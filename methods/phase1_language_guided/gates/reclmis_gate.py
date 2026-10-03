"""Execute reproduction gates against pinned upstream RecLMIS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import clip
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent
ROOT = PHASE1.parents[1]
UPSTREAM = ROOT / "upstream" / "RecLMIS"
sys.path.insert(0, str(PHASE1))
sys.path.insert(0, str(UPSTREAM))

import Config_covid19 as official_config
import nets.RecLMIS as reclmis_module
from nets.RecLMIS import RecLMIS

from adapter import load_records, split_records
from gate_common import hard_dice, load_image_mask, nonzero_gradient_summary, save_json, seed_everything, soft_dice


def tokenise(texts, device, length=18):
    tokens = clip.tokenize(list(texts), context_length=length, truncate=True).to(device)
    return tokens, (tokens != 0).int()


def build_model(device):
    reclmis_module._PT_NAME["ViT-B/32"] = str(PHASE1 / "assets" / "reclmis" / "ViT-B-32.pt")
    return RecLMIS(
        official_config,
        official_config.get_ViT_config(),
        n_channels=official_config.n_channels,
        n_classes=official_config.n_labels,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overfit-cases", type=int, default=8, choices=range(8, 17))
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()
    seed_everything(3407)
    device = torch.device(args.device)
    output = PHASE1 / "gates" / "reclmis"
    records = load_records("qata")
    splits = {name: split_records(records, name) for name in ("train", "val", "test")}

    for values in splits.values():
        for record in values[:32]:
            image, mask = load_image_mask(record)
            if image.shape != (3, 224, 224) or mask.shape != (1, 224, 224):
                raise AssertionError("Bad tensor shape for %s" % record.sample_id)

    author_path = PHASE1 / "assets" / "reclmis" / "covid19_author.pth.tar"
    model = build_model(device)
    author_state = torch.load(str(author_path), map_location="cpu")["state_dict"]
    model.load_state_dict(author_state, strict=True)

    selected = splits["train"][: args.overfit_cases]
    pairs = [load_image_mask(record) for record in selected]
    images = torch.stack([pair[0] for pair in pairs]).to(device)
    masks = torch.stack([pair[1] for pair in pairs]).to(device)
    tokens, token_mask = tokenise([record.text for record in selected], device)
    with torch.no_grad():
        _, text_embedding = model.clip.encode_text(tokens, return_hidden=True, mask=token_mask)
        official_prediction = model(images[:1], masks[:1], tokens[:1], token_mask[:1])[0]
    if official_prediction.shape != (1, 1, 224, 224):
        raise AssertionError("Author checkpoint output has wrong shape")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=official_config.learning_rate)
    bce = torch.nn.BCELoss()
    history = []
    gradients = {}
    achieved_step = None
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad()
        prediction, auxiliary = model(images, masks, tokens, token_mask)
        criterion = 0.5 * bce(prediction, masks) + 0.5 * (1.0 - soft_dice(prediction, masks))
        loss = official_config.loss_weight["loss_criterion"] * criterion
        for name, value in auxiliary.items():
            loss = loss + official_config.loss_weight[name] * value.mean()
        loss.backward()
        if step == 1:
            gradients = nonzero_gradient_summary(model, ("text_module", "interact", "rec_text"))
            if not gradients:
                raise AssertionError("No nonzero RecLMIS language gradient")
        optimizer.step()
        dice = float(hard_dice(prediction.detach(), masks).cpu())
        if step == 1 or step % 25 == 0:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "dice": dice})
        if dice >= 0.90:
            achieved_step = step
            break
    if achieved_step is None:
        raise AssertionError("RecLMIS overfit gate failed; final Dice %.6f" % dice)

    model.eval()
    with torch.no_grad():
        correct = model(images[:2], masks[:2], tokens[:2], token_mask[:2])[0]
        null_tokens, null_mask = tokenise(["", ""], device)
        null = model(images[:2], masks[:2], null_tokens, null_mask)[0]
        shuffled = model(images[:2], masks[:2], tokens[:2].flip(0), token_mask[:2].flip(0))[0]
    null_difference = float((correct - null).abs().max().cpu())
    shuffled_difference = float((correct - shuffled).abs().max().cpu())
    if null_difference == 0 or shuffled_difference == 0:
        raise AssertionError("RecLMIS text-use gate failed")

    model.load_state_dict(author_state, strict=True)
    model.eval()
    mask_dir = output / "validation_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(0, len(splits["val"]), 16):
            batch = splits["val"][index:index + 16]
            pairs = [load_image_mask(record) for record in batch]
            batch_images = torch.stack([pair[0] for pair in pairs]).to(device)
            batch_masks = torch.stack([pair[1] for pair in pairs]).to(device)
            batch_tokens, batch_token_mask = tokenise([record.text for record in batch], device)
            probability = model(batch_images, batch_masks, batch_tokens, batch_token_mask)[0].cpu().numpy()
            for record, value in zip(batch, probability):
                mask = (value[0] >= 0.5).astype(np.uint8) * 255
                if not cv2.imwrite(str(mask_dir / (record.sample_id + ".png")), mask):
                    raise IOError("Could not save %s" % record.sample_id)

    result = {
        "method": "reclmis",
        "data_gate": {name: 32 for name in splits},
        "shape_gate": {
            "image": list(images.shape), "mask": list(masks.shape),
            "token": list(tokens.shape), "embedding": list(text_embedding.shape),
            "output": list(prediction.shape),
        },
        "overfit_gate": {"dice": dice, "step": achieved_step, "history": history},
        "gradient_gate": {"nonzero_parameters": gradients},
        "inference_gate": {"saved_masks": len(list(mask_dir.glob("*.png")))},
        "text_use_gate": {"null_max_abs_difference": null_difference, "shuffled_max_abs_difference": shuffled_difference},
        "official_check_gate": {"status": "pass", "checkpoint": str(author_path), "strict_load": True},
    }
    save_json(output / "gate_results.json", result)
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "author_checkpoint_verified.pt")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
