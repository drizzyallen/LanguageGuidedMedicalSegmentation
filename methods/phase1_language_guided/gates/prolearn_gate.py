"""Execute pre-training reproduction gates against pinned upstream ProLearn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent
ROOT = PHASE1.parents[1]
UPSTREAM = ROOT / "upstream" / "ProLearn"
sys.path.insert(0, str(PHASE1))
sys.path.insert(0, str(UPSTREAM))

import utils.config as config
from models import PSA, ProLearn

from adapter import load_records, split_records
from gate_common import hard_dice, load_image_mask, nonzero_gradient_summary, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overfit-cases", type=int, default=8, choices=range(8, 17))
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()
    seed_everything(42)
    device = torch.device(args.device)
    cfg = config.load_cfg_from_cfg_file(str(UPSTREAM / "config" / "training.yaml"))
    cfg.device = str(device)
    cfg.prototype_save_path = str(UPSTREAM / "prototypes")
    cfg.num_prototypes = 16
    cfg.num_candidate = 3

    output = PHASE1 / "gates" / "prolearn"
    records = load_records("qata")
    splits = {name: split_records(records, name) for name in ("train", "val", "test")}
    for values in splits.values():
        for record in values[:32]:
            image, mask = load_image_mask(record)
            if image.shape != (3, 224, 224) or mask.shape != (1, 224, 224):
                raise AssertionError("Bad tensor shape for %s" % record.sample_id)

    prototype = PSA(cfg).to(device)
    prototype_path = UPSTREAM / "prototypes" / "prototype_qata_6_16_1024_contrastive.pkl"
    if not prototype.load(str(prototype_path), str(prototype_path.with_suffix(".pth"))):
        raise AssertionError("Official ProLearn prototype bank did not load")
    model = ProLearn(cfg, prototype=prototype).to(device)

    selected = splits["train"][: args.overfit_cases]
    pairs = [load_image_mask(record) for record in selected]
    images = torch.stack([pair[0] for pair in pairs]).to(device)
    masks = torch.stack([pair[1] for pair in pairs]).to(device)
    reports = [record.text for record in selected]
    with torch.no_grad():
        tokenised = model.model.tokenize(reports, device)
        native_tokens = tokenised["input_ids"]
        text_embedding = model.model.text_encoder(reports)
        image_embedding = prototype.encoder.encode_image(images)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []
    gradients = {}
    achieved_step = None
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch = {"image": images, "image_emb": image_embedding, "label": masks, "text": reports}
        result = model(batch)
        prediction, loss = result["logits"], result["loss"]
        loss.backward()
        if step == 1:
            gradients = nonzero_gradient_summary(model, ("prototype.matrix", "approx"))
            if not gradients:
                raise AssertionError("No nonzero ProLearn language/prototype gradient")
        optimizer.step()
        dice = float(hard_dice(prediction.detach(), masks).cpu())
        if step == 1 or step % 25 == 0:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "dice": dice})
        if dice >= 0.90:
            achieved_step = step
            break
    if achieved_step is None:
        raise AssertionError("ProLearn overfit gate failed; final Dice %.6f" % dice)

    model.eval()
    mask_dir = output / "validation_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(0, len(splits["val"]), 16):
            batch = splits["val"][index:index + 16]
            pairs = [load_image_mask(record) for record in batch]
            batch_images = torch.stack([pair[0] for pair in pairs]).to(device)
            batch_embedding = prototype.encoder.encode_image(batch_images)
            probability = model.model({"image": batch_images, "image_emb": batch_embedding}).cpu().numpy()
            for record, value in zip(batch, probability):
                mask = (value[0] >= 0.5).astype(np.uint8) * 255
                if not cv2.imwrite(str(mask_dir / (record.sample_id + ".png")), mask):
                    raise IOError("Could not save %s" % record.sample_id)

    result = {
        "method": "prolearn",
        "data_gate": {name: 32 for name in splits},
        "shape_gate": {
            "image": list(images.shape), "mask": list(masks.shape),
            "token": list(native_tokens.shape), "embedding": list(text_embedding.shape),
            "output": list(prediction.shape),
        },
        "overfit_gate": {"dice": dice, "step": achieved_step, "history": history},
        "gradient_gate": {"nonzero_parameters": gradients},
        "inference_gate": {"saved_masks": len(list(mask_dir.glob("*.png")))},
        "text_use_gate": {"status": "not_applicable", "reason": "ProLearn inference consumes image and image-derived prototype query only"},
        "official_check_gate": {"status": "pending_official_style_qata_run", "author_segmentation_checkpoint_available": False, "official_prototype_bank_loaded": True},
    }
    save_json(output / "gate_results.json", result)
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "overfit_checkpoint.pt")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
