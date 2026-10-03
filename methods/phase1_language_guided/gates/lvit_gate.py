"""Execute the pre-training reproduction gates against pinned upstream LViT."""

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
sys.path.insert(0, str(PHASE1))
sys.path.insert(0, str(ROOT / "upstream" / "LViT"))

import Config as official_config
from bert_embedding import BertEmbedding
from nets.LViT import LViT

from adapter import TextEmbeddingCache, load_records, split_records
from gate_common import (
    hard_dice,
    load_image_mask,
    nonzero_gradient_summary,
    save_json,
    seed_everything,
    soft_dice,
)


def pad_embedding(value, length=10, width=768):
    array = np.asarray(value, dtype=np.float32)[:length]
    result = np.zeros((length, width), dtype=np.float32)
    if array.size:
        result[: array.shape[0]] = array
    return result.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--overfit-cases", type=int, default=8, choices=range(8, 17))
    parser.add_argument("--max-steps", type=int, default=2500)
    args = parser.parse_args()
    seed_everything(3407)
    device = torch.device(args.device)
    output = PHASE1 / "gates" / "lvit"
    cache = TextEmbeddingCache(
        PHASE1 / "cache" / "text_embeddings",
        "lvit",
        "bert-embedding-1.0.1_bert12-768_uncased",
        permitted=True,
    )
    bert = BertEmbedding()

    records = load_records("qata")
    splits = {name: split_records(records, name) for name in ("train", "val", "test")}
    data_gate = {}
    for name, values in splits.items():
        loaded = []
        for record in values[:32]:
            image, mask = load_image_mask(record)
            if image.shape != (3, 224, 224) or mask.shape != (1, 224, 224):
                raise AssertionError("Bad tensor shape for %s" % record.sample_id)
            loaded.append(record.sample_id)
        data_gate[name] = loaded

    def embedding(record):
        return cache.get_or_compute(
            record,
            lambda text: pad_embedding(bert([text])[0][1]),
        )

    selected = splits["train"][: args.overfit_cases]
    images, masks, texts = zip(
        *((*load_image_mask(record), torch.tensor(embedding(record))) for record in selected)
    )
    images = torch.stack(images).to(device)
    masks = torch.stack(masks).to(device)
    texts = torch.stack(texts).to(device)

    config = official_config.get_CTranS_config()
    model = LViT(config, n_channels=official_config.n_channels, n_classes=official_config.n_labels).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=official_config.learning_rate)
    bce = torch.nn.BCELoss()
    history = []
    gradients = {}
    achieved_step = None
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(images, texts)
        loss = 0.5 * bce(prediction, masks) + 0.5 * (1.0 - soft_dice(prediction, masks))
        loss.backward()
        if step == 1:
            gradients = nonzero_gradient_summary(model, ("text_module",))
            if not gradients:
                raise AssertionError("No nonzero LViT language gradient")
        optimizer.step()
        dice = float(hard_dice(prediction.detach(), masks).cpu())
        if step == 1 or step % 25 == 0:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "dice": dice})
        if dice >= 0.90:
            achieved_step = step
            break
    if achieved_step is None:
        raise AssertionError("LViT overfit gate failed; final Dice %.6f" % dice)

    model.eval()
    with torch.no_grad():
        probe_images = images[:2]
        correct = model(probe_images, texts[:2])
        null = model(probe_images, torch.zeros_like(texts[:2]))
        shuffled = model(probe_images, texts[:2].flip(0))
    null_difference = float((correct - null).abs().max().cpu())
    shuffled_difference = float((correct - shuffled).abs().max().cpu())
    if null_difference == 0 or shuffled_difference == 0:
        raise AssertionError("LViT text-use gate failed")

    mask_dir = output / "validation_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(0, len(splits["val"]), 16):
            batch = splits["val"][index:index + 16]
            batch_images = torch.stack([load_image_mask(record)[0] for record in batch]).to(device)
            batch_text = torch.tensor([embedding(record) for record in batch]).to(device)
            probability = model(batch_images, batch_text).cpu().numpy()
            for record, value in zip(batch, probability):
                mask = (value[0] >= 0.5).astype(np.uint8) * 255
                if not cv2.imwrite(str(mask_dir / (record.sample_id + ".png")), mask):
                    raise IOError("Could not save %s" % record.sample_id)

    result = {
        "method": "lvit",
        "data_gate": {key: len(value) for key, value in data_gate.items()},
        "shape_gate": {
            "image": list(images.shape),
            "mask": list(masks.shape),
            "token": [args.overfit_cases, 10],
            "embedding": list(texts.shape),
            "output": list(prediction.shape),
        },
        "overfit_gate": {"dice": dice, "step": achieved_step, "history": history},
        "gradient_gate": {"nonzero_parameters": gradients},
        "inference_gate": {"saved_masks": len(list(mask_dir.glob("*.png")))},
        "text_use_gate": {"null_max_abs_difference": null_difference, "shuffled_max_abs_difference": shuffled_difference},
        "official_check_gate": {"status": "pending_official_style_qata_run", "author_checkpoint_available": False},
    }
    save_json(output / "gate_results.json", result)
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "overfit_checkpoint.pt")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

