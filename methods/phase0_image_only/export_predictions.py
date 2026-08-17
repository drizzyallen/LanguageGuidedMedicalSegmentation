"""Export complete per-case predictions from frozen Phase 0 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from determinism import configure, deterministic_loader_class
from models import panoptic_fpn, tripath_lesionnet
from models._upstream import read_manifest


class Cases(Dataset):
    def __init__(self, cases, image_size):
        self.cases, self.image_size = cases, image_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        size = (self.image_size, self.image_size)
        image = Image.open(case.image).convert("L").resize(size, Image.Resampling.BILINEAR)
        mask = Image.open(case.mask).convert("L").resize(size, Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())[None] / 255.0
        y = (torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy())[None] > 0).float()
        return x, y, case.sample_id, case.source_id


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True), nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))

    def forward(self, value): return self.layers(value)


class Down(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__(); self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(input_channels, output_channels))

    def forward(self, value): return self.layers(value)


class Up(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__(); self.up = nn.ConvTranspose2d(input_channels, input_channels // 2, 2, stride=2); self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, value, skip):
        value = self.up(value); dy, dx = skip.size(2) - value.size(2), skip.size(3) - value.size(3)
        value = F.pad(value, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, value], dim=1))


class UNet(nn.Module):
    """Exact architecture from pinned U-Net-Model.py/UnetTesting.py."""
    def __init__(self):
        super().__init__(); self.inc = DoubleConv(1, 64); self.down1 = Down(64, 128); self.down2 = Down(128, 256); self.down3 = Down(256, 512); self.down4 = Down(512, 1024); self.up1 = Up(1024, 512); self.up2 = Up(512, 256); self.up3 = Up(256, 128); self.up4 = Up(128, 64); self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, value):
        x1 = self.inc(value); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))


def metrics(prediction, target):
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(prediction, target).sum(); union = np.logical_or(prediction, target).sum(); denominator = prediction.sum() + target.sum()
    dice = (2 * intersection + 1e-7) / (denominator + 1e-7); miou = (intersection + 1e-7) / (union + 1e-7)
    if not prediction.any() and not target.any(): return dice, miou, 0.0, 0.0
    if not prediction.any() or not target.any():
        diagonal = float(np.hypot(*prediction.shape)); return dice, miou, diagonal, diagonal
    structure = np.ones((3, 3), dtype=bool)
    ps = prediction ^ binary_erosion(prediction, structure=structure, border_value=0); ts = target ^ binary_erosion(target, structure=structure, border_value=0)
    distances = np.concatenate([distance_transform_edt(~ts)[ps], distance_transform_edt(~ps)[ts]])
    return dice, miou, float(np.percentile(distances, 95)), float(distances.mean())


def load_model(method, checkpoint, device):
    state = checkpoint["model_state_dict"]
    if method == "unet": model = UNet()
    elif method == "tripath_lesionnet":
        saved = checkpoint.get("run_config", {}).get("model_config", {})
        if "context_dilations" in saved: saved["context_dilations"] = tuple(saved["context_dilations"])
        model = tripath_lesionnet.TriPathLesionNet(tripath_lesionnet.ModelConfig(**saved))
    else:
        saved = checkpoint.get("run_config", {}).get("model_config", {})
        # Checkpoint loading does not require downloading pretrained weights.
        saved["pretrained_backbone"] = False
        model = panoptic_fpn.PanopticFPN(panoptic_fpn.ModelConfig(**saved))
    model.load_state_dict(state, strict=True); model.to(device).eval(); return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["unet", "tripath_lesionnet", "panoptic_fpn"], required=True)
    parser.add_argument("--dataset", choices=["qata", "mosmed", "busi"], required=True)
    parser.add_argument("--seed", type=int, choices=[1001, 1002, 1003], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    configure(args.seed); device = torch.device(args.device)
    config = yaml.safe_load((ROOT / "configs" / f"{args.method}.yaml").read_text())
    cases = [case for case in read_manifest(ROOT / "data_manifests" / f"{args.dataset}.csv") if case.split == "test"]
    run_root = args.runs_root / args.method / args.dataset / f"seed_{args.seed}"
    checkpoint_path = run_root / "upstream" / args.dataset / "best.pt"
    if not checkpoint_path.is_file(): raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model(args.method, checkpoint, device)
    loader_type = deterministic_loader_class(args.seed, torch.utils.data.DataLoader)
    loader = loader_type(Cases(cases, config["training"]["image_size"]), batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"], pin_memory=device.type == "cuda")
    output = run_root / "predictions"; probabilities_dir = output / "probabilities"; masks_dir = output / "masks"
    probabilities_dir.mkdir(parents=True, exist_ok=True); masks_dir.mkdir(parents=True, exist_ok=True)
    rows, threshold = [], config["evaluation"]["threshold"]
    with torch.inference_mode():
        for images, targets, sample_ids, source_ids in loader:
            images = images.to(device)
            with torch.autocast(device_type=device.type, enabled=config["training"]["amp"] and device.type == "cuda"):
                values = model(images); logits = values["segmentation"] if args.method == "tripath_lesionnet" else values
            probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]; targets = targets.numpy()[:, 0].astype(bool)
            for sample_id, source_id, probability, target in zip(sample_ids, source_ids, probabilities, targets):
                prediction = probability > threshold if args.method == "unet" else probability >= threshold
                probability_name, mask_name = f"{sample_id}.npy", f"{sample_id}.png"
                np.save(probabilities_dir / probability_name, probability.astype(np.float32), allow_pickle=False)
                Image.fromarray((prediction.astype(np.uint8) * 255)).save(masks_dir / mask_name)
                rows.append((sample_id, source_id, f"probabilities/{probability_name}", f"masks/{mask_name}", *metrics(prediction, target)))
    metrics_path = output / "per_case_metrics.csv"
    with metrics_path.open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["sample_id", "source_id", "probability_path", "prediction_path", "dice", "miou", "hd95", "assd"]); writer.writerows(rows)
    with metrics_path.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    if [row["sample_id"] for row in saved] != [case.sample_id for case in cases]:
        raise RuntimeError("Exported sample order differs from frozen manifest test order")
    names = ("dice", "miou", "hd95", "assd")
    means = {name: float(np.mean([float(row[name]) for row in saved])) for name in names}
    summary = {"method": args.method, "dataset": args.dataset, "seed": args.seed,
               "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]),
               "test_samples": len(saved), "threshold": threshold,
               **{f"test_{name}": value for name, value in means.items()}}
    with (output / "test_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    recalculated = {name: float(np.mean([float(row[name]) for row in saved])) for name in names}
    if any(recalculated[name] != means[name] for name in names):
        raise RuntimeError("Per-case means do not match test summary")
    print(f"Exported {len(rows)} cases to {output}")


if __name__ == "__main__": main()
