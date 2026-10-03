"""Complete an official-style QaTa run for methods lacking author checkpoints.

This runner remains outside upstream and uses the pinned models, native losses,
optimizers, learning-rate schedules, seeds, and stopping rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent
ROOT = PHASE1.parents[1]
sys.path.insert(0, str(PHASE1))

from adapter import TextEmbeddingCache, load_records, split_records
from gate_common import hard_dice, load_image_mask, save_json, seed_everything


class QataDataset(Dataset):
    def __init__(self, records, extra):
        self.records = records
        self.extra = extra

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = load_image_mask(self.records[index])
        return image, mask, self.extra(index)


def mean_dice(model, loader, device, forward):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for images, masks, extra in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            prediction = forward(model, images, extra, device)
            total += float(hard_dice(prediction, masks).cpu()) * len(images)
            count += len(images)
    return total / count


def train_lvit(args, output):
    upstream = ROOT / "upstream" / "LViT"
    sys.path.insert(0, str(upstream))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    import Config as cfg
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    from bert_embedding import BertEmbedding
    from nets.LViT import LViT
    from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE

    seed_everything(cfg.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official gate requires CUDA")
    records = load_records("qata")
    train_records = split_records(records, "train")
    val_records = split_records(records, "val")
    cache = TextEmbeddingCache(
        PHASE1 / "cache" / "text_embeddings", "lvit",
        "bert-embedding-1.0.1_bert12-768_uncased", permitted=True,
    )
    bert = BertEmbedding()

    def embed(record):
        def compute(text):
            value = np.asarray(bert([text])[0][1], dtype=np.float32)[:10]
            padded = np.zeros((10, 768), dtype=np.float32)
            padded[:len(value)] = value
            return padded.tolist()
        return torch.tensor(cache.get_or_compute(record, compute), dtype=torch.float32)

    train_ds = QataDataset(train_records, lambda i: embed(train_records[i]))
    val_ds = QataDataset(val_records, lambda i: embed(val_records[i]))
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = LViT(cfg.get_CTranS_config(), n_channels=cfg.n_channels,
                 n_classes=cfg.n_labels).to(device)
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=3e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)

    def forward(current, images, extra, target_device):
        return current(images, extra.to(target_device, non_blocking=True))

    best = -1.0
    best_epoch = 0
    history = []
    checkpoint = output / "best_checkpoint.pt"
    for epoch in range(1, 2001):
        model.train()
        loss_sum = 0.0
        for images, masks, text in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            optimizer.zero_grad()
            prediction = model(images, text)
            loss = criterion(prediction, masks)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(images)
        scheduler.step()
        dice = mean_dice(model, val_loader, device, forward)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_ds), "val_dice": dice,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        save_json(output / "progress.json", {"history": history, "best_val_dice": best,
                                              "best_epoch": best_epoch})
        print(json.dumps(row), flush=True)
        if dice > best:
            best, best_epoch = dice, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_dice": dice}, checkpoint)
        if epoch - best_epoch > 50:
            break
    return {"status": "pass", "method": "lvit", "epochs": epoch,
            "best_epoch": best_epoch, "best_val_dice": best,
            "checkpoint": str(checkpoint), "cuda_device": args.device,
            "physical_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "native_settings": {"seed": 666, "batch_size": 2, "lr": 3e-4,
                                "max_epochs": 2000, "patience": 50,
                                "loss": "WeightedDiceBCE(0.5,0.5)"}}


def train_prolearn(args, output):
    upstream = ROOT / "upstream" / "ProLearn"
    sys.path.insert(0, str(upstream))
    import utils.config as config
    from models import PSA, ProLearn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    cfg = config.load_cfg_from_cfg_file(str(upstream / "config" / "training.yaml"))
    cfg.device = args.device
    cfg.prototype_save_path = str(upstream / "prototypes")
    cfg.num_prototypes = 16
    cfg.num_candidate = 3
    seed_everything(cfg.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official gate requires CUDA")
    prototype = PSA(cfg).to(device)
    prototype_path = upstream / "prototypes" / "prototype_qata_6_16_1024_contrastive.pkl"
    if not prototype.load(str(prototype_path), str(prototype_path.with_suffix(".pth"))):
        raise RuntimeError("Official prototype bank failed to load")
    model = ProLearn(cfg, prototype=prototype).to(device)
    records = load_records("qata")
    train_records = split_records(records, "train")
    val_records = split_records(records, "val")

    def make_extra(record):
        image, _ = load_image_mask(record)
        with torch.no_grad():
            value = prototype.encoder.encode_image(image.unsqueeze(0).to(device)).squeeze(0)
        return value.cpu()

    cache_dir = output / "image_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    def cached(records_, index):
        path = cache_dir / (records_[index].sample_id + ".pt")
        if not path.exists():
            torch.save(make_extra(records_[index]), path)
        return torch.load(path, map_location="cpu")

    train_ds = QataDataset(train_records, lambda i: cached(train_records, i))
    val_ds = QataDataset(val_records, lambda i: cached(val_records, i))
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    def forward(current, images, extra, target_device):
        return current.model({"image": images, "image_emb": extra.to(target_device, non_blocking=True)})

    best = -1.0
    best_epoch = 0
    patience = 0
    history = []
    checkpoint = output / "best_checkpoint.pt"
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, masks, image_emb in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            image_emb = image_emb.to(device, non_blocking=True)
            optimizer.zero_grad()
            result = model({"image": images, "image_emb": image_emb, "label": masks})
            loss = result["loss"]
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(images)
        dice = mean_dice(model, val_loader, device, forward)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_ds), "val_dice": dice,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if dice > best:
            best, best_epoch, patience = dice, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_dice": dice}, checkpoint)
        else:
            patience += 1
        save_json(output / "progress.json", {"history": history, "best_val_dice": best,
                                              "best_epoch": best_epoch})
        scheduler.step()
        if patience >= cfg.patience:
            break
    return {"status": "pass", "method": "prolearn", "epochs": epoch,
            "best_epoch": best_epoch, "best_val_dice": best,
            "checkpoint": str(checkpoint), "cuda_device": args.device,
            "physical_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "native_settings": {"seed": 42, "batch_size": 32, "lr": 1e-4,
                                "max_epochs": 50, "patience": 20,
                                "loss": "MONAI DiceCELoss"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("lvit", "prolearn"))
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    output = PHASE1 / "gates" / args.method / "official_qata"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = train_lvit(args, output) if args.method == "lvit" else train_prolearn(args, output)
    result["elapsed_seconds"] = time.time() - started
    save_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
