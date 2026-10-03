"""Section 1.4: full-paired, XLSX-authoritative three-seed experiment.

Upstream modules are read-only. Each process handles one method/dataset/seed.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset

PHASE1 = Path(__file__).resolve().parent
ROOT = PHASE1.parents[1]
sys.path.insert(0, str(PHASE1))
from adapter import PINNED_SHA256, TextEmbeddingCache, load_records, split_records
from gate_common import hard_dice, load_image_mask, save_json, seed_everything


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metrics(prediction, target):
    """Same surface distances, empty-mask conventions and epsilon as Phase 0."""
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    dice = (2 * intersection + 1e-7) / (denominator + 1e-7)
    miou = (intersection + 1e-7) / (union + 1e-7)
    if not prediction.any() and not target.any():
        return dice, miou, 0.0, 0.0
    if not prediction.any() or not target.any():
        diagonal = float(np.hypot(*prediction.shape))
        return dice, miou, diagonal, diagonal
    structure = np.ones((3, 3), dtype=bool)
    ps = prediction ^ binary_erosion(prediction, structure=structure, border_value=0)
    ts = target ^ binary_erosion(target, structure=structure, border_value=0)
    distances = np.concatenate([distance_transform_edt(~ts)[ps], distance_transform_edt(~ps)[ts]])
    return dice, miou, float(np.percentile(distances, 95)), float(distances.mean())


class Cases(Dataset):
    def __init__(self, records, extra):
        self.records, self.extra = records, extra

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, mask = load_image_mask(record)
        return image, mask, self.extra(record), index


def build(args):
    upstream = ROOT / "upstream" / {"lvit": "LViT", "reclmis": "RecLMIS"}[args.method]
    sys.path.insert(0, str(upstream))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.method == "lvit":
        import Config as cfg
        if visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        cfg.seed = args.seed
        cfg.task_name = "Covid19" if args.dataset == "qata" else "MosMedplus"
        from bert_embedding import BertEmbedding
        from nets.LViT import LViT
        from utils import WeightedDiceBCE, CosineAnnealingWarmRestarts
        bert = BertEmbedding()
        cache = TextEmbeddingCache(PHASE1 / "cache" / "text_embeddings", "lvit",
                                   "bert-embedding-1.0.1_bert12-768_uncased", permitted=True)

        def extra(record):
            def compute(text):
                value = np.asarray(bert([text])[0][1], dtype=np.float32)[:10]
                padded = np.zeros((10, 768), dtype=np.float32)
                padded[:len(value)] = value
                return padded.tolist()
            return torch.tensor(cache.get_or_compute(record, compute), dtype=torch.float32)

        model = LViT(cfg.get_CTranS_config(), n_channels=cfg.n_channels, n_classes=cfg.n_labels).to(args.device)
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

        def forward(images, masks, text):
            prediction = model(images, text.to(args.device))
            return prediction, criterion(prediction, masks)

        settings = dict(batch_size=2, lr=3e-4, max_epochs=2000, patience=50,
                        optimizer="Adam", scheduler="CosineAnnealingWarmRestarts(10,1,1e-4)",
                        loss="WeightedDiceBCE(0.5,0.5)")
    else:
        import importlib
        from reclmis_compat import install
        install()
        cfg = importlib.import_module("Config_covid19" if args.dataset == "qata" else "Config_MosMedPlus")
        cfg.seed = args.seed
        import clip
        import nets.RecLMIS as module
        from nets.RecLMIS import RecLMIS
        from utils import WeightedDiceBCE, CosineAnnealingWarmRestarts
        module._PT_NAME["ViT-B/32"] = str(PHASE1 / "assets" / "reclmis" / "ViT-B-32.pt")
        model = RecLMIS(cfg, cfg.get_ViT_config(), n_channels=cfg.n_channels, n_classes=cfg.n_labels).to(args.device)
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

        def extra(record):
            return clip.tokenize([record.text], context_length=cfg.token_len, truncate=True)[0]

        def forward(images, masks, tokens):
            tokens = tokens.to(args.device)
            # Evaluation supplies a zero placeholder: no ground-truth mask enters inference.
            prediction, auxiliary = model(images, masks if model.training else torch.zeros_like(masks),
                                          tokens, (tokens != 0).int())
            if model.training and set(auxiliary) != {"loss_ccl", "loss_text_rec", "loss_img_rec"}:
                raise RuntimeError("RecLMIS silently disabled its native auxiliary losses; refusing altered training")
            loss = cfg.loss_weight["loss_criterion"] * criterion(prediction, masks).mean()
            for name, value in auxiliary.items():
                loss = loss + cfg.loss_weight[name] * value.mean()
            return prediction, loss

        settings = dict(batch_size=cfg.batch_size, lr=cfg.learning_rate, max_epochs=cfg.epochs,
                        patience=cfg.early_stopping_patience, optimizer=cfg.optimizer,
                        scheduler=cfg.lr, loss="WeightedDiceBCE(0.5,0.5) + native weighted auxiliary losses")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=settings["lr"])
    if args.method == "reclmis" and args.dataset == "mosmed":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.99 ** epoch, 0.1))
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    return model, optimizer, scheduler, extra, forward, settings


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate(model, loader, forward, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for images, masks, extra, _ in loader:
            images, masks = images.to(device), masks.to(device)
            prediction, _ = forward(images, masks, extra)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Nonfinite validation prediction")
            total += float(hard_dice(prediction, masks).cpu()) * len(images)
            count += len(images)
    return total / count


def export_once(args, output, model, extra, forward, records):
    checkpoint = output / "best_checkpoint.pt"
    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["state_dict"], strict=True)
    # Exclusive marker is created before the first test forward. A partial export
    # is held for review rather than silently evaluating this checkpoint again.
    marker = output / "test_started.json"
    with marker.open("x") as stream:
        json.dump({"checkpoint_sha256": digest(checkpoint), "epoch": state["epoch"]}, stream)
    cases = split_records(records, "test")
    loader = DataLoader(Cases(cases, extra), batch_size=1, shuffle=False, num_workers=0)
    prediction_dir = output / "predictions"
    (prediction_dir / "probabilities").mkdir(parents=True)
    (prediction_dir / "masks").mkdir()
    rows = []
    model.eval()
    with torch.no_grad():
        for images, masks, value, indices in loader:
            probability, _ = forward(images.to(args.device), masks.to(args.device), value)
            probability = probability.float().cpu().numpy()[:, 0]
            for index, prob, target in zip(indices.tolist(), probability, masks.numpy()[:, 0]):
                if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                    raise RuntimeError("Invalid test probability")
                record = cases[index]
                prediction = prob >= 0.5
                np.save(prediction_dir / "probabilities" / (record.sample_id + ".npy"), prob.astype(np.float32), allow_pickle=False)
                Image.fromarray(prediction.astype(np.uint8) * 255).save(prediction_dir / "masks" / (record.sample_id + ".png"))
                rows.append((record.sample_id, record.source_id, "probabilities/" + record.sample_id + ".npy",
                             "masks/" + record.sample_id + ".png", *metrics(prediction, target)))
    if [row[0] for row in rows] != [case.sample_id for case in cases]:
        raise RuntimeError("Test export membership/order mismatch")
    with (prediction_dir / "per_case_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "source_id", "probability_path", "prediction_path", "dice", "miou", "hd95", "assd"])
        writer.writerows(rows)
    summary = dict(method=args.method, dataset=args.dataset, seed=args.seed, test_samples=len(rows),
                   checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), checkpoint_epoch=state["epoch"],
                   threshold=0.5, text_required_at_inference=True,
                   **{"test_" + name: float(np.mean([row[i + 4] for row in rows]))
                      for i, name in enumerate(["dice", "miou", "hd95", "assd"])})
    save_json(prediction_dir / "test_summary.json", summary)
    return summary


def run(args):
    output = PHASE1 / ("smoke_runs_v2" if args.smoke else "runs") / args.method / args.dataset / ("seed_%d" % args.seed)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "result.json").exists():
        print("Already complete: " + str(output), flush=True)
        return
    if (output / "test_started.json").exists():
        raise RuntimeError("Test already started; held for review to prevent repeat evaluation")
    records = load_records(args.dataset)
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model, optimizer, scheduler, extra, forward, settings = build(args)
    train = split_records(records, "train")
    val = split_records(records, "val")
    inputs = {path: digest(path) for path in PINNED_SHA256}
    inputs[str(PHASE1 / "PROVENANCE.yaml")] = digest(PHASE1 / "PROVENANCE.yaml")
    for relative in ["adapter.py", "gate_common.py"]:
        inputs[str(PHASE1 / relative)] = digest(PHASE1 / relative)
    if args.method == "reclmis":
        inputs[str(PHASE1 / "reclmis_compat.py")] = digest(PHASE1 / "reclmis_compat.py")
    source = ROOT / "upstream" / ("LViT" if args.method == "lvit" else "RecLMIS")
    for path in sorted(source.rglob("*.py")):
        inputs[str(path)] = digest(path)
    metadata = dict(method=args.method, dataset=args.dataset, seed=args.seed, paired_report_fraction=1.0,
                    split_authority="Phase 1 XLSX; Phase 0 image/mask pool", settings=settings,
                    samples={split: len(split_records(records, split)) for split in ("train", "val", "test")},
                    input_sha256=inputs, runner_sha256=digest(__file__),
                    text_required_at_inference=True, augmentation="disabled by text-safe adapter policy",
                    checkpoint_selection="validation mean per-case Dice at threshold 0.5",
                    deterministic_algorithms=True, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    torch_version=torch.__version__, smoke=args.smoke)
    metadata_path = output / "run_config.json"
    if metadata_path.exists() and not args.smoke:
        old = json.loads(metadata_path.read_text())
        if old != metadata:
            raise RuntimeError("Run inputs/configuration changed; refusing resume")
    else:
        save_json(metadata_path, metadata)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Cases(train, extra), batch_size=settings["batch_size"], shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(Cases(val, extra), batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    if args.smoke:
        images, masks, value, _ = next(iter(train_loader))
        model.train()
        optimizer.zero_grad()
        prediction, loss = forward(images.to(args.device), masks.to(args.device), value)
        if prediction.shape != masks.shape or not torch.isfinite(loss):
            raise RuntimeError("Smoke shape/loss failure")
        loss.backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters()):
            raise RuntimeError("Smoke gradient failure")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability, _ = forward(images[:1].to(args.device), masks[:1].to(args.device), value[:1])
        if not torch.isfinite(probability).all():
            raise RuntimeError("Smoke inference failure")
        atomic_checkpoint(output / "best_checkpoint.pt", dict(state_dict=model.state_dict(), epoch=0, val_dice=0.0))
        # Exercise checkpoint reload and complete export using validation records
        # relabeled only in memory. Actual held-out test images are never loaded.
        from dataclasses import replace
        summary = export_once(args, output, model, extra, forward,
                              tuple(replace(record, split="test") for record in val[:2]))
        saved_rows = list(csv.DictReader((output / "predictions" / "per_case_metrics.csv").open()))
        for name in ["dice", "miou", "hd95", "assd"]:
            assert np.mean([float(row[name]) for row in saved_rows]) == summary["test_" + name]
        save_json(output / "result.json", dict(status="smoke_pass", shape=list(probability.shape),
                                               loss=float(loss.detach().cpu()), exported_validation_cases=2,
                                               runner_sha256=digest(__file__)))
        print("SMOKE PASS " + str(output), flush=True)
        return
    start, best, best_epoch, history = 1, -1.0, 0, []
    last = output / "last_checkpoint.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best, best_epoch, history = state["epoch"] + 1, state["best"], state["best_epoch"], state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
    stopped = start - 1 - best_epoch > settings["patience"]
    for epoch in range(start, settings["max_epochs"] + 1):
        if stopped:
            break
        model.train()
        total = 0.0
        for images, masks, value, _ in train_loader:
            optimizer.zero_grad()
            _, loss = forward(images.to(args.device), masks.to(args.device), value)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(images)
        dice = evaluate(model, val_loader, forward, args.device)
        scheduler.step()
        row = dict(epoch=epoch, train_loss=total / len(train), val_dice=dice, lr=optimizer.param_groups[0]["lr"])
        history.append(row)
        if dice > best:
            best, best_epoch = dice, epoch
            atomic_checkpoint(output / "best_checkpoint.pt", dict(state_dict=model.state_dict(), epoch=epoch, val_dice=dice))
        stopped = epoch - best_epoch > settings["patience"]
        atomic_checkpoint(last, dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), epoch=epoch, best=best, best_epoch=best_epoch,
                                    history=history, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                    loader_rng=generator.get_state()))
        save_json(output / "progress.json", dict(history=history, best_val_dice=best, best_epoch=best_epoch))
        print(json.dumps(row), flush=True)
    summary = export_once(args, output, model, extra, forward, records)
    save_json(output / "result.json", dict(status="complete", best_val_dice=best, **summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["lvit", "reclmis"])
    parser.add_argument("--dataset", choices=["qata", "mosmed"], required=True)
    parser.add_argument("--seed", type=int, choices=[1001, 1002, 1003], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
