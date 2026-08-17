"""Manifest-backed entry point for frozen Phase 0 training and data checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from determinism import configure, patched_dataloaders
from models import panoptic_fpn, tripath_lesionnet, unet

METHODS = {
    "unet": unet,
    "tripath_lesionnet": tripath_lesionnet,
    "panoptic_fpn": panoptic_fpn,
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(method):
    path = ROOT / "configs" / f"{method}.yaml"
    with path.open() as stream:
        return path, yaml.safe_load(stream)


def manifest_path(dataset):
    return ROOT / "data_manifests" / f"{dataset}.csv"


def select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    return torch.device(requested)


def upstream_args(config, dataset, seed, output, device):
    training = config["training"]
    optimizer, scheduler = training["optimizer"], training["scheduler"]
    return Namespace(
        dataset=dataset, output_dir=output, image_size=training["image_size"],
        epochs=training["epochs"], batch_size=training["batch_size"],
        learning_rate=optimizer["learning_rate"], weight_decay=optimizer["weight_decay"],
        early_stopping_patience=training["early_stopping"]["patience"],
        scheduler_patience=scheduler["patience"], scheduler_factor=scheduler["factor"],
        num_workers=training["num_workers"], seed=seed, device=device,
        amp=training["amp"], max_train_samples=None, max_val_samples=None,
        max_test_samples=None, threshold=config["evaluation"]["threshold"],
        save_visualizations=False, max_visualizations=0,
        pretrained_backbone=config.get("model", {}).get("pretrained_backbone", True),
    )


def build_split(module, manifest, dataset):
    function = getattr(module, "dataset_split", None) or module.dataset_config
    return function(manifest, dataset)


def write_metadata(path, method, dataset, seed, config_path, manifest, settings):
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": method, "dataset": dataset, "seed": seed,
        "upstream_commit": "fdeb44037aba52900d852271293b165a697ecf06",
        "config": str(config_path), "config_sha256": file_sha256(config_path),
        "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "determinism": settings,
    }
    with (path / "run_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check-data", "train"])
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=["qata", "mosmed", "busi"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    module = METHODS[args.method]
    config_path, config = load_config(args.method)
    if args.seed not in config["seeds"]:
        raise SystemExit(f"Seed {args.seed} is not frozen in {config_path}")
    manifest = manifest_path(args.dataset)
    split = build_split(module, manifest, args.dataset)
    sizes = {
        "train": len(getattr(split, "train", getattr(split, "train_pairs", ()))),
        "val": len(getattr(split, "val", getattr(split, "val_pairs", ()))),
    }
    if hasattr(split, "test"):
        sizes["test"] = len(split.test)
    print(json.dumps({"method": args.method, "dataset": args.dataset, "seed": args.seed, "samples": sizes}, indent=2))
    if args.mode == "check-data":
        return

    device = select_device(args.device)
    settings = configure(args.seed)
    run_root = args.output_root / args.method / args.dataset / f"seed_{args.seed}"
    invocation_root = run_root / "upstream"
    write_metadata(run_root, args.method, args.dataset, args.seed, config_path, manifest, settings)
    upstream = module.upstream
    upstream.seed_everything(args.seed) if hasattr(upstream, "seed_everything") else None
    call_args = upstream_args(config, args.dataset, args.seed, invocation_root, device)
    with patched_dataloaders(upstream, args.seed):
        if args.method == "unet":
            result = upstream.train(call_args, split)
        elif args.method == "tripath_lesionnet":
            result = upstream.train_dataset(split, call_args, module.ModelConfig(), module.LossConfig())
        else:
            model = module.ModelConfig(pretrained_backbone=call_args.pretrained_backbone)
            result = upstream.train_dataset(split, call_args, model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
