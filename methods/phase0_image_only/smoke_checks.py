"""Non-training acceptance checks for the frozen Phase 0 benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from determinism import configure, deterministic_loader_class
from models import panoptic_fpn, tripath_lesionnet, unet
from models._upstream import read_manifest


class Integers(torch.utils.data.Dataset):
    def __len__(self): return 32
    def __getitem__(self, index): return index


def shuffled(seed):
    configure(seed)
    loader = deterministic_loader_class(seed, torch.utils.data.DataLoader)(
        Integers(), batch_size=4, shuffle=True, num_workers=2
    )
    return torch.cat(list(loader)).tolist()


def main():
    expected = {
        "qata": {"train": 5716, "val": 1429, "test": 2113},
        "mosmed": {"train": 1746, "val": 437, "test": 546},
        "busi": {"train": 413, "val": 104, "test": 130},
    }
    modules = {"unet": unet, "tripath_lesionnet": tripath_lesionnet, "panoptic_fpn": panoptic_fpn}
    for method, module in modules.items():
        config = yaml.safe_load((ROOT / "configs" / f"{method}.yaml").read_text())
        assert config["seeds"] == [1001, 1002, 1003]
        assert config["frozen"] is True
        for dataset, counts in expected.items():
            manifest = ROOT / "data_manifests" / f"{dataset}.csv"
            cases = read_manifest(manifest)
            actual = {split: sum(case.split == split for case in cases) for split in counts}
            assert actual == counts, (dataset, actual)
            split = (getattr(module, "dataset_split", None) or module.dataset_config)(manifest, dataset)
            train = getattr(split, "train", getattr(split, "train_pairs", None))
            val = getattr(split, "val", getattr(split, "val_pairs", None))
            assert len(train) == counts["train"] and len(val) == counts["val"]
            if hasattr(split, "test"): assert len(split.test) == counts["test"]
    first, repeat, other = shuffled(1001), shuffled(1001), shuffled(1002)
    assert first == repeat, "Same seed produced different shuffled order"
    assert first != other, "Different seeds unexpectedly produced identical order"
    print(json.dumps({"status": "PASS", "methods": list(modules), "datasets": expected, "seeds": [1001, 1002, 1003], "deterministic_loader": True}, indent=2))


if __name__ == "__main__": main()
