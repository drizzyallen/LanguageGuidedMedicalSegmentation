"""Validate complete Phase 1 image-mask-text coverage without training."""

from __future__ import annotations

import json

from adapter import load_records, split_records, validate_augmentation_policy


EXPECTED = {
    "qata": {"train": 5716, "val": 1429, "test": 2113},
    "mosmed": {"train": 2183, "val": 273, "test": 273},
}


def main() -> None:
    summary = {}
    for dataset, expected in EXPECTED.items():
        records = load_records(dataset)
        actual = {split: len(split_records(records, split)) for split in expected}
        if actual != expected:
            raise AssertionError("%s split mismatch: %r != %r" % (dataset, actual, expected))
        if len({record.sample_id for record in records}) != len(records):
            raise AssertionError("Duplicate joined sample_id in %s" % dataset)
        if any(not record.text.strip() for record in records):
            raise AssertionError("Empty joined report in %s" % dataset)
        summary[dataset] = {"total": len(records), "splits": actual, "status": "pass"}

    validate_augmentation_policy("train", augmentation=None, geometric=False)
    validate_augmentation_policy("val", augmentation=None)
    validate_augmentation_policy("test", augmentation=None)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
