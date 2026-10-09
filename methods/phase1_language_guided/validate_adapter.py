"""Validate complete Phase 1 image-mask-text coverage without training."""

from __future__ import annotations

import hashlib
import json

from adapter import PHASE1_MANIFESTS, ROOT, load_records, split_records, validate_augmentation_policy


# Frozen Phase 0 split counts (plan section 1.4 rule 2).
EXPECTED = {
    "qata": {"train": 5716, "val": 1429, "test": 2113},
    "mosmed": {"train": 1746, "val": 437, "test": 546},
}
PHASE0_MANIFESTS = ROOT / "methods" / "phase0_image_only" / "data_manifests"


def main() -> None:
    summary = {}
    for dataset, expected in EXPECTED.items():
        phase0 = hashlib.sha256((PHASE0_MANIFESTS / (dataset + ".csv")).read_bytes()).hexdigest()
        phase1 = hashlib.sha256((PHASE1_MANIFESTS / (dataset + ".csv")).read_bytes()).hexdigest()
        if phase0 != phase1:
            raise AssertionError("%s Phase 1 manifest is not byte-identical to Phase 0" % dataset)
        records = load_records(dataset)
        actual = {split: len(split_records(records, split)) for split in expected}
        if actual != expected:
            raise AssertionError("%s split mismatch: %r != %r" % (dataset, actual, expected))
        if len({record.sample_id for record in records}) != len(records):
            raise AssertionError("Duplicate joined sample_id in %s" % dataset)
        if any(not record.text.strip() for record in records):
            raise AssertionError("Empty joined report in %s" % dataset)
        summary[dataset] = {"total": len(records), "splits": actual, "manifest_sha256": phase1,
                            "unique_reports": len({record.text for record in records}), "status": "pass"}

    validate_augmentation_policy("train", augmentation=None, geometric=False)
    validate_augmentation_policy("val", augmentation=None)
    validate_augmentation_policy("test", augmentation=None)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
