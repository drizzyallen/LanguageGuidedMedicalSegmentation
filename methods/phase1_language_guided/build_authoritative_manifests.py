"""Build Phase 1 manifests as byte-identical copies of the frozen Phase 0 splits.

Plan section 1.4 rule 2: Phase 1 uses the same frozen train, validation and
test samples as Phase 0. The Phase 0 manifest is therefore authoritative for
membership, IDs, paths, hashes and split; report text is joined later by the
adapter. This script only copies and verifies full report coverage.

The earlier annotation-workbook split assignment (2026-09-14 to 2026-10-03) is
archived under archive/xlsx_split_20260916/.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

from adapter import load_text_lookup


ROOT = Path(__file__).resolve().parents[2]
PHASE0 = ROOT / "methods" / "phase0_image_only" / "data_manifests"
OUTPUT = Path(__file__).resolve().parent / "data_manifests"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(dataset: str) -> None:
    source = PHASE0 / (dataset + ".csv")
    target = OUTPUT / (dataset + ".csv")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    text = load_text_lookup(dataset)
    missing = [row["sample_id"] for row in rows if Path(row["image_path"]).name not in text]
    if missing:
        raise ValueError("%s: %d Phase 0 images lack a paired report" % (dataset, len(missing)))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256(source) != sha256(target):
        raise AssertionError("Copy changed %s" % dataset)
    print("%s: %s; Phase 0 split copied, 100%% paired reports, sha256=%s"
          % (dataset, dict(Counter(row["split"] for row in rows)), sha256(target)))


if __name__ == "__main__":
    build("qata")
    build("mosmed")
