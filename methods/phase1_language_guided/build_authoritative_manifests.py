"""Build Phase 1 copies using the annotation-defined Phase 1 splits."""

from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from adapter import _normalise_text_key, _xlsx_rows


ROOT = Path(__file__).resolve().parents[2]
PHASE0 = ROOT / "methods" / "phase0_image_only" / "data_manifests"
OUTPUT = Path(__file__).resolve().parent / "data_manifests"
QATA_TEXT = Path("/data/ramialle/datasets/QaTa-Text")
MOSMED_TEXT = Path("/data/ramialle/datasets/MosMedData+-Text")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
PINNED_AUTHORITIES = {
    str(QATA_TEXT / "train" / "Train_ID.xlsx"): "f904331c368ddf9e773b8239efd659fab1152952beb2624f57a2841eda994468",
    str(QATA_TEXT / "train" / "Val_ID.xlsx"): "559b0af19134b6ec0b389326e507648c2c38de97d7ec9fd2412e8aef7990df30",
}


def verify_authority(path: Path) -> None:
    expected = PINNED_AUTHORITIES.get(str(path))
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Pinned split authority changed: %s" % path)


def one_column_ids(path: Path, dataset: str) -> set[str]:
    """Read the first column of an ID-only XLSX workbook."""
    verify_authority(path)
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.iterfind(".//m:t", NS))
                  for item in root.findall("m:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        values = []
        for row in sheet.iterfind(".//m:sheetData/m:row", NS):
            cell = row.find("m:c", NS)
            if cell is None:
                continue
            value_node = cell.find("m:v", NS)
            if value_node is None:
                continue
            value = shared[int(value_node.text)] if cell.attrib.get("t") == "s" else value_node.text or ""
            if value.strip().lower() in {"image", "filename"}:
                continue
            values.append(_normalise_text_key(dataset, value))
    if len(values) != len(set(values)):
        raise ValueError("Duplicate ID in %s" % path)
    return set(values)


def text_ids(path: Path, dataset: str) -> set[str]:
    values = [_normalise_text_key(dataset, key) for key, _ in _xlsx_rows(path)]
    if len(values) != len(set(values)):
        raise ValueError("Duplicate ID in %s" % path)
    return set(values)


def authoritative_splits(dataset: str) -> dict[str, set[str]]:
    if dataset == "qata":
        return {
            "train": one_column_ids(QATA_TEXT / "train" / "Train_ID.xlsx", dataset),
            "val": one_column_ids(QATA_TEXT / "train" / "Val_ID.xlsx", dataset),
            "test": text_ids(QATA_TEXT / "Test_text_for_Covid19.xlsx", dataset),
        }
    return {
        "train": text_ids(MOSMED_TEXT / "Train_text_MosMedData+ 1.xlsx", dataset),
        "val": text_ids(MOSMED_TEXT / "Val_text_MosMedData+ 1.xlsx", dataset),
        "test": text_ids(MOSMED_TEXT / "Test_text_MosMedData+.xlsx", dataset),
    }
def build(dataset: str) -> None:
    source = PHASE0 / (dataset + ".csv")
    target = OUTPUT / (dataset + ".csv")
    splits = authoritative_splits(dataset)
    if any(splits[left] & splits[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Overlapping annotation splits for %s" % dataset)
    owner = {key: split for split, keys in splits.items() for key in keys}

    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    image_keys = {Path(row["image_path"]).name for row in rows}
    if image_keys != set(owner):
        raise ValueError(
            "%s annotation/image mismatch: annotation-only=%d image-only=%d"
            % (dataset, len(set(owner) - image_keys), len(image_keys - set(owner)))
        )
    for row in rows:
        row["split"] = owner[Path(row["image_path"]).name]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    from collections import Counter
    print("%s: %s; annotation-defined split assignments restored" % (dataset, dict(Counter(row["split"] for row in rows))))


if __name__ == "__main__":
    build("qata")
    build("mosmed")
