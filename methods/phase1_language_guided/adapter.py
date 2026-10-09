"""Shared, read-only image-mask-text records for Phase 1.

Manifests are byte-identical copies of the frozen Phase 0 manifests: IDs,
paths, hashes and split membership all come from Phase 0. Report text is
joined by image basename from the annotation workbooks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
PHASE1_MANIFESTS = Path(__file__).resolve().parent / "data_manifests"
DEFAULT_TEXT_WORKBOOKS = {
    "qata": (
        Path("/data/ramialle/datasets/QaTa-Text/Train_text_for_Covid19.xlsx"),
        Path("/data/ramialle/datasets/QaTa-Text/Test_text_for_Covid19.xlsx"),
    ),
    "mosmed": (
        Path("/data/ramialle/datasets/MosMedData+-Text/Train_text_MosMedData+ 1.xlsx"),
        Path("/data/ramialle/datasets/MosMedData+-Text/Val_text_MosMedData+ 1.xlsx"),
        Path("/data/ramialle/datasets/MosMedData+-Text/Test_text_MosMedData+.xlsx"),
    ),
}
PINNED_SHA256 = {
    str((PHASE1_MANIFESTS / "qata.csv").resolve()): "e12e88a0b570ad143cd6517f35e8d20de0e65e53cbeb46c2c1d4928e3a614a4a",
    str((PHASE1_MANIFESTS / "mosmed.csv").resolve()): "5f2cbe3a7e1cf91cb5dec1697c1d82c65bd8c46860186dfa67f584c5c6d2973e",
    str(DEFAULT_TEXT_WORKBOOKS["qata"][0].resolve()): "35c0f5250c2e8ee5e2bfbefd83823472ede28d5d587add3083449510e3a6e909",
    str(DEFAULT_TEXT_WORKBOOKS["qata"][1].resolve()): "708587c318248d305ff6e61757b4c0d7fec9037c93f86b870a5c76ccbacfc08a",
    str(DEFAULT_TEXT_WORKBOOKS["mosmed"][0].resolve()): "f029c1bd606ab6fef682d6e6d67fb875cd17f8268b834ef3ad5baf444134db33",
    str(DEFAULT_TEXT_WORKBOOKS["mosmed"][1].resolve()): "c8763cb6d636f84de9b14004c9f5c22729f6c49ce8d594af8501c75b4d7c7401",
    str(DEFAULT_TEXT_WORKBOOKS["mosmed"][2].resolve()): "f6b7d569c71af79773bfd926a7ee295ccf8ef058b5159d686323cd091d5a8db1",
}
VALID_SPLITS = ("train", "val", "test")

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS = {"m": _MAIN_NS, "r": _REL_NS}
_CELL_COLUMN = re.compile(r"[A-Z]+")


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    sample_id: str
    source_id: str
    split: str
    image_path: Path
    mask_path: Path
    text: str
    text_source: Path
    text_key: str


def _verify_pinned_file(path: Path) -> None:
    expected = PINNED_SHA256.get(str(path.resolve()))
    if expected is None:
        return
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("Pinned input changed: %s" % path)


def _normalise_text_key(dataset: str, value: str) -> str:
    """Convert workbook column-A values to frozen image basenames."""
    key = Path(str(value).strip()).name
    if dataset == "qata" and key.startswith("mask_"):
        key = key[len("mask_"):]
    return key


def _xlsx_rows(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield columns A and B from every worksheet without an Excel dependency."""
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iterfind(".//m:t", _NS))
                for item in root.findall("m:si", _NS)
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        for sheet in workbook.find("m:sheets", _NS):
            relation = sheet.attrib["{%s}id" % _REL_NS]
            target = targets[relation]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            worksheet = ET.fromstring(archive.read(target))
            first = True
            for row in worksheet.iterfind(".//m:sheetData/m:row", _NS):
                values: Dict[str, str] = {}
                for cell in row.findall("m:c", _NS):
                    column = _CELL_COLUMN.match(cell.attrib["r"]).group(0)
                    kind = cell.attrib.get("t")
                    value_node = cell.find("m:v", _NS)
                    if kind == "inlineStr":
                        value = "".join(
                            node.text or "" for node in cell.iterfind(".//m:t", _NS)
                        )
                    elif value_node is None:
                        value = ""
                    elif kind == "s":
                        value = shared[int(value_node.text)]
                    else:
                        value = value_node.text or ""
                    values[column] = value.strip()
                if first:
                    first = False
                    if values.get("A", "").lower() not in {"image", "filename"}:
                        raise ValueError("Unexpected workbook header in %s" % path)
                    if values.get("B", "").lower() not in {"text", "description"}:
                        raise ValueError("Missing text column B in %s" % path)
                    continue
                yield values.get("A", ""), values.get("B", "")


def load_text_lookup(
    dataset: str, workbook_paths: Optional[Sequence[Path]] = None
) -> Mapping[str, Tuple[str, Path]]:
    """Load a unique, nonempty filename -> (report, source) mapping."""
    if dataset not in DEFAULT_TEXT_WORKBOOKS:
        raise ValueError("Unsupported Phase 1 dataset: %s" % dataset)
    paths = tuple(workbook_paths or DEFAULT_TEXT_WORKBOOKS[dataset])
    lookup: Dict[str, Tuple[str, Path]] = {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError("Missing text workbook: %s" % path)
        _verify_pinned_file(path)
        for raw_key, text in _xlsx_rows(path):
            key = _normalise_text_key(dataset, raw_key)
            if not key:
                raise ValueError("Blank image ID in %s" % path)
            if not text.strip():
                raise ValueError("Blank report for %s in %s" % (key, path))
            previous = lookup.get(key)
            if previous is not None and previous[0] != text:
                raise ValueError("Conflicting reports for %s" % key)
            if previous is not None:
                raise ValueError("Duplicate report ID %s in %s and %s" % (key, previous[1], path))
            lookup[key] = (text.strip(), path)
    return lookup


def load_records(
    dataset: str, workbook_paths: Optional[Sequence[Path]] = None
) -> Tuple[SampleRecord, ...]:
    """Join every frozen manifest row to text by its explicit sample ID key."""
    manifest = (PHASE1_MANIFESTS / (dataset + ".csv")).resolve()
    if not manifest.is_file():
        raise FileNotFoundError("Missing frozen manifest: %s" % manifest)
    _verify_pinned_file(manifest)
    text_lookup = load_text_lookup(dataset, workbook_paths)
    records = []
    seen_ids = set()
    used_text_keys = set()
    with manifest.open(newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise ValueError("Duplicate manifest sample_id: %s" % sample_id)
            seen_ids.add(sample_id)
            split = row["split"]
            if split not in VALID_SPLITS:
                raise ValueError("Invalid split %s for %s" % (split, sample_id))
            image_path = (manifest.parent / row["image_path"]).resolve()
            mask_path = (manifest.parent / row["mask_path"]).resolve()
            text_key = image_path.name
            if text_key not in text_lookup:
                raise ValueError("No report for %s (%s)" % (sample_id, text_key))
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError("Missing image/mask for %s" % sample_id)
            text, source = text_lookup[text_key]
            records.append(
                SampleRecord(
                    dataset=dataset,
                    sample_id=sample_id,
                    source_id=row["source_id"],
                    split=split,
                    image_path=image_path,
                    mask_path=mask_path,
                    text=text,
                    text_source=source,
                    text_key=text_key,
                )
            )
            used_text_keys.add(text_key)
    if len(records) != len(seen_ids):
        raise AssertionError("Manifest join changed sample cardinality")
    return tuple(records)


def split_records(records: Iterable[SampleRecord], split: str) -> Tuple[SampleRecord, ...]:
    if split not in VALID_SPLITS:
        raise ValueError("Invalid split: %s" % split)
    return tuple(record for record in records if record.split == split)


def validate_augmentation_policy(
    split: str,
    augmentation: Optional[Callable] = None,
    geometric: bool = False,
    text_transform: Optional[Callable[[str], str]] = None,
) -> None:
    """Enforce text-safe augmentation before a method-specific loader is built."""
    if split not in VALID_SPLITS:
        raise ValueError("Invalid split: %s" % split)
    if split in {"val", "test"} and augmentation is not None:
        raise ValueError("Validation/test augmentation is disabled in Phase 1")
    if split == "train" and geometric and text_transform is None:
        raise ValueError(
            "Geometric augmentation requires a matching text transform; disable it otherwise"
        )


class TextEmbeddingCache:
    """Opt-in cache whose key includes method, encoder identity, ID, and text.

    Callers must explicitly state that the official method permits deterministic
    text-embedding caching. This class never chooses or changes an encoder.
    """

    def __init__(self, root: Path, method: str, encoder_fingerprint: str, permitted: bool):
        if not permitted:
            raise ValueError("Caching was not established as algorithm-preserving for %s" % method)
        self.root = Path(root).resolve() / method / encoder_fingerprint
        self.method = method
        self.encoder_fingerprint = encoder_fingerprint
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, record: SampleRecord) -> Path:
        digest = hashlib.sha256(
            (self.method + "\0" + self.encoder_fingerprint + "\0" + record.sample_id + "\0" + record.text).encode("utf-8")
        ).hexdigest()
        return self.root / (digest + ".json")

    def get_or_compute(self, record: SampleRecord, encode: Callable[[str], object]) -> object:
        path = self._path(record)
        if path.is_file():
            with path.open() as stream:
                return json.load(stream)["embedding"]
        embedding = encode(record.text)
        payload = {
            "sample_id": record.sample_id,
            "text_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
            "embedding": embedding,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return embedding
