"""Utilities shared by the frozen upstream adapters."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


PHASE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PHASE_ROOT.parents[1]


@dataclass(frozen=True)
class ManifestCase:
    sample_id: str
    source_id: str
    image: Path
    mask: Path
    text_path_or_id: str
    split: str


def import_upstream(module_name: str, relative_path: str) -> ModuleType:
    """Import a pinned source file without modifying or duplicating it."""
    source = REPOSITORY_ROOT / relative_path
    if not source.is_file():
        raise FileNotFoundError(f"Pinned upstream source is missing: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream module from {source}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass decoration consults sys.modules while the module is executing.
    # Match normal import semantics by registering it before exec_module.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[ManifestCase]:
    """Read and validate the shared case manifest.

    Relative image and mask paths are resolved from the manifest directory.
    """
    manifest = Path(path).resolve()
    required = {
        "sample_id", "source_id", "image_path", "mask_path",
        "text_path_or_id", "split", "image_sha256", "mask_sha256",
    }
    cases: list[ManifestCase] = []
    seen: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
            seen.add(sample_id)
            image = Path(row["image_path"])
            mask = Path(row["mask_path"])
            image = image if image.is_absolute() else manifest.parent / image
            mask = mask if mask.is_absolute() else manifest.parent / mask
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(
                    f"Missing manifest pair at line {line_number}: {image} | {mask}"
                )
            if _sha256(image) != row["image_sha256"] or _sha256(mask) != row["mask_sha256"]:
                raise ValueError(f"SHA-256 mismatch at manifest line {line_number}")
            cases.append(ManifestCase(
                sample_id, row["source_id"], image.resolve(), mask.resolve(),
                row["text_path_or_id"], row["split"],
            ))
    if not cases:
        raise ValueError(f"Manifest has no cases: {manifest}")
    return cases


def pairs_for(cases: list[ManifestCase], split: str):
    pairs = [(case.image, case.mask) for case in cases if case.split == split]
    if not pairs:
        raise ValueError(f"No manifest cases for split={split!r}")
    return pairs
