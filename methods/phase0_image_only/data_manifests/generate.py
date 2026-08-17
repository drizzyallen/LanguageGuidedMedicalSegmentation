"""Generate immutable manifests from the existing, already-frozen splits."""

from __future__ import annotations

import csv
import argparse
import hashlib
import os
import re
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
FIELDS = ["sample_id", "source_id", "image_path", "mask_path", "text_path_or_id", "split", "image_sha256", "mask_sha256"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_id(dataset, image_stem):
    if dataset == "qata" and image_stem.startswith("sub-"):
        match = re.match(r"(sub-S\d+)", image_stem)
        if match:
            return match.group(1)
    if dataset == "mosmed" and image_stem.startswith(("Morozov_", "Jun_")):
        return re.sub(r"_\d+$", "", image_stem)
    # No defensible patient/scan grouping is encoded for QaTa covid_N,
    # BUSI, or MosMed bjorke_N names; use the image as the analysis unit.
    return image_stem


def make_row(dataset, split, image, mask):
    image_id = image.stem
    source_id = cluster_id(dataset, image_id)
    return {"sample_id": f"{dataset}_{split}_{image_id}", "source_id": source_id,
            "image_path": os.path.relpath(image, OUTPUT), "mask_path": os.path.relpath(mask, OUTPUT),
            "text_path_or_id": "", "split": split, "image_sha256": sha256(image), "mask_sha256": sha256(mask)}


def directory_rows(dataset, roots, mask_prefix=""):
    rows = []
    for split, root, image_dir, mask_dir in roots:
        images, masks = root / image_dir, root / mask_dir
        image_paths = sorted(images.glob("*.png"))
        expected = {f"{mask_prefix}{image.name}" for image in image_paths}
        actual = {mask.name for mask in masks.glob("*.png")}
        if expected != actual:
            raise ValueError(f"Unpaired {dataset}/{split}: images-only={len(expected-actual)}, masks-only={len(actual-expected)}")
        rows.extend(make_row(dataset, split, image, masks / f"{mask_prefix}{image.name}") for image in image_paths)
    return rows


def mosmed_rows(datasets):
    root, rows = datasets / "MosMedData+", []
    for split in ("train", "val", "test"):
        with (root / f"mosmed_{split}_split.csv").open(newline="") as stream:
            for item in csv.DictReader(stream):
                image = root / "frames" / item["original_image_filename"]
                mask = root / "masks" / item["mask_filename"]
                if not image.is_file() or not mask.is_file():
                    raise FileNotFoundError(f"Missing MosMed pair: {image} | {mask}")
                rows.append(make_row("mosmed", split, image, mask))
    return rows


def validate(rows):
    ids = [item["sample_id"] for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id")
    ownership = {}
    for item in rows:
        key = item["image_sha256"]
        previous = ownership.setdefault(key, item["split"])
        if previous != item["split"]:
            raise ValueError(f"Identical image bytes occur across splits: {previous} and {item['split']}")


def write(name, rows):
    validate(rows)
    with (OUTPUT / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(name, {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    datasets = parser.parse_args().datasets_root.resolve()
    qata, busi = datasets / "QaTa-COV19-v2", datasets / "BUSI" / "split"
    write("qata", directory_rows("qata", [
        ("train", qata / "train_and_val" / "train_subset", "Images", "Ground-truths"),
        ("val", qata / "train_and_val" / "val", "Images", "Ground-truths"),
        ("test", qata / "Test Set", "Images", "Ground-truths")], "mask_"))
    write("mosmed", mosmed_rows(datasets))
    write("busi", directory_rows("busi", [(split, busi / split, "images", "labels") for split in ("train", "val", "test")]))


if __name__ == "__main__":
    main()
