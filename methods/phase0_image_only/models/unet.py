"""Thin wrapper around the frozen upstream U-Net training implementation."""

from ._upstream import import_upstream, pairs_for, read_manifest

SOURCE = "upstream/Summer2026Research/UNETmodel/U-Net-Model.py"
upstream = import_upstream("phase0_frozen_unet", SOURCE)


def dataset_config(manifest, dataset):
    cases = read_manifest(manifest)
    return upstream.DatasetConfig(
        dataset,
        pairs_for(cases, "train"),
        pairs_for(cases, "val"),
    )
