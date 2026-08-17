"""Thin wrapper around the frozen upstream Panoptic FPN implementation."""

from ._upstream import import_upstream, pairs_for, read_manifest

SOURCE = "upstream/Summer2026Research/PanopticFeaturePyramidNetmodel/PFPN.py"
upstream = import_upstream("phase0_frozen_panoptic_fpn", SOURCE)
PanopticFPN = upstream.PanopticFPN
ModelConfig = upstream.ModelConfig


def dataset_split(manifest, dataset):
    cases = read_manifest(manifest)
    return upstream.DatasetSplit(
        dataset,
        pairs_for(cases, "train"),
        pairs_for(cases, "val"),
        pairs_for(cases, "test"),
    )
