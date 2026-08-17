"""Thin wrapper around the frozen upstream TriPathLesionNet implementation."""

from ._upstream import import_upstream, pairs_for, read_manifest

SOURCE = "upstream/Summer2026Research/TriPathLesionNetmodel/TriPathLesionNet.py"
upstream = import_upstream("phase0_frozen_tripath", SOURCE)
TriPathLesionNet = upstream.TriPathLesionNet
ModelConfig = upstream.ModelConfig
LossConfig = upstream.LossConfig


def dataset_split(manifest, dataset):
    cases = read_manifest(manifest)
    return upstream.DatasetSplit(
        dataset,
        pairs_for(cases, "train"),
        pairs_for(cases, "val"),
        pairs_for(cases, "test"),
    )
