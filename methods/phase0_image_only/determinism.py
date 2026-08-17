"""Deterministic execution utilities for frozen benchmark runs."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch


def seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure(seed: int) -> dict[str, object]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def deterministic_loader_class(seed: int, base_class):
    """Return a DataLoader subclass with explicit worker and sampler seeds."""
    class DeterministicDataLoader(base_class):
        count = 0

        def __init__(self, *args, **kwargs):
            loader_seed = seed + type(self).count
            type(self).count += 1
            generator = torch.Generator()
            generator.manual_seed(loader_seed)
            kwargs.setdefault("generator", generator)
            kwargs.setdefault("worker_init_fn", seed_worker)
            super().__init__(*args, **kwargs)

    return DeterministicDataLoader


@contextlib.contextmanager
def patched_dataloaders(upstream, seed: int):
    """Patch loader construction only while invoking frozen upstream code."""
    original_torch_loader = torch.utils.data.DataLoader
    original_upstream_loader = getattr(upstream, "DataLoader", None)
    deterministic = deterministic_loader_class(seed, original_torch_loader)
    torch.utils.data.DataLoader = deterministic
    if original_upstream_loader is not None:
        upstream.DataLoader = deterministic
    try:
        yield
    finally:
        torch.utils.data.DataLoader = original_torch_loader
        if original_upstream_loader is not None:
            upstream.DataLoader = original_upstream_loader
