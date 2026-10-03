"""Only the fairseq.utils.softmax API used by the pinned RecLMIS source.

Calculation matches fairseq v0.10.2 utils.py. Full package installation was
attempted but failed during source builds; no upstream model file is changed.
https://github.com/facebookresearch/fairseq/blob/v0.10.2/fairseq/utils.py
"""
import importlib.util
import sys
import types
import torch
from torch.nn import functional as F


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    return F.softmax(x, dim=dim, dtype=torch.float32)


def install():
    if importlib.util.find_spec("fairseq") is not None:
        return
    package = types.ModuleType("fairseq")
    utils = types.ModuleType("fairseq.utils")
    utils.softmax = softmax
    package.utils = utils
    sys.modules["fairseq"] = package
    sys.modules["fairseq.utils"] = utils
