"""Common gate utilities; model-specific behavior stays in each gate runner."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
import torch

from adapter import SampleRecord


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_mask(record: SampleRecord, size: int = 224):
    image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError("Unreadable image/mask: %s" % record.sample_id)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().div_(255.0)
    mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
    return image, mask


def soft_dice(probability: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return ((2 * intersection + epsilon) / (denominator + epsilon)).mean()


def hard_dice(probability: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    return soft_dice((probability >= 0.5).float(), target, epsilon)


def nonzero_gradient_summary(model: torch.nn.Module, name_fragments: Iterable[str]) -> Dict[str, float]:
    fragments = tuple(name_fragments)
    values = {}
    for name, parameter in model.named_parameters():
        if any(fragment in name.lower() for fragment in fragments) and parameter.grad is not None:
            value = float(parameter.grad.detach().abs().sum().cpu())
            if value > 0:
                values[name] = value
    return values


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)

