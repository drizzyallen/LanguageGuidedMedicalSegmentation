"""Parameters and inference MACs for one method, at its native test input.

All methods are counted with the same tool (thop 0.1.1.post2209072238) on a
batch of one at the resolution and modality the method was trained and tested
with. thop counts module operations (convolutions, linear layers, norms,
activations, pooling); functional tensor products such as attention
matrix multiplications are not counted, so MACs for the attention-based
LViT-T and RecLMIS are lower bounds. FLOPs are reported as 2 x MACs.

Usage (each in the method's environment):
  myenv:                    python model_complexity.py unet|tripath_lesionnet|panoptic_fpn
  phase1_official_lvit:     python model_complexity.py lvit
  phase1_official_reclmis:  python model_complexity.py reclmis
Writes results/final_five_method/model_complexity/<method>.json.
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_five_method" / "model_complexity"


def phase0_model(method):
    phase0 = ROOT / "methods" / "phase0_image_only"
    sys.path.insert(0, str(phase0))
    import export_predictions
    checkpoint = torch.load(phase0 / "runs" / method / "qata" / "seed_1001" / "upstream" / "qata" / "best.pt",
                            map_location="cpu", weights_only=False)
    model = export_predictions.load_model(method, checkpoint, torch.device("cpu"))
    size = 384
    return model, (torch.zeros(1, 1, size, size),), "1x1x384x384 grayscale image", \
        "export_predictions.load_model from the seed-1001 QaTa best checkpoint"


def lvit_model():
    sys.path.insert(0, str(ROOT / "upstream" / "LViT"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    import Config as cfg
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from nets.LViT import LViT
    model = LViT(cfg.get_CTranS_config(), n_channels=cfg.n_channels, n_classes=cfg.n_labels)
    return model, (torch.zeros(1, 3, 224, 224), torch.zeros(1, 10, 768)), \
        "1x3x224x224 image + 1x10x768 precomputed BERT token embeddings", \
        "segmentation network only; the frozen external BERT-base text encoder (bert-embedding) is not counted"


def reclmis_model():
    phase1 = ROOT / "methods" / "phase1_language_guided"
    sys.path.insert(0, str(phase1))
    sys.path.insert(0, str(ROOT / "upstream" / "RecLMIS"))
    from reclmis_compat import install
    install()
    import importlib
    cfg = importlib.import_module("Config_covid19")
    import nets.RecLMIS as module
    module._PT_NAME["ViT-B/32"] = str(phase1 / "assets" / "reclmis" / "ViT-B-32.pt")
    model = module.RecLMIS(cfg, cfg.get_ViT_config(), n_channels=cfg.n_channels, n_classes=cfg.n_labels)
    tokens = torch.zeros(1, cfg.token_len, dtype=torch.long)
    tokens[0, :4] = torch.tensor([49406, 320, 1125, 49407])
    return model, (torch.zeros(1, 3, 224, 224), torch.zeros(1, 1, 224, 224), tokens, (tokens != 0).int()), \
        "1x3x224x224 image + 1x%d CLIP tokens" % cfg.token_len, \
        "whole module as instantiated, including its CLIP text encoder"


def main():
    method = sys.argv[1]
    sys.path.insert(0, os.environ.get("THOP_PATH", ""))
    from thop import profile
    import thop
    if method in ("unet", "tripath_lesionnet", "panoptic_fpn"):
        model, inputs, input_desc, scope = phase0_model(method)
    elif method == "lvit":
        model, inputs, input_desc, scope = lvit_model()
    else:
        model, inputs, input_desc, scope = reclmis_model()
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        macs, _ = profile(model, inputs=inputs, verbose=False)
    result = dict(method=method, parameters=params, trainable_parameters=trainable,
                  macs=float(macs), flops=2.0 * float(macs), input=input_desc, scope=scope,
                  tool="thop " + getattr(thop, "__version__", "0.1.1.post2209072238"),
                  note="thop counts module operations only; functional attention products are not counted")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (method + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print("%-18s params %.2fM  MACs %.2fG  FLOPs %.2fG" % (method, params / 1e6, macs / 1e9, 2 * macs / 1e9))


if __name__ == "__main__":
    main()
