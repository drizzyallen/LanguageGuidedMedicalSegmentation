"""Post-hoc text-use check on the final Phase 1 checkpoints (audit, not a gate rerun).

For each method/dataset/seed, the selected best checkpoint is evaluated on the
first N validation cases (test data is never loaded) with three text inputs:
  correct  - the case's own report
  null     - LViT-T: all-zero token embeddings; RecLMIS: the empty string
             (the same definitions as the historical 1.3 text-use gate)
  shuffled - the report of the next validation case with a different report
The image is identical across conditions. Reports the maximum and mean
absolute probability change and validation Dice under each condition.

Run in the method's environment, e.g.
  CUDA_VISIBLE_DEVICES=0 python evaluation/text_use_check.py lvit
Writes results/phase1/text_use_check/<method>.json.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "methods" / "phase1_language_guided"
sys.path.insert(0, str(PHASE1))
import main_experiment  # noqa: E402
from adapter import load_records, split_records  # noqa: E402
from gate_common import load_image_mask  # noqa: E402


def dice(probability, mask):
    prediction = (probability >= 0.5).float()
    dims = (1, 2, 3)
    intersection = (prediction * mask).sum(dims)
    return ((2 * intersection + 1e-6) / (prediction.sum(dims) + mask.sum(dims) + 1e-6)).mean().item()


def shuffled_partner(cases, index):
    for step in range(1, len(cases)):
        other = cases[(index + step) % len(cases)]
        if other.text != cases[index].text:
            return other
    raise RuntimeError("All probe reports are identical")


def check(method, dataset, seed, cases_n, device):
    args = argparse.Namespace(method=method, dataset=dataset, seed=seed, device=device, smoke=False)
    model, _, _, extra, forward, _ = main_experiment.build(args)
    if method == "reclmis":
        # The fairseq shim is already installed; a second install() fails its spec lookup.
        import reclmis_compat
        reclmis_compat.install = lambda: None
    run =PHASE1 / "runs" / method / dataset / ("seed_%d" % seed)
    state = torch.load(run / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    cases = split_records(load_records(dataset), "val")[:cases_n]
    images = torch.stack([load_image_mask(c)[0] for c in cases]).to(device)
    masks = torch.stack([load_image_mask(c)[1] for c in cases]).to(device)
    correct_text = torch.stack([extra(c) for c in cases])
    if method == "lvit":
        null_text = torch.zeros_like(correct_text)
    else:
        null_text = torch.stack([extra(replace(c, text="")) for c in cases])
    # The partner keeps its own sample ID so cached embeddings stay keyed to their report.
    shuffled_text = torch.stack([extra(shuffled_partner(cases, i)) for i in range(len(cases))])
    outputs = {}
    with torch.no_grad():
        for name, text in (("correct", correct_text), ("null", null_text), ("shuffled", shuffled_text)):
            chunks = [forward(images[i:i + 4], masks[i:i + 4], text[i:i + 4])[0] for i in range(0, len(cases), 4)]
            outputs[name] = torch.cat(chunks).float()
    result = dict(method=method, dataset=dataset, seed=seed, validation_cases=len(cases),
                  checkpoint_epoch=state["epoch"],
                  dice=dict((name, dice(value, masks)) for name, value in outputs.items()))
    for name in ("null", "shuffled"):
        difference = (outputs["correct"] - outputs[name]).abs()
        changed = ((outputs["correct"] >= 0.5) != (outputs[name] >= 0.5)).float().mean().item()
        result[name] = dict(max_abs_probability_change=difference.max().item(),
                            mean_abs_probability_change=difference.mean().item(),
                            fraction_pixels_label_changed=changed,
                            identical_logits=bool(difference.max().item() == 0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("method", choices=["lvit", "reclmis"])
    parser.add_argument("--cases", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(0)
    results = []
    for dataset in ("qata", "mosmed"):
        for seed in (1001, 1002, 1003):
            item = check(args.method, dataset, seed, args.cases, "cuda:0")
            results.append(item)
            print("%s %s %d  dice correct %.4f null %.4f shuffled %.4f | max |dp| null %.3f shuffled %.3f"
                  % (args.method, dataset, seed, item["dice"]["correct"], item["dice"]["null"],
                     item["dice"]["shuffled"], item["null"]["max_abs_probability_change"],
                     item["shuffled"]["max_abs_probability_change"]), flush=True)
    output = ROOT / "results" / "phase1" / "text_use_check"
    output.mkdir(parents=True, exist_ok=True)
    (output / (args.method + ".json")).write_text(json.dumps(dict(
        description=__doc__.split("\n\n")[1].strip(), results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
