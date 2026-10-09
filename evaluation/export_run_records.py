"""Write the plan's common per-run records under runs/<phase>/<method>/<dataset>/<seed>/.

Adds resolved_config.yaml, environment.txt, train_log.csv and
best_checkpoint.txt next to the shared evaluator's test_summary.json,
per_case_metrics.csv and predictions/. Everything is derived from the
original run directories, which are not modified. Run after
shared_evaluator.py, in the Phase 0 environment (myenv).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "methods" / "phase0_image_only"
PHASE1 = ROOT / "methods" / "phase1_language_guided"
ENVS = Path("/data/ramialle/miniconda3/envs")
SEEDS = (1001, 1002, 1003)
DATASETS = ("qata", "mosmed", "busi")  # BUSI: Phase 0 methods only
METHODS = {"unet": "phase0", "tripath_lesionnet": "phase0", "panoptic_fpn": "phase0",
           "lvit": "phase1", "reclmis": "phase1"}
ENV_NAME = {"phase0": "myenv", "lvit": "phase1_official_lvit", "reclmis": "phase1_official_reclmis"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def environment(name, _cache={}):
    if name not in _cache:
        python = ENVS / name / "bin" / "python"
        header = subprocess.run([str(python), "-c",
                                 "import sys, torch; print('python', sys.version.split()[0]); "
                                 "print('torch', torch.__version__); print('cuda', torch.version.cuda); "
                                 "print('cudnn', torch.backends.cudnn.version())"],
                                capture_output=True, text=True, check=True).stdout
        freeze = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
        _cache[name] = "environment: %s\ninterpreter: %s\nGPU: NVIDIA GeForce RTX 3090\n%s\n# pip freeze\n%s" % (
            name, python, header, freeze)
    return _cache[name]


def write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


def recovered_history(method, dataset, seed, history):
    """Rebuild a complete epoch log from the upstream trainer's terminal output."""
    log = (PHASE0 / "logs" / ("%s_tmux.log" % method)).read_text().splitlines()
    marker = "Output directory: %s" % (PHASE0 / "runs" / method / dataset / ("seed_%d" % seed) / "upstream" / dataset)
    start = log.index(marker)
    header = history.splitlines()[0].split(",")
    rows = []
    for line in log[start + 1:]:
        if line.startswith("Output directory:"):
            break
        if line.startswith("Epoch "):
            parts = line.split()
            values = dict(item.split("=", 1) for item in parts[2:])
            values["epoch"] = str(int(parts[1].split("/")[0]))
            values["learning_rate"] = values.pop("lr")
            rows.append(values)
    if [int(r["epoch"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError("Terminal log does not hold a complete epoch sequence")
    saved = {r.split(",")[0]: r.split(",") for r in history.splitlines()[1:]}
    for epoch, columns in saved.items():
        logged = rows[int(epoch) - 1]
        if abs(float(logged["val_dice"]) - float(columns[header.index("val_dice")])) > 1e-4:
            raise RuntimeError("Terminal log disagrees with history.csv at epoch " + epoch)
    return "\n".join([",".join(header)] + [",".join(r[h] for h in header) for r in rows]) + "\n"


def phase0_records(method, dataset, seed, output):
    run = PHASE0 / "runs" / method / dataset / ("seed_%d" % seed)
    upstream = run / "upstream" / dataset
    summary = json.loads((run / "predictions" / "test_summary.json").read_text())
    config = yaml.safe_load((PHASE0 / "configs" / (method + ".yaml")).read_text())
    resolved = dict(phase="phase0", method=method, dataset=dataset, seed=seed,
                    run_metadata=json.loads((run / "run_metadata.json").read_text()), config=config)
    if (upstream / "config.json").is_file():
        resolved["upstream_resolved_config"] = json.loads((upstream / "config.json").read_text())
    write_yaml(output / "resolved_config.yaml", resolved)
    (output / "environment.txt").write_text(environment(ENV_NAME["phase0"]))
    history = (upstream / "history.csv").read_text()
    first_epoch = history.splitlines()[1].split(",")[0]
    if first_epoch != "1":
        history = recovered_history(method, dataset, seed, history)
        (output / "train_log_note.txt").write_text(
            "upstream/%s/history.csv covers only epochs %s onward: a later relaunch of this run rewrote it.\n"
            "train_log.csv was rebuilt from the trainer's terminal log (methods/phase0_image_only/logs/%s_tmux.log),\n"
            "and its overlapping epochs were checked against history.csv. The original files are unchanged.\n"
            % (dataset, first_epoch, "unet" if method == "unet" else method))
    (output / "train_log.csv").write_text(history)
    checkpoint = upstream / "best.pt"
    (output / "best_checkpoint.txt").write_text(
        "path: %s\nsha256: %s\nepoch: %s\nselection: validation Dice (frozen Phase 0 rule)\n"
        % (checkpoint.relative_to(ROOT), sha256(checkpoint), summary["checkpoint_epoch"]))


def phase1_records(method, dataset, seed, output):
    run = PHASE1 / "runs" / method / dataset / ("seed_%d" % seed)
    config = json.loads((run / "run_config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    resolved = dict(phase="phase1", method=method, dataset=dataset, seed=seed, run_config=config,
                    training_stop=result.get("training_stop"))
    stop = run / "nonfinite_stop.json"
    if stop.is_file():
        resolved["nonfinite_stop"] = json.loads(stop.read_text())
    write_yaml(output / "resolved_config.yaml", resolved)
    (output / "environment.txt").write_text(environment(ENV_NAME[method]))
    history = json.loads((run / "progress.json").read_text())["history"]
    with (output / "train_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_dice", "lr"])
        writer.writeheader()
        writer.writerows(history)
    checkpoint = run / "best_checkpoint.pt"
    if sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash changed since test evaluation: " + str(checkpoint))
    (output / "best_checkpoint.txt").write_text(
        "path: %s\nsha256: %s\nepoch: %s\nbest_val_dice: %s\nselection: %s\n"
        % (checkpoint.relative_to(ROOT), result["checkpoint_sha256"], result["checkpoint_epoch"],
           result["best_val_dice"], config["checkpoint_selection"]))


def main():
    for method, phase in METHODS.items():
        for dataset in DATASETS:
            if dataset == "busi" and phase != "phase0":
                continue
            for seed in SEEDS:
                output = ROOT / "runs" / phase / method / dataset / str(seed)
                if not (output / "test_summary.json").is_file():
                    raise FileNotFoundError("Run shared_evaluator.py first: " + str(output))
                (phase0_records if phase == "phase0" else phase1_records)(method, dataset, seed, output)
                print("records", phase, method, dataset, seed, flush=True)


if __name__ == "__main__":
    main()
