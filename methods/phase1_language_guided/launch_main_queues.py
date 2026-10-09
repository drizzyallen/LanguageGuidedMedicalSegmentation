"""Launch validated queues detached from VPN/terminal; no upstream changes.

Usage: launch_main_queues.py lvit --dataset qata=3 --dataset mosmed=3
GPU indices are those visible to nvidia-smi at launch time.
"""
import argparse
import subprocess
from pathlib import Path
p = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("method", choices=["lvit", "reclmis"])
parser.add_argument("--dataset", action="append", required=True, metavar="DATASET=GPU",
                    help="dataset (qata or mosmed) and the GPU index for its seed queue")
a = parser.parse_args()
queues = [item.split("=", 1) for item in a.dataset]
for dataset, gpu in queues:
    if dataset not in ("qata", "mosmed") or not gpu.isdigit():
        parser.error("expected qata=N or mosmed=N, got " + dataset + "=" + gpu)
for dataset, gpu in queues:
    validation = p / "smoke_runs_v2" / a.method / dataset / "seed_1001" / "result.json"
    if not validation.exists():
        raise RuntimeError("Missing smoke validation: " + str(validation))
    log = (p / (a.method + "_" + dataset + "_queue.log")).open("a")
    child = subprocess.Popen(["bash", str(p / "run_main_queue.sh"), a.method, dataset, gpu],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (p / (a.method + "_" + dataset + "_queue.pid")).write_text(str(child.pid) + "\n")
    print(a.method, dataset, "GPU", gpu, "PID", child.pid)
