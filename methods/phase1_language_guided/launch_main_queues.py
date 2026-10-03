"""Launch validated queues detached from VPN/terminal; no upstream changes."""
import argparse
import subprocess
from pathlib import Path
p = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("method", choices=["lvit", "reclmis"])
a = parser.parse_args()
for dataset, gpu in [("qata", 0 if a.method == "lvit" else 1), ("mosmed", 3 if a.method == "lvit" else 4)]:
    validation = p / "smoke_runs_v2" / a.method / dataset / "seed_1001" / "result.json"
    if not validation.exists():
        raise RuntimeError("Missing smoke validation: " + str(validation))
    log = (p / (a.method + "_" + dataset + "_queue.log")).open("a")
    child = subprocess.Popen(["bash", str(p / "run_main_queue.sh"), a.method, dataset, str(gpu)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (p / (a.method + "_" + dataset + "_queue.pid")).write_text(str(child.pid) + "\n")
    print(a.method, dataset, "GPU", gpu, "PID", child.pid)
