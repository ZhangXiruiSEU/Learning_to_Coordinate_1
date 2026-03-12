import json
import os
import sys
import time as TM
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args(argv):
    out_path = os.environ.get("VERIFY_SUMMARY_JSON", "verify_regression_summary.json")
    task_idx = os.environ.get("VERIFY_TASK_IDX")
    if len(argv) >= 2:
        out_path = argv[1]
    if len(argv) >= 3:
        task_idx = argv[2]
    return out_path, task_idx


def main():
    import verify_jax_vs_original as verify_mod

    t0 = TM.time()
    out_path, task_idx = _parse_args(sys.argv)
    summary = verify_mod.verify_jax_planner(task_idx=task_idx, show_plots=False)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote regression summary to {out_path} in {(TM.time() - t0):.2f}s")


if __name__ == "__main__":
    main()
