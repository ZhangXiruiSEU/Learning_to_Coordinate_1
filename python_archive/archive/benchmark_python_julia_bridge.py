import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "archive"
for _path in (ROOT, ARCHIVE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from export_subp2_snapshot_for_julia import build_parser as build_export_parser
from export_subp2_snapshot_for_julia import build_snapshot_parser, build_context
from scan_subp2_snapshot import (
    build_original_step_reference,
    build_step_problem,
    capture_subp2_snapshot,
    pack_original_para2,
)
from run_julia_subp2_fullflow_experiment import build_step_export

JULIA_SCRIPT = ROOT / "julia_subp2" / "solve_step_madnlp_jump_native_eq.jl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Python<->Julia bridge overhead for single-step SubP2 solve."
    )
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-iter-admm", type=int, default=3)
    parser.add_argument("--target-admm-iter", type=int, default=1)
    parser.add_argument("--target-step", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--acceptable-tol", type=float, default=1e-4)
    parser.add_argument("--acceptable-iter", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--keep-json", action="store_true")
    return parser.parse_args()


def prepare_payload(args):
    snapshot_defaults = build_snapshot_parser().parse_args([])
    export_defaults = build_export_parser().parse_args([])
    for key, value in vars(snapshot_defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    for key, value in vars(export_defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    ctx = build_context(args)
    planner = ctx["planner"]
    orig_planner = ctx["orig_planner"]
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    dims, x_init, params_t = build_step_problem(planner, para2, args.target_step)
    orig_para2 = pack_original_para2(para2, orig_planner)
    orig_sol = orig_planner.ADMM_SubP2(orig_para2)
    x_orig = build_original_step_reference(orig_sol, planner, args.target_step)

    meta = {
        "task_idx": int(args.task_idx),
        "horizon": int(args.horizon),
        "target_admm_iter": int(args.target_admm_iter),
        "target_step": int(args.target_step),
        "decision_dim": int(np.array(x_init).shape[0]),
        "eq_dim": int(np.array(planner.ipoptax_equality(x_init, params_t, dims)).shape[0]),
        "ineq_dim": int(np.array(planner.ipoptax_inequality(x_init, params_t, dims)).shape[0]),
        "nxl": int(planner.nxl),
        "nul": int(planner.nul),
        "nxi": int(planner.nxi),
        "nui": int(planner.nui),
        "nq": int(planner.nq),
        "num_dis": int(planner.num_dis),
    }
    t0 = time.perf_counter()
    payload = build_step_export(
        planner,
        dims,
        np.array(x_init, dtype=float),
        np.array(x_orig, dtype=float),
        {k: np.array(v) for k, v in params_t.items()},
        meta,
    )
    build_ms = (time.perf_counter() - t0) * 1000.0
    return payload, build_ms


def one_run(payload, args, rep_idx):
    if args.keep_json:
        in_path = ROOT / "julia_subp2" / f"bridge_bench_{rep_idx}.json"
        out_path = ROOT / "julia_subp2" / f"bridge_bench_{rep_idx}_out.json"
    else:
        in_fd, in_name = tempfile.mkstemp(prefix=f"bridge_bench_{rep_idx}_", suffix=".json")
        os.close(in_fd)
        out_fd, out_name = tempfile.mkstemp(prefix=f"bridge_bench_{rep_idx}_out_", suffix=".json")
        os.close(out_fd)
        in_path = Path(in_name)
        out_path = Path(out_name)

    t_dump = time.perf_counter()
    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    dump_ms = (time.perf_counter() - t_dump) * 1000.0

    cmd = [
        "julia",
        str(JULIA_SCRIPT),
        str(in_path),
        str(out_path),
        f"--max-iter={args.max_iter}",
        f"--acceptable-tol={args.acceptable_tol}",
        f"--acceptable-iter={args.acceptable_iter}",
        f"--tol={args.tol}",
    ]
    env = os.environ.copy()
    env.setdefault("JULIA_DEPOT_PATH", "/tmp/julia-depot:/home/mpc/.julia")

    t_sub = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    subprocess_ms = (time.perf_counter() - t_sub) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Julia solve failed on repeat {rep_idx}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    t_load = time.perf_counter()
    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    load_ms = (time.perf_counter() - t_load) * 1000.0

    if not args.keep_json:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)

    julia_wall_ms = float(out["wall_ms"])
    bridge_overhead_ms = subprocess_ms - julia_wall_ms
    return {
        "dump_ms": dump_ms,
        "subprocess_ms": subprocess_ms,
        "load_ms": load_ms,
        "julia_wall_ms": julia_wall_ms,
        "bridge_overhead_ms": bridge_overhead_ms,
        "iter": int(out["iter"]),
        "eq_inf": float(out["eq_inf"]),
        "ineq_vio": float(out["ineq_vio"]),
        "orig_rmse": float(out["orig_rmse"]),
    }


def main():
    args = parse_args()
    payload, build_ms = prepare_payload(args)
    rows = []
    print(f"[bridge-bench] build_payload_ms={build_ms:.3f}")
    for rep in range(args.repeats):
        row = one_run(payload, args, rep)
        rows.append(row)
        print(
            f"[bridge-bench] rep={rep} dump_ms={row['dump_ms']:.3f} "
            f"subprocess_ms={row['subprocess_ms']:.3f} julia_wall_ms={row['julia_wall_ms']:.3f} "
            f"bridge_overhead_ms={row['bridge_overhead_ms']:.3f} load_ms={row['load_ms']:.3f} "
            f"iter={row['iter']} rmse={row['orig_rmse']:.3e}"
        )

    avg = {
        key: float(np.mean([r[key] for r in rows]))
        for key in ["dump_ms", "subprocess_ms", "load_ms", "julia_wall_ms", "bridge_overhead_ms", "orig_rmse"]
    }
    print("[bridge-bench] avg=", json.dumps(avg, indent=2))


if __name__ == "__main__":
    main()
