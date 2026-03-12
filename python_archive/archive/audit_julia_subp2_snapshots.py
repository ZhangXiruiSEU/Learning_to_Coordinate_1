import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "archive"
for _path in (ROOT, ARCHIVE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from export_subp2_snapshot_for_julia import _to_jsonable
from scan_subp2_snapshot import (
    build_context,
    build_parser as build_snapshot_parser,
    build_step_problem,
    build_original_step_reference,
    capture_subp2_snapshot,
    pack_original_para2,
)

JULIA_DEPOT = "/tmp/julia-depot:/home/mpc/.julia"
JULIA_AUDIT = ROOT / "julia_subp2" / "archive" / "audit_native_families.jl"
JULIA_SOLVE = ROOT / "julia_subp2" / "solve_step_madnlp_jump_native_eq.jl"


def build_parser():
    p = argparse.ArgumentParser(
        description="Audit Julia native JuMP SubP2 model across multiple exported snapshots."
    )
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--initial-model", type=int, default=4)
    p.add_argument("--max-iter-admm", type=int, default=3)
    p.add_argument("--initial-model-stage1", type=int, default=4)
    p.add_argument("--max-iter-admm-stage1", type=int, default=3)
    p.add_argument("--weight-mode-stage1", type=str, default="n", choices=["n", "f"])
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--backend", type=str, default="cpu", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument(
        "--admm-iters",
        type=str,
        default="",
        help="Comma-separated ADMM iterations to audit. Default: all available.",
    )
    p.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated step indices to audit. Default: all steps 0..horizon.",
    )
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--acceptable-tol", type=float, default=1e-4)
    p.add_argument("--acceptable-iter", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--out", type=str, default="")
    return p


def _parse_int_list(text, default_values):
    if not text:
        return list(default_values)
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def export_snapshot_dict(ctx, admm_iter, step_idx):
    planner = ctx["planner"]
    orig_planner = ctx["orig_planner"]
    args = ctx["args"]

    para2 = capture_subp2_snapshot(ctx, admm_iter, args.max_iter_admm)
    dims, x_init, params_t = build_step_problem(planner, para2, step_idx)
    orig_para2 = pack_original_para2(para2, orig_planner)
    orig_sol = orig_planner.ADMM_SubP2(orig_para2)
    x_orig = build_original_step_reference(orig_sol, planner, step_idx)
    init_eq = planner.ipoptax_equality(x_init, params_t, dims)
    init_ineq = planner.ipoptax_inequality(x_init, params_t, dims)
    orig_eq = planner.ipoptax_equality(x_orig, params_t, dims)
    orig_ineq = planner.ipoptax_inequality(x_orig, params_t, dims)

    export = {
        "meta": {
            "task_idx": int(args.task_idx),
            "horizon": int(args.horizon),
            "max_iter_admm": int(args.max_iter_admm),
            "target_admm_iter": int(admm_iter),
            "target_step": int(step_idx),
            "nxl": int(planner.nxl),
            "nul": int(planner.nul),
            "nxi": int(planner.nxi),
            "nui": int(planner.nui),
            "nq": int(planner.nq),
            "num_dis": int(planner.num_dis),
            "decision_dim": int(len(x_init)),
            "eq_dim": int(planner.ipoptax_equality(x_init, params_t, dims).shape[0]),
            "ineq_dim": int(planner.ipoptax_inequality(x_init, params_t, dims).shape[0]),
        },
        "step": {
            "dims": list(map(int, dims)),
            "x_init": x_init,
            "params_t": params_t,
            "init_reference_values": {
                "objective": float(planner.ipoptax_objective(x_init, params_t, dims)),
                "eq_vec": init_eq,
                "ineq_vec": init_ineq,
            },
            "original_step_reference": x_orig,
            "original_reference_values": {
                "objective": float(planner.ipoptax_objective(x_orig, params_t, dims)),
                "eq_vec": orig_eq,
                "ineq_vec": orig_ineq,
            },
        },
    }
    return _to_jsonable(export)


def run_julia(script_path: Path, snapshot_path: Path, extra_args=None):
    cmd = ["julia", str(script_path), str(snapshot_path)]
    if extra_args:
        cmd.extend(extra_args)
    env = {"JULIA_DEPOT_PATH": JULIA_DEPOT}
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Julia command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def parse_audit_output(text: str):
    worst = 0.0
    for line in text.splitlines():
        if "overall" in line and "max_abs_diff" in line:
            value = float(line.rsplit("=", 1)[1].strip())
            worst = max(worst, value)
    return {"worst_family_diff": worst, "raw_output": text}


def main():
    args = build_parser().parse_args()
    snapshot_defaults = build_snapshot_parser().parse_args([])
    for key, value in vars(snapshot_defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    ctx = build_context(args)
    ctx["args"] = args

    admm_iters = _parse_int_list(args.admm_iters, range(args.max_iter_admm))
    steps = _parse_int_list(args.steps, range(args.horizon + 1))
    records = []

    for admm_iter in admm_iters:
        for step_idx in steps:
            snapshot = export_snapshot_dict(ctx, admm_iter, step_idx)
            with tempfile.TemporaryDirectory(prefix="julia_subp2_audit_") as td:
                td_path = Path(td)
                snap_path = td_path / f"snapshot_a{admm_iter}_k{step_idx}.json"
                solve_out = td_path / f"solve_a{admm_iter}_k{step_idx}.json"
                with open(snap_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f)

                audit_stdout = run_julia(JULIA_AUDIT, snap_path)
                audit = parse_audit_output(audit_stdout)

                solve_stdout = run_julia(
                    JULIA_SOLVE,
                    snap_path,
                    [
                        str(solve_out),
                        f"--max-iter={args.max_iter}",
                        f"--acceptable-tol={args.acceptable_tol}",
                        f"--acceptable-iter={args.acceptable_iter}",
                        f"--tol={args.tol}",
                    ],
                )
                with open(solve_out, "r", encoding="utf-8") as f:
                    solve = json.load(f)

            record = {
                "admm_iter": int(admm_iter),
                "step": int(step_idx),
                "worst_family_diff": float(audit["worst_family_diff"]),
                "status": solve["status"],
                "iter": int(solve["iter"]),
                "wall_ms": float(solve["wall_ms"]),
                "eq_inf": float(solve["eq_inf"]),
                "ineq_vio": float(solve["ineq_vio"]),
                "orig_rmse": float(solve["orig_rmse"]),
                "orig_max_abs": float(solve["orig_max_abs"]),
                "audit_stdout": audit_stdout,
                "solve_stdout": solve_stdout,
            }
            records.append(record)
            print(
                f"[audit] a={admm_iter} k={step_idx} "
                f"family={record['worst_family_diff']:.3e} "
                f"status={record['status']} rmse={record['orig_rmse']:.3e} "
                f"wall_ms={record['wall_ms']:.1f}"
            )

    summary = {
        "task_idx": int(args.task_idx),
        "horizon": int(args.horizon),
        "max_iter_admm": int(args.max_iter_admm),
        "records": records,
    }
    out_path = Path(args.out) if args.out else ROOT / "julia_subp2" / "audit_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
