import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax.numpy as jnp
import numpy as np

import JustWorkingOnIt as JAX_Planner
import verify_jax_vs_original

JULIA_SCRIPT = ROOT / "julia_subp2" / "solve_step_madnlp_jump_native_eq.jl"
JULIA_BATCH_SCRIPT = ROOT / "julia_subp2" / "solve_batch_madnlp_jump_native.jl"


def build_parser():
    p = argparse.ArgumentParser(description="Run fullflow verify with Julia MadNLP SubP2 monkeypatch.")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--acceptable-tol", type=float, default=1e-4)
    p.add_argument("--acceptable-iter", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--keep-json", action="store_true")
    return p


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return np.array(obj).tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def build_step_export(planner, dims, x_init, x_orig, params_t, meta):
    eq_init = np.array(planner.ipoptax_equality(x_init, params_t, dims), dtype=float)
    ineq_init = np.array(planner.ipoptax_inequality(x_init, params_t, dims), dtype=float)
    eq_orig = np.array(planner.ipoptax_equality(x_orig, params_t, dims), dtype=float)
    ineq_orig = np.array(planner.ipoptax_inequality(x_orig, params_t, dims), dtype=float)
    export = {
        "meta": meta,
        "step": {
            "dims": list(map(int, dims)),
            "x_init": np.array(x_init, dtype=float),
            "params_t": {k: np.array(v) for k, v in params_t.items()},
            "init_reference_values": {
                "objective": float(np.array(planner.ipoptax_objective(x_init, params_t, dims))),
                "eq_vec": eq_init,
                "ineq_vec": ineq_init,
            },
            "original_step_reference": np.array(x_orig, dtype=float),
            "original_reference_values": {
                "objective": float(np.array(planner.ipoptax_objective(x_orig, params_t, dims))),
                "eq_vec": eq_orig,
                "ineq_vec": ineq_orig,
            },
        },
    }
    return _to_jsonable(export)


def run_julia_step(step_payload, args, suffix):
    if args.keep_json:
        in_path = ROOT / f"julia_subp2" / f"tmp_step_{suffix}.json"
        out_path = ROOT / f"julia_subp2" / f"tmp_step_{suffix}_out.json"
    else:
        in_fd, in_name = tempfile.mkstemp(prefix=f"subp2_step_{suffix}_", suffix=".json")
        os.close(in_fd)
        out_fd, out_name = tempfile.mkstemp(prefix=f"subp2_step_{suffix}_out_", suffix=".json")
        os.close(out_fd)
        in_path = Path(in_name)
        out_path = Path(out_name)

    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(step_payload, f)

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
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Julia step solve failed for {suffix}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    with open(out_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if not args.keep_json:
        try:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
    return result


def run_julia_batch(step_payloads, args, suffix):
    if args.keep_json:
        in_path = ROOT / "julia_subp2" / f"tmp_batch_{suffix}.json"
        out_path = ROOT / "julia_subp2" / f"tmp_batch_{suffix}_out.json"
    else:
        in_fd, in_name = tempfile.mkstemp(prefix=f"subp2_batch_{suffix}_", suffix=".json")
        os.close(in_fd)
        out_fd, out_name = tempfile.mkstemp(prefix=f"subp2_batch_{suffix}_out_", suffix=".json")
        os.close(out_fd)
        in_path = Path(in_name)
        out_path = Path(out_name)

    payload = {"steps": step_payloads}
    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    cmd = [
        "julia",
        str(JULIA_BATCH_SCRIPT),
        str(in_path),
        str(out_path),
        f"--max-iter={args.max_iter}",
        f"--acceptable-tol={args.acceptable_tol}",
        f"--acceptable-iter={args.acceptable_iter}",
        f"--tol={args.tol}",
    ]
    env = os.environ.copy()
    env.setdefault("JULIA_DEPOT_PATH", "/tmp/julia-depot:/home/mpc/.julia")
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Julia batch solve failed for {suffix}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    with open(out_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if not args.keep_json:
        try:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
    return result


def install_julia_subp2_patch(args):
    def patched_jax_admm_subp2(self, Para2_dict):
        planner = self
        N = self.N
        nq = int(self.nq)
        dims = (self.nxl, self.nul, self.nxi, self.nui, nq, int(self.num_dis))

        w_init_batch, params_batch = planner._prepare_subp2_batch(Para2_dict)

        meta = {
            "task_idx": int(args.task_idx),
            "horizon": int(self.N),
            "decision_dim": int(np.array(w_init_batch[0]).shape[0]),
            "eq_dim": int(np.array(planner.ipoptax_equality(w_init_batch[0], {k: v[0] for k, v in params_batch.items()}, dims)).shape[0]),
            "ineq_dim": int(np.array(planner.ipoptax_inequality(w_init_batch[0], {k: v[0] for k, v in params_batch.items()}, dims)).shape[0]),
            "nxl": int(self.nxl),
            "nul": int(self.nul),
            "nxi": int(self.nxi),
            "nui": int(self.nui),
            "nq": int(self.nq),
            "num_dis": int(self.num_dis),
        }

        payloads = []
        for k in range(N + 1):
            params_t = {kk: np.array(vv[k]) for kk, vv in params_batch.items()}
            step_payload = build_step_export(
                planner,
                dims,
                np.array(w_init_batch[k], dtype=float),
                np.array(w_init_batch[k], dtype=float),
                params_t,
                meta | {"target_step": int(k)},
            )
            payloads.append(step_payload)

        batch_out = run_julia_batch(payloads, args, "subp2")

        w_opt = []
        diag_iters = []
        diag_eq = []
        diag_ineq = []
        mode_codes = np.ones((N + 1,), dtype=int)
        for out in batch_out["results"]:
            x_sol = np.array(out["x_sol"], dtype=float)
            w_opt.append(x_sol)
            diag_iters.append(int(out["iter"]))
            diag_eq.append(float(out["eq_inf"]))
            diag_ineq.append(float(out["ineq_vio"]))

        w_opt_batch = jnp.array(np.stack(w_opt, axis=0))
        result = planner._unpack_subp2_results(w_opt_batch)
        result["diag"] = {
            "mode": np.array(["main"] * (N + 1), dtype=object),
            "mode_code": mode_codes,
            "converged": np.ones((N + 1,), dtype=bool),
            "iterations": np.array(diag_iters, dtype=int),
            "eq_inf": np.array(diag_eq, dtype=float),
            "ineq_vio": np.array(diag_ineq, dtype=float),
            "tail_mu": np.full((N + 1,), np.nan, dtype=float),
            "tail_ineq": np.full((N + 1,), np.nan, dtype=float),
            "tail_comp": np.full((N + 1,), np.nan, dtype=float),
            "tail_dual": np.full((N + 1,), np.nan, dtype=float),
            "batch_total_ms": float(batch_out.get("total_ms", np.nan)),
            "batch_build_ms_total": float(batch_out.get("build_ms_total", np.nan)),
            "batch_solve_ms_total": float(batch_out.get("solve_ms_total", np.nan)),
        }
        return result

    JAX_Planner.MPC_Planner.jax_ADMM_SubP2 = patched_jax_admm_subp2


def main():
    args = build_parser().parse_args()
    install_julia_subp2_patch(args)
    t0 = time.perf_counter()
    summary = verify_jax_vs_original.verify_jax_planner(task_idx=args.task_idx, show_plots=args.show_plots)
    total_ms = (time.perf_counter() - t0) * 1000.0
    subp2_total_ms = 0.0
    subp2_build_ms = 0.0
    subp2_solve_ms = 0.0
    for item in summary.get("subp2_summary", []):
        value = item.get("batch_total_ms")
        if value is not None:
            subp2_total_ms += float(value)
        build_value = item.get("batch_build_ms_total")
        if build_value is not None:
            subp2_build_ms += float(build_value)
        solve_value = item.get("batch_solve_ms_total")
        if solve_value is not None:
            subp2_solve_ms += float(solve_value)
    print(f"[JULIA-SUBP2-BUILD] total_ms={subp2_build_ms:.3f}")
    print(f"[JULIA-SUBP2-SOLVE] total_ms={subp2_solve_ms:.3f}")
    print(f"[JULIA-SUBP2-ONLY] total_ms={subp2_total_ms:.3f}")
    print(f"[JULIA-SUBP2-FULLFLOW] total_ms={total_ms:.3f}")
    print(summary)


if __name__ == "__main__":
    main()
