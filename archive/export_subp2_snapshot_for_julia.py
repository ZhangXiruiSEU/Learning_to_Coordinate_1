import argparse
import json
import os

import numpy as np

from scan_subp2_snapshot import (
    build_context,
    build_parser as build_snapshot_parser,
    build_step_problem,
    build_original_step_reference,
    capture_subp2_snapshot,
    pack_original_para2,
    primal_feas_metrics,
)


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


def build_parser():
    p = argparse.ArgumentParser(
        description="Export an exact SubP2 snapshot/problem contract for Julia refactor."
    )
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--initial-model", type=int, default=4)
    p.add_argument("--max-iter-admm", type=int, default=3)
    p.add_argument("--initial-model-stage1", type=int, default=4)
    p.add_argument("--max-iter-admm-stage1", type=int, default=3)
    p.add_argument("--weight-mode-stage1", type=str, default="n", choices=["n", "f"])
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--backend", type=str, default="cpu", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--target-admm-iter", type=int, default=0)
    p.add_argument("--target-step", type=int, default=0)
    p.add_argument("--include-batch", action="store_true", help="export full batched SubP2 tensors as well")
    p.add_argument("--out", type=str, default="")
    return p


def main():
    args = build_parser().parse_args()
    snapshot_defaults = build_snapshot_parser().parse_args([])
    for key, value in vars(snapshot_defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    ctx = build_context(args)
    planner = ctx["planner"]
    orig_planner = ctx["orig_planner"]

    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    dims, x_init, params_t = build_step_problem(planner, para2, args.target_step)
    eq_init, ineq_init, feas_init = primal_feas_metrics(planner, x_init, params_t, dims)
    obj_init = float(np.array(planner.ipoptax_objective(x_init, params_t, dims)))
    eq_vec_init = np.array(planner.ipoptax_equality(x_init, params_t, dims), dtype=float)
    ineq_vec_init = np.array(planner.ipoptax_inequality(x_init, params_t, dims), dtype=float)

    orig_para2 = pack_original_para2(para2, orig_planner)
    orig_sol = orig_planner.ADMM_SubP2(orig_para2)
    orig_step_ref = build_original_step_reference(orig_sol, planner, args.target_step)
    eq_orig, ineq_orig, feas_orig = primal_feas_metrics(planner, orig_step_ref, params_t, dims)
    obj_orig = float(np.array(planner.ipoptax_objective(orig_step_ref, params_t, dims)))
    eq_vec_orig = np.array(planner.ipoptax_equality(orig_step_ref, params_t, dims), dtype=float)
    ineq_vec_orig = np.array(planner.ipoptax_inequality(orig_step_ref, params_t, dims), dtype=float)

    export = {
        "meta": {
            "task_idx": args.task_idx,
            "horizon": args.horizon,
            "max_iter_admm": args.max_iter_admm,
            "target_admm_iter": args.target_admm_iter,
            "target_step": args.target_step,
            "nxl": int(planner.nxl),
            "nul": int(planner.nul),
            "nxi": int(planner.nxi),
            "nui": int(planner.nui),
            "nq": int(planner.nq),
            "num_dis": int(planner.num_dis),
            "decision_dim": int(np.array(x_init).shape[0]),
            "eq_dim": int(np.array(planner.ipoptax_equality(x_init, params_t, dims)).shape[0]),
            "ineq_dim": int(np.array(planner.ipoptax_inequality(x_init, params_t, dims)).shape[0]),
        },
        "step": {
            "dims": list(map(int, dims)),
            "x_init": np.array(x_init, dtype=float),
            "params_t": {k: np.array(v) for k, v in params_t.items()},
            "init_metrics": {
                "eq_inf": eq_init,
                "ineq_vio": ineq_init,
                "feas": feas_init,
            },
            "init_reference_values": {
                "objective": obj_init,
                "eq_vec": eq_vec_init,
                "ineq_vec": ineq_vec_init,
            },
            "original_step_reference": np.array(orig_step_ref, dtype=float),
            "original_metrics": {
                "eq_inf": eq_orig,
                "ineq_vio": ineq_orig,
                "feas": feas_orig,
            },
            "original_reference_values": {
                "objective": obj_orig,
                "eq_vec": eq_vec_orig,
                "ineq_vec": ineq_vec_orig,
            },
        },
        "para2_contract": {
            "xl_ideal": np.array(para2["xl_ideal"]),
            "ul_ideal": np.array(para2["ul_ideal"]),
            "xc_ideal": np.array(para2["xc_ideal"]),
            "uc_ideal": np.array(para2["uc_ideal"]),
            "xl_ref": np.array(para2["xl_ref"]),
            "ul_ref": np.array(para2["ul_ref"]),
            "xc_ref": np.array(para2["xc_ref"]),
            "uc_ref_single": np.array(para2["uc_ref_single"]),
            "y_xl": np.array(para2["y_xl"]),
            "y_ul": np.array(para2["y_ul"]),
            "y_xc": np.array(para2["y_xc"]),
            "y_uc": np.array(para2["y_uc"]),
            "rho_lx": float(para2["rho_lx"]),
            "rho_lu": float(para2["rho_lu"]),
            "rho_ix": float(para2["rho_ix"]),
            "rho_iu": float(para2["rho_iu"]),
            "pob1": np.array(para2["pob1"]),
            "pob2": np.array(para2["pob2"]),
            "para_l": np.array(para2["para_l"]),
            "para_i": np.array(para2["para_i"]),
            "i_admm": float(para2["i_admm"]),
        },
    }

    if args.include_batch:
        w_init_batch, params_batch = planner._prepare_subp2_batch(para2)
        export["batch"] = {
            "w_init_batch": np.array(w_init_batch),
            "params_batch": {k: np.array(v) for k, v in params_batch.items()},
        }

    out_path = args.out or (
        f"julia_subp2/subp2_snapshot_task{args.task_idx}_h{args.horizon}_"
        f"a{args.target_admm_iter}_k{args.target_step}.json"
    )
    out_path = os.path.abspath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(export), f)
    print(out_path)


if __name__ == "__main__":
    main()
