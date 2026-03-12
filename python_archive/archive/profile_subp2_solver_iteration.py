import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "archive"
for _path in (ROOT, ARCHIVE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import jax
import jax.numpy as jnp
import numpy as np

from ipoptax.linalg_helpers import project_psd_cone
from ipoptax.solver import LinearSystemFormulation, solve as ipoptax_solve
from scan_subp2_snapshot import (
    build_context,
    build_parser as build_snapshot_parser,
    build_step_problem,
    capture_subp2_snapshot,
)


def build_parser():
    p = argparse.ArgumentParser(description="Profile one SubP2 solver iteration cost breakdown.")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--max-iter-admm", type=int, default=3)
    p.add_argument("--target-admm-iter", type=int, default=1)
    p.add_argument("--target-step", type=int, default=0)
    p.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--solver-formulation", default="STABLE_DIRECT_4x4")
    p.add_argument("--tau-min", type=float, default=0.99)
    p.add_argument("--mu-min", type=float, default=1e-8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--gamma-y", type=float, default=1e-4)
    p.add_argument("--gamma-z", type=float, default=1e-2)
    p.add_argument("--armijo-factor", type=float, default=1e-4)
    p.add_argument("--feasibility-armijo-factor", type=float, default=1e-4)
    p.add_argument("--line-search-factor", type=float, default=0.2)
    p.add_argument("--line-search-min-step-size", type=float, default=1e-10)
    p.add_argument("--s0-cap", type=float, default=0.5)
    p.add_argument("--z0-init", type=float, default=1e-3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--maxit-scan",
        type=str,
        default="",
        help="Optional comma-separated max_iterations sweep for real multi-iteration solve timing, e.g. 1,2,4,8,16,32,64,120",
    )
    return p


def ensure_backend(backend):
    if backend == "cpu":
        import os

        os.environ["JAX_PLATFORMS"] = "cpu"
    elif backend == "cuda":
        import os

        os.environ["JAX_PLATFORMS"] = "cuda"


def block_tree(x):
    return jax.block_until_ready(x)


def bench(name, fn, *args, repeats=5):
    t0 = time.perf_counter()
    out0 = fn(*args)
    block_tree(out0)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    runs = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        out = fn(*args)
        block_tree(out)
        runs.append((time.perf_counter() - t1) * 1000.0)
    steady_ms = float(np.mean(runs))
    print(f"{name:32s} compile+first={compile_ms:9.3f} ms | steady_avg={steady_ms:9.3f} ms | runs={runs}")


def parse_int_list(raw):
    vals = []
    for part in raw.split(","):
        item = part.strip()
        if item:
            vals.append(int(item))
    return vals


def main():
    args = build_parser().parse_args()
    defaults = build_snapshot_parser().parse_args([])
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    ensure_backend(args.backend)

    print("[LOG] Building context", flush=True)
    ctx = build_context(args)
    print(f"[LOG] Capturing snapshot admm_iter={args.target_admm_iter}", flush=True)
    para2 = capture_subp2_snapshot(ctx, args.target_admm_iter, args.max_iter_admm)
    planner = ctx["planner"]
    dims, x_init, params_t = build_step_problem(planner, para2, args.target_step)

    f = lambda x: planner.ipoptax_objective(x, params_t, dims)
    c_raw = lambda x: planner.ipoptax_equality(x, params_t, dims)
    g_raw = lambda x: planner.ipoptax_inequality(x, params_t, dims)
    c_scale = params_t["eq_scale"]
    g_scale = params_t["ineq_scale"]
    c = lambda x: c_scale * c_raw(x)
    g = lambda x: g_scale * g_raw(x)

    eps_s = 1e-3
    c0 = c(x_init)
    g0 = g(x_init)
    s = jnp.minimum(jnp.maximum(-g0 + eps_s, eps_s), args.s0_cap)
    z = jnp.full_like(s, args.z0_init)
    y = jnp.zeros_like(c0)
    mu = jnp.maximum(jnp.array(args.mu_min, dtype=x_init.dtype), jnp.array(1e-8, dtype=x_init.dtype))

    def barrier_augmented_lagrangian(xx, ss, yy, zz, mm):
        return f(xx) + jnp.dot(yy, c(xx)) + jnp.dot(zz, g(xx) + ss) - mm * jnp.log(ss).sum()

    grad_al = jax.jit(lambda xx, ss, yy, zz, mm: jax.grad(lambda u: barrier_augmented_lagrangian(u, ss, yy, zz, mm))(xx))
    hess_al = jax.jit(lambda xx, ss, yy, zz, mm: jax.hessian(lambda u: barrier_augmented_lagrangian(u, ss, yy, zz, mm))(xx))
    jac_c = jax.jit(jax.jacfwd(c))
    jac_g = jax.jit(jax.jacfwd(g))
    psd_hess = jax.jit(
        lambda xx, ss, yy, zz, mm: project_psd_cone(
            jax.hessian(lambda u: barrier_augmented_lagrangian(u, ss, yy, zz, mm))(xx),
            delta=args.min_delta,
        )
    )

    def stable_direct_step(xx, ss, yy, zz, mm):
        d1 = grad_al(xx, ss, yy, zz, mm)
        d2 = psd_hess(xx, ss, yy, zz, mm)
        cc = jac_c(xx)
        gg = jac_g(xx)
        x_dim = xx.shape[0]
        s_dim = ss.shape[0]
        y_dim = yy.shape[0]
        z_dim = zz.shape[0]
        lhs = jnp.block(
            [
                [d2, jnp.zeros((x_dim, s_dim)), cc.T, gg.T],
                [jnp.zeros((s_dim, x_dim)), jnp.diag(zz), jnp.zeros((s_dim, y_dim)), jnp.diag(ss)],
                [cc, jnp.zeros((y_dim, s_dim)), -args.gamma_y * jnp.eye(y_dim), jnp.zeros((y_dim, z_dim))],
                [gg, jnp.eye(z_dim), jnp.zeros((z_dim, y_dim)), -args.gamma_z * jnp.eye(z_dim)],
            ]
        )
        rhs = -jnp.concatenate([d1, ss * zz - mm * jnp.ones_like(ss), c(xx), g(xx) + ss])
        return jnp.linalg.solve(lhs, rhs)

    def indirect_2x2_step(xx, ss, yy, zz, mm):
        d1 = grad_al(xx, ss, yy, zz, mm)
        d2 = psd_hess(xx, ss, yy, zz, mm)
        cc = jac_c(xx)
        gg = jac_g(xx)
        sigma = jnp.diag(zz / (ss + args.gamma_z * zz))
        y_dim = yy.shape[0]
        lhs = jnp.block(
            [
                [d2 + gg.T @ sigma @ gg, cc.T],
                [cc, -args.gamma_y * jnp.eye(y_dim)],
            ]
        )
        rhs = -jnp.concatenate([d1 + gg.T @ sigma @ (g(xx) + (mm / zz)), c(xx)])
        return jnp.linalg.solve(lhs, rhs)

    stable_direct_step = jax.jit(stable_direct_step)
    indirect_2x2_step = jax.jit(indirect_2x2_step)

    solve_one_iter = jax.jit(
        lambda xx, ss, yy, zz: ipoptax_solve(
            f=f,
            c=c,
            g=g,
            ws_x=xx,
            ws_s=ss,
            ws_y=yy,
            ws_z=zz,
            max_iterations=1,
            max_kkt_violation=0.0,
            lin_sys_formulation=getattr(LinearSystemFormulation, args.solver_formulation),
            tau_min=args.tau_min,
            mu_min=args.mu_min,
            min_delta=args.min_delta,
            gamma_y=args.gamma_y,
            gamma_z=args.gamma_z,
            armijo_factor=args.armijo_factor,
            feasibility_armijo_factor=args.feasibility_armijo_factor,
            line_search_factor=args.line_search_factor,
            line_search_min_step_size=args.line_search_min_step_size,
            positivity_floor=1e-12,
            print_logs=False,
            trace_length=0,
        )["x"]
    )

    print("")
    print(f"[PROFILE] task={args.task_idx} horizon={args.horizon} admm_iter={args.target_admm_iter} step={args.target_step}")
    print(f"[PROFILE] x_dim={x_init.shape[0]} eq_dim={c0.shape[0]} ineq_dim={g0.shape[0]}")
    print("")
    bench("grad(al_x)", grad_al, x_init, s, y, z, mu, repeats=args.repeats)
    bench("hessian(al_x)", hess_al, x_init, s, y, z, mu, repeats=args.repeats)
    bench("jacfwd(c)", jac_c, x_init, repeats=args.repeats)
    bench("jacfwd(g)", jac_g, x_init, repeats=args.repeats)
    bench("psd(hessian(al_x))", psd_hess, x_init, s, y, z, mu, repeats=args.repeats)
    bench("search_dir stable_direct_4x4", stable_direct_step, x_init, s, y, z, mu, repeats=args.repeats)
    bench("search_dir indirect_2x2", indirect_2x2_step, x_init, s, y, z, mu, repeats=args.repeats)
    bench("full solve one iteration", solve_one_iter, x_init, s, y, z, repeats=args.repeats)

    if args.maxit_scan:
        print("")
        print("[PROFILE] multi-iteration solve sweep")
        maxit_list = parse_int_list(args.maxit_scan)
        solve_cache = {}

        def get_solver(maxit):
            if maxit not in solve_cache:
                solve_cache[maxit] = jax.jit(
                    lambda xx, ss, yy, zz: ipoptax_solve(
                        f=f,
                        c=c,
                        g=g,
                        ws_x=xx,
                        ws_s=ss,
                        ws_y=yy,
                        ws_z=zz,
                        max_iterations=maxit,
                        max_kkt_violation=0.0,
                        lin_sys_formulation=getattr(LinearSystemFormulation, args.solver_formulation),
                        tau_min=args.tau_min,
                        mu_min=args.mu_min,
                        min_delta=args.min_delta,
                        gamma_y=args.gamma_y,
                        gamma_z=args.gamma_z,
                        armijo_factor=args.armijo_factor,
                        feasibility_armijo_factor=args.feasibility_armijo_factor,
                        line_search_factor=args.line_search_factor,
                        line_search_min_step_size=args.line_search_min_step_size,
                        positivity_floor=1e-12,
                        soft_stop_enabled=False,
                        plateau_stop_enabled=False,
                        print_logs=False,
                        trace_length=0,
                    )
                )
            return solve_cache[maxit]

        for maxit in maxit_list:
            solve_fn = get_solver(maxit)
            t0 = time.perf_counter()
            out0 = solve_fn(x_init, s, y, z)
            jax.block_until_ready(out0["x"])
            compile_ms = (time.perf_counter() - t0) * 1000.0
            runs = []
            iters = []
            for _ in range(args.repeats):
                t1 = time.perf_counter()
                out = solve_fn(x_init, s, y, z)
                jax.block_until_ready(out["x"])
                runs.append((time.perf_counter() - t1) * 1000.0)
                iters.append(int(out["iteration"]))
            steady_ms = float(np.mean(runs))
            avg_iter = float(np.mean(iters))
            ms_per_iter = steady_ms / max(avg_iter, 1.0)
            print(
                f"maxit={maxit:3d} | compile+first={compile_ms:9.3f} ms | "
                f"steady_avg={steady_ms:9.3f} ms | avg_iter={avg_iter:6.1f} | "
                f"ms_per_iter={ms_per_iter:8.3f} | runs={runs}"
            )


if __name__ == "__main__":
    main()
