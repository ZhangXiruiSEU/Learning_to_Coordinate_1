import argparse
import json
import math
import sys
import time
from pathlib import Path

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipoptax import solver as ipoptax_solver


def build_cfg(n_agents=10):
    angles = np.linspace(0.0, 2.0 * math.pi, n_agents, endpoint=False)
    p_ref = np.stack(
        [
            np.cos(angles),
            np.sin(angles),
            0.15 * np.sin(2.0 * angles),
        ],
        axis=1,
    )
    p_ref = p_ref / np.linalg.norm(p_ref, axis=1, keepdims=True)
    u_ref = np.stack(
        [
            0.25 * np.cos(2.0 * angles),
            0.18 * np.sin(3.0 * angles),
            -0.12 * np.cos(angles),
        ],
        axis=1,
    )
    rng = np.random.default_rng(0)
    p0 = p_ref + 0.05 * rng.normal(size=p_ref.shape)
    p0 = p0 / np.linalg.norm(p0, axis=1, keepdims=True)
    u0 = u_ref + 0.03 * rng.normal(size=u_ref.shape)
    x0 = np.concatenate([np.concatenate([p0[i], u0[i]]) for i in range(n_agents)])
    return {
        "n_agents": n_agents,
        "wx": 40.0,
        "wu": 15.0,
        "wc": 0.25,
        "min_sep": 0.45,
        "u_bound": 0.8,
        "thrust_max": 1.0,
        "thrust_min2": 0.04,
        "safe_r": 0.35,
        "obstacles": [[1.7, 0.0], [-1.55, 0.25]],
        "target_u": np.sum(u_ref, axis=0).tolist(),
        "p_ref": p_ref.tolist(),
        "u_ref": u_ref.tolist(),
        "x0": x0.tolist(),
    }


def solve_ipopt(cfg):
    n = int(cfg["n_agents"])
    wx = float(cfg["wx"])
    wu = float(cfg["wu"])
    wc = float(cfg["wc"])
    min_sep = float(cfg["min_sep"])
    u_bound = float(cfg["u_bound"])
    thrust_max = float(cfg["thrust_max"])
    thrust_min2 = float(cfg["thrust_min2"])
    safe_r = float(cfg["safe_r"])
    p_ref = np.array(cfg["p_ref"], dtype=float)
    u_ref = np.array(cfg["u_ref"], dtype=float)
    x0 = np.array(cfg["x0"], dtype=float)
    obs = np.array(cfg["obstacles"], dtype=float)
    target_u = np.array(cfg["target_u"], dtype=float)

    x = ca.MX.sym("x", 6 * n)

    def p(i):
        off = 6 * i
        return x[off : off + 3]

    def u(i):
        off = 6 * i + 3
        return x[off : off + 3]

    obj = 0
    g = []
    lbg = []
    ubg = []

    for i in range(n):
        pi = p(i)
        ui = u(i)
        obj += 0.5 * wx * ca.sumsqr(pi - p_ref[i])
        obj += 0.5 * wu * ca.sumsqr(ui - u_ref[i])
        obj += wc * ca.dot(pi, ui) ** 2
        g.append(ca.sumsqr(pi))
        lbg.append(1.0)
        ubg.append(1.0)
        for k in range(obs.shape[0]):
            g.append(safe_r**2 - ((pi[0] - obs[k, 0]) ** 2 + (pi[1] - obs[k, 1]) ** 2))
            lbg.append(-ca.inf)
            ubg.append(0.0)
        for j in range(3):
            g.append(ui[j] - u_bound)
            lbg.append(-ca.inf)
            ubg.append(0.0)
            g.append(-ui[j] - u_bound)
            lbg.append(-ca.inf)
            ubg.append(0.0)
        g.append(ca.sumsqr(ui) - thrust_max**2)
        lbg.append(-ca.inf)
        ubg.append(0.0)
        g.append(thrust_min2 - ca.sumsqr(ui))
        lbg.append(-ca.inf)
        ubg.append(0.0)

    sum_u = sum([u(i) for i in range(n)])
    for j in range(3):
        g.append(sum_u[j])
        lbg.append(float(target_u[j]))
        ubg.append(float(target_u[j]))

    for i in range(n):
        for j in range(i + 1, n):
            g.append(min_sep**2 - ca.sumsqr(p(i) - p(j)))
            lbg.append(-ca.inf)
            ubg.append(0.0)

    nlp = {"x": x, "f": obj, "g": ca.vertcat(*g)}
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 200,
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.acceptable_iter": 5,
    }

    t_build = time.perf_counter()
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    build_ms = (time.perf_counter() - t_build) * 1000.0

    def run_once():
        t = time.perf_counter()
        sol = solver(x0=x0, lbg=lbg, ubg=ubg)
        ms = (time.perf_counter() - t) * 1000.0
        return ms, float(sol["f"])

    solve1_ms, f1 = run_once()
    solve2_ms, f2 = run_once()
    return {
        "build_ms": build_ms,
        "run1": {"solve_ms": solve1_ms, "objective": f1},
        "run2": {"solve_ms": solve2_ms, "objective": f2},
    }


def make_jax_model(cfg):
    n = int(cfg["n_agents"])
    wx = jnp.array(float(cfg["wx"]))
    wu = jnp.array(float(cfg["wu"]))
    wc = jnp.array(float(cfg["wc"]))
    min_sep = jnp.array(float(cfg["min_sep"]))
    u_bound = jnp.array(float(cfg["u_bound"]))
    thrust_max = jnp.array(float(cfg["thrust_max"]))
    thrust_min2 = jnp.array(float(cfg["thrust_min2"]))
    safe_r = jnp.array(float(cfg["safe_r"]))
    p_ref = jnp.array(np.array(cfg["p_ref"], dtype=np.float64))
    u_ref = jnp.array(np.array(cfg["u_ref"], dtype=np.float64))
    obs = jnp.array(np.array(cfg["obstacles"], dtype=np.float64))
    target_u = jnp.array(np.array(cfg["target_u"], dtype=np.float64))
    x0 = jnp.array(np.array(cfg["x0"], dtype=np.float64))

    def split(x):
        xu = x.reshape((n, 6))
        p = xu[:, :3]
        u = xu[:, 3:]
        return p, u

    def f(x):
        p, u = split(x)
        track = 0.5 * wx * jnp.sum((p - p_ref) ** 2) + 0.5 * wu * jnp.sum((u - u_ref) ** 2)
        coup = wc * jnp.sum((jnp.sum(p * u, axis=1)) ** 2)
        return track + coup

    def c(x):
        p, u = split(x)
        eq_norm = jnp.sum(p**2, axis=1) - 1.0
        sum_u = jnp.sum(u, axis=0) - target_u
        return jnp.concatenate([eq_norm, sum_u], axis=0)

    def g(x):
        p, u = split(x)
        # obstacles (n * nobs)
        dp = p[:, None, :2] - obs[None, :, :]
        obs_ineq = safe_r**2 - jnp.sum(dp**2, axis=2)

        # pair spacing
        idx_i, idx_j = np.triu_indices(n, k=1)
        idx_i = jnp.array(idx_i)
        idx_j = jnp.array(idx_j)
        pair_ineq = min_sep**2 - jnp.sum((p[idx_i] - p[idx_j]) ** 2, axis=1)

        # box on u
        u_upper = u - u_bound
        u_lower = -u - u_bound

        # thrust
        u_norm2 = jnp.sum(u**2, axis=1)
        thrust_upper = u_norm2 - thrust_max**2
        thrust_lower = thrust_min2 - u_norm2

        return jnp.concatenate(
            [
                obs_ineq.reshape(-1),
                pair_ineq,
                u_upper.reshape(-1),
                u_lower.reshape(-1),
                thrust_upper,
                thrust_lower,
            ],
            axis=0,
        )

    return f, c, g, x0


def solve_jax_ipoptax(cfg, variant="baseline"):
    f, c, g, x0 = make_jax_model(cfg)
    g0 = g(x0)
    ws_s = jnp.maximum(-g0 + 1e-3, 1e-3)
    ws_y = jnp.zeros_like(c(x0))
    ws_z = jnp.ones_like(ws_s) * 1e-2

    kwargs = dict(
        f=f,
        c=c,
        g=g,
        ws_x=x0,
        ws_s=ws_s,
        ws_y=ws_y,
        ws_z=ws_z,
        max_iterations=200,
        max_kkt_violation=1e-8,
        tau_min=0.99,
        mu_min=1e-8,
        min_delta=1e-4,
        gamma_y=1e-4,
        gamma_z=1e-2,
        line_search_factor=0.2,
        line_search_min_step_size=1e-10,
        print_logs=False,
        trace_length=0,
        soft_stop_enabled=False,
        plateau_stop_enabled=False,
    )

    if variant == "soft_stop":
        kwargs.update(
            soft_stop_enabled=True,
            soft_mu_tol=1e-8,
            soft_comp_tol=1e-8,
            soft_dual_tol=5e-5,
            soft_eq_tol=1e-6,
            soft_ineq_tol=1e-6,
        )
    elif variant == "plateau_stop":
        kwargs.update(
            plateau_stop_enabled=True,
            plateau_min_iterations=12,
            plateau_patience=8,
            plateau_phi_tol=1e-7,
            plateau_ineq_tol=1e-7,
        )
    elif variant != "baseline":
        raise ValueError(f"unknown variant: {variant}")

    def run_once():
        t = time.perf_counter()
        out = ipoptax_solver.solve(**kwargs)
        # force synchronization
        x = np.array(out["x"])
        s = np.array(out["s"])
        y = np.array(out["y"])
        z = np.array(out["z"])
        del s, y, z
        ms = (time.perf_counter() - t) * 1000.0

        xj = jnp.array(x)
        obj = float(f(xj))
        eq_inf = float(jnp.max(jnp.abs(c(xj))))
        ineq_vio = float(jnp.max(jnp.maximum(g(xj), 0.0)))
        return ms, obj, bool(out["converged"]), int(out["iteration"]), eq_inf, ineq_vio

    solve1_ms, f1, conv1, it1, eq1, ineq1 = run_once()
    solve2_ms, f2, conv2, it2, eq2, ineq2 = run_once()
    return {
        "run1": {
            "solve_ms": solve1_ms,
            "objective": f1,
            "converged": conv1,
            "iterations": it1,
            "eq_inf": eq1,
            "ineq_vio": ineq1,
        },
        "run2": {
            "solve_ms": solve2_ms,
            "objective": f2,
            "converged": conv2,
            "iterations": it2,
            "eq_inf": eq2,
            "ineq_vio": ineq2,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jax-variant",
        default="all",
        choices=["all", "baseline", "soft_stop", "plateau_stop"],
    )
    args = parser.parse_args()

    cfg = build_cfg()
    ipopt = solve_ipopt(cfg)
    variants = (
        ["baseline", "soft_stop", "plateau_stop"]
        if args.jax_variant == "all"
        else [args.jax_variant]
    )
    jax_runs = {name: solve_jax_ipoptax(cfg, variant=name) for name in variants}
    print(json.dumps({"ipopt": ipopt, "jax_ipoptax": jax_runs}, indent=2))


if __name__ == "__main__":
    main()
