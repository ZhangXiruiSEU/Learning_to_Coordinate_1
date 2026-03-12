import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import casadi as ca
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
JULIA_SCRIPT = ROOT / "julia_subp2" / "archive" / "benchmark_madnlp_synthetic.jl"


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

    g_fun = ca.Function("g_fun", [x], [ca.vertcat(*g)])

    def run_once():
        t = time.perf_counter()
        sol = solver(x0=x0, lbg=lbg, ubg=ubg)
        ms = (time.perf_counter() - t) * 1000.0
        x_sol = np.array(sol["x"]).reshape(-1)
        g_val = np.array(g_fun(x_sol)).reshape(-1)
        lbg_arr = np.array(lbg, dtype=float)
        ubg_arr = np.array(ubg, dtype=float)
        eq_mask = np.isfinite(lbg_arr) & np.isfinite(ubg_arr) & np.isclose(lbg_arr, ubg_arr)
        eq_inf = float(np.max(np.abs(g_val[eq_mask] - lbg_arr[eq_mask]))) if np.any(eq_mask) else 0.0
        ineq_lower_vio = np.maximum(lbg_arr - g_val, 0.0)
        ineq_upper_vio = np.maximum(g_val - ubg_arr, 0.0)
        ineq_vio = float(np.max(np.maximum(ineq_lower_vio, ineq_upper_vio))) if g_val.size else 0.0
        return ms, float(sol["f"]), eq_inf, ineq_vio

    solve1_ms, f1, eq1, ineq1 = run_once()
    solve2_ms, f2, eq2, ineq2 = run_once()
    return {
        "build_ms": build_ms,
        "run1": {"solve_ms": solve1_ms, "objective": f1, "eq_inf": eq1, "ineq_vio": ineq1},
        "run2": {"solve_ms": solve2_ms, "objective": f2, "eq_inf": eq2, "ineq_vio": ineq2},
    }


def solve_madnlp(cfg):
    fd_in, path_in = tempfile.mkstemp(prefix="madnlp_synth_", suffix=".json")
    os.close(fd_in)
    fd_out, path_out = tempfile.mkstemp(prefix="madnlp_synth_out_", suffix=".json")
    os.close(fd_out)
    path_in = Path(path_in)
    path_out = Path(path_out)
    with open(path_in, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    env = os.environ.copy()
    env.setdefault("JULIA_DEPOT_PATH", "/tmp/julia-depot:/home/mpc/.julia")
    cmd = ["julia", str(JULIA_SCRIPT), str(path_in), str(path_out)]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Julia MadNLP benchmark failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    with open(path_out, "r", encoding="utf-8") as f:
        out = json.load(f)
    path_in.unlink(missing_ok=True)
    path_out.unlink(missing_ok=True)
    return out


def main():
    cfg = build_cfg()
    ipopt = solve_ipopt(cfg)
    madnlp = solve_madnlp(cfg)
    print(json.dumps({"ipopt": ipopt, "madnlp": madnlp}, indent=2))


if __name__ == "__main__":
    main()
