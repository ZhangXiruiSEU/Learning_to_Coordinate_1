import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import JustWorkingOnIt as JAX_Planner
import verify_jax_vs_original

# -----------------------------------------------------------------------------
# 当前稳定 persistent Julia 路线的 Python 侧总控。
#
# 主要做四件事：
# 1. 启动一个常驻 Julia worker 进程
# 2. 把 JustWorkingOnIt.py 里的 jax_ADMM_SubP2 临时替换成 Julia IPC 版本
# 3. 在上述的 IPC 版本 Python 函数中，完成 SubP2 batch payload 的打包与结果解包
# 4. 跑 fullflow verify / benchmark，并汇总桥接和求解时间
#
# 如果你现在只关心"当前正式主线"，最小阅读顺序建议是：
# 1. JuliaBatchWorker
# 2. build_batch_export_runtime(...)
# 3. install_julia_subp2_patch(...)
# 4. run_once(...)
# 5. main()
#
# 文件也按这个顺序重排过：
# - 主线直接会走的定义尽量靠前
# - 旧 snapshot/export helper 放到文件尾部
# - JuliaBatchWorker 里非主线文件模式/prepare 模式方法放到类尾部
#
# 当前主线里可先跳过的部分：
# - build_step_export(...)
#   更老的"单步完整快照"导出 helper，这个 runner 当前不直接用
# - build_step_export_runtime(...)
#   单步 runtime 导出 helper，当前也不走
# - JuliaBatchWorker.prepare_batch_payload(...)
#   留给 prepare_batch 预热/实验接口，当前正式 benchmark 不调用
# - args.keep_json 分支
#   这是把 payload 先落盘再交给 Julia 的兼容模式，当前稳定主线默认不走
# - result_format 的 legacy fallback 分支
#   当前 worker 默认走紧凑 stacked 回包格式，fallback 主要是兼容旧返回格式
# -----------------------------------------------------------------------------

JULIA_WORKER = ROOT / "julia_subp2" / "worker_batch_madnlp_jump_native.jl"
JULIA_STACKED_RESULT_FORMAT = "stacked_runtime_result_v1"
JULIA_RUNTIME_PARAM_KEYS = (
    "rho_lx",
    "rho_lu",
    "rho_ix",
    "rho_iu",
    "is_terminal",
    "xl_ideal",
    "ul_ideal",
    "y_xl",
    "y_ul",
    "xc_ideal",
    "uc_ideal",
    "y_xc",
    "y_uc",
    "Pt",
    "pob1",
    "pob2",
    "ra",
    "ro",
    "rq",
    "cl0",
    "rl",
    "t_min",
    "t_max",
    "ui_bound",
    "g",
    "ml",
    "mq",
    "fmax",
    "Jl",
    "Jl_inv",
)


# 当前 runner 的 CLI 入口参数。大部分都是 Julia worker 配置和 benchmark 选项。
def build_parser():
    p = argparse.ArgumentParser(description="Run fullflow verify with persistent Julia MadNLP SubP2 worker.")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--acceptable-tol", type=float, default=1e-4)
    p.add_argument("--acceptable-iter", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--keep-json", action="store_true")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--julia-threads", type=int, default=20)
    p.add_argument("--julia-blas-threads", type=int, default=1)
    p.add_argument("--julia-linear-solver", type=str, default="mumps")
    p.add_argument("--julia-kkt-system", type=str, default="default")
    p.add_argument("--julia-callback", type=str, default="default")
    p.add_argument("--julia-thread-schedule", type=str, choices=("default", "dynamic", "static"), default="static")
    p.add_argument("--julia-runtime-metrics", type=str, choices=("full", "none"), default="none")
    return p


# 把 numpy / jax 数组、标量等统一转换成 JSON 可序列化对象。
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


# 把 Julia 回包里的指标数组安全转成 numpy，并把 null 统一映射成 NaN。
def _json_metric_array(values):
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)

# 当前稳定主线直接会走这里。
# 作用：把 planner 侧的 batch 决策变量和运行期参数裁剪、搬到 host，再打成 Julia worker 能直接消费的紧凑 payload。
def build_batch_export_runtime(dims, w_init_batch, params_batch, meta):
    filtered_params = {k: params_batch[k] for k in JULIA_RUNTIME_PARAM_KEYS}
    host_payload = jax.device_get(
        {
            "x_init_batch": w_init_batch,
            "params_batch": filtered_params,
        }
    )
    export = {
        "format": "stacked_runtime_v1",
        "meta": meta,
        "dims": list(map(int, dims)),
        "target_steps": list(range(int(np.asarray(host_payload["x_init_batch"]).shape[0]))),
        "x_init_batch": np.asarray(host_payload["x_init_batch"], dtype=float),
        "params_batch": {
            k: np.asarray(v, dtype=float) for k, v in host_payload["params_batch"].items()
        },
    }
    return _to_jsonable(export)


# Python 侧对常驻 Julia worker 的薄封装。
# 它本身不做求解，只负责：启动进程、发 JSON 请求、收 JSON 回包、关闭进程。
class JuliaBatchWorker:
    # 当前稳定主线必经入口：
    # - 配好 Julia 线程数 / BLAS / solver / 调度策略
    # - 启动 worker_batch_madnlp_jump_native.jl
    # - 通过 ping 确认 worker 真的起来了
    def __init__(
        self,
        julia_threads: int = 0,
        blas_threads: int = 1,
        linear_solver: str = "mumps",
        kkt_system: str = "default",
        callback: str = "default",
        thread_schedule: str = "static",
    ):
        env = os.environ.copy()
        env.setdefault("JULIA_DEPOT_PATH", "/tmp/julia-depot:/home/mpc/.julia")
        if julia_threads and julia_threads > 0:
            env["JULIA_NUM_THREADS"] = str(julia_threads)
        else:
            env.setdefault("JULIA_NUM_THREADS", str(os.cpu_count() or 1))
        env["JULIA_BLAS_THREADS"] = str(max(int(blas_threads), 1))
        env["JULIA_SUBP2_LINEAR_SOLVER"] = str(linear_solver)
        env["JULIA_SUBP2_KKT_SYSTEM"] = str(kkt_system)
        env["JULIA_SUBP2_CALLBACK"] = str(callback)
        env["JULIA_SUBP2_THREAD_SCHEDULE"] = str(thread_schedule)
        self.proc = subprocess.Popen(
            ["julia", str(JULIA_WORKER)],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ping = self._request({"action": "ping"})
        self.nthreads = int(ping.get("result", {}).get("nthreads", -1))
        self.blas_threads = int(ping.get("result", {}).get("blas_threads", -1))
        self.linear_solver = str(ping.get("result", {}).get("linear_solver", linear_solver))
        self.kkt_system = str(ping.get("result", {}).get("kkt_system", kkt_system))
        self.callback = str(ping.get("result", {}).get("callback", callback))
        self.thread_schedule = str(ping.get("result", {}).get("thread_schedule", thread_schedule))

    # 核心阻塞式 IPC。
    # Python 往 stdin 写一行 JSON，然后同步等待 Julia 从 stdout 回一行 JSON。
    # 当前主线就是这条 request/response 通道，不是 socket，也不是共享内存。
    def _request(self, payload):
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = ""
            if self.proc.stderr is not None:
                try:
                    stderr = self.proc.stderr.read()
                except Exception:
                    stderr = ""
            raise RuntimeError(f"Julia worker died or produced no response.\nSTDERR:\n{stderr}")
        resp = json.loads(line)
        if not resp.get("ok", False):
            raise RuntimeError(f"Julia worker error: {resp}")
        return resp

    # 当前稳定主线直接走这里。
    # Python 直接把内存里的 batch payload 发给 Julia worker，不落盘。
    def solve_batch_payload(self, payload, args):
        return self._request(
            {
                "action": "solve_batch",
                "payload": payload,
                "max_iter": int(args.max_iter),
                "acceptable_tol": float(args.acceptable_tol),
                "acceptable_iter": int(args.acceptable_iter),
                "tol": float(args.tol),
                "runtime_metric_mode": str(args.julia_runtime_metrics),
                "result_format": JULIA_STACKED_RESULT_FORMAT,
            }
        )["result"]

    # 关闭常驻 Julia worker。正常情况先发 shutdown，再 terminate/kill 兜底。
    def close(self):
        if self.proc.poll() is not None:
            return
        try:
            self._request({"action": "shutdown"})
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 以下两个方法不是当前稳定主线必经路径，统一放到类尾部。
    # 主线默认是 solve_batch_payload(...)；只有兼容/实验模式才会走这里。
    # ------------------------------------------------------------------

    # 可先跳过：文件路径模式。
    # 只有 keep_json=True 时才会先把 payload 写到磁盘，再让 Julia 读文件。
    # 当前稳定主线默认不走这条。
    def solve_batch(self, in_path, args):
        return self._request(
            {
                "action": "solve_batch",
                "in_path": str(in_path),
                "max_iter": int(args.max_iter),
                "acceptable_tol": float(args.acceptable_tol),
                "acceptable_iter": int(args.acceptable_iter),
                "tol": float(args.tol),
                "runtime_metric_mode": str(args.julia_runtime_metrics),
                "result_format": JULIA_STACKED_RESULT_FORMAT,
            }
        )["result"]

    # 可先跳过：prepare_batch 预热/诊断接口。
    # 当前正式 benchmark 主线没有调用它，保留它主要是给实验和潜在预构建路径用。
    def prepare_batch_payload(self, payload, args):
        return self._request(
            {
                "action": "prepare_batch",
                "payload": payload,
                "max_iter": int(args.max_iter),
                "acceptable_tol": float(args.acceptable_tol),
                "acceptable_iter": int(args.acceptable_iter),
                "tol": float(args.tol),
            }
        )["result"]


# 当前稳定主线的关键接缝。
# 它会把 JustWorkingOnIt.py 里的 jax_ADMM_SubP2 方法临时替换成 Julia IPC 版本，
# 从而不改 planner 主循环调用点，就把 SubP2 切到 Julia。
def install_julia_subp2_patch(args, worker):
    # 被 monkey-patch 进去的替代版 SubP2。
    # 当前稳定路径就是：
    # planner._prepare_subp2_batch -> build_batch_export_runtime -> worker.solve_batch_payload -> planner._unpack_subp2_results
    def patched_jax_admm_subp2(self, Para2_dict):
        planner = self
        N = self.N
        nq = int(self.nq)
        nxl = int(self.nxl)
        nul = int(self.nul)
        nxi = int(self.nxi)
        nui = int(self.nui)
        num_dis = int(self.num_dis)
        dims = (nxl, nul, nxi, nui, nq, num_dis)

        t_prepare = time.perf_counter()
        w_init_batch, params_batch = planner._prepare_subp2_batch(Para2_dict)
        prepare_ms = (time.perf_counter() - t_prepare) * 1000.0
        meta = {
            "task_idx": int(args.task_idx),
            "horizon": int(self.N),
            "decision_dim": int(nxl + nul + nq * (nxi + nui)),
            "eq_dim": int(1 + nq + 6),
            "ineq_dim": int(
                2
                + 2 * nq
                + 2 * num_dis * (nq * (nq - 1) // 2)
                + 2 * nq
                + 2 * nq
                + 2 * nq * nui
                + 2 * nq
            ),
            "nxl": int(self.nxl),
            "nul": int(self.nul),
            "nxi": int(self.nxi),
            "nui": int(self.nui),
            "nq": int(self.nq),
            "num_dis": int(self.num_dis),
        }
        t_payload = time.perf_counter()
        batch_payload = build_batch_export_runtime(dims, w_init_batch, params_batch, meta)
        payload_ms = (time.perf_counter() - t_payload) * 1000.0

        prepare_batch_ms = 0.0
        prep_out = {"prepare_ms": 0.0, "built": 0.0, "mode": "skipped"}

        t_request = time.perf_counter()
        # 可先跳过：keep_json 兼容模式。
        # 当前正式主线默认直接走内存 payload，不会先把 JSON 落盘。
        if args.keep_json:
            in_path = ROOT / "julia_subp2" / "tmp_batch_persistent.json"
            with open(in_path, "w", encoding="utf-8") as f:
                json.dump(batch_payload, f, separators=(",", ":"))
            batch_out = worker.solve_batch(in_path, args)
        else:
            batch_out = worker.solve_batch_payload(batch_payload, args)
        request_ms = (time.perf_counter() - t_request) * 1000.0

        t_rebuild = time.perf_counter()
        mode_codes = np.ones((N + 1,), dtype=int)
        # 当前主线默认会走这个紧凑 stacked 回包格式。
        if batch_out.get("result_format") == JULIA_STACKED_RESULT_FORMAT:
            x_sol_batch = np.asarray(batch_out["x_sol_batch"], dtype=float)
            if x_sol_batch.ndim == 1:
                x_sol_batch = x_sol_batch.reshape((N + 1, meta["decision_dim"]), order="F")
            w_opt_batch = jnp.asarray(x_sol_batch)
            diag_iters = np.asarray(batch_out["iter_batch"], dtype=int)
            diag_eq = _json_metric_array(batch_out["eq_inf_batch"])
            diag_ineq = _json_metric_array(batch_out["ineq_vio_batch"])
        else:
            # 可先跳过：旧结果格式 fallback。
            # 只有 worker 没返回 stacked_runtime_result_v1 时才会走这里。
            w_opt = []
            diag_iters = []
            diag_eq = []
            diag_ineq = []
            for out in batch_out["results"]:
                x_sol = np.array(out["x_sol"], dtype=float)
                w_opt.append(x_sol)
                diag_iters.append(int(out["iter"]))
                eq_val = out.get("eq_inf", None)
                ineq_val = out.get("ineq_vio", None)
                diag_eq.append(np.nan if eq_val is None else float(eq_val))
                diag_ineq.append(np.nan if ineq_val is None else float(ineq_val))
            w_opt_batch = jnp.asarray(np.stack(w_opt, axis=0))
            diag_iters = np.asarray(diag_iters, dtype=int)
            diag_eq = np.asarray(diag_eq, dtype=float)
            diag_ineq = np.asarray(diag_ineq, dtype=float)
        result = planner._unpack_subp2_results(w_opt_batch)
        rebuild_ms = (time.perf_counter() - t_rebuild) * 1000.0
        valid_iters = diag_iters[diag_iters >= 0]
        iter_mean = float(np.mean(valid_iters)) if valid_iters.size else float("nan")
        iter_p50 = float(np.percentile(valid_iters, 50)) if valid_iters.size else float("nan")
        iter_p90 = float(np.percentile(valid_iters, 90)) if valid_iters.size else float("nan")
        iter_max = int(np.max(valid_iters)) if valid_iters.size else -1
        result["diag"] = {
            "mode": np.array(["main"] * (N + 1), dtype=object),
            "mode_code": mode_codes,
            "converged": np.ones((N + 1,), dtype=bool),
            "iterations": diag_iters,
            "iter_mean": iter_mean,
            "iter_p50": iter_p50,
            "iter_p90": iter_p90,
            "iter_max": iter_max,
            "eq_inf": diag_eq,
            "ineq_vio": diag_ineq,
            "tail_mu": np.full((N + 1,), np.nan, dtype=float),
            "tail_ineq": np.full((N + 1,), np.nan, dtype=float),
            "tail_comp": np.full((N + 1,), np.nan, dtype=float),
            "tail_dual": np.full((N + 1,), np.nan, dtype=float),
            "batch_total_ms": float(batch_out.get("total_ms", np.nan)),
            "batch_build_ms_total": float(batch_out.get("build_ms_total", np.nan)),
            "batch_solve_ms_total": float(batch_out.get("solve_ms_total", np.nan)),
            "batch_prepare_ms": float(prep_out.get("prepare_ms", np.nan)),
            "batch_prepare_built": float(prep_out.get("built", np.nan)),
            "bridge_prepare_ms": float(prepare_ms),
            "bridge_payload_ms": float(payload_ms),
            "bridge_worker_prepare_ms": float(prepare_batch_ms),
            "bridge_request_ms": float(request_ms),
            "bridge_rebuild_ms": float(rebuild_ms),
        }
        return result

    JAX_Planner.MPC_Planner.jax_ADMM_SubP2 = patched_jax_admm_subp2


# 单次 fullflow benchmark/verify。
# 它会先装上 Julia patch，再调用 verify_jax_vs_original 跑完整 planner，最后把 SubP2 与 bridge 指标汇总出来。
def run_once(args, worker):
    install_julia_subp2_patch(args, worker)
    t0 = time.perf_counter()
    summary = verify_jax_vs_original.verify_jax_planner(task_idx=args.task_idx, show_plots=args.show_plots)
    total_ms = (time.perf_counter() - t0) * 1000.0
    subp2_total_ms = 0.0
    subp2_build_ms = 0.0
    subp2_solve_ms = 0.0
    subp2_bridge_prepare_ms = 0.0
    subp2_bridge_payload_ms = 0.0
    subp2_bridge_worker_prepare_ms = 0.0
    subp2_bridge_request_ms = 0.0
    subp2_bridge_rebuild_ms = 0.0
    subp2_batch_prepare_ms = 0.0
    for item in summary.get("subp2_summary", []):
        subp2_total_ms += float(item.get("batch_total_ms", 0.0) or 0.0)
        subp2_build_ms += float(item.get("batch_build_ms_total", 0.0) or 0.0)
        subp2_solve_ms += float(item.get("batch_solve_ms_total", 0.0) or 0.0)
        subp2_batch_prepare_ms += float(item.get("batch_prepare_ms", 0.0) or 0.0)
        subp2_bridge_prepare_ms += float(item.get("bridge_prepare_ms", 0.0) or 0.0)
        subp2_bridge_payload_ms += float(item.get("bridge_payload_ms", 0.0) or 0.0)
        subp2_bridge_worker_prepare_ms += float(item.get("bridge_worker_prepare_ms", 0.0) or 0.0)
        subp2_bridge_request_ms += float(item.get("bridge_request_ms", 0.0) or 0.0)
        subp2_bridge_rebuild_ms += float(item.get("bridge_rebuild_ms", 0.0) or 0.0)
    return {
        "summary": summary,
        "fullflow_ms": total_ms,
        "subp2_total_ms": subp2_total_ms,
        "subp2_build_ms": subp2_build_ms,
        "subp2_solve_ms": subp2_solve_ms,
        "subp2_batch_prepare_ms": subp2_batch_prepare_ms,
        "subp2_bridge_prepare_ms": subp2_bridge_prepare_ms,
        "subp2_bridge_payload_ms": subp2_bridge_payload_ms,
        "subp2_bridge_worker_prepare_ms": subp2_bridge_worker_prepare_ms,
        "subp2_bridge_request_ms": subp2_bridge_request_ms,
        "subp2_bridge_rebuild_ms": subp2_bridge_rebuild_ms,
    }


# CLI 主入口。
# 当前从命令行跑 persistent Julia fullflow 时，实际就是从这里启动。
def main():
    args = build_parser().parse_args()
    worker = JuliaBatchWorker(
        julia_threads=args.julia_threads,
        blas_threads=args.julia_blas_threads,
        linear_solver=args.julia_linear_solver,
        kkt_system=args.julia_kkt_system,
        callback=args.julia_callback,
        thread_schedule=args.julia_thread_schedule,
    )
    try:
        print(
            "[PERSISTENT-JULIA-WORKER] "
            f"nthreads={worker.nthreads} "
            f"blas_threads={worker.blas_threads} "
            f"linear_solver={worker.linear_solver} "
            f"kkt_system={worker.kkt_system} "
            f"callback={worker.callback} "
            f"thread_schedule={worker.thread_schedule}"
        )
        results = []
        for rep in range(args.repeats):
            out = run_once(args, worker)
            results.append(out)
            print(f"[PERSISTENT-JULIA-FULLFLOW] repeat={rep+1} total_ms={out['fullflow_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-PREPARE] repeat={rep+1} total_ms={out['subp2_batch_prepare_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-BUILD] repeat={rep+1} total_ms={out['subp2_build_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-SOLVE] repeat={rep+1} total_ms={out['subp2_solve_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-ONLY] repeat={rep+1} total_ms={out['subp2_total_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-PREPARE] repeat={rep+1} total_ms={out['subp2_bridge_prepare_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-PAYLOAD] repeat={rep+1} total_ms={out['subp2_bridge_payload_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-WORKER-PREPARE] repeat={rep+1} total_ms={out['subp2_bridge_worker_prepare_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-REQUEST] repeat={rep+1} total_ms={out['subp2_bridge_request_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-REBUILD] repeat={rep+1} total_ms={out['subp2_bridge_rebuild_ms']:.3f}")
            print(out["summary"])
    finally:
        worker.close()


# -----------------------------------------------------------------------------
# 以下 helper 不是当前稳定 fullflow 主线必经路径，统一收在文件尾部。
# 保留它们是为了离线单步核对/兼容旧实验，不影响主线执行逻辑。
# -----------------------------------------------------------------------------

# 可先跳过：单步完整 snapshot 导出 helper。
# 它会把 init/original 两套参考值都带上，适合离线单步核对；
# 当前 persistent Julia fullflow runner 并不直接调用它。
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


# 可先跳过：单步 runtime 导出 helper。
# 相比上面的完整 snapshot 更轻，但当前这个 batch runner 也不直接走它。
def build_step_export_runtime(dims, x_init, params_t, meta):
    export = {
        "meta": meta,
        "step": {
            "dims": list(map(int, dims)),
            "x_init": np.array(x_init, dtype=float),
            "params_t": {k: np.array(v) for k, v in params_t.items()},
        },
    }
    return _to_jsonable(export)


# 作为脚本直接执行时，从 main() 进入。
if __name__ == "__main__":
    main()
