import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python_archive.run_julia_subp2_fullflow_persistent as base


GPU_WORKER = ROOT / "julia_subp2" / "archive" / "worker_batch_madnlp_jump_native_gpuexp.jl"
base.JULIA_WORKER = GPU_WORKER


def build_parser():
    parser = base.build_parser()
    parser.set_defaults(
        julia_linear_solver="lapackcuda",
        julia_kkt_system="dense_condensed",
        julia_callback="dense",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    worker = base.JuliaBatchWorker(
        julia_threads=args.julia_threads,
        blas_threads=args.julia_blas_threads,
        linear_solver=args.julia_linear_solver,
        kkt_system=args.julia_kkt_system,
        callback=args.julia_callback,
        thread_schedule=args.julia_thread_schedule,
    )
    try:
        print(
            "[PERSISTENT-JULIA-WORKER-GPUEXP] "
            f"nthreads={worker.nthreads} "
            f"blas_threads={worker.blas_threads} "
            f"linear_solver={worker.linear_solver} "
            f"kkt_system={worker.kkt_system} "
            f"callback={worker.callback} "
            f"thread_schedule={worker.thread_schedule}"
        )
        results = []
        for rep in range(args.repeats):
            out = base.run_once(args, worker)
            results.append(out)
            print(f"[PERSISTENT-JULIA-FULLFLOW-GPUEXP] repeat={rep+1} total_ms={out['fullflow_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-PREPARE-GPUEXP] repeat={rep+1} total_ms={out['subp2_batch_prepare_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-BUILD-GPUEXP] repeat={rep+1} total_ms={out['subp2_build_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-SOLVE-GPUEXP] repeat={rep+1} total_ms={out['subp2_solve_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-SUBP2-ONLY-GPUEXP] repeat={rep+1} total_ms={out['subp2_total_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-PREPARE-GPUEXP] repeat={rep+1} total_ms={out['subp2_bridge_prepare_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-PAYLOAD-GPUEXP] repeat={rep+1} total_ms={out['subp2_bridge_payload_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-WORKER-PREPARE-GPUEXP] repeat={rep+1} total_ms={out['subp2_bridge_worker_prepare_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-REQUEST-GPUEXP] repeat={rep+1} total_ms={out['subp2_bridge_request_ms']:.3f}")
            print(f"[PERSISTENT-JULIA-BRIDGE-REBUILD-GPUEXP] repeat={rep+1} total_ms={out['subp2_bridge_rebuild_ms']:.3f}")
            print(out["summary"])
    finally:
        worker.close()


if __name__ == "__main__":
    main()
