import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
JULIA_WORKER = ROOT / "julia_subp2" / "worker_batch_madnlp_jump_native.jl"
TEST_SNAPSHOT = ROOT / "julia_subp2" / "archive" / "test_snapshot.json"


class JuliaBatchWorker:
    def __init__(self, julia_threads: int = 0):
        env = os.environ.copy()
        env.setdefault("JULIA_DEPOT_PATH", "/tmp/julia-depot:/home/mpc/.julia")
        if julia_threads and julia_threads > 0:
            env["JULIA_NUM_THREADS"] = str(julia_threads)
        else:
            env.setdefault("JULIA_NUM_THREADS", str(os.cpu_count() or 1))
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
        ping = self.request({"action": "ping"})
        self.nthreads = int(ping.get("result", {}).get("nthreads", -1))

    def request(self, payload):
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
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

    def close(self):
        if self.proc.poll() is not None:
            return
        try:
            self.request({"action": "shutdown"})
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


def build_representative_batch(out_path: Path, nsteps: int) -> int:
    with open(TEST_SNAPSHOT, "r", encoding="utf-8") as f:
        base = json.load(f)
    step = base["step"]
    meta = base.get("meta", {})
    steps = []
    for k in range(nsteps):
        step_copy = json.loads(json.dumps(step))
        step_copy["meta"] = dict(meta)
        step_copy["meta"]["target_step"] = k
        steps.append(step_copy)
    payload = {"steps": steps}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return out_path.stat().st_size


def timed(fn, repeats):
    vals = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return vals, last


def main():
    p = argparse.ArgumentParser(description="Benchmark Julia worker bridge overhead.")
    p.add_argument("--nsteps", type=int, default=101)
    p.add_argument("--ping-repeats", type=int, default=20)
    p.add_argument("--load-repeats", type=int, default=10)
    p.add_argument("--keep-json", action="store_true")
    p.add_argument("--julia-threads", type=int, default=0)
    args = p.parse_args()

    if args.keep_json:
        batch_path = ROOT / "julia_subp2" / "tmp_bridge_batch.json"
    else:
        fd, name = tempfile.mkstemp(prefix="julia_bridge_batch_", suffix=".json")
        os.close(fd)
        batch_path = Path(name)

    size_bytes = build_representative_batch(batch_path, args.nsteps)

    write_times, _ = timed(lambda: build_representative_batch(batch_path, args.nsteps), args.load_repeats)

    worker = JuliaBatchWorker(julia_threads=args.julia_threads)
    try:
        ping_times, _ = timed(lambda: worker.request({"action": "ping"}), args.ping_repeats)
        load_times, load_resp = timed(
            lambda: worker.request({"action": "load_batch_only", "in_path": str(batch_path)}),
            args.load_repeats,
        )
    finally:
        worker.close()
        if not args.keep_json:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception:
                pass

    parse_ms = float(load_resp["result"]["load_parse_ms"])
    print(
        json.dumps(
            {
                "nsteps": args.nsteps,
                "batch_size_bytes": size_bytes,
                "write_ms_mean": statistics.mean(write_times),
                "write_ms_min": min(write_times),
                "ping_ms_mean": statistics.mean(ping_times),
                "ping_ms_min": min(ping_times),
                "load_roundtrip_ms_mean": statistics.mean(load_times),
                "load_roundtrip_ms_min": min(load_times),
                "julia_load_parse_ms_last": parse_ms,
                "load_roundtrip_minus_parse_ms_mean": statistics.mean(load_times) - parse_ms,
                "worker_nthreads": worker.nthreads,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
