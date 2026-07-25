#!/usr/bin/env python3
"""Gate B2 — Latency benchmark: single-batch inference timing."""

import json
import sys
import time
import statistics
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from inference import ModelManager


def main():
    ckpt = Path("checkpoints/best_model.pth")
    if not ckpt.exists():
        print("SKIP: checkpoints/best_model.pth not found — cannot run latency bench")
        return 0

    print("=" * 56)
    print("  Gate B2 — Latency Benchmark (batch_size=1)")
    print("=" * 56)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    t0 = time.time()
    mgr = ModelManager(checkpoint_path=str(ckpt), device=str(device.type))
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")

    params = sum(p.numel() for p in mgr.model.parameters())
    print(f"  Parameters: {params:,}")

    dummy = torch.randn(1, 3, mgr.image_size, mgr.image_size, device=device)

    print(f"\n  Warming up (5 batches) ...")
    for i in range(5):
        with torch.no_grad():
            _ = mgr.model(dummy)

    print(f"  Running 20 timed rounds ...")
    times = []
    for i in range(20):
        torch.mps.synchronize() if device.type == "mps" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            if device.type in ("cuda", "mps"):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    _ = mgr.model(dummy)
            else:
                _ = mgr.model(dummy)
        torch.mps.synchronize() if device.type == "mps" else None
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    tmin = min(times)
    tmed = statistics.median(times)
    tmean = statistics.mean(times)
    tmax = max(times)

    print(f"\n  {'Metric':<12s} {'Time (ms)':<12s}")
    print(f"  {'-'*24}")
    print(f"  {'Min':<12s} {tmin:<12.2f}")
    print(f"  {'Median':<12s} {tmed:<12.2f}")
    print(f"  {'Mean':<12s} {tmean:<12.2f}")
    print(f"  {'Max':<12s} {tmax:<12.2f}")
    print()

    ok = tmax < 5000
    verdict = "PASS" if ok else "FAIL"
    print(f"  Verdict: {verdict}  (max {tmax:.2f}ms {'<' if ok else '>='} 5000ms)")

    bench = {
        "device": str(device.type),
        "params": params,
        "load_time_s": round(load_time, 2),
        "times_ms": [round(t, 2) for t in times],
        "min_ms": round(tmin, 2),
        "median_ms": round(tmed, 2),
        "mean_ms": round(tmean, 2),
        "max_ms": round(tmax, 2),
        "pass_5s": ok,
    }
    out_path = Path("results/latency_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"  Wrote {out_path}")

    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
