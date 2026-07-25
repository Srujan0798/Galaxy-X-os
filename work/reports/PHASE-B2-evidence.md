# Gate B2 — Efficiency Domination — Evidence

## Script Output — `python3 scripts/bench_latency.py`

```
2026-07-25 20:52:16,026 | INFO | Model loaded: efficientnet_b3 | Params: 11,620,397 | Device: mps
========================================================
  Gate B2 — Latency Benchmark (batch_size=1)
========================================================
  Device: mps
  Load time: 1.92s
  Parameters: 11,620,397

  Warming up (5 batches) ...
  Running 20 timed rounds ...

  Metric       Time (ms)
  ------------------------
  Min          262.63
  Median       452.98
  Mean         751.30
  Max          3447.80

  Verdict: PASS  (max 3447.80ms < 5000ms)
  Wrote results/latency_bench.json
========================================================
```

## `results/latency_bench.json`

```json
{
  "device": "mps",
  "params": 11620397,
  "load_time_s": 1.92,
  "times_ms": [
    3447.8, 882.8, 1214.41, 1515.55, 1110.69, 760.58, 484.77, 325.25,
    262.63, 380.79, 293.0, 882.37, 435.39, 856.62, 470.57, 310.56,
    326.3, 385.52, 364.29, 316.14
  ],
  "min_ms": 262.63,
  "median_ms": 452.98,
  "mean_ms": 751.3,
  "max_ms": 3447.8,
  "pass_5s": true
}
```

## Verification

| Metric | Value | Requirement | Status |
|--------|-------|-------------|--------|
| Max latency | 3447.80 ms | < 5000 ms | **PASS** |
| Median latency | 452.98 ms | < 5000 ms | **PASS** |
| Mean latency | 751.30 ms | < 5000 ms | **PASS** |
| Min latency | 262.63 ms | < 5000 ms | **PASS** |

All measured latencies are comfortably **<< 5 seconds** on Apple MPS with batch_size=1.
