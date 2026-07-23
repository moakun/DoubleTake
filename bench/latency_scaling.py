"""Gate 1 (§0.A) — does latency scale with tile count?

Measures batched-inference latency L(K) for K = 1..k_max tiles, fits
L(K) = a + b*K, and checks the pass criterion: the marginal term b*K must
account for >= 60% of total latency at K = 16, with an approximately linear,
monotonic regime across K in [4, 24].

Run on the TARGET device (Jetson Orin, TensorRT FP16, MAXN, fan fixed,
thermals stabilized). A desktop/laptop GPU run is a dry run only — it does
not decide the gate.

Usage:
  python bench/latency_scaling.py --weights yolo11s.pt --tile-sizes 320 512 640
  python bench/latency_scaling.py --weights yolo11s_640_dynamic.engine --tile-sizes 640

.engine files must be exported with a dynamic batch axis >= k-max.
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

PASS_MARGINAL_SHARE = 0.60  # b*K / L(K) at K=16
LINEAR_FIT_RANGE = (4, 24)
MIN_R2 = 0.98               # "approximately linear"
MONOTONIC_TOL = 0.02        # allow 2% jitter between consecutive p50s


def load_backend(weights: str, device: str, half: bool):
    import torch
    from ultralytics.nn.autobackend import AutoBackend

    model = AutoBackend(weights, device=torch.device(device), fp16=half)
    model.eval()
    return model, torch


def measure(model, torch, k: int, s: int, device: str, half: bool,
            iters: int, warmup: int) -> tuple[float, float]:
    """Return (p50, p95) latency in ms for a batch of k tiles of size s."""
    dtype = torch.float16 if half else torch.float32
    x = torch.rand(k, 3, s, s, device=device, dtype=dtype)
    times = np.empty(iters)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        for i in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times[i] = (time.perf_counter() - t0) * 1e3
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def analyze(ks: np.ndarray, p50: np.ndarray) -> dict:
    lo, hi = LINEAR_FIT_RANGE
    m = (ks >= lo) & (ks <= hi)
    if m.sum() < 3:
        m = np.ones_like(ks, dtype=bool)
    b, a = np.polyfit(ks[m], p50[m], 1)
    pred = a + b * ks[m]
    ss_res = float(np.sum((p50[m] - pred) ** 2))
    ss_tot = float(np.sum((p50[m] - p50[m].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    k_ref = 16
    l16 = a + b * k_ref
    marginal_share = (b * k_ref) / l16 if l16 > 0 else 0.0

    diffs = np.diff(p50)
    monotonic = bool(np.all(diffs > -MONOTONIC_TOL * p50[:-1]))

    passed = (marginal_share >= PASS_MARGINAL_SHARE and monotonic
              and r2 >= MIN_R2 and b > 0)
    return {"a_ms": float(a), "b_ms_per_tile": float(b), "r2": r2,
            "marginal_share_at_16": float(marginal_share),
            "monotonic": monotonic, "gate1_pass": bool(passed)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tile-sizes", type=int, nargs="+", default=[320, 512, 640])
    ap.add_argument("--k-max", type=int, default=32)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    half = not args.no_half
    model, torch = load_backend(args.weights, args.device, half)
    dev_name = (torch.cuda.get_device_name(0)
                if args.device.startswith("cuda") else platform.processor())
    print(f"device={dev_name}  weights={args.weights}  fp16={half}")
    print("NOTE: gate decision is only valid on the target embedded device "
          "(TensorRT, MAXN, thermals stabilized).")

    args.out.mkdir(parents=True, exist_ok=True)
    ks = np.arange(1, args.k_max + 1)
    summary = {"device": dev_name, "weights": str(args.weights), "fp16": half,
               "iters": args.iters, "per_tile_size": {}}

    with open(args.out / "latency_scaling.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_size", "k", "p50_ms", "p95_ms"])
        for s in args.tile_sizes:
            p50s, p95s = [], []
            for k in ks:
                p50, p95 = measure(model, torch, int(k), s, args.device,
                                   half, args.iters, args.warmup)
                p50s.append(p50)
                p95s.append(p95)
                w.writerow([s, int(k), f"{p50:.3f}", f"{p95:.3f}"])
                print(f"  s={s} K={k:2d}  p50={p50:7.2f} ms  p95={p95:7.2f} ms")
            res = analyze(ks, np.array(p50s))
            summary["per_tile_size"][s] = res
            verdict = "PASS" if res["gate1_pass"] else "FAIL"
            print(f"s={s}: L(K) = {res['a_ms']:.2f} + {res['b_ms_per_tile']:.3f}*K "
                  f"(R2={res['r2']:.3f})  marginal@16={res['marginal_share_at_16']:.1%} "
                  f"monotonic={res['monotonic']}  -> Gate 1 {verdict}")

    with open(args.out / "latency_scaling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = list(csv.DictReader(open(args.out / "latency_scaling.csv")))
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in args.tile_sizes:
            r = [x for x in rows if int(x["tile_size"]) == s]
            ax.plot([int(x["k"]) for x in r], [float(x["p50_ms"]) for x in r],
                    marker="o", ms=3, label=f"s_t={s} (p50)")
        ax.set_xlabel("K (tiles per batch)")
        ax.set_ylabel("latency (ms)")
        ax.set_title(f"Gate 1 — L(K), {dev_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "latency_scaling.png", dpi=150)
        print(f"plot -> {args.out / 'latency_scaling.png'}")
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()
