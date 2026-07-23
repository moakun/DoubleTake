# Phase 0 benchmarks — feasibility gates

Two scripts, one gate each. Both must pass before any scheduler code is written
(plan §Phase 0). Gate 3 (prior-art clearance) lives in `../related_work.md` and
`../memos/`.

## Gate 1 — latency scaling (`latency_scaling.py`)

Decides: does skipping tiles actually save time, or is inference launch-bound?

```
python bench/latency_scaling.py --weights yolo11s.pt --tile-sizes 320 512 640
```

- **The gate decision is only valid on the target device** (Jetson Orin,
  TensorRT FP16 engine exported with a dynamic batch axis, MAXN, fan fixed,
  10 min warm-up). A desktop run is a plumbing dry run.
- Pass: marginal term `b·K` ≥ 60% of `L(K)` at K=16, monotone ~linear on K∈[4,24].
- Outputs: `results/latency_scaling.csv`, `_summary.json`, `.png`.

## Gate 2 — oracle headroom (`oracle_headroom.py`)

Decides: is the gain concentrated in few tiles, and does an oracle beat
objectness selection by enough to justify a learned predictor?

```
python bench/oracle_headroom.py --self-test          # no GPU/data needed
python bench/oracle_headroom.py --data <VisDrone2019-DET-val> --weights yolo11s.pt
```

- Accuracy-only → any CUDA GPU is fine (the local RTX 4060 works).
- Expects VisDrone layout: `<root>/images/*.jpg`, `<root>/annotations/*.txt`.
- Exhaustive T2 detections are cached per image in `bench/cache/…` (keyed by
  weights/s_lo/s_t/rho) — the expensive pass runs once. `--limit N` for smoke runs.
- Pass @ f=0.25: oracle ≥ 90% of exhaustive AP_small **and** oracle − objectness
  ≥ 3.0 AP_small points.
- Outputs: `results/oracle_headroom.csv`, `.png`.
- AP_small here is a self-consistent COCO-style approximation applied
  identically to all four curves; paper-grade numbers go through pycocotools.

## Requirements

`pip install -r requirements.txt` (torch per your CUDA version first, see
pytorch.org). The self-test needs numpy only.
