"""Gate 2 (§0.B) — is oracle tile-selection headroom large?

For every VisDrone-val image: run T1 (full frame at s_lo) and T2 on EVERY
tile (native res, grid overlap rho). Cache all detections. Compute per-tile
oracle gain g*(t) = #GT matched by T2-on-t but missed by T1. Build four
curves of AP_small vs tile fraction f = K/|G|:

  oracle | objectness (summed T1 conf in tile) | uniform (raster) | random (5 seeds)

Pass criteria (both required, at f = 0.25):
  1. sparsity:     oracle >= 90% of exhaustive-tiling AP_small
  2. learnability: oracle - objectness >= 3.0 AP_small points

Usage:
  python bench/oracle_headroom.py --self-test
  python bench/oracle_headroom.py --data D:/VisDrone2019-DET-val --weights yolo11s.pt

Notes: AP_small here is a self-consistent COCO-style approximation (area <
32^2, IoU 0.5, 101-pt interpolation) used identically across all curves;
paper-grade numbers go through pycocotools later.
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np

IOU_MATCH = 0.5
NMS_IOU = 0.5
SMALL_AREA = 32 * 32
FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.0]
RANDOM_SEEDS = 5
PASS_SPARSITY = 0.90
PASS_HEADROOM_PTS = 3.0
EVAL_FRACTION = 0.25

# ---------------------------------------------------------------- geometry


def tile_grid(w: int, h: int, s_t: int, rho: float) -> np.ndarray:
    """Tile origins (N,2) so tiles of size s_t lie fully inside the image."""
    stride = max(1, int(round(s_t * (1.0 - rho))))

    def axis(size):
        if size <= s_t:
            return [0]
        xs = list(range(0, size - s_t + 1, stride))
        if xs[-1] != size - s_t:
            xs.append(size - s_t)
        return xs

    return np.array([(x, y) for y in axis(h) for x in axis(w)], dtype=np.int32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(dets: np.ndarray) -> np.ndarray:
    """Per-class greedy NMS on (N,6) [x1,y1,x2,y2,conf,cls]."""
    if len(dets) == 0:
        return dets
    keep = []
    for c in np.unique(dets[:, 5]):
        d = dets[dets[:, 5] == c]
        d = d[np.argsort(-d[:, 4])]
        while len(d):
            keep.append(d[0])
            d = d[1:][iou_matrix(d[:1, :4], d[1:, :4])[0] < NMS_IOU]
    return np.array(keep)


# ---------------------------------------------------------------- matching


def match_gt(dets: np.ndarray, gt: np.ndarray) -> set:
    """Greedy per-class matching; returns set of matched GT row indices.

    dets: (N,6) [x1,y1,x2,y2,conf,cls] sorted internally by conf.
    gt:   (M,5) [x1,y1,x2,y2,cls]
    """
    matched = set()
    if len(dets) == 0 or len(gt) == 0:
        return matched
    for i in np.argsort(-dets[:, 4]):
        d = dets[i]
        cand = [j for j in range(len(gt))
                if j not in matched and gt[j, 4] == d[5]]
        if not cand:
            continue
        ious = iou_matrix(d[None, :4], gt[cand][:, :4])[0]
        k = int(np.argmax(ious))
        if ious[k] >= IOU_MATCH:
            matched.add(cand[k])
    return matched


def ap_small(per_image: list) -> float:
    """COCO-style AP_small (IoU 0.5, 101-pt), mean over classes.

    per_image: list of (dets (N,6), gt (M,5), gt_ignore (M,) bool).
    Small GT (area < SMALL_AREA, not ignored) are targets; detections
    matching ignored or non-small GT are discarded (neither TP nor FP).
    """
    classes = sorted({int(c) for _, gt, ign in per_image
                      for c in gt[~ign][:, 4]} if per_image else set())
    aps = []
    for c in classes:
        records, n_small = [], 0
        for img_i, (dets, gt, ign) in enumerate(per_image):
            g = gt[gt[:, 4] == c] if len(gt) else gt
            gi = ign[gt[:, 4] == c] if len(gt) else ign
            small = (~gi) & (np.prod(g[:, 2:4] - g[:, 0:2], axis=1) < SMALL_AREA) \
                if len(g) else np.zeros(0, bool)
            n_small += int(small.sum())
            d = dets[dets[:, 5] == c] if len(dets) else dets
            taken = np.zeros(len(g), bool)
            for i in np.argsort(-d[:, 4]) if len(d) else []:
                ious = iou_matrix(d[i][None, :4], g[:, :4])[0] if len(g) else []
                free = np.where(~taken)[0] if len(g) else []
                if len(free) and ious[free].max() >= IOU_MATCH:
                    j = free[int(np.argmax(ious[free]))]
                    taken[j] = True
                    if small[j]:
                        records.append((d[i, 4], 1))  # TP on small GT
                    # matched a non-small/ignored GT -> discard silently
                else:
                    records.append((d[i, 4], 0))      # FP
        if n_small == 0:
            continue
        if not records:
            aps.append(0.0)
            continue
        records.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in records])
        fp = np.cumsum([1 - r[1] for r in records])
        rec = tp / n_small
        prec = tp / np.maximum(tp + fp, 1e-9)
        ap = float(np.mean([prec[rec >= t].max() if np.any(rec >= t) else 0.0
                            for t in np.linspace(0, 1, 101)]))
        aps.append(ap)
    return 100.0 * float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------- selection


def select_tiles(policy: str, k: int, gains: np.ndarray, obj: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    n = len(gains)
    if k >= n:
        return np.arange(n)
    if policy == "oracle":
        order = np.argsort(-gains)
    elif policy == "objectness":
        order = np.argsort(-obj)
    elif policy == "uniform":
        order = np.arange(n)  # raster order
    elif policy == "random":
        order = rng.permutation(n)
    else:
        raise ValueError(policy)
    return order[:k]


# ---------------------------------------------------------------- pipeline


def load_visdrone_gt(ann_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt (M,5) xyxy+cls, ignore (M,) bool). Classes 1..10."""
    boxes, ignore = [], []
    for line in ann_file.read_text().strip().splitlines():
        p = [int(v) for v in line.strip().strip(",").split(",")[:8]]
        x, y, w, h, score, cat = p[0], p[1], p[2], p[3], p[4], p[5]
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h, cat])
        ignore.append(cat in (0, 11) or score == 0)
    if not boxes:
        return np.zeros((0, 5)), np.zeros(0, bool)
    return np.array(boxes, float), np.array(ignore, bool)


def run_dataset(args) -> None:
    from ultralytics import YOLO
    import cv2

    model = YOLO(args.weights)
    img_dir = Path(args.data) / "images"
    ann_dir = Path(args.data) / "annotations"
    cache_dir = Path(args.cache) / (
        f"{Path(args.weights).stem}_lo{args.s_lo}_t{args.s_t}_r{args.rho}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        images = images[:args.limit]
    print(f"{len(images)} images | cache: {cache_dir}")

    def detect(im, imgsz):
        r = model.predict(im, imgsz=imgsz, conf=args.conf, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 6))
        return np.column_stack([b.xyxy.cpu().numpy(),
                                b.conf.cpu().numpy(),
                                b.cls.cpu().numpy() + 1])  # ->VisDrone 1-based

    per_image = []
    for n, path in enumerate(images):
        cpath = cache_dir / (path.stem + ".json")
        if cpath.exists():
            c = json.loads(cpath.read_text())
        else:
            img = cv2.imread(str(path))
            h, w = img.shape[:2]
            grid = tile_grid(w, h, args.s_t, args.rho)
            t1 = detect(img, args.s_lo)
            tiles = []
            for ox, oy in grid:
                crop = img[oy:oy + args.s_t, ox:ox + args.s_t]
                d = detect(crop, args.s_t)
                if len(d):
                    d[:, [0, 2]] += ox
                    d[:, [1, 3]] += oy
                tiles.append(d.tolist())
            c = {"w": w, "h": h, "grid": grid.tolist(),
                 "t1": t1.tolist(), "tiles": tiles}
            cpath.write_text(json.dumps(c))
        per_image.append((path.stem, c))
        if (n + 1) % 25 == 0:
            print(f"  cached {n + 1}/{len(images)}")

    # ---- gains, objectness, curves
    curves = {p: {f: [] for f in FRACTIONS}
              for p in ("oracle", "objectness", "uniform", "random")}
    rngs = [np.random.default_rng(s) for s in range(RANDOM_SEEDS)]

    gts = {}
    for stem, c in per_image:
        gts[stem] = load_visdrone_gt(Path(args.data) / "annotations" / f"{stem}.txt")

    for f in FRACTIONS:
        for policy in curves:
            seeds = rngs if policy == "random" else [None]
            per_seed = []
            for rng in seeds:
                batch = []
                for stem, c in per_image:
                    gt, ign = gts[stem]
                    t1 = np.array(c["t1"]) if c["t1"] else np.zeros((0, 6))
                    grid = np.array(c["grid"])
                    tiles = [np.array(t) if t else np.zeros((0, 6))
                             for t in c["tiles"]]
                    t1_matched = match_gt(t1, gt[~ign]) if len(gt) else set()
                    gt_eval = gt[~ign]
                    gains = np.array([
                        len(match_gt(td, gt_eval) - t1_matched)
                        for td in tiles], float)
                    obj = np.zeros(len(grid))
                    if len(t1):
                        cx = (t1[:, 0] + t1[:, 2]) / 2
                        cy = (t1[:, 1] + t1[:, 3]) / 2
                        for ti, (ox, oy) in enumerate(grid):
                            m = ((cx >= ox) & (cx < ox + args.s_t)
                                 & (cy >= oy) & (cy < oy + args.s_t))
                            obj[ti] = t1[m, 4].sum()
                    k = int(round(f * len(grid)))
                    sel = select_tiles(policy, k, gains, obj, rng)
                    merged = np.vstack([t1] + [tiles[i] for i in sel]) \
                        if len(sel) else t1
                    batch.append((nms(merged), gt, ign))
                per_seed.append(ap_small(batch))
            curves[policy][f] = float(np.mean(per_seed))
        print(f"f={f:.2f}  " + "  ".join(
            f"{p}={curves[p][f]:.2f}" for p in curves))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "oracle_headroom.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fraction"] + list(curves))
        for f in FRACTIONS:
            w.writerow([f] + [f"{curves[p][f]:.3f}" for p in curves])

    exhaustive = curves["oracle"][1.0]
    o25, b25 = curves["oracle"][EVAL_FRACTION], curves["objectness"][EVAL_FRACTION]
    sparsity_ok = o25 >= PASS_SPARSITY * exhaustive
    headroom_ok = (o25 - b25) >= PASS_HEADROOM_PTS
    print(f"\nGate 2 @ f={EVAL_FRACTION}: oracle={o25:.2f} objectness={b25:.2f} "
          f"exhaustive={exhaustive:.2f}")
    print(f"  sparsity    {'PASS' if sparsity_ok else 'FAIL'} "
          f"(oracle/exhaustive = {o25 / max(exhaustive, 1e-9):.1%}, need >=90%)")
    print(f"  learnability {'PASS' if headroom_ok else 'FAIL'} "
          f"(headroom = {o25 - b25:.2f} pts, need >={PASS_HEADROOM_PTS})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for p in curves:
            ax.plot(FRACTIONS, [curves[p][f] for f in FRACTIONS],
                    marker="o", ms=3, label=p)
        ax.set_xlabel("tile fraction f = K/|G|")
        ax.set_ylabel("AP_small")
        ax.set_title("Gate 2 — oracle headroom")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "oracle_headroom.png", dpi=150)
    except ImportError:
        pass


# ---------------------------------------------------------------- self-test


def self_test() -> None:
    """Exercise geometry/matching/AP on synthetic boxes (plan §3.4 rule:
    coordinate transforms must be exact — tested before any GPU hour)."""
    # grid covers image exactly, tiles inside bounds
    g = tile_grid(1000, 600, 512, 0.2)
    assert (g >= 0).all() and (g[:, 0] <= 1000 - 512).all() \
        and (g[:, 1] <= 600 - 512).all()
    assert any((x + 512 == 1000) for x, _ in g) and any(
        (y + 512 == 600) for _, y in g), "grid must reach far edges"
    g1 = tile_grid(300, 200, 512, 0.2)
    assert len(g1) == 1 and tuple(g1[0]) == (0, 0)

    # tile->global coordinate round-trip is exact
    ox, oy = 488, 88
    local = np.array([[10.0, 20.0, 60.0, 90.0, 0.9, 1.0]])
    shifted = local.copy()
    shifted[:, [0, 2]] += ox
    shifted[:, [1, 3]] += oy
    assert np.array_equal(shifted[0, :4], np.array([498.0, 108.0, 548.0, 178.0]))

    # matching: exact overlap matches, distant box does not
    gt = np.array([[100, 100, 130, 130, 1], [500, 500, 530, 530, 2]], float)
    dets = np.array([[100, 100, 130, 130, 0.9, 1],
                     [50, 50, 60, 60, 0.8, 2]], float)
    assert match_gt(dets, gt) == {0}

    # gain: tile recovers a GT that T1 missed
    t1_matched = match_gt(dets[:1], gt)
    tile_dets = np.array([[500, 500, 530, 530, 0.7, 2]], float)
    gain = len(match_gt(tile_dets, gt) - t1_matched)
    assert gain == 1, gain

    # AP: perfect small-object detections -> 100
    gt_s = np.array([[10, 10, 30, 30, 1], [50, 50, 68, 68, 1]], float)
    ign = np.zeros(2, bool)
    dets_s = np.column_stack([gt_s[:, :4], [0.9, 0.8], gt_s[:, 4]])
    assert abs(ap_small([(dets_s, gt_s, ign)]) - 100.0) < 1e-6
    # one FP halves precision at the tail but AP over recall stays high
    dets_fp = np.vstack([dets_s, [[200, 200, 220, 220, 0.95, 1]]])
    assert 0 < ap_small([(dets_fp, gt_s, ign)]) < 100.0
    # detection on an IGNORED gt is neither TP nor FP
    gt_i = np.array([[10, 10, 30, 30, 1], [50, 50, 68, 68, 1]], float)
    ign_i = np.array([False, True])
    dets_i = np.column_stack([gt_i[:, :4], [0.9, 0.8], gt_i[:, 4]])
    assert abs(ap_small([(dets_i, gt_i, ign_i)]) - 100.0) < 1e-6

    # NMS keeps the higher-confidence duplicate
    d = np.array([[0, 0, 10, 10, 0.9, 1], [1, 1, 11, 11, 0.5, 1],
                  [100, 100, 110, 110, 0.8, 1]], float)
    kept = nms(d)
    assert len(kept) == 2 and 0.9 in kept[:, 4] and 0.8 in kept[:, 4]

    print("self-test: all assertions passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", help="VisDrone2019-DET-val root (images/ + annotations/)")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--s-lo", type=int, default=640)
    ap.add_argument("--s-t", type=int, default=512)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images")
    ap.add_argument("--cache", default="bench/cache")
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.data:
        ap.error("--data is required (or use --self-test)")
    run_dataset(args)


if __name__ == "__main__":
    main()
