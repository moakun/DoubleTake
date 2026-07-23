"""Convert VisDrone-DET annotations to YOLO format (labels/ next to images/).

VisDrone line: x,y,w,h,score,category,truncation,occlusion
Kept: categories 1..10 with score==1  ->  YOLO class 0..9, normalized cxcywh.
Skipped: category 0 (ignored regions), 11 (others), score==0.

Usage: python scripts/visdrone2yolo.py VisDrone2019-DET/VisDrone2019-DET-train ...
"""

import sys
from pathlib import Path

from PIL import Image


def convert(root: Path) -> None:
    img_dir, ann_dir = root / "images", root / "annotations"
    lbl_dir = root / "labels"
    lbl_dir.mkdir(exist_ok=True)
    n_boxes = n_imgs = 0
    for ann in ann_dir.glob("*.txt"):
        img_path = img_dir / (ann.stem + ".jpg")
        if not img_path.exists():
            print(f"  WARNING: no image for {ann.name}")
            continue
        w_img, h_img = Image.open(img_path).size  # header only, fast
        lines = []
        for raw in ann.read_text().strip().splitlines():
            p = [int(v) for v in raw.strip().strip(",").split(",")[:8]]
            x, y, w, h, score, cat = p[:6]
            if score == 0 or cat < 1 or cat > 10 or w <= 0 or h <= 0:
                continue
            cx = min(max((x + w / 2) / w_img, 0.0), 1.0)
            cy = min(max((y + h / 2) / h_img, 0.0), 1.0)
            lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} "
                         f"{min(w / w_img, 1.0):.6f} {min(h / h_img, 1.0):.6f}")
        (lbl_dir / ann.name).write_text("\n".join(lines))
        n_boxes += len(lines)
        n_imgs += 1
    print(f"{root.name}: {n_imgs} label files, {n_boxes} boxes")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        convert(Path(arg))
