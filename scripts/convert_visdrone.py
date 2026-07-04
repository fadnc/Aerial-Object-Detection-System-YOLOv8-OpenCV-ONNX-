"""
Convert VisDrone2019-DET annotations to YOLO format and copy the results
straight into this project's data/images/<split> and data/labels/<split>.

Expected input layout (matches the official VisDrone2019-DET release):
    VisDrone2019-DET-train/
        images/*.jpg
        annotations/*.txt
    VisDrone2019-DET-val/
        images/*.jpg
        annotations/*.txt

VisDrone annotation line format (comma-separated, no header):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

object_category: 0 ignored regions, 1 pedestrian, 2 people, 3 bicycle,
                 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle,
                 9 bus, 10 motor, 11 others

Usage:
    python scripts/convert_visdrone.py --source /path/to/VisDrone2019-DET-train --split train
    python scripts/convert_visdrone.py --source /path/to/VisDrone2019-DET-val --split val
"""
import argparse
import shutil
from pathlib import Path

import cv2

# VisDrone category id -> this project's class id (see data/dataset.yaml).
# We keep the full 10-class VisDrone taxonomy; only "ignored regions" (0)
# and "others" (11) are dropped, since those aren't real object classes.
VISDRONE_TO_PROJECT = {
    1: 0,   # pedestrian     -> pedestrian
    2: 1,   # people         -> people
    3: 2,   # bicycle        -> bicycle
    4: 3,   # car            -> car
    5: 4,   # van            -> van
    6: 5,   # truck          -> truck
    7: 6,   # tricycle       -> tricycle
    8: 7,   # awning-tricycle -> awning-tricycle
    9: 8,   # bus            -> bus
    10: 9,  # motor          -> motor
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def convert_annotation(ann_path: Path, img_w: int, img_h: int):
    lines_out = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            left, top, w, h, _score, category = parts[:6]
            category = int(category)
            if category not in VISDRONE_TO_PROJECT:
                continue  # ignored region or a class we don't use

            left, top, w, h = float(left), float(top), float(w), float(h)
            cx = (left + w / 2) / img_w
            cy = (top + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))

            class_id = VISDRONE_TO_PROJECT[category]
            lines_out.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines_out


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone2019-DET to YOLO format.")
    parser.add_argument("--source", required=True, help="Path to a VisDrone2019-DET-train or -val folder")
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--prefix", default="visdrone", help="Filename prefix, avoids collisions when merging datasets")
    args = parser.parse_args()

    source = Path(args.source)
    images_src = source / "images"
    annotations_src = source / "annotations"
    if not images_src.exists() or not annotations_src.exists():
        raise FileNotFoundError(
            f"Expected '{images_src}' and '{annotations_src}' to exist. "
            f"Check that --source points at a VisDrone2019-DET-train/-val folder."
        )

    images_dst = DATA_DIR / "images" / args.split
    labels_dst = DATA_DIR / "labels" / args.split
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_src.glob("*.jpg")) + sorted(images_src.glob("*.png"))
    converted, skipped = 0, 0

    for img_path in image_files:
        ann_path = annotations_src / f"{img_path.stem}.txt"
        if not ann_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]

        yolo_lines = convert_annotation(ann_path, img_w, img_h)

        new_stem = f"{args.prefix}_{img_path.stem}"
        shutil.copy(img_path, images_dst / f"{new_stem}{img_path.suffix}")
        with open(labels_dst / f"{new_stem}.txt", "w") as f:
            f.write("\n".join(yolo_lines))
        converted += 1

    print(f"Converted {converted} images ({skipped} skipped) into {images_dst} / {labels_dst}")


if __name__ == "__main__":
    main()
