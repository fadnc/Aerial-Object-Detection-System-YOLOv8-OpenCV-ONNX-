"""
Convert UAVDT-Benchmark-M (detection) ground truth to YOLO format and
copy the results into this project's data/images/<split> and
data/labels/<split>.

Expected input layout (matches the most common UAVDT-Benchmark-M mirrors):
    UAVDT/
        UAV-benchmark-M/
            M0101/
                img000001.jpg
                img000002.jpg
                ...
            M0201/
                ...
        UAV-benchmark-MOTD_v1.0/
            GT/
                M0101_gt_whole.txt
                M0201_gt_whole.txt
                ...

GT line format (comma-separated, no header):
    frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
    out-of-view, occlusion, object_category

object_category: 1 = car, 2 = truck, 3 = bus. These get mapped onto the
matching slots in this project's full 10-class scheme (see
data/dataset.yaml): car->3, truck->5, bus->8. UAVDT has no pedestrian,
van, bicycle, tricycle, or motor annotations - those classes will need
to come from VisDrone/Roboflow instead.

NOTE: UAVDT mirrors are inconsistent about folder/file naming. If this
script can't find your GT files, open one in a text editor to confirm the
column order matches above, then adjust GT_SUFFIX or the paths below.

Usage:
    python scripts/convert_uavdt.py --source /path/to/UAVDT --split train --sequences M0101 M0201
    python scripts/convert_uavdt.py --source /path/to/UAVDT --split val --sequences M0401
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2

# UAVDT category id -> this project's class id (see data/dataset.yaml:
# 3=car, 5=truck, 8=bus).
UAVDT_TO_PROJECT = {
    1: 3,  # car   -> car
    2: 5,  # truck -> truck
    3: 8,  # bus   -> bus
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GT_SUFFIX = "_gt_whole.txt"


def load_gt(gt_path: Path):
    """Group GT rows by frame index -> list of (left, top, w, h, class_id)."""
    by_frame = defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame_idx, _tid, left, top, w, h, _oov, _occ, category = parts[:9]
            category = int(category)
            if category not in UAVDT_TO_PROJECT:
                continue
            by_frame[int(frame_idx)].append(
                (float(left), float(top), float(w), float(h), UAVDT_TO_PROJECT[category])
            )
    return by_frame


def main():
    parser = argparse.ArgumentParser(description="Convert UAVDT-Benchmark-M to YOLO format.")
    parser.add_argument("--source", required=True, help="Path to the UAVDT root folder")
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--sequences", nargs="+", required=True,
                         help="Sequence folder names to include, e.g. M0101 M0201")
    parser.add_argument("--frame-stride", type=int, default=5,
                         help="Keep every Nth frame only - UAVDT is video, adjacent frames are near-duplicates")
    parser.add_argument("--prefix", default="uavdt")
    args = parser.parse_args()

    source = Path(args.source)
    images_root = source / "UAV-benchmark-M"
    gt_root = source / "UAV-benchmark-MOTD_v1.0" / "GT"

    images_dst = DATA_DIR / "images" / args.split
    labels_dst = DATA_DIR / "labels" / args.split
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    total = 0
    for seq in args.sequences:
        seq_img_dir = images_root / seq
        gt_path = gt_root / f"{seq}{GT_SUFFIX}"
        if not seq_img_dir.exists() or not gt_path.exists():
            print(f"Skipping {seq}: missing {seq_img_dir} or {gt_path}")
            continue

        by_frame = load_gt(gt_path)
        frame_files = sorted(seq_img_dir.glob("*.jpg"))

        for i, img_path in enumerate(frame_files):
            if i % args.frame_stride != 0:
                continue
            try:
                frame_idx = int("".join(c for c in img_path.stem if c.isdigit()))
            except ValueError:
                continue
            boxes = by_frame.get(frame_idx, [])
            if not boxes:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            lines_out = []
            for left, top, w, h, class_id in boxes:
                cx = (left + w / 2) / w_img
                cy = (top + h / 2) / h_img
                nw = w / w_img
                nh = h / h_img
                cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
                lines_out.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            new_stem = f"{args.prefix}_{seq}_{img_path.stem}"
            cv2.imwrite(str(images_dst / f"{new_stem}.jpg"), img)
            with open(labels_dst / f"{new_stem}.txt", "w") as f:
                f.write("\n".join(lines_out))
            total += 1

    print(f"Converted {total} frames into {images_dst} / {labels_dst}")


if __name__ == "__main__":
    main()
