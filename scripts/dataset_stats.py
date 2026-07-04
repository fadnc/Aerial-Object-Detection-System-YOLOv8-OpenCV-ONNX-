"""
Print per-class instance counts across data/labels/{train,val} so you can
sanity-check the merged dataset - e.g. spot heavy class imbalance after
combining VisDrone + UAVDT + Roboflow sources (VisDrone/Roboflow add
plenty of 'person', UAVDT adds none).

Usage:
    python scripts/dataset_stats.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402


def count_split(split):
    counts = Counter()
    label_dir = config.DATA_DIR / "labels" / split
    n_images = 0
    for label_file in label_dir.glob("*.txt"):
        n_images += 1
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                counts[class_id] += 1
    return n_images, counts


def main():
    for split in ("train", "val"):
        n_images, counts = count_split(split)
        print(f"\n{split}: {n_images} labeled images")
        if not config.CLASS_NAMES:
            print("  (could not load class names from data/dataset.yaml)")
            continue
        for class_id, name in sorted(config.CLASS_NAMES.items()):
            print(f"  {name:10s}: {counts.get(class_id, 0)}")


if __name__ == "__main__":
    main()
