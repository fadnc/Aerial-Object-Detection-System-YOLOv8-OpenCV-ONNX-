"""
Merge a Roboflow "YOLOv8" export into this project's data/ folder,
remapping class ids to match data/dataset.yaml (person, car, truck, bus).

Expected input layout (standard Roboflow YOLOv8 export):
    <roboflow_export>/
        data.yaml
        train/images/*.jpg
        train/labels/*.txt
        valid/images/*.jpg
        valid/labels/*.txt

Usage:
    python scripts/prepare_roboflow.py --source /path/to/roboflow_export --split train
    python scripts/prepare_roboflow.py --source /path/to/roboflow_export --split val
"""
import argparse
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROJECT_CLASSES = {
    "pedestrian": 0,
    "people": 1,
    "bicycle": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning-tricycle": 7,
    "bus": 8,
    "motor": 9,
}

# If your Roboflow dataset uses different words for the same concept,
# add them here (lowercase) so they still map onto our 10 classes.
# NOTE: many datasets only have a generic "person" class with no
# pedestrian/people distinction - this defaults "person" to "pedestrian",
# edit if your source is annotating loose crowds and "people" fits better.
ALIASES = {
    "person": "pedestrian",
    "pedestrians": "pedestrian",
    "cyclist": "bicycle",
    "motorcycle": "motor",
    "motorbike": "motor",
    "moto": "motor",
    "awning tricycle": "awning-tricycle",
}


def build_class_map(roboflow_yaml: Path):
    with open(roboflow_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    mapping = {}
    for rf_id, name in enumerate(names):
        key = ALIASES.get(name.lower(), name.lower())
        if key in PROJECT_CLASSES:
            mapping[rf_id] = PROJECT_CLASSES[key]
    return mapping


def remap_label_file(src_path: Path, mapping: dict):
    lines_out = []
    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rf_class = int(parts[0])
            if rf_class not in mapping:
                continue
            parts[0] = str(mapping[rf_class])
            lines_out.append(" ".join(parts))
    return lines_out


def main():
    parser = argparse.ArgumentParser(description="Merge a Roboflow YOLOv8 export into data/.")
    parser.add_argument("--source", required=True, help="Path to the extracted Roboflow export")
    parser.add_argument("--split", required=True, choices=["train", "val"], help="Destination split in this project")
    parser.add_argument("--roboflow-split", default=None,
                         help="Source folder name inside the export (default: 'train' or 'valid')")
    parser.add_argument("--prefix", default="roboflow")
    args = parser.parse_args()

    source = Path(args.source)
    rf_split = args.roboflow_split or ("valid" if args.split == "val" else "train")

    mapping = build_class_map(source / "data.yaml")
    if not mapping:
        raise ValueError(
            "No overlapping class names found between the Roboflow dataset's data.yaml "
            "and this project's classes (see PROJECT_CLASSES above). Check the 'names' "
            "field in data.yaml and add aliases to ALIASES if needed."
        )
    print(f"Class mapping (roboflow id -> project id): {mapping}")

    images_src = source / rf_split / "images"
    labels_src = source / rf_split / "labels"
    images_dst = DATA_DIR / "images" / args.split
    labels_dst = DATA_DIR / "labels" / args.split
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_path in sorted(images_src.iterdir()):
        label_path = labels_src / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        lines_out = remap_label_file(label_path, mapping)
        if not lines_out:
            continue  # image only contained classes we don't use

        new_stem = f"{args.prefix}_{img_path.stem}"
        shutil.copy(img_path, images_dst / f"{new_stem}{img_path.suffix}")
        with open(labels_dst / f"{new_stem}.txt", "w") as f:
            f.write("\n".join(lines_out))
        converted += 1

    print(f"Merged {converted} images into {images_dst} / {labels_dst}")


if __name__ == "__main__":
    main()
