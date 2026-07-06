import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

import config


def prepare_resolved_data_yaml(data_path: Path) -> Path:
    """
    Ultralytics resolves a relative 'path:' field in a dataset yaml against
    its own global `datasets_dir` setting (see ~/Ultralytics/settings.yaml
    or %APPDATA%\\Ultralytics\\settings.yaml on Windows) - NOT relative to
    the yaml file's own location. That means our checked-in, portable
    `path: data` can resolve to the wrong folder on any given machine
    (e.g. "<project>/datasets/data" instead of "<project>/data").

    To keep dataset.yaml itself relative/portable in git, write out a
    sibling file with an absolute 'path' pointing at this project's actual
    data/ folder, and train against that instead.
    """
    with open(data_path, "r") as f:
        data = yaml.safe_load(f)
    data["path"] = str(data_path.parent.resolve())

    resolved_path = data_path.parent / "dataset.resolved.yaml"
    with open(resolved_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    return resolved_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the aerial object detector.")
    parser.add_argument("--model", default=config.MODEL_NAME, help="Base checkpoint to fine-tune from")
    parser.add_argument("--data", default=str(config.DATA_YAML), help="Path to dataset yaml")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default=config.RUN_NAME)
    parser.add_argument("--device", default=None, help="e.g. '0' for GPU 0, 'cpu' for CPU")
    return parser.parse_args()


def train():
    args = parse_args()
    model = YOLO(args.model)

    resolved_data = prepare_resolved_data_yaml(Path(args.data))

    results = model.train(
        data=str(resolved_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        device=args.device,
    )

    print(f"\nTraining complete. Best weights saved under: "
          f"{config.RUNS_DIR}/{args.name}*/weights/best.pt")
    return results


if __name__ == "__main__":
    train()