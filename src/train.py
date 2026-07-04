import argparse

from ultralytics import YOLO

import config


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

    results = model.train(
        data=args.data,
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
