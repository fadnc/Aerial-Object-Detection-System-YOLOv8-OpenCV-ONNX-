"""
Export a trained .pt checkpoint to ONNX format for use with onnx_inference.py.

Usage:
    python src/export_onnx.py
    python src/export_onnx.py --weights runs/detect/aerial_detector3/weights/best.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained weights to ONNX.")
    parser.add_argument(
        "--weights", default=None,
        help="Path to a trained best.pt. Defaults to the most recently trained run.",
    )
    parser.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights) if args.weights else config.find_latest_weights()

    print(f"Exporting {weights_path} -> ONNX (imgsz={args.imgsz}, opset={args.opset})")
    model = YOLO(str(weights_path))
    exported_path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=args.simplify)

    # Ultralytics exports next to the source weights; copy it to the
    # models/ dir so onnx_inference.py can find it via config.ONNX_MODEL_PATH.
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(exported_path, config.ONNX_MODEL_PATH)
    print(f"Saved ONNX model to {config.ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()
