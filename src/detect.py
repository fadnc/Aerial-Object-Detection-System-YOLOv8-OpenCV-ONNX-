"""
Live/video object detection using a trained Ultralytics (.pt) model.
For ONNX-based inference instead, see onnx_inference.py.
"""
import argparse

import cv2
from ultralytics import YOLO

import config
from utils import FPSCounter


def parse_args():
    parser = argparse.ArgumentParser(description="Run aerial object detection.")
    parser.add_argument(
        "--source", default=config.VIDEO_SOURCE,
        help="Video file path, RTSP URL, or camera index (e.g. 0 for webcam)",
    )
    parser.add_argument("--weights", default=None, help="Path to trained .pt weights")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    parser.add_argument("--save", default=None, help="Optional path to save annotated output video")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = args.weights or str(config.find_latest_weights())
    model = YOLO(weights_path)

    # Allow numeric camera indices passed as strings, e.g. "0"
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    fps_counter = FPSCounter().start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps_counter.tick()
            results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
            annotated = results[0].plot()

            cv2.putText(
                annotated,
                f"FPS: {fps_counter.get_fps():.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Aerial Object Detection", annotated)
            if writer is not None:
                writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
