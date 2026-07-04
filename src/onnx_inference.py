"""
Run detection using an exported ONNX model (no PyTorch/Ultralytics needed
at inference time). Produce the ONNX file first with src/export_onnx.py.
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

import config
from utils import (
    FPSCounter,
    decode_yolov8_output,
    draw_detections,
    letterbox,
    scale_boxes_to_original,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX-based aerial object detection.")
    parser.add_argument("--source", default=config.VIDEO_SOURCE)
    parser.add_argument("--model", default=str(config.ONNX_MODEL_PATH))
    parser.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    parser.add_argument("--save", default=None, help="Optional path to save annotated output video")
    return parser.parse_args()


def preprocess(frame, img_size):
    padded, scale, pad_left, pad_top = letterbox(frame, img_size)
    img = padded.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return img, scale, pad_left, pad_top


def postprocess(output, scale, pad_left, pad_top, conf_threshold, iou_threshold, num_classes):
    boxes, scores, class_ids = decode_yolov8_output(output, conf_threshold, num_classes)
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    # cv2.dnn.NMSBoxes expects [x, y, w, h]
    xywh = boxes.copy()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    indices = np.array(indices).reshape(-1) if len(indices) else np.array([], dtype=int)

    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]

    boxes = scale_boxes_to_original(boxes, scale, pad_left, pad_top)
    return boxes, scores, class_ids


def main():
    args = parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    num_classes = config.NUM_CLASSES or (session.get_outputs()[0].shape[1] - 4)

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
            input_tensor, scale, pad_left, pad_top = preprocess(frame, args.imgsz)
            outputs = session.run(None, {input_name: input_tensor})[0]

            boxes, scores, class_ids = postprocess(
                outputs, scale, pad_left, pad_top, args.conf, args.iou, num_classes
            )
            annotated = draw_detections(frame, boxes, scores, class_ids, config.CLASS_NAMES)

            cv2.putText(
                annotated, f"FPS: {fps_counter.get_fps():.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            cv2.imshow("Aerial Object Detection (ONNX)", annotated)
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
