"""
Shared helpers used by both the Ultralytics (.pt) and ONNX inference paths.
"""
import time

import cv2
import numpy as np


class FPSCounter:
    """Simple rolling FPS counter.

    Usage:
        fps_counter = FPSCounter()
        while True:
            fps_counter.tick()
            ...
            fps = fps_counter.get_fps()
    """

    def __init__(self, smoothing: int = 30):
        self.smoothing = smoothing
        self._last_time = None
        self._fps = 0.0

    def start(self):
        self._last_time = time.time()
        return self

    def tick(self):
        """Call once per frame. Updates and returns the current smoothed FPS."""
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            return self._fps

        instant_fps = 1.0 / max(now - self._last_time, 1e-8)
        self._last_time = now

        # Exponential moving average so the readout doesn't jitter wildly.
        alpha = 2.0 / (self.smoothing + 1)
        self._fps = (
            instant_fps if self._fps == 0.0 else alpha * instant_fps + (1 - alpha) * self._fps
        )
        return self._fps

    def get_fps(self) -> float:
        return self._fps


def letterbox(image: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    """
    Resize an image to a square `new_shape` while preserving aspect ratio,
    padding the remainder with `color`. Returns the resized image plus the
    scale factor and (left, top) padding, so boxes predicted on the resized
    image can be mapped back to original coordinates.
    """
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = new_shape - new_w, new_shape - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, left, top


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] -> [x1, y1, x2, y2]."""
    xyxy = np.empty_like(boxes)
    xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return xyxy


def decode_yolov8_output(
    output: np.ndarray,
    conf_threshold: float,
    num_classes: int,
):
    """
    Decode a raw YOLOv8 ONNX export output of shape (1, 4 + num_classes, N)
    into boxes (xyxy, in the padded/letterboxed image space), scores, and
    class ids. Boxes below `conf_threshold` are dropped here; NMS still
    needs to be applied afterwards.
    """
    preds = np.squeeze(output).T  # -> (N, 4 + num_classes)

    class_scores = preds[:, 4:4 + num_classes]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    keep = scores >= conf_threshold
    boxes = xywh_to_xyxy(preds[keep, :4])
    scores = scores[keep]
    class_ids = class_ids[keep]
    return boxes, scores, class_ids


def scale_boxes_to_original(boxes: np.ndarray, scale: float, pad_left: int, pad_top: int):
    """Undo the letterbox transform so boxes line up with the original frame."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale
    return boxes


def draw_detections(frame, boxes, scores, class_ids, class_names: dict, color=(0, 255, 0)):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = class_names.get(int(cls_id), str(int(cls_id)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame
