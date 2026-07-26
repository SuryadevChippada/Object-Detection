"""Shared logging setup and bounding-box drawing, used by train/evaluate/infer."""

import logging

import cv2

# Fixed per-class color so boxes stay visually consistent across frames and runs
_PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (238, 130, 238),
    (255, 215, 0), (0, 206, 209), (255, 140, 0), (154, 205, 50),
    (199, 21, 133), (112, 128, 144),
]


def get_logger(name: str = "ppe_detector") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def draw_detections(frame, boxes, class_names):
    """Draw YOLO boxes + label + confidence onto a BGR frame in place, returns the frame."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        color = _PALETTE[cls_id % len(_PALETTE)]
        label = f"{class_names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    return frame


if __name__ == "__main__":
    get_logger().info("logger ok")
    # ponytail: draw_detections needs a real ultralytics Boxes object (tensor-backed
    # .xyxy/.cls/.conf) — a hand-rolled mock would be more code than the function.
    # Exercised end-to-end by the infer.py smoke test instead.
