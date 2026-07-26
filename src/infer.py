"""Inference wrapper: single entrypoint for image, video, webcam, and batch-folder detection."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_detections, get_logger

DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "weights" / "best.pt"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

logger = get_logger()


class Detector:
    """Loads a YOLO checkpoint once, reused across all input modes (image/video/webcam/batch)."""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, conf: float = 0.25):
        self.model = YOLO(str(weights))
        self.conf = conf

    def predict(self, frame):
        """Run detection on a single BGR frame, return it annotated in place."""
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        return draw_detections(frame, results.boxes, self.model.names)

    def process_image(self, image_path: Path, output_path: Path = None):
        frame = cv2.imread(str(image_path))
        annotated = self.predict(frame)
        if output_path:
            cv2.imwrite(str(output_path), annotated)
            logger.info(f"Saved: {output_path}")
        return annotated

    def process_batch(self, input_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        images = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        logger.info(f"Processing {len(images)} images from {input_dir}")
        for image_path in images:
            self.process_image(image_path, output_dir / image_path.name)

    def process_video(self, source, output_path: Path = None):
        """Handles both video files and webcam (source=int device index)."""
        cap = cv2.VideoCapture(source)
        writer = None

        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            annotated = self.predict(frame)
            if writer:
                writer.write(annotated)
            else:
                cv2.imshow("PPE Detector (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Saved: {output_path}")
        cv2.destroyAllWindows()


def detect_mode(source: str) -> str:
    if source.isdigit():
        return "webcam"
    path = Path(source)
    if path.is_dir():
        return "batch"
    if path.suffix.lower() in VIDEO_EXTS:
        return "video"
    return "image"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="image/video path, folder path, or webcam index (e.g. 0)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None,
                         help="output file (image/video) or folder (batch)")
    args = parser.parse_args()

    detector = Detector(args.weights, args.conf)
    mode = detect_mode(args.source)
    logger.info(f"Mode: {mode}")

    if mode == "webcam":
        detector.process_video(int(args.source), args.output)
    elif mode == "video":
        detector.process_video(args.source, args.output)
    elif mode == "batch":
        detector.process_batch(Path(args.source), args.output or Path(args.source) / "annotated")
    else:
        detector.process_image(Path(args.source), args.output)
