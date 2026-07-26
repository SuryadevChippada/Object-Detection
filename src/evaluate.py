"""Evaluation entrypoint: runs model.val() on the held-out test split and reports metrics."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import get_logger

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "data.yaml"
DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "weights" / "best.pt"

logger = get_logger()


def evaluate(weights: Path = DEFAULT_WEIGHTS, data: Path = DATA_PATH, split: str = "test"):
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split=split)

    precision, recall, map50, map50_95 = metrics.box.mean_results()
    logger.info(f"Precision: {precision:.3f}  Recall: {recall:.3f}")
    logger.info(f"mAP50: {map50:.3f}  mAP50-95: {map50_95:.3f}")
    logger.info(f"Full report (incl. confusion matrix): {metrics.save_dir}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.split)
