"""Training entrypoint: fine-tunes YOLOv11 on the PPE dataset via configs/train.yaml."""

from pathlib import Path

import yaml
from ultralytics import YOLO

from utils import get_logger

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "train.yaml"

logger = get_logger()


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()
    model_name = cfg.pop("model")
    data_path = cfg.pop("data")

    logger.info(f"Loading pretrained checkpoint: {model_name}")
    model = YOLO(model_name)

    logger.info(f"Starting training on {data_path} with config: {cfg}")
    results = model.train(data=data_path, **cfg)

    best_weights = results.save_dir / "weights" / "best.pt"
    logger.info(f"Training complete. Best weights: {best_weights}")
    return results


if __name__ == "__main__":
    train()
