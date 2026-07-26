# PPE Compliance Detector

Real-time construction-site PPE compliance detection, built by fine-tuning YOLOv11 (Ultralytics) on a labeled construction-safety dataset. Detects people, hardhats, safety vests, and violations (missing hardhat / missing vest) from images, video, a live webcam, or a batch folder — with a Streamlit demo app for all four.

## Why this exists

Manual PPE compliance checks on a construction site are slow and inconsistent. This fine-tunes a real-time detector to flag violations automatically from a camera feed or recorded footage — a lightweight, deployable alternative to a manual walk-through audit.

## Tech stack

- **Model:** YOLOv11s (Ultralytics), transfer-learned from COCO-pretrained weights
- **Inference:** OpenCV
- **Deployment:** Streamlit
- **Training:** Google Colab (T4 GPU)

## Dataset

Fine-tuned on the [Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) dataset from Roboflow Universe (CC BY 4.0), ~2,801 images across 10 classes: `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`.

## Setup

```bash
git clone https://github.com/SuryadevChippada/Object-Detection.git
cd Object-Detection
pip install -r requirements.txt
```

## Training

1. Export the dataset above from Roboflow in YOLO format into `data/processed/` (Roboflow generates its own `data.yaml` there — don't hand-write one).
2. Run training (Colab recommended — see hyperparameters in `configs/train.yaml`):
   ```bash
   python src/train.py
   ```
3. Copy the resulting weights into the repo:
   ```bash
   cp runs/detect/train/weights/best.pt models/weights/best.pt
   ```

## Evaluation

```bash
python src/evaluate.py --weights models/weights/best.pt --split test
```
Reports precision, recall, mAP50, and mAP50-95 on the held-out test split. A confusion matrix and PR curves are saved alongside the run.

**Results:** _to be added once training completes._

## Inference (CLI)

```bash
# Single image
python src/infer.py --source path/to/image.jpg --output out.jpg

# Video
python src/infer.py --source path/to/video.mp4 --output out.mp4

# Webcam
python src/infer.py --source 0

# Batch folder
python src/infer.py --source path/to/folder --output path/to/folder/annotated
```

## Web app

```bash
streamlit run app.py
```
Supports image upload, video upload, live webcam, and batch-folder modes, with an adjustable confidence threshold.

## Project structure

```
├── configs/
│   └── train.yaml       # training hyperparameters
├── models/weights/       # best.pt (produced by training)
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── utils.py
├── app.py                # Streamlit entrypoint
└── requirements.txt
```
