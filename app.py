"""Streamlit deployment: image, video, webcam, and batch-folder PPE detection."""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from infer import Detector  # noqa: E402

WEIGHTS = Path(__file__).parent / "models" / "weights" / "best.pt"

CONFIDENCE_LABELS = [
    (0.90, "Excellent"),
    (0.85, "Very good"),
    (0.75, "Good"),
    (0.65, "Okay"),
    (0.50, "Poor"),
]


def confidence_label(conf: float) -> str:
    for threshold, label in CONFIDENCE_LABELS:
        if conf >= threshold:
            return label
    return "Very poor"


@st.cache_resource
def load_detector() -> Detector:
    return Detector(WEIGHTS)


st.set_page_config(page_title="PPE Compliance Detector", layout="wide")
st.title("PPE Compliance Detector")

if not WEIGHTS.exists():
    st.error(f"No trained weights found at `{WEIGHTS}`. Run training first (see README).")
    st.stop()

mode = st.sidebar.radio("Input", ["Image", "Video", "Webcam", "Batch folder"])
conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.caption(f"Quality at this threshold: **{confidence_label(conf)}**")

detector = load_detector()
detector.conf = conf

if mode == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        frame = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        st.image(detector.predict(frame), channels="BGR", use_container_width=True)

elif mode == "Video":
    file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if file:
        with tempfile.NamedTemporaryFile(suffix=Path(file.name).suffix, delete=False) as tmp:
            tmp.write(file.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        frame_slot = st.empty()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_slot.image(detector.predict(frame), channels="BGR", use_container_width=True)
        cap.release()

elif mode == "Webcam":
    # ponytail: cv2.VideoCapture(0) opens the machine running `streamlit run` —
    # correct for local demo use, not a cloud-hosted deployment. Unchecking mid-loop
    # won't interrupt cleanly (Streamlit's rerun model can't preempt a running script);
    # use the on-screen Stop control, or `python src/infer.py --source 0` for a
    # cleaner continuous feed.
    run = st.checkbox("Start webcam")
    frame_slot = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        while run and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_slot.image(detector.predict(frame), channels="BGR", use_container_width=True)
        cap.release()

else:  # Batch folder
    folder = st.text_input("Folder path (images)")
    if folder and Path(folder).is_dir() and st.button("Run batch detection"):
        output_dir = Path(folder) / "annotated"
        detector.process_batch(Path(folder), output_dir)
        st.success(f"Done — annotated images saved to {output_dir}")
