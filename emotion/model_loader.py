"""Verified AG-44B loader. Does not train or modify model weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_SHA256 = "D1EF71D7A88D3C010ACAEBB175A8EA88EF04A9250BFDEE0908FE47D6ED58FE0B"
CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "emotion" / "emotion_model.keras"
CLASS_MAP_PATH = Path(__file__).resolve().parents[1] / "models" / "emotion" / "expression_classes.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_emotion_assets(model_path: Path = MODEL_PATH, class_map_path: Path = CLASS_MAP_PATH):
    """Load and validate the frozen checkpoint and its exact seven-class map."""
    model_path = Path(model_path)
    class_map_path = Path(class_map_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"AG-44B checkpoint not found: {model_path}")
    if not class_map_path.is_file():
        raise FileNotFoundError(f"Emotion class map not found: {class_map_path}")

    actual_hash = _sha256(model_path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(f"AG-44B checkpoint SHA-256 mismatch: {actual_hash}")

    with class_map_path.open("r", encoding="utf-8") as handle:
        class_map = json.load(handle)
    ordered_classes = tuple(str(class_map[str(index)]).capitalize() for index in range(7))
    if ordered_classes != CLASS_NAMES:
        raise RuntimeError(f"Unexpected emotion class order: {ordered_classes}")

    import tensorflow as tf

    # Allow TensorFlow and ONNX Runtime to share a GPU where one is available.
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except (RuntimeError, ValueError):
        # TensorFlow may already be initialized by the existing GRA_Net path.
        pass

    model = tf.keras.models.load_model(str(model_path), compile=False)
    if tuple(model.input_shape[-3:]) != (48, 48, 1):
        raise RuntimeError(f"Unexpected AG-44B input shape: {model.input_shape}")
    if int(model.output_shape[-1]) != 7:
        raise RuntimeError(f"Unexpected AG-44B output shape: {model.output_shape}")
    if int(model.count_params()) != 1_244_135:
        raise RuntimeError(f"Unexpected AG-44B parameter count: {model.count_params()}")

    # Warm once, when cached, so reported request latency excludes first-call setup.
    model(np.zeros((1, 48, 48, 1), dtype=np.float32), training=False)
    return model, CLASS_NAMES, actual_hash