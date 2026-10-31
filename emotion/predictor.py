"""Multi-face AG-44B emotion predictor with source-compatible probability TTA."""

from __future__ import annotations

import threading
import time

import numpy as np
from PIL import Image

from .face_detection import detect_faces
from .preprocessing import preprocess_face
from .tta import STANDARD, TTA5, build_input_batch


class EmotionPredictor:
    def __init__(self, model, classes, face_detector):
        self.model = model
        self.classes = tuple(classes)
        self.face_detector = face_detector
        self._model_lock = threading.RLock()
        if len(self.classes) != 7:
            raise ValueError("AG-44B must have seven emotion classes.")

    def predict(self, image, mode=TTA5, boxes=None):
        """Predict each face; confidence is the top averaged softmax probability."""
        started = time.perf_counter()
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(np.asarray(image))
            except Exception as exc:
                raise ValueError("Invalid image supplied for emotion prediction.") from exc
        image = image.convert("RGB")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("The image is empty.")
        if mode not in (STANDARD, TTA5):
            raise ValueError("Emotion mode must be 'standard' or 'tta5'.")

        face_boxes = detect_faces(image, self.face_detector) if boxes is None else [
            tuple(int(round(float(v))) for v in box)
            for box in boxes
            if len(box) == 4 and float(box[2]) > 0 and float(box[3]) > 0
        ]
        results = []
        for face_id, bbox in enumerate(face_boxes, start=1):
            face_started = time.perf_counter()
            gray, _ = preprocess_face(image, bbox)
            batch = build_input_batch(gray, mode)
            with self._model_lock:
                raw = self.model.predict(batch, batch_size=len(batch), verbose=0)
            probabilities = np.asarray(raw, dtype=np.float32)
            if probabilities.ndim == 1:
                probabilities = probabilities[None, :]
            if probabilities.shape != (len(batch), len(self.classes)):
                raise RuntimeError(f"Unexpected AG-44B prediction shape: {probabilities.shape}")
            if not np.all(np.isfinite(probabilities)):
                raise RuntimeError("AG-44B returned non-finite probabilities.")

            # The verified evaluator takes the equal arithmetic mean of the five
            # softmax vectors without a second normalization step.
            averaged = probabilities.mean(axis=0, dtype=np.float32)
            index = int(np.argmax(averaged))
            probability_map = {
                name.lower(): float(averaged[i])
                for i, name in enumerate(self.classes)
            }
            results.append({
                "face_id": face_id,
                "bbox": tuple(bbox),
                "emotion": self.classes[index],
                "confidence": float(averaged[index] * 100.0),
                "probabilities": probability_map,
                "inference_mode": mode,
                "inference_latency_ms": (time.perf_counter() - face_started) * 1000.0,
            })

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for item in results:
            item["total_latency_ms"] = elapsed_ms
        return results