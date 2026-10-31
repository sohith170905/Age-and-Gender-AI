"""Shared in-memory orchestration for uploaded image analysis."""
from typing import Callable
import time


def analyze_image_bytes(image_bytes: bytes, backend: str, *, decode: Callable, detect: Callable,
                        crop: Callable, predict: Callable, max_faces: int = 80):
    """Run the existing detector and predictor callbacks without persisting image data."""
    decode_started = time.perf_counter()
    image = decode(image_bytes)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    detection_started = time.perf_counter()
    boxes = list(detect(image))
    detection_ms = (time.perf_counter() - detection_started) * 1000.0
    truncated = len(boxes) > max_faces
    boxes = boxes[:max_faces]
    results = []
    for box in boxes:
        face_started = time.perf_counter()
        face_crop = crop(image, box, padding=0.20)
        prediction = predict(face_crop, backend=backend, source_image=image, bbox=box)
        face_ms = (time.perf_counter() - face_started) * 1000.0
        if prediction is not None:
            prediction = dict(prediction)
            prediction["timing_ms"] = {"decode": round(decode_ms, 2), "detection": round(detection_ms, 2), "face_crop_and_prediction": round(face_ms, 2)}
            results.append((tuple(box), prediction))
    return image, tuple(results), truncated
