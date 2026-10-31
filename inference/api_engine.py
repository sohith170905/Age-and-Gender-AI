"""Local model adapter for the FastAPI endpoint, preserving app preprocessing contracts."""
from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import load_config
from core.model_registry import load_registry, resolve_model_path
from services.analysis import analyze_image_bytes
from services.uploads import decode_image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT)
REGISTRY = load_registry(ROOT)


def active_model_id():
    requested = CONFIG.default_age_gender_model or REGISTRY.get("default_age_gender_model", "agevision_onnx")
    if requested not in ("agevision_onnx", "gra_net_compact"):
        requested = "agevision_onnx"
    if requested == "agevision_onnx" and not resolve_model_path(ROOT, REGISTRY["models"][requested]).is_file():
        requested = "gra_net_compact"
    return requested


@lru_cache(maxsize=2)
def load_engine(model_id):
    model = REGISTRY["models"][model_id]
    path = resolve_model_path(ROOT, model)
    if not path.is_file():
        raise RuntimeError("The selected local model is unavailable.")
    if model_id == "agevision_onnx":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 2:
            raise RuntimeError("AgeVision model input/output layout is unsupported.")
        return model_id, session, session.get_inputs()[0].name
    import tensorflow as tf
    custom = {}
    try:
        from gra_net import _GatedBlend
        custom["_GatedBlend"] = _GatedBlend
    except Exception:
        pass
    return model_id, tf.keras.models.load_model(path, compile=False, custom_objects=custom), None


@lru_cache(maxsize=1)
def load_detector():
    path = resolve_model_path(ROOT, REGISTRY["models"]["yunet"])
    if not path.is_file() or not hasattr(cv2, "FaceDetectorYN_create"):
        from face_detector import FaceDetector
        return FaceDetector()
    return cv2.FaceDetectorYN_create(str(path), "", (320, 320), 0.60, 0.30, 5000)


def detect(image):
    detector = load_detector()
    if hasattr(detector, "setInputSize"):
        width, height = image.size
        detector.setInputSize((width, height))
        _, detections = detector.detect(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        if detections is None:
            return []
        boxes = []
        for row in detections:
            x, y, w, h = [int(round(float(v))) for v in row[:4]]
            if float(row[14]) >= 0.60 and w >= 24 and h >= 24:
                boxes.append((max(0, x), max(0, y), min(w, width - max(0, x)), min(h, height - max(0, y))))
        boxes.sort(key=lambda box: (box[1], box[0]))
        return boxes
    return detector.detect(image)


def crop(image, box, padding=0.20):
    x, y, w, h = map(int, box)
    px, py = int(w * padding), int(h * padding)
    return image.crop((max(0, x-px), max(0, y-py), min(image.width, x+w+px), min(image.height, y+h+py)))


def predict(face, *, backend, source_image, bbox):
    model_id, model, input_name = load_engine(backend)
    if model_id == "agevision_onnx":
        face = crop(source_image, bbox, padding=0.10)
        resized = face.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None, ...])
        age_logits, gender_logits = model.run(None, {input_name: tensor})
        age_cumulative = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(age_logits).reshape(1, -1)[0], -88, 88)))
        estimated = float(np.clip(age_cumulative.sum(), 0, 100))
        cumulative = np.minimum.accumulate(np.clip(age_cumulative, 0, 1))
        distribution = np.clip(np.concatenate(([1-cumulative[0]], cumulative[:-1]-cumulative[1:], [cumulative[-1]])), 0, 1)
        distribution /= max(float(distribution.sum()), 1e-12)
        age = int(round(estimated))
        lo, hi = max(0, age-5), min(100, age+5)
        logits = np.asarray(gender_logits, dtype=np.float32).reshape(1, -1)[0]
        probs = np.exp(logits - np.max(logits)); probs /= max(float(probs.sum()), 1e-12)
        gender_index = int(np.argmax(probs))
        return {"gender": "Male" if gender_index == 1 else "Female", "gender_confidence": round(float(probs[gender_index]*100), 1), "age": age, "age_raw": round(estimated, 2), "age_group": f"{(age//10)*10}-{min(99,(age//10)*10+9)}", "age_confidence": round(float(distribution[lo:hi+1].sum()*100), 1), "age_consistent": None}
    input_shape = getattr(model, "input_shape", (None, 32, 32, 3))
    size = tuple(int(value) for value in input_shape[1:3])
    arr = np.asarray(face.convert("RGB").resize(size), dtype=np.float32)[None, ...] / 255.0
    outputs = model.predict(arr, verbose=0)
    gender_probs = np.nan_to_num(np.asarray(outputs[0][0], dtype=np.float32).reshape(-1), nan=0, posinf=0, neginf=0)
    gender_index = 1 if len(gender_probs) > 1 and float(gender_probs[1]) >= 0.32 else 0
    age_probs = np.nan_to_num(np.asarray(outputs[1][0], dtype=np.float32).reshape(-1), nan=0, posinf=0, neginf=0)
    class_index = int(np.argmax(age_probs)); low=class_index*10; high=low+9
    raw = float(np.nan_to_num(np.asarray(outputs[2]).reshape(-1)[0], nan=25, posinf=116, neginf=0))
    paper_head = "age_remainder" in getattr(model, "output_names", [])
    research_head = getattr(model, "name", "") == "ResearchAgeGenderMobileNetV3"
    if paper_head:
        estimated = low + float(np.clip(raw*9, 0, 9)); consistent=True
    elif research_head:
        estimated = float(np.clip(raw*116, 0, 116)); consistent=low <= estimated <= high
    else:
        estimated = float(np.clip(raw, 0, 116)); consistent=low <= estimated <= high
    age = int(np.clip(round(estimated), 0, 116))
    age_conf = min(float(age_probs[class_index]*100), 50.0) if not consistent else float(age_probs[class_index]*100)
    return {"gender": "Male" if gender_index == 0 else "Female", "gender_confidence": round(float(gender_probs[gender_index]*100),1), "age": age, "age_raw": round(estimated,2), "age_group": f"{(age//10)*10}-{(age//10)*10+9}", "age_confidence": round(age_conf,1), "age_consistent": consistent}


def analyze(image_bytes: bytes, include_emotion: bool = False):
    backend = active_model_id()
    image, results, truncated = analyze_image_bytes(
        image_bytes, backend, decode=lambda data: decode_image(data, max_bytes=CONFIG.max_upload_bytes, max_pixels=CONFIG.max_image_pixels),
        detect=detect, crop=crop, predict=predict, max_faces=CONFIG.max_faces,
    )
    faces = []
    for box, prediction in results:
        faces.append({"bbox": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]}, **prediction})
    if include_emotion and faces:
        from emotion.model_loader import load_emotion_assets
        from emotion.predictor import EmotionPredictor
        from emotion.tta import TTA5
        model, classes, digest = load_emotion_assets()
        predictor = EmotionPredictor(model, classes, detect)
        emotions = predictor.predict(image, mode=TTA5, boxes=tuple(tuple(face["bbox"][key] for key in ("x", "y", "width", "height")) for face in faces))
        for face, emotion in zip(faces, emotions):
            top3 = sorted(emotion["probabilities"].items(), key=lambda item: item[1], reverse=True)[:3]
            face["expression"] = {
                "label": emotion["emotion"],
                "confidence": emotion["confidence"],
                "top3": [{"label": label, "confidence": round(float(probability) * 100, 2)} for label, probability in top3],
            }
        expression_digest = digest
    else:
        expression_digest = None
    return {"model": REGISTRY["models"][backend]["display_name"], "model_version": REGISTRY["models"][backend]["version"], "face_count": len(faces), "truncated": truncated, "faces": faces, "expression_model": "Expression Model v1" if include_emotion else None, "expression_sha256": expression_digest}
