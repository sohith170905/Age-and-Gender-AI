"""Focused mocked tests for batch emotion inference and expression exports."""
import ast
import csv
import io
import json

import pytest

from emotion.tta import TTA5
from services.batch_emotions import predict_batch_emotions
from services.exports import csv_bytes, json_bytes, result_rows


def prediction(age, gender):
    return {
        "gender": gender,
        "gender_confidence": 91.2,
        "age": age,
        "age_group": f"{(age // 10) * 10}-{(age // 10) * 10 + 9}",
        "age_confidence": 84.5,
        "uncertain": False,
    }


def emotion(face_id, bbox, label, confidence, probabilities):
    return {
        "face_id": face_id,
        "bbox": bbox,
        "emotion": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def test_single_face_batch_calls_predictor_with_tta5_and_box():
    image_bytes = b"single-image"
    boxes = [(10, 20, 30, 40)]
    expected = [emotion(1, boxes[0], "Happy", 88.0, {"happy": 0.88})]
    calls = []

    def predictor(data, mode, *, boxes):
        calls.append((data, mode, boxes))
        return expected, "test-digest"

    results, error = predict_batch_emotions(image_bytes, boxes, predictor, TTA5)

    assert error is None
    assert results == expected
    assert calls == [(image_bytes, TTA5, boxes)]


def test_multiple_face_results_stay_aligned_and_age_gender_are_unchanged():
    boxes = [(10, 20, 30, 40), (50, 60, 70, 80)]
    predictions = [prediction(21, "Female"), prediction(34, "Male")]
    emotions = [
        emotion(1, boxes[0], "Happy", 88.0, {"happy": 0.88, "neutral": 0.12}),
        emotion(2, boxes[1], "Sad", 76.0, {"sad": 0.76, "angry": 0.24}),
    ]
    results, error = predict_batch_emotions(
        b"group-image", boxes,
        lambda data, mode, *, boxes: (emotions, "test-digest"), TTA5,
    )

    rows = result_rows(boxes, predictions, results)
    assert error is None
    assert [row["expression"] for row in rows] == ["Happy", "Sad"]
    assert [row["expression_confidence"] for row in rows] == [88.0, 76.0]
    assert [(row["estimated_age"], row["gender"]) for row in rows] == [
        (21, "Female"), (34, "Male")
    ]


def test_batch_exports_include_expression_label_confidence_and_top_three():
    boxes = [(5, 8, 40, 42)]
    predictions = [prediction(26, "Female")]
    emotion_result = emotion(
        1, boxes[0], "Happy", 82.5,
        {"happy": 0.825, "neutral": 0.10, "sad": 0.05, "angry": 0.025},
    )
    emotions, error = predict_batch_emotions(
        b"image", boxes, lambda *_args, **_kwargs: ([emotion_result], "digest"), TTA5
    )
    rows = result_rows(boxes, predictions, emotions)

    csv_rows = list(csv.DictReader(io.StringIO(csv_bytes(rows).decode("utf-8-sig"))))
    json_rows = json.loads(json_bytes(rows))
    assert error is None
    for row in (csv_rows[0], json_rows[0]):
        assert row["expression"] == "Happy"
        assert float(row["expression_confidence"]) == pytest.approx(82.5)
        assert row["estimated_age"] in ("26", 26)
        assert row["gender"] == "Female"
    assert ast.literal_eval(csv_rows[0]["expression_top3"]) == [
        ("happy", 0.825), ("neutral", 0.1), ("sad", 0.05)
    ]
    assert json_rows[0]["expression_top3"] == [
        ["happy", 0.825], ["neutral", 0.1], ["sad", 0.05]
    ]


def test_no_detected_faces_skips_emotion_predictor():
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("Predictor must not run without detected faces")

    results, error = predict_batch_emotions(b"no-faces", [], should_not_run, TTA5)
    assert results == []
    assert error is None


def test_emotion_failure_preserves_age_gender_rows_and_marks_empty_expressions():
    boxes = [(4, 6, 32, 34)]
    predictions = [prediction(45, "Male")]

    def failing_predictor(*_args, **_kwargs):
        raise RuntimeError("mock inference failure")

    emotions, error = predict_batch_emotions(b"image", boxes, failing_predictor, TTA5)
    rows = result_rows(boxes, predictions, emotions)

    assert isinstance(error, RuntimeError)
    assert rows[0]["estimated_age"] == 45
    assert rows[0]["gender"] == "Male"
    assert rows[0]["expression"] is None
    assert rows[0]["expression_confidence"] is None
    assert rows[0]["expression_top3"] == []


def test_mismatched_emotion_count_is_treated_as_inference_failure():
    boxes = [(0, 0, 20, 20), (30, 30, 20, 20)]
    emotions = [emotion(1, boxes[0], "Neutral", 50, {"neutral": 0.5})]
    results, error = predict_batch_emotions(
        b"image", boxes, lambda *_args, **_kwargs: (emotions, "digest"), TTA5
    )
    assert results is None
    assert isinstance(error, ValueError)
