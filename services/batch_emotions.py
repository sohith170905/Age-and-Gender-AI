"""Small, failure-isolated emotion inference helper for batch image exports."""


def predict_batch_emotions(image_bytes, boxes, predictor, mode):
    """Return ordered per-box emotion results, or an error without losing core results."""
    if not boxes:
        return [], None

    try:
        results, _model_digest = predictor(image_bytes, mode, boxes=boxes)
        if results is None or len(results) != len(boxes):
            raise ValueError("Emotion results did not match the detected face boxes.")
        return results, None
    except Exception as exc:
        return None, exc
