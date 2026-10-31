"""Adapter for AgeVision AI's existing cached YuNet detection function."""


def detect_faces(image, detector):
    """Return valid [x, y, width, height] boxes from a supplied detector callback.

    The app passes its existing detect_group_faces function so this module does
    not create a second YuNet instance or change the app's detection thresholds.
    """
    if detector is None or not callable(detector):
        raise RuntimeError("An existing face detector callback is required.")
    boxes = detector(image)
    normalized = []
    for box in boxes or []:
        if len(box) != 4:
            continue
        x, y, width, height = (int(round(float(value))) for value in box)
        if width > 0 and height > 0:
            normalized.append((x, y, width, height))
    return normalized