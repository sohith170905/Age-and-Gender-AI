"""AG-44B tight-crop preprocessing, kept separate from age/gender crops."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def preprocess_face(image: Image.Image, bbox):
    """Crop with zero padding, BGR-to-gray, INTER_AREA resize, then /255."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Invalid image supplied to emotion preprocessing.")

    x, y, width, height = (int(round(float(value))) for value in bbox)
    image_height, image_width = rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_width, x + width), min(image_height, y + height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Emotion face crop is empty or outside the image.")

    # The source uses BGR face crops, OpenCV grayscale conversion, INTER_AREA,
    # and float32 normalization to [0, 1].
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=-1)
    return resized, tensor