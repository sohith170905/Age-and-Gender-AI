"""Quality hints wrap inference and never alter model input."""
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityResult:
    score: float
    flags: tuple[str, ...]
    uncertain: bool


def image_quality(image: Image.Image, *, dark_threshold: float = 38.0, bright_threshold: float = 225.0) -> tuple[float, tuple[str, ...], float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luminance = float(gray.mean())
    flags = []
    if luminance < dark_threshold:
        flags.append("The image is very dark; improve the lighting.")
    elif luminance > bright_threshold:
        flags.append("The image is very bright; reduce glare or backlighting.")
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return luminance, tuple(flags), blur_variance


def face_quality(image: Image.Image, bbox, *, min_blur_variance: float = 18.0, min_side: int = 48) -> QualityResult:
    x, y, width, height = (int(round(float(value))) for value in bbox)
    image_width, image_height = image.size
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_width, x + width), min(image_height, y + height)
    flags = []
    if x2 <= x1 or y2 <= y1:
        return QualityResult(0.0, ("Face crop is outside the image.",), True)
    if x1 <= 1 or y1 <= 1 or x2 >= image_width - 1 or y2 >= image_height - 1:
        flags.append("Face is close to the image edge and may be partly cut off.")
    if width < min_side or height < min_side:
        flags.append("Face is small; move closer for a more reliable estimate.")
    crop = np.asarray(image.convert("RGB"), dtype=np.uint8)[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < min_blur_variance:
        flags.append("Face appears blurry; hold the camera steady or use a sharper image.")
    score = max(0.0, 100.0 - 25.0 * len(flags))
    return QualityResult(score, tuple(flags), bool(flags))


def prediction_uncertain(prediction: dict, quality: QualityResult, *, gender_threshold: float = 55.0, age_threshold: float = 10.0) -> bool:
    gender_conf = float(prediction.get("gender_confidence", 0.0))
    age_conf = prediction.get("age_confidence")
    low_age_conf = age_conf is not None and float(age_conf) < age_threshold
    return quality.uncertain or gender_conf < gender_threshold or low_age_conf
