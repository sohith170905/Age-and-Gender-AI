"""Exact source-defined Standard and TTA-5 image views for AG-44B."""

from __future__ import annotations

import numpy as np
from PIL import Image

STANDARD = "standard"
TTA5 = "tta5"


def build_input_batch(gray_face: np.ndarray, mode: str = TTA5) -> np.ndarray:
    """Build NHWC float32 views using the original FER-ACC-01 transforms."""
    gray = np.asarray(gray_face, dtype=np.uint8)
    if gray.shape != (48, 48):
        raise ValueError(f"Expected a 48x48 grayscale face, got {gray.shape}.")

    original = gray.astype(np.float32) / 255.0
    if mode == STANDARD:
        return np.ascontiguousarray(original[None, ..., None], dtype=np.float32)
    if mode != TTA5:
        raise ValueError("Emotion mode must be 'standard' or 'tta5'.")

    # T1: NumPy horizontal flip along image width.
    flipped = np.flip(original, axis=1)
    source_pil = Image.fromarray(gray, mode="L")

    # T2/T3: PIL rotate defaults (same center, no expansion, black fill),
    # using bilinear resampling as in evaluate_tta_exp5a.py.
    resampling = Image.Resampling.BILINEAR
    rot_minus_4 = np.asarray(source_pil.rotate(-4, resample=resampling), dtype=np.float32) / 255.0
    rot_plus_4 = np.asarray(source_pil.rotate(4, resample=resampling), dtype=np.float32) / 255.0

    # T4: exact source crop box and bilinear resize.
    zoom = source_pil.crop((1, 1, 47, 47)).resize((48, 48), resample=resampling)
    zoom_array = np.asarray(zoom, dtype=np.float32) / 255.0

    views = [original, flipped, rot_minus_4, rot_plus_4, zoom_array]
    return np.ascontiguousarray(np.stack(views, axis=0)[..., None], dtype=np.float32)