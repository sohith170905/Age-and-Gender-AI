"""Safe, in-memory image decoding. No image bytes are written to disk."""
from io import BytesIO
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """User-correctable image upload error."""


def decode_image(image_bytes: bytes, *, max_bytes: int, max_pixels: int) -> Image.Image:
    if not image_bytes:
        raise ImageValidationError("Choose a non-empty JPG, PNG, or WEBP image.")
    if len(image_bytes) > max_bytes:
        raise ImageValidationError(f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as source:
                if source.format not in ALLOWED_FORMATS:
                    raise ImageValidationError("Use a JPG, PNG, or WEBP image.")
                if source.width <= 0 or source.height <= 0 or source.width * source.height > max_pixels:
                    raise ImageValidationError("Image dimensions are too large to process safely.")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.info.clear()
                return image
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("This file is not a valid supported image.") from exc
