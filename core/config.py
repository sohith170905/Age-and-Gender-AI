"""Environment-backed application settings with project-relative defaults."""
from dataclasses import dataclass
import os
from pathlib import Path


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AppConfig:
    root: Path
    app_name: str
    app_version: str
    max_upload_bytes: int
    max_image_pixels: int
    max_faces: int
    max_batch_images: int
    gender_uncertain_below: float
    age_confidence_uncertain_below: float
    blur_variance_threshold: float
    dark_luminance_threshold: float
    bright_luminance_threshold: float
    retention_days: int
    database_url: str
    default_age_gender_model: str
    api_key: str
    api_rate_limit_per_minute: int
    camera_frame_interval: int
    camera_max_width: int
    camera_max_height: int
    camera_stun_url: str


def load_config(root: Path | None = None) -> AppConfig:
    project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    return AppConfig(
        root=project_root,
        app_name=os.getenv("AGE_GENDER_APP_NAME", "Age & Gender AI"),
        app_version=os.getenv("AGE_GENDER_APP_VERSION", "2.0.0"),
        max_upload_bytes=_int_env("AGE_GENDER_MAX_UPLOAD_BYTES", 16 * 1024 * 1024),
        max_image_pixels=_int_env("AGE_GENDER_MAX_IMAGE_PIXELS", 40_000_000),
        max_faces=_int_env("AGE_GENDER_MAX_FACES", 80),
        max_batch_images=_int_env("AGE_GENDER_MAX_BATCH_IMAGES", 5),
        gender_uncertain_below=_float_env("AGE_GENDER_UNCERTAIN_CONFIDENCE", 55.0),
        age_confidence_uncertain_below=_float_env("AGE_CONFIDENCE_UNCERTAIN_BELOW", 10.0),
        blur_variance_threshold=_float_env("AGE_GENDER_BLUR_THRESHOLD", 18.0),
        dark_luminance_threshold=_float_env("AGE_GENDER_DARK_THRESHOLD", 38.0),
        bright_luminance_threshold=_float_env("AGE_GENDER_BRIGHT_THRESHOLD", 225.0),
        retention_days=_int_env("AGE_GENDER_RETENTION_DAYS", 0, minimum=0),
        database_url=os.getenv("DATABASE_URL", "").strip(),
        default_age_gender_model=os.getenv("AGE_GENDER_MODEL", "").strip(),
        api_key=os.getenv("AGE_GENDER_API_KEY", "").strip(),
        api_rate_limit_per_minute=_int_env("AGE_GENDER_API_RATE_LIMIT", 60),
        camera_frame_interval=_int_env("AGE_GENDER_CAMERA_FRAME_INTERVAL", 5),
        camera_max_width=_int_env("AGE_GENDER_CAMERA_MAX_WIDTH", 640),
        camera_max_height=_int_env("AGE_GENDER_CAMERA_MAX_HEIGHT", 480),
        camera_stun_url=os.getenv("AGE_GENDER_CAMERA_STUN_URL", "").strip(),
    )
