"""Validated reader for the registry stored outside immutable model assets."""
import json
from pathlib import Path
from typing import Any

REGISTRY_RELATIVE_PATH = Path("config") / "model_registry.json"


def load_registry(root: Path) -> dict[str, Any]:
    path = Path(root).resolve() / REGISTRY_RELATIVE_PATH
    with path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    models = registry.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Model registry must define at least one model.")
    for model_id, model in models.items():
        if not isinstance(model, dict) or not model.get("display_name"):
            raise ValueError(f"Model registry entry {model_id!r} is incomplete.")
        relative = Path(model.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Model path for {model_id!r} must stay inside the project.")
    return registry


def age_gender_model_id(backend: str) -> str:
    return "agevision_onnx" if backend == "clientscan" else "gra_net_compact"


def resolve_model_path(root: Path, model: dict[str, Any]) -> Path:
    return (Path(root).resolve() / model["path"]).resolve()


def short_hash(model: dict[str, Any], length: int = 12) -> str:
    return str(model.get("sha256", "Not available"))[:length]
