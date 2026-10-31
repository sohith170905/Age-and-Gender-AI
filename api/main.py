"""Optional authenticated REST API. Streamlit remains independent of this service."""
from collections import defaultdict, deque
from hashlib import sha256
import hmac
import os
import threading
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException
from email.parser import BytesParser
from email.policy import default as email_policy
from email.message import Message

from core.config import load_config
from core.model_registry import load_registry
from inference.api_engine import analyze

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
_CONFIG = load_config(_ROOT)
_REGISTRY = load_registry(_ROOT)
_MAX_BYTES = _CONFIG.max_upload_bytes
_RATE_LIMIT = _CONFIG.api_rate_limit_per_minute
_WINDOW = 60.0
_REQUESTS = defaultdict(deque)
_LOCK = threading.Lock()
_BEARER = HTTPBearer(auto_error=False)

app = FastAPI(title=_CONFIG.app_name + " API", version=_CONFIG.app_version, docs_url="/docs", redoc_url="/redoc")


class ErrorBody(BaseModel):
    code: str
    message: str


def error(code: str, message: str, status: int):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_request: Request, exc: StarletteHTTPException):
    code = {401: "unauthorized", 403: "forbidden", 413: "image_too_large", 429: "rate_limited"}.get(exc.status_code, "request_error")
    return error(code, str(exc.detail), exc.status_code)


async def require_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER)):
    expected = os.getenv("AGE_GENDER_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="API key authentication is not configured.")
    if credentials is None or credentials.scheme.lower() != "bearer" or not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=401, detail="A valid bearer API key is required.")
    digest = sha256(credentials.credentials.encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _LOCK:
        events = _REQUESTS[digest]
        while events and now - events[0] >= _WINDOW:
            events.popleft()
        if len(events) >= _RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Request rate limit exceeded. Try again shortly.")
        events.append(now)
    return digest


@app.get("/health")
def health():
    return {"status": "ok", "service": _CONFIG.app_name, "version": _CONFIG.app_version}


@app.get("/api/v1/models")
def models(_key: str = Depends(require_api_key)):
    return {"models": [
        {"id": model_id, "name": value["display_name"], "version": value["version"], "task": value["task"], "input_size": value["input_size"]}
        for model_id, value in _REGISTRY["models"].items()
    ]}


@app.post("/api/v1/analyze")
async def analyze_image(request: Request, include_emotion: bool = False, _key: str = Depends(require_api_key)):
    # Read the multipart request into memory ourselves: Starlette UploadFile uses
    # a spooled temporary file, which would violate the no-image-to-disk rule.
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_BYTES + 128_000:
                return error("image_too_large", "Image exceeds the configured upload limit.", 413)
        except ValueError:
            return error("invalid_request", "Invalid Content-Length header.", 400)
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_BYTES + 128_000:
            return error("image_too_large", "Image exceeds the configured upload limit.", 413)
        chunks.append(chunk)
    body = b"".join(chunks)
    content_type = request.headers.get("content-type", "").lower()
    payload = b""
    if content_type.startswith("multipart/form-data"):
        header = Message()
        header["content-type"] = request.headers.get("content-type", "")
        boundary = header.get_param("boundary", header="content-type")
        if not boundary:
            return error("invalid_request", "Multipart boundary is missing.", 400)
        envelope = (f"Content-Type: {request.headers.get('content-type')}\r\nMIME-Version: 1.0\r\n\r\n").encode("ascii") + body
        try:
            parsed = BytesParser(policy=email_policy).parsebytes(envelope)
            for part in parsed.iter_parts():
                disposition = part.get_content_disposition()
                name = part.get_param("name", header="content-disposition")
                if disposition == "form-data" and name == "image":
                    payload = part.get_payload(decode=True) or b""
                elif disposition == "form-data" and name == "include_emotion":
                    include_value = part.get_payload(decode=True) or b""
                    include_emotion = include_value.decode("utf-8", "ignore").strip().lower() in {"true", "1", "yes"}
        except Exception:
            return error("invalid_request", "Could not parse the multipart image request.", 400)
    elif content_type in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
        payload = body
    else:
        return error("unsupported_media_type", "Send a multipart image field or a JPG, PNG, or WEBP request body.", 415)
    if not payload:
        return error("missing_image", "Include a non-empty image field.", 422)
    if len(payload) > _MAX_BYTES:
        return error("image_too_large", "Image exceeds the configured upload limit.", 413)
    try:
        result = analyze(payload, include_emotion=include_emotion)
    except ValueError as exc:
        return error("invalid_image", str(exc), 422)
    except RuntimeError:
        return error("model_unavailable", "The selected local model is not available.", 503)
    except Exception:
        return error("inference_failed", "Local inference failed. Check the server model configuration.", 500)
    return result
