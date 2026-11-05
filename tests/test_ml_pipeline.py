"""Synthetic-only regression checks. These tests never read project datasets."""
from pathlib import Path
import csv
import io
import json
import hashlib

import numpy as np
import pytest
from PIL import Image

from core.model_registry import load_registry, resolve_model_path
from services.analysis import analyze_image_bytes
from services.exports import csv_bytes, json_bytes, pdf_report
from services.quality import face_quality, image_quality, prediction_uncertain
from services.tracking import TemporalFaceTracker
from services.uploads import ImageValidationError, decode_image

ROOT = Path(__file__).resolve().parents[1]


def synthetic_png(size=(128, 128), color=(120, 130, 140)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_registry_paths_and_hashes_match_immutable_assets():
    registry = load_registry(ROOT)
    for model_id, model in registry["models"].items():
        path = resolve_model_path(ROOT, model)
        assert path.is_file(), model_id
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        assert hasher.hexdigest().upper() == model["sha256"], model_id


def test_decode_image_accepts_synthetic_png_and_removes_metadata():
    image = decode_image(synthetic_png(), max_bytes=1024 * 1024, max_pixels=100_000)
    assert image.mode == "RGB"
    assert image.size == (128, 128)
    assert image.info == {}


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_decode_image_rejects_invalid_payload(payload):
    with pytest.raises(ImageValidationError):
        decode_image(payload, max_bytes=1024, max_pixels=100_000)


def test_decode_image_rejects_byte_and_pixel_overages():
    payload = synthetic_png((400, 300))
    with pytest.raises(ImageValidationError):
        decode_image(payload, max_bytes=32, max_pixels=1_000_000)
    with pytest.raises(ImageValidationError):
        decode_image(payload, max_bytes=1_000_000, max_pixels=100)


def test_quality_hints_warn_without_mutating_input():
    image = Image.new("RGB", (96, 96), (8, 8, 8))
    original = np.asarray(image).copy()
    _, flags, _ = image_quality(image)
    quality = face_quality(image, (0, 0, 32, 32))
    assert flags
    assert quality.uncertain
    assert prediction_uncertain({"gender_confidence": 99, "age_confidence": 99}, quality)
    np.testing.assert_array_equal(np.asarray(image), original)


def test_analysis_orchestrator_caps_faces_and_uses_callbacks_in_memory():
    calls = []
    image, results, truncated = analyze_image_bytes(
        synthetic_png(), "test", decode=lambda data: decode_image(data, max_bytes=10000, max_pixels=100000),
        detect=lambda im: [(1, 2, 30, 30), (40, 5, 30, 30)],
        crop=lambda im, box, padding: (im, box),
        predict=lambda item, **kwargs: calls.append(item[1]) or {"age": 22}, max_faces=1,
    )
    assert image.size == (128, 128)
    assert truncated is True
    assert len(results) == 1
    assert calls == [(1, 2, 30, 30)]


def test_exports_handle_mixed_success_error_rows():
    rows = [{"face": 1, "age": 22}, {"face": None, "error": "no face"}]
    csv_data = csv_bytes(rows)
    parsed = list(csv.DictReader(io.StringIO(csv_data.decode("utf-8-sig"))))
    assert len(parsed) == 2
    assert "error" in parsed[0] and "age" in parsed[1]
    assert json.loads(json_bytes(rows)) == rows
    assert pdf_report(rows).startswith(b"%PDF-")


def test_tracker_keeps_prediction_box_pairs_and_smooths_age():
    tracker = TemporalFaceTracker()
    box = (10, 10, 40, 40)
    first = tracker.update([box], [{"age": 20, "gender": "Female"}])
    second = tracker.update([(11, 10, 40, 40)], [{"age": 40, "gender": "Male"}])
    third = tracker.update([(12, 10, 40, 40)], [{"age": 20, "gender": "Female"}])
    assert first[0]["age"] == 20
    assert second[0]["age"] == 27
    assert second[0]["gender"] == "Male"  # deterministic latest vote on a tie
    assert third[0]["gender"] == "Female"  # majority wins once available


def test_gra_net_synthetic_baseline_regression():
    import tensorflow as tf
    from gra_net import _GatedBlend
    with np.load(ROOT / "tests/baseline/gra.npz") as baseline:
        model = tf.keras.models.load_model(ROOT / "models/gra_net_age_tuned.keras", compile=False, custom_objects={"_GatedBlend": _GatedBlend})
        outputs = model.predict(baseline["input"], verbose=0)
        for index, output in enumerate(outputs):
            np.testing.assert_allclose(output, baseline[f"output_{index:02d}"], atol=1e-5, rtol=1e-5)


def test_agevision_onnx_synthetic_baseline_regression():
    import onnxruntime as ort
    with np.load(ROOT / "tests/baseline/agevision.npz") as baseline:
        session = ort.InferenceSession(str(ROOT / "models/faceage_dino_fp32.onnx"), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: baseline["input"]})
        for index, output in enumerate(outputs):
            np.testing.assert_allclose(output, baseline[f"output_{index:02d}"], atol=1e-5, rtol=1e-5)


def test_expression_synthetic_baseline_regression_and_checkpoint_hash():
    from emotion.model_loader import load_emotion_assets
    with np.load(ROOT / "tests/baseline/emotion.npz") as baseline:
        model, classes, digest = load_emotion_assets()
        output = model.predict(baseline["input"], verbose=0)
        np.testing.assert_allclose(output, baseline["output_00"], atol=1e-5, rtol=1e-5)
        assert classes == ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
        assert digest == "D1EF71D7A88D3C010ACAEBB175A8EA88EF04A9250BFDEE0908FE47D6ED58FE0B"


def test_api_health_auth_and_invalid_image_are_safe(monkeypatch):
    import asyncio
    import httpx
    import api.main as api_module
    app = api_module.app
    monkeypatch.setenv("AGE_GENDER_API_KEY", "synthetic-test-key")

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/api/v1/models")).status_code == 401
            headers = {"Authorization": "Bearer synthetic-test-key"}
            assert (await client.get("/api/v1/models", headers=headers)).status_code == 200
            response = await client.post("/api/v1/analyze", headers=headers, files={"image": ("fake.png", b"not an image", "image/png")})
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "invalid_image"
            blank = synthetic_png(color=(255, 255, 255))
            monkeypatch.setattr(api_module, "analyze", lambda _payload, include_emotion=False: {"model": "synthetic", "face_count": 0, "truncated": False, "faces": [], "expression_model": None, "expression_sha256": None})
            no_face = await client.post("/api/v1/analyze", headers=headers, files={"image": ("blank.png", blank, "image/png")})
            assert no_face.status_code == 200
            assert no_face.json()["face_count"] == 0
    asyncio.run(exercise())


def test_streamlit_guest_navigation_smoke(monkeypatch):
    from streamlit.testing.v1 import AppTest
    database = ROOT / "tmp" / "synthetic_app_smoke.db"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + database.as_posix())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any(button.label == "Continue as Guest" for button in at.button)
    assert any(button.label == "Create Account" for button in at.button)
    assert any(button.label == "Sign In" for button in at.button)
    next(button for button in at.button if button.label == "Continue as Guest").click().run()
    assert at.session_state["guest_mode"] is True
    assert at.session_state["navigation_page"] == "🏠 Home"
    assert not any("Sign In" in button.label or "Log Out" in button.label for button in at.button)
    assert not any(el.value == "Privacy and responsible use" for el in at.title)
    assert not at.exception
    assert at.session_state["theme_preference"] == "Dark"
    at.session_state["theme_preference"] = "Light"
    at.run()
    nav = next(radio for radio in at.radio if radio.label == "Navigation")
    for page in nav.options:
        nav.set_value(page).run()
        assert at.session_state["navigation_page"] == page
        assert not at.exception, f"Light theme / {page}"
        nav = next(radio for radio in at.radio if radio.label == "Navigation")
        subpage_label = {"🔍 Analyze": "Analyze", "🕘 History": "History", "⚙️ Settings": "Settings", "ℹ️ About AI": "About AI"}.get(page)
        if subpage_label:
            subpage = next(radio for radio in at.radio if radio.label == subpage_label)
            for option in subpage.options:
                subpage.set_value(option).run()
                assert not at.exception, f"{page} / {option}"
                nav = next(radio for radio in at.radio if radio.label == "Navigation")
                nav.set_value(page).run()
                subpage = next(radio for radio in at.radio if radio.label == subpage_label)
    at.session_state["theme_preference"] = "Dark"
    at.run()
    assert not at.exception
    nav = next(radio for radio in at.radio if radio.label == "Navigation")
    for page in nav.options:
        nav.set_value(page).run()
        assert not at.exception, f"Dark theme / {page}"
        nav = next(radio for radio in at.radio if radio.label == "Navigation")


def test_streamlit_account_registration_and_signin_flow(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    database = tmp_path / "account-flow.db"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + database.as_posix())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(button for button in at.button if button.label == "Create Account").click().run()
    next(widget for widget in at.text_input if widget.label == "Full Name").set_value("Taylor Example")
    next(widget for widget in at.text_input if widget.label == "Email Address").set_value("taylor@example.com")
    next(widget for widget in at.text_input if widget.label.startswith("Password (")).set_value("test-password-94")
    next(widget for widget in at.text_input if widget.label == "Confirm Password").set_value("test-password-94")
    next(button for button in at.button if button.label == "Create Account").click().run()
    assert not at.exception
    assert any("Account created" in element.value for element in at.success)

    next(widget for widget in at.text_input if widget.label == "Email Address").set_value("TAYLOR@example.com")
    next(widget for widget in at.text_input if widget.label == "Password").set_value("test-password-94")
    next(button for button in at.button if button.label == "Sign In").click().run()
    assert at.session_state["authenticated_user_email"] == "taylor@example.com"
    assert not at.session_state.get("guest_mode", False)
    assert not at.exception


def test_user_scoped_history_consent_and_deletion_are_isolated():
    from sqlalchemy import create_engine, text
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE prediction_history(user_id TEXT, item TEXT)"))
        connection.execute(text("CREATE TABLE user_consent(user_id TEXT PRIMARY KEY, version TEXT, accepted_at TEXT)"))
        connection.execute(text("CREATE TABLE user_preferences(user_id TEXT PRIMARY KEY, retention_days INTEGER)"))
        connection.execute(text("INSERT INTO prediction_history VALUES ('user-a','a1'),('user-b','b1')"))
        connection.execute(text("INSERT INTO user_consent VALUES ('user-a','v1','2026-10-31'),('user-b','v1','2026-10-31')"))
        connection.execute(text("INSERT INTO user_preferences VALUES ('user-a',30),('user-b',90)"))
    with engine.connect() as connection:
        assert [row[0] for row in connection.execute(text("SELECT item FROM prediction_history WHERE user_id=:id"), {"id": "user-a"})] == ["a1"]
        assert connection.execute(text("SELECT retention_days FROM user_preferences WHERE user_id=:id"), {"id": "user-b"}).scalar_one() == 90
    with engine.begin() as connection:
        for table in ("prediction_history", "user_consent", "user_preferences"):
            connection.execute(text(f"DELETE FROM {table} WHERE user_id=:id"), {"id": "user-a"})
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM prediction_history WHERE user_id='user-a'")).scalar_one() == 0
        assert connection.execute(text("SELECT COUNT(*) FROM prediction_history WHERE user_id='user-b'")).scalar_one() == 1
        assert connection.execute(text("SELECT COUNT(*) FROM user_consent WHERE user_id='user-b'")).scalar_one() == 1
    engine.dispose()


def test_api_rate_limit_is_per_api_key(monkeypatch):
    import asyncio
    import httpx
    import api.main as api_module
    monkeypatch.setenv("AGE_GENDER_API_KEY", "rate-test-key")
    monkeypatch.setattr(api_module, "_RATE_LIMIT", 2)
    api_module._REQUESTS.clear()
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_module.app), base_url="http://test") as client:
            headers = {"Authorization": "Bearer rate-test-key"}
            assert (await client.get("/api/v1/models", headers=headers)).status_code == 200
            assert (await client.get("/api/v1/models", headers=headers)).status_code == 200
            response = await client.get("/api/v1/models", headers=headers)
            assert response.status_code == 429
            assert response.json()["error"]["code"] == "rate_limited"
    asyncio.run(exercise())
    api_module._REQUESTS.clear()


def test_environment_configuration_is_bounded(monkeypatch):
    from core.config import load_config
    monkeypatch.setenv("AGE_GENDER_MAX_UPLOAD_BYTES", "4096")
    monkeypatch.setenv("AGE_GENDER_MAX_BATCH_IMAGES", "3")
    monkeypatch.setenv("AGE_GENDER_CAMERA_FRAME_INTERVAL", "2")
    config = load_config(ROOT)
    assert config.max_upload_bytes == 4096
    assert config.max_batch_images == 3
    assert config.camera_frame_interval == 2
    assert config.root == ROOT.resolve()

def test_authentication_has_no_backdoor_demo_login(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DEV_FAKE_USER", "1")
    monkeypatch.setenv("ENV", "production")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert all("demo account" not in button.label.lower() for button in at.button)
    labels = [b.label for b in at.button]
    assert "Continue as Guest" in labels
    assert "Create Account" in labels
    assert "Sign In" in labels
