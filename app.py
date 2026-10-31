# ============================================================
# CLASSIFICATION OF AGE AND GENDER FROM FACIAL IMAGES
# GRA_Net - Streamlit Application
# ============================================================

import json
import hashlib
import io
import logging
from logging.handlers import RotatingFileHandler
import os
import threading
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from emotion.model_loader import load_emotion_assets
from emotion.predictor import EmotionPredictor
from emotion.tta import TTA5 as EMOTION_TTA5
from core.config import load_config
from core.model_registry import load_registry, age_gender_model_id, resolve_model_path, short_hash
from services.uploads import decode_image, ImageValidationError
from services.quality import image_quality, face_quality, prediction_uncertain
from services.tracking import TemporalFaceTracker
from services.analysis import analyze_image_bytes as run_image_analysis
from services.exports import result_rows, csv_bytes, json_bytes, pdf_report


# ============================================================
# PAGE CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
APP_CONFIG = load_config(PROJECT_ROOT)
MODEL_REGISTRY = load_registry(PROJECT_ROOT)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOGGER = logging.getLogger("age_gender_ai")
if not LOGGER.handlers:
    handler = RotatingFileHandler(LOG_DIR / "application.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

st.set_page_config(
    page_title=APP_CONFIG.app_name,
    page_icon=str(PROJECT_ROOT / "assets" / "agevision_favicon.svg"),
    layout="wide",
    initial_sidebar_state="expanded"
)

theme_preference = st.session_state.get("theme_preference", "Light")
if theme_preference == "Dark":
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], .stApp { background: #151827; color: #f3f5ff; }
        [data-testid="stSidebar"] { background: #1d2133; }
        .result-card { background: #202438; border-color: #353b56; }
        .main-title, .section-title, h1, h2, h3 { color: #f3f5ff; }
        .subtitle { color: #c4c9df; }
        [data-testid="stSidebar"] * { color: #f3f5ff !important; }
        [data-testid="stAppViewContainer"] .stRadio label,
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
        [data-testid="stAppViewContainer"] [data-testid="stAlert"],
        [data-testid="stAppViewContainer"] [data-testid="stAlert"] * { color: #f3f5ff !important; }
        [data-testid="stAppViewContainer"] .stCaption { color: #c4c9df !important; }
        .history-table { background: #202438 !important; color: #f3f5ff !important; }
        .history-table th { background: #303858 !important; color: #ffffff !important; border-color: #59617f !important; }
        .history-table td { background: #202438 !important; color: #f3f5ff !important; border-color: #454d6d !important; }
        .section-title-image { color: #c5b4ff !important; border-left-color: #9b7aff !important; }
        .section-title-live { color: #72efe7 !important; border-left-color: #2dd4bf !important; }
        [data-testid="stDownloadButton"] > button { background: #252b45 !important; border: 1px solid #59617f !important; color: #f3f5ff !important; }
        [data-testid="stDownloadButton"] > button:hover { background: #303858 !important; border-color: #8e79f5 !important; color: #ffffff !important; }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: #f3f5ff !important; }
        [data-testid="stHeader"], [data-testid="stToolbar"] { background: #151827 !important; }
        [data-testid="stHeader"] *, [data-testid="stToolbar"] * { color: #f3f5ff !important; }
        [data-testid="stAppViewContainer"] div[data-testid="stButton"] > button,
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: #252b45 !important;
            border: 1px solid #59617f !important;
            color: #f3f5ff !important;
        }
        [data-testid="stAppViewContainer"] div[data-testid="stButton"] > button:hover,
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: #303858 !important;
            border-color: #8e79f5 !important;
            color: #ffffff !important;
        }
        [data-testid="stAppViewContainer"] div[data-testid="stButton"] > button[kind="primary"],
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(100deg, #6729e7, #4f54f4) !important;
            border: 0 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Public access is protected by Google OpenID Connect. Do not initialize the
# model or expose app data until a verified account is present.
try:
    auth_settings = dict(st.secrets.get("auth", {}))
except Exception:
    auth_settings = {}

required_auth_settings = (
    "redirect_uri",
    "cookie_secret",
    "client_id",
    "client_secret",
    "server_metadata_url",
)
GOOGLE_AUTH_CONFIGURED = all(auth_settings.get(key) for key in required_auth_settings)
IS_LOGGED_IN = bool(st.user.is_logged_in)
IS_GUEST = not IS_LOGGED_IN and bool(st.session_state.get("guest_mode", False))

if not IS_LOGGED_IN and not IS_GUEST:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 8% 6%, #f5d7ff 0, transparent 28%),
                        radial-gradient(circle at 92% 92%, #c5d6ff 0, transparent 30%),
                        #eef3ff;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"], section[data-testid="stSidebar"] {
            display: none !important;
        }
        .main .block-container {
            max-width: 1160px; min-height: 100vh; padding: 16px 20px;
            display: flex; flex-direction: column; justify-content: center;
        }
        div[data-testid="stHorizontalBlock"]:has(.login-hero) {
            gap: 0 !important; min-height: min(620px, calc(100vh - 32px));
            background: #fff; border: 1px solid rgba(255,255,255,.9);
            border-radius: 24px; overflow: hidden;
            box-shadow: 0 22px 60px rgba(49,66,120,.16);
        }
        div[data-testid="stHorizontalBlock"]:has(.login-hero) > div[data-testid="stColumn"] {
            padding: clamp(18px, 2.2vw, 32px); display: flex;
            flex-direction: column; justify-content: center;
        }
        div[data-testid="stHorizontalBlock"]:has(.login-hero) > div[data-testid="stColumn"]:first-child {
            background: linear-gradient(145deg, #f6f4ff 0%, #e8edff 58%, #d8f1ff 100%);
        }
        .login-hero { color: #151b4b; }
        .login-brand { display:flex; align-items:center; gap:14px; color:#121943; font-size:19px; font-weight:800; }
        .login-brand svg { width:44px; height:44px; flex:0 0 44px; }
        .hero-title { margin:24px 0 10px; font-size:clamp(34px,3vw,50px); line-height:1.02; letter-spacing:-1.8px; font-weight:850; color:#111844; }
        .hero-title span { background:linear-gradient(90deg,#7329e8,#2677ff); -webkit-background-clip:text; background-clip:text; color:transparent; }
        .hero-copy { max-width:420px; color:#646b8b; font-size:16px; line-height:1.5; }
        .hero-art { position:relative; height:190px; max-width:400px; margin:6px auto 0; width:100%; border-radius:22px; background:radial-gradient(circle at 52% 46%,rgba(149,132,255,.30),transparent 58%); }
        .hero-art svg { height:100%; width:100%; overflow:visible; }
        .face-stat { position:absolute; background:rgba(255,255,255,.88); color:#141b4c; border:1px solid rgba(112,131,220,.18); border-radius:14px; padding:12px 16px; box-shadow:0 12px 28px rgba(53,74,150,.14); font-size:13px; }
        .face-stat b { display:block; color:#2768ed; font-size:21px; margin-top:3px; }
        .face-stat.gender { right:0; top:36%; }
        .face-stat.age { left:0; bottom:17%; }
        .hero-steps { margin-top:12px; padding-top:14px; border-top:1px solid rgba(80,95,160,.18); color:#171d50; font-size:16px; font-weight:800; text-align:center; }
        .hero-steps span { color:#8c69ed; padding:0 9px; }
        .hero-foot { text-align:center; color:#707895; font-size:13px; margin-top:10px; }
        .login-panel { max-width:480px; width:100%; margin:auto; color:#141a47; }
        .welcome-icon { text-align:center; margin-bottom:12px; }
        .welcome-icon svg { width:52px; height:52px; }
        .welcome-title { text-align:center; color:#151a49; font-size:32px; font-weight:850; letter-spacing:-.8px; }
        .welcome-copy { text-align:center; color:#7a819c; font-size:14px; margin:4px 0 24px; }
        .login-label { color:#242a54; font-size:13px; font-weight:700; margin:16px 0 6px; }
        .login-helper { color:#7b829c; font-size:13px; line-height:1.5; margin-top:10px; text-align:center; }
        .login-note { color:#737b98; font-size:13px; text-align:center; margin-top:18px; }
        div[data-testid="stButton"] > button { min-height:46px; border-radius:10px; font-weight:700; border:1px solid #d8ddf0; transition:transform .15s ease, box-shadow .15s ease; }
        div[data-testid="stButton"] > button:hover { transform:translateY(-1px); box-shadow:0 8px 22px rgba(91,70,190,.16); border-color:#7651ec; color:#4b2dcc; }
        div[data-testid="stButton"] > button[kind="primary"] { background:linear-gradient(100deg,#6729e7,#4f54f4); border:0; color:white; }
        .or-divider { display:flex; align-items:center; gap:12px; color:#8a90a8; margin:16px 0 10px; font-size:12px; }
        .or-divider:before, .or-divider:after { content:""; height:1px; flex:1; background:#e3e6f1; }
        @media (max-width:850px) {
            .main .block-container { padding:12px; }
            div[data-testid="stHorizontalBlock"]:has(.login-hero) { min-height:auto; }
            div[data-testid="stHorizontalBlock"]:has(.login-hero) > div[data-testid="stColumn"] { padding:28px; }
            .hero-title { margin-top:28px; font-size:44px; }
            .hero-art { height:230px; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    left, right = st.columns([1.05, 0.95], gap=None)
    with left:
        st.markdown(
            """
            <div class="login-hero">
              <div class="login-brand">
                <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
                  <path d="M10 23V10h13M41 10h13v13M54 41v13H41M23 54H10V41" stroke="#6428e8" stroke-width="4" stroke-linecap="round"/>
                  <path d="M21 30c0-9 5-15 12-15s12 6 12 15v5c0 9-5 15-12 15s-12-6-12-15v-5Z" fill="#E7E9FF" stroke="#22265A" stroke-width="2.5"/>
                  <path d="M25 30c2-1 4-1 6 0m8 0c-2-1-4-1-6 0M29 39c2 2 4 2 6 0" stroke="#22265A" stroke-width="2" stroke-linecap="round"/>
                  <path d="M18 29c1-13 7-20 16-20 8 0 13 6 14 15-4-1-7-4-9-8-4 5-11 8-21 8" fill="#D5D8FF" stroke="#22265A" stroke-width="2.5" stroke-linejoin="round"/>
                </svg>
                <span>Age &amp; Gender<br>Classifier</span>
              </div>
              <div class="hero-title">AI that<br><span>Understands<br>Faces</span></div>
              <div class="hero-copy">A deep learning model for exploring face detection and age and gender estimates from images.</div>
              <div class="hero-art">
                <svg viewBox="0 0 460 300" fill="none" aria-hidden="true">
                  <circle cx="230" cy="150" r="116" fill="url(#orb)"/>
                  <path d="M101 248c16-68 62-100 129-100s113 32 129 100" fill="#D3E3FF"/>
                  <path d="M164 104c0-47 26-72 66-72s66 25 66 72v53c0 49-27 80-66 80s-66-31-66-80v-53Z" fill="#F4CBBE" stroke="#28336C" stroke-width="4"/>
                  <path d="M158 105c4-58 29-88 74-88 45 0 72 30 76 84-23-2-45-15-58-33-21 22-51 34-92 37Z" fill="#20254F"/>
                  <path d="M187 113c10-7 22-7 32 0m28 0c10-7 22-7 32 0" stroke="#252A60" stroke-width="5" stroke-linecap="round"/>
                  <circle cx="204" cy="126" r="5" fill="#252A60"/><circle cx="263" cy="126" r="5" fill="#252A60"/>
                  <path d="m233 127-8 34 13 2m-26 26c14 12 31 12 45 0" stroke="#A95F69" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M182 103 208 223m24-104v130m25-127-27 126m-61-79h121m-111-28h100m-97 56h91" stroke="#FFFFFF" stroke-opacity=".66" stroke-width="1.5"/>
                  <path d="M143 108V77h34M317 77h34v31M351 191v31h-34M177 222h-34v-31" stroke="#5B53EE" stroke-width="5" stroke-linecap="round"/>
                  <defs><radialGradient id="orb"><stop stop-color="#B8B5FF" stop-opacity=".8"/><stop offset="1" stop-color="#B8E8FF" stop-opacity=".05"/></radialGradient></defs>
                </svg>
                <div class="face-stat gender">Gender<b>Estimate</b></div>
                <div class="face-stat age">Age<b>Analysis</b></div>
              </div>
              <div class="hero-steps">Upload <span>•</span> Analyze <span>•</span> Predict</div>
              <div class="hero-foot">AI-powered estimates for learning and research</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        st.markdown(
            """
            <div class="login-panel">
              <div class="welcome-icon">
                <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
                  <path d="M10 24V10h14M40 10h14v14M54 40v14H40M24 54H10V40" stroke="#6B32E8" stroke-width="4" stroke-linecap="round"/>
                  <circle cx="32" cy="25" r="10" fill="#E6EAFF" stroke="#25306B" stroke-width="2.5"/>
                  <path d="M15 52c2-11 8-17 17-17s15 6 17 17M27 26c2-2 4-2 6 0m-3 5v2" stroke="#25306B" stroke-width="2.5" stroke-linecap="round"/>
                </svg>
              </div>
              <div class="welcome-title">Welcome Back</div>
              <div class="welcome-copy">Sign in to your account to continue</div>
              <div class="login-label">Secure sign-in</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if GOOGLE_AUTH_CONFIGURED:
            st.button("Sign in with Google", on_click=st.login, type="primary", use_container_width=True, key="login_google")
        else:
            st.warning("Google sign-in is not configured yet. You can still try the app as a guest.")
        st.markdown('<div class="or-divider">or</div>', unsafe_allow_html=True)
        if st.button("Continue as Guest", use_container_width=True, key="login_guest"):
            st.session_state["guest_mode"] = True
            st.rerun()
        st.markdown(
            '<div class="login-note">Guest access works without an account. Guest predictions are not saved.</div>'
            '<div class="login-helper">Uploaded images and camera frames are processed for analysis and are not stored.</div>',
            unsafe_allow_html=True
        )
    st.stop()

if IS_LOGGED_IN:
    st.session_state.pop("guest_mode", None)
    CURRENT_USER_ID = str(st.user.get("sub", ""))
    if not CURRENT_USER_ID:
        st.error("Your sign-in did not provide a stable account identifier. Please sign out and try again.")
        st.stop()
else:
    CURRENT_USER_ID = None


# ============================================================
# PATHS
# ============================================================

BASE_DIR = PROJECT_ROOT
ACTIVE_REGISTRY_ID = MODEL_REGISTRY.get("default_age_gender_model", "gra_net_compact")
GRA_REGISTRY_MODEL = MODEL_REGISTRY["models"]["gra_net_compact"]
MODEL_PATH = resolve_model_path(BASE_DIR, GRA_REGISTRY_MODEL)
DATA_DIR = BASE_DIR / "data"
MAX_UPLOAD_BYTES = APP_CONFIG.max_upload_bytes
MAX_IMAGE_PIXELS = APP_CONFIG.max_image_pixels
MAX_FACES = APP_CONFIG.max_faces


def load_uploaded_image(uploaded_file):
    """Decode supported uploads in memory, stripping metadata and applying EXIF orientation."""
    try:
        return decode_image(uploaded_file.getvalue(), max_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_IMAGE_PIXELS)
    except ImageValidationError as exc:
        st.error(str(exc))
        return None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #b8b8c5;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .result-card {
        background: #20232d;
        border: 1px solid #353947;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .metric-label {
        color: #b8b8c5;
        font-size: 15px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
    }

    .face-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
    }

    .history-table-wrap { overflow-x: auto; width: 100%; }
    .history-table { width: 100%; border-collapse: collapse; color: #17203f; background: #ffffff; font-size: 13px; }
    .history-table th { background: #e8ecf8; color: #17203f; font-weight: 700; text-align: left; padding: 10px 8px; border: 1px solid #d6dced; white-space: nowrap; }
    .history-table td { background: #ffffff; color: #17203f; padding: 9px 8px; border: 1px solid #dfe3ef; white-space: nowrap; }
    .section-title-image, .section-title-live { padding-left: 12px; border-left: 6px solid; }
    .section-title-image { color: #5b32d6; border-left-color: #7b4dff; }
    .section-title-live { color: #007f83; border-left-color: #00bfa6; }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AGE GROUP
# ============================================================

def get_age_group(age):

    age = int(round(age))

    if age < 10:
        return "0-9"

    elif age < 20:
        return "10-19"

    elif age < 30:
        return "20-29"

    elif age < 40:
        return "30-39"

    elif age < 50:
        return "40-49"

    elif age < 60:
        return "50-59"

    elif age < 70:
        return "60-69"

    elif age < 80:
        return "70-79"

    elif age < 90:
        return "80-89"

    elif age < 100:
        return "90-99"

    elif age < 110:
        return "100-109"

    else:
        return "110-116"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading the AI model for your first analysis…")
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:

        import tensorflow as tf

        custom_objects = {}
        try:
            from gra_net import _GatedBlend
            custom_objects["_GatedBlend"] = _GatedBlend
        except Exception:
            pass

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects,
        )

        return model

    except Exception as e:
        LOGGER.error("gra_model_load_failed error_type=%s", type(e).__name__)
        st.error("The local age/gender model could not be loaded. Check the Model section in Settings.")
        return None



@st.cache_resource(show_spinner="Loading local AgeVision AI ONNX model on first analysis…")
def load_clientscan_model():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("AgeVision AI needs onnxruntime. Install the updated requirements.") from exc

    local_path = resolve_model_path(BASE_DIR, MODEL_REGISTRY["models"]["agevision_onnx"])
    if not local_path.is_file():
        raise RuntimeError(
            "AgeVision AI model weights are not installed locally. This app does not "
            "download gated model files; select GRA_Net or install the configured local checkpoint."
        )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    session = ort.InferenceSession(str(local_path), sess_options=options, providers=providers)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 2:
        raise RuntimeError("AgeVision AI ONNX has an unexpected input/output layout.")
    return session, inputs[0].name


# ============================================================
# FACE DETECTOR
# ============================================================
# FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        cascade_path
    )

    return detector


face_cascade = load_face_detector()


# YuNet detector used only by Group Analysis. The existing detector remains
# available to Image Analysis without changing the prediction pipeline.
@st.cache_resource
def load_group_face_detector():
    model_path = resolve_model_path(BASE_DIR, MODEL_REGISTRY["models"]["yunet"])
    if not model_path.is_file():
        return None
    try:
        return cv2.FaceDetectorYN_create(
            str(model_path), "", (320, 320), 0.60, 0.30, 5000
        )
    except (AttributeError, cv2.error):
        return None


def detect_group_faces(image):
    """Detect image faces with YuNet and conservative cleanup."""
    detector = load_group_face_detector()
    if detector is None:
        return detect_faces(image)

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        return []

    detector.setInputSize((width, height))
    try:
        _, detections = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except cv2.error:
        return []
    if detections is None:
        return []

    # Keep reasonable-confidence detections, rejecting only tiny boxes that
    # are unlikely to be real faces at this image resolution.
    candidates = []
    min_side = max(24, int(round(min(width, height) * 0.012)))
    for row in detections:
        x, y, w, h = (int(round(v)) for v in row[:4])
        score = float(row[14])
        if score < 0.60 or w < min_side or h < min_side:
            continue
        # YuNet can return low-confidence background regions as faces on
        # detailed group photos.  Small candidates must therefore have a
        # stronger score, and every candidate must have face-like landmark
        # geometry before it reaches the prediction pipeline.
        if min(w, h) < 32 and score < 0.65:
            continue
        landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        eye_right, eye_left, nose, mouth_right, mouth_left = landmarks
        eye_distance = float(np.linalg.norm(eye_left - eye_right))
        eye_mid_y = float((eye_right[1] + eye_left[1]) * 0.5)
        mouth_y = float((mouth_right[1] + mouth_left[1]) * 0.5)
        if eye_distance < 0.12 * w or eye_distance > 1.05 * w:
            continue
        if abs(float(eye_right[1] - eye_left[1])) > 0.45 * h:
            continue
        if nose[1] < eye_mid_y - 0.15 * h or nose[1] > eye_mid_y + 0.95 * h:
            continue
        if mouth_y < nose[1] - 0.25 * h or mouth_y > nose[1] + 0.95 * h:
            continue
        if w / float(h) < 0.55 or w / float(h) > 1.80:
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        w, h = x2 - x1, y2 - y1
        if w < min_side or h < min_side:
            continue
        candidates.append((score, (x1, y1, w, h)))

    # Confidence-ordered suppression removes close duplicate boxes while
    # allowing nearby, differently sized faces in group photographs.
    candidates.sort(key=lambda item: item[0], reverse=True)
    kept = []
    for score, box in candidates:
        x, y, w, h = box
        duplicate = False
        for _, prior in kept:
            px, py, pw, ph = prior
            ix = max(0, min(x + w, px + pw) - max(x, px))
            iy = max(0, min(y + h, py + ph) - max(y, py))
            intersection = ix * iy
            if not intersection:
                continue
            union = w * h + pw * ph - intersection
            smaller = min(w * h, pw * ph)
            if intersection / float(union) > 0.30 or intersection / float(smaller) > 0.55:
                duplicate = True
                break
        if not duplicate:
            kept.append((score, box))

    faces = [box for _, box in kept]
    faces.sort(key=lambda box: (box[1], box[0]))
    return faces

# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image):
    """
    Group-friendly face detection.

    Uses multiple Haar detection passes so that smaller faces
    in group photographs are not missed.
    """

    img = np.array(image.convert("RGB"))
    original_height, original_width = img.shape[:2]

    # Limit very large uploads before the three Haar passes; scale boxes back
    # to source coordinates below so prediction crops remain full resolution.
    detection_scale = min(1.0, 1600.0 / max(original_width, original_height))
    if detection_scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(original_width * detection_scale)), int(round(original_height * detection_scale))),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # --------------------------------------------------------
    # Prepare multiple versions of the image
    # --------------------------------------------------------

    gray_equalized = cv2.equalizeHist(gray)

    # Slightly enlarged image helps detect smaller faces
    scale_up = 1.5
    enlarged = cv2.resize(
        gray_equalized,
        None,
        fx=scale_up,
        fy=scale_up,
        interpolation=cv2.INTER_CUBIC
    )

    all_faces = []

    # --------------------------------------------------------
    # Detection PASS 1 - normal image
    # --------------------------------------------------------

    faces1 = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(35, 35)
    )

    if faces1 is not None:
        for x, y, w, h in faces1:
            all_faces.append((int(x), int(y), int(w), int(h)))

    # --------------------------------------------------------
    # Detection PASS 2 - equalized image
    # --------------------------------------------------------

    faces2 = face_cascade.detectMultiScale(
        gray_equalized,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if faces2 is not None:
        for x, y, w, h in faces2:
            all_faces.append((int(x), int(y), int(w), int(h)))

    # --------------------------------------------------------
    # Detection PASS 3 - enlarged image
    # --------------------------------------------------------

    faces3 = face_cascade.detectMultiScale(
        enlarged,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if faces3 is not None:
        for x, y, w, h in faces3:
            x = int(x / scale_up)
            y = int(y / scale_up)
            w = int(w / scale_up)
            h = int(h / scale_up)
            all_faces.append((x, y, w, h))

    # --------------------------------------------------------
    # If nothing detected
    # --------------------------------------------------------

    if not all_faces:
        return []

    # Restore source-image coordinates for crop extraction and annotations.
    if detection_scale < 1.0:
        inverse_scale = 1.0 / detection_scale
        all_faces = [
            (int(round(x * inverse_scale)), int(round(y * inverse_scale)),
             int(round(w * inverse_scale)), int(round(h * inverse_scale)))
            for x, y, w, h in all_faces
        ]
        width, height = original_width, original_height

    # --------------------------------------------------------
    # Basic filtering
    # --------------------------------------------------------

    filtered = []
    image_area = width * height

    for x, y, w, h in all_faces:
        if w <= 0 or h <= 0:
            continue

        # Smaller faces are allowed for group photographs
        if w < 28 or h < 28:
            continue

        ratio = w / float(h)
        if ratio < 0.55 or ratio > 1.80:
            continue

        area = w * h
        area_ratio = area / float(image_area)
        if area_ratio < 0.0008:
            continue

        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            continue

        filtered.append((x, y, w, h))

    # --------------------------------------------------------
    # Remove overlapping duplicates
    # --------------------------------------------------------

    filtered = remove_duplicate_faces(filtered, threshold=0.35)

    # --------------------------------------------------------
    # Sort faces: top-to-bottom, then left-to-right
    # --------------------------------------------------------

    filtered.sort(key=lambda b: (b[1], b[0]))

    return filtered


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left = max(
        x1,
        x2
    )

    top = max(
        y1,
        y2
    )

    right = min(
        x1 + w1,
        x2 + w2
    )

    bottom = min(
        y1 + h1,
        y2 + h2
    )

    inter_w = max(
        0,
        right - left
    )

    inter_h = max(
        0,
        bottom - top
    )

    intersection = (
        inter_w * inter_h
    )

    area1 = w1 * h1
    area2 = w2 * h2

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:
        return 0

    return (
        intersection
        / float(union)
    )


def remove_duplicate_faces(
    faces,
    threshold=0.30
):

    if len(faces) <= 1:
        return faces

    # Largest first
    faces = sorted(
        faces,
        key=lambda b: b[2] * b[3],
        reverse=True
    )

    keep = []

    for face in faces:

        duplicate = False

        for existing in keep:

            if calculate_iou(
                face,
                existing
            ) > threshold:

                duplicate = True
                break

        if not duplicate:

            keep.append(
                face
            )

    return keep


# ============================================================
# CROP FACE
# ============================================================

def crop_with_padding(
    image,
    box,
    padding=0.20
):

    x, y, w, h = box

    image_width, image_height = (
        image.size
    )

    pad_x = int(
        w * padding
    )

    pad_y = int(
        h * padding
    )

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        image_width,
        x + w + pad_x
    )

    y2 = min(
        image_height,
        y + h + pad_y
    )

    return image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# PREPROCESS FACE
# ============================================================

def preprocess_face(face_image, size=(32, 32)):

    face_image = face_image.convert(
        "RGB"
    )

    face_image = face_image.resize(size)

    arr = np.asarray(
        face_image,
        dtype=np.float32
    )

    arr = arr / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    return arr


# ============================================================
# MODEL PREDICTION
# ============================================================

GENDER_FEMALE_THRESHOLD = 0.32


def age_confidence_text(prediction):
    value = prediction.get("age_confidence")
    return "Not provided by this model" if value is None else f"{float(value):.1f}%"


def classify_gender_probabilities(probabilities):
    """Use the validation-selected threshold for the female output class."""
    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0
    if values.size < 2 or not np.all(np.isfinite(values[:2])):
        return int(np.argmax(np.nan_to_num(values)))
    # UTKFace label 0 is male and label 1 is female.
    return 1 if float(values[1]) >= GENDER_FEMALE_THRESHOLD else 0

def predict_clientscan_face(face_image, source_image=None, bbox=None):
    """Use AgeVision AI's documented 224px ImageNet and CORAL inference contract."""
    session, input_name = load_clientscan_model()
    if source_image is not None and bbox is not None:
        face_image = crop_with_padding(source_image, bbox, padding=0.10)
    rgb = np.asarray(face_image.convert("RGB"), dtype=np.uint8)
    if rgb.size == 0:
        raise ValueError("The detected face crop is empty.")
    resized = Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None, ...])
    age_logits, gender_logits = session.run(None, {input_name: tensor})
    age_logits = np.asarray(age_logits, dtype=np.float32).reshape(1, -1)
    gender_logits = np.asarray(gender_logits, dtype=np.float32).reshape(1, -1)
    if age_logits.shape[1] != 100 or gender_logits.shape[1] != 2:
        raise ValueError("AgeVision AI returned unexpected prediction dimensions.")
    age_probabilities = 1.0 / (1.0 + np.exp(-np.clip(age_logits[0], -88.0, 88.0)))
    estimated_age = float(np.clip(age_probabilities.sum(), 0.0, 100.0))

    # Convert the model's cumulative CORAL probabilities into an age distribution.
    # Report the probability mass within +/- 5 years of the displayed estimate.
    # This is model-derived confidence, not an independently calibrated score.
    cumulative_age = np.minimum.accumulate(np.clip(age_probabilities, 0.0, 1.0))
    age_distribution = np.concatenate((
        [1.0 - cumulative_age[0]],
        cumulative_age[:-1] - cumulative_age[1:],
        [cumulative_age[-1]],
    ))
    age_distribution = np.clip(age_distribution, 0.0, 1.0)
    distribution_total = float(age_distribution.sum())
    if distribution_total > 0.0:
        age_distribution /= distribution_total
    age = int(round(estimated_age))
    lower_age, upper_age = max(0, age - 5), min(100, age + 5)
    age_confidence = float(age_distribution[lower_age:upper_age + 1].sum() * 100.0)
    shifted = gender_logits[0] - float(np.max(gender_logits[0]))
    gender_probabilities = np.exp(shifted)
    gender_probabilities /= max(float(gender_probabilities.sum()), 1e-12)
    gender_index = int(np.argmax(gender_probabilities))  # 0=female, 1=male
    gender = "Male" if gender_index == 1 else "Female"
    confidence = float(gender_probabilities[gender_index] * 100.0)
    return {
        "gender": gender,
        "gender_confidence": round(confidence, 1),
        "age": age,
        "age_raw": round(estimated_age, 2),
        "age_group": get_age_group(age),
        "age_confidence": round(age_confidence, 1),
        "age_consistent": None,
        "model_backend": "AgeVision AI",
    }


def predict_face(face_image, show_errors=True, backend="gra_net", source_image=None, bbox=None):

    if backend == "clientscan":
        try:
            return predict_clientscan_face(face_image, source_image=source_image, bbox=bbox)
        except Exception as exc:
            LOGGER.error("agevision_inference_failed error_type=%s", type(exc).__name__)
            if show_errors:
                st.error("AgeVision AI could not produce a result. Check the local model status in Settings.")
                return None
            raise

    # Load TensorFlow only when a prediction is requested. This lets the login
    # and dashboard render immediately instead of blocking app startup.
    model = load_model()
    if model is None:
        return None

    if model is None:
        return None

    try:

        # ====================================================
        # PREPROCESS
        # ====================================================

        input_shape = getattr(model, "input_shape", (None, 32, 32, 3))
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        model_size = tuple(int(value) for value in input_shape[1:3])
        x = preprocess_face(face_image, size=model_size)

        outputs = model.predict(
            x,
            verbose=0
        )

        if len(outputs) < 3:
            return None

        # ====================================================
        # OUTPUT 0 = GENDER
        # ====================================================

        gender_probs = np.asarray(
            outputs[0][0],
            dtype=np.float32
        ).flatten()

        # Make sure probabilities are valid
        gender_probs = np.nan_to_num(
            gender_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        gender_index = classify_gender_probabilities(gender_probs)

        gender_confidence = (
            float(
                gender_probs[gender_index]
            ) * 100.0
        )

        # IMPORTANT:
        # Model output:
        # index 0 = Male
        # index 1 = Female

        if gender_index == 0:
            gender = "Male"
        else:
            gender = "Female"

        # Low confidence should not be presented
        # as a strong prediction.
        if gender_confidence < 55:
            gender_status = "Uncertain"
        else:
            gender_status = gender

        # ====================================================
        # OUTPUT 1 = AGE CLASSIFICATION
        # ====================================================

        age_class_probs = np.asarray(
            outputs[1][0],
            dtype=np.float32
        ).flatten()

        age_class_probs = np.nan_to_num(
            age_class_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        age_class_index = int(
            np.argmax(age_class_probs)
        )

        age_confidence = (
            float(
                age_class_probs[
                    age_class_index
                ]
            ) * 100.0
        )

        # The class represents decades:
        #
        # 0 = 0-9
        # 1 = 10-19
        # 2 = 20-29
        # 3 = 30-39
        # ...
        # 9 = 90-99

        class_low = age_class_index * 10
        class_high = (
            class_low + 9
        )

        # ====================================================
        # OUTPUT 2 = AGE REGRESSION
        # ====================================================

        regression_value = float(
            np.asarray(
                outputs[2]
            ).reshape(-1)[0]
        )

        regression_value = float(
            np.nan_to_num(
                regression_value,
                nan=25.0,
                posinf=116.0,
                neginf=0.0
            )
        )

        paper_age_head = "age_remainder" in getattr(model, "output_names", [])
        research_age_head = getattr(model, "name", "") == "ResearchAgeGenderMobileNetV3"

        if paper_age_head:
            # Paper protocol: output 2 is a normalized 0-1 remainder within
            # the selected decade, so decode it back to 0-9 before combining
            # it with the age-class prediction.
            remainder = float(np.clip(regression_value * 9.0, 0.0, 9.0))
            estimated_age = float(np.clip(class_low + remainder, 0.0, 116.0))
            regression_inside_class = True
        elif research_age_head:
            # The optional MobileNetV3 research candidate trains the direct
            # age head on a stable 0-1 target and decodes it here.
            estimated_age = float(np.clip(regression_value * 116.0, 0.0, 116.0))
            regression_inside_class = class_low <= estimated_age <= class_high
        else:
            # Existing production models expose a direct age-regression head.
            regression_value = float(np.clip(regression_value, 0, 116))
            estimated_age = regression_value
            regression_inside_class = class_low <= estimated_age <= class_high

        # ====================================================
        # AGE ESTIMATION
        # ====================================================
        #
        # Use the regression output directly.
        #
        # DO NOT do:
        #
        # decade * 10 + 9
        #
        # That was causing:
        #
        # 34 -> 29
        # 51 -> 69
        # 23 -> 29
        #
        # The regression output itself is the exact-age
        # estimate.
        # ====================================================

        # ====================================================
        # SANITY CHECK BETWEEN CLASSIFICATION AND REGRESSION
        # ====================================================
        #
        # If regression and classification strongly disagree,
        # don't silently invent an age.
        #
        # Example:
        #
        # classification = 20-29
        # regression = 28
        #
        # -> perfectly consistent
        #
        # classification = 20-29
        # regression = 67
        #
        # -> inconsistent
        #
        # In inconsistent cases we report the regression
        # result but reduce the age confidence.
        # ====================================================

        if regression_inside_class:

            final_age_confidence = age_confidence

        else:

            # The two age heads disagree.
            # Reduce confidence rather than changing
            # the actual regression prediction.

            final_age_confidence = min(
                age_confidence,
                50.0
            )

        # ====================================================
        # FINAL AGE
        # ====================================================

        final_age = int(
            round(
                estimated_age
            )
        )

        final_age = int(
            np.clip(
                final_age,
                0,
                116
            )
        )

        # ====================================================
        # AGE GROUP
        # ====================================================

        age_group = get_age_group(
            final_age
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "gender": gender,

            "gender_display":
                gender_status,

            "gender_confidence":
                round(
                    gender_confidence,
                    1
                ),

            "age":
                final_age,

            "age_raw":
                round(
                    estimated_age,
                    2
                ),

            "age_group":
                age_group,

            "age_class":
                age_class_index,

            "age_class_range":
                f"{class_low}-{class_high}",

            "age_confidence":
                round(
                    final_age_confidence,
                    1
                ),

            "age_consistent":
                regression_inside_class
        }

    except Exception as e:

        if show_errors:
            st.error(
                f"Prediction error: {e}"
            )

        return None
    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    gender_index = classify_gender_probabilities(gender_probs)

    if gender_index == 0:

        gender = "Male"

    else:

        gender = "Female"

    gender_confidence = (
        float(
            gender_probs[
                gender_index
            ]
        )
        * 100
    )

    # --------------------------------------------------------
    # AGE
    #
    # IMPORTANT:
    #
    # Use regression output DIRECTLY.
    #
    # Do NOT do:
    # decade * 10 + digit
    #
    # because the regression output is already producing
    # values such as 23, 34, 51 etc.
    # --------------------------------------------------------

    age = float(
        np.clip(
            age_regression,
            0,
            116
        )
    )

    age = round(
        age
    )

    # --------------------------------------------------------
    # Age classification confidence
    # --------------------------------------------------------

    age_class_index = int(
        np.argmax(
            age_class_probs
        )
    )

    age_confidence = (
        float(
            age_class_probs[
                age_class_index
            ]
        )
        * 100
    )

    age_group = get_age_group(
        age
    )

    return {

        "gender": gender,

        "gender_confidence":
            gender_confidence,

        "age": int(age),

        "age_group":
            age_group,

        "age_confidence":
            age_confidence
    }


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    faces,
    predictions=None
):

    output = image.copy()

    draw = ImageDraw.Draw(
        output
    )

    for i, box in enumerate(
        faces,
        start=1
    ):

        x, y, w, h = box

        # Green box
        draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h
            ],
            outline=(0, 255, 120),
            width=4
        )

        # Label
        if (
            predictions
            and i - 1 < len(predictions)
        ):

            pred = predictions[
                i - 1
            ]

            label = (
                f"Face {i} | "
                f"{pred['gender']} | "
                f"{pred['age']} yrs"
                + (f" | {pred['emotion']}" if pred.get("emotion") else "")
            )

        else:

            label = (
                f"Face {i}"
            )

        # Font
        try:

            font = ImageFont.truetype(
                "arial.ttf",
                16
            )

        except:

            font = ImageFont.load_default()

        # Text box
        bbox = draw.textbbox(
            (
                0,
                0
            ),
            label,
            font=font
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        text_height = (
            bbox[3] - bbox[1]
        )

        label_y = max(
            0,
            y - text_height - 10
        )

        draw.rectangle(
            [
                x,
                label_y,
                x + text_width + 12,
                label_y + text_height + 8
            ],
            fill=(0, 255, 120)
        )

        draw.text(
            (
                x + 6,
                label_y + 3
            ),
            label,
            fill=(0, 0, 0),
            font=font
        )

    return output


# ============================================================
# HISTORY
# ============================================================

def get_database_engine():
    """Create the local or hosted database and initialize private history."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        try:
            database_url = str(st.secrets.get("database", {}).get("url", "")).strip()
        except Exception:
            database_url = ""

    if not database_url:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"

    if database_url.startswith("postgres://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
    elif database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]

    if database_url.startswith("sqlite:"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(database_url, pool_pre_ping=True)

    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                user_id TEXT NOT NULL,
                analysis_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                face_index INTEGER NOT NULL,
                gender TEXT NOT NULL,
                gender_confidence REAL NOT NULL,
                estimated_age INTEGER NOT NULL,
                age_group TEXT NOT NULL,
                age_confidence REAL NOT NULL,
                UNIQUE (user_id, analysis_id, face_index)
            )
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS emotion_prediction_history (
                user_id TEXT NOT NULL,
                analysis_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                face_index INTEGER NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                inference_mode TEXT NOT NULL,
                model_id TEXT NOT NULL,
                UNIQUE (user_id, analysis_id, face_index)
            )
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS user_consent (
                user_id TEXT PRIMARY KEY,
                consent_version TEXT NOT NULL,
                accepted_at TEXT NOT NULL
            )
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                retention_days INTEGER NOT NULL DEFAULT 0
            )
        """))
        additions = {
            "model_id": "TEXT NOT NULL DEFAULT 'unknown'",
            "model_version": "TEXT NOT NULL DEFAULT 'unknown'",
            "source_type": "TEXT NOT NULL DEFAULT 'image'",
            "uncertain": "INTEGER NOT NULL DEFAULT 0",
            "quality_score": "REAL NOT NULL DEFAULT 100",
            "quality_flags": "TEXT NOT NULL DEFAULT '[]'",
        }
        if engine.dialect.name == "sqlite":
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(prediction_history)")).fetchall()}
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE prediction_history ADD COLUMN {name} {declaration}"))
        else:
            for name, declaration in additions.items():
                connection.execute(text(f"ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS {name} {declaration}"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_prediction_user_created ON prediction_history(user_id, created_at)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_emotion_user_created ON emotion_prediction_history(user_id, created_at)"))
    return engine


@st.cache_resource
def cached_database_engine():
    return get_database_engine()


CONSENT_VERSION = "2026-10-31-v1"

def consent_is_recorded():
    if IS_GUEST:
        return bool(st.session_state.get("guest_consent_accepted"))
    if not CURRENT_USER_ID:
        return False
    try:
        with cached_database_engine().connect() as connection:
            row = connection.execute(text("SELECT consent_version FROM user_consent WHERE user_id=:user_id"), {"user_id": CURRENT_USER_ID}).first()
        return bool(row and row[0] == CONSENT_VERSION)
    except Exception:
        return False

def record_consent():
    accepted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if IS_GUEST:
        st.session_state["guest_consent_accepted"] = True
    elif CURRENT_USER_ID:
        with cached_database_engine().begin() as connection:
            connection.execute(text("INSERT INTO user_consent(user_id, consent_version, accepted_at) VALUES(:user_id,:version,:accepted) ON CONFLICT(user_id) DO UPDATE SET consent_version=:version, accepted_at=:accepted"), {"user_id": CURRENT_USER_ID, "version": CONSENT_VERSION, "accepted": accepted_at})

def enforce_consent_gate():
    if not IS_GUEST and not IS_LOGGED_IN:
        return
    if consent_is_recorded():
        return
    st.title("Privacy and responsible use")
    st.write("Images are processed locally by this application in memory. Signed-in accounts can save prediction metadata; Guest Mode saves nothing. Uploaded images, camera frames, crops, and face embeddings are not stored.")
    st.warning("Age, gender, and visible-expression results are estimates. Do not use them to make decisions about a person. Only analyze images you have permission to use.")
    accepted = st.checkbox("I understand how image processing and saved history work, and I have permission to analyze my images.", key="consent_check")
    if st.button("Continue", type="primary", disabled=not accepted, key="consent_continue"):
        try:
            record_consent()
            st.rerun()
        except Exception:
            st.error("Consent could not be saved. Please try again.")
    st.stop()

def get_retention_days():
    if not CURRENT_USER_ID:
        return APP_CONFIG.retention_days
    try:
        with cached_database_engine().connect() as connection:
            row = connection.execute(text("SELECT retention_days FROM user_preferences WHERE user_id=:user_id"), {"user_id": CURRENT_USER_ID}).first()
        return int(row[0]) if row else APP_CONFIG.retention_days
    except Exception:
        return APP_CONFIG.retention_days

def set_retention_days(days):
    days = int(days)
    if days not in (0, 30, 90) or not CURRENT_USER_ID:
        return
    with cached_database_engine().begin() as connection:
        connection.execute(text("INSERT INTO user_preferences(user_id,retention_days) VALUES(:user_id,:days) ON CONFLICT(user_id) DO UPDATE SET retention_days=:days"), {"user_id": CURRENT_USER_ID, "days": days})
    apply_retention_cleanup()

def apply_retention_cleanup():
    days = get_retention_days()
    if days <= 0 or not CURRENT_USER_ID:
        return
    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat(timespec="seconds")
    with cached_database_engine().begin() as connection:
        connection.execute(text("DELETE FROM prediction_history WHERE user_id=:user_id AND created_at < :cutoff"), {"user_id": CURRENT_USER_ID, "cutoff": cutoff})
        connection.execute(text("DELETE FROM emotion_prediction_history WHERE user_id=:user_id AND created_at < :cutoff"), {"user_id": CURRENT_USER_ID, "cutoff": cutoff})

def delete_user_account_data():
    if not CURRENT_USER_ID:
        return
    with cached_database_engine().begin() as connection:
        for table in ("prediction_history", "emotion_prediction_history", "user_consent", "user_preferences"):
            connection.execute(text(f"DELETE FROM {table} WHERE user_id=:user_id"), {"user_id": CURRENT_USER_ID})

def delete_history_row(analysis_id, face_index):
    if IS_GUEST or not CURRENT_USER_ID:
        return
    with cached_database_engine().begin() as connection:
        connection.execute(text("DELETE FROM prediction_history WHERE user_id=:user_id AND analysis_id=:analysis_id AND face_index=:face_index"), {"user_id": CURRENT_USER_ID, "analysis_id": analysis_id, "face_index": int(face_index)})


def get_analysis_id(mode, image_data):
    """Return a per-session random ID; only the random ID is persisted, never the image digest."""
    idempotency_key = hashlib.sha256(mode.encode("utf-8") + b":" + image_data).hexdigest()
    analysis_ids = st.session_state.setdefault("_analysis_ids", {})
    analysis_id = analysis_ids.setdefault(idempotency_key, uuid.uuid4().hex)
    if len(analysis_ids) > 64:
        for old_key in list(analysis_ids)[:-32]:
            analysis_ids.pop(old_key, None)
    return analysis_id


def save_history(filename, mode, predictions, image_data=None):
    """Store prediction metadata only, scoped to the signed-in account."""
    if IS_GUEST or not predictions:
        return None

    seed = image_data or f"{mode}:{filename}:{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
    analysis_id = get_analysis_id(mode, seed)
    if len(analysis_ids) > 64:
        for old_key in list(analysis_ids)[:-32]:
            analysis_ids.pop(old_key, None)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        {
            "user_id": CURRENT_USER_ID,
            "analysis_id": analysis_id,
            "created_at": timestamp,
            "mode": mode,
            "face_index": index,
            "gender": prediction["gender"],
            "gender_confidence": round(float(prediction["gender_confidence"]), 2),
            "estimated_age": int(prediction["age"]),
            "age_group": prediction["age_group"],
            "age_confidence": (
                -1.0 if prediction.get("age_confidence") is None
                else round(float(prediction["age_confidence"]), 2)
            ),
            "model_id": age_gender_model_id(selected_backend),
            "model_version": MODEL_REGISTRY["models"][age_gender_model_id(selected_backend)]["version"],
            "source_type": str(mode),
            "uncertain": int(bool(prediction.get("uncertain", False))),
            "quality_score": float(prediction.get("quality_score", 100.0)),
            "quality_flags": json.dumps(list(prediction.get("quality_flags", ())), ensure_ascii=False),
        }
        for index, prediction in enumerate(predictions, start=1)
    ]

    try:
        with cached_database_engine().begin() as connection:
            connection.execute(text("""
                INSERT INTO prediction_history (
                    user_id, analysis_id, created_at, mode, face_index, gender,
                    gender_confidence, estimated_age, age_group, age_confidence,
                    model_id, model_version, source_type, uncertain, quality_score, quality_flags
                ) VALUES (
                    :user_id, :analysis_id, :created_at, :mode, :face_index, :gender,
                    :gender_confidence, :estimated_age, :age_group, :age_confidence,
                    :model_id, :model_version, :source_type, :uncertain, :quality_score, :quality_flags
                ) ON CONFLICT (user_id, analysis_id, face_index) DO NOTHING
            """), rows)
        if not IS_GUEST:
            apply_retention_cleanup()
        return analysis_id
    except Exception:
        LOGGER.warning("prediction_history_save_failed")
        st.warning("Your prediction is ready, but it could not be added to history. Please try again later.")
        return analysis_id


def load_user_history():
    query = text("""
        SELECT
            analysis_id AS "Analysis ID",
            created_at AS "Timestamp",
            mode AS "Mode",
            face_index AS "Face",
            gender AS "Gender",
            gender_confidence AS "Gender Confidence",
            estimated_age AS "Estimated Age",
            age_group AS "Age Group",
            age_confidence AS "Age Confidence",
            model_id AS "Model",
            model_version AS "Model Version",
            uncertain AS "Uncertain"
        FROM prediction_history
        WHERE user_id = :user_id
        ORDER BY created_at DESC, analysis_id DESC, face_index ASC
    """)
    with cached_database_engine().connect() as connection:
        history = pd.read_sql_query(query, connection, params={"user_id": CURRENT_USER_ID})
    if not history.empty:
        history.loc[history["Age Confidence"] < 0, "Age Confidence"] = "N/A"
    return history


def delete_user_history():
    with cached_database_engine().begin() as connection:
        connection.execute(
            text("DELETE FROM prediction_history WHERE user_id = :user_id"),
            {"user_id": CURRENT_USER_ID}
        )
        connection.execute(
            text("DELETE FROM emotion_prediction_history WHERE user_id = :user_id"),
            {"user_id": CURRENT_USER_ID}
        )


@st.cache_resource(show_spinner=False)
def cached_emotion_predictor():
    model, classes, digest = load_emotion_assets()
    return EmotionPredictor(model, classes, detect_group_faces), digest


def predict_emotions_from_bytes(image_bytes, mode, boxes=None):
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    predictor, digest = cached_emotion_predictor()
    normalized_boxes = None if boxes is None else tuple(tuple(int(value) for value in box) for box in boxes)
    return predictor.predict(image, mode=mode, boxes=normalized_boxes), digest


def save_emotion_history(image_bytes, results, mode, analysis_id=None):
    """Save expression metadata for signed-in users; never persist images or guests."""
    if IS_GUEST or not CURRENT_USER_ID or not results:
        return
    # A random per-session ID can join expression labels to the same age analysis.
    analysis_id = analysis_id or get_analysis_id(f"expression:{mode}", image_bytes)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        {
            "user_id": CURRENT_USER_ID,
            "analysis_id": analysis_id,
            "created_at": created_at,
            "face_index": int(result["face_id"]),
            "emotion": result["emotion"],
            "confidence": round(float(result["confidence"]), 3),
            "inference_mode": mode,
            "model_id": "expression_v1",
        }
        for result in results
    ]
    with cached_database_engine().begin() as connection:
        connection.execute(text("""
            INSERT INTO emotion_prediction_history (
                user_id, analysis_id, created_at, face_index, emotion,
                confidence, inference_mode, model_id
            ) VALUES (
                :user_id, :analysis_id, :created_at, :face_index, :emotion,
                :confidence, :inference_mode, :model_id
            ) ON CONFLICT (user_id, analysis_id, face_index) DO NOTHING
        """), rows)
    apply_retention_cleanup()


def load_emotion_history():
    if IS_GUEST or not CURRENT_USER_ID:
        return pd.DataFrame()
    query = text("""
        SELECT analysis_id AS "Analysis ID", created_at AS "Timestamp", face_index AS "Face",
               emotion AS "Emotion", confidence AS "Confidence (%)",
               inference_mode AS "Inference", model_id AS "Model"
        FROM emotion_prediction_history
        WHERE user_id = :user_id
        ORDER BY created_at DESC, analysis_id DESC, face_index ASC
    """)
    with cached_database_engine().connect() as connection:
        return pd.read_sql_query(query, connection, params={"user_id": CURRENT_USER_ID})
enforce_consent_gate()
if IS_LOGGED_IN:
    try:
        apply_retention_cleanup()
    except Exception:
        LOGGER.warning("retention_cleanup_failed")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(APP_CONFIG.app_name.upper())

st.sidebar.markdown(
    """
    **Classification of Age and Gender
    from Facial Images**
    """
)

st.sidebar.divider()

pages = [
    "🏠 Home",
    "🔍 Analyze",
    "🎭 Facial Emotion Detection",
    "📷 Live Camera",
    "🕘 History",
    "⚙️ Settings",
    "ℹ️ About AI",
]

page = st.sidebar.radio(
    "Navigation",
    pages
)

history_view = "History"
analysis_mode = "Single Person"
settings_section = "Account"
about_section = "How It Works"

if page == "🔍 Analyze":
    analysis_mode = st.sidebar.radio(
        "Analyze",
        ["Single Person", "Group Analysis"],
        key="analysis_mode",
    )
elif page == "🕘 History":
    history_view = st.sidebar.radio(
        "History",
        ["History", "Analytics / Statistics", "Emotion History"],
        key="history_section",
    )
elif page == "⚙️ Settings":
    settings_section = st.sidebar.radio(
        "Settings",
        ["Account", "Appearance", "Privacy", "Security", "Model"],
        key="settings_section",
    )
elif page == "ℹ️ About AI":
    about_section = st.sidebar.radio(
        "About AI",
        ["How It Works", "Model", "Expression Model", "Dataset & Training", "Evaluation", "Responsible AI"],
        key="about_section",
    )

st.sidebar.divider()
if st.session_state.get("prediction_backend") == "FaceAge ClientScan":
    st.session_state["prediction_backend"] = "AgeVision AI"
if "prediction_backend" not in st.session_state:
    configured_id = APP_CONFIG.default_age_gender_model or ACTIVE_REGISTRY_ID
    configured_model = MODEL_REGISTRY["models"].get(configured_id, {})
    configured_path = resolve_model_path(BASE_DIR, configured_model) if configured_model else Path()
    if configured_id == "agevision_onnx" and not configured_path.is_file():
        configured_id = "gra_net_compact"
    st.session_state["prediction_backend"] = MODEL_REGISTRY["models"][configured_id]["display_name"]
selected_backend = "clientscan" if st.session_state.get("prediction_backend") == MODEL_REGISTRY["models"]["agevision_onnx"]["display_name"] else "gra_net"
active_model = MODEL_REGISTRY["models"][age_gender_model_id(selected_backend)]
st.sidebar.caption(f"Prediction model: {active_model['display_name']}")
st.sidebar.caption(f"Version {APP_CONFIG.app_version}")
if IS_GUEST:
    st.sidebar.caption("Guest mode · predictions are not saved")
    if GOOGLE_AUTH_CONFIGURED:
        st.sidebar.button("Sign in with Google", on_click=st.login, use_container_width=True)
    if st.sidebar.button("Exit guest mode", use_container_width=True):
        st.session_state.pop("guest_mode", None)
        st.rerun()
else:
    st.sidebar.caption(
        f"Signed in: {st.user.get('email', st.user.get('name', 'Google account'))}"
    )
    st.sidebar.button("Sign out", on_click=st.logout, use_container_width=True)
st.sidebar.divider()

if selected_backend == "clientscan":
    st.sidebar.info("AgeVision AI loads a local model file during analysis. No Hugging Face token or account details are sent.")
else:
    st.sidebar.info("AI model loads when you start an analysis.")

def analyze_uploaded_image(image_bytes, backend):
    """Run the shared in-memory analysis orchestration with the existing model callbacks."""
    try:
        _, results, truncated = run_image_analysis(
            image_bytes, backend, decode=lambda data: decode_image(data, max_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_IMAGE_PIXELS),
            detect=detect_group_faces, crop=crop_with_padding,
            predict=lambda crop, backend, source_image, bbox: predict_face(crop, show_errors=False, backend=backend, source_image=source_image, bbox=bbox),
            max_faces=MAX_FACES,
        )
        return results, None, truncated
    except ImageValidationError as exc:
        return tuple(), str(exc), False
    except Exception as exc:
        LOGGER.error("image_inference_failed error_type=%s", type(exc).__name__)
        return tuple(), "The selected local model could not produce a result. Check its status in Settings.", False

# Real-time processor keeps frames only in memory and predicts every configured Nth frame.
try:
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    VideoProcessorBase = object
    webrtc_streamer = None
    av = None
    WEBRTC_AVAILABLE = False


class LiveInferenceProcessor(VideoProcessorBase):
    backend = "gra_net"
    include_expression = False

    def __init__(self):
        self.frame_number = 0
        self.last_rgb = None
        self.last_boxes = []
        self.last_predictions = []
        self.tracker = TemporalFaceTracker()
        self.last_latency_ms = 0.0
        self.last_fps = 0.0
        self.last_tick = time.perf_counter()

    def recv(self, frame):
        started = time.perf_counter()
        rgb = frame.to_ndarray(format="rgb24")
        self.frame_number += 1
        if self.frame_number % APP_CONFIG.camera_frame_interval == 0:
            height, width = rgb.shape[:2]
            scale = min(1.0, APP_CONFIG.camera_max_width / max(width, 1), APP_CONFIG.camera_max_height / max(height, 1))
            if scale < 1.0:
                small = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            else:
                small = rgb
            small_image = Image.fromarray(small, mode="RGB")
            luminance = float(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).mean())
            if luminance < APP_CONFIG.dark_luminance_threshold:
                boxes, predictions = [], []
                status = "Too dark: face a light source"
            else:
                small_boxes = detect_group_faces(small_image)[:MAX_FACES]
                boxes = [tuple(int(round(value / scale)) for value in (x, y, w, h)) for x, y, w, h in small_boxes]
                predictions, valid_boxes = [], []
                full_image = Image.fromarray(rgb, mode="RGB")
                for box in boxes:
                    crop = crop_with_padding(full_image, box, padding=0.20)
                    try:
                        prediction = predict_face(crop, show_errors=False, backend=self.backend, source_image=full_image, bbox=box)
                    except Exception as exc:
                        LOGGER.error("camera_inference_failed error_type=%s", type(exc).__name__)
                        continue
                    if prediction is not None:
                        valid_boxes.append(box)
                        predictions.append(prediction)
                boxes = valid_boxes
                if self.include_expression and boxes:
                    try:
                        emotion_predictor, _ = cached_emotion_predictor()
                        emotions = emotion_predictor.predict(full_image, mode=EMOTION_TTA5, boxes=tuple(boxes))
                        for prediction, emotion in zip(predictions, emotions):
                            prediction["emotion"] = emotion["emotion"]
                    except Exception as exc:
                        LOGGER.info("camera_expression_skipped error_type=%s", type(exc).__name__)
                predictions = self.tracker.update(boxes, predictions)
                status = f"Faces: {len(boxes)}" if boxes else "No face detected"
            annotated = draw_detections(Image.fromarray(rgb, mode="RGB"), boxes, predictions)
            draw = ImageDraw.Draw(annotated)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except OSError:
                font = ImageFont.load_default()
            draw.rectangle((8, 8, min(annotated.width - 8, 340), 36), fill=(20, 24, 39))
            draw.text((14, 14), status, fill=(255, 255, 255), font=font)
            self.last_rgb = np.asarray(annotated, dtype=np.uint8)
            self.last_boxes, self.last_predictions = boxes, predictions
            now = time.perf_counter()
            self.last_fps = 1.0 / max(now - self.last_tick, 1e-6)
            self.last_tick = now
            self.last_latency_ms = (now - started) * 1000.0
        output = self.last_rgb if self.last_rgb is not None else rgb
        return av.VideoFrame.from_ndarray(output, format="rgb24")


def _live_processor_factory(backend: str, include_expression: bool):
    class ConfiguredLiveProcessor(LiveInferenceProcessor):
        pass
    ConfiguredLiveProcessor.backend = backend
    ConfiguredLiveProcessor.include_expression = include_expression
    return ConfiguredLiveProcessor


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">AGE & GENDER AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Classification of Age and Gender from Facial Images
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        """
        Upload a photograph and choose the appropriate
        analysis mode. Choose Single Person for one face or Group Analysis for multiple faces.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            MODEL_REGISTRY["models"][age_gender_model_id(selected_backend)].get("display_name", "Age and gender model")
        )

    with col2:

        st.metric(
            "Input Size",
            MODEL_REGISTRY["models"][age_gender_model_id(selected_backend)].get("input_size", "Not available")
        )

    with col3:

        st.metric(
            "Face Detector",
            "YuNet + Haar"
        )

    st.divider()

    st.subheader(
        "How to use"
    )

    st.write(
        """
        1. Use **Analyze** for a photograph containing one person or multiple people.
        2. Every detected face is cropped and displayed separately.
        3. Age and gender are estimates and can be incorrect.
        """
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "🔍 Analyze":

    st.markdown(
        '<div class="section-title section-title-image">🔍 Analyze</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an image containing one person or multiple people. Choose Single Person for one face or Group Analysis for multiple faces. "
        "The existing YuNet detector filters small background detections, "
        "then the selected age-and-gender model predicts each detected face."
    )
    st.caption("Only upload images you have permission to analyze. Images are processed in memory and are not saved.")

    batch_files = st.file_uploader(
        "Or upload several images for batch analysis",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_image_analysis",
    )
    if batch_files:
        if len(batch_files) > APP_CONFIG.max_batch_images:
            st.warning(f"Batch size is limited to {APP_CONFIG.max_batch_images} images. The first {APP_CONFIG.max_batch_images} will be analyzed.")
        batch_files = batch_files[:APP_CONFIG.max_batch_images]
        progress = st.progress(0, text="Preparing batch analysis…")
        batch_rows = []
        for batch_index, batch_file in enumerate(batch_files, start=1):
            data = batch_file.getvalue()
            batch_results, batch_error, batch_truncated = analyze_uploaded_image(data, selected_backend)
            if batch_error:
                batch_rows.append({"image": f"Image {batch_index}", "status": batch_error, "faces": 0})
            else:
                batch_predictions = [prediction for _, prediction in batch_results]
                batch_boxes = [box for box, _ in batch_results]
                batch_image = decode_image(data, max_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_IMAGE_PIXELS)
                if batch_predictions:
                    _luminance, hints, _blur = image_quality(batch_image, dark_threshold=APP_CONFIG.dark_luminance_threshold, bright_threshold=APP_CONFIG.bright_luminance_threshold)
                    for pred, box in zip(batch_predictions, batch_boxes):
                        quality = face_quality(batch_image, box, min_blur_variance=APP_CONFIG.blur_variance_threshold)
                        pred["quality_score"], pred["quality_flags"] = quality.score, quality.flags
                        pred["uncertain"] = prediction_uncertain(pred, quality, gender_threshold=APP_CONFIG.gender_uncertain_below, age_threshold=APP_CONFIG.age_confidence_uncertain_below)
                    for row in result_rows(batch_boxes, batch_predictions):
                        row["image"] = f"Image {batch_index}"
                        row["status"] = "Uncertain" if row["uncertain"] else "Complete"
                        if batch_truncated:
                            row["status"] += " · face limit reached"
                        batch_rows.append(row)
                else:
                    batch_rows.append({"image": f"Image {batch_index}", "status": "No faces detected", "faces": 0})
                save_history("batch image", "Batch", batch_predictions, image_data=data)
            progress.progress(batch_index / len(batch_files), text=f"Processed {batch_index} of {len(batch_files)} images")
        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)
        st.download_button("Download batch results (CSV)", csv_bytes(batch_rows), "age_gender_batch.csv", "text/csv")
        st.download_button("Download batch results (JSON)", json_bytes(batch_rows), "age_gender_batch.json", "application/json")
        progress.empty()
        st.caption("Batch image contents are processed in memory and not included in the exports.")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_analysis"
    )
    include_emotion = st.checkbox("Also estimate facial expression", key="include_ag44b_emotion")
    emotion_mode = EMOTION_TTA5
    if include_emotion:
        st.caption("Expression estimates use High-accuracy mode.")

    if uploaded_file:
        image = load_uploaded_image(uploaded_file)
        if image is None:
            st.stop()

        analysis_results, analysis_error, analysis_truncated = analyze_uploaded_image(uploaded_file.getvalue(), selected_backend)
        if analysis_error:
            st.error(f"The selected model could not run: {analysis_error}")
            if selected_backend == "clientscan":
                st.info("Check Settings → Model for the expected local ONNX path. This app does not use an HF token or download the gated model.")
            st.stop()
        faces = [face for face, _ in analysis_results]
        if analysis_truncated:
            st.warning(f"More than {MAX_FACES} faces were detected. Only the first {MAX_FACES} will be analyzed.")
        if len(faces) == 0:
            st.error("No face detected. Try a clearer, front-facing image.")
        elif analysis_mode == "Single Person" and len(faces) > 1:
            st.error("This image contains multiple faces. Use Group Analysis instead.")
        else:
            predictions = [prediction for _, prediction in analysis_results]
            predicted_boxes = faces
            image_luminance, image_flags, image_blur = image_quality(
                image, dark_threshold=APP_CONFIG.dark_luminance_threshold,
                bright_threshold=APP_CONFIG.bright_luminance_threshold
            )
            for hint in image_flags:
                st.warning(hint)
            face_quality_results = [
                face_quality(image, box, min_blur_variance=APP_CONFIG.blur_variance_threshold)
                for box in predicted_boxes
            ]
            for idx, (prediction, quality) in enumerate(zip(predictions, face_quality_results), start=1):
                prediction["quality_score"] = quality.score
                prediction["quality_flags"] = quality.flags
                prediction["uncertain"] = prediction_uncertain(
                    prediction, quality, gender_threshold=APP_CONFIG.gender_uncertain_below,
                    age_threshold=APP_CONFIG.age_confidence_uncertain_below
                )
                if prediction["uncertain"]:
                    st.warning(f"Face {idx}: estimate is uncertain. Review image quality and treat this result cautiously.")
            cropped_faces = [crop_with_padding(image, face, padding=0.20) for face in predicted_boxes]

            if not predictions:
                st.error("Faces were detected, but the model could not produce predictions.")
            else:
                count = len(predictions)
                st.success(f"✅ {count} face(s) analyzed successfully.")
                summary_columns = st.columns(4)
                summary_columns[0].metric("Faces", count)
                summary_columns[1].metric("Mean estimated age", f"{np.mean([item['age'] for item in predictions]):.1f}")
                summary_columns[2].metric("Male estimates", sum(item.get("gender") == "Male" for item in predictions))
                summary_columns[3].metric("Uncertain", sum(bool(item.get("uncertain")) for item in predictions))
                age_groups = pd.Series([item.get("age_group", "Unknown") for item in predictions]).value_counts().sort_index()
                st.caption("Age-group distribution")
                st.bar_chart(age_groups)
                if any(pred.get("age_confidence") is not None and float(pred["age_confidence"]) < 60.0 for pred in predictions):
                    st.warning(
                        "Some age estimates have low confidence. Glasses, pose, lighting, and older faces can cause large errors; treat these values as estimates."
                    )

                emotion_results = []
                emotion_hash = None
                emotion_error = None
                if include_emotion:
                    try:
                        with st.spinner("Estimating facial emotion for the same detected faces…"):
                            emotion_results, emotion_hash = predict_emotions_from_bytes(
                                uploaded_file.getvalue(),
                                emotion_mode,
                                boxes=predicted_boxes,
                            )
                        if len(emotion_results) != len(predicted_boxes):
                            emotion_error = "Emotion results did not match the detected face boxes."
                    except Exception as exc:
                        LOGGER.error("expression_inference_failed error_type=%s", type(exc).__name__)
                        emotion_error = "The local expression model could not produce results. Check its status in About AI → Expression Model."
                    if emotion_error:
                        st.error(f"Emotion inference could not run: {emotion_error}")
                    elif not emotion_results:
                        st.info("No face was detected for emotion analysis.")
                    else:
                        st.subheader("Expression distribution")
                        st.bar_chart(pd.Series([item["emotion"] for item in emotion_results]).value_counts())
                        st.caption(
                            f"Expression Model v1 · High-accuracy mode · {emotion_hash[:12]} · "
                            f"{emotion_results[0]['total_latency_ms']:.0f} ms total"
                        )

                annotated = draw_detections(image, predicted_boxes, predictions)
                st.subheader("📷 Given Image")

                # UI-only preview scaling: annotate in original coordinates first,
                # then resize the complete annotated image uniformly for display.
                max_preview_width, max_preview_height = 850, 600
                preview_scale = min(
                    1.0,
                    max_preview_width / float(annotated.width),
                    max_preview_height / float(annotated.height),
                )
                preview = annotated
                if preview_scale < 1.0:
                    preview = annotated.resize(
                        (
                            max(1, int(round(annotated.width * preview_scale))),
                            max(1, int(round(annotated.height * preview_scale))),
                        ),
                        Image.Resampling.LANCZOS,
                    )

                with st.container(horizontal_alignment="center"):
                    st.image(preview, width=preview.width)
                st.caption("Green boxes show the faces used for prediction.")

                st.divider()
                st.subheader("✂️ Cropped Faces and Prediction Details")
                st.info(f"Total Faces: {count}")

                crop_columns = st.columns(min(4, count))
                for i, (crop, pred) in enumerate(zip(cropped_faces, predictions)):
                    with crop_columns[i % len(crop_columns)]:
                        with st.container(border=True):
                            st.image(crop, width=min(280, crop.width))
                            st.markdown(f"### 👤 Face {i + 1}")
                            st.write(f"⚥ **Gender:** {pred['gender']}")
                            st.write(f"🎯 **Gender Confidence:** {pred['gender_confidence']:.1f}%")
                            st.write(f"🎂 **Estimated Age:** {pred['age']} years")
                            st.write(f"👥 **Age Group:** {pred['age_group']}")
                            st.write(f"📊 **Age Confidence:** {age_confidence_text(pred)}")
                            if pred.get("uncertain"):
                                st.warning("Uncertain estimate")
                            if pred.get("quality_flags"):
                                st.caption(" · ".join(pred["quality_flags"]))
                            if include_emotion and not emotion_error and i < len(emotion_results):
                                emotion_result = emotion_results[i]
                                st.divider()
                                st.markdown(f"**🎭 Facial emotion: {emotion_result['emotion']}**")
                                st.write(f"🎯 **Expression confidence:** {emotion_result['confidence']:.1f}%")
                                top3 = sorted(emotion_result["probabilities"].items(), key=lambda item: item[1], reverse=True)[:3]
                                st.caption("Top expressions: " + " · ".join(f"{name.title()} {value*100:.1f}%" for name, value in top3))
                            timing = pred.get("timing_ms", {})
                            if timing:
                                st.caption(f"Decode {timing.get('decode', 0):.0f} ms · detection {timing.get('detection', 0):.0f} ms · face inference {timing.get('face_crop_and_prediction', 0):.0f} ms")

                export_rows = result_rows(predicted_boxes, predictions, emotion_results if include_emotion and not emotion_error else None)
                annotated_buffer = io.BytesIO()
                annotated.save(annotated_buffer, format="PNG")
                export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                export_col1.download_button("Annotated PNG", annotated_buffer.getvalue(), "analysis_annotated.png", "image/png")
                export_col2.download_button("Results CSV", csv_bytes(export_rows), "analysis_results.csv", "text/csv")
                export_col3.download_button("Results JSON", json_bytes(export_rows), "analysis_results.json", "application/json")
                export_col4.download_button("PDF report", pdf_report(export_rows), "analysis_report.pdf", "application/pdf")

                if include_emotion and emotion_results and not emotion_error:
                    save_key = hashlib.sha256(
                        uploaded_file.getvalue() + emotion_mode.encode("utf-8") + b"analyze"
                    ).hexdigest()
                    if st.session_state.get("last_saved_emotion_key") != save_key:
                        try:
                            save_emotion_history(
                                uploaded_file.getvalue(), emotion_results, emotion_mode,
                                analysis_id=get_analysis_id(analysis_mode, uploaded_file.getvalue()),
                            )
                            st.session_state["last_saved_emotion_key"] = save_key
                        except Exception:
                            st.warning("Emotion results are ready, but private emotion history could not be saved.")

                save_history(
                    uploaded_file.name,
                    analysis_mode,
                    predictions,
                    image_data=uploaded_file.getvalue()
                )

# LIVE CAMERA
# ============================================================

elif page == "🎭 Facial Emotion Detection":
    st.markdown('<div class="section-title section-title-image">🎭 Facial Emotion Detection</div>', unsafe_allow_html=True)
    st.write("Estimate seven visible facial expressions using the expression model. High-accuracy mode averages multiple views for steadier predictions.")
    st.caption("Emotion is a visual expression estimate, not a diagnosis. Only analyze images you have permission to use.")
    input_tab, camera_tab = st.tabs(["Upload image", "Camera snapshot"])
    with input_tab:
        emotion_upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="ag44b_upload")
    with camera_tab:
        emotion_camera = st.camera_input("Capture a photo for emotion analysis", key="ag44b_camera")
    emotion_file = emotion_upload or emotion_camera
    emotion_mode = EMOTION_TTA5
    st.caption("High-accuracy mode is used for every expression prediction.")
    if emotion_file is not None:
        image_bytes = emotion_file.getvalue()
        try:
            emotion_image = decode_image(image_bytes, max_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_IMAGE_PIXELS)
        except ImageValidationError as exc:
            st.error(str(exc))
        else:
            try:
                with st.spinner("Loading the expression model and analyzing detected faces…"):
                    emotion_results, emotion_hash = predict_emotions_from_bytes(image_bytes, emotion_mode)
            except Exception as exc:
                LOGGER.error("expression_inference_failed error_type=%s", type(exc).__name__)
                st.error("Expression analysis could not run. Check that the local model file is available.")
            else:
                if not emotion_results:
                    st.warning("No face was detected. Try a clear, front-facing image with a larger face.")
                else:
                    annotated_emotion = emotion_image.copy()
                    draw = ImageDraw.Draw(annotated_emotion)
                    try:
                        label_font = ImageFont.truetype("arial.ttf", max(14, min(22, emotion_image.width // 45)))
                    except OSError:
                        label_font = ImageFont.load_default()
                    for result in emotion_results:
                        x, y, width, height = result["bbox"]
                        draw.rectangle((x, y, x + width, y + height), outline=(0, 220, 120), width=max(2, emotion_image.width // 500))
                        label = f"{result['emotion']} · {result['confidence']:.0f}%"
                        label_box = draw.textbbox((x, max(0, y - 24)), label, font=label_font)
                        draw.rectangle(label_box, fill=(0, 220, 120))
                        draw.text((x, max(0, y - 24)), label, fill=(0, 0, 0), font=label_font)
                    st.image(annotated_emotion, caption="Detected faces with expression labels", width=850)
                    st.success(f"Analyzed {len(emotion_results)} detected face(s).")
                    st.caption(f"Expression model SHA-256: {emotion_hash} · High-accuracy mode · {emotion_results[0]['total_latency_ms']:.0f} ms total")
                    emotion_columns = st.columns(min(3, len(emotion_results)))
                    for index, result in enumerate(emotion_results):
                        with emotion_columns[index % len(emotion_columns)]:
                            with st.container(border=True):
                                st.subheader(f"Face {result['face_id']} · {result['emotion']}")
                                st.metric("Emotion confidence", f"{result['confidence']:.1f}%")
                                probability_df = pd.DataFrame({
                                    "Probability": result["probabilities"]
                                }).sort_values("Probability", ascending=False)
                                st.bar_chart(probability_df)
                                st.caption(f"Inference: {result['inference_latency_ms']:.0f} ms")
                    save_key = hashlib.sha256(image_bytes + emotion_mode.encode("utf-8") + b"emotion-page").hexdigest()
                    if st.session_state.get("last_saved_emotion_key") != save_key:
                        try:
                            save_emotion_history(
                                image_bytes, emotion_results, emotion_mode,
                                analysis_id=get_analysis_id("emotion-page", image_bytes),
                            )
                            st.session_state["last_saved_emotion_key"] = save_key
                        except Exception:
                            st.warning("Predictions are ready, but private emotion history could not be saved.")

elif page == "📷 Live Camera":
    st.markdown('<div class="section-title section-title-live">📷 Live Camera</div>', unsafe_allow_html=True)
    st.write("Use real-time camera analysis or take a snapshot. Both use the same local detector and age/gender prediction pipeline as Analyze.")
    st.caption("Frames stay in memory and are never saved. Camera access requires localhost or HTTPS.")
    live_tab, snapshot_tab = st.tabs(["Live stream", "Snapshot fallback"])
    try:
        request_headers = st.context.headers
        host = request_headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
        origin = request_headers.get("Origin", "").lower()
        forwarded_proto = request_headers.get("X-Forwarded-Proto", "").lower()
    except Exception:
        host, origin, forwarded_proto = "", "", ""
    secure_camera_context = host in {"localhost", "127.0.0.1", "::1"} or origin.startswith("https://") or forwarded_proto == "https" or os.environ.get("AGE_GENDER_HTTPS") == "1"

    with live_tab:
        if not WEBRTC_AVAILABLE:
            st.warning("Real-time video support is not installed. Use the Snapshot fallback tab.")
        elif not secure_camera_context:
            st.warning("Live camera is blocked on plain HTTP. Open localhost on this computer or use HTTPS on another device. Snapshot mode may still be available.")
        else:
            st.info("Select Start to begin live analysis. Select Stop when finished.")
            include_live_expression = st.checkbox("Include visible-expression estimate (uses more CPU)", value=False, key="live_expression")
            rtc_config = {"iceServers": [{"urls": APP_CONFIG.camera_stun_url}]} if APP_CONFIG.camera_stun_url else {"iceServers": []}
            context = webrtc_streamer(
                key="age_gender_live_stream",
                video_processor_factory=_live_processor_factory(selected_backend, include_live_expression),
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": {"width": APP_CONFIG.camera_max_width, "height": APP_CONFIG.camera_max_height}, "audio": False},
                async_processing=True,
            )
            if context.state.playing:
                st.success("Camera is on. Select Stop when finished.")
                processor = context.video_processor
                if processor is not None:
                    st.caption(f"Approx. {processor.last_fps:.1f} processed FPS · {processor.last_latency_ms:.0f} ms last inference")
            else:
                st.caption("The camera is off. Press Start to allow camera access. If access is denied, allow this site in browser permissions; if no device appears, check that a camera is connected and not in use by another app.")

    with snapshot_tab:
        camera_file = st.camera_input("Take a camera snapshot", key="live_camera_snapshot")
        if camera_file is not None:
            image = load_uploaded_image(camera_file)
            if image is not None:
                image_bytes = camera_file.getvalue()
                analysis_results, analysis_error, analysis_truncated = analyze_uploaded_image(image_bytes, selected_backend)
                if analysis_error:
                    st.error(analysis_error)
                elif not analysis_results:
                    st.warning("No face detected. Move closer to the camera and try again.")
                else:
                    if analysis_truncated:
                        st.warning(f"More than {MAX_FACES} faces were detected. Only the first {MAX_FACES} will be analyzed.")
                    boxes = [box for box, _ in analysis_results]
                    predictions = [pred for _, pred in analysis_results]
                    include_snapshot_expression = st.checkbox("Also estimate visible expression", value=False, key="snapshot_expression")
                    emotion_results = []
                    if include_snapshot_expression:
                        try:
                            emotion_predictor, _ = cached_emotion_predictor()
                            emotion_results = emotion_predictor.predict(image, mode=EMOTION_TTA5, boxes=tuple(boxes))
                            for pred, emotion in zip(predictions, emotion_results):
                                pred["emotion"] = emotion["emotion"]
                        except Exception as exc:
                            LOGGER.info("snapshot_expression_failed error_type=%s", type(exc).__name__)
                            st.warning("Expression estimates are unavailable for this snapshot.")
                    annotated = draw_detections(image, boxes, predictions)
                    scale = min(1.0, 850 / max(annotated.width, 1), 600 / max(annotated.height, 1))
                    if scale < 1:
                        annotated = annotated.resize((round(annotated.width * scale), round(annotated.height * scale)), Image.Resampling.LANCZOS)
                    with st.container(horizontal_alignment="center"):
                        st.image(annotated, width=annotated.width, caption="Green boxes show the faces used for prediction.")
                    st.subheader("Face results")
                    columns = st.columns(min(4, len(boxes)))
                    for index, (box, pred) in enumerate(zip(boxes, predictions)):
                        with columns[index % len(columns)]:
                            with st.container(border=True):
                                st.image(crop_with_padding(image, box, padding=0.20), width=240)
                                st.markdown(f"### Face {index + 1}")
                                st.write(f"Gender: {pred['gender']} · {pred['gender_confidence']:.1f}%")
                                st.write(f"Estimated age: {pred['age']} · {pred['age_group']}")
                                st.write(f"Age confidence: {age_confidence_text(pred)}")
                                if pred.get("emotion"):
                                    st.write(f"Visible expression: {pred['emotion']}")
                    snapshot_analysis_id = get_analysis_id("Live Camera Snapshot", image_bytes)
                    save_history(getattr(camera_file, "name", "camera_snapshot"), "Live Camera Snapshot", predictions, image_data=image_bytes)
                    if include_snapshot_expression and emotion_results:
                        try:
                            save_emotion_history(image_bytes, emotion_results, EMOTION_TTA5, analysis_id=snapshot_analysis_id)
                        except Exception:
                            st.warning("Expression results are ready, but private expression history could not be saved.")


# ============================================================
# ANALYTICS
# ============================================================
elif page == "🕘 History" and history_view == "Emotion History":
    st.markdown('<div class="section-title">🎭 Expression Prediction History</div>', unsafe_allow_html=True)
    if IS_GUEST:
        st.info("Guest emotion predictions are not saved. Sign in to keep private emotion history.")
    else:
        try:
            emotion_history = load_emotion_history()
        except Exception:
            st.error("Emotion history is temporarily unavailable. Please try again later.")
        else:
            if emotion_history.empty:
                st.info("You do not have any saved emotion predictions yet.")
            else:
                emotion_view = emotion_history.drop(columns=["Analysis ID"], errors="ignore")
                st.dataframe(emotion_view, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download emotion history as CSV",
                    data=emotion_view.to_csv(index=False),
                    file_name="my_emotion_history.csv",
                    mime="text/csv",
                )

elif page == "🕘 History" and history_view == "Analytics / Statistics":

    st.markdown(
        '<div class="section-title">📈 Analytics / Statistics</div>',
        unsafe_allow_html=True
    )
    try:
        df = load_user_history()
    except Exception:
        st.error("Your analytics are temporarily unavailable. Please try again later.")
    else:
        if IS_GUEST:
            st.info("Guest predictions are not saved. Sign in with Google to keep private history and view analytics.")
        elif df.empty:
            st.info("Your analytics will appear after your first prediction.")
        else:
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Predictions", len(df))
            metric2.metric("Average estimated age", f"{df['Estimated Age'].mean():.1f}")
            metric3.metric("Male estimates", int((df["Gender"] == "Male").sum()))
            metric4.metric("Female estimates", int((df["Gender"] == "Female").sum()))
            st.subheader("Gender estimates")
            st.bar_chart(df["Gender"].value_counts())
            st.subheader("Age groups")
            st.bar_chart(df["Age Group"].value_counts())
            try:
                emotion_metrics = load_emotion_history()
            except Exception:
                emotion_metrics = pd.DataFrame()
            st.subheader("Expression summary")
            if emotion_metrics.empty:
                st.info("Expression statistics will appear after an expression analysis.")
            else:
                st.bar_chart(emotion_metrics["Emotion"].value_counts())
                emotion_metrics["Day"] = pd.to_datetime(emotion_metrics["Timestamp"], utc=True, errors="coerce").dt.date
                trend = emotion_metrics.dropna(subset=["Day"]).groupby(["Day", "Emotion"]).size().unstack(fill_value=0)
                if not trend.empty:
                    st.subheader("Expression trend")
                    st.line_chart(trend)
# ============================================================
# HISTORY
# ============================================================

elif page == "🔍 Analyze":
    analysis_mode = st.sidebar.radio(
        "Analyze",
        ["Single Person", "Group Analysis"],
        key="analysis_mode",
    )
elif page == "🕘 History":
    st.markdown('<div class="section-title">🕘 Your Prediction History</div>', unsafe_allow_html=True)
    try:
        df = load_user_history()
    except Exception as exc:
        LOGGER.error("history_load_failed error_type=%s", type(exc).__name__)
        st.error("Your private history is temporarily unavailable. Please try again later.")
    else:
        if IS_GUEST:
            st.info("Guest predictions are not saved. Sign in with Google to save, download, and manage private history.")
        elif df.empty:
            st.info("You do not have any saved predictions yet.")
        else:
            try:
                emotion_rows = load_emotion_history()
            except Exception:
                emotion_rows = pd.DataFrame()
            if not emotion_rows.empty:
                emotion_lookup = emotion_rows[["Analysis ID", "Face", "Emotion"]].drop_duplicates(["Analysis ID", "Face"])
                df = df.merge(emotion_lookup, how="left", on=["Analysis ID", "Face"])
            else:
                df["Emotion"] = "Not analyzed"
            df["Emotion"] = df["Emotion"].fillna("Not analyzed")
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            search_text = filter_col1.text_input("Search history", placeholder="Gender, mode, emotion, or age group")
            gender_filter = filter_col2.selectbox("Gender", ["All", "Male", "Female"])
            minimum_age, maximum_age = filter_col3.slider("Estimated age range", 0, 100, (0, 100))
            emotion_options = ["All"] + sorted(value for value in df["Emotion"].dropna().unique() if value != "Not analyzed") + (["Not analyzed"] if (df["Emotion"] == "Not analyzed").any() else [])
            emotion_filter = filter_col4.selectbox("Expression", emotion_options)
            valid_dates = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce").dropna().dt.date
            if not valid_dates.empty:
                date_bounds = (min(valid_dates), max(valid_dates))
                chosen_dates = st.date_input("Date range", value=date_bounds, min_value=date_bounds[0], max_value=date_bounds[1], key="history_date_range")
            else:
                chosen_dates = ()
            filtered = df.copy()
            if search_text:
                mask = filtered.astype(str).apply(lambda col: col.str.contains(search_text, case=False, regex=False)).any(axis=1)
                filtered = filtered[mask]
            if gender_filter != "All":
                filtered = filtered[filtered["Gender"] == gender_filter]
            if emotion_filter != "All":
                filtered = filtered[filtered["Emotion"] == emotion_filter]
            if len(chosen_dates) == 2:
                row_dates = pd.to_datetime(filtered["Timestamp"], utc=True, errors="coerce").dt.date
                filtered = filtered[(row_dates >= chosen_dates[0]) & (row_dates <= chosen_dates[1])]
            filtered = filtered[filtered["Estimated Age"].between(minimum_age, maximum_age)]
            page_size = 25
            page_count = max(1, (len(filtered) + page_size - 1) // page_size)
            current_page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1, key="history_page")
            shown = filtered.iloc[(int(current_page) - 1) * page_size:int(current_page) * page_size]
            st.dataframe(shown.drop(columns=["Analysis ID"], errors="ignore"), use_container_width=True, hide_index=True)
            full_export = df.drop(columns=["Analysis ID"], errors="ignore")
            st.download_button("Download my history as CSV", full_export.to_csv(index=False), "my_prediction_history.csv", "text/csv")
            if not filtered.empty:
                row_choices = {
                    f"{row['Timestamp']} · Face {row['Face']} · {row['Gender']} · age {row['Estimated Age']}": (row["Analysis ID"], int(row["Face"]))
                    for _, row in shown.iterrows()
                }
                if row_choices:
                    selected_row = st.selectbox("Choose a history row to remove", list(row_choices), key="history_delete_row_choice")
                    if st.button("Delete selected row", key="delete_history_row"):
                        try:
                            delete_history_row(*row_choices[selected_row])
                            st.success("Selected history row deleted.")
                            st.rerun()
                        except Exception:
                            st.error("Selected row could not be deleted.")
            confirm_delete = st.checkbox("I understand this permanently deletes all my saved prediction history.", key="delete_history_all_confirm")
            if st.button("Delete all my history", type="secondary", disabled=not confirm_delete, key="delete_history_all"):
                try:
                    delete_user_history()
                    st.success("Your saved history was deleted.")
                    st.rerun()
                except Exception:
                    st.error("History could not be deleted. Please try again later.")
# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.markdown(
        '<div class="section-title">⚙️ Settings</div>',
        unsafe_allow_html=True
    )

    if settings_section == "Account":
        st.subheader("Account")
        if IS_GUEST:
            st.info("You are using Guest Mode. Sign in with Google to save private history.")
            if GOOGLE_AUTH_CONFIGURED:
                st.button("Sign in with Google", on_click=st.login, type="primary", key="account_google_sign_in")
        else:
            st.write(f"Name: {st.user.get('name', 'Not available')}")
            st.write(f"Email: {st.user.get('email', 'Not available')}")

    elif settings_section == "Appearance":
        st.subheader("Appearance")
        saved_theme = st.session_state.get("theme_preference", "Light")
        saved_theme_index = 1 if saved_theme == "Dark" else 0
        selected_theme = st.radio(
            "Theme",
            ["Light", "Dark"],
            index=saved_theme_index,
            horizontal=True,
            key="theme_selector",
        )
        if selected_theme != saved_theme:
            st.session_state["theme_preference"] = selected_theme
            st.rerun()
        st.caption("Your theme stays active while you move between pages.")

    elif settings_section == "Privacy":
        st.subheader("Privacy & Terms")
        st.info("Images and camera frames are processed in memory and are not saved. Signed-in users can save prediction metadata privately. Guest predictions are not saved.")
        st.warning("Only analyze images you have permission to use. Results are estimates, not identity or medical assessments, and must not be used for decisions about people.")
        if IS_GUEST:
            st.caption("Sign in to manage saved data. Guest sessions save no history.")
        else:
            retention_options = {"Off (keep until deleted)": 0, "30 days": 30, "90 days": 90}
            selected_retention = st.selectbox("Automatically delete history after", list(retention_options), index=list(retention_options.values()).index(get_retention_days()) if get_retention_days() in retention_options.values() else 0, key="retention_setting")
            if st.button("Save retention setting", key="save_retention"):
                set_retention_days(retention_options[selected_retention])
                st.success("Retention preference saved.")
            try:
                privacy_history = load_user_history()
            except Exception:
                privacy_history = pd.DataFrame()
            if not privacy_history.empty:
                export_history = privacy_history.drop(columns=["Analysis ID"], errors="ignore")
                st.download_button("Download my data as CSV", export_history.to_csv(index=False), "my_age_gender_data.csv", "text/csv")
                st.download_button("Download my data as JSON", export_history.to_json(orient="records", indent=2), "my_age_gender_data.json", "application/json")
            try:
                export_emotions = load_emotion_history()
            except Exception:
                export_emotions = pd.DataFrame()
            if not export_emotions.empty:
                export_emotions = export_emotions.drop(columns=["Analysis ID"], errors="ignore")
                st.download_button("Download expression data as CSV", export_emotions.to_csv(index=False), "my_expression_data.csv", "text/csv")
                st.download_button("Download expression data as JSON", export_emotions.to_json(orient="records", indent=2), "my_expression_data.json", "application/json")
            delete_confirmation = st.checkbox("I understand this deletes this app's saved data and consent record (not my Google account).", key="delete_account_confirm")
            if st.button("Delete my app data", disabled=not delete_confirmation, key="delete_app_data"):
                try:
                    delete_user_account_data()
                    st.success("Your saved app data was deleted.")
                    st.session_state.pop("consent_check", None)
                    st.rerun()
                except Exception:
                    st.error("Your app data could not be deleted. Please try again.")

    elif settings_section == "Model":
        st.subheader("Prediction model")
        options = [MODEL_REGISTRY["models"]["gra_net_compact"]["display_name"], MODEL_REGISTRY["models"]["agevision_onnx"]["display_name"]]
        selected = st.session_state.get("prediction_backend", options[0])
        if selected not in options:
            selected = MODEL_REGISTRY["models"]["gra_net_compact"]["display_name"]
        st.selectbox(
            "Choose the model used for image and live-camera predictions",
            options,
            index=options.index(selected),
            key="prediction_backend",
        )
        if st.session_state.get("prediction_backend") == MODEL_REGISTRY["models"]["agevision_onnx"]["display_name"]:
            expected_path = resolve_model_path(BASE_DIR, MODEL_REGISTRY["models"]["agevision_onnx"])
            st.info(
                "AgeVision AI uses its locally installed ONNX model. The app does not "
                "download model files or send account details to Hugging Face."
            )
            model_meta = MODEL_REGISTRY["models"]["agevision_onnx"]
            st.write(f"**Model:** {model_meta['display_name']} · **Version:** {model_meta['version']} · **Input:** {model_meta['input_size']}")
            st.caption(f"SHA-256: {short_hash(model_meta)}")
            if expected_path.is_file():
                st.success("Local model: loaded")
            else:
                st.warning("Local model: unavailable. GRA_Net remains available.")
            st.caption("Age confidence is model-derived and not independently calibrated.")
            if st.button("Retry AgeVision AI for the last uploaded image", key="retry_clientscan"):
                analyze_uploaded_image.clear()
                st.rerun()
        else:
            model_meta = MODEL_REGISTRY["models"]["gra_net_compact"]
            st.write(f"**Model:** {model_meta['display_name']} · **Version:** {model_meta['version']} · **Input:** {model_meta['input_size']}")
            st.caption(f"SHA-256: {short_hash(model_meta)}")
            st.success("Local model: loaded" if resolve_model_path(BASE_DIR, model_meta).is_file() else "Local model: unavailable")

    elif settings_section == "Security":
        st.subheader("Security")
        if IS_GUEST:
            st.info("Sign in with Google to manage your account security.")
            if GOOGLE_AUTH_CONFIGURED:
                st.button("Sign in with Google", on_click=st.login, type="primary", key="security_google_sign_in")
        else:
            st.write(f"Signed in as {st.user.get('email', st.user.get('name', 'Google account'))}.")
            st.caption("This browser session uses Google sign-in. App history contains metadata only; uploaded images and frames are not stored.")
            st.write("Session started for this app tab. Signing out ends the Google-authenticated app session.")
            if st.button("Sign out", key="settings_sign_out"):
                st.logout()

# MODEL
# ============================================================

elif page == "ℹ️ About AI" and about_section == "Model":

    st.markdown(
        '<div class="section-title">🧠 Active Model</div>',
        unsafe_allow_html=True
    )

    about_model = MODEL_REGISTRY["models"][age_gender_model_id(selected_backend)]
    st.caption(f"Active model: {about_model['display_name']}")
    st.info(f"{about_model['display_name']} estimates age and gender from detected faces.")
    st.write(f"**Version:** {about_model['version']}  ·  **Input:** {about_model['input_size']}")
    st.write("**Training data:** " + str(about_model.get("dataset", "Not available")))
    st.write("**Metrics:** Not available for this exact model version.")
    st.caption("The project does not claim the original research paper’s accuracy for this checkpoint.")
    st.write("**Known limitations:** " + " ".join(about_model.get("limitations", [])))
    with st.expander("Technical details"):
        st.write({"model_id": age_gender_model_id(selected_backend), "input": about_model.get("input_size"), "outputs": about_model.get("outputs"), "sha256": about_model.get("sha256")})

# EXPRESSION MODEL CARD
# ============================================================

elif page == "ℹ️ About AI" and about_section == "Expression Model":
    expression_model = MODEL_REGISTRY["models"]["expression_v1"]
    st.markdown('<div class="section-title">🎭 Expression Model</div>', unsafe_allow_html=True)
    st.info("This model estimates visible facial expression labels. It cannot determine a person’s true feelings or diagnose a condition.")
    st.write(f"**Version:** {expression_model['version']} · **Input:** {expression_model['input_size']}")
    st.write("**Classes:** " + ", ".join(expression_model["class_order"]))
    st.write("**Metrics:** Not available for this exact checkpoint.")
    st.warning("Expression estimates are substantially less reliable than age/gender estimates. Minority labels such as disgust and fear can be especially difficult.")
    with st.expander("Technical details"):
        st.write({"model_id": "expression_v1", "outputs": expression_model["outputs"], "sha256": expression_model["sha256"], "inference": "high-accuracy multi-view averaging"})

# DATASETS
# ============================================================

elif page == "ℹ️ About AI" and about_section == "Dataset & Training":

    st.markdown(
        '<div class="section-title">📚 Dataset & Training</div>',
        unsafe_allow_html=True
    )
    current_model = MODEL_REGISTRY["models"][age_gender_model_id(selected_backend)]
    st.write("Current model dataset: " + str(current_model.get("dataset", "Not available")))
    st.caption("Dataset details are shown only when documented for the selected checkpoint.")
    st.write("The existing training utilities remain separate from app inference. No training or evaluation runs were performed during this upgrade.")

# EVALUATION
# ============================================================

elif page == "ℹ️ About AI" and about_section == "Evaluation":

    st.markdown(
        '<div class="section-title">📈 Evaluation</div>',
        unsafe_allow_html=True
    )
    active_meta = MODEL_REGISTRY["models"][age_gender_model_id(selected_backend)]
    metrics_source = active_meta.get("metrics_source")
    if metrics_source:
        evaluation_file = (BASE_DIR / metrics_source).resolve()
        if BASE_DIR not in evaluation_file.parents or not evaluation_file.is_file():
            st.info("Metrics are not available for this exact model version.")
        else:
            st.caption(f"Source: {metrics_source}")
            st.code(evaluation_file.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Metrics are not available for this exact model version.")

# PRIVACY
# ============================================================

elif page == "ℹ️ About AI" and about_section == "Responsible AI":

    st.markdown(
        '<div class="section-title">🛡️ Responsible AI</div>',
        unsafe_allow_html=True
    )

    st.warning(
        """
        Age and gender predictions are model estimates and
        may be incorrect.
        """
    )

    st.write(
        """
        Results can vary depending on:

        • Lighting

        • Face orientation

        • Image quality

        • Occlusion

        • Glasses or accessories

        • Face size

        • Background

        • Dataset/model limitations
        """
    )

    st.info(
        """
        Uploaded images and live camera frames are processed by the app server
        and are not saved as image files. Google-signed-in users can save
        prediction metadata privately to their account and manage it on the
        History page. Guest predictions are not saved.
        """
    )

    st.warning(
        "The gender label is the model’s visual classification, not a person’s identity. Do not use these estimates for hiring, credit, insurance, policing, access control, or other decisions about people. Results vary with lighting, pose, occlusion, age, and image quality."
    )






# ABOUT AI DEFAULT
# ============================================================

elif page == "ℹ️ About AI":

    st.markdown(
        '<div class="section-title">ℹ️ How It Works</div>',
        unsafe_allow_html=True
    )
    st.write(
        f"The app detects faces, crops each face, then applies the documented preprocessing for {MODEL_REGISTRY['models'][age_gender_model_id(selected_backend)]['display_name']}."
    )
    st.write(
        "Results are visual estimates. Confidence values come from model outputs and are not independently calibrated."
    )
    st.subheader("Available application features")
    st.caption(f"Active model: {MODEL_REGISTRY['models'][age_gender_model_id(selected_backend)]['display_name']} · {MODEL_REGISTRY['models'][age_gender_model_id(selected_backend)]['input_size']}")
    st.write("Home, Analyze with Single Person or Group Analysis, Live Camera, private History, Analytics / Statistics, Settings, and this About AI section.")
