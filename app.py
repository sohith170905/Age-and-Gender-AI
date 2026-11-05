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

import streamlit as st
from core.config import load_config
from core.model_registry import load_registry, age_gender_model_id, resolve_model_path, short_hash


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

st.session_state.setdefault("theme_preference", "Dark")
theme_preference = st.session_state["theme_preference"]
AUTH_USER_ID = st.session_state.get("authenticated_user_id")
IS_LOGGED_IN = bool(AUTH_USER_ID)
IS_GUEST = bool(st.session_state.get("guest_mode")) and not IS_LOGGED_IN
CURRENT_USER_ID = AUTH_USER_ID if IS_LOGGED_IN else None
AUTH_ENTRY_PENDING = not (IS_LOGGED_IN or IS_GUEST)
AUTH_WELCOME_PENDING = AUTH_ENTRY_PENDING and st.session_state.get("auth_entry_mode", "welcome") == "welcome"

if theme_preference == "Dark" and not AUTH_WELCOME_PENDING:
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
    """ if not AUTH_WELCOME_PENDING else "",
    unsafe_allow_html=True
)


# Shared contrast fixes for interactive controls and secondary text in dark mode.
if theme_preference == "Dark" and not AUTH_WELCOME_PENDING:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] small,
        [data-testid="stAppViewContainer"] .subtitle,
        [data-testid="stAppViewContainer"] .metric-label {
            color: #d5deed !important;
            opacity: 1 !important;
        }

        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] {
            background: #20263a !important;
            color: #e2e8f0 !important;
            border: 1px dashed #7180a4 !important;
            border-radius: 12px !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] * {
            color: #e2e8f0 !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] button {
            background: #6d3be8 !important;
            color: #ffffff !important;
            border: 1px solid #a78bfa !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] button:hover {
            background: #8056f2 !important;
            border-color: #c4b5fd !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] {
            background: #252b45 !important;
            color: #f8fafc !important;
            border: 1px solid #7180a4 !important;
            border-radius: 9px !important;
            opacity: 1 !important;
        }
        /* Streamlit renders the selected filename inside nested surfaces; theme each one. */
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] *:not(button) {
            background-color: #252b45 !important;
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] p,
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] span,
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] small {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] button {
            background: #303858 !important;
            color: #ffffff !important;
            border-color: #a78bfa !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] button:hover {
            background: #493b70 !important;
        }
        /* Current Streamlit versions render selected uploads as file chips. */
        [data-testid="stAppViewContainer"] [data-testid="stFileChips"] [data-testid="stFileChip"] {
            background: #252b45 !important;
            color: #f8fafc !important;
            border: 1px solid #7180a4 !important;
            border-radius: 9px !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileChips"] [data-testid="stFileChipName"],
        [data-testid="stAppViewContainer"] [data-testid="stFileChips"] [data-testid="stFileChip"] div,
        [data-testid="stAppViewContainer"] [data-testid="stFileChips"] [data-testid="stFileChip"] span,
        [data-testid="stAppViewContainer"] [data-testid="stFileChips"] [data-testid="stFileChip"] small {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileChips"] [data-testid="stFileChipDeleteBtn"] button {
            background: #303858 !important;
            color: #ffffff !important;
            border: 1px solid #a78bfa !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileChips"] [data-testid="stFileChipDeleteBtn"] button:hover {
            background: #493b70 !important;
        }

        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"],
        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] div {
            background: #151a2d !important;
            color: #e2e8f0 !important;
            border-color: #475569 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] p,
        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] span {
            color: #d5deed !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] button {
            background: #6d3be8 !important;
            color: #ffffff !important;
            border: 1px solid #a78bfa !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] button:disabled {
            background: #343b55 !important;
            color: #e2e8f0 !important;
            border-color: #7180a4 !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] video {
            background: #151a2d !important;
            filter: brightness(1.25) contrast(1.04);
        }
        [data-testid="stAppViewContainer"] [data-testid="stCameraInput"] img {
            filter: brightness(1.25) contrast(1.04);
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: .35rem;
            border-bottom: 1px solid #475569 !important;
        }
        [data-testid="stTabs"] [role="tab"] {
            color: #e2e8f0 !important;
            opacity: 1 !important;
            background: #20263a !important;
            border-radius: 9px 9px 0 0;
        }
        [data-testid="stTabs"] [role="tab"] *,
        [data-testid="stTabs"] [role="tab"]:hover * {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        [data-testid="stTabs"] [role="tab"]:hover {
            background: #303853 !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: #c4b5fd !important;
            border-bottom: 3px solid #8b5cf6 !important;
            background: #262b42 !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] * {
            color: #c4b5fd !important;
        }

        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-baseweb="input"] input,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-baseweb="textarea"] textarea,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-testid="stNumberInput"] input,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-testid="stDateInput"] input,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-testid="stTimeInput"] input {
            background: #20263a !important;
            color: #f1f5ff !important;
            border-color: #64748b !important;
            caret-color: #c4b5fd !important;
        }
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-baseweb="input"] input::placeholder,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-baseweb="textarea"] textarea::placeholder {
            color: #cbd5e1 !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
            background: #20263a !important;
            color: #f1f5ff !important;
            border-color: #64748b !important;
        }
        [data-baseweb="popover"] [role="option"] {
            background: #20263a !important;
            color: #e2e8f0 !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [role="option"][aria-selected="true"] {
            background: #3b315d !important;
            color: #ffffff !important;
        }

        html body .history-table,
        html body .history-table td {
            background: #202438 !important;
            color: #f3f5ff !important;
            border-color: #59617f !important;
        }
        html body .history-table th {
            background: #303858 !important;
            color: #ffffff !important;
            border-color: #7180a4 !important;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border-color: #475569 !important;
        }
        .stMarkdown table {
            color: #f1f5ff !important;
            background: #202438 !important;
        }
        .stMarkdown table th {
            color: #ffffff !important;
            background: #303858 !important;
        }
        .stMarkdown table td {
            color: #f1f5ff !important;
            background: #202438 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if theme_preference == "Light" and not AUTH_WELCOME_PENDING:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { color: #17203f !important; }
        [data-testid="stSidebar"] { background: #ffffff !important; }
        [data-testid="stSidebar"] *,
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
        [data-testid="stAppViewContainer"] .stRadio label,
        [data-testid="stAppViewContainer"] .stCheckbox label {
            color: #17203f !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] small,
        [data-testid="stAppViewContainer"] .subtitle,
        [data-testid="stAppViewContainer"] .metric-label {
            color: #4b556b !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            color: #17203f !important;
            border: 1px dashed #aab3c5 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] * {
            color: #27334e !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] button {
            background: #6d35e8 !important;
            color: #ffffff !important;
            border-color: #6d35e8 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploaderFile"] {
            background: #f2f4fa !important;
            color: #17203f !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-baseweb="input"] input,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-baseweb="textarea"] textarea,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-auth-signin-page)) [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #17203f !important;
            border-color: #aab3c5 !important;
            opacity: 1 !important;
        }
        [data-baseweb="popover"] [role="option"] {
            background: #ffffff !important;
            color: #17203f !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [role="option"][aria-selected="true"] {
            background: #eee8ff !important;
            color: #321b73 !important;
        }
        [data-testid="stTabs"] [role="tab"] {
            color: #34405a !important;
            background: #f0f2f8 !important;
            opacity: 1 !important;
        }
        [data-testid="stTabs"] [role="tab"] * { color: #34405a !important; opacity: 1 !important; }
        [data-testid="stTabs"] [role="tab"]:hover { background: #e7eafa !important; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: #5427c5 !important;
            background: #ffffff !important;
            border-bottom: 3px solid #6d35e8 !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] * { color: #5427c5 !important; }
        [data-testid="stDownloadButton"] > button,
        [data-testid="stAppViewContainer"] div[data-testid="stButton"] > button,
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            color: #17203f !important;
            background: #ffffff !important;
            border: 1px solid #aab3c5 !important;
            opacity: 1 !important;
        }
        [data-testid="stDownloadButton"] > button:hover,
        [data-testid="stAppViewContainer"] div[data-testid="stButton"] > button:hover,
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            color: #321b73 !important;
            background: #f0ebff !important;
            border-color: #6d35e8 !important;
        }
        [data-testid="stAppViewContainer"] div[data-testid="stButton"] > button[kind="primary"] {
            color: #ffffff !important;
            background: #6d35e8 !important;
            border-color: #6d35e8 !important;
        }
        .history-table, .history-table td { background: #ffffff !important; color: #17203f !important; }
        .history-table th { background: #e8ecf8 !important; color: #17203f !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

tab_inactive_color = "#334155" if theme_preference == "Light" else "#cbd5e1"
tab_active_color = "#6d3fea" if theme_preference == "Light" else "#c4b5fd"
tab_divider_color = "#cbd5e1" if theme_preference == "Light" else "#475569"
st.markdown(
    f"""
    <style>
    .stApp :is(.st-key-emotion-page-tabs, .st-key-live-camera-tabs) [data-testid="stTabs"] [role="tablist"] {{
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
        gap: 1.25rem !important;
        border-bottom: 1px solid {tab_divider_color} !important;
    }}
    .stApp :is(.st-key-emotion-page-tabs, .st-key-live-camera-tabs) [data-testid="stTabs"] [role="tab"] {{
        color: {tab_inactive_color} !important;
        opacity: 1 !important;
        background: transparent !important;
        border: 0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }}
    .stApp :is(.st-key-emotion-page-tabs, .st-key-live-camera-tabs) [data-testid="stTabs"] [role="tab"] *,
    .stApp :is(.st-key-emotion-page-tabs, .st-key-live-camera-tabs) [data-testid="stTabs"] [role="tab"]:hover {{
        color: {tab_inactive_color} !important;
        opacity: 1 !important;
        background: transparent !important;
    }}
    .stApp :is(.st-key-emotion-page-tabs, .st-key-live-camera-tabs) [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: {tab_active_color} !important;
        background: transparent !important;
        border-bottom: 2px solid {tab_active_color} !important;
    }}
    .stApp :is(.st-key-emotion-page-tabs, .st-key-live-camera-tabs) [data-testid="stTabs"] [role="tab"][aria-selected="true"] * {{
        color: {tab_active_color} !important;
    }}
    </style>
    """ if not AUTH_WELCOME_PENDING else "",
    unsafe_allow_html=True,
)

def _chart_colors():
    if st.session_state.get("theme_preference", "Dark") == "Dark":
        return "#151827", "#e2e8f0", "#3b435c"
    return "#f7f8fc", "#17203f", "#d6dced"


def render_bar_chart(data):
    """Render a compact bar chart with colors that follow the selected app theme."""
    background, foreground, grid = _chart_colors()
    if isinstance(data, pd.Series):
        series = data.copy()
        frame = pd.DataFrame({"category": series.index.map(str), "value": series.to_numpy()})
    else:
        frame = data.copy()
        numeric_columns = list(frame.select_dtypes(include="number").columns)
        if not numeric_columns:
            return
        value_column = numeric_columns[0]
        frame = pd.DataFrame({"category": frame.index.map(str), "value": frame[value_column].to_numpy()})
    chart = (
        alt.Chart(frame)
        .mark_bar(color="#8b5cf6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("category:N", sort=None, title=None, axis=alt.Axis(labelColor=foreground, labelAngle=0, labelLimit=180, domainColor=grid, tickColor=grid)),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(labelColor=foreground, gridColor=grid, domainColor=grid, tickColor=grid)),
            tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("value:Q", title="Count")],
        )
        .properties(height=260, background=background)
        .configure_view(fill=background, stroke=grid)
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def render_distribution_chart(data, preferred_order):
    """Render Group Analysis counts on a consistent, light chart canvas."""
    counts = {str(category): int(value) for category, value in data.items()}
    categories = list(preferred_order) + sorted(set(counts) - set(preferred_order))
    frame = pd.DataFrame({
        "category": categories,
        "value": [counts.get(category, 0) for category in categories],
    })
    upper_bound = max(1.0, max(frame["value"].tolist(), default=0) * 1.18)
    x_axis = alt.X(
        "category:N", sort=categories, title=None,
        axis=alt.Axis(labelAngle=0, labelColor="#243047", labelFontSize=12,
                      labelLimit=140, domainColor="#cbd5e1", tickColor="#cbd5e1"),
    )
    y_axis = alt.Y(
        "value:Q", title=None, scale=alt.Scale(domain=[0, upper_bound], nice=False),
        axis=alt.Axis(labelColor="#4b5563", labelFontSize=11,
                      gridColor="#e5eaf1", domainColor="#cbd5e1", tickColor="#cbd5e1"),
    )
    bars = alt.Chart(frame).mark_bar(
        color="#1976d2", cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=x_axis,
        y=y_axis,
        tooltip=[alt.Tooltip("category:N", title="Category"),
                 alt.Tooltip("value:Q", title="Faces", format="d")],
    )
    labels = alt.Chart(frame).mark_text(
        align="center", baseline="bottom", dy=-6, color="#243047",
        fontSize=13, fontWeight="bold",
    ).encode(
        x=x_axis,
        y=y_axis,
        text=alt.Text("value:Q", format="d"),
    )
    chart = (
        alt.layer(bars, labels)
        .properties(height=380, background="#ffffff")
        .configure_view(fill="#ffffff", stroke="#e5eaf1")
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def render_line_chart(data):
    """Render time-series data with readable axes in both app themes."""
    background, foreground, grid = _chart_colors()
    frame = data.copy().reset_index()
    x_field = frame.columns[0]
    frame[x_field] = frame[x_field].astype(str)
    melted = frame.melt(id_vars=[x_field], var_name="series", value_name="value")
    order = frame[x_field].drop_duplicates().tolist()
    palette = ["#8b5cf6", "#22d3ee", "#f59e0b", "#34d399", "#fb7185", "#60a5fa"]
    chart = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(f"{x_field}:O", sort=order, title=None, axis=alt.Axis(labelColor=foreground, labelAngle=-25, domainColor=grid, tickColor=grid)),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(labelColor=foreground, gridColor=grid, domainColor=grid, tickColor=grid)),
            color=alt.Color("series:N", scale=alt.Scale(range=palette), legend=alt.Legend(labelColor=foreground, titleColor=foreground)),
            tooltip=[alt.Tooltip(f"{x_field}:O", title="Date"), alt.Tooltip("series:N", title="Series"), alt.Tooltip("value:Q", title="Count")],
        )
        .properties(height=280, background=background)
        .configure_view(fill=background, stroke=grid)
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


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

    face_cascade = load_face_detector()
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
    from sqlalchemy import create_engine, text

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
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS app_users (
                user_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
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
def cached_database_engine(database_schema_revision: str = "accounts-v1"):
    """Cache the engine by schema revision so hot reloads cannot retain an old schema engine."""
    return get_database_engine()


def _start_guest_session():
    for key in ("authenticated_user_id", "authenticated_user_name", "authenticated_user_email"):
        st.session_state.pop(key, None)
    st.session_state["guest_mode"] = True
    st.session_state["guest_consent_accepted"] = True
    st.session_state.setdefault("guest_prediction_history", [])
    st.session_state.setdefault("guest_emotion_history", [])
    st.session_state["auth_entry_mode"] = "welcome"
    st.rerun()


def _return_to_auth_entry():
    theme = st.session_state.get("theme_preference", "Dark")
    st.session_state.clear()
    st.session_state["theme_preference"] = theme
    st.session_state["auth_entry_mode"] = "welcome"
    st.rerun()


def _start_account_session(account):
    st.session_state.pop("guest_mode", None)
    st.session_state.pop("guest_prediction_history", None)
    st.session_state.pop("guest_emotion_history", None)
    st.session_state["authenticated_user_id"] = account["user_id"]
    st.session_state["authenticated_user_name"] = account["full_name"]
    st.session_state["authenticated_user_email"] = account["email"]
    st.session_state["auth_entry_mode"] = "welcome"
    st.rerun()


def _render_auth_entry():
    if IS_LOGGED_IN or IS_GUEST:
        return

    mode = st.session_state.get("auth_entry_mode", "welcome")
    if mode == "welcome":
        st.markdown(
            """
            <style>
            .stApp:has(.st-key-auth-welcome-shell) {
                background: radial-gradient(ellipse at 83% 12%, rgba(118,37,228,.46), transparent 34%),
                    radial-gradient(ellipse at 51% 58%, rgba(79,27,159,.30), transparent 47%),
                    linear-gradient(118deg,#100b2b 0%,#160d37 48%,#1b0b43 100%) !important;
            }
            .stApp:has(.st-key-auth-welcome-shell) [data-testid="stHeader"],
            .stApp:has(.st-key-auth-welcome-shell) [data-testid="stToolbar"],
            .stApp:has(.st-key-auth-welcome-shell) [data-testid="stSidebar"] { display:none !important; }
            .stApp:has(.st-key-auth-welcome-shell) .main .block-container {
                max-width:1240px; min-height:100vh; padding:clamp(24px,4vh,48px) clamp(18px,5vw,64px) 34px;
                position:relative; z-index:1; display:flex; flex-direction:column; justify-content:center;
            }
            .st-key-auth-welcome-shell { position:relative; z-index:1; width:100%; max-width:1120px; margin:0 auto; }
            .auth-welcome-bg { position:fixed; inset:0; width:100vw; height:100vh; z-index:0; pointer-events:none; }
            .auth-welcome-hero { text-align:center; color:#f8f5ff; margin:0 auto 48px; position:relative; z-index:2; }
            .auth-welcome-mark { display:block; width:80px; height:80px; margin:0 auto 10px; filter:drop-shadow(0 0 12px rgba(185,97,255,.55)); }
            .auth-welcome-title { margin:0; color:#fbf9ff; font-size:clamp(34px,5vw,58px); line-height:1.04; font-weight:850; letter-spacing:.035em; text-shadow:0 4px 30px rgba(178,105,255,.26); }
            .auth-welcome-subtitle { margin:8px 0 0; color:#d7c4ff; font-size:clamp(18px,2.2vw,27px); font-weight:500; }
            .auth-welcome-description { margin:17px 0 0; color:#d8d2ee; font-size:clamp(15px,1.55vw,20px); }
            .st-key-auth-guest-card, .st-key-auth-account-card {
                height:100%; min-height:340px; padding:32px 34px 30px !important; box-sizing:border-box; text-align:center !important; color:#f5efff;
                border:1.5px solid transparent !important; border-radius:26px !important;
                background:linear-gradient(145deg,rgba(31,20,67,.90),rgba(22,18,56,.83)) padding-box,
                    linear-gradient(120deg,#d25bff,#6c60ff 58%,#41c9e6) border-box !important;
                box-shadow:0 14px 36px rgba(7,5,26,.24),inset 0 0 24px rgba(126,73,226,.07);
                display:flex; flex-direction:column; justify-content:center;
            }
            .auth-option-icon { display:block; width:56px; height:56px; margin:0 auto 12px; color:#e8dcff; filter:drop-shadow(0 0 7px rgba(179,111,255,.36)); }
            .auth-option-title { color:#f7f2ff; font-size:clamp(21px,2.1vw,29px); font-weight:750; margin:0 0 8px; }
            .auth-option-copy { color:#d9d1ef; font-size:16px; line-height:1.5; max-width:390px; margin:0 auto 21px; min-height:48px; }
            .st-key-auth-guest-card [data-testid="stButton"], .st-key-auth-account-card [data-testid="stButton"] { margin-top:auto; }
            .st-key-auth-guest-card [data-testid="stButton"] button, .st-key-auth-account-card [data-testid="stButton"] button {
                min-height:53px; border-radius:13px; border:1px solid rgba(224,174,255,.65); color:#fff;
                background:linear-gradient(105deg,#893eff,#712ce9 54%,#a43de8); font-size:17px; font-weight:750;
                box-shadow:0 0 24px rgba(138,62,255,.36),inset 0 1px rgba(255,255,255,.2); transition:transform .16s ease,box-shadow .16s ease;
            }
            .st-key-auth-guest-card [data-testid="stButton"] button:hover, .st-key-auth-account-card [data-testid="stButton"] button:hover {
                transform:translateY(-2px); box-shadow:0 0 32px rgba(166,84,255,.58); border-color:#e9bcff; color:#fff;
            }
            .st-key-auth-guest-card [data-testid="stButton"] button::after { content:"  →"; font-size:21px; }
            .auth-or-divider { height:300px; width:1px; margin:auto; position:relative; background:linear-gradient(transparent,rgba(190,139,255,.62) 18%,rgba(190,139,255,.62) 82%,transparent); }
            .auth-or-divider span { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:54px; height:54px; display:grid; place-items:center; border:1px solid #9864e6; border-radius:50%; background:#21134a; color:#f8f2ff; font-weight:700; }
            .auth-signin-prompt { text-align:center; color:#eee8fb; font-size:17px; margin:28px 0 8px; }
            .st-key-auth-signin-button { max-width:320px; width:min(100%,320px); margin:0 auto; }
            .st-key-auth-signin-button [data-testid="stButton"] button { min-height:50px; border-radius:13px; border:1px solid #d6bdff; background:rgba(22,15,48,.7); color:#f6f0ff; font-weight:700; font-size:17px; }
            .st-key-auth-signin-button [data-testid="stButton"] button:hover { background:rgba(116,55,213,.36); border-color:#f1dbff; color:#fff; }
            @media (max-width:760px) {
                .stApp:has(.st-key-auth-welcome-shell) .main .block-container { padding:28px 18px 36px; justify-content:flex-start; }
                .auth-welcome-hero { margin-bottom:32px; } .auth-welcome-mark { width:68px; height:68px; }
                .st-key-auth-guest-card, .st-key-auth-account-card { min-height:285px; padding:26px 22px !important; }
                .auth-or-divider { display:none; }
                [data-testid="stColumn"]:has(.auth-or-divider) { display:none !important; }
            }
            @media (max-width:520px) {
                .stApp:has(.st-key-auth-welcome-shell) [data-testid="stHorizontalBlock"] { flex-direction:column; gap:14px; }
                .stApp:has(.st-key-auth-welcome-shell) [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width:100% !important; flex:1 1 100% !important; }
                .auth-welcome-title { font-size:clamp(29px,9vw,40px); }
                .auth-welcome-description { max-width:360px; margin-left:auto; margin-right:auto; }
            }
            @media (prefers-reduced-motion:reduce) { .st-key-auth-guest-card button,.st-key-auth-account-card button,.st-key-auth-signin-button button { transition:none !important; } }
            </style>
            <svg class="auth-welcome-bg" viewBox="0 0 1680 950" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <linearGradient id="bottomWaveA" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#6624a3" stop-opacity=".62"/><stop offset="1" stop-color="#4420bd" stop-opacity=".12"/></linearGradient>
                <linearGradient id="bottomWaveB" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#37206f" stop-opacity=".15"/><stop offset=".67" stop-color="#6f20d2" stop-opacity=".72"/><stop offset="1" stop-color="#9b30ff" stop-opacity=".62"/></linearGradient>
                <radialGradient id="faceGlow"><stop stop-color="#a545ff" stop-opacity=".32"/><stop offset="1" stop-color="#a545ff" stop-opacity="0"/></radialGradient>
              </defs>
              <ellipse cx="260" cy="330" rx="310" ry="390" fill="url(#faceGlow)"/>
              <g fill="none" stroke="#9b43ff" stroke-opacity=".48" stroke-width="1.4"><path d="M-130 92C85 42 112-72 365-98"/><path d="M-130 115C88 68 130-43 404-72"/><path d="M-130 139C95 95 147-8 444-44"/><path d="M-130 164C107 121 164 26 486-16"/><path d="M-130 190C116 148 184 62 526 15"/><path d="M-130 217C127 176 200 98 563 48"/><path d="M-130 244C142 204 218 137 599 81"/><path d="M-130 271C155 233 237 175 635 114"/></g>
              <g fill="none" stroke="#8f57ff" stroke-opacity=".36" stroke-width="1"><path d="M88 90C166 85 218 119 245 182C270 242 239 280 254 328C270 378 317 399 310 430C304 453 273 458 274 478C275 503 309 511 297 535C290 550 264 552 261 569C257 589 282 604 270 621C255 643 216 636 200 665C179 704 184 758 162 795"/><path d="M116 112C188 112 227 146 244 202M105 150C178 147 221 178 247 225M96 190C173 183 219 211 249 253M91 233C168 221 219 241 253 278M91 277C168 258 218 272 255 306M100 322C173 296 224 302 260 333M118 365C187 330 235 333 269 358M137 408C199 371 246 367 278 386M150 454C210 415 252 409 288 418M153 503C215 463 262 453 293 456M151 552C211 512 253 497 286 495M145 602C201 563 244 543 272 534M136 653C187 618 225 590 260 572M126 705C171 672 206 643 243 615"/><path d="M113 120L96 190L91 277L100 322L118 365L137 408L150 454L153 503L151 552L145 602L136 653L126 705M244 202L247 253L255 306L260 333L269 358L278 386L288 418L293 456L286 495L272 534L260 572L243 615M178 147L168 221L173 296L187 330L199 371L210 415L215 463L211 512L201 563L187 618L171 672"/></g>
              <g fill="#a77aff" fill-opacity=".75"><circle cx="113" cy="120" r="2"/><circle cx="96" cy="190" r="2"/><circle cx="91" cy="277" r="2"/><circle cx="100" cy="322" r="2"/><circle cx="118" cy="365" r="2"/><circle cx="137" cy="408" r="2"/><circle cx="150" cy="454" r="2"/><circle cx="153" cy="503" r="2"/><circle cx="151" cy="552" r="2"/><circle cx="145" cy="602" r="2"/><circle cx="136" cy="653" r="2"/><circle cx="126" cy="705" r="2"/><circle cx="244" cy="202" r="2"/><circle cx="247" cy="253" r="2"/><circle cx="255" cy="306" r="2"/><circle cx="260" cy="333" r="2"/><circle cx="269" cy="358" r="2"/><circle cx="278" cy="386" r="2"/><circle cx="288" cy="418" r="2"/><circle cx="293" cy="456" r="2"/><circle cx="286" cy="495" r="2"/><circle cx="272" cy="534" r="2"/><circle cx="260" cy="572" r="2"/></g>
              <path d="M0 812C178 813 247 934 424 919C575 906 620 842 786 861C946 879 1017 956 1179 921C1362 881 1445 758 1680 813V950H0Z" fill="url(#bottomWaveA)"/><path d="M0 864C206 930 321 838 473 851C655 867 711 958 911 943C1103 929 1252 842 1393 825C1518 811 1602 840 1680 866V950H0Z" fill="url(#bottomWaveB)" opacity=".8"/>
              <g fill="none" stroke="#9f43ff" stroke-opacity=".48" stroke-width="2"><path d="M0 824C190 840 255 935 428 912C586 891 628 838 784 859C948 881 1018 953 1181 914C1367 870 1445 757 1680 811"/><path d="M0 899C195 947 328 867 475 871C646 875 728 955 910 933C1108 910 1254 825 1395 811C1524 798 1610 827 1680 849"/></g>
            </svg>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="auth-welcome-shell"):
            st.markdown(
                """
                <div class="auth-welcome-hero">
                  <svg class="auth-welcome-mark" viewBox="0 0 100 100" fill="none" aria-hidden="true">
                    <path d="M17 35V20a5 5 0 0 1 5-5h15M63 15h15a5 5 0 0 1 5 5v15M83 65v15a5 5 0 0 1-5 5H63M37 85H22a5 5 0 0 1-5-5V65" stroke="#bd5dff" stroke-width="5" stroke-linecap="round"/>
                    <path d="M49 27c-13 0-23 10-23 24v6c0 13 10 23 23 23V27Z" fill="#a450f2" stroke="#eee3ff" stroke-width="3"/><path d="M51 27c13 0 23 10 23 24v6c0 13-10 23-23 23V27Z" fill="#d8b6ff" stroke="#eee3ff" stroke-width="3"/>
                    <path d="M37 47c4-3 8-3 12 0m-13 12c4 3 8 3 12 0m5-12c4-3 8-3 12 0m-12 12c4 3 8 3 12 0" stroke="#f7f0ff" stroke-width="2.4" stroke-linecap="round"/><path d="M50 25v57" stroke="#7e36d8" stroke-width="2"/>
                  </svg>
                  <h1 class="auth-welcome-title">AGE AND GENDER AI</h1>
                  <p class="auth-welcome-subtitle">AI Powered Human Analysis System</p>
                  <p class="auth-welcome-description">Analyze age, gender, and facial emotions with deep learning.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            guest_column, divider_column, account_column = st.columns([1, 0.10, 1], gap="small", vertical_alignment="center")
            with guest_column:
                with st.container(key="auth-guest-card"):
                    st.markdown('<svg class="auth-option-icon" viewBox="0 0 64 64" fill="none" aria-hidden="true"><circle cx="25" cy="21" r="10" stroke="currentColor" stroke-width="3"/><circle cx="43" cy="25" r="8" stroke="currentColor" stroke-width="3"/><path d="M6 53c1-11 8-18 19-18s18 7 19 18H6Zm34-15c10-2 17 4 18 15H48" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg><h2 class="auth-option-title">Continue as Guest</h2><p class="auth-option-copy">Try the application without creating an account.</p>', unsafe_allow_html=True)
                    if st.button("Continue as Guest", type="primary", use_container_width=True, key="auth_continue_guest"):
                        _start_guest_session()
            with divider_column:
                st.markdown('<div class="auth-or-divider"><span>OR</span></div>', unsafe_allow_html=True)
            with account_column:
                with st.container(key="auth-account-card"):
                    st.markdown('<svg class="auth-option-icon" viewBox="0 0 64 64" fill="none" aria-hidden="true"><circle cx="28" cy="20" r="10" stroke="currentColor" stroke-width="3"/><path d="M9 54c1-13 8-20 19-20s18 7 19 20H9Z" stroke="currentColor" stroke-width="3" stroke-linejoin="round"/><path d="M49 18v18m-9-9h18" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/></svg><h2 class="auth-option-title">Create Account</h2><p class="auth-option-copy">Register to save your analysis history and access personalized features.</p>', unsafe_allow_html=True)
                    if st.button("Create Account", type="primary", use_container_width=True, key="auth_open_register"):
                        st.session_state["auth_entry_mode"] = "register"
                        st.rerun()
            st.markdown('<div class="auth-signin-prompt">Already have an account?</div>', unsafe_allow_html=True)
            with st.container(key="auth-signin-button"):
                if st.button("Sign In", icon=":material/login:", use_container_width=True, key="auth_open_login"):
                    st.session_state["auth_entry_mode"] = "signin"
                    st.rerun()
    elif mode == "register":
        register_dark = theme_preference != "Light"
        form_bg = "rgba(24, 17, 55, .82)" if register_dark else "rgba(255, 255, 255, .90)"
        form_text = "#f5f0ff" if register_dark else "#17203f"
        form_muted = "#b8a9d8" if register_dark else "#64708b"
        input_bg = "#332a50" if register_dark else "#eff1fa"
        input_border = "rgba(184, 155, 244, .45)" if register_dark else "rgba(113, 81, 190, .30)"
        svg_face = "#d9c4ff" if register_dark else "#6540ad"
        register_icon_color = "%23d9c4ff" if register_dark else "%236540ad"
        st.markdown(
            f"""
            <style>
            .stApp:has(.st-key-auth-register-page) {{
                background:
                    radial-gradient(ellipse at 84% 12%, rgba(118,37,228,.38), transparent 34%),
                    radial-gradient(ellipse at 50% 58%, rgba(79,27,159,.22), transparent 48%),
                    {"linear-gradient(118deg,#100b2b 0%,#160d37 48%,#1b0b43 100%)" if register_dark else "linear-gradient(118deg,#f0eaff 0%,#f8f6ff 52%,#e9e7ff 100%)"} !important;
            }}
            .stApp:has(.st-key-auth-register-page) [data-testid="stHeader"],
            .stApp:has(.st-key-auth-register-page) [data-testid="stToolbar"],
            .stApp:has(.st-key-auth-register-page) [data-testid="stSidebar"] {{ display:none !important; }}
            .stApp:has(.st-key-auth-register-page) .main .block-container {{
                width:100%; max-width:none; min-height:100vh; padding:24px 18px;
                display:flex; align-items:center; justify-content:center; position:relative;
            }}
            .auth-register-bg {{ position:fixed; inset:0; z-index:0; width:100vw; height:100vh; pointer-events:none; }}
            .st-key-auth-register-page {{
                position:relative; z-index:1; width:min(100%,820px); margin:auto;
                box-sizing:border-box; padding:26px clamp(24px,3.25vw,52px);
                color:{form_text}; background:{form_bg};
                border:1px solid {input_border}; border-radius:28px;
                box-shadow:0 20px 60px rgba(12,7,35,.28), inset 0 0 30px rgba(137,75,230,.07);
                backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); transform:translateY(-84px);
            }}
            .auth-register-header {{ text-align:center; margin:0 auto 14px; color:{form_text}; }}
            .auth-register-logo {{ display:block; width:68px; height:68px; margin:0 auto 8px; filter:drop-shadow(0 0 10px rgba(171,91,255,.5)); }}
            .auth-register-title {{ margin:0; color:{form_text}; font-size:clamp(30px,4vw,42px); line-height:1.08; font-weight:850; letter-spacing:.01em; }}
            .auth-register-subtitle {{ display:flex; align-items:center; gap:16px; margin:14px 0 0; color:{form_muted}; font-size:19px; line-height:1.3; }}
            .auth-register-subtitle::before,.auth-register-subtitle::after {{ content:""; height:1px; flex:1; background:linear-gradient(90deg,transparent,#a04aff 40%,#7041d7); }}
            .auth-register-subtitle::after {{ transform:scaleX(-1); }}
            .st-key-auth-register-page form {{ border:0 !important; padding:0 !important; background:transparent !important; }}
            .st-key-auth-register-page form [data-testid="stVerticalBlock"] {{ gap:.62rem !important; }}
            .st-key-auth-register-page [data-testid="stWidgetLabel"] {{ color:{form_text} !important; font-weight:650; }}
            .st-key-auth-register-page [data-testid="stTextInput"] {{ margin-bottom:4px; }}
            .st-key-auth-register-page [data-testid="stTextInput"] [data-baseweb="input"] {{
                min-height:54px; border:1px solid {input_border} !important; border-radius:14px !important;
                background:{input_bg} !important; box-shadow:none !important; transition:border-color .16s ease,box-shadow .16s ease;
            }}
            .st-key-auth-register-page [data-testid="stTextInput"] div:has(> input),
            .st-key-auth-register-page [data-testid="stTextInput"] div:has(> button) {{ background:{input_bg} !important; }}
            .st-key-auth-register-page [data-testid="stTextInput"] div:has(> input) {{
                border:1px solid {input_border} !important; border-radius:14px !important;
                box-shadow:none !important;
            }}
            .st-key-auth-register-page [data-testid="stTextInput"] div:has(> input):focus-within {{
                border-color:#a85bff !important; box-shadow:0 0 0 3px rgba(145,74,237,.18) !important;
            }}
            .st-key-auth-register-page [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{
                border-color:#a85bff !important; box-shadow:0 0 0 3px rgba(145,74,237,.18),0 0 18px rgba(145,74,237,.16) !important;
            }}
            .st-key-auth-register-page [data-testid="stTextInput"] input {{
                min-height:52px; color:{form_text} !important; background-color:{input_bg} !important;
                border:0 !important; border-radius:13px !important; caret-color:#b86aff; font-size:15px;
            }}
            .st-key-auth-register-page [data-testid="stTextInput"] input::placeholder {{ color:{form_muted} !important; opacity:.82; }}
            .st-key-auth-register-page [data-testid="stTextInput"] button {{ color:{form_text} !important; background:{input_bg} !important; }}
            .st-key-auth-register-page [data-testid="stWidgetLabel"] {{ display:flex; align-items:center; gap:8px; }}
            .st-key-auth-register-full-name [data-testid="stWidgetLabel"]::before,
            .st-key-auth-register-email [data-testid="stWidgetLabel"]::before,
            .st-key-auth-register-password [data-testid="stWidgetLabel"]::before,
            .st-key-auth-register-confirm-password [data-testid="stWidgetLabel"]::before {{
                content:""; display:inline-block; flex:0 0 18px; width:18px; height:18px;
                background-repeat:no-repeat; background-position:center; background-size:18px;
            }}
            .st-key-auth-register-full-name [data-testid="stWidgetLabel"]::before,
            .st-key-auth-register-full-name input {{
                background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{register_icon_color}' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4 21a8 8 0 0 1 16 0'/%3E%3C/svg%3E");
            }}
            .st-key-auth-register-email [data-testid="stWidgetLabel"]::before,
            .st-key-auth-register-email input {{
                background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{register_icon_color}' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='5' width='18' height='14' rx='2'/%3E%3Cpath d='m3 7 9 6 9-6'/%3E%3C/svg%3E");
            }}
            .st-key-auth-register-password [data-testid="stWidgetLabel"]::before,
            .st-key-auth-register-password input,
            .st-key-auth-register-confirm-password [data-testid="stWidgetLabel"]::before,
            .st-key-auth-register-confirm-password input {{
                background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{register_icon_color}' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='10' width='16' height='11' rx='2'/%3E%3Cpath d='M8 10V7a4 4 0 0 1 8 0v3'/%3E%3C/svg%3E");
            }}
            .st-key-auth-register-page [data-testid="stTextInput"] input {{ background-repeat:no-repeat; background-position:16px center; background-size:19px; padding-left:48px !important; }}
            .st-key-auth-register-page [data-testid="stFormSubmitButton"] button {{
                min-height:60px; margin-top:4px; border:1px solid rgba(231,196,255,.65); border-radius:15px;
                background:linear-gradient(105deg,#a32ee8 0%,#792bf1 54%,#5535e8 100%);
                color:#fff; font-size:17px !important; font-weight:750;
                box-shadow:0 8px 24px rgba(120,43,230,.30),inset 0 1px rgba(255,255,255,.25);
                transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease;
            }}
            .st-key-auth-register-page [data-testid="stFormSubmitButton"] button:hover {{ transform:translateY(-1px); box-shadow:0 11px 30px rgba(139,55,241,.42); border-color:#efd8ff; color:#fff; }}
            .st-key-auth-register-page [data-testid="stFormSubmitButton"] button:disabled {{ opacity:.68; cursor:wait; transform:none; }}
            .st-key-auth-register-page .st-key-auth-register-back {{ margin-top:8px; }}
            .st-key-auth-register-page .st-key-auth-register-back button {{
                min-height:48px; padding:0 20px; border:1px solid {input_border}; border-radius:13px;
                background:rgba(113,91,158,.16); color:{form_text}; font-weight:650;
            }}
            .st-key-auth-register-page .st-key-auth-register-back button:hover {{ border-color:#a976ee; background:rgba(137,74,214,.20); color:{form_text}; }}
            @media (max-width:600px) {{
                .stApp:has(.st-key-auth-register-page) .main .block-container {{ padding:16px 12px; align-items:flex-start; }}
                .st-key-auth-register-page {{ padding:24px 20px; border-radius:22px; margin:10px auto; transform:none; }}
                .auth-register-logo {{ width:64px; height:64px; }}
                .auth-register-title {{ font-size:clamp(27px,8vw,36px); }}
                .auth-register-subtitle {{ font-size:17px; gap:10px; }}
                .auth-register-header {{ margin-bottom:18px; }}
                .auth-register-bg {{ opacity:.68; }}
            }}
            @media (prefers-reduced-motion:reduce) {{ .st-key-auth-register-page * {{ transition:none !important; }} }}
            </style>
            <svg class="auth-register-bg" viewBox="0 0 1680 950" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <linearGradient id="regWave" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#6825a7" stop-opacity=".58"/><stop offset="1" stop-color="#5225c8" stop-opacity=".12"/></linearGradient>
                <radialGradient id="regGlow"><stop stop-color="#a545ff" stop-opacity=".30"/><stop offset="1" stop-color="#a545ff" stop-opacity="0"/></radialGradient>
              </defs>
              <ellipse cx="250" cy="340" rx="320" ry="390" fill="url(#regGlow)"/>
              <g fill="none" stroke="#9b43ff" stroke-opacity=".42" stroke-width="1.4"><path d="M-130 90C85 42 112-72 365-98"/><path d="M-130 120C88 68 130-43 404-72"/><path d="M-130 152C95 95 147-8 444-44"/><path d="M-130 186C107 121 164 26 486-16"/><path d="M-130 222C116 148 184 62 526 15"/><path d="M-130 260C127 176 200 98 563 48"/><path d="M-130 300C142 204 218 137 599 81"/></g>
              <g fill="none" stroke="{svg_face}" stroke-opacity=".32" stroke-width="1"><path d="M82 132C162 84 237 132 257 220C273 286 248 319 279 365C302 398 337 406 322 436C311 456 283 458 283 477C283 499 313 507 302 526C292 545 270 542 266 562C260 587 285 606 268 624C246 648 218 632 194 666C170 700 182 746 148 787"/><path d="M111 164C184 135 231 171 248 232M96 205C172 176 228 207 252 260M91 251C169 218 226 244 258 291M103 300C177 261 229 281 267 324M126 351C193 306 245 321 281 360M145 406C209 361 256 371 298 398M150 462C214 415 265 418 311 423M150 520C211 473 264 465 305 457M141 581C198 536 249 516 282 491M128 645C181 604 225 574 265 540M119 703C168 664 208 630 248 589"/><path d="M111 164L96 205L91 251L103 300L126 351L145 406L150 462L150 520L141 581L128 645L119 703M248 232L252 260L258 291L267 324L281 360L298 398L311 423L305 457L282 491L265 540L248 589"/></g>
              <g fill="#b391ff" fill-opacity=".64"><circle cx="111" cy="164" r="2"/><circle cx="96" cy="205" r="2"/><circle cx="91" cy="251" r="2"/><circle cx="103" cy="300" r="2"/><circle cx="126" cy="351" r="2"/><circle cx="145" cy="406" r="2"/><circle cx="150" cy="462" r="2"/><circle cx="150" cy="520" r="2"/><circle cx="141" cy="581" r="2"/><circle cx="128" cy="645" r="2"/><circle cx="119" cy="703" r="2"/><circle cx="248" cy="232" r="2"/><circle cx="258" cy="291" r="2"/><circle cx="281" cy="360" r="2"/><circle cx="311" cy="423" r="2"/><circle cx="282" cy="491" r="2"/><circle cx="248" cy="589" r="2"/></g>
              <path d="M0 814C178 815 247 934 424 919C575 906 620 842 786 861C946 879 1017 956 1179 921C1362 881 1445 758 1680 813V950H0Z" fill="url(#regWave)"/>
              <path d="M0 866C206 930 321 838 473 851C655 867 711 958 911 943C1103 929 1252 842 1393 825C1518 811 1602 840 1680 866" fill="none" stroke="#9f43ff" stroke-opacity=".48" stroke-width="2"/>
            </svg>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="auth-register-page"):
            st.markdown(
                """
                <div class="auth-register-header">
                  <svg class="auth-register-logo" viewBox="0 0 100 100" fill="none" aria-hidden="true">
                    <path d="M17 35V20a5 5 0 0 1 5-5h15M63 15h15a5 5 0 0 1 5 5v15M83 65v15a5 5 0 0 1-5 5H63M37 85H22a5 5 0 0 1-5-5V65" stroke="#bd5dff" stroke-width="5" stroke-linecap="round"/>
                    <path d="M49 27c-13 0-23 10-23 24v6c0 13 10 23 23 23V27Z" fill="#a450f2" stroke="#eee3ff" stroke-width="3"/><path d="M51 27c13 0 23 10 23 24v6c0 13-10 23-23 23V27Z" fill="#d8b6ff" stroke="#eee3ff" stroke-width="3"/>
                    <path d="M37 47c4-3 8-3 12 0m-13 12c4 3 8 3 12 0m5-12c4-3 8-3 12 0m-12 12c4-3 8-3 12 0" stroke="#f7f0ff" stroke-width="2.4" stroke-linecap="round"/>
                  </svg>
                  <h1 class="auth-register-title">AGE AND GENDER AI</h1>
                  <div class="auth-register-subtitle"><span>Create your account</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.pop("auth_notice", None):
                st.success("Account created. Sign in with your new email and password.")
            with st.form("account_registration_form", clear_on_submit=True):
                with st.container(key="auth-register-full-name"):
                    full_name = st.text_input("Full Name", placeholder="Enter your full name", max_chars=120)
                with st.container(key="auth-register-email"):
                    email = st.text_input("Email Address", placeholder="Enter your email address", max_chars=254)
                with st.container(key="auth-register-password"):
                    password = st.text_input("Password (12–256 characters)", placeholder="Enter a password (12–256 characters)", type="password", max_chars=256)
                with st.container(key="auth-register-confirm-password"):
                    confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password", max_chars=256)
                submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True, icon=":material/person_add:")
            if submitted:
                if not full_name.strip():
                    st.error("Enter your full name.")
                else:
                    from services.accounts import DuplicateEmailError, is_valid_email, register_account

                    if not is_valid_email(email):
                        st.error("Enter a valid email address.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif len(password) < 12 or len(password) > 256:
                        st.error("Use a password between 12 and 256 characters.")
                    else:
                        registration_stage = "database_initialization"
                        try:
                            registration_engine = cached_database_engine()
                            registration_stage = "account_insert"
                            register_account(registration_engine, full_name, email, password)
                        except DuplicateEmailError:
                            st.error("An account already exists for this email address.")
                        except Exception as exc:
                            database_cause = getattr(exc, "orig", None)
                            cause_text = str(database_cause or "").casefold()
                            if "no such table: app_users" in cause_text:
                                failure_code = "missing_app_users_table"
                            elif "database is locked" in cause_text:
                                failure_code = "sqlite_busy"
                            elif "unable to open database file" in cause_text:
                                failure_code = "database_unavailable"
                            else:
                                failure_code = "account_persistence_failed"
                            LOGGER.error(
                                "account_registration_failed stage=%s error_type=%s failure_code=%s",
                                registration_stage,
                                type(exc).__name__,
                                failure_code,
                            )
                            st.error("Account registration is temporarily unavailable. Please try again.")
                        else:
                            st.session_state["auth_notice"] = True
                            st.session_state["auth_entry_mode"] = "signin"
                            st.rerun()
            with st.container(key="auth-register-back"):
                if st.button("←  Back", key="auth_register_back"):
                    st.session_state["auth_entry_mode"] = "welcome"
                    st.rerun()
    else:
        st.markdown(
            """
            <style>
            .stApp:has(.st-key-auth-signin-page) {
                background:
                    radial-gradient(ellipse at 89% 8%, rgba(121,39,224,.46), transparent 34%),
                    radial-gradient(ellipse at 50% 60%, rgba(67,31,151,.29), transparent 48%),
                    linear-gradient(118deg,#100b2b 0%,#160d37 50%,#1b0b43 100%) !important;
            }
            .stApp:has(.st-key-auth-signin-page) [data-testid="stHeader"],
            .stApp:has(.st-key-auth-signin-page) [data-testid="stToolbar"],
            .stApp:has(.st-key-auth-signin-page) [data-testid="stSidebar"] { display:none !important; }
            .stApp:has(.st-key-auth-signin-page) .main .block-container {
                max-width:1180px; min-height:100vh; padding:28px 24px 42px;
                display:flex; flex-direction:column; justify-content:center; position:relative; z-index:1;
            }
            .auth-signin-bg { position:fixed; inset:0; width:100vw; height:100vh; z-index:0; pointer-events:none; }
            .st-key-auth-signin-page {
                box-sizing:border-box; width:min(100%,890px); margin:0 auto; padding:30px clamp(28px,4vw,64px) 28px;
                color:#f7f2ff; border:1.5px solid transparent; border-radius:30px;
                background:linear-gradient(145deg,rgba(28,21,59,.94),rgba(17,14,48,.91)) padding-box,
                    linear-gradient(125deg,#ba52fa,#6843db 56%,#9e36ed) border-box;
                box-shadow:0 22px 66px rgba(5,3,24,.36),0 0 34px rgba(116,51,217,.13),inset 0 1px rgba(255,255,255,.06);
                backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
            }
            .auth-signin-header { text-align:center; margin:0 auto 27px; }
            .auth-signin-logo { display:block; width:82px; height:82px; margin:0 auto 6px; filter:drop-shadow(0 0 12px rgba(184,88,255,.55)); }
            .auth-signin-title { margin:0; color:#fbf9ff; font-size:clamp(32px,4.1vw,52px); line-height:1.08; letter-spacing:.018em; font-weight:850; text-align:center; }
            .auth-signin-subtitle { display:flex; align-items:center; justify-content:center; gap:22px; margin:15px 0 0; color:#ddd0f6; font-size:clamp(17px,2vw,24px); font-weight:500; }
            .auth-signin-subtitle:before,.auth-signin-subtitle:after { content:""; display:block; width:min(18vw,175px); height:1px; background:linear-gradient(90deg,transparent,#963df1); }
            .auth-signin-subtitle:after { background:linear-gradient(90deg,#963df1,transparent); }
            .st-key-auth-signin-email-label,.st-key-auth-signin-password-label { margin:0 0 7px; color:#D8C7FF; font-size:16px; font-weight:650; }
            .st-key-auth-signin-email-label p,.st-key-auth-signin-password-label p { margin:0; }
            .st-key-auth-signin-email-label svg,.st-key-auth-signin-password-label svg { width:21px; height:21px; vertical-align:-5px; margin-right:10px; color:#f0e8ff; }
            .st-key-auth-signin-email [data-testid="stTextInput"] input,
            .st-key-auth-signin-password [data-testid="stTextInput"] input {
                width:100%; height:54px; min-height:54px; box-sizing:border-box; border:0 !important; border-radius:0;
                background-color:#292044 !important; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;
                opacity:1 !important; caret-color:#FFFFFF !important; font-size:17px; padding:10px 16px 10px 58px;
                box-shadow:none !important;
            }
            .st-key-auth-signin-email [data-testid="stTextInput"] input:-webkit-autofill,
            .st-key-auth-signin-email [data-testid="stTextInput"] input:-webkit-autofill:hover,
            .st-key-auth-signin-email [data-testid="stTextInput"] input:-webkit-autofill:focus,
            .st-key-auth-signin-password [data-testid="stTextInput"] input:-webkit-autofill,
            .st-key-auth-signin-password [data-testid="stTextInput"] input:-webkit-autofill:hover,
            .st-key-auth-signin-password [data-testid="stTextInput"] input:-webkit-autofill:focus {
                -webkit-text-fill-color:#FFFFFF !important; box-shadow:0 0 0 1000px #292044 inset !important;
                caret-color:#FFFFFF !important; opacity:1 !important;
            }
            .st-key-auth-signin-email [data-baseweb="input"],
            .st-key-auth-signin-password [data-baseweb="input"] {
                width:100%; height:56px; min-height:56px; box-sizing:border-box; overflow:hidden;
                border:1px solid #7661A8 !important; border-radius:14px !important;
                background:#292044 !important; box-shadow:none !important;
                transition:border-color .16s ease,box-shadow .16s ease;
            }
            .st-key-auth-signin-email [data-baseweb="input"]:focus-within,
            .st-key-auth-signin-password [data-baseweb="input"]:focus-within { border-color:#B660FF !important; box-shadow:0 0 0 2px rgba(153,74,238,.23),0 0 16px rgba(151,63,242,.18) !important; }
            .st-key-auth-signin-email [data-testid="stTextInput"] input {
                background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23D8C7FF' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='5' width='18' height='14' rx='2'/%3E%3Cpath d='m3 7 9 6 9-6'/%3E%3C/svg%3E");
                background-repeat:no-repeat; background-position:24px center; background-size:24px;
            }
            .st-key-auth-signin-password [data-testid="stTextInput"] input {
                background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23D8C7FF' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='10' width='16' height='11' rx='2'/%3E%3Cpath d='M8 10V7a4 4 0 0 1 8 0v3'/%3E%3C/svg%3E");
                background-repeat:no-repeat; background-position:20px center; background-size:22px; padding-right:58px;
            }
            .st-key-auth-signin-password [data-testid="stTextInput"] button[aria-label*="password" i],
            .st-key-auth-signin-password [data-testid="stTextInput"] button[title*="password" i] { display:none !important; }
            .st-key-auth-signin-email [data-testid="stTextInput"] input::placeholder,
            .st-key-auth-signin-password [data-testid="stTextInput"] input::placeholder { color:#B9ACD8 !important; opacity:1; }
            .st-key-auth-signin-email [data-testid="stWidgetLabel"],.st-key-auth-signin-password [data-testid="stWidgetLabel"] { display:none; }
            .st-key-auth-signin-password-row { position:relative; width:100%; }
            .st-key-auth-signin-password-row .st-key-auth-signin-password,
            .st-key-auth-signin-email,.st-key-auth-signin-password { width:100%; }
            .st-key-auth-signin-password-row .st-key-auth-signin-visibility {
                position:absolute; z-index:4; right:5px; top:7px; width:42px; height:42px; margin:0;
                display:flex; align-items:center; justify-content:center;
            }
            .st-key-auth-signin-visibility [data-testid="stButton"] { width:42px; height:42px; margin:0 !important; }
            .st-key-auth-signin-visibility [data-testid="stButton"] button {
                display:flex; align-items:center; justify-content:center; width:42px; height:42px; min-height:42px;
                padding:0 !important; margin:0; border:0 !important; border-radius:9px; font-size:0;
                color:#E4D8FA; background:transparent !important; box-shadow:none !important;
            }
            .st-key-auth-signin-visibility [data-testid="stButton"] button span { font-size:22px !important; line-height:1; }
            .st-key-auth-signin-visibility [data-testid="stButton"] button:hover { color:#fff; background:rgba(182,96,255,.12) !important; }
            .st-key-auth-signin-visibility [data-testid="stButton"] button:focus-visible { outline:2px solid #B660FF; outline-offset:-3px; }
            .st-key-auth-signin-submit [data-testid="stButton"] button {
                min-height:68px; margin-top:15px; border:1px solid rgba(226,171,255,.58); border-radius:15px;
                color:#fff; font-size:21px; font-weight:750;
                background:linear-gradient(105deg,#ad38ed 0%,#762bea 57%,#6129e9 100%);
                box-shadow:0 8px 24px rgba(117,37,225,.34),inset 0 1px rgba(255,255,255,.22); transition:filter .16s ease,transform .16s ease,box-shadow .16s ease;
            }
            .st-key-auth-signin-submit [data-testid="stButton"] button:hover { color:#fff; filter:brightness(1.09); transform:translateY(-1px); box-shadow:0 11px 30px rgba(137,48,240,.45); }
            .st-key-auth-signin-back { margin-top:17px; }
            .st-key-auth-signin-back [data-testid="stButton"] button {
                min-height:53px; padding:0 20px; border:1px solid rgba(206,188,240,.58); border-radius:13px;
                color:#f4efff; background:rgba(94,83,128,.27); font-size:17px; font-weight:650;
            }
            .st-key-auth-signin-back [data-testid="stButton"] button:hover { color:#fff; border-color:#c79bff; background:rgba(118,67,181,.38); }
            @media(max-width:600px) {
                .stApp:has(.st-key-auth-signin-page) .main .block-container { padding:16px 12px 24px; justify-content:center; }
                .st-key-auth-signin-page { padding:25px 19px 22px; border-radius:23px; }
                .auth-signin-logo { width:68px; height:68px; }
                .auth-signin-header { margin-bottom:20px; }
                .auth-signin-title { font-size:clamp(25px,8vw,36px); }
                .auth-signin-subtitle { gap:10px; font-size:17px; }
                .auth-signin-subtitle:before,.auth-signin-subtitle:after { width:min(12vw,55px); }
                .st-key-auth-signin-email [data-testid="stTextInput"] input,.st-key-auth-signin-password [data-testid="stTextInput"] input { height:54px; min-height:54px; font-size:15px; padding-left:50px; }
                .st-key-auth-signin-email [data-testid="stTextInput"] input,.st-key-auth-signin-password [data-testid="stTextInput"] input { background-position:17px center; background-size:21px; }
                .st-key-auth-signin-password-row .st-key-auth-signin-visibility { right:3px; top:7px; }
                .st-key-auth-signin-submit [data-testid="stButton"] button { min-height:60px; font-size:18px; }
            }
            @media(prefers-reduced-motion:reduce) { .st-key-auth-signin-page * { transition:none !important; } }
            </style>
            <svg class="auth-signin-bg" viewBox="0 0 1680 950" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <radialGradient id="signinGlow"><stop stop-color="#9d45ff" stop-opacity=".28"/><stop offset="1" stop-color="#9d45ff" stop-opacity="0"/></radialGradient>
                <linearGradient id="signinWave" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#6925a9" stop-opacity=".58"/><stop offset="1" stop-color="#5424c9" stop-opacity=".14"/></linearGradient>
                <linearGradient id="signinWaveLine" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#9f43ff" stop-opacity=".22"/><stop offset=".7" stop-color="#a338f5" stop-opacity=".68"/><stop offset="1" stop-color="#9f43ff" stop-opacity=".30"/></linearGradient>
              </defs>
              <ellipse cx="245" cy="350" rx="315" ry="410" fill="url(#signinGlow)"/>
              <g fill="none" stroke="#9b43ff" stroke-opacity=".5" stroke-width="1.5"><path d="M-130 68C75 37 140-88 370-92"/><path d="M-130 91C87 58 151-62 405-67"/><path d="M-130 115C99 79 166-34 442-40"/><path d="M-130 140C111 100 180-4 480-13"/><path d="M-130 166C124 121 195 27 521 15"/><path d="M-130 193C136 144 210 59 559 44"/><path d="M-130 221C148 168 226 91 597 74"/></g>
              <g fill="none" stroke="#9564ff" stroke-opacity=".43" stroke-width="1"><path d="M80 150C157 101 235 131 258 218C275 280 246 317 276 359C299 392 333 402 320 428C311 447 282 456 283 476C285 497 310 507 300 526C292 542 268 543 264 562C258 584 281 601 267 621C251 643 220 635 196 661C171 689 178 741 150 782"/><path d="M110 177C182 139 229 170 247 228M98 218C174 177 225 209 251 258M94 261C170 216 226 248 258 293M106 306C179 262 234 285 267 326M126 352C193 307 244 324 282 361M143 402C207 358 259 374 299 400M151 455C215 414 267 422 311 425M151 509C215 463 269 464 306 456M144 567C205 518 259 508 290 492M132 626C190 580 236 554 272 530M119 686C173 638 214 610 251 578"/><path d="M110 177L98 218L94 261L106 306L126 352L143 402L151 455L151 509L144 567L132 626L119 686M247 228L251 258L258 293L267 326L282 361L299 400L311 425L306 456L290 492L272 530L251 578"/></g>
              <g fill="#ab7bff" fill-opacity=".72"><circle cx="110" cy="177" r="2"/><circle cx="98" cy="218" r="2"/><circle cx="94" cy="261" r="2"/><circle cx="106" cy="306" r="2"/><circle cx="126" cy="352" r="2"/><circle cx="143" cy="402" r="2"/><circle cx="151" cy="455" r="2"/><circle cx="151" cy="509" r="2"/><circle cx="144" cy="567" r="2"/><circle cx="132" cy="626" r="2"/><circle cx="119" cy="686" r="2"/><circle cx="247" cy="228" r="2"/><circle cx="258" cy="293" r="2"/><circle cx="282" cy="361" r="2"/><circle cx="311" cy="425" r="2"/><circle cx="290" cy="492" r="2"/><circle cx="272" cy="530" r="2"/></g>
              <path d="M0 811C178 812 247 934 424 919C575 906 620 842 786 861C946 879 1017 956 1179 921C1362 881 1445 758 1680 813V950H0Z" fill="url(#signinWave)"/>
              <path d="M0 864C206 930 321 838 473 851C655 867 711 958 911 943C1103 929 1252 842 1393 825C1518 811 1602 840 1680 866" fill="none" stroke="url(#signinWaveLine)" stroke-width="2"/>
              <path d="M0 901C195 947 328 867 475 871C646 875 728 955 910 933C1108 910 1254 825 1395 811C1524 798 1610 827 1680 849" fill="none" stroke="url(#signinWaveLine)" stroke-width="1.6"/>
            </svg>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="auth-signin-page"):
            st.markdown(
                """
                <div class="auth-signin-header">
                  <svg class="auth-signin-logo" viewBox="0 0 100 100" fill="none" aria-hidden="true">
                    <path d="M17 35V20a5 5 0 0 1 5-5h15M63 15h15a5 5 0 0 1 5 5v15M83 65v15a5 5 0 0 1-5 5H63M37 85H22a5 5 0 0 1-5-5V65" stroke="#bd5dff" stroke-width="5" stroke-linecap="round"/>
                    <path d="M49 27c-13 0-23 10-23 24v6c0 13 10 23 23 23V27Z" fill="#a450f2" stroke="#eee3ff" stroke-width="3"/><path d="M51 27c13 0 23 10 23 24v6c0 13-10 23-23 23V27Z" fill="#d8b6ff" stroke="#eee3ff" stroke-width="3"/>
                    <path d="M37 47c4-3 8-3 12 0m-13 12c4 3 8 3 12 0m5-12c4-3 8-3 12 0m-12 12c4 3 8 3 12 0" stroke="#f7f0ff" stroke-width="2.4" stroke-linecap="round"/><path d="M50 25v57" stroke="#7e36d8" stroke-width="2"/>
                  </svg>
                  <h1 class="auth-signin-title">AGE AND GENDER AI</h1>
                  <div class="auth-signin-subtitle"><span>Sign In to Your Account</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.pop("auth_notice", None):
                st.success("Account created. You can now sign in.")
            with st.container(key="auth-signin-email-label"):
                st.markdown('<p><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>Email Address</p>', unsafe_allow_html=True)
            with st.container(key="auth-signin-email"):
                email = st.text_input("Email Address", placeholder="Enter your email address", max_chars=254, key="auth_signin_email", label_visibility="collapsed")
            with st.container(key="auth-signin-password-label"):
                st.markdown('<p><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>Password</p>', unsafe_allow_html=True)
            with st.container(key="auth-signin-password-row"):
                password_visible = bool(st.session_state.get("auth_signin_password_visible", False))
                with st.container(key="auth-signin-password"):
                    password = st.text_input("Password", placeholder="Enter your password", type="default" if password_visible else "password", key="auth_signin_password", label_visibility="collapsed")
                with st.container(key="auth-signin-visibility"):
                    if st.button(" ", icon=":material/visibility_off:" if password_visible else ":material/visibility:", help="Hide password" if password_visible else "Show password", key="auth_signin_toggle_password"):
                        st.session_state["auth_signin_password_visible"] = not password_visible
                        st.rerun()
            if st.button("Sign In", type="primary", use_container_width=True, icon=":material/login:", key="auth-signin-submit"):
                if not email.strip() or not password:
                    st.error("Enter your email address and password.")
                else:
                    from services.accounts import authenticate_account

                    try:
                        with st.spinner("Signing in…"):
                            account = authenticate_account(cached_database_engine(), email, password)
                    except Exception as exc:
                        LOGGER.error("account_signin_failed error_type=%s", type(exc).__name__)
                        st.error("Sign-in is temporarily unavailable. Please try again.")
                    else:
                        if account:
                            _start_account_session(account)
                        st.error("Email or password is incorrect.")
            with st.container(key="auth-signin-back"):
                if st.button("Back", icon=":material/arrow_back:", key="auth_signin_back"):
                    st.session_state["auth_entry_mode"] = "welcome"
                    st.rerun()
    st.stop()


CONSENT_VERSION = "2026-10-31-v1"


def _load_dashboard_dependencies():
    """Import analysis and UI libraries only after a user enters the dashboard."""
    global alt, cv2, np, pd, create_engine, text
    global Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
    global load_emotion_assets, EmotionPredictor, EMOTION_TTA5
    global decode_image, ImageValidationError, image_quality, face_quality, prediction_uncertain
    global TemporalFaceTracker, run_image_analysis, result_rows, csv_bytes, json_bytes, pdf_report
    global predict_batch_emotions

    import altair as alt
    import cv2
    import numpy as np
    import pandas as pd
    from sqlalchemy import create_engine, text
    from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
    from emotion.model_loader import load_emotion_assets
    from emotion.predictor import EmotionPredictor
    from emotion.tta import TTA5 as EMOTION_TTA5
    from services.uploads import decode_image, ImageValidationError
    from services.quality import image_quality, face_quality, prediction_uncertain
    from services.tracking import TemporalFaceTracker
    from services.analysis import analyze_image_bytes as run_image_analysis
    from services.exports import result_rows, csv_bytes, json_bytes, pdf_report
    from services.batch_emotions import predict_batch_emotions

def consent_is_recorded():
    if IS_GUEST:
        return bool(st.session_state.get("guest_consent_accepted"))
    if not CURRENT_USER_ID:
        return False
    consent_cache_key = (CURRENT_USER_ID, CONSENT_VERSION)
    if st.session_state.get("_consent_cache_key") == consent_cache_key:
        return True
    try:
        with cached_database_engine().connect() as connection:
            row = connection.execute(text("SELECT consent_version FROM user_consent WHERE user_id=:user_id"), {"user_id": CURRENT_USER_ID}).first()
        accepted = bool(row and row[0] == CONSENT_VERSION)
        if accepted:
            st.session_state["_consent_cache_key"] = consent_cache_key
        return accepted
    except Exception:
        return False

def record_consent():
    accepted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if IS_GUEST:
        st.session_state["guest_consent_accepted"] = True
    elif CURRENT_USER_ID:
        with cached_database_engine().begin() as connection:
            connection.execute(text("INSERT INTO user_consent(user_id, consent_version, accepted_at) VALUES(:user_id,:version,:accepted) ON CONFLICT(user_id) DO UPDATE SET consent_version=:version, accepted_at=:accepted"), {"user_id": CURRENT_USER_ID, "version": CONSENT_VERSION, "accepted": accepted_at})
        st.session_state["_consent_cache_key"] = (CURRENT_USER_ID, CONSENT_VERSION)

def enforce_consent_gate():
    if consent_is_recorded():
        return
    # Keep a compact inline fallback for an existing session whose consent
    # could not be recorded; never route users to a standalone consent page.
    history_notice = "Guest history stays only in this session." if IS_GUEST else "Prediction metadata is saved only to your registered account."
    st.info(f"Age, gender, and expression results are estimates. Only use images you have permission to analyze. {history_notice} Uploaded images and camera frames are not stored.")
    accepted = st.checkbox("I understand", key="consent_check")
    if st.button("Continue", type="primary", disabled=not accepted, key="consent_continue"):
        try:
            record_consent()
            st.rerun()
        except Exception:
            st.error("Your confirmation could not be saved. Please try again.")
    st.stop()

def get_retention_days():
    if not CURRENT_USER_ID:
        return APP_CONFIG.retention_days
    retention_cache = st.session_state.get("_retention_days_cache")
    if retention_cache and retention_cache[0] == CURRENT_USER_ID:
        return int(retention_cache[1])
    try:
        with cached_database_engine().connect() as connection:
            row = connection.execute(text("SELECT retention_days FROM user_preferences WHERE user_id=:user_id"), {"user_id": CURRENT_USER_ID}).first()
        days = int(row[0]) if row else APP_CONFIG.retention_days
        st.session_state["_retention_days_cache"] = (CURRENT_USER_ID, days)
        return days
    except Exception:
        return APP_CONFIG.retention_days

def set_retention_days(days):
    days = int(days)
    if days not in (0, 30, 90) or not CURRENT_USER_ID:
        return
    with cached_database_engine().begin() as connection:
        connection.execute(text("INSERT INTO user_preferences(user_id,retention_days) VALUES(:user_id,:days) ON CONFLICT(user_id) DO UPDATE SET retention_days=:days"), {"user_id": CURRENT_USER_ID, "days": days})
    st.session_state["_retention_days_cache"] = (CURRENT_USER_ID, days)
    apply_retention_cleanup(force=True)

def apply_retention_cleanup(force=False):
    days = get_retention_days()
    if days <= 0 or not CURRENT_USER_ID:
        return
    cleanup_day = datetime.now(timezone.utc).date().isoformat()
    cleanup_marker = (CURRENT_USER_ID, days, cleanup_day)
    if not force and st.session_state.get("_retention_cleanup_marker") == cleanup_marker:
        return
    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat(timespec="seconds")
    with cached_database_engine().begin() as connection:
        connection.execute(text("DELETE FROM prediction_history WHERE user_id=:user_id AND created_at < :cutoff"), {"user_id": CURRENT_USER_ID, "cutoff": cutoff})
        connection.execute(text("DELETE FROM emotion_prediction_history WHERE user_id=:user_id AND created_at < :cutoff"), {"user_id": CURRENT_USER_ID, "cutoff": cutoff})
    st.session_state["_retention_cleanup_marker"] = cleanup_marker

def delete_user_account_data():
    if not CURRENT_USER_ID:
        return
    with cached_database_engine().begin() as connection:
        for table in ("prediction_history", "emotion_prediction_history", "user_consent", "user_preferences"):
            connection.execute(text(f"DELETE FROM {table} WHERE user_id=:user_id"), {"user_id": CURRENT_USER_ID})

def delete_history_row(analysis_id, face_index):
    if IS_GUEST:
        st.session_state["guest_prediction_history"] = [
            row for row in st.session_state.get("guest_prediction_history", [])
            if not (row.get("Analysis ID") == analysis_id and int(row.get("Face", 0)) == int(face_index))
        ]
        return
    if not CURRENT_USER_ID:
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
    """Persist for registered users and keep guest history in session state only."""
    if not predictions:
        return None

    seed = image_data or f"{mode}:{filename}:{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
    analysis_id = get_analysis_id(mode, seed)
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

    if IS_GUEST:
        session_rows = [
            {
                "Analysis ID": row["analysis_id"],
                "Timestamp": row["created_at"],
                "Mode": row["mode"],
                "Face": row["face_index"],
                "Gender": row["gender"],
                "Gender Confidence": row["gender_confidence"],
                "Estimated Age": row["estimated_age"],
                "Age Group": row["age_group"],
                "Age Confidence": "N/A" if row["age_confidence"] < 0 else row["age_confidence"],
                "Model": row["model_id"],
                "Model Version": row["model_version"],
                "Uncertain": row["uncertain"],
            }
            for row in rows
        ]
        session_history = st.session_state.setdefault("guest_prediction_history", [])
        session_history.extend(session_rows)
        del session_history[:-1000]
        return analysis_id
    if not CURRENT_USER_ID:
        return None

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
    if IS_GUEST:
        history = pd.DataFrame(
            st.session_state.get("guest_prediction_history", []),
            columns=[
                "Analysis ID", "Timestamp", "Mode", "Face", "Gender", "Gender Confidence",
                "Estimated Age", "Age Group", "Age Confidence", "Model", "Model Version", "Uncertain",
            ],
        )
        return history
    if not CURRENT_USER_ID:
        return pd.DataFrame()
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
    if IS_GUEST:
        st.session_state["guest_prediction_history"] = []
        st.session_state["guest_emotion_history"] = []
        return
    if not CURRENT_USER_ID:
        return
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
    """Persist for registered users and keep guest expression history in-session."""
    if not results:
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
    if IS_GUEST:
        guest_rows = [
            {
                "Analysis ID": row["analysis_id"],
                "Timestamp": row["created_at"],
                "Face": row["face_index"],
                "Emotion": row["emotion"],
                "Confidence (%)": row["confidence"],
                "Inference": row["inference_mode"],
                "Model": row["model_id"],
            }
            for row in rows
        ]
        session_history = st.session_state.setdefault("guest_emotion_history", [])
        session_history.extend(guest_rows)
        del session_history[:-1000]
        return
    if not CURRENT_USER_ID:
        return
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
    if IS_GUEST:
        return pd.DataFrame(
            st.session_state.get("guest_emotion_history", []),
            columns=["Analysis ID", "Timestamp", "Face", "Emotion", "Confidence (%)", "Inference", "Model"],
        )
    if not CURRENT_USER_ID:
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
_render_auth_entry()
if IS_LOGGED_IN:
    from sqlalchemy import text
enforce_consent_gate()
_load_dashboard_dependencies()
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
    pages,
    key="navigation_page",
)

history_view = "History"
analysis_mode = "Single Person"
settings_section = "Appearance"
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
        ["Appearance", "Privacy", "Model"],
        key="anonymous_settings_section",
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
    st.sidebar.caption("Guest Mode · history lasts for this session")
    if st.sidebar.button("Exit Guest Mode", use_container_width=True):
        _return_to_auth_entry()
elif IS_LOGGED_IN:
    st.sidebar.caption(f"Signed in as {st.session_state.get('authenticated_user_name', st.session_state.get('authenticated_user_email', 'Registered user'))}")
    if st.sidebar.button("Log Out", use_container_width=True):
        _return_to_auth_entry()
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
        infer_this_frame = self.frame_number % APP_CONFIG.camera_frame_interval == 0
        if infer_this_frame:
            height, width = rgb.shape[:2]
            scale = min(1.0, APP_CONFIG.camera_max_width / max(width, 1), APP_CONFIG.camera_max_height / max(height, 1))
            small = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
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
            self.last_boxes, self.last_predictions = boxes, predictions
            self.last_status = status
            now = time.perf_counter()
            self.last_fps = 1.0 / max(now - self.last_tick, 1e-6)
            self.last_tick = now
            self.last_latency_ms = (now - started) * 1000.0

        # Draw the latest predictions on every live frame so video stays fluid
        # while inference runs only every configured Nth frame.
        annotated = draw_detections(Image.fromarray(rgb, mode="RGB"), self.last_boxes, self.last_predictions)
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        status = getattr(self, "last_status", "Starting analysis…")
        draw.rectangle((8, 8, min(annotated.width - 8, 340), 36), fill=(20, 24, 39))
        draw.text((14, 14), status, fill=(255, 255, 255), font=font)
        return av.VideoFrame.from_ndarray(np.asarray(annotated, dtype=np.uint8), format="rgb24")


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
        """
        <style>
        /* Keep Analyze upload controls legible in the app's dark theme. */
        [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            background: #20263a !important;
            border: 1px dashed #7180a4 !important;
            border-radius: 12px !important;
            color: #f1f5ff !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
            color: #e2e8f0 !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
            background: #6d3be8 !important;
            border: 1px solid #a78bfa !important;
            color: #ffffff !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover {
            background: #8056f2 !important;
            border-color: #c4b5fd !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] small,
        [data-testid="stAppViewContainer"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
            color: #cbd5e1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
        [data-testid="stAppViewContainer"] .stCaption {
            color: #cbd5e1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    if analysis_mode == "Group Analysis":
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
            batch_emotion_failures = 0
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
                        batch_emotions, batch_emotion_error = predict_batch_emotions(
                            data,
                            batch_boxes,
                            predict_emotions_from_bytes,
                            EMOTION_TTA5,
                        )
                        if batch_emotion_error is not None:
                            LOGGER.error(
                                "batch_expression_inference_failed image_index=%s error_type=%s",
                                batch_index,
                                type(batch_emotion_error).__name__,
                            )
                            batch_emotion_failures += 1

                        for row in result_rows(batch_boxes, batch_predictions, batch_emotions):
                            row["image"] = f"Image {batch_index}"
                            row["status"] = "Uncertain" if row["uncertain"] else "Complete"
                            if batch_truncated:
                                row["status"] += " · face limit reached"
                            if batch_emotion_error is not None:
                                row["status"] += " · expression unavailable"
                            batch_rows.append(row)
                    else:
                        batch_rows.append({"image": f"Image {batch_index}", "status": "No faces detected", "faces": 0})
                    save_history("batch image", "Batch", batch_predictions, image_data=data)
                progress.progress(batch_index / len(batch_files), text=f"Processed {batch_index} of {len(batch_files)} images")
            if batch_emotion_failures:
                st.warning(
                    f"Expression estimates were unavailable for {batch_emotion_failures} image(s). "
                    "Age and gender results are still included."
                )
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

                if analysis_mode == "Group Analysis":
                    age_chart_column, expression_chart_column = st.columns(2, gap="medium", wrap=True)
                    with age_chart_column:
                        with st.container(border=True, height=500):
                            st.subheader("Age-group Distribution")
                            age_chart_groups = age_groups.rename(index=lambda category: str(category).replace("-", "–"))
                            render_distribution_chart(age_chart_groups, ["10–19", "20–29", "30–39"])
                    with expression_chart_column:
                        with st.container(border=True, height=500):
                            st.subheader("Expression Distribution")
                            if include_emotion and not emotion_error and emotion_results:
                                emotion_counts = pd.Series(
                                    [str(item.get("emotion", "Unknown")).title() for item in emotion_results]
                                ).value_counts()
                                render_distribution_chart(
                                    emotion_counts, ["Angry", "Happy", "Neutral", "Sad"]
                                )
                            elif not include_emotion:
                                st.info("Enable facial expression estimates to see expression counts.")
                            elif emotion_error:
                                st.info("Expression counts are unavailable because expression analysis could not finish.")
                            else:
                                st.info("No expression results are available for this image.")
                if include_emotion and not emotion_error and emotion_results:
                    st.caption(
                        f"Expression Model v1 · High-accuracy mode · {emotion_hash[:12]} · "
                        f"{emotion_results[0]['total_latency_ms']:.0f} ms total"
                    )

                annotated = draw_detections(image, predicted_boxes, predictions)

                if analysis_mode == "Single Person":
                    with st.container(key="single-analysis-layout"):
                        st.markdown(
                            """
                            <style>
                            .st-key-single-analysis-layout [data-testid="stImage"] img {
                                display: block;
                                max-width: 100% !important;
                                max-height: 540px !important;
                                width: auto !important;
                                height: auto !important;
                                object-fit: contain !important;
                                border-radius: 12px !important;
                                margin-left: auto;
                                margin-right: auto;
                            }
                            .st-key-single-left-panel,
                            .st-key-single-right-panel {
                                box-sizing: border-box;
                                height: 100%;
                                min-height: 680px;
                                padding: 1rem 1.1rem;
                                border: 1px solid rgba(148, 163, 184, 0.34);
                                border-radius: 16px;
                                background: rgba(21, 26, 45, 0.22);
                            }
                            .st-key-single-image-stage [data-testid="stImage"] {
                                display: flex;
                                min-height: 560px;
                                align-items: center;
                                justify-content: center;
                            }
                            .st-key-single-image-stage [data-testid="stImage"] img {
                                max-width: min(100%, 420px) !important;
                                max-height: 540px !important;
                            }
                            .st-key-single-crop-stage [data-testid="stImage"] {
                                display: flex;
                                justify-content: center;
                            }
                            .st-key-single-crop-stage [data-testid="stImage"] img {
                                max-width: min(100%, 260px) !important;
                                max-height: 260px !important;
                            }
                            .st-key-single-emotion-details {
                                box-sizing: border-box;
                                height: 100%;
                                padding-left: 1rem;
                                border-left: 1px solid rgba(148, 163, 184, 0.38);
                            }
                            @media (max-width: 850px) {
                                .st-key-single-left-panel,
                                .st-key-single-right-panel {
                                    min-height: 0;
                                }
                                .st-key-single-image-stage [data-testid="stImage"] {
                                    min-height: 0;
                                }
                                .st-key-single-emotion-details {
                                    height: auto;
                                    padding: 1rem 0 0;
                                    border-left: 0;
                                    border-top: 1px solid rgba(148, 163, 184, 0.38);
                                }
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        image_column, details_column = st.columns(
                            [1, 1], gap="large", wrap=True
                        )

                        with image_column:
                            with st.container(border=True, key="single-left-panel"):
                                st.subheader("📷 Given Image")
                                # Scale the already-annotated preview uniformly so face
                                # boxes and labels remain aligned with the source image.
                                max_preview_width, max_preview_height = 420, 540
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
                                with st.container(key="single-image-stage"):
                                    st.image(preview, width=preview.width)
                                st.caption("Green boxes show the faces used for prediction.")

                        with details_column:
                            with st.container(border=True, key="single-right-panel"):
                                st.subheader("✂️ Cropped Faces and Prediction Details")
                                st.info(f"Total Faces: {count}")
                                for i, (crop, pred) in enumerate(zip(cropped_faces, predictions)):
                                    crop_scale = min(
                                        1.0,
                                        260 / float(crop.width),
                                        260 / float(crop.height),
                                    )
                                    crop_preview = crop
                                    if crop_scale < 1.0:
                                        crop_preview = crop.resize(
                                            (
                                                max(1, int(round(crop.width * crop_scale))),
                                                max(1, int(round(crop.height * crop_scale))),
                                            ),
                                            Image.Resampling.LANCZOS,
                                        )
                                    with st.container(key="single-crop-stage"):
                                        st.image(crop_preview, width=crop_preview.width)
                                    st.markdown(f"### 👤 Face {i + 1}")

                                    age_gender_column, emotion_column = st.columns(
                                        [1, 1], gap="medium", wrap=True
                                    )
                                    with age_gender_column:
                                        st.markdown("#### Age & Gender")
                                        st.write(f"⚥ **Gender:** {pred['gender']}")
                                        st.write(f"🎯 **Gender Confidence:** {pred['gender_confidence']:.1f}%")
                                        st.write(f"🎂 **Estimated Age:** {pred['age']} years")
                                        st.write(f"👥 **Age Group:** {pred['age_group']}")
                                        st.write(f"📊 **Age Confidence:** {age_confidence_text(pred)}")
                                        if pred.get("uncertain"):
                                            st.warning("Uncertain estimate")
                                        if pred.get("quality_flags"):
                                            st.caption(" · ".join(pred["quality_flags"]))

                                    with emotion_column:
                                        with st.container(key="single-emotion-details"):
                                            st.markdown("#### Facial Emotion")
                                            if not include_emotion:
                                                st.caption("Facial emotion estimation is turned off.")
                                            elif emotion_error:
                                                st.caption("A facial emotion estimate is unavailable for this image.")
                                            elif i < len(emotion_results):
                                                emotion_result = emotion_results[i]
                                                st.markdown(
                                                    f"**🎭 {emotion_result['emotion']}**"
                                                )
                                                st.write(
                                                    f"🎯 **Expression confidence:** {emotion_result['confidence']:.1f}%"
                                                )
                                                probabilities = emotion_result.get("probabilities") or {}
                                                top3 = sorted(
                                                    probabilities.items(),
                                                    key=lambda item: item[1],
                                                    reverse=True,
                                                )[:3]
                                                if top3:
                                                    st.caption(
                                                        "Top expressions: "
                                                        + " · ".join(
                                                            f"{name.title()} {value*100:.1f}%"
                                                            for name, value in top3
                                                        )
                                                    )
                                            else:
                                                st.caption("No facial emotion result is available for this face.")

                                            timing = pred.get("timing_ms", {})
                                            if timing:
                                                st.caption(
                                                    f"Decode {timing.get('decode', 0):.0f} ms · "
                                                    f"detection {timing.get('detection', 0):.0f} ms · "
                                                    f"face inference {timing.get('face_crop_and_prediction', 0):.0f} ms"
                                                )
                else:
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
                    st.subheader("✂️ Face Analysis Results")
                    st.info(f"Total Faces: {count}")
                    card_background = "#ffffff" if theme_preference == "Light" else "#11182b"
                    card_text = "#17203f" if theme_preference == "Light" else "#f8fafc"
                    st.markdown(
                        f"""
                        <style>
                        [class*="st-key-group-face-card-"] {{
                            height: 100%; padding: 14px !important; box-sizing: border-box;
                            background: {card_background} !important; color: {card_text} !important;
                            border: 1px solid rgba(139, 92, 246, .42) !important;
                            border-radius: 14px !important;
                        }}
                        [class*="st-key-group-face-card-"] [data-testid="stImage"] img {{
                            display: block; width: 100% !important; height: 172px !important;
                            object-fit: contain !important; border-radius: 10px !important;
                        }}
                        .group-face-label {{ margin-top: 6px; text-align: center; font-weight: 700; color: {card_text}; }}
                        .group-section-heading {{ margin: 2px 0 7px; color: #8b5cf6; font-size: 14px; font-weight: 700; }}
                        .group-detail-row {{ display: flex; justify-content: space-between; gap: 8px; padding: 3px 0; font-size: 13px; border-bottom: 1px solid rgba(148, 163, 184, .16); }}
                        .group-detail-label {{ opacity: .76; }}
                        .group-detail-value {{ text-align: right; font-weight: 600; overflow-wrap: anywhere; }}
                        .group-gender-badge {{ display: inline-block; padding: 2px 8px; color: #166534; background: #dcfce7; border-radius: 999px; font-weight: 700; }}
                        .group-emotion-list {{ font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
                        @media (max-width: 420px) {{
                            [class*="st-key-group-face-card-"] {{ padding: 9px !important; }}
                            [class*="st-key-group-face-card-"] [data-testid="stImage"] img {{ height: 135px !important; }}
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    face_card_columns = st.columns(2, gap="medium", wrap=True)
                    for i, (crop, pred) in enumerate(zip(cropped_faces, predictions)):
                        with face_card_columns[i % 2]:
                            with st.container(border=True, key=f"group-face-card-{i + 1}"):
                                image_column, details_column = st.columns([35, 65], gap="small")
                                with image_column:
                                    st.image(crop, width=min(220, crop.width))
                                    st.markdown(f'<div class="group-face-label">Face {i + 1}</div>', unsafe_allow_html=True)
                                with details_column:
                                    st.markdown('<div class="group-section-heading">Personal Details</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="group-detail-row"><span class="group-detail-label">Gender</span><span class="group-gender-badge">{pred["gender"]}</span></div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="group-detail-row"><span class="group-detail-label">Gender confidence</span><span class="group-detail-value">{pred["gender_confidence"]:.1f}%</span></div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="group-detail-row"><span class="group-detail-label">Estimated age</span><span class="group-detail-value">{pred["age"]} years</span></div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="group-detail-row"><span class="group-detail-label">Age group</span><span class="group-detail-value">{pred["age_group"]}</span></div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="group-detail-row"><span class="group-detail-label">Age confidence</span><span class="group-detail-value">{age_confidence_text(pred)}</span></div>', unsafe_allow_html=True)
                                    if pred.get("uncertain"):
                                        st.caption("Uncertain estimate")
                                    if pred.get("quality_flags"):
                                        st.caption(" · ".join(pred["quality_flags"]))

                                    st.markdown('<div class="group-section-heading" style="margin-top:10px">Facial Emotion</div>', unsafe_allow_html=True)
                                    if include_emotion and not emotion_error and i < len(emotion_results):
                                        emotion_result = emotion_results[i]
                                        st.markdown(f'<div class="group-detail-row"><span class="group-detail-label">Facial emotion</span><span class="group-detail-value">{emotion_result["emotion"]}</span></div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="group-detail-row"><span class="group-detail-label">Expression confidence</span><span class="group-detail-value">{emotion_result["confidence"]:.1f}%</span></div>', unsafe_allow_html=True)
                                        top3 = sorted(emotion_result.get("probabilities", {}).items(), key=lambda item: item[1], reverse=True)[:3]
                                        top_expression_text = " · ".join(f"{name.title()} {value * 100:.1f}%" for name, value in top3) or "Unavailable"
                                        st.markdown(f'<div class="group-emotion-list"><b>Top expressions:</b> {top_expression_text}</div>', unsafe_allow_html=True)
                                    elif emotion_error:
                                        st.caption("Expression estimate unavailable")
                                    else:
                                        st.caption("Facial emotion estimation is turned off.")
                                    timing = pred.get("timing_ms", {})
                                    if timing:
                                        st.caption(
                                            f"Decode {timing.get('decode', 0):.0f} ms · detection {timing.get('detection', 0):.0f} ms · face inference {timing.get('face_crop_and_prediction', 0):.0f} ms"
                                        )

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
    st.markdown(
        """
        <style>
        .st-key-emotion-camera-preview [data-testid="stCameraInput"] {
            width: min(100%, 720px) !important;
            max-width: 720px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            background: #0f172a !important;
            border: 1px solid #303a52 !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        .st-key-emotion-camera-preview [data-testid="stCameraInput"] video,
        .st-key-emotion-camera-preview [data-testid="stCameraInput"] img {
            display: block !important;
            width: 100% !important;
            max-width: 720px !important;
            height: auto !important;
            max-height: 420px !important;
            object-fit: contain !important;
            margin-left: auto !important;
            margin-right: auto !important;
            border-radius: 12px !important;
        }
        @media (max-width: 768px) {
            .st-key-emotion-camera-preview [data-testid="stCameraInput"] {
                width: 100% !important;
                max-width: 100% !important;
            }
            .st-key-emotion-camera-preview [data-testid="stCameraInput"] video,
            .st-key-emotion-camera-preview [data-testid="stCameraInput"] img {
                max-height: 60vh !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    input_tab, camera_tab = st.tabs(["Upload image", "Camera snapshot"], key="emotion-page-tabs")
    with input_tab:
        emotion_upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="ag44b_upload")
    with camera_tab:
        with st.container(key="emotion-camera-preview"):
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
                                render_bar_chart(probability_df)
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
    st.markdown(
        """
        <style>
        [data-testid="stTabs"] [role="tablist"] { gap: .35rem; border-bottom: 1px solid #39425b; }
        [data-testid="stTabs"] [role="tab"] { color: #e2e8f0 !important; opacity: 1 !important; background: #20263a !important; border-radius: 9px 9px 0 0; }
        [data-testid="stTabs"] [role="tab"] * { color: #e2e8f0 !important; opacity: 1 !important; }
        [data-testid="stTabs"] [role="tab"]:hover { color: #ffffff !important; background: #303853 !important; }
        [data-testid="stTabs"] [role="tab"]:hover * { color: #ffffff !important; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #c4b5fd !important; border-bottom: 3px solid #8b5cf6 !important; background: #262b42 !important; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] * { color: #c4b5fd !important; }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .stCaption, .stCaption * { color: #cbd5e1 !important; }
        [data-testid="stCustomComponentV1"] { display: block !important; width: 100% !important; max-width: 750px !important; margin: 0 auto !important; box-sizing: border-box !important; background: #0f172a !important; border: 1px solid #303a52 !important; border-radius: 12px !important; overflow: hidden !important; }
        [data-testid="stCameraInput"] { background: #0f172a !important; border: 1px solid #303a52 !important; border-radius: 12px !important; color: #e2e8f0 !important; overflow: hidden !important; }
        [data-testid="stCameraInput"] div { background-color: #0f172a !important; color: #cbd5e1 !important; }
        [data-testid="stCameraInput"] p, [data-testid="stCameraInput"] span { color: #cbd5e1 !important; }
        [data-testid="stCameraInput"] a { color: #c4b5fd !important; }
        [data-testid="stCameraInput"] a:hover { color: #ffffff !important; }
        [data-testid="stCameraInput"] video { background: #0f172a !important; }
        [data-testid="stCameraInput"] button:not(:disabled) { background: #7c3aed !important; color: #ffffff !important; border-color: #8b5cf6 !important; }
        [data-testid="stCameraInput"] button:disabled { background: #303853 !important; color: #cbd5e1 !important; border-color: #475569 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title section-title-live">📷 Live Camera</div>', unsafe_allow_html=True)
    st.write("Use real-time camera analysis or take a snapshot. Both use the same local detector and age/gender prediction pipeline as Analyze.")
    st.markdown('<p class="camera-helper-text">Frames stay in memory and are never saved. Camera access requires localhost or HTTPS.</p>', unsafe_allow_html=True)
    live_tab, snapshot_tab = st.tabs(["Live stream", "Snapshot fallback"], key="live-camera-tabs")
    snapshot_card_background = "#ffffff" if theme_preference == "Light" else "#0f172a"
    snapshot_card_border = "#CBD5E1" if theme_preference == "Light" else "#303a52"
    snapshot_card_text = "#17203f" if theme_preference == "Light" else "#f8fafc"
    st.markdown(
        f"""
        <style>
        .stApp .st-key-live-camera-snapshot-card {{
            width: min(100%, 750px) !important;
            max-width: 750px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 10px !important;
            box-sizing: border-box !important;
            background: {snapshot_card_background} !important;
            border: 1px solid {snapshot_card_border} !important;
            border-radius: 14px !important;
        }}
        .live-camera-snapshot-title {{
            margin: 0 0 8px !important;
            color: {snapshot_card_text} !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            text-align: left !important;
        }}
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] {{
            width: 100% !important;
            padding: 0 !important;
            color: {snapshot_card_text} !important;
            background: {snapshot_card_background} !important;
            border: 0 !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }}
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] div {{
            background-color: {snapshot_card_background} !important;
            color: {snapshot_card_text} !important;
        }}
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] label,
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] p,
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] span {{
            color: {snapshot_card_text} !important;
        }}
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] video,
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] img {{
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            height: 375px !important;
            max-height: 375px !important;
            object-fit: contain !important;
            background: {snapshot_card_background} !important;
            border-radius: 10px !important;
        }}
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] button {{
            display: block !important;
            width: 100% !important;
            min-height: 42px !important;
            color: #ffffff !important;
            background: #6d3fea !important;
            border: 1px solid #6d3fea !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: background-color 140ms ease, box-shadow 140ms ease !important;
        }}
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] button:not(:disabled):hover {{
            color: #ffffff !important;
            background: #5b2fc7 !important;
            border-color: #5b2fc7 !important;
            box-shadow: 0 3px 10px rgba(57, 31, 130, 0.24) !important;
        }}
        .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] button:disabled {{
            color: #ffffff !important;
            background: #6d3fea !important;
            border-color: #6d3fea !important;
            opacity: 0.65 !important;
        }}
        @media (max-width: 768px) {{
            .stApp .st-key-live-camera-snapshot-card {{
                width: 100% !important;
                max-width: 100% !important;
            }}
            .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] video,
            .stApp .st-key-live-camera-snapshot-card [data-testid="stCameraInput"] img {{
                height: min(375px, 55vh) !important;
                max-height: min(375px, 55vh) !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
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
                video_html_attrs={
                    "autoPlay": True,
                    "muted": True,
                    "playsInline": True,
                    "controls": True,
                    "style": {
                        "width": "100%",
                        "maxWidth": "750px",
                        "height": "min(420px, 56vw)",
                        "objectFit": "contain",
                        "display": "block",
                        "margin": "0 auto",
                        "backgroundColor": "#f1f5f9" if theme_preference == "Light" else "#0f172a",
                        "borderRadius": "12px",
                    },
                },
            )
            if context.state.playing:
                st.success("Camera is on. Select Stop when finished.")
                processor = context.video_processor
                if processor is not None:
                    st.markdown(f'<p class="camera-helper-text">Approx. {processor.last_fps:.1f} processed FPS · {processor.last_latency_ms:.0f} ms last inference</p>', unsafe_allow_html=True)
            elif context.state.signalling:
                st.info("Starting camera… Allow camera access in your browser if prompted.")
            else:
                st.warning("Camera is off or unavailable. Allow camera access and select a device; if permission is blocked or no camera is connected, use Snapshot fallback.")

    with snapshot_tab:
        with st.container(key="live-camera-snapshot-card", width=750):
            st.markdown('<div class="live-camera-snapshot-title">Take a camera snapshot</div>', unsafe_allow_html=True)
            camera_file = st.camera_input("Camera Preview", key="live_camera_snapshot")
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
    try:
        emotion_history = load_emotion_history()
    except Exception:
        st.error("Expression history is temporarily unavailable. Please try again.")
    else:
        if emotion_history.empty:
            st.info("No expression history is available for this session/account yet.")
        else:
            emotion_view = emotion_history.drop(columns=["Analysis ID"], errors="ignore")
            st.dataframe(emotion_view, use_container_width=True, hide_index=True)
            st.download_button(
                "Download expression history as CSV",
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
        st.error("Analytics are temporarily unavailable. Please try again.")
    else:
        if df.empty:
            st.info("Analytics will appear after predictions are made in this session/account.")
        else:
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Predictions", len(df))
            metric2.metric("Average estimated age", f"{df['Estimated Age'].mean():.1f}")
            metric3.metric("Male estimates", int((df["Gender"] == "Male").sum()))
            metric4.metric("Female estimates", int((df["Gender"] == "Female").sum()))
            st.subheader("Gender estimates")
            render_bar_chart(df["Gender"].value_counts())
            st.subheader("Age groups")
            render_bar_chart(df["Age Group"].value_counts())
            emotion_metrics = load_emotion_history()
            st.subheader("Expression summary")
            if emotion_metrics.empty:
                st.info("Expression statistics will appear after expression analysis.")
            else:
                render_bar_chart(emotion_metrics["Emotion"].value_counts())
                emotion_metrics["Day"] = pd.to_datetime(emotion_metrics["Timestamp"], utc=True, errors="coerce").dt.date
                trend = emotion_metrics.dropna(subset=["Day"]).groupby(["Day", "Emotion"]).size().unstack(fill_value=0)
                if not trend.empty:
                    st.subheader("Expression trend")
                    render_line_chart(trend)
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
        if df.empty:
            st.info("No prediction history is available for this session/account yet.")
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

    if settings_section == "Appearance":
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
        st.info("Images and camera frames are processed in memory and are not saved. Registered accounts save prediction metadata privately; guest history lasts only for the current session.")
        st.warning("Only analyze images you have permission to use. Results are estimates, not identity or medical assessments, and must not be used for decisions about people.")
        if IS_GUEST:
            st.caption("Guest prediction history is temporary and is cleared when you exit Guest Mode or close the session.")
        else:
            retention_options = {"Off (keep until deleted)": 0, "30 days": 30, "90 days": 90}
            current_retention = get_retention_days()
            retention_index = list(retention_options.values()).index(current_retention) if current_retention in retention_options.values() else 0
            selected_retention = st.selectbox("Automatically delete history after", list(retention_options), index=retention_index, key="retention_setting")
            if st.button("Save retention setting", key="save_retention"):
                set_retention_days(retention_options[selected_retention])
                st.success("Retention setting saved.")
            privacy_history = load_user_history()
            if not privacy_history.empty:
                export_history = privacy_history.drop(columns=["Analysis ID"], errors="ignore")
                st.download_button("Download my data as CSV", export_history.to_csv(index=False), "my_age_gender_data.csv", "text/csv")
                st.download_button("Download my data as JSON", export_history.to_json(orient="records", indent=2), "my_age_gender_data.json", "application/json")
            export_emotions = load_emotion_history()
            if not export_emotions.empty:
                export_emotions = export_emotions.drop(columns=["Analysis ID"], errors="ignore")
                st.download_button("Download expression data as CSV", export_emotions.to_csv(index=False), "my_expression_data.csv", "text/csv")
                st.download_button("Download expression data as JSON", export_emotions.to_json(orient="records", indent=2), "my_expression_data.json", "application/json")
            delete_confirmation = st.checkbox("I understand this deletes this app's saved data and consent record.", key="delete_account_confirm")
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
        and are not saved as image files. Registered accounts save prediction
        metadata privately; guest prediction history stays in the current session.
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

# Final light-theme contrast overrides. Kept last so page-specific dark styles cannot win.
if theme_preference == "Light":
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #F8FAFC; --sidebar-bg: #FFFFFF; --card-bg: #FFFFFF;
            --text-primary: #172554; --text-secondary: #475569; --text-muted: #64748B;
            --accent: #6D3FEA; --border: #CBD5E1; --selected-bg: #EDE9FE;
            --info-bg: #DBEAFE; --warning-bg: #FEF3C7; --success-bg: #DCFCE7; --error-bg: #FEE2E2;
        }
        .stApp, .stApp [data-testid="stAppViewContainer"] { background: var(--app-bg) !important; color: var(--text-primary) !important; }
        .stApp [data-testid="stHeader"], .stApp [data-testid="stToolbar"] { background: var(--app-bg) !important; }
        .stApp .camera-helper-text { color: var(--text-secondary) !important; margin: 0 0 .75rem 0; }
        .stApp [data-testid="stAppViewContainer"] h1,
        .stApp [data-testid="stAppViewContainer"] h2,
        .stApp [data-testid="stAppViewContainer"] h3,
        .stApp [data-testid="stAppViewContainer"] h4 { color: #172554 !important; }
        .stApp [data-testid="stAppViewContainer"] p,
        .stApp [data-testid="stAppViewContainer"] li { color: #1f2937; }
        .stApp [data-testid="stAppViewContainer"] .stCaption,
        .stApp [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
        .stApp [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
        .stApp [data-testid="stAppViewContainer"] small,
        .stApp [data-testid="stAppViewContainer"] .subtitle,
        .stApp [data-testid="stAppViewContainer"] .metric-label,
        .stApp [data-testid="stAppViewContainer"] p[style*="#cbd5e1"],
        .stApp [data-testid="stAppViewContainer"] p[style*="#d5deed"] { color: #475569 !important; opacity: 1 !important; }
        .stApp [data-testid="stSidebar"] { background: #ffffff !important; }
        .stApp [data-testid="stSidebar"] *,
        .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #1e293b !important; opacity: 1 !important; }
        .stApp [data-testid="stSidebar"] [aria-checked="true"] + div,
        .stApp [data-testid="stSidebar"] [aria-checked="true"] ~ div { color: #5427c5 !important; }
        .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { background: #ffffff !important; border: 1px dashed #cbd5e1 !important; border-radius: 12px !important; }
        .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * { color: #334155 !important; opacity: 1 !important; }
        .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button { background: #6d3fea !important; color: #ffffff !important; border-color: #6d3fea !important; }
        .stApp [data-testid="stCameraInput"] { background: #ffffff !important; border: 1px solid #cbd5e1 !important; color: #1f2937 !important; }
        .stApp [data-testid="stCameraInput"] div,
        .stApp [data-testid="stCameraInput"] p,
        .stApp [data-testid="stCameraInput"] span { background-color: #ffffff !important; color: #334155 !important; opacity: 1 !important; }
        .stApp [data-testid="stCameraInput"] button:not(:disabled) { background: #6d3fea !important; color: #ffffff !important; }
        .stApp [data-testid="stTabs"] [role="tab"] { color: #1e293b !important; background: #f1f5f9 !important; opacity: 1 !important; }
        .stApp [data-testid="stTabs"] [role="tab"] * { color: #1e293b !important; opacity: 1 !important; }
        .stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #5427c5 !important; background: #ffffff !important; border-bottom: 3px solid #6d3fea !important; }
        .stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] * { color: #5427c5 !important; }
        .stApp [data-testid="stAlert"] { border: 1px solid #cbd5e1 !important; }
        .stApp [data-testid="stAlert"] p,
        .stApp [data-testid="stAlert"] span,
        .stApp [data-testid="stAlert"] div { color: #1e293b !important; opacity: 1 !important; }
        .stApp [data-testid="stDownloadButton"] button,
        .stApp [data-testid="stButton"] button { color: #1e293b !important; opacity: 1 !important; }
        .stApp [data-testid="stDownloadButton"] button:hover,
        .stApp [data-testid="stButton"] button:hover { color: #5427c5 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
