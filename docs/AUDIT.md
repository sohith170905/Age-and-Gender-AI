# Initial audit — 2026-10-31

## Architecture and file map

The application is a single-file Streamlit UI with separate inference helpers for emotion and face detection. `app.py` owns page routing, presentation, upload decoding, age/gender inference selection, database access, history, and analytics. `gra_net.py` defines model architectures; it does not own runtime UI. `face_detector.py` implements detection and drawing. `emotion/` contains the frozen emotion-model loader, seven-class labels, preprocessing, TTA, and prediction wrapper. `data_utils.py`, `train*.py`, `fine_tune*.py`, `evaluate*.py`, `tta_eval.py`, `metrics_cal.py`, `bias_analysis.py`, `nearest_face.py`, and `predict.py` are offline training/evaluation utilities and are not imported as app pages.

| Area | Existing files | Role |
|---|---|---|
| Streamlit entrypoint and UI | `app.py` | Login/guest gate, navigation, analysis, camera snapshot, history, settings, About pages |
| Age/gender model definitions and inference | `gra_net.py`, `app.py` | Keras architectures and image-to-age/gender prediction |
| Alternative age/gender inference | `app.py` | Local ONNX Runtime model at `models/faceage_dino_fp32.onnx` |
| Face detection | `face_detector.py`, `app.py`, `emotion/face_detection.py` | YuNet/Haar detection, crop and overlay helpers |
| Emotion inference | `emotion/__init__.py`, `emotion/model_loader.py`, `emotion/predictor.py`, `emotion/preprocessing.py`, `emotion/tta.py`, `emotion/face_detection.py` | Frozen seven-class Keras model and TTA-5 |
| Persistence/auth | `app.py` | Streamlit Google OIDC; SQLAlchemy/SQLite or configured PostgreSQL; metadata tables for age/gender and emotion |
| Existing checks | `tests/test_ml_pipeline.py` | Existing GRA_Net tests; currently loads a real UTKFace image at import/setup and must not be run under the no-dataset constraint |
| Configuration/dependencies | `.streamlit/config.toml`, `.gitignore`, `requirements.txt` | Theme/server settings, exclusions, broad unpinned dependency ranges |
| Setup/docs | `README.md`, `run_https.py`, `docs/GRA_Net_Project_Report.*`, `docs/GRA_Net_Project_Presentation.*` | Local startup, local HTTPS, project reports |
| Model/data/runtime assets | `models/`, `data/`, `results/`, `tmp/`, `notebooks/`, `assets/` | Weights, datasets, generated outputs, notebooks and static assets; datasets and generated user imagery were not opened |

## Inference pipeline

- GRA_Net: `app.py:399-432` lazily loads `MODEL_PATH` (`models/gra_net_age_tuned.keras` by default). `app.py:1015-1405` resizes RGB face crops to the model input dimensions, scales pixels to `[0,1]`, calls the existing three-output model, and maps the existing gender/age-class/regression heads to displayed predictions.
- AgeVision AI: `app.py:434-465` creates a local ONNX Runtime session. `app.py:959-1012` applies the existing 224px ImageNet-normalized input and age/gender output decoding.
- Detection and image analysis: `app.py:474-714`, `app.py:759-913`, `app.py:1409-1520`, and `app.py:1816-1840` implement cached detectors, face filtering/crops, overlays, upload analysis.
- Emotion: `emotion/model_loader.py`, `emotion/preprocessing.py`, `emotion/tta.py`, and `emotion/predictor.py` validate and load the frozen seven-class model, crop grayscale 48×48 faces, create source-compatible views, and average softmax outputs.
- All three model paths were run on seeded synthetic tensors only before edits. Outputs are saved under `tests/baseline/` for later numerical comparison. No dataset was used.

## Authentication and database

Google authentication uses Streamlit OIDC configured through `[auth]` secrets (`app.py:88-104`, `app.py:218-247`). Guest mode is session state and skips persistence. `app.py:1524-1583` creates a SQLAlchemy engine, defaulting to SQLite at `data/app.db` and allowing `DATABASE_URL`/Streamlit secrets for hosted databases. Age/gender and emotion metadata are saved in separate user-scoped tables; image bytes are used transiently for deterministic analysis IDs and are not inserted into these tables (`app.py:1585-1718`).

## Existing pages

`app.py:1739-1783` defines Home, Analyze (Single Person / Group Analysis), Facial Emotion Detection, Live Camera, History (History / Analytics-Statistics / Emotion History), Settings (Account / Appearance / Privacy / Security / Model), and About AI (How It Works / GRA_Net Model / Dataset & Training / Evaluation / Responsible AI). Live Camera is currently `st.camera_input`, not a WebRTC stream (`app.py:2119-2238`).

## Installed dependencies

The working environment is Python 3.11.9 and Streamlit 1.64.0. It includes TensorFlow 2.18.1, Keras 3.15.1, ONNX Runtime 1.31.1, OpenCV 4.10/4.11, Pillow 12.3.0, SQLAlchemy 2.0.54, pandas 2.3.3, NumPy 1.26.4, Streamlit-WebRTC 0.78.1, FastAPI 0.142.2, Uvicorn 0.53.0, pytest 9.1.1, and pydantic-settings 2.15.0. `requirements.txt` currently uses ranges rather than pinned versions. The complete installed package list was captured in the task log; no dependency was installed for this audit.

## Baseline gaps P1–P9

| ID | Finding | Evidence |
|---|---|---|
| P1 | Model identity is duplicated: Home labels GRA_Net, sidebar and Settings select a different active backend, while About AI hardcodes a tuned GRA_Net description. | `app.py:1786-1796`, `app.py:1864-1878`, `app.py:2375-2406`, `app.py:2423-2455` |
| P2 | Settings displays the machine-specific absolute ONNX model path. | `app.py:2388-2400` |
| P3 | Internal names AG-44B and TTA-5 appear in the normal user interface. | `app.py:1925-1983`, `app.py:2054-2064`, `app.py:2241-2243`; emotion package technical identifiers in `emotion/` |
| P4 | Live Camera is a one-shot snapshot widget with no true real-time stream, FPS/latency loop, tracking or smoothing. | `app.py:2119-2238`; `streamlit-webrtc` is already installed but unused by this page |
| P5 | Account displays profile details but has no account-data export/deletion flow; Security only signs out; History offers CSV and bulk history delete but no search, filters, pagination or row deletion. | `app.py:2297-2329`, `app.py:2333-2365`, `app.py:2408-2420` |
| P6 | No first-run consent or retention preference exists; Privacy is informational text only. | `app.py:2366-2374`, `app.py:2498-2540` |
| P7 | Dependency ranges are not pinned; no app health endpoint, API, Docker setup or deployment guide exists. Streamlit toolbar mode is not configured. | `requirements.txt:1-10`, `.streamlit/config.toml`, repository inventory |
| P8 | Analysis handles one uploaded image per run; no batch workflow or PNG/JSON/PDF export. Quality screening and clear uncertain-state coverage are incomplete. | `app.py:1906-2050`, `app.py:2119-2238`; upload validation is limited to size/pixel/decode checks at `app.py:270-283` |
| P9 | Emotion limitations and comparative reliability are not consistently explained on About AI model cards. | `app.py:2423-2575`; no emotion model card or existing verified metric source was identified in the UI code |

## Baseline constraints and files

No files under `models/` or `experiments/` were changed. Before-edit SHA-256 values for every file under `models/` are recorded in `docs/model_hashes_before.json`. The synthetic output archives are `tests/baseline/gra.npz`, `tests/baseline/agevision.npz`, and `tests/baseline/emotion.npz`. The existing dataset-reading test was not run; it will be replaced with synthetic-only tests before the test suite is executed.
