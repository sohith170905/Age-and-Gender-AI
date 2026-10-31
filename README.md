# AgeVision AI

A local Streamlit application for face detection, age/gender estimates, and optional facial-expression estimates. The project preserves its existing GRA_Net and local AgeVision ONNX models; it does not train, download, convert, or rewrite checkpoints during app use.

## Start the app on Windows

The project currently includes a ready-to-use `venv`:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Or use `scripts\run_app.ps1`. If setting up a new machine, use Python 3.11, create a virtual environment, and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Keep the app bound to localhost unless it is behind an authenticated HTTPS reverse proxy. Browser camera access on another device needs HTTPS; see [deployment notes](docs/DEPLOYMENT.md).

## Screenshots

Add approved application screenshots here. Do not include real people's images without their permission.

## Architecture

```text
Browser ──> Streamlit app (app.py) ──> validated in-memory image pipeline
                                         ├─> local YuNet/Haar detector
                                         ├─> local AgeVision ONNX or GRA_Net model
                                         └─> optional local expression model
Signed-in metadata ──> SQLAlchemy ──> SQLite (default) or PostgreSQL
Optional REST client ──> FastAPI ──> same in-memory service + local models
```

The Streamlit app does not call the API. Image bytes and camera frames are not written to disk or saved in the database.

## Features

- Home, image analysis (single or group), facial-expression analysis, live camera, private history/analytics, Settings, and About AI.
- Local GRA_Net or AgeVision ONNX inference; the local expression model with high-accuracy multi-view averaging.
- Upload validation and image-quality notices; images and camera frames remain in memory and are not written as files.
- Signed-in prediction history stores metadata, not face images. Guests do not save predictions.
- In-memory CSV, JSON, annotated-PNG, and PDF exports.
- Optional bearer-protected FastAPI service.

## Google sign-in (optional)

Configure Streamlit’s native Google OIDC in `.streamlit/secrets.toml` (never commit it):

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "replace-with-a-long-random-secret"
client_id = "your-client-id.apps.googleusercontent.com"
client_secret = "your-client-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Register the same redirect URL in the Google OAuth web client. Without credentials, guest mode remains available.

## Configuration

The process reads OS environment variables. `.env.example` is a reference and is not auto-loaded.

| Variable | Purpose | Default |
|---|---|---|
| `AGE_GENDER_MODEL` | `agevision_onnx` or `gra_net_compact` | `agevision_onnx` (falls back to GRA_Net if absent) |
| `AGE_GENDER_MAX_UPLOAD_BYTES` | Per-image upload cap | 16 MiB |
| `AGE_GENDER_MAX_IMAGE_PIXELS` | Decoded image pixel cap | 40,000,000 |
| `AGE_GENDER_MAX_FACES` | Faces analyzed per image | 80 |
| `AGE_GENDER_MAX_BATCH_IMAGES` | Images in one batch | 5 |
| `AGE_GENDER_RETENTION_DAYS` | Default signed-in metadata retention | 0 (Off) |
| `DATABASE_URL` | Optional SQLAlchemy database URL | local SQLite file |
| `AGE_GENDER_API_KEY` | Enables authenticated API routes | unset (API unavailable) |
| `AGE_GENDER_API_RATE_LIMIT` | Requests per API key/minute | 60 |
| `AGE_GENDER_CAMERA_FRAME_INTERVAL` | Frames between inference passes | 5 |
| `AGE_GENDER_CAMERA_MAX_WIDTH` / `_HEIGHT` | Camera processing dimensions | 640 × 480 |
| `AGE_GENDER_CAMERA_STUN_URL` | Optional STUN URL for WebRTC | empty |

Settings also let signed-in users select Off, 30-day, or 90-day history retention and choose which local age/gender model to use.

## Run the optional API

Set a long random key in the current PowerShell session, then run `scripts\run_api.ps1`:

```powershell
$env:AGE_GENDER_API_KEY = "<long-random-secret>"
.\scripts\run_api.ps1
```

The API binds to `127.0.0.1:8000`; health is `GET /health`, authenticated model metadata is `GET /api/v1/models`, and predictions use `POST /api/v1/analyze` with a multipart `image` field. See [API and deployment notes](docs/DEPLOYMENT.md).

## Verification

Synthetic-only regression tests cover upload validation, quality flags, orchestration, exports, tracker behavior, registry hashes, and numerical inference against frozen synthetic baselines. They do not read face datasets:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Read [the upgrade report](docs/REALWORLD_UPGRADE_REPORT.md) for the exact test outcomes and model checksums after verification.

## Model and use limitations

Age, gender, and expression are estimates and can be incorrect. Confidence is not independently calibrated; poor image quality can make results less reliable. Do not use outputs to make decisions about people. The repository does not assert that the checkpoints meet an accuracy target or reproduce published-paper results.

## Troubleshooting

- **Model unavailable:** confirm the chosen local checkpoint exists in `models/`; the app never downloads model files. Settings → Model shows the checkpoint status and short checksum.
- **Camera permission or blank camera:** allow camera access in the browser, close other applications using the camera, use `http://localhost` on the same computer, or use HTTPS for another device. Snapshot mode remains available as a fallback.
- **ONNX Runtime fails to load:** use the pinned requirements in a fresh Python 3.11 virtual environment and check CPU support for the installed `onnxruntime` wheel.
- **Google sign-in does not appear:** check the local `[auth]` settings and exact registered redirect URI. Guest mode works without Google credentials.
- **History is empty:** guest results are not saved. Sign in and grant consent before making predictions to save metadata.
- **App feels slow:** models load lazily on the first analysis. For CPU live camera, keep the configured reduced frame size and inference interval; turn off optional expression analysis.
