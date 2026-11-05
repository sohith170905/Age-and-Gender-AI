# AgeVision AI (Age-and-Gender-AI)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142.2-009688.svg)](https://fastapi.tiangolo.com/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Tracked-informational.svg)](https://git-lfs.github.com/)

A modular computer vision application powered by Streamlit and local deep learning models for face detection, age and gender estimation, and optional facial expression classification. The project performs all inference locally using pre-trained neural networks without sending image data to external cloud APIs or third-party servers.

---

## 1. Project Introduction

**AgeVision AI** provides an interactive web application and optional REST API designed for demographic and affective visual analysis. The system combines multi-task Convolutional Neural Networks (GRA_Net architecture) and deep ONNX-optimized vision models to detect faces and compute real-time estimates of age, gender, and facial expressions from uploaded photos or live camera streams.

---

## 2. Project Objectives

- **On-Device Inference**: Run computer vision pipelines locally without reliance on third-party cloud APIs.
- **Privacy by Design**: Process uploaded images and camera frames strictly in memory without writing image files to disk.
- **Modular Architecture**: Separate UI presentation (`app.py`), backend business logic (`services/`), and inference engines (`inference/`, `emotion/`).
- **Reproducible Engineering**: Maintain frozen model baselines, automated synthetic test suites, and Git LFS pointer integrity.

---

## 3. Key Features

- **Interactive Multi-Page Streamlit UI**:
  - **Single & Group Face Analysis**: Bounding box detection, age prediction, gender classification, and confidence scoring.
  - **Facial Expression Analysis**: 7-class emotion classification (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger) with multi-view Test-Time Augmentation (TTA).
  - **Live WebRTC Camera Stream**: Real-time video frame processing with an automatic snapshot fallback mode.
  - **User History & Analytics**: Local, user-scoped prediction history with interactive charts and aggregate demographic metrics.
  - **Model Settings & Audit**: Inspect loaded checkpoints, verify SHA-256 integrity, switch between active models, and configure data retention.
- **Privacy & Security Controls**:
  - **No Image Persistence**: Uploads and camera frames are decoded directly in memory and discarded after processing.
  - **Secure Local Authentication**: Salted PBKDF2-HMAC-SHA256 password hashing for registered user accounts.
  - **Ephemeral Guest Mode**: Explore the application without creating an account; guest data is never written to the database.
  - **Configurable Retention**: Choose 0-day (Off), 30-day, or 90-day automatic metadata retention, with instant one-click data deletion.
- **In-Memory Multi-Format Export**:
  - Export prediction results directly to CSV, JSON, annotated PNG crops, or structured PDF reports without writing temporary files.
- **Optional Bearer-Protected REST API**:
  - Standalone FastAPI service providing endpoints for health monitoring, model metadata inspection, and authenticated image analysis.

---

## 4. Technology Stack

- **Application & Presentation**: [Streamlit](https://streamlit.io/), [Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc)
- **Computer Vision & Image Processing**: OpenCV (`opencv-python-headless`), Pillow (`PIL`), PyAV (`av`)
- **Deep Learning Frameworks**: TensorFlow / Keras, [ONNX Runtime](https://onnxruntime.ai/)
- **Data & Numeric Computing**: NumPy, Pandas, Scikit-learn
- **Database & Persistence**: SQLAlchemy 2.0, SQLite (default local) / PostgreSQL (`psycopg`)
- **REST Backend**: FastAPI, Uvicorn, Pydantic v2
- **Testing & Quality**: Pytest, Streamlit AppTest framework

---

## 5. Project Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       Client Browser                        │
└──────────────┬──────────────────────────────┬───────────────┘
               │ HTTP / WebSockets            │ HTTPS / WebRTC
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit Application (app.py)              │
├─────────────────────────────────────────────────────────────┤
│  • Auth Manager (services/accounts.py)                      │
│  • Quality Checks & Blurring (services/quality.py)          │
│  • Orchestration Pipeline (services/analysis.py)            │
│  • Multi-Format Exporters (services/exports.py)             │
│  • In-Memory Prediction Tracking (services/tracking.py)     │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│      Face Detection          │ │    Deep Inference Models   │
├──────────────────────────────┤ ├────────────────────────────┤
│ • OpenCV YuNet (ONNX)        │ │ • GRA_Net Compact (Keras)  │
│ • Haar Cascade Fallback      │ │ • AgeVision DINO (ONNX)    │
│                              │ │ • Expression CNN (Keras)   │
└──────────────────────────────┘ └────────────────────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Database Storage (Local SQLite)              │
│    • User accounts (hashed passwords)                       │
│    • User-scoped prediction metadata (no images stored)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Directory Structure

```text
Age-and-Gender-AI/
├── .env.example                 # Environment configuration template
├── .gitattributes               # Git LFS tracking configuration
├── .gitignore                   # Exclusions for credentials, data, and artifacts
├── Dockerfile                   # Production container definition
├── README.md                    # Project documentation
├── app.py                       # Main Streamlit web application
├── docker-compose.yml           # Container orchestration definition
├── requirements.txt             # Production dependencies (pinned)
├── requirements-dev.txt         # Development & testing dependencies
├── run_app.ps1                  # Windows launcher for Streamlit app
├── run_api.ps1                  # Windows launcher for FastAPI service
├── api/
│   └── main.py                  # Standalone FastAPI REST service
├── assets/
│   └── agevision_favicon.svg    # Application branding icon
├── config/
│   └── model_registry.json      # Model metadata and SHA-256 checksums
├── core/
│   ├── config.py                # Environment configuration loader
│   └── model_registry.py        # Model registry verification utilities
├── data/
│   └── README.md                # Dataset storage guidelines (datasets ignored)
├── docs/
│   ├── AUDIT.md                 # Security and baseline audit record
│   ├── DEPLOYMENT.md            # Production deployment recommendations
│   └── model_hashes_after.json  # Reference cryptographic model checksums
├── emotion/
│   ├── model_loader.py          # Emotion model loader
│   ├── predictor.py             # Emotion inference handler
│   └── tta.py                   # Multi-view Test-Time Augmentation
├── inference/
│   └── api_engine.py            # In-memory batch inference engine
├── models/                      # Deep learning weights (tracked via Git LFS)
│   ├── emotion/
│   │   ├── emotion_model.keras  # Facial expression classifier
│   │   └── expression_classes.json
│   ├── faceage_dino_fp32.onnx   # High-resolution age/gender ONNX model
│   ├── face_detection_yunet_2023mar.onnx
│   └── gra_net*.keras           # GRA_Net multi-task checkpoints
├── notebooks/
│   └── GRA_Net_Complete_Colab.ipynb # Google Colab starter notebook
├── services/
│   ├── accounts.py              # Salted PBKDF2 authentication service
│   ├── analysis.py              # Pipeline orchestration service
│   ├── batch_emotions.py        # Multi-face batch emotion classification
│   ├── exports.py               # In-memory CSV, JSON, PNG, and PDF exporters
│   ├── quality.py               # Lighting, blur, and contrast validators
│   └── tracking.py              # History tracking and retention service
└── tests/
    ├── baseline/                # Synthetic baseline arrays
    ├── test_accounts.py         # Authentication service unit tests
    ├── test_batch_emotions.py   # Batch emotion classification tests
    └── test_ml_pipeline.py      # Streamlit UI & ML pipeline regression tests
```

---

## 7. Installation Instructions

### Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version **3.11** (recommended: 3.11.9)
- **Git & Git LFS**: Ensure Git and [Git LFS](https://git-lfs.com/) are installed on your machine.

---

## 8. Virtual Environment Setup (Windows)

Clone the repository and create a dedicated Python 3.11 virtual environment:

```powershell
# Clone the repository
git clone https://github.com/sohit170905/Age-and-Gender-AI.git
cd Age-and-Gender-AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

*(On macOS / Linux, run: `python3 -m venv venv && source venv/bin/activate`)*

---

## 9. Dependency Installation

Upgrade `pip` and install the pinned dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To install development and testing tools:

```powershell
pip install -r requirements-dev.txt
```

---

## 10. Environment Variable Configuration

Copy the `.env.example` template to `.env`:

```powershell
Copy-Item .env.example .env
```

Key environment configuration variables:

| Variable | Description | Default |
|---|---|---|
| `AGE_GENDER_MODEL` | Selected model (`agevision_onnx` or `gra_net_compact`) | `agevision_onnx` |
| `AGE_GENDER_MAX_UPLOAD_BYTES` | Maximum allowed file upload size in bytes | `16777216` (16 MiB) |
| `AGE_GENDER_MAX_IMAGE_PIXELS` | Maximum decoded image pixel limit | `40000000` (40 MP) |
| `AGE_GENDER_MAX_FACES` | Maximum faces detected and processed per image | `80` |
| `AGE_GENDER_MAX_BATCH_IMAGES` | Maximum number of images in a batch upload | `5` |
| `DATABASE_URL` | SQLAlchemy connection string (defaults to SQLite) | `sqlite:///data/app.db` |
| `AGE_GENDER_RETENTION_DAYS` | Default retention period in days (`0` = Off) | `0` |
| `AGE_GENDER_API_KEY` | Bearer API token required for the FastAPI backend | *Unset (API disabled)* |
| `AGE_GENDER_API_RATE_LIMIT` | Allowed requests per minute per API key | `60` |
| `AGE_GENDER_CAMERA_FRAME_INTERVAL` | Interval between processed camera frames | `5` |

---

## 11. Dataset Setup Instructions (Optional)

The application ships with trained model weights and **does not require a dataset to run**.

If you plan to run evaluation or training scripts (`train.py`, `evaluate.py`):
1. Download a licensed dataset (e.g., the [UTKFace](https://susanqq.github.io/UTKFace/) dataset).
2. Place the unzipped image files into `data/UTKFace/`.
3. The `data/UTKFace/` folder is explicitly ignored by Git to avoid uploading private or unlicensed images to GitHub.

---

## 12. Model File Requirements & Checksums

The project uses frozen local model weights stored in the `models/` folder:

| Model Identifier | File Path | Format | Task |
|---|---|---|---|
| `agevision_onnx` | `models/faceage_dino_fp32.onnx` | ONNX | Age & Gender Estimation |
| `gra_net_compact` | `models/gra_net_age_tuned.keras` | Keras | Age & Gender Estimation |
| `gra_net_v1` | `models/gra_net.keras` | Keras | Multi-task Baseline |
| `expression_v1` | `models/emotion/emotion_model.keras` | Keras | 7-Class Emotion Classification |
| `yunet_detector` | `models/face_detection_yunet_2023mar.onnx` | ONNX | Face Detection |

All weights and configurations are registered with cryptographic SHA-256 hashes in `config/model_registry.json`.

---

## 13. Git LFS Installation and Model Download

Model binary files are managed using [Git Large File Storage (LFS)](https://git-lfs.com/).

Before cloning or pulling model files, ensure Git LFS is initialized:

```powershell
# Initialize Git LFS on your machine
git lfs install

# If models were cloned as pointers, fetch the actual binary weights:
git lfs pull
```

Verify that all model files are tracked properly:

```powershell
git lfs ls-files
```

---

## 14. Application Launch Instructions

Launch the Streamlit web application:

```powershell
streamlit run app.py
```

Or execute the Windows helper script:

```powershell
.\run_app.ps1
```

Once running, navigate to `http://localhost:8501` in your web browser.

---

## 15. Testing Instructions

The repository includes comprehensive synthetic unit and integration tests. The test suite uses synthetic arrays and in-memory databases, requiring no external datasets:

```powershell
# Run the complete test suite
pytest tests/ -v
```

To run registry integrity checks:

```powershell
pytest tests/test_ml_pipeline.py -k "registry" -v
```

---

## 16. Optional REST API Service

An independent FastAPI service is available for programmatic integration:

```powershell
# Set an API key in your session
$env:AGE_GENDER_API_KEY = "your-secure-random-token"

# Start the API server
.\run_api.ps1
# Or directly via Uvicorn:
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- **Health check**: `GET http://127.0.0.1:8000/health`
- **Models endpoint**: `GET http://127.0.0.1:8000/api/v1/models` *(Requires `Authorization: Bearer <token>`)*
- **Analysis endpoint**: `POST http://127.0.0.1:8000/api/v1/analyze` *(Multipart form upload with field `image`)*

---

## 17. Important Limitations & Ethical Considerations

- **Statistical Estimates**: Predictions for age, gender, and facial expressions are algorithmic estimates based on learned visual features and may produce errors or biased outputs.
- **Image Quality Dependence**: Poor illumination, extreme head poses, motion blur, and low image resolution significantly degrade prediction reliability.
- **Non-Diagnostic & Non-Decisional**: Outputs must **never** be used for automated decision-making, access control, legal determination, surveillance, or health assessments.
- **Privacy Compliance**: Ensure you possess explicit consent from individuals before capturing or processing their facial images.

---

## 18. License

This repository does not currently include an open-source license file. All rights are reserved by the repository owner. Contact the repository maintainer for licensing and reuse permissions.
