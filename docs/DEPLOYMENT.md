# Deployment and operations

## Local Windows run

1. Open PowerShell in the project directory.
2. Activate `venv` or `.venv` and run `streamlit run app.py` (or `scripts\run_app.ps1`).
3. Open the local URL Streamlit prints.
4. For a phone camera, serve through HTTPS. `run_https.py` is for local development only and uses a self-signed certificate; do not expose that listener directly to the public internet.

## Docker Compose

Docker packages both the Streamlit app and optional API. The app is the default service and is published only on loopback:

```powershell
docker compose build app
docker compose up -d app
```

Then open `http://localhost:8501`. Model files are mounted read-only; the SQLite data and logs use named volumes. No dataset folder, user image, or upload cache is copied into the image. The API is optional:

```powershell
$env:AGE_GENDER_API_KEY = "<long-random-secret>"
docker compose --profile api up -d --build
```

The API is published on `127.0.0.1:8000`; configure an HTTPS reverse proxy with authentication, TLS, request limits, and firewall rules before making either service reachable outside the host. Do not expose Streamlit or the API directly to the public internet. Set a persistent `DATABASE_URL` only when using a secured database service with access controls and backups.

## API contract

- `GET /health`: basic status/version; contains no user information.
- `GET /api/v1/models`: bearer-authenticated model names, versions, tasks, and input sizes.
- `POST /api/v1/analyze`: bearer-authenticated JPG, PNG, or WEBP. Use multipart field `image`; optional multipart field `include_emotion=true` adds expression estimates. API images are parsed and processed in memory.
- Responses include face count, truncation status, bounding boxes, age/gender estimates, uncertainty/quality flags, and optional expression estimates. The API does not write image bytes or predictions to the Streamlit history database.
- The API is disabled with HTTP 503 until `AGE_GENDER_API_KEY` is set. Authenticated calls use an in-process rate limit; multiple API workers each have an independent counter.

Example PowerShell request (replace the key and image path):

```powershell
$headers = @{ Authorization = "Bearer $env:AGE_GENDER_API_KEY" }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/analyze -Headers $headers -Form @{ image = Get-Item .\example.jpg }
```

## Configuration

Copy `.env.example` as a reference for environment variable names. The Streamlit process reads its OS environment and `.streamlit/secrets.toml`; the example file is not auto-loaded. Important controls include:

- `AGE_GENDER_MODEL`: `agevision_onnx` or `gra_net_compact`.
- `AGE_GENDER_MAX_UPLOAD_BYTES`, `AGE_GENDER_MAX_IMAGE_PIXELS`, `AGE_GENDER_MAX_FACES`.
- `AGE_GENDER_RETENTION_DAYS`: metadata retention, `0`, `30`, or `90` in Settings.
- `AGE_GENDER_CAMERA_FRAME_INTERVAL`, `AGE_GENDER_CAMERA_MAX_WIDTH`, `AGE_GENDER_CAMERA_MAX_HEIGHT`.
- `AGE_GENDER_API_KEY`, `AGE_GENDER_API_RATE_LIMIT` for the optional API.

## Storage and privacy

Uploads and camera frames are decoded in memory, are not saved as image files, and are not included in logs. Signed-in history stores prediction metadata scoped to the account; guest history is not persisted. Privacy settings support export and deletion of this app’s records. Deletion does not remove the user’s Google account or external database backups. Protect database volumes, `.streamlit/secrets.toml`, API keys, backups, and reverse-proxy logs. Do not use real people’s images for deployment checks without permission.

## Backup and rollback

Before changing application code, preserve the repository and verify the model files against `docs/model_hashes_before.json`. Roll back application changes from the recorded Git commit or the pre-upgrade archive. Model weights are immutable for this upgrade and must retain their recorded checksums.

## Hosting notes

For a VPS, Azure VM/App Service, AWS EC2/ECS, or Render, run behind a maintained HTTPS reverse proxy/load balancer, restrict inbound ports, configure OAuth with the public HTTPS callback URL, keep secrets in the host's secret manager, and mount the existing model files as read-only storage. Check that the host has enough disk and memory for the local checkpoints before deployment. Do not use an ephemeral filesystem for the SQLite database if saved history must persist; prefer managed PostgreSQL with TLS, least-privilege credentials, backups, and retention controls.

Hugging Face Spaces can run Streamlit or Docker, but verify its storage persistence, memory/disk limits, and secrets settings before using it for signed-in history or the large ONNX model. Never add gated model credentials to the image or repository. No external deployment provider was configured or tested as part of this local upgrade.

For version control, `.gitattributes` configures Git LFS patterns for model binaries. If models are pushed to a remote, install Git LFS and check the remote storage quotas and access controls first. This upgrade did not push any model or dataset files.

## Production checklist

- [ ] Set Google OIDC callback and secrets in the deployment secret manager.
- [ ] Use HTTPS and confirm cross-device camera permissions.
- [ ] Bind app/API behind the reverse proxy; keep origin services private.
- [ ] Set API key if API is enabled; rotate it periodically and protect the proxy.
- [ ] Configure a persistent, secured database and test restore procedures.
- [ ] Mount model weights read-only and confirm the registry checksums.
- [ ] Set retention and backup policies consistent with privacy requirements.
- [ ] Test consent, sign-in, guest behavior, account data export/deletion, and access isolation.
- [ ] Review logs and ensure they contain no images, tokens, or personal identifiers.
