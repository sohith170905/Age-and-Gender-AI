# AgeVision AI production upgrade report

**Project folder:** `C:\Users\sohit\Downloads\GRA_Net_Complete_Project\GRA_Net_Complete`

**Branch:** `realworld-upgrade`
**Verification date:** 2026-10-31

## Scope and safety

- Kept the existing Streamlit app, GRA_Net checkpoints, AgeVision ONNX checkpoint, YuNet detector, and expression checkpoint. No model was trained, converted, quantized, or exported.
- Did not read training, validation, or test dataset images and did not download datasets or weights.
- Image bytes and camera frames are processed in memory; prediction history stores user-scoped metadata only.
- The tracked model set contains 16 files. Their pre-upgrade and post-upgrade SHA-256 values and sizes match exactly (zero changed, added, or missing files). See [`model_hashes_before.json`](model_hashes_before.json) and [`model_hashes_after.json`](model_hashes_after.json).


| Model file | SHA-256 before | SHA-256 after | Bytes |
|---|---|---|---:|
| `models/emotion/emotion_model.keras` | `D1EF71D7A88D3C010ACAEBB175A8EA88EF04A9250BFDEE0908FE47D6ED58FE0B` | `D1EF71D7A88D3C010ACAEBB175A8EA88EF04A9250BFDEE0908FE47D6ED58FE0B` | 5116389 |
| `models/emotion/expression_classes.json` | `D743888CA8CB8C7EEF33EF027922E6C05942E4424B5670AFCE627E77AB2700FB` | `D743888CA8CB8C7EEF33EF027922E6C05942E4424B5670AFCE627E77AB2700FB` | 132 |
| `models/face_detection_yunet_2023mar.onnx` | `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4` | `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4` | 232589 |
| `models/faceage_dino_fp32.onnx` | `B0FEFB4473978DAE8128803D4577DE2586F974198991F2C6252A5609DFFB3233` | `B0FEFB4473978DAE8128803D4577DE2586F974198991F2C6252A5609DFFB3233` | 1219158136 |
| `models/gra_net_age_balanced.keras` | `D39867EBAC9E06DD0429CD155B8E7A935B1E215B9B274391517B2F3A23C2BD43` | `D39867EBAC9E06DD0429CD155B8E7A935B1E215B9B274391517B2F3A23C2BD43` | 55392409 |
| `models/gra_net_age_focus_candidate.keras` | `251EB2F1616EB44DCE18A2B5E84026F5B251495FB290E0D9E00E4782C64685BD` | `251EB2F1616EB44DCE18A2B5E84026F5B251495FB290E0D9E00E4782C64685BD` | 55392547 |
| `models/gra_net_age_robust.keras` | `B064933102FFBEFB436D2A99FF47B6059599048A8B2592DE03D84B5A13C07E8B` | `B064933102FFBEFB436D2A99FF47B6059599048A8B2592DE03D84B5A13C07E8B` | 55392547 |
| `models/gra_net_age_tuned.keras` | `4F6ECCDDB4A58D81620472714D80D7056271641A47B862DBBEAA9E57E54BF7E0` | `4F6ECCDDB4A58D81620472714D80D7056271641A47B862DBBEAA9E57E54BF7E0` | 55392357 |
| `models/gra_net_paper_utkface.keras` | `43B2E6E43564A11166B366C3EDF6CFB1F71F56D2FDB37EA683C827B175F3FCAE` | `43B2E6E43564A11166B366C3EDF6CFB1F71F56D2FDB37EA683C827B175F3FCAE` | 92794508 |
| `models/gra_net_v2.keras` | `069EB8DCE4A0CCF784B9D4508984B46F4CDA2FC12C8525EC960AE460969FD56A` | `069EB8DCE4A0CCF784B9D4508984B46F4CDA2FC12C8525EC960AE460969FD56A` | 55395367 |
| `models/gra_net.keras` | `3C7C15C9ABDD8A2ACDC476350E912E3E81A9F8EBFD2C0EB810BF068CE22BB9D5` | `3C7C15C9ABDD8A2ACDC476350E912E3E81A9F8EBFD2C0EB810BF068CE22BB9D5` | 55395367 |
| `models/paper_fast_benchmark.keras` | `1E7DAE3E0264590C765F19DF92A31FB63E0C150A47358F083DDB8A9929C6D8AA` | `1E7DAE3E0264590C765F19DF92A31FB63E0C150A47358F083DDB8A9929C6D8AA` | 92794508 |
| `models/paper_fast_smoke.keras` | `7C6A5D51DDC02BA0A372A0EE36B59271E03A3480F9793CB2F27BF2467B7C51A0` | `7C6A5D51DDC02BA0A372A0EE36B59271E03A3480F9793CB2F27BF2467B7C51A0` | 92794508 |
| `models/paper_smoke_bounded.keras` | `50E01EFFD00867CC1A97850E5F99434E4A9715724EE50D8998374D66AE49ACB6` | `50E01EFFD00867CC1A97850E5F99434E4A9715724EE50D8998374D66AE49ACB6` | 291843736 |
| `models/paper_smoke.keras` | `6C278F1784169DC891B185655668284DB1185EA1FB69CEA101DC7127DD46B28F` | `6C278F1784169DC891B185655668284DB1185EA1FB69CEA101DC7127DD46B28F` | 291843735 |
| `models/research_mobilenet_age.keras` | `C6C8F8C05D24C9D01CE086A14663656AE976B96726B588D5AE45ACE20633CFA6` | `C6C8F8C05D24C9D01CE086A14663656AE976B96726B588D5AE45ACE20633CFA6` | 9040298 |
- Added `.gitattributes` rules for the 15 binary model weights so Git LFS stores model objects without changing the checked-out model bytes. Dataset folders, secrets, local databases, logs, backups, caches, and temporary images were excluded from staging.

## Delivered

- Registry-driven model metadata and bounded process configuration; shared in-memory upload, quality, prediction, and export services.
- Existing pages and workflows preserved, with batch limits, uncertain-quality notices, prediction timing, summarized results, and privacy-safe result exports.
- Live WebRTC camera processing at reduced resolution and frame rate, with explicit camera guidance, face/prediction box alignment, temporal smoothing, and snapshot fallback.
- Consent-gated, per-user history and analytics, retention preferences, export/delete actions, and account data deletion.
- Optional bearer-authenticated FastAPI routes with bounded request bodies, input validation, rate limiting, and structured errors.
- Windows start scripts, pinned dependencies, Streamlit configuration, Docker files, setup/deployment documentation, and API client example.
- Docker uses a non-root user, read-only root filesystem, writable data/log volumes, read-only model mount, reduced capabilities, and temporary cache paths under `/tmp`.

## Verification

- Python compile check: passed.
- Dependency consistency (`pip check`): passed; no broken requirements.
- Synthetic-only pytest suite: **17 passed in 34.41 seconds**. Checks cover registry hashes, input validation, image quality, shared inference orchestration, exports, live tracker behavior, frozen synthetic inference outputs for the existing age/gender/expression models, API auth/validation/rate limiting, user-scoped storage isolation, and guest navigation through all Streamlit sections in light and dark themes.
- Local Streamlit preview: `http://127.0.0.1:8502/`; Streamlit health endpoint returned HTTP 200.
- SHA-256 verification: **16/16 model files unchanged**.
- Git LFS: installed and configured; all 15 binary model files are staged through LFS.

## Limits and manual checks

- Docker is not installed in this environment, so image build and container startup could not be verified here.
- Automated tests do not grant browser camera permission or exercise a real camera, Google OAuth provider, external HTTPS reverse proxy, or PostgreSQL server. Verify these in the intended deployment environment.
- No dataset was evaluated, so this upgrade makes no new accuracy claim and does not certify a published accuracy target. The displayed prediction confidence remains model-derived and is not independently calibrated.
- Run Git LFS push after connecting a remote that supports LFS. No remote push or external deployment was performed.

## Start locally

From the project folder, run `streamlit run app.py`. The helper `scripts\run_app.ps1` starts the same app with the project virtual environment.
