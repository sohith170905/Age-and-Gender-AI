# Implementation decisions

- Keep the existing Streamlit entrypoint named `app.py` and preserve `streamlit run app.py`.
- Keep the GRA_Net, AgeVision AI ONNX, YuNet/Haar, and emotion models and their output semantics unchanged. Treat model files as immutable; verify hashes at the end.
- Use the existing environment and dependencies first. Pin only versions already present unless a feature truly needs an additional dependency.
- Do not inspect, use, or copy any training/validation/test dataset. Regression checks use seeded synthetic tensors only.
- Keep inference local. Store metadata only for signed-in accounts; guest mode stores nothing. Never persist uploaded images, camera frames, crops, or embeddings.
- Keep the existing SQLite/PostgreSQL SQLAlchemy boundary unless a specific gap requires a minimal migration. Account deletion means deleting this app’s records and local consent/settings metadata; it cannot delete a Google account.
- Default retention is Off to preserve current history. Let signed-in users explicitly choose Off, 30, or 90 days.
- Preserve camera snapshot as a fallback while adding a WebRTC stream. Camera transport is required for real-time inference; frame handling remains in memory.
- Show product-friendly model and inference names in normal UI; keep implementation identifiers and exact hashes under a collapsed Technical Details section.
- If a model has no reliable metric file matching its version, display “Not available” and do not borrow metrics from another model or paper.
- The large production-upgrade checklist is implemented in small layers around the current inference routines. Existing inference preprocessing and output decoding are not refactored unless regression baselines prove equivalence.
