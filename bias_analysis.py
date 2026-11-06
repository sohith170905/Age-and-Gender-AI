"""Offline demographic and age bias analysis utility.

This script calculates validation and test evaluation metrics (bias and MAE)
across gender and age brackets from offline cached predictions.
It is an optional offline benchmarking tool and is not required by the live Streamlit app (app.py).
"""

from pathlib import Path
import sys
import numpy as np


def run_bias_analysis(predictions_path: str = "tmp/age_tuned_predictions.npz", dataset_path: str = "data/UTKFace"):
    pred_file = Path(predictions_path)
    data_dir = Path(dataset_path)

    missing_items = []
    if not pred_file.exists():
        missing_items.append(f"prediction cache '{pred_file}'")

    dataset_images = sorted(data_dir.glob("*.jpg.chip.jpg")) if data_dir.exists() else []
    if not dataset_images:
        missing_items.append(f"dataset directory '{data_dir}' containing '*.jpg.chip.jpg' images")

    if missing_items:
        msg = (
            "Optional offline bias analysis cannot run: missing "
            + " and ".join(missing_items)
            + ".\nNote: This is an offline benchmarking tool. "
            "For the interactive web application, please launch 'app.py' "
            "(Streamlit Community Cloud: Settings -> General -> Main file path -> app.py)."
        )
        # Graceful UI when launched via Streamlit Community Cloud
        try:
            import streamlit as st
            if hasattr(st, "runtime") and st.runtime.exists():
                st.title("AgeVision AI – Bias Analysis (Offline Utility)")
                st.warning("Offline evaluation artifacts are not present in this cloud deployment.")
                st.info(
                    "**Notice:** `bias_analysis.py` is an offline benchmarking tool. "
                    "The live application entrypoint is **`app.py`**.\n\n"
                    "**To fix your Streamlit Community Cloud deployment:**\n"
                    "1. Open your app on [share.streamlit.io](https://share.streamlit.io).\n"
                    "2. Click **Manage app** (bottom-right) → **⋮** (three dots) → **Settings**.\n"
                    "3. In the **General** tab, change **Main file path** to: `app.py`.\n"
                    "4. Click **Save** and the app will reload the full application."
                )
                st.stop()
        except Exception:
            pass

        print(msg, file=sys.stderr)
        return False

    z = np.load(pred_file)
    ages = np.array([int(p.stem.split("_")[0]) for p in dataset_images])
    genders = np.array([int(p.stem.split("_")[1]) for p in dataset_images])
    reg = z["age_reg"]
    val = z["val_idx"]
    test = z["test_idx"]

    for split, idx in [("val", val), ("test", test)]:
        print(split)
        for g in [0, 1]:
            m = genders[idx] == g
            err = reg[idx][m] - ages[idx][m]
            print("gender", g, "n", m.sum(), "bias", err.mean(), "mae", np.abs(err).mean())
        for lo, hi in [
            (0, 9),
            (10, 19),
            (20, 29),
            (30, 39),
            (40, 49),
            (50, 59),
            (60, 69),
            (70, 79),
            (80, 89),
            (90, 116),
        ]:
            m = (ages[idx] >= lo) & (ages[idx] <= hi)
            if m.sum():
                print(
                    f"{lo}-{hi}",
                    m.sum(),
                    "bias",
                    round((reg[idx][m] - ages[idx][m]).mean(), 2),
                    "mae",
                    round(np.abs(reg[idx][m] - ages[idx][m]).mean(), 2),
                )
    return True


if __name__ == "__main__":
    run_bias_analysis()
