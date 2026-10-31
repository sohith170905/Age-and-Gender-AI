"""Evaluate a paper-protocol candidate with the paper's age/gender metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error

from data_utils import make_arrays
# Import the model module so the registered custom gate is available when a
# saved paper candidate is deserialized.
import gra_net  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="results/paper_utkface_split.npz")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default="results/paper_utkface_evaluation.json")
    args = parser.parse_args()

    split = np.load(args.split)
    X, labels = make_arrays(args.data, args.max_items)
    test_idx = split["test_idx"]
    model = tf.keras.models.load_model(
        args.model, compile=False, custom_objects={"_GatedBlend": gra_net._GatedBlend}
    )
    outputs = model.predict(X[test_idx], batch_size=32, verbose=1)

    gender_true = np.asarray(labels["gender"])[test_idx]
    age_class_true = np.asarray(labels["age_class"])[test_idx]
    age_true = np.asarray(labels["age_regression"], dtype=np.float32)[test_idx]
    gender_pred = np.argmax(np.asarray(outputs[0]), axis=1)
    age_class_pred = np.argmax(np.asarray(outputs[1]), axis=1)
    remainder_pred = np.clip(np.asarray(outputs[2]).reshape(-1) * 9.0, 0, 9)
    age_pred = np.clip(age_class_pred * 10 + np.rint(remainder_pred), 0, 116)

    results = {
        "dataset": "UTKFace",
        "model": args.model,
        "test_samples": int(len(test_idx)),
        "gender_accuracy": float(accuracy_score(gender_true, gender_pred)),
        "age_class_accuracy": float(accuracy_score(age_class_true, age_class_pred)),
        "exact_age_accuracy": float(accuracy_score(age_true.astype(int), age_pred.astype(int))),
        "age_mae": float(mean_absolute_error(age_true, age_pred)),
        "paper_targets": {
            "gender_accuracy": 0.992,
            "age_accuracy": 0.937,
            "age_mae": 1.07,
        },
    }
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
