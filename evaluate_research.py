"""Evaluate the optional higher-resolution research candidate on the fixed split."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error

from train_research import collect_examples, make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    examples = collect_examples(args.data)
    split = np.load("results/dataset_split.npz")
    test_idx = split["test_idx"]
    labels = [examples[i] for i in test_idx]
    y_gender = np.asarray([row[1] for row in labels], dtype=np.int32)
    y_age_class = np.asarray([row[2] for row in labels], dtype=np.int32)
    y_age = np.asarray([row[3] for row in labels], dtype=np.float32)
    ds = make_dataset(examples, test_idx, args.size, args.batch_size, np.ones(10, dtype=np.float32), False)
    model = tf.keras.models.load_model(args.model, compile=False)
    outputs = model.predict(ds, verbose=1)
    gender_pred = np.argmax(np.asarray(outputs[0]), axis=1)
    age_class_pred = np.argmax(np.asarray(outputs[1]), axis=1)
    age_pred = np.clip(np.asarray(outputs[2]).reshape(-1), 0.0, 1.0) * 116.0
    print("Gender Accuracy:", f"{accuracy_score(y_gender, gender_pred) * 100:.2f}%")
    print("Age Classification Accuracy:", f"{accuracy_score(y_age_class, age_class_pred) * 100:.2f}%")
    print("Age MAE:", f"{mean_absolute_error(y_age, age_pred):.2f} years")
    result = Path("results/research_model_evaluation.txt")
    result.write_text(
        "Research candidate test results\n"
        f"Gender Accuracy: {accuracy_score(y_gender, gender_pred) * 100:.2f}%\n"
        f"Age Classification Accuracy: {accuracy_score(y_age_class, age_class_pred) * 100:.2f}%\n"
        f"Age MAE: {mean_absolute_error(y_age, age_pred):.2f} years\n",
        encoding="utf-8",
    )
    print("Saved:", result)


if __name__ == "__main__":
    main()
