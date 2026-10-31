"""Fine-tune the verified compact checkpoint for age-group accuracy.

The production checkpoint is never overwritten. Age-class samples receive
moderate inverse-frequency weights while gender and direct-age regression keep
unit weights, preserving the existing multi-task objective.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_utils import make_arrays
from gra_net import _GatedBlend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/gra_net_age_balanced.keras")
    parser.add_argument("--out", default="models/gra_net_age_tuned.keras")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    X, labels = make_arrays(args.data)
    split = np.load("results/dataset_split.npz")
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    age_class = np.asarray(labels["age_class"], dtype=np.int32)

    counts = np.bincount(age_class[train_idx], minlength=10).astype(np.float32)
    # Square-root inverse frequency is intentionally moderate: full inverse
    # frequency would make the rare 90+ classes dominate the model.
    weights = np.sqrt(counts.mean() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    train_age_weight = weights[age_class[train_idx]]
    val_age_weight = np.ones(len(val_idx), dtype=np.float32)

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        custom_objects={"_GatedBlend": _GatedBlend},
    )
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=args.lr),
        loss={
            "gender": "sparse_categorical_crossentropy",
            "age_class": "sparse_categorical_crossentropy",
            "age_regression": "mae",
        },
        metrics={
            "gender": "accuracy",
            "age_class": "accuracy",
            "age_regression": "mae",
        },
    )
    y = {
        "gender": np.asarray(labels["gender"], dtype=np.int32),
        "age_class": age_class,
        "age_regression": np.asarray(labels["age_regression"], dtype=np.float32),
    }
    train_y = {key: value[train_idx] for key, value in y.items()}
    val_y = {key: value[val_idx] for key, value in y.items()}
    train_weights = {
        "gender": np.ones(len(train_idx), dtype=np.float32),
        "age_class": train_age_weight,
        "age_regression": np.ones(len(train_idx), dtype=np.float32),
    }
    val_weights = {
        "gender": np.ones(len(val_idx), dtype=np.float32),
        "age_class": val_age_weight,
        "age_regression": np.ones(len(val_idx), dtype=np.float32),
    }
    print("Age-class weights:", [round(float(value), 3) for value in weights])
    output_names = ["gender", "age_class", "age_regression"]
    history = model.fit(
        X[train_idx],
        [train_y[key] for key in output_names],
        sample_weight=[train_weights[key] for key in output_names],
        validation_data=(
            X[val_idx],
            [val_y[key] for key in output_names],
            [val_weights[key] for key in output_names],
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_age_class_accuracy", mode="max", factor=0.5,
                patience=1, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_age_class_accuracy", mode="max", patience=2,
                restore_best_weights=True, verbose=1
            ),
        ],
        verbose=1,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    Path("results/age_tuned_history.json").write_text(
        __import__("json").dumps(
            {key: [float(value) for value in values] for key, values in history.history.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved age-tuned candidate: {output}")


if __name__ == "__main__":
    main()
