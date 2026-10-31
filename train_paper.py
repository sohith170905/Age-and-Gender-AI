"""Train a paper-protocol GRA_Net candidate without replacing production files."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_utils import make_arrays
from gra_net import build_paper_gra_net, build_paper_gra_net_fast, compile_paper_model


def _split_indices(gender, age_class, seed=42):
    """Create a reproducible 70/15/15 split with joint-label stratification."""
    indices = np.arange(len(gender))
    joint = gender.astype(np.int32) * 10 + age_class.astype(np.int32)
    counts = np.bincount(joint)
    if len(counts) and np.min(counts[counts > 0]) >= 2:
        stratify = joint
    else:
        stratify = gender

    train_val, test = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=stratify
    )
    train, val = train_test_split(
        train_val,
        test_size=(0.15 / 0.85),
        random_state=seed,
        stratify=stratify[train_val],
    )
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default="models/gra_net_paper_utkface.keras")
    parser.add_argument(
        "--variant", choices=("fast", "large"), default="fast",
        help="fast is the CPU-feasible paper-protocol variant; large is the 24M-parameter reference variant",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    print("Loading UTKFace-style data...")
    X, labels = make_arrays(args.data, args.max_items)
    gender = np.asarray(labels["gender"], dtype=np.int32)
    age_class = np.asarray(labels["age_class"], dtype=np.int32)
    ages = np.asarray(labels["age_regression"], dtype=np.float32)
    # Train the bounded sigmoid remainder head on a normalized 0-1 target;
    # evaluation converts it back to the paper's 0-9 remainder.
    age_remainder = (np.mod(ages, 10.0) / 9.0).astype(np.float32)

    train_idx, val_idx, test_idx = _split_indices(
        gender, age_class, seed=args.seed
    )
    Path("results").mkdir(exist_ok=True)
    np.savez(
        "results/paper_utkface_split.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    print(
        f"Images: {len(X):,} | train: {len(train_idx):,} | "
        f"validation: {len(val_idx):,} | test: {len(test_idx):,}"
    )

    y_train = {
        "gender": gender[train_idx],
        "age_class": age_class[train_idx],
        "age_remainder": age_remainder[train_idx],
    }
    y_val = {
        "gender": gender[val_idx],
        "age_class": age_class[val_idx],
        "age_remainder": age_remainder[val_idx],
    }

    builder = build_paper_gra_net_fast if args.variant == "fast" else build_paper_gra_net
    model = compile_paper_model(builder())
    model.summary()
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
    ]
    history = model.fit(
        X[train_idx],
        y_train,
        validation_data=(X[val_idx], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    Path("results/paper_utkface_history.json").write_text(
        json.dumps(history_data, indent=2), encoding="utf-8"
    )
    Path("results/paper_utkface_manifest.json").write_text(
        json.dumps(
            {
                "paper_protocol": "GRA_Net IEEE Access 2021",
                "dataset": "UTKFace",
                "input_size": [32, 32, 3],
                "age_labels": "decade classification + age remainder regression",
                "optimizer": "Nadam",
                "learning_rate": 0.001,
                "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "seed": args.seed,
                "model": str(output_path),
                "variant": args.variant,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved candidate model: {output_path}")


if __name__ == "__main__":
    main()
