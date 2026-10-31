import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from gra_net import build_gra_net, compile_model
from data_utils import make_arrays


def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None
    )

    parser.add_argument(
        "--out",
        default="models/gra_net_v2.keras"
    )

    args = parser.parse_args()


    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset...")

    X, y = make_arrays(
        args.data,
        args.max_items
    )


    print("\nDataset loaded.")

    print(
        "Total images:",
        len(X)
    )


    # ========================================================
    # LABELS
    # ========================================================

    gender_labels = np.asarray(
        y["gender"]
    )

    age_class_labels = np.asarray(
        y["age_class"]
    )

    age_regression_labels = np.asarray(
        y["age_regression"]
    )


    # ========================================================
    # CREATE INDICES
    # ========================================================

    indices = np.arange(
        len(X)
    )


    # ========================================================
    # FIRST SPLIT
    #
    # 85% temporary
    # 15% test
    # ========================================================

    train_val_idx, test_idx = train_test_split(

        indices,

        test_size=0.15,

        random_state=42,

        stratify=gender_labels

    )


    # ========================================================
    # SECOND SPLIT
    #
    # 70% train
    # 15% validation
    #
    # Validation is 15 / 85 of the remaining data.
    # ========================================================

    train_idx, val_idx = train_test_split(

        train_val_idx,

        test_size=(0.15 / 0.85),

        random_state=42,

        stratify=gender_labels[
            train_val_idx
        ]

    )


    # ========================================================
    # PRINT SPLIT INFORMATION
    # ========================================================

    print("\n========================================")
    print("DATASET SPLIT")
    print("========================================")

    print(
        f"Training:   {len(train_idx):,} "
        f"({len(train_idx) / len(X) * 100:.2f}%)"
    )

    print(
        f"Validation: {len(val_idx):,} "
        f"({len(val_idx) / len(X) * 100:.2f}%)"
    )

    print(
        f"Test:       {len(test_idx):,} "
        f"({len(test_idx) / len(X) * 100:.2f}%)"
    )

    print(
        f"Total:      {len(X):,}"
    )

    print("========================================\n")


    # ========================================================
    # SAVE SPLIT INDICES
    # ========================================================

    Path("results").mkdir(
        exist_ok=True
    )


    np.savez(

        "results/dataset_split.npz",

        train_idx=train_idx,

        val_idx=val_idx,

        test_idx=test_idx

    )


    print(
        "Saved dataset split:"
        " results/dataset_split.npz"
    )


    # ========================================================
    # PREPARE TRAINING DATA
    # ========================================================

    X_train = X[train_idx]
    X_val = X[val_idx]


    y_train = {

        "gender":
            gender_labels[train_idx],

        "age_class":
            age_class_labels[train_idx],

        "age_regression":
            age_regression_labels[train_idx]

    }


    y_val = {

        "gender":
            gender_labels[val_idx],

        "age_class":
            age_class_labels[val_idx],

        "age_regression":
            age_regression_labels[val_idx]

    }


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print("\nBuilding model...")

    model = build_gra_net()

    model = compile_model(
        model
    )


    model.summary()


    # ========================================================
    # TRAIN
    # ========================================================

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================\n")


    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=args.epochs,

        batch_size=args.batch_size,

        shuffle=True,

        verbose=1

    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    output_path = Path(
        args.out
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    model.save(
        output_path
    )


    print(
        "\nSaved model:",
        output_path
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_data = {

        key: [
            float(value)
            for value in values
        ]

        for key, values
        in history.history.items()

    }


    with open(
        "results/training_history_v2.json",
        "w"
    ) as file:

        json.dump(

            history_data,

            file,

            indent=2

        )


    print(
        "Saved training history:"
        " results/training_history_v2.json"
    )


    # ========================================================
    # FINAL TRAINING SUMMARY
    # ========================================================

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    for key, values in history.history.items():

        if values:

            print(
                f"{key}: "
                f"{values[-1]:.4f}"
            )

    print("========================================")


if __name__ == "__main__":

    main()