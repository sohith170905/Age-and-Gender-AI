import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)

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
        "--model",
        required=True
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None
    )

    args = parser.parse_args()


    # ========================================================
    # CHECK SPLIT FILE
    # ========================================================

    split_path = Path(
        "results/dataset_split.npz"
    )


    if not split_path.exists():

        raise FileNotFoundError(

            "Dataset split not found.\n"
            "Run train.py first so that "
            "results/dataset_split.npz "
            "is created."

        )


    # ========================================================
    # LOAD SPLIT
    # ========================================================

    split = np.load(
        split_path
    )


    train_idx = split[
        "train_idx"
    ]

    val_idx = split[
        "val_idx"
    ]

    test_idx = split[
        "test_idx"
    ]


    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset...")

    X, y = make_arrays(

        args.data,

        args.max_items

    )


    print(
        "Total images:",
        len(X)
    )


    # ========================================================
    # TEST DATA ONLY
    # ========================================================

    X_test = X[
        test_idx
    ]


    y_gender = np.asarray(
        y["gender"]
    )[test_idx]


    y_age_class = np.asarray(
        y["age_class"]
    )[test_idx]


    y_age_regression = np.asarray(
        y["age_regression"]
    )[test_idx]


    print(
        "\nTest images:",
        len(X_test)
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading model..."
    )


    model = tf.keras.models.load_model(
        args.model
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\nRunning predictions..."
    )


    predictions = model.predict(

        X_test,

        batch_size=32,

        verbose=1

    )


    # ========================================================
    # GENDER
    # ========================================================

    gender_probabilities = np.asarray(
        predictions[0]
    )


    gender_predictions = np.argmax(

        gender_probabilities,

        axis=1

    )


    gender_accuracy = accuracy_score(

        y_gender,

        gender_predictions

    )


    # ========================================================
    # AGE CLASSIFICATION
    # ========================================================

    age_class_probabilities = np.asarray(
        predictions[1]
    )


    age_class_predictions = np.argmax(

        age_class_probabilities,

        axis=1

    )


    age_class_accuracy = accuracy_score(

        y_age_class,

        age_class_predictions

    )


    # ========================================================
    # AGE REGRESSION
    # ========================================================

    age_predictions = np.asarray(

        predictions[2]

    ).reshape(-1)


    age_mae = mean_absolute_error(

        y_age_regression,

        age_predictions

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    gender_matrix = confusion_matrix(

        y_gender,

        gender_predictions

    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("========================================")
    print("FINAL UNSEEN TEST RESULTS")
    print("========================================")

    print(
        f"\nTest Samples: "
        f"{len(X_test):,}"
    )

    print(
        f"\nGender Accuracy: "
        f"{gender_accuracy * 100:.2f}%"
    )

    print(
        f"Age Classification Accuracy: "
        f"{age_class_accuracy * 100:.2f}%"
    )

    print(
        f"Age MAE: "
        f"{age_mae:.2f} years"
    )


    print(
        "\nGender Confusion Matrix:"
    )

    print(
        gender_matrix
    )


    print(
        "\nGender Classification Report:"
    )

    print(

        classification_report(

            y_gender,

            gender_predictions,

            target_names=[
                "Male",
                "Female"
            ],

            digits=4

        )

    )


    print(
        "\n========================================"
    )

    print(
        "TRAIN / VALIDATION / TEST"
    )

    print(
        f"Train:      {len(train_idx):,}"
    )

    print(
        f"Validation: {len(val_idx):,}"
    )

    print(
        f"Test:       {len(test_idx):,}"
    )

    print(
        "========================================"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )


    with open(

        "results/final_test_results.txt",

        "w"

    ) as file:

        file.write(
            "FINAL UNSEEN TEST RESULTS\n"
        )

        file.write(
            "==========================\n\n"
        )

        file.write(
            f"Test Samples: "
            f"{len(X_test)}\n"
        )

        file.write(
            f"Gender Accuracy: "
            f"{gender_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Age Classification Accuracy: "
            f"{age_class_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Age MAE: "
            f"{age_mae:.2f} years\n\n"
        )

        file.write(
            "Gender Confusion Matrix:\n"
        )

        file.write(
            str(gender_matrix)
        )

        file.write(
            "\n\nClassification Report:\n"
        )

        file.write(

            classification_report(

                y_gender,

                gender_predictions,

                target_names=[
                    "Male",
                    "Female"
                ],

                digits=4

            )

        )


    print(
        "\nSaved:"
        " results/final_test_results.txt"
    )


if __name__ == "__main__":

    main()