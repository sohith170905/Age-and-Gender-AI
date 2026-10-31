"""Train an optional higher-resolution age/gender candidate.

This follows the practical pattern used by public age/gender projects such as
yu4u/age-gender-estimation and eshagarwal/face-analyzer: an ImageNet-pretrained
MobileNetV3 backbone with separate multi-task heads.  It is deliberately kept
separate from the verified GRA_Net checkpoint until the fixed test split shows
an improvement.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_utils import parse_utkface_filename


def collect_examples(root):
    examples = []
    for path in sorted(Path(root).rglob("*.jpg.chip.jpg")):
        age, gender = parse_utkface_filename(path)
        if age is None or gender not in (0, 1):
            continue
        examples.append((str(path), int(gender), int(np.clip(age // 10, 0, 9)), float(age)))
    if not examples:
        raise ValueError("No compatible UTKFace-style images found.")
    return examples


def make_dataset(examples, indices, size, batch_size, class_weights, training):
    paths = np.asarray([examples[i][0] for i in indices])
    gender = np.asarray([examples[i][1] for i in indices], dtype=np.int32)
    age_class = np.asarray([examples[i][2] for i in indices], dtype=np.int32)
    # Normalize the continuous age target for stable multi-task optimization.
    age = np.asarray([examples[i][3] / 116.0 for i in indices], dtype=np.float32)
    age_weight = class_weights[age_class].astype(np.float32)
    one = np.ones(len(indices), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, (gender, age_class, age), (one, age_weight, one))
    )

    def load(path, labels, sample_weights):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [size, size], antialias=True)
        image = tf.cast(image, tf.float32)
        if training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.08)
            image = tf.clip_by_value(image, 0.0, 255.0)
        return image, labels, sample_weights

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(indices), 4096), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(size):
    inputs = tf.keras.Input(shape=(size, size, 3), name="image")
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(size, size, 3),
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        pooling="avg",
        name="mobilenetv3_backbone",
    )
    backbone.trainable = False
    features = backbone(inputs, training=False)
    features = tf.keras.layers.Dropout(0.25)(features)
    gender = tf.keras.layers.Dense(2, activation="softmax", name="gender")(features)
    age_class = tf.keras.layers.Dense(10, activation="softmax", name="age_class")(features)
    age_regression = tf.keras.layers.Dense(1, activation="sigmoid", name="age_regression")(features)
    model = tf.keras.Model(inputs, [gender, age_class, age_regression], name="ResearchAgeGenderMobileNetV3")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss={
            "gender": "sparse_categorical_crossentropy",
            "age_class": "sparse_categorical_crossentropy",
            "age_regression": tf.keras.losses.Huber(delta=8.0),
        },
        loss_weights={"gender": 1.0, "age_class": 1.5, "age_regression": 0.35},
        metrics={"gender": "accuracy", "age_class": "accuracy", "age_regression": "mae"},
    )
    return model, backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="models/research_mobilenet_age.keras")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    examples = collect_examples(args.data)
    split = np.load("results/dataset_split.npz")
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    counts = np.bincount([examples[i][2] for i in train_idx], minlength=10).astype(np.float32)
    class_weights = np.sqrt(counts.mean() / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    print("Examples:", len(examples), "train:", len(train_idx), "validation:", len(val_idx))
    print("Age-class weights:", [round(float(x), 3) for x in class_weights])

    train_ds = make_dataset(examples, train_idx, args.size, args.batch_size, class_weights, True)
    val_ds = make_dataset(examples, val_idx, args.size, args.batch_size, np.ones(10, dtype=np.float32), False)
    model, backbone = build_model(args.size)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_age_class_accuracy", mode="max", factor=0.5, patience=1, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_age_class_accuracy", mode="max", patience=2, restore_best_weights=True, verbose=1
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

    if args.fine_tune_epochs > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss={
                "gender": "sparse_categorical_crossentropy",
                "age_class": "sparse_categorical_crossentropy",
                "age_regression": tf.keras.losses.Huber(delta=8.0),
            },
            loss_weights={"gender": 1.0, "age_class": 1.5, "age_regression": 0.35},
            metrics={"gender": "accuracy", "age_class": "accuracy", "age_regression": "mae"},
        )
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks, verbose=1)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print("Saved research candidate:", output)


if __name__ == "__main__":
    main()
