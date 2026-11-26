import shutil

import tensorflow as tf

from .config import (
    CACHE_DIR,
    CLASSES,
    MODEL_PATH,
    TRAIN_DIR,
    UPLOAD_DIR
)
from .evaluation import evaluate, load_test_set
from .model import build_cnn
from .model_registry import register_model
from .preprocessing import load_dataset
from .training import fit_model


def count_uploads():
    return sum(
        1
        for class_name in CLASSES
        if (UPLOAD_DIR / class_name).is_dir()
        for _ in (UPLOAD_DIR / class_name).glob("*.wav")
    )


def merge_uploads():
    count = 0

    for class_name in CLASSES:
        source_dir = UPLOAD_DIR / class_name

        if source_dir.is_dir():
            destination_dir = TRAIN_DIR / class_name
            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for file_path in list(
                source_dir.glob("*.wav")
            ):
                shutil.move(
                    str(file_path),
                    str(destination_dir / file_path.name)
                )
                count += 1

    (CACHE_DIR / "viz.json").unlink(missing_ok=True)

    return count


def train_model(epochs=40, progress=None):
    X, y, groups, _ = load_dataset([TRAIN_DIR])

    if len(y) == 0:
        raise ValueError(
            "data/train is empty - run "
            "`python -m src.data_acquisition` first."
        )

    model = build_cnn()

    history = fit_model(
        model, X, y, groups,
        epochs=epochs,
        lr=1e-3,
        progress=progress
    )

    metrics = evaluate(model, *load_test_set())

    return register_model(
        model, history, metrics,
        "train", len(y), 0,
        force_promote=True
    )


def retrain_model(
    epochs=15,
    from_scratch=False,
    progress=None
):
    new_files = count_uploads()

    if new_files == 0:
        raise ValueError(
            "No uploaded data waiting - upload labelled files first."
        )

    X, y, groups, _ = load_dataset(
        [TRAIN_DIR, UPLOAD_DIR]
    )

    if from_scratch or not MODEL_PATH.exists():
        model = build_cnn()
        learning_rate = 1e-3
    else:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        learning_rate = 1e-4

    history = fit_model(
        model, X, y, groups,
        epochs=epochs,
        lr=learning_rate,
        progress=progress
    )

    metrics = evaluate(model, *load_test_set())

    entry = register_model(
        model, history, metrics,
        "retrain", len(y), new_files,
        force_promote=False
    )

    merge_uploads()

    return entry
