import json

import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from .config import CLASSES, TEST_DIR
from .preprocessing import load_dataset


def _clean(obj):
    return json.loads(json.dumps(obj, default=float))


def evaluate(model, X, y):
    probabilities = model.predict(
        X, batch_size=64, verbose=0
    )

    predictions = probabilities.argmax(axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    labels = list(range(len(CLASSES)))

    return _clean({
        "accuracy": float(
            (predictions == y).mean()
        ),
        "loss": float(
            tf.keras.losses.sparse_categorical_crossentropy(
                y, probabilities
            ).numpy().mean()
        ),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "report": classification_report(
            y,
            predictions,
            labels=labels,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=labels
        ).tolist(),
        "n_test": int(len(y))
    })


def load_test_set():
    X_test, y_test, _, _ = load_dataset(
        [TEST_DIR], verbose=False
    )

    if len(y_test) == 0:
        raise ValueError(
            "data/test is empty - run "
            "`python -m src.data_acquisition` first."
        )

    return X_test, y_test
