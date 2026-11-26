
from tensorflow.keras import layers, models, regularizers

from .config import CLASSES, INPUT_SHAPE


def build_baseline(input_shape=INPUT_SHAPE, n_classes=len(CLASSES)):
    return models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="baseline_cnn")


def build_cnn(input_shape=INPUT_SHAPE, n_classes=len(CLASSES)):
    reg = regularizers.l2(1e-4)

    model = models.Sequential(
        [layers.Input(input_shape)],
        name="urbansound_cnn"
    )

    for filters in (32, 64, 128):
        model.add(layers.Conv2D(
            filters, 3,
            padding="same",
            activation="relu",
            kernel_regularizer=reg
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=reg
    ))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(n_classes, activation="softmax"))

    return model


# Backward compatibility for existing API and notebook imports.

from .augmentation import augment_spectrogram as _augment
from .training import fit_model
from .evaluation import evaluate, _clean, load_test_set as _test_set
from .model_registry import (
    load_registry,
    current_model_info,
    register_model as _register
)


# Lazy wrappers prevent circular imports with retraining.py.

def count_uploads():
    from .retraining import count_uploads as function
    return function()


def merge_uploads():
    from .retraining import merge_uploads as function
    return function()


def train_model(epochs=40, progress=None):
    from .retraining import train_model as function
    return function(epochs=epochs, progress=progress)


def retrain_model(
    epochs=15,
    from_scratch=False,
    progress=None
):
    from .retraining import retrain_model as function
    return function(
        epochs=epochs,
        from_scratch=from_scratch,
        progress=progress
    )
