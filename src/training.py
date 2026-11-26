import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, optimizers

from .augmentation import augment_spectrogram
from .config import INPUT_SHAPE
from .preprocessing import features_cached


def _read_cached_feature(path):
    # Load one spectrogram from the existing on-disk feature cache.
    if hasattr(path, "item"):
        path = path.item()
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    return features_cached(str(path)).astype(np.float32, copy=False)


def _load_feature(path, label):
    # TensorFlow wrapper: only the current example is loaded into memory.
    feature = tf.numpy_function(
        _read_cached_feature,
        [path],
        Tout=tf.float32
    )
    feature.set_shape(INPUT_SHAPE)
    label.set_shape([])
    return feature, label


def _make_dataset(paths, labels, batch_size, shuffle=False, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Shuffle small path/label records, not full spectrogram arrays.
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 512),
            seed=42,
            reshuffle_each_iteration=True
        )

    # Decode one cached spectrogram at a time using a single worker.
    dataset = dataset.map(_load_feature, num_parallel_calls=1)

    if augment:
        dataset = dataset.map(
            augment_spectrogram,
            num_parallel_calls=1
        )

    dataset = dataset.batch(batch_size).prefetch(1)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options)


def fit_model(
    model, paths, y, groups,
    epochs=40, lr=1e-3,
    progress=None, batch_size=8,
    augment=True
):
    paths = np.asarray(paths, dtype=str)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    if len(paths) != len(y) or len(y) != len(groups):
        raise ValueError("paths, labels, and groups must have equal lengths.")
    if len(y) < 2:
        raise ValueError("At least two training examples are required.")

    tr, va = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=42
        ).split(paths, y, groups)
    )

    classes = np.unique(y[tr])
    class_weights = dict(zip(
        map(int, classes),
        compute_class_weight(
            "balanced",
            classes=classes,
            y=y[tr]
        )
    ))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3
        )
    ]

    if progress:
        training_callbacks.append(
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs:
                    progress(epoch, logs)
            )
        )

    train_ds = _make_dataset(
        paths[tr], y[tr],
        batch_size=batch_size,
        shuffle=True,
        augment=augment
    )
    val_ds = _make_dataset(
        paths[va], y[va],
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=training_callbacks,
        verbose=2
    )

    return {
        key: [round(float(value), 4) for value in values]
        for key, values in history.history.items()
        if key != "lr"
    }
