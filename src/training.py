import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, optimizers

from .augmentation import augment_spectrogram


def fit_model(
    model, X, y, groups,
    epochs=40, lr=1e-3,
    progress=None, batch_size=32,
    augment=True
):
    tr, va = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=42
        ).split(X, y, groups)
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

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X[tr], y[tr])
    ).shuffle(len(tr), seed=42)

    if augment:
        train_ds = train_ds.map(
            augment_spectrogram,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    train_ds = train_ds.batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (X[va], y[va])
    ).batch(batch_size)

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
