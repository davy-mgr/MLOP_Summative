import datetime as dt
import json
import shutil

import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers

from .config import (CACHE_DIR, CLASSES, INPUT_SHAPE, MODEL_PATH, MODELS_DIR,
                     REGISTRY_PATH, TEST_DIR, TRAIN_DIR, UPLOAD_DIR)
from .preprocessing import load_dataset

def build_baseline(input_shape=INPUT_SHAPE, n_classes=len(CLASSES)):
    return models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="baseline_cnn")


def build_cnn(input_shape=INPUT_SHAPE, n_classes=len(CLASSES)):
    reg = regularizers.l2(1e-4)
    m = models.Sequential([layers.Input(input_shape)], name="urbansound_cnn")
    for f in (32, 64, 128):
        m.add(layers.Conv2D(f, 3, padding="same", activation="relu", kernel_regularizer=reg))
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D())
        m.add(layers.Dropout(0.25))
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(128, activation="relu", kernel_regularizer=reg))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(n_classes, activation="softmax"))
    return m

def _augment(x, y):
    M, T = tf.shape(x)[0], tf.shape(x)[1]
    x = tf.roll(x, tf.random.uniform([], -(T // 5), T // 5 + 1, dtype=tf.int32), axis=1)
    f = tf.random.uniform([], 0, 9, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, M - f, dtype=tf.int32)
    t = tf.random.uniform([], 0, 31, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, T - t, dtype=tf.int32)
    fmask = ((tf.range(M) >= f0) & (tf.range(M) < f0 + f))[:, None, None]
    tmask = ((tf.range(T) >= t0) & (tf.range(T) < t0 + t))[None, :, None]
    x = tf.where(fmask | tmask, tf.zeros_like(x), x)       # 0 = the clip mean after standardising
    return x + tf.random.normal(tf.shape(x), stddev=0.05), y


def fit_model(model, X, y, groups, epochs=40, lr=1e-3, progress=None, batch_size=32, augment=True):
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(X, y, groups))
    classes = np.unique(y[tr])
    cw = dict(zip(map(int, classes), compute_class_weight("balanced", classes=classes, y=y[tr])))
    model.compile(optimizer=optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
    if progress:
        cbs.append(callbacks.LambdaCallback(on_epoch_end=lambda e, logs: progress(e, logs)))
    train_ds = tf.data.Dataset.from_tensor_slices((X[tr], y[tr])).shuffle(len(tr), seed=42)
    if augment:
        train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X[va], y[va])).batch(batch_size)
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=cw, callbacks=cbs, verbose=2)
    return {k: [round(float(v), 4) for v in vals] for k, vals in h.history.items() if k != "lr"}


def _clean(obj):
    return json.loads(json.dumps(obj, default=float))


def evaluate(model, X, y):
    probs = model.predict(X, batch_size=64, verbose=0)
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    labels = list(range(len(CLASSES)))
    return _clean({
        "accuracy": float((pred == y).mean()),
        "loss": float(tf.keras.losses.sparse_categorical_crossentropy(y, probs).numpy().mean()),
        "precision_macro": p, "recall_macro": r, "f1_macro": f,
        "report": classification_report(y, pred, labels=labels, target_names=CLASSES,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "n_test": int(len(y)),
    })

def load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"current": None, "versions": []}


def current_model_info():
    reg = load_registry()
    return next((v for v in reg["versions"] if v["version"] == reg["current"]), None)


def _register(model, history, metrics, kind, n_train, n_new, force_promote):
    reg = load_registry()
    version = len(reg["versions"]) + 1
    prev = current_model_info()
    promoted = bool(force_promote or prev is None or
                    metrics["accuracy"] >= prev["metrics"]["accuracy"] - 0.02)
    (MODELS_DIR / "versions").mkdir(parents=True, exist_ok=True)
    vfile = MODELS_DIR / "versions" / f"urbansound_cnn_v{version}.h5"
    model.save(vfile)
    if promoted:
        shutil.copy2(vfile, MODEL_PATH)
        reg["current"] = version
    entry = {"version": version, "kind": kind, "promoted": promoted, "file": vfile.name,
             "trained_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
             "n_train": int(n_train), "n_new_files": int(n_new),
             "metrics": metrics, "history": history}
    reg["versions"].append(entry)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    return entry


def count_uploads():
    return sum(1 for c in CLASSES if (UPLOAD_DIR / c).is_dir() for _ in (UPLOAD_DIR / c).glob("*.wav"))


def merge_uploads():
    n = 0
    for c in CLASSES:
        d = UPLOAD_DIR / c
        if d.is_dir():
            (TRAIN_DIR / c).mkdir(parents=True, exist_ok=True)
            for p in list(d.glob("*.wav")):
                shutil.move(str(p), str(TRAIN_DIR / c / p.name))
                n += 1
    (CACHE_DIR / "viz.json").unlink(missing_ok=True)   # visualisations must be recomputed
    return n


def _test_set():
    Xt, yt, _, _ = load_dataset([TEST_DIR], verbose=False)
    if len(yt) == 0:
        raise ValueError("data/test is empty - run `python -m src.data_acquisition` first.")
    return Xt, yt


def train_model(epochs=40, progress=None):
    X, y, g, _ = load_dataset([TRAIN_DIR])
    if len(y) == 0:
        raise ValueError("data/train is empty - run `python -m src.data_acquisition` first.")
    model = build_cnn()
    history = fit_model(model, X, y, g, epochs=epochs, lr=1e-3, progress=progress)
    metrics = evaluate(model, *_test_set())
    return _register(model, history, metrics, "train", len(y), 0, force_promote=True)


def retrain_model(epochs=15, from_scratch=False, progress=None):
    n_new = count_uploads()
    if n_new == 0:
        raise ValueError("No uploaded data waiting - upload labelled files first.")
    X, y, g, _ = load_dataset([TRAIN_DIR, UPLOAD_DIR])
    if from_scratch or not MODEL_PATH.exists():
        model, lr = build_cnn(), 1e-3
    else:
        model, lr = tf.keras.models.load_model(MODEL_PATH, compile=False), 1e-4
    history = fit_model(model, X, y, g, epochs=epochs, lr=lr, progress=progress)
    metrics = evaluate(model, *_test_set())
    entry = _register(model, history, metrics, "retrain", len(y), n_new, force_promote=False)
    merge_uploads()
    return entry