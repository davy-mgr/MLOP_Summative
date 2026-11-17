"""Load the live model and predict a single audio clip."""
import threading

import numpy as np
import tensorflow as tf

from .config import CLASSES, MODEL_PATH
from .preprocessing import load_audio, mel_db

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"No model at {MODEL_PATH}. Train one first.")
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return _model


def reload_model():
    """Call after retraining so the API serves the new weights."""
    global _model
    with _lock:
        _model = None


def predict(source, return_spectrogram=False):
    """source: path or file-like (.wav). Returns predicted class, confidence and all probabilities."""
    model = get_model()
    y = load_audio(source)
    x = mel_db(y)
    with _lock:
        probs = model(x[None], training=False).numpy()[0]
    i = int(np.argmax(probs))
    out = {"class": CLASSES[i], "confidence": float(probs[i]),
           "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)}}
    if return_spectrogram:
        out["spectrogram"] = mel_db(y, normalize=False)[..., 0].round(1).tolist()
    return out
