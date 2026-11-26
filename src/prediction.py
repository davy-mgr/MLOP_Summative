
import threading
import time

import numpy as np
import tensorflow as tf

from .config import CLASSES, MODEL_PATH
from .preprocessing import load_audio, mel_db

_model = None
_lock = threading.Lock()


def get_model():
    global _model

    print("MODEL: waiting for model lock", flush=True)
    with _lock:
        if _model is None:
            print(f"MODEL: loading from {MODEL_PATH}", flush=True)
            t0 = time.perf_counter()

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"No model at {MODEL_PATH}. Train one first."
                )

            _model = tf.keras.models.load_model(
                MODEL_PATH, compile=False
            )

            print(
                f"MODEL: loaded in {time.perf_counter() - t0:.2f}s",
                flush=True
            )
        else:
            print("MODEL: already loaded", flush=True)

        return _model


def reload_model():
    global _model
    with _lock:
        _model = None


def predict(source, return_spectrogram=False):
    print("PREDICTION: starting", flush=True)

    t0 = time.perf_counter()
    model = get_model()
    print(
        f"PREDICTION: get_model took {time.perf_counter() - t0:.2f}s",
        flush=True
    )

    t0 = time.perf_counter()
    y = load_audio(source)
    print(
        f"PREDICTION: load_audio took {time.perf_counter() - t0:.2f}s",
        flush=True
    )

    t0 = time.perf_counter()
    x = mel_db(y)
    print(
        f"PREDICTION: mel_db took {time.perf_counter() - t0:.2f}s",
        flush=True
    )

    t0 = time.perf_counter()
    probs = model(x[None], training=False).numpy()[0]
    print(
        f"PREDICTION: TensorFlow inference took {time.perf_counter() - t0:.2f}s",
        flush=True
    ) 

    i = int(np.argmax(probs))
    out = {
        "class": CLASSES[i],
        "confidence": float(probs[i]),
        "probabilities": {
            c: float(p) for c, p in zip(CLASSES, probs)
        },
    }

    if return_spectrogram:
        t0 = time.perf_counter()
        out["spectrogram"] = mel_db(
            y, normalize=False
        )[..., 0].round(1).tolist()
        print(
            f"PREDICTION: spectrogram generation took {time.perf_counter() - t0:.2f}s",
            flush=True
        )

    print("PREDICTION: finished", flush=True)
    return out
