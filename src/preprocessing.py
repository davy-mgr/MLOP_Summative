"""Audio -> log-mel spectrogram features."""
import hashlib
from pathlib import Path

import librosa
import numpy as np

from .config import (CACHE_DIR, CLASSES, HOP_LENGTH, N_FFT, N_MELS, N_SAMPLES, SR)


def load_audio(source):
    """Load a path or file-like object as mono, SR Hz, padded/trimmed to a fixed length."""
    y, _ = librosa.load(source, sr=SR, mono=True, duration=4.0)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    return y[:N_SAMPLES]


def mel_db(y, normalize=True):
    """Log-mel spectrogram in dB, shape (N_MELS, frames, 1)."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    db = librosa.power_to_db(m, ref=1.0, top_db=80.0)
    if normalize:
        db = (db - db.mean()) / (db.std() + 1e-6)
    return db[..., None].astype("float32")


def audio_to_features(source):
    return mel_db(load_audio(source))


def _cache_file(path):
    key = hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()
    return CACHE_DIR / f"{key}.npy"


def features_cached(path):
    f = _cache_file(path)
    if f.exists():
        return np.load(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x = audio_to_features(str(path))
    np.save(f, x)
    return x


def list_files(root):
    """[(path, class_name)] for root/<class>/*.wav, ordered by CLASSES."""
    root, out = Path(root), []
    for c in CLASSES:
        d = root / c
        if d.is_dir():
            out += [(str(p), c) for p in sorted(d.glob("*.wav"))]
    return out


def load_dataset(roots, verbose=True):
    """Return X, y, groups, paths for one or more root folders.

    `groups` = freesound id of the source recording (first token of the UrbanSound8K filename),
    used so a validation split never shares a recording with the training split.
    """
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    files = [f for r in roots for f in list_files(r)]
    if not files:
        return np.empty((0,) + (N_MELS, 1 + N_SAMPLES // HOP_LENGTH, 1), "float32"), np.array([], int), np.array([]), []
    X = []
    for i, (p, _) in enumerate(files):
        X.append(features_cached(p))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(files)}")
    y = np.array([CLASSES.index(c) for _, c in files])
    groups = np.array([Path(p).stem.split("-")[0] for p, _ in files])
    return np.stack(X), y, groups, [p for p, _ in files]
