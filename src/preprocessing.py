import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from scipy.fft import rfft
from scipy.signal.windows import hann

import hashlib
from pathlib import Path

import librosa
import numpy as np

from .config import (CACHE_DIR, CLASSES, HOP_LENGTH, N_FFT, N_MELS, N_SAMPLES, SR)


def load_audio(source):
    # Read WAV data directly as float32.
    y, original_sr = sf.read(
        source,
        dtype="float32",
        always_2d=True
    )

    # Convert stereo or multichannel audio to mono.
    y = y.mean(axis=1)

    # Resample only when the original sample rate differs.
    if original_sr != SR:
        divisor = gcd(int(original_sr), int(SR))
        up = SR // divisor
        down = int(original_sr) // divisor
        y = resample_poly(y, up, down).astype(np.float32)

    # Keep a maximum of four seconds.
    y = y[:N_SAMPLES]

    # Pad shorter clips to the model's required input length.
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))

    return y.astype(np.float32)


# Build the same Slaney-normalized mel filter bank used by Librosa.
_MEL_FILTER = librosa.filters.mel(
    sr=SR,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=0.0,
    fmax=SR / 2,
    htk=False,
    norm="slaney",
    dtype=np.float32,
)

_STFT_WINDOW = hann(N_FFT, sym=False).astype(np.float32)


def mel_db(y, normalize=True):
    # Match Librosa's centered STFT with constant-zero padding.
    y = np.asarray(y, dtype=np.float32)
    y = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="constant")

    # Create overlapping frames without copying the entire signal.
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP_LENGTH]

    # Hann-windowed power spectrogram.
    spectrum = rfft(frames * _STFT_WINDOW, axis=-1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)

    # Apply the Librosa-compatible mel filter bank.
    mel = _MEL_FILTER @ power.T

    # Match librosa.power_to_db(ref=1.0, top_db=80.0).
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max() - 80.0)

    if normalize:
        db = (db - db.mean()) / (db.std() + 1e-6)

    return db[..., None].astype(np.float32)

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
    root, out = Path(root), []
    for c in CLASSES:
        d = root / c
        if d.is_dir():
            out += [(str(p), c) for p in sorted(d.glob("*.wav"))]
    return out


def load_dataset(roots, verbose=True):
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
