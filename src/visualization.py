"""Data behind the dashboard visualisations (class balance, spectral features, mean spectrograms)."""
import json
import random

import librosa

from .config import CACHE_DIR, CLASSES, SR, TEST_DIR, TRAIN_DIR, UPLOAD_DIR
from .preprocessing import load_audio, mel_db

VIZ_CACHE = CACHE_DIR / "viz.json"


def class_distribution():
    def count(root):
        return {c: (len(list((root / c).glob("*.wav"))) if (root / c).is_dir() else 0) for c in CLASSES}
    return {"train": count(TRAIN_DIR), "test": count(TEST_DIR), "uploads": count(UPLOAD_DIR)}


def compute_viz(sample_per_class=20, force=False):
    """Per-class spectral centroid / zero-crossing rate / RMS plus the mean log-mel spectrogram."""
    if VIZ_CACHE.exists() and not force:
        return json.loads(VIZ_CACHE.read_text())
    rng, feats, specs = random.Random(42), {}, {}
    for c in CLASSES:
        files = sorted((TRAIN_DIR / c).glob("*.wav"))
        if not files:
            continue
        cen, zcr, rms, mels = [], [], [], []
        for p in rng.sample(files, min(sample_per_class, len(files))):
            y = load_audio(str(p))
            cen.append(float(librosa.feature.spectral_centroid(y=y, sr=SR).mean()))
            zcr.append(float(librosa.feature.zero_crossing_rate(y).mean()))
            rms.append(float(librosa.feature.rms(y=y).mean()))
            mels.append(mel_db(y, normalize=False)[..., 0])
        feats[c] = {"spectral_centroid": cen, "zero_crossing_rate": zcr, "rms": rms}
        specs[c] = (sum(mels) / len(mels)).round(1).tolist()
    out = {"features": feats, "mean_spectrograms": specs}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_CACHE.write_text(json.dumps(out))
    return out
