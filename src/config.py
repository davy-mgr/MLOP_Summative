"""Central configuration for the UrbanSound8K pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"      # new data waiting to be used for retraining
CACHE_DIR = DATA_DIR / "cache"         # cached log-mel features + viz cache
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "urbansound_cnn.h5"      # the "live" model
REGISTRY_PATH = MODELS_DIR / "registry.json"       # version history + metrics

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]

# Audio / feature settings
SR = 22050
DURATION = 4.0                       # UrbanSound8K clips are <= 4 s
N_SAMPLES = int(SR * DURATION)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
INPUT_SHAPE = (N_MELS, 1 + N_SAMPLES // HOP_LENGTH, 1)   # (64, 173, 1)

# Retraining trigger: when this many new labelled files are waiting, retrain automatically
AUTO_RETRAIN_THRESHOLD = int(os.getenv("AUTO_RETRAIN_THRESHOLD", "50"))
