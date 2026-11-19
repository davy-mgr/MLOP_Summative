import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = []
for p in sorted((ROOT / "data" / "test").rglob("*.wav"))[:: 20][:40]:
    SAMPLES.append((p.name, p.read_bytes()))
assert SAMPLES, "No .wav files in data/test - run src.data_acquisition first"


class AudioUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(10)
    def predict(self):
        name, data = random.choice(SAMPLES)
        self.client.post("/predict", files={"file": (name, data, "audio/wav")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
