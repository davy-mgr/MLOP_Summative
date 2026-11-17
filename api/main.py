"""FastAPI service: predict, upload, train/retrain, status, metrics, visualisation data."""
import io
import sys
import threading
import time
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model as ml                      # noqa: E402
from src import prediction, visualization        # noqa: E402
from src.config import AUTO_RETRAIN_THRESHOLD, CLASSES, UPLOAD_DIR  # noqa: E402

app = FastAPI(title="UrbanSound8K Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STARTED = time.time()
latencies = deque(maxlen=500)          # seconds, for production monitoring
counters = {"predictions": 0, "errors": 0}

job_lock = threading.Lock()
job = {"running": False, "kind": None, "started_at": None, "finished_at": None,
       "epoch": 0, "epochs": 0, "message": "idle", "result": None, "error": None}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------- background jobs
def start_job(kind, runner, epochs):
    """runner(progress_cb) -> result dict. Returns False if a job is already running."""
    with job_lock:
        if job["running"]:
            return False
        job.update(running=True, kind=kind, started_at=_now(), finished_at=None, epoch=0,
                   epochs=epochs, message="starting (loading + preprocessing data)", result=None, error=None)

    def progress(e, logs):
        job.update(epoch=e + 1, message=f"epoch {e + 1}/{epochs} - val_acc {logs.get('val_accuracy', 0):.3f}")

    def run():
        try:
            res = runner(progress)
            prediction.reload_model()
            job.update(result={k: res[k] for k in ("version", "promoted", "n_train", "n_new_files")} |
                       {"accuracy": res["metrics"]["accuracy"], "f1_macro": res["metrics"]["f1_macro"]},
                       message="finished")
        except Exception as exc:                     # noqa: BLE001
            job.update(error=str(exc), message="failed")
        finally:
            job.update(running=False, finished_at=_now())

    threading.Thread(target=run, daemon=True).start()
    return True


# ----------------------------------------------------------------------- health / status
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    info = ml.current_model_info()
    lat = np.array(latencies) * 1000 if latencies else None
    return {
        "uptime_seconds": round(time.time() - STARTED, 1),
        "started_at": datetime.fromtimestamp(STARTED, timezone.utc).isoformat(timespec="seconds"),
        "model_loaded": info is not None,
        "model_version": info["version"] if info else None,
        "last_trained_at": info["trained_at"] if info else None,
        "model_accuracy": info["metrics"]["accuracy"] if info else None,
        "predictions_served": counters["predictions"],
        "errors": counters["errors"],
        "latency_ms_avg": round(float(lat.mean()), 1) if lat is not None else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat is not None else None,
        "pending_uploads": ml.count_uploads(),
        "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD,
        "job": job,
    }


@app.get("/metrics")
def metrics():
    reg = ml.load_registry()
    info = ml.current_model_info()
    return {"current": info, "versions": [
        {"version": v["version"], "kind": v["kind"], "promoted": v["promoted"],
         "trained_at": v["trained_at"], "n_train": v["n_train"],
         "accuracy": v["metrics"]["accuracy"], "f1_macro": v["metrics"]["f1_macro"]}
        for v in reg["versions"]]}


# ----------------------------------------------------------------------- prediction
@app.post("/predict")
def predict(file: UploadFile = File(...), spectrogram: bool = Query(False)):
    data = file.file.read()
    t0 = time.perf_counter()
    try:
        result = prediction.predict(io.BytesIO(data), return_spectrogram=spectrogram)
    except FileNotFoundError:
        raise HTTPException(503, "No trained model yet. POST /train first.")
    except Exception as exc:                         # noqa: BLE001
        counters["errors"] += 1
        raise HTTPException(400, f"Could not process audio: {exc}")
    dt_s = time.perf_counter() - t0
    latencies.append(dt_s)
    counters["predictions"] += 1
    result["latency_ms"] = round(dt_s * 1000, 1)
    return result


# ----------------------------------------------------------------------- bulk upload
def _save_wav(data: bytes, label: str):
    try:
        sf.info(io.BytesIO(data))
    except Exception:                                # noqa: BLE001
        return False
    d = UPLOAD_DIR / label
    d.mkdir(parents=True, exist_ok=True)
    (d / f"upload_{uuid.uuid4().hex}.wav").write_bytes(data)
    return True


@app.post("/upload")
def upload(files: list[UploadFile] = File(...), label: str = Form(None)):
    """Upload many .wav files (with `label`) or .zip files containing <class_name>/*.wav folders."""
    if label and label not in CLASSES:
        raise HTTPException(400, f"label must be one of {CLASSES}")
    saved, rejected = 0, []
    for f in files:
        data = f.file.read()
        name = f.filename or "file"
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m in z.namelist():
                    if not m.lower().endswith(".wav") or m.startswith("__MACOSX"):
                        continue
                    folder = Path(m).parent.name
                    lab = folder if folder in CLASSES else label
                    if lab and _save_wav(z.read(m), lab):
                        saved += 1
                    else:
                        rejected.append(m)
        elif name.lower().endswith(".wav") and label:
            if _save_wav(data, label):
                saved += 1
            else:
                rejected.append(name)
        else:
            rejected.append(name)

    pending = ml.count_uploads()
    auto = False
    if pending >= AUTO_RETRAIN_THRESHOLD:            # automatic retraining trigger
        auto = start_job("auto-retrain", lambda cb: ml.retrain_model(10, False, cb), 10)
    return {"saved": saved, "rejected": rejected, "pending_uploads": pending,
            "auto_retrain_started": auto}


# ----------------------------------------------------------------------- train / retrain
@app.post("/train")
def train(epochs: int = Query(40, ge=1, le=200)):
    if not start_job("train", lambda cb: ml.train_model(epochs, cb), epochs):
        raise HTTPException(409, "A training job is already running.")
    return {"started": True, "epochs": epochs}


@app.post("/retrain")
def retrain(epochs: int = Query(15, ge=1, le=200), from_scratch: bool = Query(False)):
    if ml.count_uploads() == 0:
        raise HTTPException(400, "No uploaded data waiting. Upload labelled files first.")
    if not start_job("retrain", lambda cb: ml.retrain_model(epochs, from_scratch, cb), epochs):
        raise HTTPException(409, "A training job is already running.")
    return {"started": True, "epochs": epochs, "from_scratch": from_scratch}


@app.get("/retrain/status")
def retrain_status():
    return job


# ----------------------------------------------------------------------- visualisation data
@app.get("/viz/class-distribution")
def viz_distribution():
    return visualization.class_distribution()


@app.get("/viz/features")
def viz_features():
    return visualization.compute_viz()["features"]


@app.get("/viz/mean-spectrograms")
def viz_specs():
    return visualization.compute_viz()["mean_spectrograms"]
