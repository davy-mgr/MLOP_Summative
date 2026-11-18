# UrbanSound8K Sound Classification - End-to-End ML Pipeline

**Video demo:** <YOUTUBE LINK>  
**Live app (UI):** <URL>  **API docs:** <URL>/docs  
**GitHub:** <REPO URL>

## Project description
Classifies 4-second urban audio clips into 10 classes (air conditioner, car horn, children playing, dog bark,
drilling, engine idling, gun shot, jackhammer, siren, street music) using log-mel spectrograms and a CNN.
The pipeline covers data acquisition, preprocessing, training, evaluation, a FastAPI service, a Streamlit
dashboard, bulk-upload + retraining (manual button and automatic trigger), Docker deployment and Locust load testing.

**Use case:** <ONE PARAGRAPH linking this to your Intro-to-ML summative use case, e.g. noise monitoring / urban safety>

## Repository layout
```
notebook/urbansound_classification.ipynb   EDA, preprocessing, training, evaluation, prediction, retraining
src/   config.py  data_acquisition.py  preprocessing.py  model.py  prediction.py  visualization.py
api/main.py            FastAPI service
app/streamlit_app.py   UI (predict / visualisations / upload + retrain / monitoring)
loadtest/                load test + scaling script
data/train, data/test  class-per-folder .wav files (built from UrbanSound8K)
models/urbansound_cnn.h5   live model;  registry.json = version history + metrics
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10-3.11
pip install -r requirements-dev.txt

# 1. Build data/train (folds 1-9) and data/test (fold 10) from your UrbanSound8K download
python -m src.data_acquisition --source /path/to/UrbanSound8K

# 2. Train the first model (or run the notebook)
python -c "from src.model import train_model; train_model(40)"

# 3. Run API and UI
uvicorn api.main:app --port 8000            # docs at http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```
Docker: `docker compose up --build` â†’ UI http://localhost:8501, API http://localhost:8000.

## API
| Endpoint | Purpose |
|---|---|
| `POST /predict` | one .wav â†’ class, confidence, probabilities |
| `POST /upload` | many .wav files + `label`, or .zip with `<class>/*.wav` folders |
| `POST /retrain` | fine-tune (or `from_scratch=true`) on train + uploaded data |
| `POST /train` | train from scratch on the original data |
| `GET /status`, `/health`, `/metrics`, `/retrain/status` | uptime, versions, latency, evaluation metrics, job progress |
| `GET /viz/*` | data behind the dashboard visualisations |

**Retraining trigger:** the Retrain button in the UI (`POST /retrain`) **or** automatically once
`AUTO_RETRAIN_THRESHOLD` (default 50) uploaded files are waiting. A retrained model is only promoted to the
live model if its test accuracy is within 2 points of the current model; every version is kept in `models/versions/`.

## Results
| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Baseline CNN | | | | |
| Optimised CNN | | | | |

## Flood-request simulation (Locust)
```bash
bash loadtest/run_loadtest.sh 50 10 60s      # 50 users, spawn 10/s, 60 s, for 1 / 2 / 3 API containers
```
| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Req/s |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

*(paste `results/summary.md` here and add a sentence on what changed as containers increased)*

## Deployment
Deployed on <Render / Railway / GCP Cloud Run>: two services built from `Dockerfile` (API) and `Dockerfile.ui`
(UI, `API_URL` = the API's public URL). For deployment, bake a small data subset into the image:
`python -m src.data_acquisition --source ... --max-per-class 30` (the full dataset is ~6 GB, so keep it out of Git).

**Note:** in-memory job state and the uploads folder are per-container, so with several replicas use the load-balanced
endpoint for `/predict` and send upload/retrain calls to a single instance (or use shared storage).
