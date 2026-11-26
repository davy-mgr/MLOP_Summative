# UrbanSound8K Sound Classification — End-to-End MLOps Pipeline

**Video demonstration:** 

**Cloud API:** https://urbansound-api-44471.containers.snapdeploy.app

**API documentation:** https://urbansound-api-44471.containers.snapdeploy.app/docs

**Streamlit dashboard:** https://davy-mlopsummative.streamlit.app/


---

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for classifying environmental audio recordings into ten urban sound categories using the UrbanSound8K dataset.

The system accepts four-second audio recordings and predicts their sound category using a Convolutional Neural Network (CNN) trained on log-mel spectrogram representations.

The ten supported classes are:

- Air conditioner
- Car horn
- Children playing
- Dog bark
- Drilling
- Engine idling
- Gun shot
- Jackhammer
- Siren
- Street music

The project demonstrates the machine learning lifecycle, including data acquisition, audio preprocessing, model development, evaluation, model versioning, prediction serving, bulk data ingestion, retraining, containerization, cloud deployment, monitoring, and load testing.

### Problem and use case

Urban environments contain different sources of environmental noise, including traffic, construction activities, machinery, and public activity. Identifying these sound sources can support urban noise monitoring and environmental assessment.

This project demonstrates how machine learning can automatically classify short environmental audio recordings, providing a foundation for systems that monitor sound patterns and identify potentially disruptive noise sources.

## 2. System Architecture

The application consists of four main components:

1. **Data and preprocessing pipeline:** Acquires UrbanSound8K audio files, organizes training and testing datasets, extracts audio features, and caches processed spectrograms.
2. **CNN model:** Learns sound patterns from log-mel spectrograms and produces probabilities for the ten sound classes.
3. **FastAPI backend:** Provides prediction, upload, training, retraining, status, metrics, and visualization endpoints.
4. **Streamlit dashboard:** Provides an interactive interface for audio prediction, visualizations, bulk uploads, retraining, and monitoring.

### Prediction workflow

1. A user uploads a WAV audio file through the dashboard or API.
2. The API reads the audio and loads the trained model.
3. Audio is converted to mono, resampled to 22,050 Hz, and standardized to four seconds.
4. A 64-band log-mel spectrogram is extracted.
5. The CNN predicts probabilities for the ten classes.
6. The API returns the predicted class, confidence, class probabilities, and prediction latency.

## 3. Repository Structure

```text
MLOP_Summative/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── config.py
│   ├── data_acquisition.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── visualization.py
│
├── notebook/
│   └── urbansound_classification.ipynb
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   ├── urbansound_cnn.h5
│   ├── registry.json
│   └── versions/
│
├── loadtest/
│
├── results/
│   ├── summary.md
│   ├── containers_1_stats.csv
│   ├── containers_2_stats.csv
│   └── containers_3_stats.csv
│
├── evidence/
│   ├── evidence_streamlit_prediction.png
│   ├── evidence_api_status.png
│   ├── evidence_api_prediction.png
│   ├── evidence_api_status.json
│   └── evidence_api_prediction.json
│
├── Dockerfile
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
├── requirements-ui.txt
├── requirements-dev.txt
└── README.md
```

## 4. Dataset and Preprocessing

The project uses the UrbanSound8K dataset.

| Dataset partition | Audio files |
|---|---:|
| Training | 7,895 |
| Testing | 837 |
| Total | 8,732 |

The preprocessing pipeline performs:

- Audio loading and decoding.
- Conversion to mono.
- Resampling to 22,050 Hz.
- Trimming or zero-padding to four seconds.
- Extraction of 64-band mel spectrograms.
- Conversion to decibel scale.
- Feature normalization.
- Caching of processed features.

The resulting model input shape is:

```text
(64, 173, 1)
```

The dataset is not included in the Git repository. Users must obtain UrbanSound8K separately and prepare the training and testing directories.

## 5. Installation and Execution

### Requirements

- Python 3.11 recommended.
- Docker and Docker Compose.
- UrbanSound8K dataset.
- Git.

Docker is the recommended execution environment because it provides a consistent TensorFlow runtime.

### Dataset preparation

Download and extract UrbanSound8K from https://drive.google.com/drive/folders/1ESoHooW3ffwkYmXCVy1JLlqv_7zRsGdK?usp=sharing , then run:

```powershell
python -m src.data_acquisition --source "C:\path\to\UrbanSound8K"
```

This prepares the training and testing datasets.

### Local Python installation

For a compatible Python 3.11 environment:

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements-dev.txt
```

Train the initial model:

```powershell
python -c "from src.model import train_model; train_model(40)"
```

### Run the API

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

API documentation:

http://localhost:8000/docs

### Run the dashboard

In another terminal:

```powershell
$env:API_URL="http://localhost:8000"

streamlit run app/streamlit_app.py
```

Dashboard:

http://localhost:8501

### Docker deployment

Build and start the local services:

```powershell
docker compose up --build
```

The local services are available at:

| Component | Address |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| API through Nginx | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |

## 6. Model Development and Evaluation

A CNN architecture is trained using log-mel spectrogram features to distinguish between the ten environmental sound categories.

The model is evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Validation and test loss

### Model comparison

The following results were obtained from the project's model evaluation and retraining experiments.

| Metric | Version 1 | Version 2 |
|---|---:|---:|
| Accuracy | 78.38% | 79.81% |
| Macro precision | 80.38% | 81.41% |
| Macro recall | 80.39% | 82.10% |
| Macro F1-score | 79.48% | 81.29% |
| Loss | 0.8078 | 0.7168 |

Version 2 was produced through a retraining experiment incorporating three additional audio files.

Compared with Version 1:

- Accuracy improved by approximately 1.43 percentage points.
- Macro F1-score improved by approximately 1.81 percentage points.
- Macro recall improved by approximately 1.70 percentage points.
- Test loss decreased.

These results demonstrate the model versioning and retraining workflow.

**Deployment distinction:** The currently deployed cloud API reports model version 1, with an accuracy of approximately 78.38%. Version 2 results are reported as a separate retraining experiment.

## 7. Model Versioning and Retraining

The application supports model version management through a model registry.

The registry records the current model version, historical versions, and associated evaluation metrics.

The retraining workflow supports:

1. Uploading new labeled audio files.
2. Adding uploaded files to the training data.
3. Triggering retraining manually through the dashboard or API.
4. Evaluating the candidate model.
5. Promoting an eligible model to the current version.
6. Retaining historical model artifacts for version tracking and rollback.

The retraining endpoint supports fine-tuning and training from scratch.

The configured automatic retraining threshold is 50 uploaded files.

A candidate model is promoted only when its test accuracy is within two percentage points of the existing model's accuracy.

## 8. API Endpoints

The FastAPI service exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Predict the sound category of an uploaded WAV file |
| `/upload` | POST | Upload multiple labeled audio files or a ZIP archive |
| `/train` | POST | Train a model from the original dataset |
| `/retrain` | POST | Trigger model retraining |
| `/status` | GET | Retrieve application and model status |
| `/health` | GET | Check API health |
| `/metrics` | GET | Retrieve prediction and performance metrics |
| `/retrain/status` | GET | Check retraining progress |
| `/viz/*` | GET | Retrieve visualization data |

Interactive API documentation is available through `/docs`.

### Example prediction

Request:

```powershell
curl.exe -X POST `
  -F "file=@data\test\drilling\181624-4-5-0.wav" `
  http://localhost:8000/predict
```

Example prediction response:

```json
{
  "class": "drilling",
  "confidence": 0.9980151653,
  "latency_ms": 556.7
}
```

The API also returns probabilities for all ten sound categories.

## 9. Dashboard Functionality

The Streamlit dashboard provides:

- Single audio-file prediction.
- Predicted class and confidence display.
- Class probability visualizations.
- Audio-related visualizations.
- Bulk audio upload.
- Manual model retraining.
- Model version and status information.
- API health and uptime monitoring.
- Prediction and latency monitoring.

The dashboard communicates with the FastAPI backend through the configured `API_URL`.

## 10. Load Testing and Scalability

Locust was used to evaluate API performance under simulated concurrent user requests.

The experiment tested three API container configurations.

### Benchmark results

| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Max (ms) | Req/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 146 | 0 | 1,246 | 1,000 | 3,200 | 5,005 | 5.0 |
| 2 | 113 | 0 | 1,611 | 390 | 13,000 | 15,892 | 3.9 |
| 3 | 35 | 0 | 6,659 | 840 | 23,000 | 26,028 | 1.2 |

### Interpretation

All three configurations recorded zero failures in the reported Locust runs. However, increasing the number of API containers did not improve throughput in this experiment. The one-container configuration achieved the highest throughput at 5.0 requests per second and the lowest p95 latency at 3.2 seconds. The two- and three-container configurations experienced substantially higher tail latency and lower throughput.

### API deployment configuration

- Application: FastAPI.
- ASGI server: Uvicorn.
- Container port: 8000.
- Deployment branch: `main`.
- API Dockerfile: `Dockerfile.api`.
- Container memory: 512 MB.
- Container CPU allocation: 0.25 vCPU.

### Cloud verification results

The deployed API was tested using both the Streamlit dashboard and direct HTTP requests.

The health endpoint returned:

```json
{
  "status": "ok"
}
```

The API status endpoint confirmed:

- Model loaded successfully.
- Model version 1 active.
- Prediction requests served successfully.
- Zero recorded API errors.
- No pending uploads.

The status response after two successful prediction requests reported:

| Metric | Observed value |
|---|---:|
| Model loaded | True |
| Model version | 1 |
| Predictions served | 2 |
| Errors | 0 |
| Average latency | 732.9 ms |
| p95 latency | 1,266.2 ms |

A direct cloud prediction request also returned HTTP 200 with a valid prediction and class probability distribution. Its reported inference latency was 140.3 ms.

The API and Streamlit-to-API integration were therefore verified successfully.

Deployment screenshots and JSON responses are retained in the `evidence/` directory.

## 12. Limitations and Future Improvements

Potential improvements include:

- Increasing cloud CPU and memory resources for inference.
- Further optimizing audio decoding and preprocessing.
- Improving model inference efficiency.
- Introducing shared storage for uploaded audio and model artifacts.
- Using persistent storage for model versions and training data.
- Improving horizontal scaling and request distribution.
- Expanding retraining experiments with larger batches of newly labeled data.
- Monitoring prediction latency and resource utilization in production.
- Improving classification performance for commonly confused sound categories.

## 13. Conclusion

This project demonstrates an end-to-end MLOps workflow for environmental audio classification.

It combines audio preprocessing, CNN model training, model evaluation, version management, API serving, interactive visualization, bulk data ingestion, retraining, Docker containerization, cloud deployment, and load testing.

The experiments demonstrate measurable improvement through retraining and provide practical evidence of how deployment resources and container configuration affect inference performance and scalability.

The final cloud deployment successfully serves prediction requests through the FastAPI backend, with the Streamlit dashboard communicating with the deployed API.
