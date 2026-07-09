# Heart Disease Prediction MLOps Assignment

This is my end-to-end MLOps assignment for predicting heart disease risk using the UCI Cleveland Heart Disease dataset.

In this project, I show the full workflow: downloading data, cleaning and preprocessing it, generating EDA plots, training models, tracking experiments in MLflow, serving the best model through FastAPI, packaging it with Docker, checking it through GitHub Actions, and preparing Kubernetes deployment files.

Repository link: `https://github.com/Rajvik85/MLOPS-Assignment-1`

## Project Flow

I explain the project in this order:

1. Download the raw dataset from UCI.
2. Clean the data and prepare preprocessing steps.
3. Generate EDA plots for understanding the dataset.
4. Train Logistic Regression and Random Forest models.
5. Track metrics, plots, and model artifacts in MLflow.
6. Save the best model as `models/best_model.joblib`.
7. Serve predictions using a FastAPI API and browser UI.
8. Run tests and quality checks.
9. Build and run the Docker image.
10. Show CI/CD through GitHub Actions.
11. Show deployment files for Kubernetes.

## Main Files

```text
src/download_data.py     Downloads the UCI dataset
src/preprocess.py        Cleans data and builds preprocessing pipeline
src/eda.py               Creates EDA plots
src/train.py             Trains models and logs MLflow experiments
src/api.py               Serves the trained model using FastAPI
tests/                   Unit tests for preprocessing, EDA, training, and API
Dockerfile               Docker image definition for the API
.github/workflows/       GitHub Actions CI pipeline
kubernetes/              Deployment and service YAML files
examples/                Sample prediction request JSON
screenshots/             Proof screenshots for report
output/                  Final report files
```

## Setup

I use a clean Python environment and install everything from `requirements.txt`.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Pipeline

Download the dataset:

```bash
python src/download_data.py
```

Generate EDA plots:

```bash
python -m src.eda --data data/raw/processed.cleveland.data --output-dir plots
```

Train both models and save the best one:

```bash
MLFLOW_ALLOW_FILE_STORE=true python -m src.train --model both --save-path models/best_model.joblib
```

Open MLflow:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --host 127.0.0.1 --port 5001
```

Then open:

```text
http://127.0.0.1:5001
```

## Run Tests

I use these commands to check formatting, linting, tests, and coverage:

```bash
black --check src/ tests/
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 src/ tests/ --count --max-line-length=127 --statistics
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Run The API Locally

Start the FastAPI app:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the browser UI:

```bash
open http://127.0.0.1:8000
```

Open Swagger docs:

```bash
open http://127.0.0.1:8000/docs
```

Test from terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/patient_request.json
curl http://127.0.0.1:8000/metrics | head
```

## Docker

Build the image:

```bash
docker build -t heart-disease-api:check .
```

Run the container:

```bash
docker rm -f heart-disease-api-check 2>/dev/null || true
docker run -d --name heart-disease-api-check -p 8766:8000 heart-disease-api:check
```

Test the container:

```bash
curl http://127.0.0.1:8766/health
curl -X POST http://127.0.0.1:8766/predict \
  -H "Content-Type: application/json" \
  -d @examples/patient_request.json
```

## Kubernetes

The Kubernetes files are in the `kubernetes/` folder.

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods -l app=heart-disease-api
kubectl get svc heart-disease-api-service
```

For local testing, I use port forwarding:

```bash
kubectl port-forward service/heart-disease-api-service 8080:80
```

Then test:

```bash
curl http://127.0.0.1:8080/health
```

## Report And Evidence

My final report and proof screenshots are stored here:

```text
output/MLOps_Assignment_01_Final_Report.docx
output/pdf/MLOps_Assignment_01_Final_Report.pdf
screenshots/
```

The report includes setup instructions, EDA and modelling choices, MLflow tracking summary, architecture diagram, CI/CD proof, deployment proof, and repository link.
