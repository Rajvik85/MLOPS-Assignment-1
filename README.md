# Heart Disease Prediction MLOps Pipeline

This repository contains a production-grade, end-to-end Machine Learning Operations (MLOps) pipeline for predicting heart disease risk based on patient clinical health data. The pipeline uses the **UCI Heart Disease Cleveland Dataset** and demonstrates best practices in data engineering, model development, experiment tracking, API containerization, Kubernetes orchestration, CI/CD, and monitoring.

**GitHub username**: `Rajvik85`  
**Recommended repository URL**: `https://github.com/Rajvik85/MLOPS-Assignment-1`

---

## 1. Architectural Overview

The project structure is split into modular components:

```text
├── .github/workflows/
│   └── mlops_ci.yaml         # GitHub Actions CI workflow
├── data/
│   ├── raw/                  # Downloaded raw dataset from UCI Repository
├── kubernetes/
│   ├── deployment.yaml       # K8s Deployment configurations (scaling, probes)
│   └── service.yaml          # K8s Service configuration (LoadBalancer)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis with visualizations
│   └── 02_model_training.ipynb # Model training run demo and selection
├── src/
│   ├── __init__.py
│   ├── api.py                # FastAPI serving code with Prometheus metrics
│   ├── download_data.py      # Automated dataset downloader
│   ├── eda.py                # Reproducible EDA plot generator
│   ├── preprocess.py         # Sklearn preprocessing Pipeline
│   └── train.py              # MLflow integrated grid-tuning & training script
├── tests/
│   ├── __init__.py
│   ├── test_api.py           # Endpoint integration tests (Mocked models)
│   └── test_preprocess.py    # Unit tests for preprocessing functions
├── Dockerfile                # Secure, multi-stage production Docker build
├── examples/                 # Sample JSON request payloads
├── screenshots/              # API, MLflow, Docker, and Kubernetes evidence
├── requirements.txt          # Python package dependencies
├── Report.md                 # Markdown version of the assignment report
├── output/
│   ├── MLOps_Assignment_01_Final_Report.docx
│   └── pdf/MLOps_Assignment_01_Final_Report.pdf
└── README.md                 # Project execution guide
```

---

## 2. Installation & Setup

Ensure you have **Python 3.10+**, **Docker**, and a local Kubernetes cluster (like **Minikube** or **Docker Desktop Kubernetes**) installed.

### Virtual Environment Configuration
Initialize a Python virtual environment and install the required dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Execution Guide

### Step 3.1: Data Ingestion
Download the Heart Disease dataset directly from the UCI Repository:

```bash
python3 src/download_data.py
```
This saves the raw file to `data/raw/processed.cleveland.data`.

### Step 3.2: Exploratory Data Analysis (EDA)
Generate all required EDA plots for the report:

```bash
python3 -m src.eda
```

This creates class balance, missing value, histogram, correlation heatmap, and feature relationship plots under `plots/`.

You can also explore the dataset interactively using the Jupyter Notebook:
* Location: `notebooks/01_eda.ipynb`
* To run:
  ```bash
  jupyter notebook notebooks/01_eda.ipynb
  ```

### Step 3.3: Model Training & Tuning (with MLflow Tracking)
Train, tune, compare Logistic Regression and Random Forest, and save the best model by ROC-AUC:

```bash
MLFLOW_ALLOW_FILE_STORE=true python3 -m src.train --model both
```

You can also train one model at a time:

```bash
# Train Logistic Regression
MLFLOW_ALLOW_FILE_STORE=true python3 -m src.train --model lr

# Train Random Forest
MLFLOW_ALLOW_FILE_STORE=true python3 -m src.train --model rf
```
The script will perform 5-fold cross-validation grid search to find the optimal hyperparameters, log accuracy, precision, recall, F1-score, ROC-AUC, Confusion Matrix, and ROC Curve plots directly to **MLflow**, and save the selected production pipeline locally to `models/best_model.joblib`.

### Step 3.4: Visualizing Experiments in MLflow Dashboard
To launch the local MLflow dashboard and review metrics and logged curves, execute:
```bash
mlflow ui
```
Navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 4. Serving Predictions via API

We serve the best model pipeline via a production-grade **FastAPI** web application.

### Start the Local API Server
Start Uvicorn to run the server:
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

### Use the Browser Prediction UI
Open the user-facing prediction form in your browser:

```bash
open http://127.0.0.1:8000
```

The page lets a user enter patient health values and calls the same `/predict`
API endpoint used by automated systems. It displays the prediction, confidence,
and model version directly in the browser. The form loads a random realistic
sample patient record each time the page opens, and the **Load Random Sample**
button can be used to load another random record before prediction. The loaded
sample record is displayed on the page as JSON so users can see exactly which
values are being sent to the model.

You can also inspect the automatically generated API documentation:

```bash
open http://127.0.0.1:8000/docs
```

### Validate API Endpoints
1. **Health Check**:
   ```bash
   curl -X GET http://127.0.0.1:8000/health
   ```
2. **Prometheus Metrics**:
   ```bash
   curl -X GET http://127.0.0.1:8000/metrics
   ```
3. **Model Prediction**:
   Submit clinical features for inference using `curl`:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" \
        -H "Content-Type: application/json" \
        --data @examples/patient_request.json
   ```
   **Expected Response**:
   ```json
   {
     "prediction": 0,
     "confidence": 0.178,
     "model_version": "1.0.0"
   }
   ```

---

## 5. Automated Testing & Code Quality

Verify the codebase integrity by running lint checks and automated unit tests.

### Code Formatting and Linting
We enforce standard PEP-8 rules using `black` (formatter) and `flake8` (linter):

```bash
# Check formatting
black --check src/ tests/

# Check styling and syntax
flake8 src/ tests/ --ignore=E501
```

### Run Unit & Integration Tests
Execute the unit tests covering the preprocessing modules and the FastAPI routing endpoints (using mock classes to isolate endpoints):

```bash
python3 -m pytest tests/ --cov=src -v
```

---

## 6. Containerization & Kubernetes Orchestration

### Build Docker Image
To package the API with its dependencies, models, and preprocessors:

```bash
docker build -t heart-disease-api:latest -t heart-disease-api:1.0.1 .
```

### Run Docker Container Locally
```bash
docker run -p 8000:8000 heart-disease-api:latest
```
Open `http://127.0.0.1:8000` to use the browser UI, or test the container
endpoint using the same `curl` commands.

### Kubernetes Deployment
Deploy the containerized application to your local Kubernetes cluster (Minikube / Docker Desktop):

```bash
# Apply deployment config
kubectl apply -f kubernetes/deployment.yaml

# Apply service config (LoadBalancer)
kubectl apply -f kubernetes/service.yaml

# Check status of pods and service
kubectl get pods
kubectl get service heart-disease-api-service
```
For Minikube users, run `minikube service heart-disease-api-service` to retrieve the external API endpoint url.

---
