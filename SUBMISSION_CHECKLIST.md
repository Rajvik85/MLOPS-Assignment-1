# MLOps Assignment 01 Submission Checklist

Use this checklist before uploading the GitHub repository and final report.

GitHub username: `Rajvik85`

GitHub repositories page: `https://github.com/Rajvik85?tab=repositories`

Recommended repository URL: `https://github.com/Rajvik85/MLOPS-Assignment-1`

Current status checked on 2026-07-05: the GitHub profile shows no public
repositories yet. Create a new public repository named `MLOPS-Assignment-1`
before final submission.

## Rubric Coverage

| Requirement | Marks | Evidence in this repository | Final action |
|---|---:|---|---|
| Data acquisition and EDA | 5 | `src/download_data.py`, `src/eda.py`, `notebooks/01_eda.ipynb`, `plots/eda_*.png` | Run `python -m src.eda` and include plots in report. |
| Feature engineering and model development | 8 | `src/preprocess.py`, `src/train.py`, `notebooks/02_model_training.ipynb` | Run both models and document best model choice. |
| Experiment tracking | 5 | `mlruns/`, MLflow logging in `src/train.py`, model/plot artifacts | Capture MLflow screenshots. |
| Model packaging and reproducibility | 7 | `models/best_model.joblib`, `requirements.txt`, sklearn `Pipeline` | Verify clean setup commands work. |
| CI/CD and automated testing | 8 | `.github/workflows/mlops_ci.yaml`, `tests/` | Push to GitHub and capture passing workflow screenshot. |
| Docker containerization | 5 | `Dockerfile`, `.dockerignore`, `src/api.py`, `examples/patient_request.json` | Build image, run container, test `/predict`. |
| Production deployment | 7 | `kubernetes/deployment.yaml`, `kubernetes/service.yaml` | Deploy to Minikube or Docker Desktop Kubernetes and capture screenshots. |
| Monitoring and logging | 3 | FastAPI logs, `/metrics` via `prometheus-fastapi-instrumentator` | Capture `/metrics` screenshot. |
| Documentation and reporting | 2 | `README.md`, `Report.md`, `Study_Notes.md` | Export final report as PDF/DOCX and add repository link. |

## Final Commands To Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/download_data.py
python -m src.eda
python -m src.train --model both
python -m pytest tests/ --cov=src -v
black --check src/ tests/
flake8 src/ tests/ --count --max-line-length=127 --statistics
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Test the API in another terminal:

```bash
curl -X GET http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/patient_request.json
curl -X GET http://127.0.0.1:8000/metrics
```

## Docker Proof

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/patient_request.json
```

## Kubernetes Proof

For Docker Desktop Kubernetes:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods
kubectl get svc heart-disease-api-service
kubectl port-forward service/heart-disease-api-service 8000:80
```

For Minikube:

```bash
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
minikube service heart-disease-api-service
```

## Must Not Forget

- Create the public repository `MLOPS-Assignment-1` under `Rajvik85`.
- Use `https://github.com/Rajvik85/MLOPS-Assignment-1` as the final repository URL.
- Add actual screenshots into `screenshots/`.
- Record a short pipeline video and include its link or file as required by the
  instructor.
- Do not upload `venv/`, `.pytest_cache/`, or `__pycache__/` to GitHub.
