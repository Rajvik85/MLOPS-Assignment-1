# Screenshot Evidence Checklist

Add final screenshots in this folder before submission. The assignment asks for
proof of CI/CD, deployment, experiment tracking, and monitoring.

Required screenshots:

1. `01_mlflow_experiments.png` - MLflow experiment list showing Logistic Regression
   and Random Forest runs.
2. `02_mlflow_metrics.png` - MLflow run metrics and artifacts, including ROC curve
   and confusion matrix.
3. `03_github_actions_success.png` - GitHub Actions workflow run with lint, tests,
   training, and Docker build passing.
4. `04_docker_api_health.png` - Docker container running locally and `/health`
   returning healthy.
5. `05_docker_predict_response.png` - `/predict` endpoint returning prediction and
   confidence from the Docker container.
6. `06_kubernetes_pods_service.png` - `kubectl get pods` and `kubectl get svc`
   showing the API deployment and service.
7. `07_kubernetes_predict_response.png` - `/predict` response through the
   Kubernetes service or Minikube tunnel URL.
8. `08_prometheus_metrics.png` - `/metrics` endpoint showing Prometheus metrics.

Suggested short video flow:

1. Show the GitHub repository structure.
2. Run `pytest`, `black --check`, and `flake8`.
3. Show MLflow runs and metrics.
4. Start the API locally or through Docker.
5. Send a `/predict` request.
6. Show Kubernetes pod/service status.
7. Open `/metrics` to demonstrate monitoring.
