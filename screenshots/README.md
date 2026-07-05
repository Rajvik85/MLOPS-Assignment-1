# Screenshot Evidence Index

This folder contains the selected and renamed screenshots for report compilation.
The original raw screenshots remain in `/Users/rajeshnatarajan/Documents/Snapshots`.

## MLflow Evidence

| File | Use in report |
| --- | --- |
| `mlflow/01_mlflow_experiments_list.png` | Shows the `Heart_Disease_Classification` experiment exists in MLflow. |
| `mlflow/02_mlflow_logistic_regression_overview.png` | Shows Logistic Regression run metrics, parameters, status, source, and run ID. |
| `mlflow/03_mlflow_random_forest_overview.png` | Shows Random Forest run metrics, parameters, status, source, and run ID. |
| `mlflow/04_mlflow_logistic_regression_metrics_charts.png` | Shows Logistic Regression metric charts. |
| `mlflow/05_mlflow_random_forest_metrics_charts.png` | Shows Random Forest metric charts. |
| `mlflow/06_mlflow_logistic_confusion_matrix.png` | Shows Logistic Regression confusion matrix artifact. |
| `mlflow/07_mlflow_logistic_roc_curve.png` | Shows Logistic Regression ROC curve artifact. |
| `mlflow/08_mlflow_random_forest_confusion_matrix.png` | Shows Random Forest confusion matrix artifact. |
| `mlflow/09_mlflow_random_forest_roc_curve.png` | Shows Random Forest ROC curve artifact. |

## API And Monitoring Evidence

| File | Use in report |
| --- | --- |
| `api/01_local_api_health_predict_metrics.png` | Shows local API server, `/health`, `/predict`, and `/metrics` usage. |
| `api/02_prometheus_metrics_detail.png` | Shows detailed Prometheus metrics output. |

## Docker Evidence

| File | Use in report |
| --- | --- |
| `docker/01_container_or_local_predict_response.png` | Shows successful `/predict` response. Use as supporting API evidence if Docker-specific proof is not required separately. |

## Kubernetes Evidence

| File | Use in report |
| --- | --- |
| `kubernetes/01_kubernetes_pods_service_health_predict.png` | Main Kubernetes proof: pods running, service exposed, `/health` successful, and `/predict` successful through port-forwarded service. |
| `kubernetes/02_kubernetes_pods_service_health_predict_backup.png` | Backup Kubernetes proof with the same evidence. |

## Still Needed If Report Requires It

| File | Reason |
| --- | --- |
| `github/01_github_actions_success.png` | A GitHub Actions success screenshot was not present in the attached Snapshots set. Capture the green Actions run from GitHub and save it with this name if the final report needs visual CI proof. |

