"""
Model training and experiment tracking script.
Trains Logistic Regression and Random Forest models, performs hyperparameter tuning,
evaluates performance metrics, and logs metadata and artifacts to MLflow.
"""

import os
import argparse
import logging
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

import mlflow
import mlflow.sklearn

from src.preprocess import (
    load_and_clean_data,
    build_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train")


def create_plots(y_test, y_pred, y_prob, run_name):
    """
    Generates and saves Confusion Matrix and ROC Curve plots for evaluation.
    """
    plot_paths = {}
    os.makedirs("plots", exist_ok=True)

    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Disease", "Disease"]
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Confusion Matrix - {run_name}")
    cm_path = f"plots/{run_name}_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    plot_paths["confusion_matrix"] = cm_path

    # 2. ROC Curve Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc_val:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {run_name}")
    plt.legend(loc="lower right")
    roc_path = f"plots/{run_name}_roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    plot_paths["roc_curve"] = roc_path

    return plot_paths


def train_and_evaluate(raw_data_path, model_type="rf", mlflow_tracking_uri=None):
    """
    Main training function: loads data, splits, preprocesses, tunes, and tracks with MLflow.
    """
    # 1. Set MLflow tracking URI if provided
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set experiment
    mlflow.set_experiment("Heart_Disease_Classification")

    # 2. Load and clean data
    logger.info("Loading and cleaning data...")
    df = load_and_clean_data(raw_data_path)

    # Separate features and target
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_COLUMN]

    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 3. Create preprocessing and training pipeline
    preprocessor = build_preprocessor()

    if model_type == "lr":
        run_name = "Logistic_Regression"
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"],
        }
    elif model_type == "rf":
        run_name = "Random_Forest"
        classifier = RandomForestClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    full_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        logger.info(f"Starting MLflow run: {run_name}")

        # 4. Hyperparameter tuning via GridSearchCV
        logger.info(f"Tuning hyperparameters for {model_type}...")
        grid_search = GridSearchCV(
            full_pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_

        logger.info(f"Best hyperparameters: {best_params}")

        # Log best parameters to MLflow
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # 5. Evaluate on test set
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        logger.info(
            "Evaluation Metrics: "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"ROC-AUC={roc_auc:.4f}"
        )

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # 6. Generate plots and log them as artifacts
        plots = create_plots(y_test, y_pred, y_prob, run_name)
        for plot_name, plot_path in plots.items():
            mlflow.log_artifact(plot_path, artifact_path="plots")

        # 7. Log model to MLflow
        mlflow.sklearn.log_model(best_pipeline, name="model")
        logger.info("Model logged to MLflow.")

        # Return best pipeline and evaluation metrics
        return best_pipeline, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }


def train_all_and_select_best(raw_data_path, selection_metric="roc_auc"):
    """
    Train all required models, compare them, and return the best pipeline.
    """
    results = []
    for model_type in ["lr", "rf"]:
        pipeline, metrics = train_and_evaluate(raw_data_path, model_type=model_type)
        results.append(
            {
                "model_type": model_type,
                "pipeline": pipeline,
                "metrics": metrics,
            }
        )

    selected = max(results, key=lambda item: item["metrics"][selection_metric])
    logger.info(
        "Selected %s as the production model using %s=%.4f",
        selected["model_type"],
        selection_metric,
        selected["metrics"][selection_metric],
    )
    return selected["pipeline"], selected["metrics"], selected["model_type"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train heart disease classification models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/processed.cleveland.data",
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["lr", "rf", "both"],
        help=(
            "Model type: lr (Logistic Regression), rf (Random Forest), "
            "or both (train both and save the best by ROC-AUC)"
        ),
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/best_model.joblib",
        help="Output path to save the best model file",
    )

    args = parser.parse_args()

    # Verify file path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, args.data)
    save_path = os.path.join(project_dir, args.save_path)

    # Train model or compare all required models
    if args.model == "both":
        pipeline, metrics, selected_model = train_all_and_select_best(data_path)
        logger.info("Best model from comparison: %s", selected_model)
    else:
        pipeline, metrics = train_and_evaluate(data_path, model_type=args.model)

    # Save model artifact locally
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Saved best model pipeline locally to: {save_path}")
