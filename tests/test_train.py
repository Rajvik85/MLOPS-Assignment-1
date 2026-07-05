"""
Tests for model evaluation utilities and model-selection logic.
"""

from pathlib import Path

import numpy as np

from src import train


def test_create_plots_generates_confusion_matrix_and_roc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.05, 0.95, 0.2, 0.8])

    plot_paths = train.create_plots(y_test, y_pred, y_prob, "Unit_Test_Model")

    assert set(plot_paths) == {"confusion_matrix", "roc_curve"}
    for file_path in plot_paths.values():
        assert Path(file_path).exists()
        assert Path(file_path).stat().st_size > 0


def test_train_all_and_select_best_uses_roc_auc(monkeypatch):
    def fake_train_and_evaluate(
        raw_data_path, model_type="rf", mlflow_tracking_uri=None
    ):
        if model_type == "lr":
            return "lr_pipeline", {"roc_auc": 0.95, "accuracy": 0.86}
        return "rf_pipeline", {"roc_auc": 0.91, "accuracy": 0.88}

    monkeypatch.setattr(train, "train_and_evaluate", fake_train_and_evaluate)

    pipeline, metrics, selected_model = train.train_all_and_select_best("dummy.data")

    assert pipeline == "lr_pipeline"
    assert selected_model == "lr"
    assert metrics["roc_auc"] == 0.95
