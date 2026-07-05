"""
Preprocessing pipeline for the Heart Disease dataset.
Handles data cleaning, missing values, encoding, and scaling in a reproducible manner.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Standard column names for the Cleveland Heart Disease Dataset
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# Feature lists
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COLUMN = "target"


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw CSV data, handles missing values represented as '?', and binary-encodes the target.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    # Read the dataset (no headers in the raw UCI file)
    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)

    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Convert ca and thal to float since they might contain NaNs as strings
    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
    df["thal"] = pd.to_numeric(df["thal"], errors="coerce")

    # Map target: 0 -> 0 (no heart disease), 1,2,3,4 -> 1 (presence of heart disease)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x > 0 else 0)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Creates and returns the scikit-learn ColumnTransformer for preprocessing.
    """
    # Pipeline for numerical features
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline for categorical features
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def save_pipeline(pipeline: ColumnTransformer, output_path: str):
    """
    Serializes and saves the fitted preprocessor pipeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)


def load_pipeline(pipeline_path: str) -> ColumnTransformer:
    """
    Loads a serialized preprocessor pipeline from disk.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline not found at: {pipeline_path}")
    return joblib.load(pipeline_path)
