"""
Unit tests for the preprocessing functions and Pipeline in src/preprocess.py.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from src.preprocess import (
    load_and_clean_data,
    build_preprocessor,
    COLUMN_NAMES,
    TARGET_COLUMN,
)


@pytest.fixture
def mock_raw_data_file():
    """
    Fixture that creates a temporary raw data CSV representing the Cleveland format.
    Includes normal rows and rows with missing value '?' indicators.
    """
    # Create temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".data")

    # Define mock dataset matching the 14 columns
    # cols: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
    mock_data = [
        "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0",  # Normal row (target 0)
        "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2",  # Normal row (target > 0, should become 1)
        "67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1",  # Normal row (target > 0)
        "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0",  # Normal row
        "41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0",  # Missing 'ca' as '?'
        "56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,1",  # Missing 'thal' as '?'
    ]

    with open(temp_path, "w") as f:
        f.write("\n".join(mock_data))

    yield temp_path

    # Cleanup after test
    os.close(temp_fd)
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_load_and_clean_data(mock_raw_data_file):
    """
    Tests that load_and_clean_data replaces '?' with NaN and correctly maps target variables.
    """
    df = load_and_clean_data(mock_raw_data_file)

    assert len(df) == 6
    assert list(df.columns) == COLUMN_NAMES

    # Test target mapping (0 remains 0, positive integers map to 1)
    assert df.loc[0, TARGET_COLUMN] == 0
    assert df.loc[1, TARGET_COLUMN] == 1
    assert df.loc[2, TARGET_COLUMN] == 1

    # Test missing value conversion ('?' -> NaN)
    assert pd.isna(df.loc[4, "ca"])
    assert pd.isna(df.loc[5, "thal"])

    # Test correct data types
    assert df["ca"].dtype == np.float64 or df["ca"].dtype == np.float32
    assert df["thal"].dtype == np.float64 or df["thal"].dtype == np.float32


def test_preprocessor_pipeline(mock_raw_data_file):
    """
    Tests that the built preprocessor successfully fits and transforms a dataframe.
    """
    df = load_and_clean_data(mock_raw_data_file)

    features = [c for c in COLUMN_NAMES if c != TARGET_COLUMN]
    X = df[features]

    preprocessor = build_preprocessor()

    # Fit and transform
    X_trans = preprocessor.fit_transform(X)

    # Verify shape and contents (no NaNs should remain in the output)
    assert X_trans.shape[0] == 6
    assert not np.isnan(X_trans).any()
