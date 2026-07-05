#!/usr/bin/env python3
"""
Script to download the Heart Disease dataset from the UCI Machine Learning Repository.
Saves the raw file in the data/raw directory.
"""

import os
import urllib.request
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_data")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed.cleveland.data")


def download_dataset():
    """
    Downloads the processed Cleveland dataset from UCI repository.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        logger.info(f"Downloading dataset from: {DATA_URL}")

        # Download the file
        urllib.request.urlretrieve(DATA_URL, OUTPUT_FILE)

        # Check if file exists and has size
        if os.path.exists(OUTPUT_FILE):
            size_kb = os.path.getsize(OUTPUT_FILE) / 1024
            logger.info(
                f"Successfully downloaded dataset to: {OUTPUT_FILE} ({size_kb:.2f} KB)"
            )
        else:
            raise FileNotFoundError(
                "Download completed but destination file was not found."
            )

    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    download_dataset()
