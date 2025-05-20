"""
settings.py

Application constants.

Author: Anthony Morin
Created: 2025-05-18
Project: Wine Quality Prediction - Streamlit UI
License: MIT
"""

from pathlib import Path

# Constants for dataset download
DATASET_NAME = "brendan45774/wine-quality"
EXPECTED_FILES = ["winequality-red.csv", "winequality-white.csv"]
DOWNLOAD_DIR = Path("data/raw")
RESULT_DIR = Path("data/results")
