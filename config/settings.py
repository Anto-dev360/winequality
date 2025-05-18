"""
Author : Anthony Morin
Description : Application constants.
"""

from pathlib import Path

# Constants for dataset download
DATASET_NAME = "brendan45774/wine-quality"
EXPECTED_FILES = ["winequality-red.csv", "winequality-white.csv"]
DOWNLOAD_DIR = Path("data/raw")