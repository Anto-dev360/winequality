"""
Author : Anthony Morin
Description : Loading data.
"""

import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from pathlib import Path

from config.settings import (
    DATASET_NAME,
    EXPECTED_FILES,
    DOWNLOAD_DIR,
)

def setup_kaggle_credentials():
    """
    Copies the kaggle.json file from the local project directory (.kaggle/kaggle.json)
    to the user's home directory (~/.kaggle/kaggle.json).

    Raises:
        FileNotFoundError: If the local .kaggle/kaggle.json file is not found.
    """
    project_kaggle_path = Path(".kaggle/kaggle.json")
    user_kaggle_dir = Path.home() / ".kaggle"
    user_kaggle_path = user_kaggle_dir / "kaggle.json"

    if not project_kaggle_path.exists():
        raise FileNotFoundError("The file '.kaggle/kaggle.json' was not found in the project directory.")

    user_kaggle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(project_kaggle_path, user_kaggle_path)
    print(f"✅ kaggle.json copied to: {user_kaggle_path}")

def authenticate_kaggle():
    """
    Authenticate the Kaggle API using the kaggle.json token in the local .kaggle directory.

    Raises:
        FileNotFoundError: If the kaggle.json file is not found.

    Returns:
        KaggleApi: An authenticated instance of the Kaggle API.
    """

    setup_kaggle_credentials()

    api = KaggleApi()
    api.authenticate()
    return api


def is_dataset_present():
    """
    Check if the expected dataset files are already present in the download directory.

    Returns:
        bool: True if both CSV files exist, False otherwise.
    """
    return all((DOWNLOAD_DIR / file).exists() for file in EXPECTED_FILES)

def download_wine_dataset():
    """
    Download the wine quality dataset from Kaggle, unless it is already downloaded.

    Downloads and unzips the dataset to the data/raw directory.
    """
    if is_dataset_present():
        print("✅ Dataset already present. Skipping download.")
        return

    print("📥 Downloading the dataset from Kaggle...")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    api = authenticate_kaggle()
    api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_DIR, unzip=True)
    print("✅ Download complete and files extracted.")


def load_wine_dataframes():
    """
    Load the red and white wine quality datasets into pandas DataFrames.

    Returns:
        tuple: A tuple containing the red and white wine DataFrames.

    Raises:
        FileNotFoundError: If the CSV files are not found.
    """
    red_path = DOWNLOAD_DIR / "winequality-red.csv"
    white_path = DOWNLOAD_DIR / "winequality-white.csv"

    if not red_path.exists() or not white_path.exists():
        raise FileNotFoundError("Data files are missing. Please download the dataset first.")

    df_red = pd.read_csv(red_path, sep=";")
    df_white = pd.read_csv(white_path, sep=";")
    return df_red, df_white