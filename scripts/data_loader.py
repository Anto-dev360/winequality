"""
data_loader.py

Functions to load and manages wine dataset

Author: Anthony Morin
Created: 2025-05-18
Project: Wine Quality Prediction - Streamlit UI
License: MIT
"""

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi

from config.settings import DATASET_NAME, DOWNLOAD_DIR, EXPECTED_FILES


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
        raise FileNotFoundError(
            "The file '.kaggle/kaggle.json' was not found in the project directory."
        )

    user_kaggle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(project_kaggle_path, user_kaggle_path)
    st.toast(f"kaggle.json copied to: {user_kaggle_path}", icon="✅")


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
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    api = authenticate_kaggle()
    api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_DIR, unzip=True)


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
        raise FileNotFoundError(
            "Data files are missing. Please download the dataset first."
        )

    df_red = pd.read_csv(red_path, sep=";")
    df_white = pd.read_csv(white_path, sep=";")
    return df_red, df_white


def merge_wine_dataframes(df_red, df_white):
    """
    Merge red and white wine DataFrames into a single DataFrame.

    Adds a 'color' column to distinguish between red and white wines.

    Args:
        df_red (pd.DataFrame): DataFrame containing red wine data.
        df_white (pd.DataFrame): DataFrame containing white wine data.

    Returns:
        pd.DataFrame: Merged DataFrame with an added 'color' column.
    """
    df_red["color"] = "red"
    df_white["color"] = "white"
    return pd.concat([df_red, df_white], ignore_index=True)


@st.cache_data
def fetch_data():
    """
    Cacheable version that downloads and loads the dataset without Streamlit elements.

    Returns:
    pd.DataFrame: The merged dataset containing both red and white wines.
                    Returns an empty DataFrame if an error occurs.
    """
    download_wine_dataset()
    df_red, df_white = load_wine_dataframes()
    df = merge_wine_dataframes(df_red, df_white)
    return df
