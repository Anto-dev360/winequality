"""
app.py

This is the main entry point of the Streamlit application for the
Wine Quality Explorer project.

It defines the layout, page navigation, and application flow. Each page
(About, Dataset, EDA, etc.) is implemented in a separate function in the
visualization module. The data is cached to improve performance.

Sections available via the sidebar:
- About: Introduction to the project
- Dataset Viewer: View and download the merged dataset
- EDA: Feature distributions and outlier detection
- Skewness & Kurtosis: Shape statistics of numerical features
- Bivariate Analysis: Relationship between features and wine quality
- Model Results: Training and accuracy scores of ML classifiers
- Decision Tree: Downloadable PDF of the decision tree visualization

Author: Anthony Morin
Created: 2025-05-18
Project: Wine Quality Prediction - Streamlit UI
License: MIT
"""

import pandas as pd
import streamlit as st

from scripts.data_loader import (download_wine_dataset, fetch_data,
                                 is_dataset_present, load_wine_dataframes,
                                 merge_wine_dataframes)
from scripts.visualization import (show_about, show_bivariate_analysis,
                                   show_dataset, show_decision_tree, show_eda,
                                   show_model_results, show_skew_kurtosis)


def load_data():
    """
    Handles cached data loading and wraps error display for the Streamlit UI.

    This function performs the following:
    1. Downloads the dataset from Kaggle if not already present.
    2. Loads the red and white wine CSV files.
    3. Merges them into a single DataFrame with a 'color' column.
    4. Displays an error message in the Streamlit UI if loading fails.

    Returns:
        pd.DataFrame: The merged dataset containing both red and white wines.
                      Returns an empty DataFrame if an error occurs.
    """
    try:
        if not is_dataset_present():
            st.toast("📥 Downloading the dataset from Kaggle...")
        df = fetch_data()
        st.toast("✅ Dataset ready.")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()


def main():
    """
    Main function for launching the Streamlit web application.

    This function:
    - Sets the page configuration for Streamlit.
    - Displays a sidebar menu for page navigation.
    - Loads the dataset via `load_data()`.
    - Renders the selected section (About, EDA, Model Results, etc.)
      by calling the corresponding UI function from `visualization.py`.

    Navigation options:
        - About
        - Dataset Viewer
        - EDA (Exploratory Data Analysis)
        - Skewness & Kurtosis
        - Bivariate Analysis
        - Model Results
        - Decision Tree Visualization
    """
    st.set_page_config(page_title="Wine Quality Explorer", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "About",
            "Dataset Viewer",
            "EDA",
            "Skewness & Kurtosis",
            "Bivariate Analysis",
            "Model Results",
            "Decision Tree",
        ],
    )

    df = load_data()

    if page == "About":
        show_about()
    elif page == "Dataset Viewer":
        show_dataset(df)
    elif page == "EDA":
        show_eda(df)
    elif page == "Skewness & Kurtosis":
        show_skew_kurtosis(df)
    elif page == "Bivariate Analysis":
        show_bivariate_analysis(df)
    elif page == "Model Results":
        show_model_results(df)
    elif page == "Decision Tree":
        show_decision_tree(df)


if __name__ == "__main__":
    main()
