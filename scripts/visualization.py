"""
visualization.py

This module contains the Streamlit-based user interface components for the
Wine Quality Explorer application.

It includes functions that render various sections of the interactive app:
- About the project
- Dataset viewer and download
- Exploratory Data Analysis (EDA)
- Skewness and kurtosis statistics
- Bivariate analysis
- Model training and performance evaluation
- Decision tree visualization and export

The goal is to provide an educational, interactive way to explore machine
learning techniques applied to a real-world dataset.

Author: Anthony Morin
Created: 2025-05-18
Project: Wine Quality Prediction - Streamlit UI
License: MIT
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from config.settings import RESULT_DIR
from scripts.analysis_utils import (compute_skew_kurtosis, prepare_data,
                                    train_models)


def show_about():
    """
    Display the introductory section of the Streamlit app.

    This section explains the purpose of the application, provides learning context for users,
    and introduces the technologies and libraries used with external resource links.
    """
    st.title("🍷 Wine Quality Explorer")

    st.markdown(
        """
    Welcome to the **Wine Quality Explorer**!
    This interactive application is designed for educational purposes to help you understand
    how machine learning techniques can be used to predict the quality of wine based on measurable chemical properties.

    The dataset used comes from [Kaggle](https://www.kaggle.com/datasets/brendan45774/wine-quality),
    originally sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

    You'll be able to:
    - Explore and visualize physicochemical data from both **red** and **white** wines.
    - Analyze distributions, relationships, and statistical properties of the features.
    - Train multiple machine learning models (Logistic Regression, SVM, Naive Bayes, Decision Tree...).
    - Understand model performance and how features impact predictions.

    ---
    ### 🧰 Key Libraries & Tools

    - [Streamlit](https://streamlit.io) – for building the interactive web app
    - [Pandas](https://pandas.pydata.org) – for data handling and manipulation
    - [Seaborn](https://seaborn.pydata.org) / [Matplotlib](https://matplotlib.org) – for data visualization
    - [Scikit-learn](https://scikit-learn.org) – for machine learning models and data preprocessing
    - [Graphviz](https://graphviz.org) – for visualizing the decision tree

    ---
    ### 📘 How to use this app

    Use the navigation menu on the left to browse different sections of the app:

    - **Dataset Viewer**: Preview and download the dataset.
    - **EDA**: Visual exploration of each feature.
    - **Skewness & Kurtosis**: Analyze distribution shapes.
    - **Bivariate Analysis**: Explore feature vs quality relationships.
    - **Model Results**: Train and evaluate ML models.
    - **Decision Tree**: View the tree structure used for classification.

    > This project aims to be an accessible and practical introduction to machine learning on real-world data.
    """
    )


def show_dataset(df):
    """
    Display the merged wine dataset, its features, and provide a download option.

    Args:
        df (pd.DataFrame): The combined dataset containing both red and white wine samples.
    """
    st.header("📄 Wine Dataset Viewer")

    st.markdown(
        """
    The dataset consists of two types of wines:
    - **Red wines** and
    - **White wines**
    merged into a single dataset with an added column called `color` to distinguish them.

    Each row represents a unique wine sample with several physicochemical properties.

    ---
    ### 🔬 Features

    The following columns describe the characteristics of each wine:

    - `fixed acidity`
    - `volatile acidity`
    - `citric acid`
    - `residual sugar`
    - `chlorides`
    - `free sulfur dioxide`
    - `total sulfur dioxide`
    - `density`
    - `pH`
    - `sulphates`
    - `alcohol`
    - `color` (either `red` or `white`)

    ---
    ### 🎯 Target Label

    - `quality`: A score between **0** and **10**, indicating the quality of the wine as rated by testers.

    ---
    """
    )

    with st.expander("🔍 View full dataset"):
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download dataset as CSV",
        data=csv,
        file_name="wine_quality.csv",
        mime="text/csv",
    )


def show_eda(df):
    """
    Display Exploratory Data Analysis (EDA) visualizations for all numeric features.

    For each numeric column, plots a KDE distribution and a boxplot to analyze feature distribution and outliers.

    Args:
        df (pd.DataFrame): The combined dataset containing numeric features for EDA.
    """
    st.header("📊 Exploratory Data Analysis")

    st.markdown(
        """
    Exploratory Data Analysis (EDA) helps us understand the distribution and variability of each feature before applying any model.

    - **KDE plots** (Kernel Density Estimation) give us a smoothed curve representing the distribution of values.
    - **Boxplots** summarize key statistics (median, quartiles) and highlight potential **outliers**.

    By visually analyzing each feature, we can detect skewness, multimodality, or abnormal values that may affect model performance.

    ---
    """
    )

    df_features = df.select_dtypes(include="number")
    for col in df_features.columns:
        st.subheader(f"Feature: {col}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.kdeplot(df[col], ax=ax[0], fill=True, color="orange")
        ax[0].set_title("Distribution")
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title("Boxplot")
        st.pyplot(fig)


def show_skew_kurtosis(df):
    """
    Compute and display skewness and kurtosis statistics of the dataset's numeric features.

    Args:
        df (pd.DataFrame): The combined dataset from which skewness and kurtosis will be derived.
    """
    st.header("📈 Skewness and Kurtosis")

    st.markdown(
        """
    **Skewness** tells us about the asymmetry of a distribution.
    - A value near 0 indicates a symmetric distribution.
    - Positive skew means a longer tail on the right.
    - Negative skew means a longer tail on the left.

    **Kurtosis** indicates the 'tailedness' of the distribution:
    - Higher kurtosis (>3) means more outliers (heavy tails).
    - Lower kurtosis (<3) indicates light tails or fewer extreme values.

    These metrics help us understand how far the data deviates from normality, which is important when applying models that assume Gaussian distributions.
    """
    )

    st.markdown("---")

    stats_df = compute_skew_kurtosis(df)
    st.dataframe(stats_df)


def show_bivariate_analysis(df):
    """
    Perform and display bivariate visualizations to study the relationship between each
    numeric feature and wine quality.

    For each feature:
    - Displays a violin plot showing the distribution across quality levels.
    - Displays a strip plot (optimized) to show individual sample spread.

    Args:
        df (pd.DataFrame): The combined dataset containing wine features and quality labels.
    """
    st.header("📉 Bivariate Analysis")

    st.markdown(
        """
    This analysis helps us understand how each physicochemical feature correlates with wine quality.
    Violin plots show the density distribution across quality scores,
    while strip plots (lighter than swarm plots) display individual data points without overlap.

    ---
    """
    )

    df_features = df.select_dtypes(include="number").drop(columns="quality")

    for col in df_features.columns:
        with st.expander(f"🔎 {col} vs Quality", expanded=False):
            st.subheader(f"Feature: {col} by Quality")
            fig, ax = plt.subplots(1, 2, figsize=(16, 5))

            # Violin plot
            sns.violinplot(data=df, x="quality", y=col, ax=ax[0])
            ax[0].set_title(f"Violin Plot: {col}")

            # Strip plot (optimized alternative to swarm)
            sampled_df = df if len(df) < 1000 else df.sample(1000, random_state=42)
            sns.stripplot(
                data=sampled_df,
                x="quality",
                y=col,
                ax=ax[1],
                alpha=0.5,
                size=3,
                jitter=0.2,
            )
            ax[1].set_title(f"Strip Plot: {col} (sampled)")

            st.pyplot(fig)


def show_model_results(df):
    """
    Train and evaluate multiple machine learning models using the dataset.

    Displays the accuracy of:
        - Logistic Regression
        - Support Vector Machine (SVM)
        - Stochastic Gradient Descent (SGD)
        - Gaussian Naive Bayes
        - Decision Tree

    Args:
        df (pd.DataFrame): The combined dataset used for training/testing.
    """
    st.header("🤖 Model Results")

    st.markdown(
        """
    In this section, we train and evaluate several common classification algorithms on the wine dataset.
    Each model has different strengths and assumptions, and comparing them helps us understand which one best suits our data.

    We report their **accuracy** on a test set that was not used during training.

    ---
    **Models evaluated:**
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Stochastic Gradient Descent (SGD)
    - Gaussian Naive Bayes
    - Decision Tree

    ---
    """
    )

    X_train, X_test, y_train, y_test = prepare_data(df)
    st.info("🔁 Training models... This may take a few seconds.")
    train_models(X_train, X_test, y_train, y_test, X_train.columns.tolist())


def show_decision_tree(df):
    """
    Display and offer download of the pre-trained decision tree visualization (PDF).

    Assumes that a decision tree has already been exported to `wine.pdf`.

    Args:
        df (pd.DataFrame): The dataset (unused directly, but required for function signature consistency).
    """
    st.header("🌳 Decision Tree Visualization")
    st.markdown(
        """A Decision Tree model was trained and exported as a PDF.
    You can open it using the button below to visualize how decisions are made."""
    )

    pdf_path = RESULT_DIR / "wine.pdf"

    if pdf_path.exists():
        with open(pdf_path, "rb") as file:
            st.download_button(
                "Download Decision Tree PDF",
                data=file,
                file_name="wine.pdf",
                mime="application/pdf",
            )
    else:
        st.warning("📁 PDF not found. Make sure the decision tree has been generated.")
