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
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.analysis_utils import (
    compute_skew_kurtosis,
    prepare_data,
    train_models
)

def show_about():
    """
    Display the introductory section of the Streamlit app.

    Provides a description of the application's educational purpose and the key Python libraries used.
    This section is intended to give users context before exploring the dataset and models.
    """
    st.title("🍷 Wine Quality Explorer")
    st.markdown("""
    This application is designed for educational purposes to demonstrate how machine learning models can be applied
    to classify the quality of wine using physicochemical features.
    It uses the Wine Quality dataset (red and white) from Kaggle.

    **Key Libraries:**
    - Streamlit
    - Pandas, Seaborn, Matplotlib
    - Scikit-learn
    - Graphviz
    """)

def show_dataset(df):
    """
    Display the merged wine dataset in an expandable data table with a download option.

    Args:
        df (pd.DataFrame): The combined dataset containing both red and white wine samples.
    """
    st.header("📄 Merged Dataset Viewer")
    with st.expander("View dataset"):
        st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download dataset as CSV", data=csv, file_name="wine_quality.csv", mime="text/csv")

def show_eda(df):
    """
    Display Exploratory Data Analysis (EDA) visualizations for all numeric features.

    For each numeric column, plots a KDE distribution and a boxplot to analyze feature distribution and outliers.

    Args:
        df (pd.DataFrame): The combined dataset containing numeric features for EDA.
    """
    st.header("📊 Exploratory Data Analysis")
    df_features = df.select_dtypes(include='number')
    for col in df_features.columns:
        st.subheader(f"Feature: {col}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.kdeplot(df[col], ax=ax[0], fill=True, color='orange')
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
    stats_df = compute_skew_kurtosis(df)
    st.dataframe(stats_df)

def show_bivariate_analysis(df):
    """
    Perform and display bivariate visualizations to study the relationship between features and wine quality.

    Includes:
        - Violin plot: sulphates vs. quality
        - Swarm plot: total sulfur dioxide vs. quality
        - Bar plot: average total sulfur dioxide per quality level

    Args:
        df (pd.DataFrame): The combined dataset containing wine features and quality labels.
    """
    st.header("📉 Bivariate Analysis")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.violinplot(data=df, x='quality', y='sulphates', ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.swarmplot(data=df, x='quality', y='total sulfur dioxide', ax=ax2)
    st.pyplot(fig2)

    quality_cat = df.quality.unique()
    quality_cat.sort()
    qual_TSD = [[q, df['total sulfur dioxide'][df['quality'] == q].mean()] for q in quality_cat]
    df_qual_TSD = pd.DataFrame(qual_TSD, columns=['Quality', 'Mean TSD'])

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Quality', y='Mean TSD', data=df_qual_TSD, ax=ax3)
    st.pyplot(fig3)

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
    X_train, X_test, y_train, y_test = prepare_data(df)
    st.info("Training models... this may take a moment.")
    train_models(X_train, X_test, y_train, y_test, X_train.columns.tolist())

def show_decision_tree(df):
    """
    Display and offer download of the pre-trained decision tree visualization (PDF).

    Assumes that a decision tree has already been exported to `wine.pdf`.

    Args:
        df (pd.DataFrame): The dataset (unused directly, but required for function signature consistency).
    """
    st.header("🌳 Decision Tree Visualization")
    st.markdown("""A Decision Tree model was trained and exported as a PDF.
    You can open it using the button below to visualize how decisions are made.""")
    with open("wine.pdf", "rb") as file:
        st.download_button("Download Decision Tree PDF", data=file, file_name="wine.pdf", mime="application/pdf")
