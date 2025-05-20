"""
analysis_utils.py

Wine Quality Analysis and Classification

This module provides functions for exploratory data analysis (EDA) and machine learning
on the Wine Quality dataset from Kaggle. It includes functionality for previewing,
merging, visualizing distributions, analyzing bivariate relationships, and training
various classification models to predict wine quality.

Author: Anthony Morin
Created: 2025-05-18
Project: Wine Quality Prediction - Streamlit UI
License: MIT
"""

import shutil

import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from config.settings import RESULT_DIR


def preview_data(df_red, df_white):
    """
    Display the first few rows of red and white wine datasets.

    Args:
        df_red (pd.DataFrame): Red wine dataset.
        df_white (pd.DataFrame): White wine dataset.
    """
    st.subheader("Red wine preview")
    st.dataframe(df_red.head(3))
    st.subheader("White wine preview")
    st.dataframe(df_white.head(3))


def plot_quality_distribution(df):
    """
    Plot the distribution of wine quality ratings.

    Args:
        df (pd.DataFrame): Combined wine dataset.
    """
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    df["quality"].value_counts(normalize=True).plot.bar(rot=0, color="#066b8b")
    plt.ylabel("Quality")
    plt.xlabel("% Distribution per Category")

    plt.subplot(1, 2, 2)
    sns.countplot(data=df, y="quality")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df):
    """
    Plot distribution and boxplot for each numerical feature in the dataset.

    Args:
        df (pd.DataFrame): Combined wine dataset.
    """
    df_features = df.select_dtypes(include="number")
    num_columns = df_features.columns.tolist()

    plt.figure(figsize=(18, 40))
    for i, col in enumerate(num_columns, 1):
        plt.subplot(8, 4, i)
        sns.kdeplot(df[col], color="#d1aa00", fill=True)
        plt.subplot(8, 4, i + 11)
        df[col].plot.box()
    plt.tight_layout()
    plt.show()


def compute_skew_kurtosis(df):
    """
    Compute skewness and kurtosis for numeric features in the dataset.

    Args:
        df (pd.DataFrame): Combined wine dataset.

    Returns:
        pd.DataFrame: DataFrame containing skewness and kurtosis.
    """
    df_features = df.select_dtypes(include="number")
    return pd.DataFrame(
        data=[df_features.skew(), df_features.kurtosis()],
        index=["skewness", "kurtosis"],
    )


def prepare_data(df):
    """
    Prepare features and labels and split them into training and testing sets.

    Args:
        df (pd.DataFrame): Combined wine dataset.

    Returns:
        tuple: Features and labels split into training and test sets.
    """
    X = df.select_dtypes(include="number").drop(columns="quality")
    y = df["quality"]
    return train_test_split(X, y, test_size=0.20)


def train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train multiple classification models and report their accuracy.
    Includes Logistic Regression, SVM, SGD, Naive Bayes, and Decision Tree.
    Also renders a decision tree as PDF if Graphviz is available.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        feature_names (list): Feature names for tree visualization.
    """
    st.subheader("📈 Model Accuracy Scores")

    # Standardize features for models sensitive to scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": svm.SVC(),
        "SGD": SGDClassifier(),
        "GaussianNB": GaussianNB(),
        "DecisionTree": tree.DecisionTreeClassifier(),
    }

    for name, model in models.items():
        try:
            if name in ["LogisticRegression", "SVM", "SGD"]:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
            else:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
            st.success(f"{name} accuracy: {score:.2f}")
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {e}")

        if name == "DecisionTree":
            try:
                dot_data = tree.export_graphviz(
                    model,
                    out_file=None,
                    feature_names=feature_names,
                    class_names=[str(c) for c in sorted(y_train.unique())],
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = graphviz.Source(dot_data)
                if shutil.which("dot"):
                    pdf_path = RESULT_DIR / "wine.pdf"

                    # Remove old file if needed
                    if pdf_path.exists():
                        try:
                            pdf_path.unlink()  # force deletion if previously locked
                        except Exception as e:
                            st.warning(f"⚠️ Could not overwrite existing PDF: {e}")
                    else:
                        pdf_path.parent.mkdir(parents=True, exist_ok=True)

                    graph.render(pdf_path.with_suffix(""), cleanup=True)
                    st.toast("✅ Decision tree rendered and saved as 'wine.pdf'")
                else:
                    st.warning("⚠️ Graphviz not found — skipping tree rendering.")
            except Exception as e:
                st.warning(f"⚠️ Could not export decision tree: {e}")
