"""
Author : Anthony Morin
Description : Main file of the scikit-learn application.
"""

from scripts.data_loader import (
    download_wine_dataset,
    load_wine_dataframes,
    merge_wine_dataframes,
)
from scripts.model_utils import (
    preview_data,
    plot_quality_distribution,
    plot_feature_distributions,
    compute_skew_kurtosis,
    bivariate_analysis,
    prepare_data,
    train_models
)


def main():
    """
    Main entry point of the wine quality analysis and prediction application.

    This function performs a full machine learning workflow:
    1. Downloads the red and white wine datasets from Kaggle if not already available locally.
    2. Loads the datasets into pandas DataFrames and merges them with a 'color' column.
    3. Displays an overview of the quality distribution using bar plots and count plots.
    4. Filters out non-numeric features and analyzes distributions and outliers using KDE and box plots.
    5. Computes skewness and kurtosis for all numeric features.
    6. Performs bivariate analysis between selected features and wine quality using violin, swarm, and bar plots.
    7. Prepares the data for classification (features and labels) and splits into training and test sets.
    8. Trains and evaluates several classification algorithms:
        - Logistic Regression
        - Support Vector Machines
        - Stochastic Gradient Descent
        - Gaussian Naive Bayes
        - Decision Tree (with export to a Graphviz visual)

    Dependencies:
        - Requires the Kaggle API key (`kaggle.json`) to be available in the user's ~/.kaggle directory.
        - Requires packages: pandas, matplotlib, seaborn, scikit-learn, graphviz, kaggle.

    Raises:
        FileNotFoundError: If dataset files or API credentials are missing.
        KaggleApiException: If Kaggle API authentication fails.
        Any scikit-learn or pandas error during model training or data preprocessing.
    """
    download_wine_dataset()
    df_red, df_white = load_wine_dataframes()

    preview_data(df_red, df_white)

    df = merge_wine_dataframes(df_red, df_white)

    plot_quality_distribution(df)

    plot_feature_distributions(df)

    skew_kurt = compute_skew_kurtosis(df)
    print("\nSkewness and Kurtosis:\n", skew_kurt)

    bivariate_analysis(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    train_models(X_train, X_test, y_train, y_test, X_train.columns.to_list())


if __name__ == "__main__":
    main()