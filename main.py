"""
Author : Anthony Morin
Description : Main file of the scikit-learn application.
"""

from scripts.data_loader import (
    download_wine_dataset,
    load_wine_dataframes,
)


def main():
    """
    Main entry point of the application.

    This function performs the following steps:
    1. Downloads the wine quality dataset from Kaggle if it is not already available locally.
    2. Loads the red and white wine datasets into separate pandas DataFrames.
    3. Prints the first few rows of each DataFrame for preview purposes.

    Dependencies:
        - Requires the Kaggle API to be configured and kaggle.json to be present in ~/.kaggle.
        - Requires access to the internet for initial download (unless data is cached locally).

    Raises:
        FileNotFoundError: If dataset files are missing and cannot be downloaded.
        KaggleApiException: If Kaggle API authentication fails.
    """
    download_wine_dataset()
    df_red, df_white = load_wine_dataframes()

    # Print preview
    print(df_red.head(3))
    print(df_white.head(3))

if __name__ == "__main__":
    main()