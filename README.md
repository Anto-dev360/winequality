# 🍷 Wine Quality Prediction

> An interactive educational Streamlit application that visualizes and classifies red and white wine quality using machine learning and exploratory data analysis (EDA).

## 📌 Description

This project uses the **Wine Quality dataset** (from Kaggle) to build machine learning models that predict the **quality** of a wine based on its physicochemical characteristics. It is built using **Python**, **scikit-learn**, **pandas**, **matplotlib**, and **seaborn**.

Key goals:

- Practice data loading, cleaning, merging.
- Conduct exploratory data analysis (EDA) with visualizations.
- Compute skewness and kurtosis of features.
- Perform bivariate analysis.
- Train and evaluate multiple classifiers.

Now supports both **red and white wines**, and includes modular code for better clarity and reusability.

## 📁 Dataset

The dataset is publicly available on Kaggle:
🔗 [Wine Quality Dataset – Kaggle](https://www.kaggle.com/datasets/brendan45774/wine-quality/)

## 🧪 Features

Each sample represents a wine with the following features:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- `color` (added column: red or white)

Target variable:

- `quality` (integer score between 0 and 10)

# UI tabs:
- **About**: Description and goals of the application.
- **Dataset Viewer**: Interactive view and download of merged dataset.
- **EDA**: Distribution and boxplots of wine features.
- **Skewness & Kurtosis**: Shape characteristics of distributions.
- **Bivariate Analysis**: Relationship between features and wine quality.
- **Model Results**: Accuracy scores for multiple classifiers.
- **Decision Tree**: Visual decision tree to explain predictions.

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Anto-dev360/winequality.git
cd winequality
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API (to download the dataset)

1. Visit [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Click **Create New API Token**
3. Move `kaggle.json` to the appropriate location:

```bash
mkdir ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Optional: Install Graphviz (for Decision Tree export)

To visualize and export the decision tree as a PDF (`wine.pdf`), make sure the Graphviz system executable is installed.

#### Windows:
1. Download the installer: [https://graphviz.org/download/](https://graphviz.org/download/)
2. Run the installer and choose the default location (e.g. `C:\Program Files\Graphviz`)
3. Add the Graphviz binary folder to your system PATH:
    - Add: `C:\Program Files\Graphviz\bin`

4. Restart your terminal or IDE.

#### macOS (Homebrew):
```bash
brew install graphviz
```

#### Linux (Debian/Ubuntu):
```bash
sudo apt-get install graphviz
```

You can test the installation with:
```bash
dot -V
```

## 🚀 How to Run

```bash
streamlit run app.py
```

This will:
- Download and load the dataset
- Merge and annotate red/white wines
- Generate distribution and box plots
- Compute skewness and kurtosis
- Visualize feature relationships with wine quality
- Train and evaluate 5 classifiers
- Predict wine quality
- Display result with streamlit

## 🧠 Models Used

The following classification algorithms from `scikit-learn` are trained and evaluated:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Stochastic Gradient Descent (SGD)**
- **Naive Bayes (GaussianNB)**
- **Decision Tree Classifier**

The decision tree is also visualized and saved as `wine.pdf`.

## 📊 Visualizations

The project includes:
- Bar plots of wine quality distribution
- KDE and box plots of all numeric features
- Violin and swarm plots for feature vs quality
- Mean sulfur dioxide across quality levels
- Decision tree visualization

## 📂 Project Structure

```
.
├── main.py                  # Entry point of the project
├── scripts/
|   ├── analysis_utils.py    # Modular functions for EDA and modeling
|   ├── visualization.py     # Modular functions for result display with streamlit
|   └── data_loader.py       # Modular functions for data loading
├── config
|   └── settings.py          # Application constants
├── data/raw                 # Directory containing the downloaded data
├── wine.pdf                 # Exported decision tree diagram
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## ✨ Author

**Anto-dev360** – _First modular ML project using scikit-learn_
📫 GitHub: [@Anto-dev360](https://github.com/Anto-dev360)

## 📘 License

This project is open-source and available under the [MIT License](LICENSE).