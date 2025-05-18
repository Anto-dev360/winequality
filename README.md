# 🍷 Wine Quality Prediction

> First project with scikit-learn: predict the quality of a wine based on different physicochemical measurements.

## 📌 Description

This project is a beginner-friendly machine learning model built with **scikit-learn** to predict the **quality of white wines**.
It uses the **Wine Quality dataset** from UCI, available via **Kaggle**.

The goal is to practice data preprocessing, model training, evaluation, and visualization.

## 📁 Dataset

The dataset is publicly available on Kaggle:
🔗 [Wine Quality Dataset – Kaggle](https://www.kaggle.com/datasets/brendan45774/wine-quality/)

## 🧪 Features used

Each record in the dataset represents a white wine sample, with the following features:

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

The target variable is:

- `quality` (integer score from 0 to 10)

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Anto-dev360/winequality.git
cd winequality
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API

To download the dataset, you must configure access to the Kaggle API:

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Click **Create New API Token**
3. A file named `kaggle.json` will download

Then, move it to the appropriate location:

```bash
mkdir ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
```


## 🚀 How to Run

You can run the project using the provided script or notebook.

### ▶️ Run the script

```bash
python main.py
```


## 🧠 Model

The project uses:

- **Random Forest Classifier** from `scikit-learn`
- Data normalization with `StandardScaler`
- Train/test split and evaluation (accuracy, confusion matrix)


## ✨ Author

**Anto-dev360** – _First ML project using scikit-learn_
📫 GitHub: [https://github.com/Anto-dev360](https://github.com/Anto-dev360)

## 📘 License

This project is open-source and available under the [MIT License](LICENSE).
