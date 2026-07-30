# Credit Card Fraud Detection using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green?logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)

## 📌 Overview

This project implements a **Machine Learning model for detecting fraudulent credit card transactions** using **Logistic Regression**.

The objective is to classify transactions into two categories:

- **0 → Normal transaction**
- **1 → Fraudulent transaction**

The project follows a complete Machine Learning workflow:

- Dataset loading from Kaggle
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Handling class imbalance
- Model training
- Model validation
- Final test evaluation
- Feature importance analysis


---

# 📂 Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset characteristics:

- **284,807 transactions**
- **30 input features**
- **1 target variable (`Class`)**

Features:

- `Time` → Seconds elapsed between transactions
- `Amount` → Transaction amount
- `V1 - V28` → PCA-transformed numerical features
- `Class` → Target variable

Target:

| Value | Meaning |
|---|---|
| 0 | Normal transaction |
| 1 | Fraud transaction |


Because fraudulent transactions represent a very small percentage of the dataset, this is considered a **highly imbalanced classification problem**.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- KaggleHub


---

# 🔎 Exploratory Data Analysis (EDA)

Before training the model, the dataset was analyzed to understand its structure and characteristics.

## Dataset Inspection

Performed:

- Dataset information
- Data types analysis
- Statistical description
- Missing value detection
- Duplicate detection


## Class Distribution Analysis

The fraud dataset is highly imbalanced, therefore the distribution between:

- Normal transactions
- Fraudulent transactions

was visualized.

This step is important because accuracy alone can be misleading for fraud detection problems.


## Transaction Analysis

Explored:

- Transaction amount distribution
- Transaction time distribution
- Difference between fraud and normal transaction amounts


## Correlation Analysis

Performed:

- Feature correlation heatmap
- Identification of features most correlated with fraud


---

# Data Preprocessing

## Duplicate Removal

Duplicate transactions were removed before training.

```python
df = df.drop_duplicates()
```


## Dataset Splitting

The dataset was divided into:

| Dataset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

A **stratified split** was used to maintain the same fraud ratio across all datasets.


## Feature Scaling

The following features were standardized:

- `Time`
- `Amount`

Using:

```python
StandardScaler()
```

Feature scaling improves Logistic Regression performance because it is sensitive to feature magnitude.


---

# Machine Learning Model

## Logistic Regression

The selected model is:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
```

## Why Logistic Regression?

Logistic Regression was chosen because it:

- Is suitable for binary classification
- Produces probability predictions
- Handles high-dimensional data efficiently
- Provides interpretable coefficients


---

# ⚖️ Handling Class Imbalance

Fraud transactions are extremely rare compared to normal transactions.

To prevent the model from being biased toward the majority class:

```python
class_weight="balanced"
```

was applied.

This automatically increases the importance of fraud samples during training.


---

# 📊 Model Evaluation

Accuracy is not a reliable metric for this dataset because a model predicting all transactions as normal could still achieve very high accuracy.

Therefore, the model was evaluated using:

## Precision

Measures:

> Among all transactions predicted as fraud, how many were actually fraud?

High precision reduces false fraud alerts.


## Recall

Measures:

> Among all real fraud transactions, how many were detected?

High recall reduces missed fraud cases.


## F1-score

The harmonic mean between precision and recall.

Useful when balancing false positives and false negatives.


## ROC-AUC Score

Measures the ability of the model to distinguish between:

- Fraud transactions
- Normal transactions

A value closer to 1 indicates better performance.


## Precision-Recall Curve

Because the dataset is highly imbalanced, the Precision-Recall curve provides a more informative evaluation than ROC alone.


---

# 🔍 Feature Importance

Logistic Regression coefficients were analyzed to identify:

- Features contributing toward fraud prediction
- Features contributing toward normal prediction

This provides interpretability and helps understand model decisions.


**Emna Chebbi**

GitHub:  
https://github.com/Emna-chebbi
