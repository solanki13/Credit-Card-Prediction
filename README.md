# Credit-Card-Prediction
The goal is to build a machine learning model that can learn from historical credit data and classify users into “default” or “non-default” classes, helping financial institutions assess credit risk and prevent potential losses.


This project focuses on predicting the likelihood of a customer defaulting on their credit card payment for the upcoming month using machine learning techniques. We use the Taiwan Credit Card Default dataset from the UCI repository and implement full preprocessing, EDA, model training, and evaluation using a clean ML pipeline.

---

## 📂 Dataset Information

- **Source**: UCI Machine Learning Repository
- **Instances**: 30,000
- **Features**: 23
- **Target**: `default.payment.next.month` (1 = Default, 0 = No Default)

Each record contains demographic details, credit limit, repayment history, and bill/payment data over 6 months.

---

## 🎯 Problem Statement

Given customer attributes and credit history, can we predict whether they will default on their next payment?

This is a **binary classification** problem:  
- 0 → No default  
- 1 → Will default

---

## ⚙️ Technologies Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (for preprocessing, modeling, evaluation)
- Jupyter Notebook / Colab
- GridSearchCV for hyperparameter tuning

---

## 📊 Workflow

### 1. Data Cleaning
- Dropped unnecessary ID column
- Fixed categorical outliers (e.g., EDUCATION and MARRIAGE unknown categories)

### 2. Exploratory Data Analysis (EDA)
- Target class distribution
- Feature distributions (histograms, KDEs)
- Correlation heatmap
- Boxplots to inspect outliers
- Monthly payment and bill trend visualizations

### 3. Feature Engineering
- Standardized numeric features
- Handled class imbalance with stratified train-test split
- Removed highly correlated or redundant features (optional)

### 4. Model Building
- Created a full pipeline with:
  - ColumnTransformer
  - StandardScaler
  - RandomForestClassifier

### 5. Hyperparameter Tuning
- Used GridSearchCV to optimize `n_estimators`, `max_depth`, and `min_samples_split`

### 6. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

---

## ✅ Results

- **Best Model**: Random Forest Classifier
- **Accuracy Achieved**: ~82% on test data
- Confusion matrix and classification report confirm balanced performance
- ROC-AUC score: ~0.75 (good for imbalanced data)

---

---

## 🚀 Future Work

- Apply SMOTE or Class Weights to handle imbalance better
- Test with XGBoost and LightGBM for improved performance
- Deploy via Flask or Streamlit
- Add model explainability using SHAP or LIME




