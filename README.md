# 🫀 Heart Disease Prediction - Machine Learning Analysis

## 📌 Project Overview
This repository contains an end-to-end applied machine learning project focused on healthcare analytics. The goal is to build a predictive classification model that can accurately identify whether a patient has heart disease based on their clinical test results and physical attributes. 

Working with a real-world structured dataset (`HeartDiseaseTrain-Test.csv`), this project demonstrates the complete data science lifecycle: from raw data exploration and visualization to preprocessing, model training, and performance evaluation.

## 📊 Dataset Information
The dataset consists of patient diagnostic records, featuring a mix of continuous numerical data and text-based categorical data. 

**Target Variable:**
* `target`: `1` (Heart Disease Present) / `0` (No Heart Disease)

**Key Clinical Features Include:**
* **Demographics:** `age`, `sex`
* **Symptoms:** `chest_pain_type` (e.g., Typical angina, Asymptomatic), `exercise_induced_angina`
* **Vitals & Labs:** `resting_blood_pressure`, `cholestoral`, `fasting_blood_sugar`
* **Diagnostic Results:** `rest_ecg`, `Max_heart_rate`, `oldpeak`, `slope`, `vessels_colored_by_flourosopy`, `thalassemia`

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (Random Forest Classifier, StandardScaler, Metrics)
* **Environment:** Jupyter Notebook

## ⚙️ Methodology & Pipeline

1. **Exploratory Data Analysis (EDA):**
   * Handled missing values and analyzed statistical distributions.
   * Visualized class balance, age distributions via KDE plots, and correlation heatmaps.
   * Mapped cardiovascular efficiency by comparing Maximum Heart Rate against Age.

2. **Data Preprocessing:**
   * **One-Hot Encoding:** Converted descriptive categorical text features (e.g., "Male", "Typical angina") into machine-readable binary numeric columns (`pd.get_dummies`).
   * **Feature Scaling:** Applied `StandardScaler` to continuous variables to ensure distance-based calculations were not skewed by different units of measurement (e.g., cholesterol in mg/dl vs. oldpeak).
   * **Data Splitting:** Stratified 80/20 Train-Test split to maintain class ratios.

3. **Modeling:**
   * Trained a **Random Forest Classifier** (`n_estimators=100`).
   * Chosen for its robustness against overfitting and its ability to handle complex, non-linear relationships in medical data.

4. **Evaluation:**
   * Analyzed Model Accuracy, Precision, Recall, and F1-Scores.
   * Generated a visual **Confusion Matrix** to evaluate False Positives vs. False Negatives (critical in medical screening).
   * Extracted **Feature Importances** to determine the primary physiological drivers behind the model's predictions.

