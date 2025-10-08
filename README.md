# 💳 Loan Defaulter Prediction using XGBoost & Big Data Analytics

> **Tech Stack:** Python · PySpark · IBM SPSS Modeler · XGBoost · LightGBM · Sklearn · Seaborn

---

## 📘 Overview

This project develops a **machine-learning framework to predict potential loan defaulters** and strengthen financial risk management for lending institutions.
Using a large-scale dataset from **Alibaba Tianchi (1.2 million records)**, the study applies **XGBoost** as the primary model and benchmarks it against Logistic Regression, CART Decision Tree, and LightGBM.
The solution integrates **PySpark** for distributed data processing and **SPSS Modeler** for visual experimentation, delivering an end-to-end pipeline from data cleaning to model evaluation.

---

## 🧩 Project Objectives

* Predict loan default probability from borrower attributes.
* Identify **key risk factors** influencing default behaviour.
* Achieve **≥ 80 % accuracy** and a **high AUC score** to ensure business reliability.
* Demonstrate scalable data-mining practice using open-source and enterprise tools.

---

## ⚙️ Methodology

### 1️⃣ Data Source

* **Dataset:** Alibaba Tianchi Financial Risk Control Competition
* **Records:** ≈ 1.2 million; 47 features (15 anonymised)
* **Split:** 800 K train · 200 K test A · 200 K test B

### 2️⃣ Pre-processing & EDA

* Missing-value imputation (mean / mode).
* Outlier detection + normalisation.
* Feature-type conversion (int/float/object → numeric).
* Correlation heat-maps & statistical tests (χ² / t-test).

### 3️⃣ Feature Engineering & Model Building

* Encoding techniques: One-Hot & Index Encoding.
* Models: Logistic Regression | Decision Tree | LightGBM | XGBoost.
* Cross-validation: 5-fold CV for robust evaluation.
* Hyperparameter tuning via GridSearchCV (`max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`).

### 4️⃣ Evaluation Metrics

* **AUC (Area Under ROC)** as primary metric.
* Accuracy, Precision, Recall, F1-Score for model robustness.
* Comparison of baseline and tuned models for stability check.

---

## 📊 Results

| Model                | AUC (Initial) | AUC (Improved) | Accuracy |
| :------------------- | :-----------: | :------------: | :------: |
| Logistic Regression  |     0.7044    |        –       |   0.79   |
| Decision Tree (CART) |     0.7000    |        –       |   0.78   |
| XGBoost              |     0.7249    |   **0.7269**   | **0.81** |

**Key Findings:**

* 🧠 **XGBoost outperformed all other models**, delivering superior balance between precision and recall.
* 🏆 **Top predictors:** `issueDateDT`, `annualIncome`, `dti (debt-to-income)`, `employmentTitle`.
* 📈 Encoding and tuning raised accuracy from 0.48 → 0.81 and AUC to 0.7269.
* 🧱 Feature-importance analysis provides interpretable insights for credit risk managers.

---

## 💡 Insights & Business Implications

* Enables banks to flag high-risk borrowers early and reduce non-performing loans.
* Supports policy design for income verification and DTI ratio limits.
* Demonstrates how **AI-driven credit risk models** can enhance lending decisions and regulatory compliance.

---

## 🚀 Future Enhancements

* Implement SMOTE or generative augmentation for class imbalance.
* Incorporate SHAP values for model explainability.
* Deploy via Flask API on AWS with streaming prediction capability.
* Integrate economic indicators or transaction behaviour for richer feature space.



Would you like me to make a **simplified short version** (about 150 words) for your GitHub repository *description box* or pinned-repo summary as well?
