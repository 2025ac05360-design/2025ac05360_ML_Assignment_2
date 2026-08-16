# Sensorless Drive Diagnosis — Fault Classification

## Problem Statement
Electric motor drives are expensive to instrument with dedicated fault sensors, so this project explores whether the drive's condition — healthy or one of several fault states — can be diagnosed purely from statistical features extracted from its phase current signals. This is a multi-class classification problem: given 48 numeric features per sample, predict which of 11 operating conditions the drive is in.

## Dataset Description
- **Source:** UCI Machine Learning Repository — [Dataset for Sensorless Drive Diagnosis](https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis) (ID 325)
- **Instances:** 58,509
- **Features:** 48 (real-valued; mean, standard deviation, skewness and kurtosis of Empirical Mode Decomposition components extracted from two phase-current signals)
- **Target:** 11 classes (1 = healthy drive, 2–11 = distinct fault/operating conditions), perfectly balanced at 5,319 samples per class
- **Missing values:** none

## GitHub Repository
`<paste your repo link here after you push>`

## Models Used

Evaluated with a stratified 75/25 train-test split (`random_state=42`), features standardized with `StandardScaler`. Precision/Recall/F1/AUC use macro averaging (fair across all 11 classes since AUC is one-vs-rest).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9177 | 0.9952 | 0.9176 | 0.9177 | 0.9175 | 0.9095 |
| Decision Tree | 0.9840 | 0.9912 | 0.9840 | 0.9840 | 0.9840 | 0.9824 |
| kNN | 0.8255 | 0.9727 | 0.8285 | 0.8255 | 0.8258 | 0.8082 |
| Naive Bayes | 0.7453 | 0.9843 | 0.7887 | 0.7453 | 0.7066 | 0.7286 |
| Random Forest (Ensemble) | 0.9989 | 1.0000 | 0.9989 | 0.9989 | 0.9989 | 0.9988 |

*(6th model — pending confirmation from course coordinator on whether one is required; see `planner.md`. Add its row here once implemented.)*

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong result (92% accuracy, 0.995 AUC) despite being a linear model — the 48 engineered statistical features are apparently close to linearly separable for most fault classes. The gap between AUC (0.995) and accuracy (0.918) suggests the model ranks classes well but its hard decision boundary misclassifies some borderline cases between visually similar fault states. |
| Decision Tree | Performs very well (98.4% accuracy) with a single unpruned tree — the fault classes likely correspond to fairly clean, axis-aligned thresholds in a few of the 48 features, which trees exploit directly without needing feature scaling. |
| kNN | Clearly the weakest of the five (82.6% accuracy) even after standardization. With 48 dimensions and 11 tightly-packed classes, distance-based neighbor voting likely suffers from the curse of dimensionality; increasing k or using a different distance metric would be worth trying if this model is kept. |
| Naive Bayes | Weakest overall (74.5% accuracy, and the lowest F1 at 0.71 despite a respectable 0.984 AUC). Gaussian Naive Bayes assumes features are independent given the class, which is a poor fit here since many of the 48 features are derived from the same underlying current signal and are highly correlated. |
| Random Forest (Ensemble) | Near-perfect (99.9% accuracy, AUC 1.0000). Bagging many decision trees smooths out the single tree's few remaining errors and handles the correlated, non-linear feature interactions better than any other model tested. |
| **Overall Winner** | **Random Forest (Ensemble)** — best score on every single metric, with virtually no misclassifications on the held-out test set. |

## Streamlit App

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload `test_data.csv` (or any CSV with the same 48 `feature_*` columns, optionally plus a `target` column), pick a model from the sidebar dropdown, and the app shows accuracy/AUC/precision/recall/F1/MCC, the confusion matrix, and a downloadable predictions CSV.

## Live App
`<paste your Streamlit Community Cloud link here once deployed>`
