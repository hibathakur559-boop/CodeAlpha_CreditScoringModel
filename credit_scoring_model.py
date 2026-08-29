"""
==========================================================
TASK 1: CREDIT SCORING MODEL - CodeAlpha ML Internship
==========================================================
Objective:
    Predict whether a person is CREDITWORTHY (Good) or 
    NOT CREDITWORTHY (Bad) based on their financial history.

Approach:
    - Feature engineering from financial data
    - Train 3 classification models: Logistic Regression,
      Decision Tree, Random Forest
    - Evaluate using Precision, Recall, F1-Score, ROC-AUC

Author: [Your Name Here]
==========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

# Make results reproducible every time we run the script
np.random.seed(42)


# ----------------------------------------------------------
# STEP 1: CREATE / LOAD THE DATASET
# ----------------------------------------------------------
# NOTE: In a real project you would download a dataset like the
# "German Credit Data" or "UCI Credit Card Default" dataset.
# Here we GENERATE a realistic synthetic dataset so the project
# runs instantly with no external downloads (great for beginners
# and for showing you understand every column).
#
# If you want to use a REAL dataset instead, just replace this
# section with:
#     df = pd.read_csv("your_dataset.csv")

def create_dataset(n_samples=2000):
    income = np.random.normal(50000, 20000, n_samples).clip(10000, 200000)
    age = np.random.randint(18, 70, n_samples)
    debts = np.random.normal(15000, 10000, n_samples).clip(0, 100000)
    loan_amount = np.random.normal(20000, 15000, n_samples).clip(1000, 150000)
    payment_history_score = np.random.randint(0, 100, n_samples)  # 0=bad,100=great
    num_credit_lines = np.random.randint(0, 15, n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    late_payments = np.random.poisson(1.5, n_samples)

    # Debt-to-income ratio: a very common real-world credit feature
    debt_to_income = debts / income

    # Create a "creditworthiness score" using a weighted formula,
    # then add some randomness (noise) to make it realistic.
    score = (
        0.35 * (payment_history_score / 100) +
        0.25 * (1 - debt_to_income.clip(0, 1)) +
        0.15 * (income / 200000) +
        0.10 * (employment_years / 40) -
        0.15 * (late_payments / 10) +
        np.random.normal(0, 0.05, n_samples)
    )

    # Convert score into a binary label: 1 = Good (creditworthy), 0 = Bad
    creditworthy = (score > np.median(score)).astype(int)

    df = pd.DataFrame({
        "income": income.round(2),
        "age": age,
        "debts": debts.round(2),
        "loan_amount": loan_amount.round(2),
        "payment_history_score": payment_history_score,
        "num_credit_lines": num_credit_lines,
        "employment_years": employment_years,
        "late_payments": late_payments,
        "debt_to_income": debt_to_income.round(3),
        "creditworthy": creditworthy
    })
    return df


print("Step 1: Creating dataset...")
df = create_dataset()
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nTarget distribution:\n{df['creditworthy'].value_counts()}")


# ----------------------------------------------------------
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# ----------------------------------------------------------
print("\nStep 2: Generating visualizations...")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.countplot(x="creditworthy", data=df)
plt.title("Class Distribution (0=Bad, 1=Good)")

plt.subplot(2, 2, 2)
sns.histplot(df["income"], bins=30, kde=True)
plt.title("Income Distribution")

plt.subplot(2, 2, 3)
sns.boxplot(x="creditworthy", y="debt_to_income", data=df)
plt.title("Debt-to-Income by Creditworthiness")

plt.subplot(2, 2, 4)
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
print("Saved: eda_plots.png")


# ----------------------------------------------------------
# STEP 3: PREPROCESSING
# ----------------------------------------------------------
print("\nStep 3: Preprocessing data...")

X = df.drop("creditworthy", axis=1)
y = df["creditworthy"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# ----------------------------------------------------------
# STEP 4: TRAIN MODELS
# ----------------------------------------------------------
print("\nStep 4: Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
}

results = {}

for name, model in models.items():
    # Logistic Regression benefits from scaled data; tree models don't need it
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba
    }


# ----------------------------------------------------------
# STEP 5: EVALUATE & COMPARE MODELS
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

summary = pd.DataFrame({
    name: {
        "Accuracy": f"{res['accuracy']:.3f}",
        "Precision": f"{res['precision']:.3f}",
        "Recall": f"{res['recall']:.3f}",
        "F1-Score": f"{res['f1']:.3f}",
        "ROC-AUC": f"{res['roc_auc']:.3f}"
    }
    for name, res in results.items()
}).T

print(summary)
summary.to_csv("model_comparison.csv")
print("\nSaved: model_comparison.csv")

# Detailed report for the best model (by ROC-AUC)
best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
print(f"\nBest model: {best_model_name}")
print("\nClassification Report:")
print(classification_report(y_test, results[best_model_name]["y_pred"]))


# ----------------------------------------------------------
# STEP 6: PLOT CONFUSION MATRIX & ROC CURVES
# ----------------------------------------------------------
print("\nStep 6: Plotting confusion matrix & ROC curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix for best model
cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
axes[0].set_title(f"Confusion Matrix - {best_model_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ROC curves for all models
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Random Guess")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves")
axes[1].legend()

plt.tight_layout()
plt.savefig("evaluation_plots.png", dpi=150)
print("Saved: evaluation_plots.png")

# ----------------------------------------------------------
# STEP 7: FEATURE IMPORTANCE (Random Forest)
# ----------------------------------------------------------
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved: feature_importance.png")

print("\n" + "=" * 60)
print("DONE! All results and plots have been saved.")
print("=" * 60)
