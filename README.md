# CodeAlpha_CreditScoringModel

## 📌 Task 1: Credit Scoring Model
**Internship:** Machine Learning Internship @ CodeAlpha

### 🎯 Objective
Predict an individual's **creditworthiness** (Good vs Bad) using financial history data such as income, debts, payment history, and loan amount.

### ⚙️ Approach
- Generated a realistic financial dataset (income, debts, payment history, credit lines, etc.)
- Performed feature engineering (e.g., debt-to-income ratio)
- Trained and compared **3 classification models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluated using **Precision, Recall, F1-Score, and ROC-AUC**

### 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.930 | 0.943 | 0.915 | 0.929 | 0.979 |
| Decision Tree | 0.863 | 0.880 | 0.840 | 0.859 | 0.904 |
| Random Forest | 0.910 | 0.927 | 0.890 | 0.908 | 0.968 |

**Best Model:** Logistic Regression (highest ROC-AUC)

### 📁 Project Structure
```
CodeAlpha_CreditScoringModel/
├── credit_scoring_model.py     # Main script (run this)
├── requirements.txt            # Dependencies
├── model_comparison.csv        # Output: model metrics
├── eda_plots.png               # Output: exploratory data analysis
├── evaluation_plots.png        # Output: confusion matrix + ROC curves
├── feature_importance.png      # Output: feature importance chart
└── README.md
```

### ▶️ How to Run
```bash
pip install -r requirements.txt
python credit_scoring_model.py
```

### 🧠 Key Learnings
- How to engineer meaningful financial features (e.g., debt-to-income ratio)
- Comparing classification algorithms fairly using multiple metrics (not just accuracy)
- Why ROC-AUC is important for imbalanced/binary classification problems
- Interpreting model results using confusion matrices and feature importance

### 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---
**#CodeAlpha #MachineLearning #Internship**
