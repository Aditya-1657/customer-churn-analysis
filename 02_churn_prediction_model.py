"""
Customer Churn Analysis - Predictive Model
Builds a Logistic Regression + Random Forest model to predict churn
and extracts the top factors driving churn.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------
# 1. LOAD CLEAN DATA
# ---------------------------------------------------------
df = pd.read_csv("data/telco_churn_clean.csv")

# ---------------------------------------------------------
# 2. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------
df_model = df.copy()
target = df_model.pop("Churn").map({"No": 0, "Yes": 1})

cat_cols = df_model.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, target, test_size=0.2, random_state=42, stratify=target
)

scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------
# 4. LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------
# 5. RANDOM FOREST MODEL
# ---------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 6. EVALUATE BOTH MODELS
# ---------------------------------------------------------
def evaluate(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred), 3),
        "Recall": round(recall_score(y_true, y_pred), 3),
        "F1-score": round(f1_score(y_true, y_pred), 3),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 3),
    }

results = pd.DataFrame([
    evaluate("Logistic Regression", y_test, y_pred_lr, y_prob_lr),
    evaluate("Random Forest", y_test, y_pred_rf, y_prob_rf),
])
print(results.to_string(index=False))
results.to_csv("model_results.csv", index=False)

# ---------------------------------------------------------
# 7. CONFUSION MATRIX (Random Forest - best model typically)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.title("Confusion Matrix - Random Forest", fontsize=13, fontweight="bold")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("images/07_confusion_matrix.png")
plt.close()

# ---------------------------------------------------------
# 8. FEATURE IMPORTANCE (what actually drives churn)
# ---------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(12)

plt.figure(figsize=(7, 5.5))
top_features.sort_values().plot(kind="barh", color="#2E86AB")
plt.title("Top 12 Factors Driving Customer Churn", fontsize=13, fontweight="bold")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("images/08_feature_importance.png")
plt.close()

print("\nTop churn drivers:\n", top_features)

with open("model_results.txt", "w") as f:
    f.write(results.to_string(index=False))
    f.write("\n\nTop churn drivers (Random Forest feature importance):\n")
    f.write(top_features.to_string())

print("\nModel evaluation + feature importance charts saved.")
