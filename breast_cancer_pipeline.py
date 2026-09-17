"""
Breast Cancer Wisconsin Dataset — Train/Test + Evaluation
Purpose: Comparison dataset to see how accuracy differs when features are
strongly predictive of the target (unlike churn, where key drivers are missing).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# ============================================================
# 1. LOAD
# ============================================================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign

print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"\nClass balance:\n{df['target'].value_counts()}")

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 3. MODEL TRAINING + EVALUATION
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = {}
fig, axes = plt.subplots(1, len(models), figsize=(16, 4.5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    print(f"\n{'='*55}")
    print(f"MODEL: {name}")
    print(f"{'='*55}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_test, preds, target_names=["Malignant", "Benign"]))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"]
    )
    ax.set_title(f"{name}\nAcc: {acc:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
print("\nSaved confusion matrix plot -> confusion_matrices.png")

results_df = pd.DataFrame(results).T
print("\n", results_df)
results_df.to_csv("breast_cancer_results_summary.csv")
