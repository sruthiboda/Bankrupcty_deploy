import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# =============== CONFIG ===============
DATA_PATHS = [
    "D:/Amazon_ML/student_resource/1year.csv",
    "D:/Amazon_ML/student_resource/2year.csv",
    "D:/Amazon_ML/student_resource/3year.csv",
    "D:/Amazon_ML/student_resource/4year.csv",
    "D:/Amazon_ML/student_resource/5year.csv"
]
SCALER_PATH = "scaler_all.pkl"
XGB_MODEL_PATH = "xgb_all.joblib"
ANN_MODEL_PATH = "ann_all.keras"
META_MODEL_PATH = "meta_logreg_all.joblib"
# ======================================

print("📊 Loading datasets...")
dfs = [pd.read_csv(path) for path in DATA_PATHS]
df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"✅ Combined dataframe shape: {df.shape}")

# ====== CLEANING ======
df = df.dropna()
df = df.drop_duplicates()
X = df.drop(columns=["class"])
y = df["class"]

# Split same as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ====== SCALING ======
scaler = joblib.load(SCALER_PATH)
X_test_scaled = scaler.transform(X_test)

# ====== LOAD MODELS ======
print("🔄 Loading trained models...")
xgb_model = joblib.load(XGB_MODEL_PATH)
ann_model = load_model(ANN_MODEL_PATH)
meta_model = joblib.load(META_MODEL_PATH)

# ====== PREDICTIONS ======
print("🔍 Generating predictions...")
xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
ann_pred = ann_model.predict(X_test_scaled, verbose=0).flatten()

# Stack predictions for meta model
meta_input = np.vstack([xgb_pred, ann_pred]).T
meta_pred_prob = meta_model.predict_proba(meta_input)[:, 1]
meta_pred = (meta_pred_prob > 0.5).astype(int)

# ====== METRICS ======
def print_metrics(name, y_true, y_pred, y_prob=None):
    print(f"\n📈 --- {name} PERFORMANCE ---")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    if auc: print(f"ROC-AUC:   {auc:.4f}")
    return acc, prec, rec, f1, auc

print_metrics("XGBoost", y_test, (xgb_pred > 0.5).astype(int), xgb_pred)
print_metrics("ANN", y_test, (ann_pred > 0.5).astype(int), ann_pred)
print_metrics("Stacked Meta-Model", y_test, meta_pred, meta_pred_prob)

# ====== CONFUSION MATRIX ======
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, meta_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Stacked Meta Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ====== ROC CURVES ======
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pred)
fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_pred)
fpr_meta, tpr_meta, _ = roc_curve(y_test, meta_pred_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")
plt.plot(fpr_ann, tpr_ann, label="ANN")
plt.plot(fpr_meta, tpr_meta, label="Meta-Model", linewidth=2.5)
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ====== FEATURE IMPORTANCE ======
plt.figure(figsize=(10,6))
importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
top_features = X.columns[sorted_idx][:20]
sns.barplot(x=importance[sorted_idx][:20], y=top_features, palette="viridis")
plt.title("Top 20 Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.show()

# ====== EDA PLOTS ======
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\n✅ Evaluation complete. Metrics, confusion matrix, ROC, and EDA plotted successfully.")
