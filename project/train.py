# train_all.py
import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# CONFIG
# -------------------------
DATA_FOLDER = os.getcwd()
CSV_GLOBS = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))
OUT_DIR = os.path.join(DATA_FOLDER, "hybrid_models_all")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 64

print("🚀 Training on combined dataset from:", CSV_GLOBS)

# -------------------------
# Load + validate + concat
# -------------------------
dfs = []
expected_cols = None
for path in CSV_GLOBS:
    print("📂 Reading", path)
    df = pd.read_csv(path)
    # drop rows with all NaNs
    df = df.dropna(how="all")
    # Ensure last col is class
    if expected_cols is None:
        expected_cols = list(df.columns)
    else:
        if list(df.columns) != expected_cols:
            raise SystemExit(f"❌ Column mismatch between files. {path} columns differ.")
    dfs.append(df)

if not dfs:
    raise SystemExit("❌ No CSV files found to merge.")

big_df = pd.concat(dfs, ignore_index=True)
print(f"✅ Merged dataframe shape: {big_df.shape}")

# -------------------------
# Basic cleaning + label handling
# -------------------------
# Assume last column is class
label_col = big_df.columns[-1]
X = big_df.iloc[:, :-1].copy()
y = big_df[label_col].copy()

# Clean labels: force 0/1
if y.dtype == "object" or not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("🔤 LabelEncoder classes:", list(le.classes_))

y = y.astype(int)

# remove rows with any NaNs in features
rows_before = X.shape[0]
mask = X.isnull().any(axis=1)
if mask.any():
    X = X[~mask]
    y = y[~mask]
    print(f"🧹 Dropped {mask.sum()} rows with NaNs; remaining {X.shape[0]} rows.")

# remove duplicate rows if any (optional)
dups = X.duplicated().sum()
if dups:
    print(f"🧾 Found {dups} duplicate feature rows. Dropping duplicates.")
    X = X.drop_duplicates()
    y = y.loc[X.index]

# -------------------------
# Train/test split (stratified)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Class distribution (train):", np.bincount(y_train))
print("Class distribution (test):", np.bincount(y_test))

# -------------------------
# Scale features and save scaler
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler_all.pkl"))
print("💾 Saved scaler to scaler_all.pkl")

# -------------------------
# XGBoost training (with scale_pos_weight)
# -------------------------
# compute scale_pos_weight = #neg / #pos
num_pos = int(y_train.sum())
num_neg = y_train.shape[0] - num_pos
scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0
print("⚖ scale_pos_weight (neg/pos) =", scale_pos_weight)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight
)
print("🟩 Training XGBoost...")
xgb.fit(X_train_scaled, y_train)
xgb_pred_proba = xgb.predict_proba(X_test_scaled)[:, 1]
joblib.dump(xgb, os.path.join(OUT_DIR, "xgb_all.joblib"))
print("💾 Saved XGBoost to xgb_all.joblib")

# -------------------------
# ANN (Keras) training with class weights
# -------------------------
# compute class weight
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print("⚖ class_weights for ANN:", class_weights_dict)

def build_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

print("🔵 Building and training ANN...")
ann = build_ann(X_train_scaled.shape[1])
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
ann.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[es],
    verbose=1
)
# Save ANN in Keras native format
ann.save(os.path.join(OUT_DIR, "ann_all.keras"))
print("💾 Saved ANN to ann_all.keras")

# -------------------------
# Build stacking meta-model on validation predictions
# -------------------------
print("🔗 Building stacking features for meta-model using 5-fold CV on train set...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
meta_features = np.zeros((X_train_scaled.shape[0], 2))  # columns: xgb_proba, ann_proba
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    # train per-fold models
    xgb_fold = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight
    )
    xgb_fold.fit(X_train_scaled[tr_idx], y_train.iloc[tr_idx])
    xgb_val_proba = xgb_fold.predict_proba(X_train_scaled[val_idx])[:, 1]

    ann_fold = build_ann(X_train_scaled.shape[1])
    ann_fold.fit(
        X_train_scaled[tr_idx], y_train.iloc[tr_idx],
        validation_split=0.1, epochs=30, batch_size=BATCH_SIZE, class_weight=class_weights_dict,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)], verbose=0
    )
    ann_val_proba = ann_fold.predict(X_train_scaled[val_idx]).flatten()

    meta_features[val_idx, 0] = xgb_val_proba
    meta_features[val_idx, 1] = ann_val_proba
    print(f"  fold {fold+1} done")

# Fit meta-model
meta_clf = LogisticRegression(max_iter=2000)
meta_clf.fit(meta_features, y_train)
joblib.dump(meta_clf, os.path.join(OUT_DIR, "meta_logreg_all.joblib"))
print("💾 Saved meta logistic regression to meta_logreg_all.joblib")

# -------------------------
# Final test-time evaluation using trained base models (xgb & ann) + meta-model
# -------
