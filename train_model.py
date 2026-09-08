import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# -------------------------------------------------------------
# 1. ENSURE OUTPUT DIRECTORY EXISTS
# -------------------------------------------------------------
os.makedirs('models', exist_ok=True)

# -------------------------------------------------------------
# 2. LOAD DATASET
# -------------------------------------------------------------
data_path = os.path.join('data', 'dataset.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError("Dataset file 'data/dataset.csv' not found! Please run generate_data.py first.")

df = pd.read_csv(data_path)
print(f"Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

# Separate Features (X) and Target Label (y)
X = df.drop(columns=['label'])
y = df['label']
feature_names = list(X.columns)

print(f"Feature Columns ({len(feature_names)}): {feature_names}")

# -------------------------------------------------------------
# 3. TRAIN-TEST SPLIT & FEATURE SCALING
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# 4. TRAIN ML MODELS (XGBoost & Random Forest Baseline)
# -------------------------------------------------------------
print("\n--- Training Models ---")

# 4a. XGBoost Classifier
print("Training XGBoost Classifier (n_estimators=150, learning_rate=0.08, max_depth=5)...")
xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=5,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

# 4b. Random Forest Classifier Baseline
print("Training Random Forest Classifier Baseline (n_estimators=100)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# -------------------------------------------------------------
# 5. METRIC COMPUTATION & MODEL COMPARISON
# -------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred) * 100,
        'Precision': precision_score(y_true, y_pred) * 100,
        'Recall': recall_score(y_true, y_pred) * 100,
        'F1-Score': f1_score(y_true, y_pred) * 100,
        'ROC-AUC': roc_auc_score(y_true, y_proba) * 100
    }

xgb_metrics = compute_metrics(y_test, y_pred_xgb, y_proba_xgb)
rf_metrics = compute_metrics(y_test, y_pred_rf, y_proba_rf)

comparison_df = pd.DataFrame([rf_metrics, xgb_metrics], index=['RandomForest Baseline', 'XGBoost Classifier'])

print("\n==================================================")
print("MODEL PERFORMANCE COMPARISON SUMMARY")
print("==================================================")
print(comparison_df.round(2).to_string())
print("==================================================\n")

print("Detailed XGBoost Classification Report:\n")
print(classification_report(y_test, y_pred_xgb, target_names=['Legitimate (0)', 'Phishing (1)']))

# -------------------------------------------------------------
# 6. SAVE MODEL ARTIFACTS TO 'models/' FOLDER
# -------------------------------------------------------------
model_path = os.path.join('models', 'xgb_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')
features_path = os.path.join('models', 'features.pkl')

joblib.dump(xgb_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(feature_names, features_path)

print("Saved Model Artifacts:")
print(f"   [1] Model File    --> {model_path}")
print(f"   [2] Scaler File   --> {scaler_path}")
print(f"   [3] Features List --> {features_path}")

# -------------------------------------------------------------
# 7. GENERATE & SAVE CONFUSION MATRIX PLOT
# -------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=['Legitimate (0)', 'Phishing (1)'],
    yticklabels=['Legitimate (0)', 'Phishing (1)'],
    annot_kws={'size': 14, 'weight': 'bold'}
)
plt.title('Web Shield - XGBoost Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=10, fontweight='bold')
plt.ylabel('True Label', fontsize=10, fontweight='bold')
plt.tight_layout()

matrix_img_path = os.path.join('models', 'confusion_matrix.png')
plt.savefig(matrix_img_path, dpi=300)
print(f"   [4] Heatmap Chart --> {matrix_img_path}")

print("\nAll 4 model artifacts successfully updated in 'models/' directory!")
