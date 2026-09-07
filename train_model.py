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
    accuracy_score, precision_score, recall_score, f1_score, 
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
    raise FileNotFoundError("❌ 'data/dataset.csv' not found! Please run generate_data.py first.")

df = pd.read_csv(data_path)
print(f"📊 Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

# Separate Features (X) and Target Label (y)
X = df.drop(columns=['label'])
y = df['label']

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
# 4. TRAIN ML MODELS (XGBoost & Random Forest)
# -------------------------------------------------------------
print("\n⚡ Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1, 
    random_state=42, 
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = xgb_model.predict(X_test_scaled)

# -------------------------------------------------------------
# 5. PRINT EVALUATION METRICS FOR REVIEW SLIDES
# -------------------------------------------------------------
acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred) * 100
rec = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

print("\n==================================================")
print("🎯 MODEL PERFORMANCE EVALUATION (Slide 7 Metrics)")
print("==================================================")
print(f"   Accuracy : {acc:.2f}%")
print(f"   Precision: {prec:.2f}%")
print(f"   Recall   : {rec:.2f}%")
print(f"   F1-Score : {f1:.2f}%")
print("==================================================\n")

print("Detailed Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# -------------------------------------------------------------
# 6. SAVE MODEL ARTIFACTS TO 'models/' FOLDER
# -------------------------------------------------------------
model_path = os.path.join('models', 'xgb_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')
features_path = os.path.join('models', 'features.pkl')

joblib.dump(xgb_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(list(X.columns), features_path)

print("✅ Saved Model    -->", model_path)
print("✅ Saved Scaler   -->", scaler_path)
print("✅ Saved Features -->", features_path)

# -------------------------------------------------------------
# 7. GENERATE & SAVE CONFUSION MATRIX PLOT (For Slide 7)
# -------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=['Legitimate (0)', 'Phishing (1)'],
    yticklabels=['Legitimate (0)', 'Phishing (1)'],
    annot_kws={'size': 14, 'weight': 'bold'}
)
plt.title('Web Shield - Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=10, fontweight='bold')
plt.ylabel('True Label', fontsize=10, fontweight='bold')
plt.tight_layout()

matrix_img_path = os.path.join('models', 'confusion_matrix.png')
plt.savefig(matrix_img_path, dpi=300)
print("✅ Saved Confusion Matrix Graphic -->", matrix_img_path)
print("\n🚀 All tasks complete! Ready for presentation screenshots.")
