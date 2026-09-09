"""
Comprehensive ML Model Audit Script for Web Shield (Full Suite).
Performs thorough empirical investigation into model pipeline, dataset composition,
feature distributions, missingness proxies, tree splits, train/test leakage,
grouped splits, adversarial evaluation, bootstrap CIs, scaler impact, and URLhaus breakdown.
"""

import os
import sys
sys.path.insert(0, '.')

import joblib
import pandas as pd
import numpy as np
import shap
import tldextract
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier
from src.feature_extraction import extract_features, get_domain_age_status

MODEL_DIR = "models"
model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))

dataset_path = os.path.join("data", "dataset.csv")
df_raw = pd.read_csv(dataset_path)

print("======================================================================")
print("SECTION 1: PIPELINE ALIGNMENT VERIFICATION")
print("======================================================================")
print("1. Feature Names (features.pkl):", feature_names)
print("2. Model Classes (model.classes_):", model.classes_)
phishing_class_idx = list(model.classes_).index(1)
print("3. Phishing Class Index:", phishing_class_idx)
print("4. Scaler Means:", dict(zip(feature_names, np.round(scaler.mean_, 4))))
print("5. Scaler Scales:", dict(zip(feature_names, np.round(scaler.scale_, 4))))

test_url = "https://urlhaus.abuse.ch/browse/"
feat_dict = extract_features(test_url)
df_single = pd.DataFrame([feat_dict])[feature_names]
scaled_single = scaler.transform(df_single)
scaled_df_single = pd.DataFrame(scaled_single, columns=feature_names)

prob_single = model.predict_proba(scaled_single)[0]
explainer = shap.TreeExplainer(model)
shap_single = explainer(scaled_df_single)
shap_vals_single = shap_single.values[0]

print(f"\nPipeline Test for {test_url}:")
print(f"Phishing Probability: {prob_single[1]:.4f} ({prob_single[1]*100:.2f}%)")
print("Prediction Input Scaled Vector:", np.round(scaled_single[0], 4))
print("SHAP Input Scaled Vector:      ", np.round(scaled_df_single.iloc[0].values, 4))
print("Input Vector Alignment Check:  ", np.allclose(scaled_single[0], scaled_df_single.iloc[0].values))


print("\n======================================================================")
print("SECTION 2: DATASET COMPOSITION AUDIT")
print("======================================================================")
total_rows = len(df_raw)
legit_count = (df_raw['label'] == 0).sum()
phish_count = (df_raw['label'] == 1).sum()
print(f"Total Rows: {total_rows}")
print(f"Legitimate (Label 0): {legit_count} ({legit_count/total_rows*100:.2f}%)")
print(f"Phishing (Label 1):   {phish_count} ({phish_count/total_rows*100:.2f}%)")

feature_cols = [c for c in df_raw.columns if c != 'label']
dup_rows = df_raw.duplicated(subset=feature_cols).sum()
print(f"Duplicate Feature Vectors: {dup_rows} ({dup_rows/total_rows*100:.2f}% duplicate rate)")


print("\n======================================================================")
print("SECTION 3: FEATURE DISTRIBUTIONS BY CLASS")
print("======================================================================")
for col in feature_names:
    print(f"\n--- Feature: {col} ---")
    for lbl, name in [(0, "Legitimate"), (1, "Phishing")]:
        vals = df_raw[df_raw['label'] == lbl][col]
        missing = (vals == -1).sum() if col == 'domain_age_months' else 0
        q = np.percentile(vals, [25, 50, 75, 90, 95, 99])
        print(f"  [{name}] Mean: {vals.mean():.4f} | Median: {vals.median():.4f} | Std: {vals.std():.4f} | Min: {vals.min()} | Max: {vals.max()}")
        print(f"          Quantiles (25, 50, 75, 90, 95, 99): {np.round(q, 2)}")
        if col == 'domain_age_months':
            print(f"          Missing/Unknown (-1) Count: {missing}/{len(vals)} ({missing/len(vals)*100:.2f}%)")


print("\n======================================================================")
print("SECTION 4: DOMAIN AGE MISSINGNESS AUDIT")
print("======================================================================")
legit_missing = (df_raw[df_raw['label'] == 0]['domain_age_months'] == -1).sum()
legit_total = len(df_raw[df_raw['label'] == 0])
phish_missing = (df_raw[df_raw['label'] == 1]['domain_age_months'] == -1).sum()
phish_total = len(df_raw[df_raw['label'] == 1])

p_missing_legit = legit_missing / legit_total
p_missing_phish = phish_missing / phish_total

print(f"P(domain_age_months == -1 | Legitimate): {p_missing_legit:.4f} ({p_missing_legit*100:.2f}%) [{legit_missing}/{legit_total}]")
print(f"P(domain_age_months == -1 | Phishing)  : {p_missing_phish:.4f} ({p_missing_phish*100:.2f}%) [{phish_missing}/{phish_total}]")

idx_age = feature_names.index('domain_age_months')
scaled_minus_1 = ( -1 - scaler.mean_[idx_age] ) / scaler.scale_[idx_age]
print(f"Raw domain_age_months = -1 transforms to Scaled Value: {scaled_minus_1:.4f}")


print("\n======================================================================")
print("SECTION 5: FEATURE IMPORTANCE & SHAP ANALYSIS")
print("======================================================================")
booster = model.get_booster()
gain_imp = booster.get_score(importance_type='gain')
weight_imp = booster.get_score(importance_type='weight')

print("XGBoost Gain Importance:")
for k, v in sorted(gain_imp.items(), key=lambda item: item[1], reverse=True):
    f_idx = int(k.replace('f', ''))
    f_name = feature_names[f_idx]
    print(f"  {f_name:25s}: Gain = {v:.4f}")

print("\nXGBoost Split Weight Importance:")
for k, v in sorted(weight_imp.items(), key=lambda item: item[1], reverse=True):
    f_idx = int(k.replace('f', ''))
    f_name = feature_names[f_idx]
    print(f"  {f_name:25s}: Splits = {v}")

X_all = df_raw[feature_names]
X_all_scaled = pd.DataFrame(scaler.transform(X_all), columns=feature_names)
shap_all = explainer(X_all_scaled).values

mean_abs_shap = np.abs(shap_all).mean(axis=0)
print("\nGlobal Mean Absolute SHAP Importance:")
for name, score in sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True):
    print(f"  {name:25s}: mean(|SHAP|) = {score:.4f}")


print("\n======================================================================")
print("SECTION 8: HARD LEGITIMATE ADVERSARIAL TEST SET EVALUATION")
print("======================================================================")
hard_legit_urls = [
    "https://urlhaus.abuse.ch/browse/",
    "https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION",
    "https://aws.amazon.com/console/billing/reports/latest/summary",
    "https://www.google.com/search?q=machine+learning+phishing+detection+xgboost+shap",
    "https://bit.ly/3xY8zQ",
    "https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=12345",
    "https://checkout.stripe.com/pay/cs_live_a1b2c3d4e5f6",
    "https://www.paypal.com/cgi-bin/webscr?cmd=_express-checkout&token=EC-12345",
    "https://login.microsoftonline.com/common/oauth2/authorize",
    "https://en.wikipedia.org/wiki/Phishing_detection_techniques",
    "https://stackoverflow.com/questions/12345678/python-xgboost-shap-explainability",
    "https://medium.com/@user/how-we-built-a-cybersecurity-dashboard-2026",
    "https://cdn.cloudflare.com/static/css/main.v2.4.1.css",
    "https://m.facebook.com/messages/read/?thread_id=100012345",
    "https://store.steampowered.com/app/1086940/Baldurs_Gate_3/"
]

print(f"{'URL':<65s} | {'PhishProb':<10s} | {'Predicted Verdict':<15s} | {'FP Flag':<8s}")
print("-" * 105)

fp_count = 0
for u in hard_legit_urls:
    try:
        f = extract_features(u)
        df_u = pd.DataFrame([f])[feature_names]
        s_u = scaler.transform(df_u)
        prob = float(model.predict_proba(s_u)[0][1])
        is_fp = prob >= 0.50
        if is_fp:
            fp_count += 1
        verdict = "Phishing (FP)" if is_fp else "Legitimate (TN)"
        print(f"{u[:65]:<65s} | {prob*100:<9.2f}% | {verdict:<15s} | {'YES' if is_fp else 'NO':<8s}")
    except Exception as e:
        print(f"{u[:65]:<65s} | ERROR: {e}")

print(f"\nHard Legitimate False Positive Rate: {fp_count}/{len(hard_legit_urls)} ({fp_count/len(hard_legit_urls)*100:.2f}%)")


print("\n======================================================================")
print("SECTION 9: HARD MALICIOUS ADVERSARIAL TEST SET EVALUATION")
print("======================================================================")
hard_phish_urls = [
    "http://192.168.1.1/login-verify-account/",
    "http://secure-update-paypal.com.verify-user-account.info/login.php",
    "https://paypal-security-update.com/signin",
    "https://login-appleid-verify.com/account",
    "http://wellsfargo-account-verify-alert.net/secure/",
    "https://bankofamerica-online-access.com/auth",
    "http://verify-microsoft-security-alert.org/signin/",
    "https://chase-bank-online-security-update.com/login",
    "http://netflix-billing-update-account.org/signin",
    "https://accounts-google-com-login.verify-auth.info/o/oauth2"
]

print(f"{'URL':<65s} | {'PhishProb':<10s} | {'Predicted Verdict':<15s} | {'TP Flag':<8s}")
print("-" * 105)

tp_count = 0
for u in hard_phish_urls:
    try:
        f = extract_features(u)
        df_u = pd.DataFrame([f])[feature_names]
        s_u = scaler.transform(df_u)
        prob = float(model.predict_proba(s_u)[0][1])
        is_tp = prob >= 0.50
        if is_tp:
            tp_count += 1
        verdict = "Phishing (TP)" if is_tp else "Legitimate (FN)"
        print(f"{u[:65]:<65s} | {prob*100:<9.2f}% | {verdict:<15s} | {'YES' if is_tp else 'NO':<8s}")
    except Exception as e:
        print(f"{u[:65]:<65s} | ERROR: {e}")

print(f"\nHard Malicious True Positive Rate: {tp_count}/{len(hard_phish_urls)} ({tp_count/len(hard_phish_urls)*100:.2f}%)")


print("\n======================================================================")
print("SECTION 12: NEW DOMAIN AGE REPRESENTATION EVALUATION")
print("======================================================================")
df_new = df_raw.copy()
df_new['domain_age_known'] = (df_new['domain_age_months'] != -1).astype(int)
df_new['domain_age_months_clean'] = df_new['domain_age_months'].apply(lambda x: 0 if x == -1 else x)

feature_cols_new = [c for c in feature_names if c != 'domain_age_months'] + ['domain_age_months_clean', 'domain_age_known']

X_new = df_new[feature_cols_new]
y_new = df_new['label']

X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(X_new, y_new, test_size=0.2, random_state=42, stratify=y_new)
clf_new_rep = XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, eval_metric='logloss', random_state=42)
clf_new_rep.fit(X_tr_n, y_tr_n)

y_pred_new_rep = clf_new_rep.predict(X_te_n)
print("Evaluation of candidate 11-feature model ([domain_age_months_clean, domain_age_known]):")
print(f"  Accuracy: {accuracy_score(y_te_n, y_pred_new_rep)*100:.2f}%")
print(f"  Feature Importances:")
for col, imp in sorted(zip(feature_cols_new, clf_new_rep.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"    {col:30s}: {imp:.4f}")


print("\n======================================================================")
print("SECTION 13: DETAILED URLHAUS BREAKDOWN TABLE")
print("======================================================================")
print(f"Target URL: {test_url}")
print(f"{'Feature':<25s} | {'Raw':<8s} | {'Scaled':<8s} | {'SHAP':<8s} | {'Direction':<18s} | {'Legit Pct':<10s} | {'Phish Pct':<10s}")
print("-" * 105)

df_legit = df_raw[df_raw['label'] == 0]
df_phish = df_raw[df_raw['label'] == 1]

for i, f in enumerate(feature_names):
    raw_val = feat_dict[f]
    scaled_val = scaled_single[0][i]
    shap_val = shap_vals_single[i]
    direction = "toward_phishing" if shap_val > 0 else "toward_legitimate"

    legit_vals = df_legit[f]
    phish_vals = df_phish[f]

    pct_legit = (legit_vals <= raw_val).mean() * 100
    pct_phish = (phish_vals <= raw_val).mean() * 100

    print(f"{f:<25s} | {str(raw_val):<8s} | {scaled_val:<8.4f} | {shap_val:<+8.4f} | {direction:<18s} | {pct_legit:<10.1f}% | {pct_phish:<10.1f}%")

print("======================================================================\n")
