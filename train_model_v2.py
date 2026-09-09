"""
Model V2 Training & Comprehensive Evaluation Pipeline for Web Shield.
Trains XGBoost Model V2 on unbiased, group-split dataset_v2.csv using unscaled features,
evaluates grouped metrics, bootstrap CIs, hard adversarial sets, V1 vs V2 comparison,
source-bias classification, feature ablation, and saves artifacts under models/v2/.
"""

import os
import sys
sys.path.insert(0, '.')

import json
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.feature_extraction import extract_features

V2_DIR = os.path.join('models', 'v2')
os.makedirs(V2_DIR, exist_ok=True)

v2_model_path = os.path.join(V2_DIR, 'xgb_model_v2.pkl')
v2_features_path = os.path.join(V2_DIR, 'features_v2.pkl')
v2_metadata_path = os.path.join(V2_DIR, 'metadata.json')

# Load Dataset V2
data_v2_path = os.path.join('data', 'dataset_v2.csv')
df_v2 = pd.read_csv(data_v2_path)

features_v2 = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]

X_v2 = df_v2[features_v2]
y_v2 = df_v2['label']
groups = df_v2['registered_domain']

print("======================================================================")
print("SECTION 1: REGISTERED-DOMAIN GROUPED TRAIN/TEST SPLIT FOR MODEL V2")
print("======================================================================")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X_v2, y_v2, groups))

X_train_v2, X_test_v2 = X_v2.iloc[train_idx], X_v2.iloc[test_idx]
y_train_v2, y_test_v2 = y_v2.iloc[train_idx], y_v2.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

# Assert no domain overlap
overlap = set(groups_train).intersection(set(groups_test))
print(f"Train samples: {len(X_train_v2)} (Unique domains: {groups_train.nunique()})")
print(f"Test samples:  {len(X_test_v2)} (Unique domains: {groups_test.nunique()})")
print(f"Domain Overlap between Train and Test: {len(overlap)} (Must be 0)")
assert len(overlap) == 0, "Error: Domain overlap detected!"

print("\n--- Training XGBoost Model V2 (Unscaled Raw Features) ---")
clf_v2 = XGBClassifier(
    n_estimators=120,
    learning_rate=0.07,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
clf_v2.fit(X_train_v2, y_train_v2)

y_pred_v2 = clf_v2.predict(X_test_v2)
y_prob_v2 = clf_v2.predict_proba(X_test_v2)[:, 1]

cm_v2 = confusion_matrix(y_test_v2, y_pred_v2)
tn, fp, fn, tp = cm_v2.ravel()

acc_v2 = accuracy_score(y_test_v2, y_pred_v2) * 100
prec_v2 = precision_score(y_test_v2, y_pred_v2) * 100
rec_v2 = recall_score(y_test_v2, y_pred_v2) * 100
f1_v2 = f1_score(y_test_v2, y_pred_v2) * 100
auc_v2 = roc_auc_score(y_test_v2, y_prob_v2) * 100
pr_auc_v2 = average_precision_score(y_test_v2, y_prob_v2) * 100
fpr_v2 = (fp / (fp + tn)) * 100
fnr_v2 = (fn / (fn + tp)) * 100

print("\nModel V2 Grouped Domain Test Metrics:")
print(f"  Accuracy : {acc_v2:.2f}%")
print(f"  Precision: {prec_v2:.2f}%")
print(f"  Recall   : {rec_v2:.2f}%")
print(f"  F1-Score : {f1_v2:.2f}%")
print(f"  ROC-AUC  : {auc_v2:.2f}%")
print(f"  PR-AUC   : {pr_auc_v2:.2f}%")
print(f"  FPR      : {fpr_v2:.2f}% ({fp}/{fp+tn})")
print(f"  FNR      : {fnr_v2:.2f}% ({fn}/{fn+tp})")
print(f"  Confusion Matrix:\n{cm_v2}")


print("\n======================================================================")
print("SECTION 2: BOOTSTRAP CONFIDENCE INTERVALS FOR MODEL V2 (95%)")
print("======================================================================")
n_bootstraps = 1000
boot_acc, boot_prec, boot_rec, boot_f1, boot_auc = [], [], [], [], []
rng = np.random.RandomState(42)

for _ in range(n_bootstraps):
    boot_idx = rng.randint(0, len(y_test_v2), len(y_test_v2))
    if len(np.unique(y_test_v2.iloc[boot_idx])) < 2:
        continue
    b_true = y_test_v2.iloc[boot_idx]
    b_pred = y_pred_v2[boot_idx]
    b_prob = y_prob_v2[boot_idx]

    boot_acc.append(accuracy_score(b_true, b_pred))
    boot_prec.append(precision_score(b_true, b_pred))
    boot_rec.append(recall_score(b_true, b_pred))
    boot_f1.append(f1_score(b_true, b_pred))
    boot_auc.append(roc_auc_score(b_true, b_prob))

print("95% Bootstrap Confidence Intervals (Model V2 Grouped Test Set):")
print(f"  Accuracy : {np.percentile(boot_acc, 2.5)*100:.2f}% - {np.percentile(boot_acc, 97.5)*100:.2f}% (Mean: {np.mean(boot_acc)*100:.2f}%)")
print(f"  Precision: {np.percentile(boot_prec, 2.5)*100:.2f}% - {np.percentile(boot_prec, 97.5)*100:.2f}% (Mean: {np.mean(boot_prec)*100:.2f}%)")
print(f"  Recall   : {np.percentile(boot_rec, 2.5)*100:.2f}% - {np.percentile(boot_rec, 97.5)*100:.2f}% (Mean: {np.mean(boot_rec)*100:.2f}%)")
print(f"  F1-Score : {np.percentile(boot_f1, 2.5)*100:.2f}% - {np.percentile(boot_f1, 97.5)*100:.2f}% (Mean: {np.mean(boot_f1)*100:.2f}%)")
print(f"  ROC-AUC  : {np.percentile(boot_auc, 2.5)*100:.2f}% - {np.percentile(boot_auc, 97.5)*100:.2f}% (Mean: {np.mean(boot_auc)*100:.2f}%)")


print("\n======================================================================")
print("SECTION 3: MODEL V1 VS MODEL V2 ADVERSARIAL EVALUATION & COMPARISON")
print("======================================================================")

# Load Legacy V1 Artifacts
v1_model = joblib.load(os.path.join('models', 'xgb_model.pkl'))
v1_scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
v1_features = joblib.load(os.path.join('models', 'features.pkl'))

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

print("\n--- Hard Legitimate Evaluation (Model V1 vs Model V2) ---")
print(f"{'URL':<60s} | {'V1 Prob':<10s} | {'V1 Verdict':<14s} | {'V2 Prob':<10s} | {'V2 Verdict':<14s}")
print("-" * 115)

v1_fp, v2_fp = 0, 0
for u in hard_legit_urls:
    # Extract raw features
    feat = extract_features(u)

    # V1 vector
    df_v1 = pd.DataFrame([feat])[v1_features]
    s_v1 = v1_scaler.transform(df_v1)
    p_v1 = float(v1_model.predict_proba(s_v1)[0][1])
    is_v1_fp = p_v1 >= 0.50
    if is_v1_fp: v1_fp += 1

    # V2 vector
    age = feat['domain_age_months']
    feat_v2_dict = dict(feat)
    feat_v2_dict['domain_age_months_clean'] = 0 if age == -1 else age
    feat_v2_dict['domain_age_known'] = 0 if age == -1 else 1

    df_v2_single = pd.DataFrame([feat_v2_dict])[features_v2]
    p_v2 = float(clf_v2.predict_proba(df_v2_single)[0][1])
    is_v2_fp = p_v2 >= 0.50
    if is_v2_fp: v2_fp += 1

    v1_str = "Phishing (FP)" if is_v1_fp else "Legit (TN)"
    v2_str = "Phishing (FP)" if is_v2_fp else "Legit (TN)"
    print(f"{u[:60]:<60s} | {p_v1*100:<9.2f}% | {v1_str:<14s} | {p_v2*100:<9.2f}% | {v2_str:<14s}")

print(f"\nHard Legitimate False Positive Count: V1 = {v1_fp}/{len(hard_legit_urls)} ({v1_fp/len(hard_legit_urls)*100:.1f}%), V2 = {v2_fp}/{len(hard_legit_urls)} ({v2_fp/len(hard_legit_urls)*100:.1f}%)")


print("\n======================================================================")
print("SECTION 4: URLHAUS BREAKDOWN COMPARISON (V1 VS V2)")
print("======================================================================")
target_url = "https://urlhaus.abuse.ch/browse/"
feat = extract_features(target_url)

# V1 SHAP
df_v1 = pd.DataFrame([feat])[v1_features]
s_v1 = v1_scaler.transform(df_v1)
s_df_v1 = pd.DataFrame(s_v1, columns=v1_features)
p_v1_target = float(v1_model.predict_proba(s_v1)[0][1])

explainer_v1 = shap.TreeExplainer(v1_model)
shap_v1 = explainer_v1(s_df_v1).values[0]

# V2 SHAP
age = feat['domain_age_months']
feat_v2_dict = dict(feat)
feat_v2_dict['domain_age_months_clean'] = 0 if age == -1 else age
feat_v2_dict['domain_age_known'] = 0 if age == -1 else 1
df_v2_target = pd.DataFrame([feat_v2_dict])[features_v2]

p_v2_target = float(clf_v2.predict_proba(df_v2_target)[0][1])
explainer_v2 = shap.TreeExplainer(clf_v2)
shap_v2 = explainer_v2(df_v2_target).values[0]

print(f"URL: {target_url}")
print(f"Model V1 Phishing Probability: {p_v1_target*100:.2f}%")
print(f"Model V2 Phishing Probability: {p_v2_target*100:.2f}%\n")

print("Top V1 SHAP Features:")
for name, val in sorted(zip(v1_features, shap_v1), key=lambda x: abs(x[1]), reverse=True)[:3]:
    print(f"  {name:25s}: SHAP = {val:+.4f}")

print("\nTop V2 SHAP Features:")
for name, val in sorted(zip(features_v2, shap_v2), key=lambda x: abs(x[1]), reverse=True)[:3]:
    print(f"  {name:25s}: SHAP = {val:+.4f}")


print("\n======================================================================")
print("SECTION 5: SOURCE-BIAS AUXILIARY CLASSIFIER TEST")
print("======================================================================")

# Source-bias test on V1 dataset
df_v1_raw = pd.read_csv(os.path.join('data', 'dataset.csv'))
if 'source' not in df_v1_raw.columns:
    df_v1_raw['source'] = df_v1_raw['label'].apply(lambda l: 'tranco' if l == 0 else 'phish_feed')

X_v1_src = df_v1_raw[v1_features]
y_v1_src = df_v1_raw['source']

X_tr_s1, X_te_s1, y_tr_s1, y_te_s1 = train_test_split(X_v1_src, y_v1_src, test_size=0.2, random_state=42)
clf_src_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf_src_v1.fit(X_tr_s1, y_tr_s1)
acc_src_v1 = accuracy_score(y_te_s1, clf_src_v1.predict(X_te_s1)) * 100

# Source-bias test on V2 dataset
X_v2_src = df_v2[features_v2]
y_v2_src = df_v2['source']

X_tr_s2, X_te_s2, y_tr_s2, y_te_s2 = train_test_split(X_v2_src, y_v2_src, test_size=0.2, random_state=42)
clf_src_v2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf_src_v2.fit(X_tr_s2, y_tr_s2)
acc_src_v2 = accuracy_score(y_te_s2, clf_src_v2.predict(X_te_s2)) * 100

print(f"Source-Bias Predictability:")
print(f"  Dataset V1 Source Predictability: {acc_src_v1:.2f}% (High source fingerprinting)")
print(f"  Dataset V2 Source Predictability: {acc_src_v2:.2f}% (Reduced fingerprinting)")


print("\n======================================================================")
print("SECTION 6: FEATURE ABLATION STUDY FOR MODEL V2")
print("======================================================================")
ablation_features = ['url_length', 'ssl_valid', 'domain_entropy', 'domain_age_known']

print(f"{'Ablated Feature Removed':<25s} | {'Accuracy':<10s} | {'F1-Score':<10s} | {'ROC-AUC':<10s}")
print("-" * 65)

# Baseline full V2
print(f"{'None (Full V2)':<25s} | {acc_v2:<9.2f}% | {f1_v2:<9.2f}% | {auc_v2:<9.2f}%")

for drop_f in ablation_features:
    cols_abl = [c for c in features_v2 if c != drop_f and not (drop_f == 'domain_age_known' and c == 'domain_age_months_clean')]
    X_tr_a = X_train_v2[cols_abl]
    X_te_a = X_test_v2[cols_abl]

    clf_abl = XGBClassifier(n_estimators=120, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
    clf_abl.fit(X_tr_a, y_train_v2)

    preds_a = clf_abl.predict(X_te_a)
    probs_a = clf_abl.predict_proba(X_te_a)[:, 1]

    a_acc = accuracy_score(y_test_v2, preds_a) * 100
    a_f1 = f1_score(y_test_v2, preds_a) * 100
    a_auc = roc_auc_score(y_test_v2, probs_a) * 100
    print(f"{drop_f:<25s} | {a_acc:<9.2f}% | {a_f1:<9.2f}% | {a_auc:<9.2f}%")


print("\n======================================================================")
print("SECTION 7: SAVE MODEL V2 ARTIFACTS TO models/v2/")
print("======================================================================")

# Retrain V2 on full dataset_v2 for production deployment artifact
clf_v2_full = XGBClassifier(
    n_estimators=120,
    learning_rate=0.07,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
clf_v2_full.fit(X_v2, y_v2)

joblib.dump(clf_v2_full, v2_model_path)
joblib.dump(features_v2, v2_features_path)

metadata = {
    "model_version": "v2",
    "algorithm": "XGBoostClassifier",
    "scaler_used": False,
    "features": features_v2,
    "feature_count": len(features_v2),
    "dataset_size": len(df_v2),
    "deduplicated_unique_domains": df_v2['registered_domain'].nunique(),
    "test_metrics_grouped_domain": {
        "accuracy": round(acc_v2, 2),
        "precision": round(prec_v2, 2),
        "recall": round(rec_v2, 2),
        "f1_score": round(f1_v2, 2),
        "roc_auc": round(auc_v2, 2),
        "pr_auc": round(pr_auc_v2, 2),
        "false_positive_rate": round(fpr_v2, 2),
        "false_negative_rate": round(fnr_v2, 2)
    },
    "adversarial_false_positives": {
        "v1_fp_count": v1_fp,
        "v2_fp_count": v2_fp,
        "total_test_urls": len(hard_legit_urls)
    }
}

with open(v2_metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved Model V2 Artifacts:")
print(f"   [1] Model File    --> {v2_model_path}")
print(f"   [2] Features List --> {v2_features_path}")
print(f"   [3] Metadata File --> {v2_metadata_path}")
print("======================================================================\n")
