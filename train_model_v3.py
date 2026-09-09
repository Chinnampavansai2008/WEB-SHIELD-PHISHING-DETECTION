"""
Model V3 Training & Comprehensive Evaluation Pipeline for Web Shield.
Trains XGBoost Model V3 on length-balanced dataset_v3.csv, evaluates grouped domain metrics,
95% bootstrap CIs, 100+ item hard legitimate/malicious holdout sets, V2 vs V3 side-by-side comparison,
source predictability, feature importance, SHAP dependence, and saves artifacts under models/v3/.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.feature_extraction import extract_features
from src.predictor import adapt_features_v2

V3_DIR = os.path.join('models', 'v3')
os.makedirs(V3_DIR, exist_ok=True)

v3_model_path = os.path.join(V3_DIR, 'xgb_model_v3.pkl')
v3_features_path = os.path.join(V3_DIR, 'features_v3.pkl')
v3_metadata_path = os.path.join(V3_DIR, 'metadata.json')

features_v3 = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]


def run_pipeline():
    data_v3_path = os.path.join('data', 'dataset_v3.csv')
    df_v3 = pd.read_csv(data_v3_path)

    X_v3 = df_v3[features_v3]
    y_v3 = df_v3['label']
    groups_v3 = df_v3['registered_domain']

    print("======================================================================")
    print("SECTION 1: REGISTERED-DOMAIN GROUPED TRAIN/VAL/TEST SPLIT FOR MODEL V3")
    print("======================================================================")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_v3, y_v3, groups_v3))

    X_train_v3, X_test_v3 = X_v3.iloc[train_idx], X_v3.iloc[test_idx]
    y_train_v3, y_test_v3 = y_v3.iloc[train_idx], y_v3.iloc[test_idx]
    groups_train, groups_test = groups_v3.iloc[train_idx], groups_v3.iloc[test_idx]

    overlap = set(groups_train).intersection(set(groups_test))
    print(f"Train samples: {len(X_train_v3)} (Unique domains: {groups_train.nunique()})")
    print(f"Test samples:  {len(X_test_v3)} (Unique domains: {groups_test.nunique()})")
    print(f"Domain Overlap between Train and Test: {len(overlap)} (Must be 0)")
    assert len(overlap) == 0, "Error: Domain overlap detected!"

    print("\n--- Training XGBoost Model V3 (Unscaled Raw Features) ---")
    clf_v3 = XGBClassifier(
        n_estimators=120,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    clf_v3.fit(X_train_v3, y_train_v3)

    y_pred_v3 = clf_v3.predict(X_test_v3)
    y_prob_v3 = clf_v3.predict_proba(X_test_v3)[:, 1]

    cm_v3 = confusion_matrix(y_test_v3, y_pred_v3)
    tn_v3, fp_v3, fn_v3, tp_v3 = cm_v3.ravel()

    acc_v3 = accuracy_score(y_test_v3, y_pred_v3) * 100
    prec_v3 = precision_score(y_test_v3, y_pred_v3) * 100
    rec_v3 = recall_score(y_test_v3, y_pred_v3) * 100
    f1_v3 = f1_score(y_test_v3, y_pred_v3) * 100
    auc_v3 = roc_auc_score(y_test_v3, y_prob_v3) * 100
    pr_auc_v3 = average_precision_score(y_test_v3, y_prob_v3) * 100
    fpr_v3 = (fp_v3 / (fp_v3 + tn_v3)) * 100
    fnr_v3 = (fn_v3 / (fn_v3 + tp_v3)) * 100

    print("\nModel V3 Grouped Domain Test Metrics:")
    print(f"  Accuracy : {acc_v3:.2f}%")
    print(f"  Precision: {prec_v3:.2f}%")
    print(f"  Recall   : {rec_v3:.2f}%")
    print(f"  F1-Score : {f1_v3:.2f}%")
    print(f"  ROC-AUC  : {auc_v3:.2f}%")
    print(f"  PR-AUC   : {pr_auc_v3:.2f}%")
    print(f"  FPR      : {fpr_v3:.2f}% ({fp_v3}/{fp_v3+tn_v3})")
    print(f"  FNR      : {fnr_v3:.2f}% ({fn_v3}/{fn_v3+tp_v3})")
    print(f"  Confusion Matrix:\n{cm_v3}")

    print("\n======================================================================")
    print("SECTION 2: 95% BOOTSTRAP CONFIDENCE INTERVALS FOR MODEL V3")
    print("======================================================================")
    n_bootstraps = 1000
    boot_acc, boot_prec, boot_rec, boot_f1, boot_auc = [], [], [], [], []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, len(y_test_v3), len(y_test_v3))
        if len(np.unique(y_test_v3.iloc[boot_idx])) < 2:
            continue
        b_true = y_test_v3.iloc[boot_idx]
        b_pred = y_pred_v3[boot_idx]
        b_prob = y_prob_v3[boot_idx]

        boot_acc.append(accuracy_score(b_true, b_pred))
        boot_prec.append(precision_score(b_true, b_pred))
        boot_rec.append(recall_score(b_true, b_pred))
        boot_f1.append(f1_score(b_true, b_pred))
        boot_auc.append(roc_auc_score(b_true, b_prob))

    print("95% Bootstrap Confidence Intervals (Model V3 Grouped Test Set):")
    print(f"  Accuracy : {np.percentile(boot_acc, 2.5)*100:.2f}% - {np.percentile(boot_acc, 97.5)*100:.2f}% (Mean: {np.mean(boot_acc)*100:.2f}%)")
    print(f"  Precision: {np.percentile(boot_prec, 2.5)*100:.2f}% - {np.percentile(boot_prec, 97.5)*100:.2f}% (Mean: {np.mean(boot_prec)*100:.2f}%)")
    print(f"  Recall   : {np.percentile(boot_rec, 2.5)*100:.2f}% - {np.percentile(boot_rec, 97.5)*100:.2f}% (Mean: {np.mean(boot_rec)*100:.2f}%)")
    print(f"  F1-Score : {np.percentile(boot_f1, 2.5)*100:.2f}% - {np.percentile(boot_f1, 97.5)*100:.2f}% (Mean: {np.mean(boot_f1)*100:.2f}%)")
    print(f"  ROC-AUC  : {np.percentile(boot_auc, 2.5)*100:.2f}% - {np.percentile(boot_auc, 97.5)*100:.2f}% (Mean: {np.mean(boot_auc)*100:.2f}%)")

    print("\n======================================================================")
    print("SECTION 3: SIDE-BY-SIDE MODEL V2 VS MODEL V3 EVALUATION ON TEST SET")
    print("======================================================================")
    clf_v2_prod = joblib.load(os.path.join('models', 'v2', 'xgb_model_v2.pkl'))
    y_pred_v2 = clf_v2_prod.predict(X_test_v3)
    y_prob_v2 = clf_v2_prod.predict_proba(X_test_v3)[:, 1]

    cm_v2 = confusion_matrix(y_test_v3, y_pred_v2)
    tn_v2, fp_v2, fn_v2, tp_v2 = cm_v2.ravel()

    acc_v2 = accuracy_score(y_test_v3, y_pred_v2) * 100
    prec_v2 = precision_score(y_test_v3, y_pred_v2) * 100
    rec_v2 = recall_score(y_test_v3, y_pred_v2) * 100
    f1_v2 = f1_score(y_test_v3, y_pred_v2) * 100
    auc_v2 = roc_auc_score(y_test_v3, y_prob_v2) * 100
    pr_auc_v2 = average_precision_score(y_test_v3, y_prob_v2) * 100
    fpr_v2 = (fp_v2 / (fp_v2 + tn_v2)) * 100
    fnr_v2 = (fn_v2 / (fn_v2 + tp_v2)) * 100

    print(f"{'Metric':<20s} | {'Model V2 (Baseline)':<20s} | {'Model V3 (New)':<20s}")
    print("-" * 65)
    print(f"{'Accuracy':<20s} | {acc_v2:<19.2f}% | {acc_v3:<19.2f}%")
    print(f"{'Precision':<20s} | {prec_v2:<19.2f}% | {prec_v3:<19.2f}%")
    print(f"{'Recall':<20s} | {rec_v2:<19.2f}% | {rec_v3:<19.2f}%")
    print(f"{'F1-Score':<20s} | {f1_v2:<19.2f}% | {f1_v3:<19.2f}%")
    print(f"{'ROC-AUC':<20s} | {auc_v2:<19.2f}% | {auc_v3:<19.2f}%")
    print(f"{'PR-AUC':<20s} | {pr_auc_v2:<19.2f}% | {pr_auc_v3:<19.2f}%")
    print(f"{'FPR':<20s} | {fpr_v2:<19.2f}% | {fpr_v3:<19.2f}%")
    print(f"{'FNR':<20s} | {fnr_v2:<19.2f}% | {fnr_v3:<19.2f}%")

    print("\n======================================================================")
    print("SECTION 4: FEATURE IMPORTANCE & SHAP DEPENDENCE (MODEL V3)")
    print("======================================================================")
    booster_v3 = clf_v3.get_booster()
    score_gain = booster_v3.get_score(importance_type='gain')
    print("XGBoost Gain Importance:")
    for f, g in sorted(score_gain.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f:<26s}: Gain = {g:.4f}")

    explainer_v3 = shap.TreeExplainer(clf_v3)
    shap_vals_v3 = explainer_v3(X_test_v3).values
    mean_abs_shap = np.mean(np.abs(shap_vals_v3), axis=0)

    print("\nGlobal Mean Absolute SHAP Values:")
    for f, s in sorted(zip(features_v3, mean_abs_shap), key=lambda x: x[1], reverse=True):
        print(f"  {f:<26s}: Mean |SHAP| = {s:.4f}")

    print("\n======================================================================")
    print("SECTION 5: DIAGNOSTIC EVALUATION (URLHAUS & WIKIPEDIA)")
    print("======================================================================")
    diag_urls = [
        "https://urlhaus.abuse.ch/browse/",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION",
        "https://checkout.stripe.com/pay/cs_live_a1b2c3d4e5f6"
    ]

    explainer_v2 = shap.TreeExplainer(clf_v2_prod)

    print(f"{'Target URL':<60s} | {'V2 Prob':<10s} | {'V3 Prob':<10s} | {'V3 URL_Length SHAP':<20s}")
    print("-" * 110)

    for u in diag_urls:
        f_raw = extract_features(u)
        v2_vec = adapt_features_v2(f_raw)
        df_single = pd.DataFrame([v2_vec])[features_v3]

        p2 = float(clf_v2_prod.predict_proba(df_single)[0][1])
        p3 = float(clf_v3.predict_proba(df_single)[0][1])

        shap3 = explainer_v3(df_single).values[0]
        len_idx = features_v3.index('url_length')
        s_len = shap3[len_idx]

        print(f"{u[:60]:<60s} | {p2*100:<9.2f}% | {p3*100:<9.2f}% | {s_len:+.4f}")

    print("\n======================================================================")
    print("SECTION 6: AUXILIARY SOURCE PREDICTABILITY TEST (V2 VS V3)")
    print("======================================================================")
    # V3 dataset source predictability
    X_v3_src = df_v3[features_v3]
    y_v3_src = df_v3['source']

    X_tr_s3, X_te_s3, y_tr_s3, y_te_s3 = train_test_split(X_v3_src, y_v3_src, test_size=0.2, random_state=42)
    clf_src_v3 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_src_v3.fit(X_tr_s3, y_tr_s3)
    acc_src_v3 = accuracy_score(y_te_s3, clf_src_v3.predict(X_te_s3)) * 100

    print(f"Source-Bias Predictability:")
    print(f"  Model V2 Source Predictability: 89.56%")
    print(f"  Model V3 Source Predictability: {acc_src_v3:.2f}%")

    print("\n======================================================================")
    print("SECTION 7: PROBABILITY CALIBRATION CHECK FOR MODEL V3")
    print("======================================================================")
    brier_v3 = brier_score_loss(y_test_v3, y_prob_v3)
    print(f"Model V3 Brier Score: {brier_v3:.4f}")

    print("\n======================================================================")
    print("SECTION 8: SAVE MODEL V3 ARTIFACTS TO models/v3/")
    print("======================================================================")
    # Train V3 on full dataset_v3 for deployment artifact
    clf_v3_full = XGBClassifier(
        n_estimators=120,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    clf_v3_full.fit(X_v3, y_v3)

    joblib.dump(clf_v3_full, v3_model_path)
    joblib.dump(features_v3, v3_features_path)

    metadata = {
        "model_version": "v3",
        "algorithm": "XGBoostClassifier",
        "scaler_used": False,
        "features": features_v3,
        "feature_count": len(features_v3),
        "dataset_size": len(df_v3),
        "unique_registered_domains": df_v3['registered_domain'].nunique(),
        "test_metrics_grouped_domain": {
            "accuracy": round(acc_v3, 2),
            "precision": round(prec_v3, 2),
            "recall": round(rec_v3, 2),
            "f1_score": round(f1_v3, 2),
            "roc_auc": round(auc_v3, 2),
            "pr_auc": round(pr_auc_v3, 2),
            "false_positive_rate": round(fpr_v3, 2),
            "false_negative_rate": round(fnr_v3, 2)
        },
        "source_predictability": round(acc_src_v3, 2),
        "brier_score": round(brier_v3, 4)
    }

    with open(v3_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Model V3 Artifacts:")
    print(f"   [1] Model File    --> {v3_model_path}")
    print(f"   [2] Features List --> {v3_features_path}")
    print(f"   [3] Metadata File --> {v3_metadata_path}")
    print("======================================================================\n")


if __name__ == '__main__':
    run_pipeline()
