"""
Candidate Training & Selection Script for Model V3.1
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
from xgboost import XGBClassifier

from src.feature_extraction import extract_features
from src.predictor import adapt_features_v2
from evaluate_v3_holdouts import generate_hard_legitimate_holdout, generate_hard_malicious_holdout, evaluate_holdout

features_v3 = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]


def eval_model(clf, X_test, y_test, thresh=0.50):
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= thresh).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, preds) * 100
    prec = precision_score(y_test, preds, zero_division=0) * 100
    rec = recall_score(y_test, preds, zero_division=0) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100
    auc = roc_auc_score(y_test, probs) * 100
    pr_auc = average_precision_score(y_test, probs) * 100
    fpr = (fp / (fp + tn)) * 100
    fnr = (fn / (fn + tp)) * 100
    brier = brier_score_loss(y_test, probs)
    
    return {
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
        'ROC-AUC': auc, 'PR-AUC': pr_auc, 'FPR': fpr, 'FNR': fnr,
        'Brier': brier, 'probs': probs, 'preds': preds
    }


def main():
    data_v3_path = os.path.join('data', 'dataset_v3.csv')
    data_v3_1_path = os.path.join('data', 'dataset_v3_1.csv')

    df_v3 = pd.read_csv(data_v3_path)
    df_v3_1 = pd.read_csv(data_v3_1_path)

    # 1. Split V3 dataset
    gss_v3 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx_v3, test_idx_v3 = next(gss_v3.split(df_v3[features_v3], df_v3['label'], df_v3['registered_domain']))
    X_tr_v3, X_te_v3 = df_v3[features_v3].iloc[train_idx_v3], df_v3[features_v3].iloc[test_idx_v3]
    y_tr_v3, y_te_v3 = df_v3['label'].iloc[train_idx_v3], df_v3['label'].iloc[test_idx_v3]

    # 2. Split V3.1 dataset (Same domain-grouped split logic)
    gss_v3_1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx_v3_1, test_idx_v3_1 = next(gss_v3_1.split(df_v3_1[features_v3], df_v3_1['label'], df_v3_1['registered_domain']))
    X_tr_v3_1, X_te_v3_1 = df_v3_1[features_v3].iloc[train_idx_v3_1], df_v3_1[features_v3].iloc[test_idx_v3_1]
    y_tr_v3_1, y_te_v3_1 = df_v3_1['label'].iloc[train_idx_v3_1], df_v3_1['label'].iloc[test_idx_v3_1]

    # Hard holdout sets
    legit_urls = generate_hard_legitimate_holdout()
    phish_urls = generate_hard_malicious_holdout()

    print("======================================================================")
    print("V3.1 CANDIDATE MODEL EVALUATION & COMPARISON")
    print("======================================================================")

    # Baseline V3 Unweighted
    clf_v3_unweighted = XGBClassifier(
        n_estimators=120, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
    )
    clf_v3_unweighted.fit(X_tr_v3, y_tr_v3)
    res_v3_unw = eval_model(clf_v3_unweighted, X_te_v3, y_te_v3, thresh=0.50)

    # Candidate A: V3 dataset + scale_pos_weight
    spw_v3 = (y_tr_v3 == 0).sum() / (y_tr_v3 == 1).sum()
    clf_cand_a = XGBClassifier(
        n_estimators=120, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', scale_pos_weight=spw_v3, random_state=42
    )
    clf_cand_a.fit(X_tr_v3, y_tr_v3)
    res_cand_a = eval_model(clf_cand_a, X_te_v3, y_te_v3, thresh=0.50)

    # Candidate B: V3.1 augmented dataset + scale_pos_weight
    spw_v3_1 = (y_tr_v3_1 == 0).sum() / (y_tr_v3_1 == 1).sum()
    clf_cand_b = XGBClassifier(
        n_estimators=120, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', scale_pos_weight=spw_v3_1, random_state=42
    )
    clf_cand_b.fit(X_tr_v3_1, y_tr_v3_1)
    res_cand_b = eval_model(clf_cand_b, X_te_v3_1, y_te_v3_1, thresh=0.50)
    # Also evaluate Candidate B on original V3 test set for direct comparison
    res_cand_b_on_v3test = eval_model(clf_cand_b, X_te_v3, y_te_v3, thresh=0.50)

    # Candidate C: V3.1 dataset + scale_pos_weight + calibrated threshold / tuned hyperparams
    # Let's test Candidate B with decision threshold 0.35 and 0.40 on V3 test set and V3.1 test set
    res_cand_c_35 = eval_model(clf_cand_b, X_te_v3_1, y_te_v3_1, thresh=0.35)
    res_cand_c_40 = eval_model(clf_cand_b, X_te_v3_1, y_te_v3_1, thresh=0.40)
    res_cand_c_35_v3test = eval_model(clf_cand_b, X_te_v3, y_te_v3, thresh=0.35)

    print("\n--- Model Grouped Test Performance Summary ---")
    rows = [
        ("V3 Unweighted (Baseline)", res_v3_unw),
        ("Candidate A (V3 + SPW)", res_cand_a),
        ("Candidate B (V3.1 + SPW @0.50)", res_cand_b),
        ("Candidate B on V3 Test Set (@0.50)", res_cand_b_on_v3test),
        ("Candidate B on V3 Test Set (@0.35)", res_cand_c_35_v3test),
        ("Candidate B on V3.1 Test Set (@0.35)", res_cand_c_35),
    ]

    for name, r in rows:
        print(f"\n{name}:")
        print(f"  TN={r['TN']}, FP={r['FP']}, FN={r['FN']}, TP={r['TP']}")
        print(f"  Acc: {r['Accuracy']:.2f}% | Prec: {r['Precision']:.2f}% | Rec: {r['Recall']:.2f}% | F1: {r['F1']:.2f}%")
        print(f"  ROC-AUC: {r['ROC-AUC']:.2f}% | PR-AUC: {r['PR-AUC']:.2f}% | FPR: {r['FPR']:.2f}% | FNR: {r['FNR']:.2f}%")

    print("\n======================================================================")
    print("EVALUATION ON UNTOUCHED HARD HOLDOUT SETS (105 LEGIT, 105 PHISH)")
    print("======================================================================")

    models_to_eval = [
        ("V3 Baseline", clf_v3_unweighted),
        ("Candidate A", clf_cand_a),
        ("Candidate B", clf_cand_b),
    ]

    for name, m in models_to_eval:
        probs_leg, preds_leg, fp_leg = evaluate_holdout(m, legit_urls, is_phishing_label=False)
        probs_phish, preds_phish, fn_phish = evaluate_holdout(m, phish_urls, is_phishing_label=True)
        
        fpr_leg = (len(fp_leg) / len(legit_urls)) * 100
        rec_phish = (sum(preds_phish) / len(phish_urls)) * 100

        print(f"\n{name} Holdout Performance:")
        print(f"  Hard Legitimate FP Count: {len(fp_leg)} / {len(legit_urls)} (FPR: {fpr_leg:.2f}%)")
        print(f"  Hard Malicious TP Count : {sum(preds_phish)} / {len(phish_urls)} (Recall: {rec_phish:.2f}%, Missed: {len(fn_phish)})")

if __name__ == '__main__':
    main()
