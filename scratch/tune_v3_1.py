"""
Grid & Feature Tuning Script for Model V3.1 Candidate Selection
"""

import os
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

from evaluate_v3_holdouts import generate_hard_legitimate_holdout, generate_hard_malicious_holdout, evaluate_holdout

features_v3 = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]

def main():
    data_v3_1_path = os.path.join('data', 'dataset_v3_1.csv')
    df = pd.read_csv(data_v3_1_path)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df[features_v3], df['label'], df['registered_domain']))

    X_tr, X_te = df[features_v3].iloc[train_idx], df[features_v3].iloc[test_idx]
    y_tr, y_te = df['label'].iloc[train_idx], df['label'].iloc[test_idx]

    legit_urls = generate_hard_legitimate_holdout()
    phish_urls = generate_hard_malicious_holdout()

    spw_default = (y_tr == 0).sum() / (y_tr == 1).sum()

    print(f"Dataset V3.1 Train size: {len(X_tr)} (Legit: {(y_tr==0).sum()}, Phish: {(y_tr==1).sum()})")
    print(f"Dataset V3.1 Test size:  {len(X_te)} (Legit: {(y_te==0).sum()}, Phish: {(y_te==1).sum()})")
    print(f"Base scale_pos_weight: {spw_default:.4f}\n")

    param_grid = [
        {'n_estimators': 120, 'max_depth': 3, 'learning_rate': 0.05, 'spw_mult': 1.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.05, 'spw_mult': 1.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.07, 'spw_mult': 1.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.07, 'spw_mult': 1.5, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.07, 'spw_mult': 2.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'spw_mult': 1.5, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'spw_mult': 2.5, 'subsample': 0.8, 'colsample_bytree': 0.8},
    ]

    for i, p in enumerate(param_grid):
        spw = spw_default * p['spw_mult']
        clf = XGBClassifier(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            scale_pos_weight=spw,
            eval_metric='logloss',
            random_state=42
        )
        clf.fit(X_tr, y_tr)

        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.50).astype(int)

        cm = confusion_matrix(y_te, preds)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_te, preds) * 100
        prec = precision_score(y_te, preds, zero_division=0) * 100
        rec = recall_score(y_te, preds, zero_division=0) * 100
        f1 = f1_score(y_te, preds, zero_division=0) * 100
        auc = roc_auc_score(y_te, probs) * 100
        pr_auc = average_precision_score(y_te, probs) * 100
        fpr = (fp / (fp + tn)) * 100

        # Evaluate on holdouts
        probs_leg, preds_leg, fp_leg = evaluate_holdout(clf, legit_urls, is_phishing_label=False)
        probs_phish, preds_phish, fn_phish = evaluate_holdout(clf, phish_urls, is_phishing_label=True)
        
        h_fpr = (len(fp_leg) / len(legit_urls)) * 100
        h_rec = (sum(preds_phish) / len(phish_urls)) * 100

        print(f"Config {i+1}: depth={p['max_depth']}, lr={p['learning_rate']}, n_est={p['n_estimators']}, spw={spw:.2f}")
        print(f"  Test: TN={tn}, FP={fp}, FN={fn}, TP={tp} | Acc={acc:.2f}%, Prec={prec:.2f}%, Rec={rec:.2f}%, F1={f1:.2f}%, FPR={fpr:.2f}%, AUC={auc:.2f}%")
        print(f"  Hard Holdout: Hard Leg FP={len(fp_leg)}/105 (FPR={h_fpr:.2f}%), Hard Mal Rec={sum(preds_phish)}/93 ({h_rec:.2f}%)\n")

if __name__ == '__main__':
    main()
