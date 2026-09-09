"""
Check Target Tradeoff Script across all Candidates & Thresholds
"""

import os
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

features_v3 = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]

def main():
    data_v3 = pd.read_csv(os.path.join('data', 'dataset_v3.csv'))
    data_v3_1 = pd.read_csv(os.path.join('data', 'dataset_v3_1.csv'))

    gss_v3 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx_v3, te_idx_v3 = next(gss_v3.split(data_v3[features_v3], data_v3['label'], data_v3['registered_domain']))
    X_tr_v3, X_te_v3 = data_v3[features_v3].iloc[tr_idx_v3], data_v3[features_v3].iloc[te_idx_v3]
    y_tr_v3, y_te_v3 = data_v3['label'].iloc[tr_idx_v3], data_v3['label'].iloc[te_idx_v3]

    gss_v3_1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx_v3_1, te_idx_v3_1 = next(gss_v3_1.split(data_v3_1[features_v3], data_v3_1['label'], data_v3_1['registered_domain']))
    X_tr_v3_1, X_te_v3_1 = data_v3_1[features_v3].iloc[tr_idx_v3_1], data_v3_1[features_v3].iloc[te_idx_v3_1]
    y_tr_v3_1, y_te_v3_1 = data_v3_1['label'].iloc[tr_idx_v3_1], data_v3_1['label'].iloc[te_idx_v3_1]

    models = {
        'V3 Baseline (Unweighted)': XGBClassifier(n_estimators=120, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42).fit(X_tr_v3, y_tr_v3),
        'Candidate A (V3 + SPW=3.43)': XGBClassifier(n_estimators=120, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', scale_pos_weight=3.43, random_state=42).fit(X_tr_v3, y_tr_v3),
        'Candidate B (V3.1 + SPW=2.90)': XGBClassifier(n_estimators=120, learning_rate=0.07, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', scale_pos_weight=2.90, random_state=42).fit(X_tr_v3_1, y_tr_v3_1),
        'Candidate B2 (V3.1 + SPW=4.0)': XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', scale_pos_weight=4.0, random_state=42).fit(X_tr_v3_1, y_tr_v3_1),
    }

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    for m_name, model in models.items():
        print(f"\n==================================================")
        print(f"MODEL: {m_name}")
        print(f"==================================================")
        # Evaluate on V3 test set
        p_v3 = model.predict_proba(X_te_v3)[:, 1]
        print("--- Evaluated on V3 Grouped Test Set (345 Legit, 99 Phish) ---")
        for t in thresholds:
            preds = (p_v3 >= t).astype(int)
            cm = confusion_matrix(y_te_v3, preds)
            tn, fp, fn, tp = cm.ravel()
            rec = recall_score(y_te_v3, preds)*100
            fpr = (fp/(fp+tn))*100
            f1 = f1_score(y_te_v3, preds)*100
            flag = " ** TARGET HIT (Rec>=90%, FPR<=5%) **" if rec>=90 and fpr<=5.0 else ""
            print(f"  Thresh {t:.2f}: Rec={rec:5.2f}%, FPR={fpr:5.2f}%, F1={f1:5.2f}% {flag}")

        # Evaluate on V3.1 test set
        p_v3_1 = model.predict_proba(X_te_v3_1)[:, 1]
        print("--- Evaluated on V3.1 Grouped Test Set (367 Legit, 135 Phish) ---")
        for t in thresholds:
            preds = (p_v3_1 >= t).astype(int)
            cm = confusion_matrix(y_te_v3_1, preds)
            tn, fp, fn, tp = cm.ravel()
            rec = recall_score(y_te_v3_1, preds)*100
            fpr = (fp/(fp+tn))*100
            f1 = f1_score(y_te_v3_1, preds)*100
            flag = " ** TARGET HIT (Rec>=90%, FPR<=5%) **" if rec>=90 and fpr<=5.0 else ""
            print(f"  Thresh {t:.2f}: Rec={rec:5.2f}%, FPR={fpr:5.2f}%, F1={f1:5.2f}% {flag}")

if __name__ == '__main__':
    main()
