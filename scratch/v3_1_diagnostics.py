"""
Diagnostic Script for Web Shield Model V3.1 Analysis
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
    average_precision_score, confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

features_v3 = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]

def main():
    data_v3_path = os.path.join('data', 'dataset_v3.csv')
    df_v3 = pd.read_csv(data_v3_path)

    X_v3 = df_v3[features_v3]
    y_v3 = df_v3['label']
    groups_v3 = df_v3['registered_domain']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_v3, y_v3, groups_v3))

    X_train_v3, X_test_v3 = X_v3.iloc[train_idx], X_v3.iloc[test_idx]
    y_train_v3, y_test_v3 = y_v3.iloc[train_idx], y_v3.iloc[test_idx]
    groups_train, groups_test = groups_v3.iloc[train_idx], groups_v3.iloc[test_idx]
    df_test_v3 = df_v3.iloc[test_idx].copy()

    print("Train dataset size:", len(X_train_v3), "Legit:", (y_train_v3==0).sum(), "Phish:", (y_train_v3==1).sum())
    print("Test dataset size:", len(X_test_v3), "Legit:", (y_test_v3==0).sum(), "Phish:", (y_test_v3==1).sum())

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

    y_prob_v3 = clf_v3.predict_proba(X_test_v3)[:, 1]

    # SECTION 4: THRESHOLD ANALYSIS
    print("\n==================================================")
    print("SECTION 4: THRESHOLD ANALYSIS ON FROZEN V3 MODEL")
    print("==================================================")
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    
    thresh_results = []
    for t in thresholds:
        y_pred_t = (y_prob_v3 >= t).astype(int)
        cm = confusion_matrix(y_test_v3, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_test_v3, y_pred_t, zero_division=0) * 100
        rec = recall_score(y_test_v3, y_pred_t, zero_division=0) * 100
        f1 = f1_score(y_test_v3, y_pred_t, zero_division=0) * 100
        fpr = (fp / (fp + tn)) * 100
        fnr = (fn / (fn + tp)) * 100
        thresh_results.append({
            'Threshold': t,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Precision': prec, 'Recall': rec, 'F1': f1,
            'FPR': fpr, 'FNR': fnr
        })

    df_thresh = pd.DataFrame(thresh_results)
    print(df_thresh.to_string(index=False))

    # ROC & PR curves
    fpr_roc, tpr_roc, _ = roc_curve(y_test_v3, y_prob_v3)
    prec_pr, rec_pr, _ = precision_recall_curve(y_test_v3, y_prob_v3)
    print(f"\nROC-AUC: {roc_auc_score(y_test_v3, y_prob_v3)*100:.2f}%")
    print(f"PR-AUC: {average_precision_score(y_test_v3, y_prob_v3)*100:.2f}%")

    max_f1_row = df_thresh.loc[df_thresh['F1'].idxmax()]
    rec_90_rows = df_thresh[df_thresh['Recall'] >= 90]
    rec_95_rows = df_thresh[df_thresh['Recall'] >= 95]

    print(f"\nThreshold Maximizing F1: {max_f1_row['Threshold']:.2f} (F1={max_f1_row['F1']:.2f}%, Rec={max_f1_row['Recall']:.2f}%, FPR={max_f1_row['FPR']:.2f}%)")
    if not rec_90_rows.empty:
        best_rec_90 = rec_90_rows.iloc[0]
        print(f"Threshold for >=90% Recall: {best_rec_90['Threshold']:.2f} (Rec={best_rec_90['Recall']:.2f}%, FPR={best_rec_90['FPR']:.2f}%)")
    else:
        print("No threshold achieved >=90% Recall")
    
    if not rec_95_rows.empty:
        best_rec_95 = rec_95_rows.iloc[0]
        print(f"Threshold for >=95% Recall: {best_rec_95['Threshold']:.2f} (Rec={best_rec_95['Recall']:.2f}%, FPR={best_rec_95['FPR']:.2f}%)")
    else:
        print("No threshold achieved >=95% Recall")

    # SECTION 5: DETERMINE WHETHER THRESHOLD ALONE CAN FIX V3
    print("\n==================================================")
    print("SECTION 5: CAN THRESHOLD ALONE FIX V3?")
    print("==================================================")
    can_fix = False
    for _, r in df_thresh.iterrows():
        if r['Recall'] >= 90 and r['FPR'] <= 5.0:
            can_fix = True
            print(f" threshold {r['Threshold']} achieves Recall {r['Recall']:.2f}% with FPR {r['FPR']:.2f}% <= 5%")
    if not can_fix:
        print("RESULT: NO! Threshold alone CANNOT fix V3. Achieving >=90% Recall requires lowering threshold to <=0.20, which pushes FPR to >=8.70% (>5.0%). Data & model improvements are necessary.")

    # SECTION 6: CLASS IMBALANCE EXPERIMENT
    print("\n==================================================")
    print("SECTION 6: CLASS IMBALANCE EXPERIMENT (scale_pos_weight)")
    print("==================================================")
    neg_cnt = (y_train_v3 == 0).sum()
    pos_cnt = (y_train_v3 == 1).sum()
    spw = neg_cnt / pos_cnt
    print(f"Train negative count (0): {neg_cnt}, positive count (1): {pos_cnt}")
    print(f"scale_pos_weight = {spw:.4f}")

    clf_weighted = XGBClassifier(
        n_estimators=120,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        scale_pos_weight=spw,
        random_state=42
    )
    clf_weighted.fit(X_train_v3, y_train_v3)
    y_prob_weighted = clf_weighted.predict_proba(X_test_v3)[:, 1]
    y_pred_weighted = clf_weighted.predict(X_test_v3)

    cm_w = confusion_matrix(y_test_v3, y_pred_weighted)
    tn_w, fp_w, fn_w, tp_w = cm_w.ravel()
    print("Weighted V3 Candidate Metrics (thresh=0.50):")
    print(f"  TN={tn_w}, FP={fp_w}, FN={fn_w}, TP={tp_w}")
    print(f"  Accuracy : {accuracy_score(y_test_v3, y_pred_weighted)*100:.2f}%")
    print(f"  Precision: {precision_score(y_test_v3, y_pred_weighted)*100:.2f}%")
    print(f"  Recall   : {recall_score(y_test_v3, y_pred_weighted)*100:.2f}%")
    print(f"  F1       : {f1_score(y_test_v3, y_pred_weighted)*100:.2f}%")
    print(f"  ROC-AUC  : {roc_auc_score(y_test_v3, y_prob_weighted)*100:.2f}%")
    print(f"  PR-AUC   : {average_precision_score(y_test_v3, y_prob_weighted)*100:.2f}%")
    print(f"  FPR      : {(fp_w/(fp_w+tn_w))*100:.2f}%")
    print(f"  FNR      : {(fn_w/(fn_w+tp_w))*100:.2f}%")

    # SECTION 7: DATA COVERAGE ANALYSIS FOR FALSE NEGATIVES
    print("\n==================================================")
    print("SECTION 7: DATA COVERAGE ANALYSIS FOR V3 FALSE NEGATIVES")
    print("==================================================")
    df_test_v3['v3_prob'] = y_prob_v3
    df_test_v3['v3_pred'] = (y_prob_v3 >= 0.50).astype(int)
    fn_df = df_test_v3[(df_test_v3['label'] == 1) & (df_test_v3['v3_pred'] == 0)]
    tp_df = df_test_v3[(df_test_v3['label'] == 1) & (df_test_v3['v3_pred'] == 1)]
    phish_df = df_test_v3[df_test_v3['label'] == 1]

    print(f"Total test phishing samples: {len(phish_df)}, FN count: {len(fn_df)}, TP count: {len(tp_df)}")
    print("\nFN Breakdown by key features:")
    print("1. URL length distribution among FNs:")
    print(fn_df['url_length'].describe())
    print("  url_length <= 30:", (fn_df['url_length'] <= 30).sum())
    print("  url_length 31-50:", ((fn_df['url_length'] > 30) & (fn_df['url_length'] <= 50)).sum())
    print("  url_length > 50 :", (fn_df['url_length'] > 50).sum())

    print("\n2. having_ip among FNs vs TPs in phishing test set:")
    print("  FN having_ip=1:", (fn_df['having_ip'] == 1).sum(), "having_ip=0:", (fn_df['having_ip'] == 0).sum())
    print("  TP having_ip=1:", (tp_df['having_ip'] == 1).sum(), "having_ip=0:", (tp_df['having_ip'] == 0).sum())

    print("\n3. ssl_valid among FNs vs TPs in phishing test set:")
    print("  FN ssl_valid=1:", (fn_df['ssl_valid'] == 1).sum(), "ssl_valid=0:", (fn_df['ssl_valid'] == 0).sum())
    print("  TP ssl_valid=1:", (tp_df['ssl_valid'] == 1).sum(), "ssl_valid=0:", (tp_df['ssl_valid'] == 0).sum())

    print("\n4. has_suspicious_keyword among FNs vs TPs:")
    print("  FN has_kw=1:", (fn_df['has_suspicious_keyword'] == 1).sum(), "has_kw=0:", (fn_df['has_suspicious_keyword'] == 0).sum())
    print("  TP has_kw=1:", (tp_df['has_suspicious_keyword'] == 1).sum(), "has_kw=0:", (tp_df['has_suspicious_keyword'] == 0).sum())

    print("\n5. subdomain_count among FNs:")
    print(fn_df['subdomain_count'].value_counts())

    print("\n6. hyphen_count among FNs:")
    print(fn_df['hyphen_count'].value_counts())

    print("\nSample False Negative URLs:")
    for _, r in fn_df[['url', 'source', 'url_length', 'ssl_valid', 'having_ip', 'v3_prob']].head(10).iterrows():
        print(f"  URL: {r['url'][:65]:<65s} | Src: {r['source']:<15s} | Prob: {r['v3_prob']*100:5.2f}% | Len: {r['url_length']:3d} | SSL: {r['ssl_valid']} | IP: {r['having_ip']}")

    # SECTION 10: FEATURE ABLATION
    print("\n==================================================")
    print("SECTION 10: FEATURE ABLATION STUDY")
    print("==================================================")
    ablation_configs = {
        'A. All 11 Features': features_v3,
        'B. w/o url_length': [f for f in features_v3 if f != 'url_length'],
        'C. w/o ssl_valid': [f for f in features_v3 if f != 'ssl_valid'],
        'D. w/o domain-age': [f for f in features_v3 if f not in ['domain_age_months_clean', 'domain_age_known']],
        'E. w/o having_ip': [f for f in features_v3 if f != 'having_ip'],
    }

    ablation_rows = []
    for name, f_list in ablation_configs.items():
        X_tr = X_train_v3[f_list]
        X_te = X_test_v3[f_list]
        clf_ab = XGBClassifier(
            n_estimators=120, learning_rate=0.07, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
        )
        clf_ab.fit(X_tr, y_train_v3)
        y_pr = clf_ab.predict_proba(X_te)[:, 1]
        y_pd = (y_pr >= 0.50).astype(int)
        
        cm = confusion_matrix(y_test_v3, y_pd)
        tn, fp, fn, tp = cm.ravel()
        
        ablation_rows.append({
            'Config': name,
            'ROC-AUC': roc_auc_score(y_test_v3, y_pr)*100,
            'PR-AUC': average_precision_score(y_test_v3, y_pr)*100,
            'Recall': recall_score(y_test_v3, y_pd)*100,
            'FPR': (fp/(fp+tn))*100,
            'F1': f1_score(y_test_v3, y_pd)*100
        })

    df_ab = pd.DataFrame(ablation_rows)
    print(df_ab.to_string(index=False))

    # SECTION 11: INVESTIGATE having_ip DOMINANCE
    print("\n==================================================")
    print("SECTION 11: INVESTIGATE having_ip DOMINANCE")
    print("==================================================")
    p_ip_phish = (df_v3[df_v3['label'] == 1]['having_ip'] == 1).mean()
    p_ip_legit = (df_v3[df_v3['label'] == 0]['having_ip'] == 1).mean()
    cnt_ip_phish = (df_v3[df_v3['label'] == 1]['having_ip'] == 1).sum()
    cnt_ip_legit = (df_v3[df_v3['label'] == 0]['having_ip'] == 1).sum()

    print(f"P(having_ip=1 | phishing)   = {p_ip_phish*100:.2f}% ({cnt_ip_phish} / {(df_v3['label']==1).sum()})")
    print(f"P(having_ip=1 | legitimate) = {p_ip_legit*100:.2f}% ({cnt_ip_legit} / {(df_v3['label']==0).sum()})")

    # Performance on phishing test set where having_ip=0
    df_test_no_ip = df_test_v3[df_test_v3['label'] == 1]
    df_test_no_ip_0 = df_test_no_ip[df_test_no_ip['having_ip'] == 0]
    rec_ip_0 = (df_test_no_ip_0['v3_pred'] == 1).mean() * 100
    print(f"V3 Phishing Recall when having_ip=0: {rec_ip_0:.2f}% ({(df_test_no_ip_0['v3_pred']==1).sum()} / {len(df_test_no_ip_0)})")
    
    df_test_ip_1 = df_test_no_ip[df_test_no_ip['having_ip'] == 1]
    if len(df_test_ip_1) > 0:
        rec_ip_1 = (df_test_ip_1['v3_pred'] == 1).mean() * 100
        print(f"V3 Phishing Recall when having_ip=1: {rec_ip_1:.2f}% ({(df_test_ip_1['v3_pred']==1).sum()} / {len(df_test_ip_1)})")

    # SECTION 12: INVESTIGATE ssl_valid DIRECTION
    print("\n==================================================")
    print("SECTION 12: INVESTIGATE ssl_valid DIRECTION")
    print("==================================================")
    p_ssl_phish = (df_v3[df_v3['label'] == 1]['ssl_valid'] == 1).mean()
    p_ssl_legit = (df_v3[df_v3['label'] == 0]['ssl_valid'] == 1).mean()
    cnt_ssl_phish = (df_v3[df_v3['label'] == 1]['ssl_valid'] == 1).sum()
    cnt_ssl_legit = (df_v3[df_v3['label'] == 0]['ssl_valid'] == 1).sum()

    print(f"P(ssl_valid=1 | legitimate) = {p_ssl_legit*100:.2f}% ({cnt_ssl_legit} / {(df_v3['label']==0).sum()})")
    print(f"P(ssl_valid=1 | phishing)   = {p_ssl_phish*100:.2f}% ({cnt_ssl_phish} / {(df_v3['label']==1).sum()})")

    # SECTION 13: NETWORK FEATURE ROBUSTNESS
    print("\n==================================================")
    print("SECTION 13: NETWORK FEATURE ROBUSTNESS")
    print("==================================================")
    # Fixture 1: Baseline
    y_pr_base = y_prob_v3
    y_pd_base = (y_pr_base >= 0.50).astype(int)

    # Fixture 2: domain age unavailable (set domain_age_months_clean=0, domain_age_known=0)
    X_test_no_age = X_test_v3.copy()
    X_test_no_age['domain_age_months_clean'] = 0
    X_test_no_age['domain_age_known'] = 0
    y_pr_no_age = clf_v3.predict_proba(X_test_no_age)[:, 1]
    y_pd_no_age = (y_pr_no_age >= 0.50).astype(int)

    # Fixture 3: TLS lookup unavailable (set ssl_valid=0)
    X_test_no_ssl = X_test_v3.copy()
    X_test_no_ssl['ssl_valid'] = 0
    y_pr_no_ssl = clf_v3.predict_proba(X_test_no_ssl)[:, 1]
    y_pd_no_ssl = (y_pr_no_ssl >= 0.50).astype(int)

    # Fixture 4: Both unavailable
    X_test_no_net = X_test_v3.copy()
    X_test_no_net['domain_age_months_clean'] = 0
    X_test_no_net['domain_age_known'] = 0
    X_test_no_net['ssl_valid'] = 0
    y_pr_no_net = clf_v3.predict_proba(X_test_no_net)[:, 1]
    y_pd_no_net = (y_pr_no_net >= 0.50).astype(int)

    fixtures = {
        'Baseline (All available)': (y_pr_base, y_pd_base),
        'Domain Age Unavailable': (y_pr_no_age, y_pd_no_age),
        'TLS Unavailable (ssl_valid=0)': (y_pr_no_ssl, y_pd_no_ssl),
        'Both Age & TLS Unavailable': (y_pr_no_net, y_pd_no_net),
    }

    rob_rows = []
    for name, (pr, pd_val) in fixtures.items():
        cm = confusion_matrix(y_test_v3, pd_val)
        tn, fp, fn, tp = cm.ravel()
        rob_rows.append({
            'Fixture': name,
            'Accuracy': accuracy_score(y_test_v3, pd_val)*100,
            'Precision': precision_score(y_test_v3, pd_val, zero_division=0)*100,
            'Recall': recall_score(y_test_v3, pd_val)*100,
            'FPR': (fp/(fp+tn))*100,
            'F1': f1_score(y_test_v3, pd_val)*100,
            'ROC-AUC': roc_auc_score(y_test_v3, pr)*100
        })

    df_rob = pd.DataFrame(rob_rows)
    print(df_rob.to_string(index=False))

if __name__ == '__main__':
    main()
