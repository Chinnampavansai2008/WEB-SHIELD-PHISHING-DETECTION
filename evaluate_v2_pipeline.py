"""
Evaluation & Calibration Pipeline for Model V2 Integration.
Evaluates Model V2 hard malicious performance, Brier score calibration,
and shadow comparison on sanity test cases.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    brier_score_loss, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit

from src.predictor import get_predictor
from src.shadow_comparator import run_shadow_comparison


def run_hard_malicious_evaluation():
    """Runs hard malicious evaluation on diverse phishing test vectors."""
    print("======================================================================")
    print("SECTION 1: HARD MALICIOUS EVALUATION FOR MODEL V2")
    print("======================================================================")

    data_v2_path = os.path.join('data', 'dataset_v2.csv')
    df_v2 = pd.read_csv(data_v2_path)
    
    # Grouped split test set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(gss.split(df_v2, df_v2['label'], df_v2['registered_domain']))
    df_test = df_v2.iloc[test_idx]
    
    # Filter phishing test samples
    df_phish = df_test[df_test['label'] == 1].copy()

    predictor = get_predictor(force_version="v2")
    
    y_true = []
    y_pred = []
    y_prob = []

    for _, row in df_phish.iterrows():
        raw_feat = {
            'url_length': row['url_length'],
            'having_ip': row['having_ip'],
            'has_at_symbol': row['has_at_symbol'],
            'redirect_count': row['redirect_count'],
            'subdomain_count': row['subdomain_count'],
            'hyphen_count': row['hyphen_count'],
            'domain_entropy': row['domain_entropy'],
            'has_suspicious_keyword': row['has_suspicious_keyword'],
            'ssl_valid': row['ssl_valid'],
            'domain_age_months': -1 if row['domain_age_known'] == 0 else row['domain_age_months_clean']
        }

        prob, _, _ = predictor.predict_raw(raw_feat)
        pred = 1 if prob >= 0.50 else 0

        y_true.append(1)
        y_pred.append(pred)
        y_prob.append(prob)

    rec = recall_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=1) * 100
    f1 = f1_score(y_true, y_pred) * 100
    pr_auc = average_precision_score(y_true, y_prob) * 100
    fnr = (1.0 - (rec / 100.0)) * 100
    fn_count = sum(1 for p in y_pred if p == 0)

    print(f"Total Hard Malicious Test Samples: {len(df_phish)}")
    print(f"  Recall               : {rec:.2f}%")
    print(f"  False Negative Rate  : {fnr:.2f}% ({fn_count}/{len(df_phish)})")
    print(f"  Precision            : {prec:.2f}%")
    print(f"  F1-Score             : {f1:.2f}%")
    print(f"  PR-AUC               : {pr_auc:.2f}%")

    return {
        "recall": round(rec, 2),
        "fnr": round(fnr, 2),
        "precision": round(prec, 2),
        "f1": round(f1, 2),
        "pr_auc": round(pr_auc, 2),
        "fn_count": fn_count,
        "total": len(df_phish)
    }


def run_calibration_check():
    """Computes Brier score and reliability calibration curve bins for Model V2."""
    print("\n======================================================================")
    print("SECTION 2: PROBABILITY CALIBRATION CHECK FOR MODEL V2")
    print("======================================================================")

    data_v2_path = os.path.join('data', 'dataset_v2.csv')
    df_v2 = pd.read_csv(data_v2_path)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(gss.split(df_v2, df_v2['label'], df_v2['registered_domain']))
    df_test = df_v2.iloc[test_idx]

    predictor = get_predictor(force_version="v2")

    y_true = []
    y_prob = []

    for _, row in df_test.iterrows():
        raw_feat = {
            'url_length': row['url_length'],
            'having_ip': row['having_ip'],
            'has_at_symbol': row['has_at_symbol'],
            'redirect_count': row['redirect_count'],
            'subdomain_count': row['subdomain_count'],
            'hyphen_count': row['hyphen_count'],
            'domain_entropy': row['domain_entropy'],
            'has_suspicious_keyword': row['has_suspicious_keyword'],
            'ssl_valid': row['ssl_valid'],
            'domain_age_months': -1 if row['domain_age_known'] == 0 else row['domain_age_months_clean']
        }

        prob, _, _ = predictor.predict_raw(raw_feat)
        y_true.append(row['label'])
        y_prob.append(prob)

    brier = brier_score_loss(y_true, y_prob)

    print(f"Model V2 Brier Score: {brier:.4f} (Lower is better, 0.0 = perfect calibration)")
    print("\nReliability / Calibration Curve Bins (5 Probability Bins):")
    print(f"{'Bin Range':<18s} | {'Sample Count':<12s} | {'Mean Pred Prob':<16s} | {'Actual Positive Rate':<20s}")
    print("-" * 72)

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    y_t = np.array(y_true)
    y_p = np.array(y_prob)

    for low, high in bins:
        mask = (y_p >= low) & (y_p < high) if high < 1.0 else (y_p >= low) & (y_p <= high)
        cnt = np.sum(mask)
        if cnt > 0:
            mean_prob = np.mean(y_p[mask])
            act_pos = np.mean(y_t[mask])
            print(f"{low*100:4.1f}% - {high*100:4.1f}%    | {cnt:<12d} | {mean_prob*100:<15.2f}% | {act_pos*100:<19.2f}%")
        else:
            print(f"{low*100:4.1f}% - {high*100:4.1f}%    | 0            | N/A              | N/A")

    return {
        "brier_score": round(brier, 4)
    }


def run_sanity_cases_shadow():
    """Runs shadow comparison across representative sanity cases."""
    print("\n======================================================================")
    print("SECTION 3: REPRESENTATIVE SANITY CASES SHADOW COMPARISON (V1 vs V2)")
    print("======================================================================")

    urls = [
        "https://www.google.com",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION",
        "https://aws.amazon.com/console/billing/reports/latest/summary",
        "https://www.google.com/search?q=machine+learning+phishing+detection",
        "https://accounts.google.com/o/oauth2/auth?response_type=code",
        "https://urlhaus.abuse.ch/browse/",
        "http://paypa1.com/login",
        "http://192.168.1.1/auth/login.php"
    ]

    print(f"{'URL':<55s} | {'V1 Prob':<10s} | {'V1 Verdict':<12s} | {'V2 Prob':<10s} | {'V2 Verdict':<12s} | {'Delta':<8s}")
    print("-" * 115)

    for u in urls:
        res = run_shadow_comparison(u)
        print(f"{res['url'][:55]:<55s} | {res['v1_probability']*100:<9.2f}% | {res['v1_verdict']:<12s} | {res['v2_probability']*100:<9.2f}% | {res['v2_verdict']:<12s} | {res['probability_delta']:+7.4f}")


if __name__ == '__main__':
    run_hard_malicious_evaluation()
    run_calibration_check()
    run_sanity_cases_shadow()
