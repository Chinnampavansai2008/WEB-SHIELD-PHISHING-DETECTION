"""
Final Model Audit, Selection, and Comparison Script for Web Shield
"""

import os
import sys
sys.path.insert(0, '.')

import json
import hashlib
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

from src.feature_extraction import extract_features
from src.predictor import adapt_features_v2
from evaluate_v3_holdouts import generate_hard_legitimate_holdout, generate_hard_malicious_holdout

features_cols = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]


def file_sha256(filepath):
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("======================================================================")
    print("SECTION 1: ARTIFACT SHA256 HASHES (FROZEN MODELS)")
    print("======================================================================")
    
    artifacts = [
        ("V2 Model", "models/v2/xgb_model_v2.pkl"),
        ("V2 Features", "models/v2/features_v2.pkl"),
        ("V2 Metadata", "models/v2/metadata.json"),
        ("V3 Model", "models/v3/xgb_model_v3.pkl"),
        ("V3 Features", "models/v3/features_v3.pkl"),
        ("V3 Metadata", "models/v3/metadata.json"),
        ("V3.1 Model", "models/v3_1/xgb_model_v3_1.pkl"),
        ("V3.1 Features", "models/v3_1/features_v3_1.pkl"),
        ("V3.1 Metadata", "models/v3_1/metadata.json"),
        ("Dataset V2", "data/dataset_v2.csv"),
        ("Dataset V3", "data/dataset_v3.csv"),
        ("Dataset V3.1", "data/dataset_v3_1.csv"),
    ]

    for name, path in artifacts:
        print(f"  {name:<15s} ({path}): {file_sha256(path)}")

    # Load frozen models
    clf_v2 = joblib.load("models/v2/xgb_model_v2.pkl")
    clf_v3 = joblib.load("models/v3/xgb_model_v3.pkl")
    clf_v3_1 = joblib.load("models/v3_1/xgb_model_v3_1.pkl")

    print("\n======================================================================")
    print("SECTION 2 & 3: DEFINING & RECONSTRUCTING FROZEN EVALUATION SETS")
    print("======================================================================")

    # Set A: LEGACY_V2_GROUPED_TEST (from dataset_v2.csv)
    df_v2 = pd.read_csv("data/dataset_v2.csv")
    v2_features_list = joblib.load("models/v2/features_v2.pkl")
    gss_v2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx_v2, te_idx_v2 = next(gss_v2.split(df_v2[v2_features_list], df_v2['label'], df_v2['registered_domain']))
    
    df_v2_test = df_v2.iloc[te_idx_v2].copy()
    v2_test_sha256 = hashlib.sha256(df_v2_test['url'].str.cat(sep='\n').encode()).hexdigest()
    
    print(f"Set A: LEGACY_V2_GROUPED_TEST ({len(df_v2_test)} samples)")
    print(f"  Dataset: dataset_v2.csv ({file_sha256('data/dataset_v2.csv')})")
    print(f"  Legitimate: {(df_v2_test['label']==0).sum()}, Phishing: {(df_v2_test['label']==1).sum()}")
    print(f"  Unique Registered Domains: {df_v2_test['registered_domain'].nunique()}")
    print(f"  Test URLs Hash (SHA256): {v2_test_sha256}")

    # Set B: V3_GROUPED_TEST (from dataset_v3.csv)
    df_v3 = pd.read_csv("data/dataset_v3.csv")
    gss_v3 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx_v3, te_idx_v3 = next(gss_v3.split(df_v3[features_cols], df_v3['label'], df_v3['registered_domain']))
    
    df_v3_test = df_v3.iloc[te_idx_v3].copy()
    v3_test_sha256 = hashlib.sha256(df_v3_test['url'].str.cat(sep='\n').encode()).hexdigest()
    
    print(f"\nSet B: V3_GROUPED_TEST ({len(df_v3_test)} samples)")
    print(f"  Dataset: dataset_v3.csv ({file_sha256('data/dataset_v3.csv')})")
    print(f"  Legitimate: {(df_v3_test['label']==0).sum()}, Phishing: {(df_v3_test['label']==1).sum()}")
    print(f"  Unique Registered Domains: {df_v3_test['registered_domain'].nunique()}")
    print(f"  Test URLs Hash (SHA256): {v3_test_sha256}")

    # Set C: HARD_HOLDOUT
    legit_urls = generate_hard_legitimate_holdout()
    phish_urls = generate_hard_malicious_holdout()
    
    all_hard_urls = legit_urls + phish_urls
    hard_sha256 = hashlib.sha256("\n".join(all_hard_urls).encode()).hexdigest()
    
    print(f"\nSet C: HARD_HOLDOUT ({len(all_hard_urls)} samples)")
    print(f"  Legitimate: {len(legit_urls)}, Phishing: {len(phish_urls)}")
    print(f"  Hard Holdout URLs Hash (SHA256): {hard_sha256}")


if __name__ == '__main__':
    main()
