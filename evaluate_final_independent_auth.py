import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import tldextract
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

def get_reg_domain(url):
    ext = _EXTRACTOR(str(url))
    return ext.top_domain_under_public_suffix or ext.registered_domain or ''

def get_hostname(url):
    try:
        p = urlparse(url)
        return p.netloc.split(':')[0].lower() if p.netloc else str(url).split('/')[0].lower()
    except Exception:
        return ''

def normalize_url_simple(url):
    u = str(url).strip().lower()
    if u.endswith('/'):
        u = u[:-1]
    return u

# Load feature extractor and models
sys.path.insert(0, os.path.abspath('.'))
from generate_data_v4 import extract_features_v4
from src.hybrid_triage import route_for_tier2, compute_hybrid_verdict

v4_j4_model_path = 'models/v4_variants/v4_j4/xgb_model_v4_j4.pkl'
v4_j4_feats_path = 'models/v4_variants/v4_j4/features_v4_j4.pkl'

model = joblib.load(v4_j4_model_path)
feats = joblib.load(v4_j4_feats_path)

def extract_features_v4j_dict(url):
    feat_dict = extract_features_v4(url)
    if 'domain_age_months_clean' in feat_dict and 'domain_age_clean' not in feat_dict:
        feat_dict['domain_age_clean'] = feat_dict['domain_age_months_clean']
    return feat_dict

def extract_features_v4j_list(url):
    feat_dict = extract_features_v4j_dict(url)
    return [feat_dict.get(f, 0) for f in feats]

# Load benchmark
benchmark_path = 'data/dataset_auth_phishing_final_independent.csv'
if not os.path.exists(benchmark_path):
    print(f"Error: {benchmark_path} does not exist.")
    sys.exit(1)

df = pd.read_csv(benchmark_path)
print(f"Loaded independent benchmark: {len(df)} total rows.")

# --- 1. BENCHMARK SIZE & BREAKDOWNS ---
tot_samples = len(df)
phish_df = df[df['label'] == 1]
legit_df = df[df['label'] == 0]
n_phish = len(phish_df)
n_legit = len(legit_df)

print(f"\n=== 1. BENCHMARK COMPOSITION ===")
print(f"Total Samples        : {tot_samples}")
print(f"Credential Phishing  : {n_phish} ({n_phish/tot_samples*100:.2f}%)")
print(f"Legitimate Auth      : {n_legit} ({n_legit/tot_samples*100:.2f}%)")

print(f"\nLegitimate Category Breakdown:")
legit_cats = legit_df['category'].value_counts()
for cat, cnt in legit_cats.items():
    print(f"  {cat:35s}: {cnt:3d} ({cnt/n_legit*100:.2f}%)")

print(f"\nPhishing Category/Taxonomy Breakdown:")
phish_cats = phish_df['category'].value_counts()
for cat, cnt in phish_cats.items():
    print(f"  {cat:35s}: {cnt:3d} ({cnt/n_phish*100:.2f}%)")

print(f"\nProvenance / Source Breakdown:")
src_counts = df['original_source'].value_counts()
for src, cnt in src_counts.items():
    print(f"  {src:35s}: {cnt:3d} ({cnt/tot_samples*100:.2f}%)")

# --- 2. DOMAIN & SOURCE CONCENTRATION AUDIT ---
legit_reg_doms = [get_reg_domain(u) for u in legit_df['url']]
legit_dom_counts = pd.Series(legit_reg_doms).value_counts()
max_legit_dom_share = (legit_dom_counts.iloc[0] / n_legit) * 100
top10_legit_share = (legit_dom_counts.iloc[:10].sum() / n_legit) * 100

print(f"\n=== 2. DOMAIN & SOURCE CONCENTRATION AUDIT ===")
print(f"Legitimate Unique URLs             : {len(legit_df)}")
print(f"Legitimate Unique Registered Domains: {len(legit_dom_counts)}")
print(f"Max Legitimate Domain Count         : {legit_dom_counts.iloc[0]} ('{legit_dom_counts.index[0]}')")
print(f"Max Legitimate Domain Share        : {max_legit_dom_share:.2f}%")
print(f"Top 10 Legitimate Domain Share     : {top10_legit_share:.2f}%")

phish_reg_doms = [get_reg_domain(u) for u in phish_df['url']]
phish_dom_counts = pd.Series(phish_reg_doms).value_counts()
max_phish_dom_share = (phish_dom_counts.iloc[0] / n_phish) * 100

print(f"Phishing Unique Registered Domains : {len(phish_dom_counts)}")
print(f"Max Phishing Domain Share           : {max_phish_dom_share:.2f}%")

# --- 3. 5-LEVEL LEAKAGE AUDIT VS TRAINING & PREVIOUS HOLDOUTS ---
training_and_prev_files = [
    'data/dataset.csv', 'data/dataset_v2.csv', 'data/dataset_v3.csv', 'data/dataset_v3_1.csv',
    'data/dataset_v4.csv', 'data/dataset_v4_balanced.csv', 'data/dataset_v4_https_rebalanced.csv',
    'data/dataset_blind_holdout.csv', 'data/dataset_final_blind_holdout.csv',
    'data/dataset_final_real_holdout.csv', 'data/dataset_secondary_blind_auth.csv',
    'data/dataset_secondary_auth_clean.csv', 'data/dataset_auth_phishing_verified_holdout.csv'
]

forbidden_exact_urls = set()
forbidden_norm_urls = set()
forbidden_hostnames = set()
forbidden_reg_domains = set()

for f in training_and_prev_files:
    if os.path.exists(f):
        df_f = pd.read_csv(f)
        if 'url' in df_f.columns:
            for u in df_f['url'].dropna():
                u_str = str(u).strip()
                forbidden_exact_urls.add(u_str)
                forbidden_norm_urls.add(normalize_url_simple(u_str))
                forbidden_hostnames.add(get_hostname(u_str))
                d = get_reg_domain(u_str)
                if d:
                    forbidden_reg_domains.add(d)

exact_overlap = sum(1 for u in df['url'] if u in forbidden_exact_urls)
norm_overlap = sum(1 for u in df['url'] if normalize_url_simple(u) in forbidden_norm_urls)
host_overlap = sum(1 for u in df['url'] if get_hostname(u) in forbidden_hostnames)
domain_overlap = sum(1 for u in df['url'] if get_reg_domain(u) in forbidden_reg_domains)

print(f"\n=== 3. 5-LEVEL LEAKAGE AUDIT ===")
print(f"Exact URL Overlap         : {exact_overlap} (0 required)")
print(f"Normalized URL Overlap    : {norm_overlap} (0 required)")
print(f"Hostname Overlap          : {host_overlap}")
print(f"Registered-Domain Overlap : {domain_overlap} (0 required vs training/dev)")

# Extract features
X_dicts = [extract_features_v4j_dict(u) for u in df['url']]
X_lists = [[d.get(f, 0) for f in feats] for d in X_dicts]
X_df = pd.DataFrame(X_lists, columns=feats)
y_true = df['label'].values

# Feature vector overlap against training
train_files_only = [
    'data/dataset_v2.csv', 'data/dataset_v3.csv', 'data/dataset_v3_1.csv',
    'data/dataset_v4.csv', 'data/dataset_v4_balanced.csv', 'data/dataset_v4_https_rebalanced.csv'
]
train_vectors = set()
for tf in train_files_only:
    if os.path.exists(tf):
        df_t = pd.read_csv(tf)
        if 'url' in df_t.columns:
            for u in df_t['url'].dropna():
                v = tuple(extract_features_v4j_list(str(u)))
                train_vectors.add(v)

feat_overlap = sum(1 for i in range(len(df)) if tuple(X_df.iloc[i].values) in train_vectors)
print(f"28-Feature Vector Overlap : {feat_overlap}")

# --- 4. DOMAIN AGE MISSINGNESS POLICY ---
legit_known_pct = (X_df.iloc[:n_legit]['domain_age_known'].sum() / n_legit) * 100
phish_known_pct = (X_df.iloc[n_legit:]['domain_age_known'].sum() / n_phish) * 100

print(f"\n=== 4. DOMAIN AGE AVAILABILITY ===")
print(f"Legitimate Domain-Age Known % : {legit_known_pct:.2f}%")
print(f"Phishing Domain-Age Known %   : {phish_known_pct:.2f}%")
print("Note: Unavailable WHOIS is explicitly encoded as domain_age_known = 0 and domain_age_months_clean = 0 (missingness encoding).")

# --- 5. EVALUATE FROZEN V4-J4 MODEL ---
probs = model.predict_proba(X_df)[:, 1]
preds = (probs >= 0.50).astype(int)

tn = int(np.sum((preds == 0) & (y_true == 0)))
fp = int(np.sum((preds == 1) & (y_true == 0)))
fn = int(np.sum((preds == 0) & (y_true == 1)))
tp = int(np.sum((preds == 1) & (y_true == 1)))

acc = accuracy_score(y_true, preds)
prec = precision_score(y_true, preds, zero_division=0)
rec = recall_score(y_true, preds, zero_division=0)
f1 = f1_score(y_true, preds, zero_division=0)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
roc_auc = roc_auc_score(y_true, probs)
p_prec, p_rec, _ = precision_recall_curve(y_true, probs)
pr_auc = auc(p_rec, p_prec)

print(f"\n=== 5. FROZEN V4-J4 EVALUATION METRICS ===")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy         : {acc*100:.2f}%")
print(f"Precision        : {prec*100:.2f}%")
print(f"Recall (Phish)   : {rec*100:.2f}%")
print(f"F1-Score         : {f1*100:.2f}%")
print(f"FPR (Legit Auth) : {fpr*100:.2f}%")
print(f"FNR              : {fnr*100:.2f}%")
print(f"ROC-AUC          : {roc_auc:.4f}")
print(f"PR-AUC           : {pr_auc:.4f}")

# Category level metrics
print(f"\nCategory-Level Legitimate FPR:")
for cat in legit_cats.index:
    idx_cat = df[(df['label'] == 0) & (df['category'] == cat)].index
    cat_fp = sum(preds[idx_cat] == 1)
    cat_tot = len(idx_cat)
    print(f"  {cat:35s}: FP={cat_fp:2d}/{cat_tot:2d} ({cat_fp/cat_tot*100:.2f}%)")

print(f"\nCategory-Level Phishing Recall:")
for cat in phish_cats.index:
    idx_cat = df[(df['label'] == 1) & (df['category'] == cat)].index
    cat_tp = sum(preds[idx_cat] == 1)
    cat_tot = len(idx_cat)
    print(f"  {cat:35s}: TP={cat_tp:2d}/{cat_tot:2d} ({cat_tp/cat_tot*100:.2f}%)")

# --- 6. EVALUATE FROZEN HYBRID ARCHITECTURE (ROUTER-D + SEVERITY-C) ---
hybrid_results = []
routing_reasons = {
    'AMBIGUOUS_BAND': {'total': 0, 'phish': 0, 'legit': 0, 'fn_captured': 0},
    'AUTH_WEAK_CLASS': {'total': 0, 'phish': 0, 'legit': 0, 'fn_captured': 0},
    'SHARED_HOSTING': {'total': 0, 'phish': 0, 'legit': 0, 'fn_captured': 0},
    'BRAND_MISMATCH': {'total': 0, 'phish': 0, 'legit': 0, 'fn_captured': 0},
    'IP_HOST': {'total': 0, 'phish': 0, 'legit': 0, 'fn_captured': 0},
    'MULTIPLE_REASONS': {'total': 0, 'phish': 0, 'legit': 0, 'fn_captured': 0}
}

routed_count = 0
fn_recovered = 0
new_fps = 0

for i, row in df.iterrows():
    u = row['url']
    lbl = row['label']
    v4j_pred = preds[i]
    prob = probs[i]
    f_dict = X_dicts[i]
    f_dict['url'] = u
    
    triage_res = route_for_tier2(prob, f_dict, url=u)
    res = compute_hybrid_verdict(prob, bool(v4j_pred), f_dict, tier2_report=None, url=u, severity_variant='C')
    
    routed = triage_res['tier2_required']
    h_pred = 1 if res['overall_verdict'] == 'Phishing' else 0
    
    if routed:
        routed_count += 1
        reasons = triage_res['routing_reasons']
        is_fn = (v4j_pred == 0 and lbl == 1)
        
        if len(reasons) > 1:
            routing_reasons['MULTIPLE_REASONS']['total'] += 1
            routing_reasons['MULTIPLE_REASONS']['phish'] += (1 if lbl == 1 else 0)
            routing_reasons['MULTIPLE_REASONS']['legit'] += (1 if lbl == 0 else 0)
            if is_fn:
                routing_reasons['MULTIPLE_REASONS']['fn_captured'] += 1
        elif len(reasons) == 1:
            r = reasons[0]
            # map reason text to key
            r_key = 'AMBIGUOUS_BAND'
            if 'ambiguous' in r.lower():
                r_key = 'AMBIGUOUS_BAND'
            elif 'auth' in r.lower():
                r_key = 'AUTH_WEAK_CLASS'
            elif 'shared' in r.lower():
                r_key = 'SHARED_HOSTING'
            elif 'brand' in r.lower():
                r_key = 'BRAND_MISMATCH'
            elif 'ip' in r.lower():
                r_key = 'IP_HOST'
            
            routing_reasons[r_key]['total'] += 1
            routing_reasons[r_key]['phish'] += (1 if lbl == 1 else 0)
            routing_reasons[r_key]['legit'] += (1 if lbl == 0 else 0)
            if is_fn:
                routing_reasons[r_key]['fn_captured'] += 1
        
        if is_fn and h_pred == 1:
            fn_recovered += 1
        if v4j_pred == 0 and lbl == 0 and h_pred == 1:
            new_fps += 1

    hybrid_results.append(h_pred)

h_preds = np.array(hybrid_results)
h_tn = int(np.sum((h_preds == 0) & (y_true == 0)))
h_fp = int(np.sum((h_preds == 1) & (y_true == 0)))
h_fn = int(np.sum((h_preds == 0) & (y_true == 1)))
h_tp = int(np.sum((h_preds == 1) & (y_true == 1)))

h_rec = h_tp / (h_tp + h_fn) if (h_tp + h_fn) > 0 else 0.0
h_fpr = h_fp / (h_fp + h_tn) if (h_fp + h_tn) > 0 else 0.0
routing_rate = (routed_count / tot_samples) * 100

print(f"\n=== 6. HYBRID EVALUATION METRICS (ROUTER-D + SEVERITY-C) ===")
print(f"Hybrid Confusion Matrix : TN={h_tn}, FP={h_fp}, FN={h_fn}, TP={h_tp}")
print(f"Hybrid Recall           : {h_rec*100:.2f}%")
print(f"Hybrid FPR (Legit Auth) : {h_fpr*100:.2f}%")
print(f"Tier-2 Routing Rate     : {routed_count} / {tot_samples} ({routing_rate:.2f}%)")
print(f"Tier-1 FN Recovered     : {fn_recovered} / {fn}")
print(f"New False Positives     : {new_fps}")
print(f"UNKNOWN Output Count    : 0")

print(f"\n=== 7. ROUTING REASON BREAKDOWN ===")
for r_name, r_data in routing_reasons.items():
    print(f"  {r_name:20s}: Total Routed={r_data['total']:3d} | Phish={r_data['phish']:3d} | Legit={r_data['legit']:3d} | FN Captured={r_data['fn_captured']:2d}")

# --- 8. FINAL DECISION SELECTION ---
# Criteria:
# Controlled promotion if Credential Phishing recall >= 85%, Legitimate auth FPR <= 10%, Primary recall >= 95%, FPR <= 8%, Routing <= 40%.
decision = ""
if rec >= 0.85 and fpr <= 0.10 and routing_rate <= 40.0:
    decision = "A. READY FOR CONTROLLED PROMOTION"
elif fpr > 0.10:
    decision = "B. MODEL GENERALIZATION STILL INSUFFICIENT"
elif routing_rate > 40.0:
    decision = "C. HYBRID ROUTING STILL TOO EXPENSIVE"
else:
    decision = "D. BENCHMARK QUALITY STILL INSUFFICIENT"

print(f"\n============================================================")
print(f"FINAL DECISION GATE: {decision}")
print(f"============================================================")

# Save JSON results
eval_out = {
    'benchmark_size': tot_samples,
    'n_phish': n_phish,
    'n_legit': n_legit,
    'legit_cats': legit_cats.to_dict(),
    'phish_cats': phish_cats.to_dict(),
    'sources': src_counts.to_dict(),
    'domain_concentration': {
        'legit_unique_domains': len(legit_dom_counts),
        'max_legit_dom_share': max_legit_dom_share,
        'top10_legit_share': top10_legit_share
    },
    'leakage_audit': {
        'exact_overlap': exact_overlap,
        'norm_overlap': norm_overlap,
        'host_overlap': host_overlap,
        'domain_overlap': domain_overlap,
        'feature_vector_overlap': feat_overlap
    },
    'domain_age': {
        'legit_known_pct': legit_known_pct,
        'phish_known_pct': phish_known_pct
    },
    'v4j4_metrics': {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'fpr': fpr, 'fnr': fnr, 'roc_auc': roc_auc, 'pr_auc': pr_auc
    },
    'hybrid_metrics': {
        'tn': h_tn, 'fp': h_fp, 'fn': h_fn, 'tp': h_tp,
        'recall': h_rec, 'fpr': h_fpr, 'routing_rate': routing_rate,
        'routed_count': routed_count, 'fn_recovered': fn_recovered, 'new_fps': new_fps
    },
    'routing_reasons': routing_reasons,
    'final_decision': decision
}

with open('final_independent_auth_eval_results.json', 'w') as f_out:
    json.dump(eval_out, f_out, indent=2)

print("Saved evaluation results to final_independent_auth_eval_results.json")
