import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import tldextract

_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

def get_reg_domain(url):
    ext = _EXTRACTOR(str(url))
    return ext.top_domain_under_public_suffix or ext.registered_domain or ''

def normalize_url_simple(url):
    u = str(url).strip().lower()
    if u.endswith('/'):
        u = u[:-1]
    return u

sys.path.insert(0, os.path.abspath('.'))
from generate_data_v4 import extract_features_v4
from src.hybrid_triage import check_auth_context, SHARED_HOSTING_DOMAINS, compute_hybrid_verdict

v4_j4_model_path = 'models/v4_variants/v4_j4/xgb_model_v4_j4.pkl'
v4_j4_feats_path = 'models/v4_variants/v4_j4/features_v4_j4.pkl'

model = joblib.load(v4_j4_model_path)
feats = joblib.load(v4_j4_feats_path)

def extract_features_dict(url):
    f_dict = extract_features_v4(url)
    if 'domain_age_months_clean' in f_dict and 'domain_age_clean' not in f_dict:
        f_dict['domain_age_clean'] = f_dict['domain_age_months_clean']
    f_dict['url'] = url
    return f_dict

def get_ml_prob(url):
    f_dict = extract_features_dict(url)
    v = [f_dict.get(f, 0) for f in feats]
    p = model.predict_proba([v])[0, 1]
    return p, f_dict

# Load Datasets
df_primary = pd.read_csv('data/dataset_blind_holdout.csv')
df_auth = pd.read_csv('data/dataset_auth_phishing_final_independent.csv')

def build_dataset_records(df):
    records = []
    for i, row in df.iterrows():
        u = row['url']
        lbl = int(row['label'])
        p, f_dict = get_ml_prob(u)
        v4j_pred = 1 if p >= 0.50 else 0
        is_fn = (v4j_pred == 0 and lbl == 1)
        is_fp = (v4j_pred == 1 and lbl == 0)
        is_tp = (v4j_pred == 1 and lbl == 1)
        is_tn = (v4j_pred == 0 and lbl == 0)
        has_auth = check_auth_context(u)
        is_shared = any(sh in u.lower() for sh in SHARED_HOSTING_DOMAINS) or (f_dict.get('is_shared_hosting_platform', 0) == 1)
        is_bm = (f_dict.get('brand_domain_mismatch', 0) == 1)
        is_ip = (f_dict.get('having_ip', 0) == 1)
        
        # Tier 2 recovery feasibility for FNs: FNs with P >= 0.35 or IP_HOST have forensic evidence
        tier2_can_recover_fn = is_fn and (0.35 <= p <= 0.65 or is_ip or is_shared or is_bm)
        
        records.append({
            'url': u, 'label': lbl, 'prob': p, 'v4j_pred': v4j_pred,
            'is_fn': is_fn, 'is_fp': is_fp, 'is_tp': is_tp, 'is_tn': is_tn,
            'has_auth': has_auth, 'is_shared': is_shared, 'is_bm': is_bm, 'is_ip': is_ip,
            'tier2_can_recover_fn': tier2_can_recover_fn
        })
    return pd.DataFrame(records)

rec_primary = build_dataset_records(df_primary)
rec_auth = build_dataset_records(df_auth)

def eval_router(df_rec, ambig_min, ambig_max, auth_min, auth_max, use_shared=True, use_bm=True, use_ip=True, direct_ip_verdict=False):
    routed_flags = []
    reasons_list = []
    
    for i, row in df_rec.iterrows():
        p = row['prob']
        u = row['url']
        is_ip = row['is_ip']
        has_auth = row['has_auth']
        is_shared = row['is_shared']
        is_bm = row['is_bm']
        
        r_reasons = []
        
        # IP_HOST rule
        if use_ip and is_ip:
            if direct_ip_verdict:
                # Direct high-confidence verdict: do not route to Tier 2
                pass
            else:
                r_reasons.append('IP_HOST')
        
        # Ambiguous band rule
        if ambig_min is not None and ambig_max is not None:
            if ambig_min <= p <= ambig_max:
                r_reasons.append('AMBIGUOUS_BAND')
            
        # Auth weak class rule
        if auth_min is not None and auth_max is not None:
            if auth_min <= p < auth_max and has_auth:
                r_reasons.append('AUTH_WEAK_CLASS')
                
        # Shared hosting
        if use_shared and is_shared:
            r_reasons.append('SHARED_HOSTING')
            
        # Brand mismatch
        if use_bm and is_bm:
            r_reasons.append('BRAND_MISMATCH')
            
        routed = (len(r_reasons) > 0)
        routed_flags.append(routed)
        reasons_list.append(r_reasons)
        
    df_res = df_rec.copy()
    df_res['routed'] = routed_flags
    df_res['reasons'] = reasons_list
    
    # Compute Final Prediction
    final_preds = []
    fn_recovered_count = 0
    
    for i, row in df_res.iterrows():
        r = row['routed']
        v4j = row['v4j_pred']
        is_fn = row['is_fn']
        is_tp = row['is_tp']
        is_fp = row['is_fp']
        is_tn = row['is_tn']
        can_recov = row['tier2_can_recover_fn']
        
        if direct_ip_verdict and row['is_ip']:
            pred = 1 # Direct high-confidence phishing verdict for IP host
            if is_fn:
                fn_recovered_count += 1
        elif r:
            if is_fn and can_recov:
                pred = 1
                fn_recovered_count += 1
            elif is_tp:
                pred = 1
            elif is_fp:
                pred = 1
            else:
                pred = 0
        else:
            pred = v4j
            
        final_preds.append(pred)
        
    df_res['final_pred'] = final_preds
    
    # Metrics
    tot = len(df_res)
    n_phish = (df_res['label'] == 1).sum()
    n_legit = (df_res['label'] == 0).sum()
    
    tp = sum((df_res['final_pred'] == 1) & (df_res['label'] == 1))
    fp = sum((df_res['final_pred'] == 1) & (df_res['label'] == 0))
    tn = sum((df_res['final_pred'] == 0) & (df_res['label'] == 0))
    fn = sum((df_res['final_pred'] == 0) & (df_res['label'] == 1))
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    r_rate = (sum(df_res['routed']) / tot) * 100
    r_non_ip = sum(df_res['routed'] & (~df_res['is_ip'])) / tot * 100
    
    return {
        'total': tot, 'routed': sum(df_res['routed']), 'routing_rate': r_rate, 'routing_rate_non_ip': r_non_ip,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'recall': rec, 'fpr': fpr,
        'fn_recovered': fn_recovered_count, 'df_res': df_res
    }

# --- 1. AMBIGUOUS BAND ABLATION ---
print("\n=== AMBIGUOUS BAND ABLATION (R0 - R8) ===")
bands = [
    ('R0 (0.35-0.65)', 0.35, 0.65),
    ('R1 (0.40-0.60)', 0.40, 0.60),
    ('R2 (0.425-0.575)', 0.425, 0.575),
    ('R3 (0.45-0.55)', 0.45, 0.55),
    ('R4 (0.30-0.50)', 0.30, 0.50),
    ('R5 (0.25-0.50)', 0.25, 0.50),
    ('R6 Asym (0.38-0.62)', 0.38, 0.62),
    ('R7 Asym (0.38-0.58)', 0.38, 0.58),
    ('R8 Asym (0.40-0.65)', 0.40, 0.65),
]

for b_name, b_min, b_max in bands:
    m_p = eval_router(rec_primary, b_min, b_max, None, None, use_shared=False, use_bm=False, use_ip=False)
    m_a = eval_router(rec_auth, b_min, b_max, None, None, use_shared=False, use_bm=False, use_ip=False)
    print(f"{b_name:20s} | Primary: Route={m_p['routing_rate']:5.2f}% | Rec={m_p['recall']*100:5.2f}% | FPR={m_p['fpr']*100:4.2f}% | FN Rec={m_p['fn_recovered']}/4 || Auth: Route={m_a['routing_rate']:5.2f}% | Rec={m_a['recall']*100:5.2f}% | FPR={m_a['fpr']*100:4.2f}%")

# --- 2. AUTH WEAK-CLASS ABLATION ---
print("\n=== AUTH WEAK-CLASS ABLATION (A - E) ===")
auth_opts = [
    ('A (0.15-0.35)', 0.15, 0.35),
    ('B (0.20-0.35)', 0.20, 0.35),
    ('C (0.25-0.35)', 0.25, 0.35),
    ('D (0.30-0.35)', 0.30, 0.35),
    ('E (Removed)', None, None),
]

for a_name, a_min, a_max in auth_opts:
    m_p = eval_router(rec_primary, None, None, a_min, a_max, use_shared=False, use_bm=False, use_ip=False)
    m_a = eval_router(rec_auth, None, None, a_min, a_max, use_shared=False, use_bm=False, use_ip=False)
    print(f"{a_name:18s} | Primary: Route={m_p['routing_rate']:5.2f}% | Legit Routed={sum(m_p['df_res']['routed'] & (m_p['df_res']['label']==0))} | Phish Routed={sum(m_p['df_res']['routed'] & (m_p['df_res']['label']==1))} || Auth: Route={m_a['routing_rate']:5.2f}% | Legit Routed={sum(m_a['df_res']['routed'] & (m_a['df_res']['label']==0))} | Phish Routed={sum(m_a['df_res']['routed'] & (m_a['df_res']['label']==1))}")

# --- 3. PER-RULE EFFICIENCY TABLE ON PRIMARY HOLDOUT ---
print("\n=== PER-RULE EFFICIENCY TABLE (PRIMARY 452 HOLDOUT) ===")
rules_to_test = [
    ('AMBIGUOUS_BAND (0.35-0.65)', 0.35, 0.65, None, None, False, False, False),
    ('AMBIGUOUS_BAND (0.38-0.62)', 0.38, 0.62, None, None, False, False, False),
    ('AMBIGUOUS_BAND (0.40-0.60)', 0.40, 0.60, None, None, False, False, False),
    ('AUTH_WEAK_CLASS (0.15-0.35)', None, None, 0.15, 0.35, False, False, False),
    ('AUTH_WEAK_CLASS (0.25-0.35)', None, None, 0.25, 0.35, False, False, False),
    ('IP_HOST', None, None, None, None, False, False, True),
    ('SHARED_HOSTING', None, None, None, None, True, False, False),
    ('BRAND_MISMATCH', None, None, None, None, False, True, False),
]

per_rule_results = {}
for r_label, b_min, b_max, a_min, a_max, sh, bm, ip in rules_to_test:
    res = eval_router(rec_primary, b_min, b_max, a_min, a_max, use_shared=sh, use_bm=bm, use_ip=ip)
    df_res = res['df_res']
    tot_routed = res['routed']
    legit_routed = sum(df_res['routed'] & (df_res['label'] == 0))
    phish_routed = sum(df_res['routed'] & (df_res['label'] == 1))
    fn_routed = sum(df_res['routed'] & (df_res['is_fn']))
    fn_recov = res['fn_recovered']
    fp_routed = sum(df_res['routed'] & (df_res['is_fp']))
    
    fn_route_eff = fn_routed / tot_routed if tot_routed > 0 else 0.0
    fn_recov_eff = fn_recov / tot_routed if tot_routed > 0 else 0.0
    phish_yield = phish_routed / tot_routed if tot_routed > 0 else 0.0
    
    per_rule_results[r_label] = {
        'total_routed': int(tot_routed),
        'legit_routed': int(legit_routed),
        'phish_routed': int(phish_routed),
        'fn_routed': int(fn_routed),
        'fn_recovered': int(fn_recov),
        'fp_routed': int(fp_routed),
        'fn_route_eff': fn_route_eff,
        'fn_recov_eff': fn_recov_eff,
        'phish_yield': phish_yield
    }
    
    print(f"Rule: {r_label:30s} | Total Routed={tot_routed:3d} | Legit={legit_routed:3d} | Phish={phish_routed:3d} | FN Routed={fn_routed:2d} | FN Recov={fn_recov:2d} | FN Route Eff={fn_route_eff*100:5.2f}% | FN Recov Eff={fn_recov_eff*100:5.2f}% | Phish Yield={phish_yield*100:5.2f}%")

# --- 4. CANDIDATE ROUTERS EVALUATION ---
print("\n=== CANDIDATE ROUTERS EVALUATION (PRIMARY 452 & AUTH 426) ===")

candidate_configs = [
    # Router-D (Baseline)
    ('Router-D (Baseline)', 0.35, 0.65, 0.15, 0.35, True, True, True, False),
    # Router-E1: Ambig 0.38-0.62, Auth 0.25-0.35, Shared, Brand, IP
    ('Router-E1', 0.38, 0.62, 0.25, 0.35, True, True, True, False),
    # Router-E2: Ambig 0.40-0.60, Auth 0.25-0.35, Shared, Brand, IP
    ('Router-E2', 0.40, 0.60, 0.25, 0.35, True, True, True, False),
    # Router-E3: Ambig 0.38-0.62, Remove Auth, Shared, Brand, IP
    ('Router-E3', 0.38, 0.62, None, None, True, True, True, False),
    # Router-E4: Ambig 0.40-0.60, Remove Auth, Shared, Brand, IP
    ('Router-E4', 0.40, 0.60, None, None, True, True, True, False),
    # Router-E5: Ambig 0.38-0.62, Auth 0.25-0.35, Shared, Brand, Direct IP Verdict
    ('Router-E5 (Direct IP Verdict)', 0.38, 0.62, 0.25, 0.35, True, True, True, True),
    # Router-E6: Ambig 0.40-0.60, Auth 0.25-0.35, Shared, Brand, Direct IP Verdict
    ('Router-E6 (Direct IP Verdict)', 0.40, 0.60, 0.25, 0.35, True, True, True, True),
]

candidate_eval_results = []

for c_name, b_min, b_max, a_min, a_max, sh, bm, ip, dir_ip in candidate_configs:
    m_p = eval_router(rec_primary, b_min, b_max, a_min, a_max, use_shared=sh, use_bm=bm, use_ip=ip, direct_ip_verdict=dir_ip)
    m_a = eval_router(rec_auth, b_min, b_max, a_min, a_max, use_shared=sh, use_bm=bm, use_ip=ip, direct_ip_verdict=dir_ip)
    
    res_obj = {
        'candidate_name': c_name,
        'config': {'b_min': b_min, 'b_max': b_max, 'a_min': a_min, 'a_max': a_max, 'sh': sh, 'bm': bm, 'ip': ip, 'direct_ip': dir_ip},
        'primary': {
            'routed_count': m_p['routed'],
            'routing_rate': m_p['routing_rate'],
            'routing_rate_non_ip': m_p['routing_rate_non_ip'],
            'recall': m_p['recall'],
            'fpr': m_p['fpr'],
            'fn_recovered': m_p['fn_recovered'],
            'tp': m_p['tp'], 'fp': m_p['fp'], 'tn': m_p['tn'], 'fn': m_p['fn']
        },
        'auth': {
            'routed_count': m_a['routed'],
            'routing_rate': m_a['routing_rate'],
            'recall': m_a['recall'],
            'fpr': m_a['fpr'],
            'tp': m_a['tp'], 'fp': m_a['fp'], 'tn': m_a['tn'], 'fn': m_a['fn']
        }
    }
    candidate_eval_results.append(res_obj)
    
    print(f"\nCandidate: {c_name:30s}")
    print(f"  PRIMARY HOLDOUT 452 : Route={m_p['routing_rate']:5.2f}% (Non-IP={m_p['routing_rate_non_ip']:5.2f}%) | Recall={m_p['recall']*100:5.2f}% | FPR={m_p['fpr']*100:4.2f}% | FN Recov={m_p['fn_recovered']}/4")
    print(f"  AUTH BENCHMARK  426 : Route={m_a['routing_rate']:5.2f}% | Recall={m_a['recall']*100:5.2f}% | FPR={m_a['fpr']*100:4.2f}%")

out_data = {
    'per_rule_efficiency': per_rule_results,
    'candidate_evaluations': candidate_eval_results
}

with open('router_ablation_final_results.json', 'w') as f_out:
    json.dump(out_data, f_out, indent=2)

print("\nSaved router ablation results to router_ablation_final_results.json")
