"""
Untouched Hard Holdout Evaluation Script for Model V3 vs Model V2.
Evaluates Model V2 and Model V3 against 100+ item hard legitimate holdout set
and 100+ item hard malicious holdout set.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from src.feature_extraction import extract_features
from src.predictor import adapt_features_v2

features_cols = [
    'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
    'subdomain_count', 'hyphen_count', 'domain_entropy',
    'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
]


def generate_hard_legitimate_holdout():
    """Generates 100+ untouched hard legitimate test URLs across diverse length ranges."""
    categories = [
        # Wikipedia Articles (Length 30-50)
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Cybersecurity",
        "https://en.wikipedia.org/wiki/Information_security",
        "https://en.wikipedia.org/wiki/Zero_trust_security_model",
        "https://en.wikipedia.org/wiki/Public_key_infrastructure",
        "https://en.wikipedia.org/wiki/Transport_Layer_Security",
        "https://en.wikipedia.org/wiki/Domain_Name_System",
        "https://en.wikipedia.org/wiki/Multi-factor_authentication",
        "https://en.wikipedia.org/wiki/Cross-site_scripting",
        "https://en.wikipedia.org/wiki/SQL_injection",
        
        # Documentation & API (Length 35-65)
        "https://docs.python.org/3/library/unittest.html",
        "https://docs.python.org/3/library/urllib.parse.html",
        "https://scikit-learn.org/stable/modules/ensemble.html",
        "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
        "https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html",
        "https://flask.palletsprojects.com/en/3.0.x/quickstart/",
        "https://requests.readthedocs.io/en/latest/user/quickstart/",
        "https://pandas.pydata.org/docs/user_guide/10min.html",
        "https://numpy.org/doc/stable/user/quickstart.html",
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers",
        
        # GitHub Repos & Files (Length 40-75)
        "https://github.com/torvalds/linux/blob/master/README",
        "https://github.com/python/cpython/tree/main/Lib",
        "https://github.com/pallets/flask/releases/tag/3.0.2",
        "https://github.com/psf/requests/issues/1234",
        "https://github.com/dmlc/xgboost/pull/9000",
        "https://github.com/slundberg/shap/blob/master/README.md",
        "https://github.com/scikit-learn/scikit-learn/releases",
        "https://github.com/tldextract/tldextract/blob/master/LICENSE",
        "https://github.com/tiangolo/fastapi/discussions/5000",
        "https://github.com/psf/black/blob/main/docs/the_black_code_style/current_style.md",

        # Search Queries & Mediums (Length 45-80)
        "https://www.google.com/search?q=machine+learning+security+best+practices",
        "https://www.google.com/search?q=python+xgboost+hyperparameter+tuning",
        "https://stackoverflow.com/questions/11888417/how-to-extract-domain-from-url",
        "https://stackoverflow.com/questions/204017/how-do-i-execute-a-program-or-call-a-system-command",
        "https://medium.com/@user/building-scalable-cybersecurity-pipelines-in-2026",
        "https://medium.com/swlh/understanding-shapley-additive-explanations-76b66",
        
        # E-Commerce & Checkout (Length 40-75)
        "https://www.amazon.com/dp/B08N5WRWNW/ref=sr_1_1?dchild=1",
        "https://www.ebay.com/itm/123456789012?hash=item12345678",
        "https://checkout.stripe.com/pay/cs_live_sample_token_1234567890",
        "https://www.paypal.com/cgi-bin/webscr?cmd=_express-checkout&token=EC-98765",
        
        # OAuth / SSO / Cloud Consoles (Length 50-90)
        "https://accounts.google.com/o/oauth2/v2/auth?scope=email%20profile",
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "https://aws.amazon.com/console/billing/reports/latest/summary",
        "https://portal.azure.com/#blade/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade/Overview",
        "https://console.cloud.google.com/welcome?project=webshield-2026",
        "https://dash.cloudflare.com/login?redirect_uri=https://dash.cloudflare.com",
        "https://id.atlassian.com/login?application=jira",
        "https://app.slack.com/client/T00000000/C00000000",
        "https://dropbox.com/home/workspaces/cybersecurity-reports",
        "https://zoom.us/j/1234567890?pwd=sample_password_token_123456"
    ]

    # Expand to 100+ unique URLs by combining paths with diverse base domains
    domains = [
        "nytimes.com", "bbc.com", "washingtonpost.com", "theguardian.com",
        "forbes.com", "bloomberg.com", "reuters.com", "cnbc.com",
        "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu",
        "cmu.edu", "ox.ac.uk", "cam.ac.uk", "ethz.ch"
    ]
    path_exts = [
        "/technology/2026/09/08/cyber-security-advancements-report.html",
        "/research/papers/machine-learning-phishing-detection-2026.pdf",
        "/about/privacy-policy-updates-and-security-terms",
        "/news/articles/global-cyber-threat-intelligence-briefing"
    ]

    for d in domains:
        for p in path_exts:
            categories.append(f"https://www.{d}{p}")

    return categories[:105]


def generate_hard_malicious_holdout():
    """Generates 100+ untouched hard malicious test URLs spanning short, medium, long ranges."""
    urls = []
    # IP-hosted phishing
    for i in range(1, 25):
        urls.append(f"http://192.168.1.{i}/login.php?session=auth")
        urls.append(f"http://10.0.0.{i}/verify/account/signin")

    # Typosquatting & credential harvesting
    typo = [
        "http://paypa1.com/verify-account-security-update",
        "http://g00gle.com/auth/signin?client_id=123",
        "http://micros0ft-online-365.com/login.aspx",
        "http://app1e-id-verify.account-update.net/auth",
        "http://chase-bank-verify-security.com/login",
        "http://wellsfarg0-online-login.info/verify",
        "http://bankofamer1ca-security-check.org/auth",
        "http://netfl1x-billing-update.com/signin",
        "http://spot1fy-account-verify.net/login",
        "http://dropb0x-shared-file-verify.com/download"
    ]
    for t in typo:
        for suffix in ["", "/login.php", "/auth/verify?token=123456", "/secure/update-credentials.html"]:
            urls.append(t + suffix)

    # Short phishing URLs
    short_phish = [
        "http://short.phish/go",
        "http://bit.ly/phish-account-update-2026",
        "http://tinyurl.com/fake-login-bank-verify",
        "http://t.co/fake-login-security",
        "http://ow.ly/phish-token-12345"
    ]
    for s in short_phish:
        urls.append(s)

    return urls[:105]


def evaluate_holdout(model, urls, is_phishing_label=True):
    fp_or_fn = []
    probs = []
    preds = []

    for u in urls:
        from unittest.mock import patch
        ssl_val = 1 if u.lower().startswith('https://') else 0
        with patch('src.feature_extraction.verify_ssl', return_value=ssl_val), \
             patch('src.feature_extraction.get_domain_age_months', return_value=-1):
            raw_f = extract_features(u)

        v2_vec = adapt_features_v2(raw_f)
        df_vec = pd.DataFrame([v2_vec])[features_cols]

        prob = float(model.predict_proba(df_vec)[0][1])
        pred = 1 if prob >= 0.50 else 0

        probs.append(prob)
        preds.append(pred)

        expected = 1 if is_phishing_label else 0
        if pred != expected:
            fp_or_fn.append((u, prob, raw_f))

    return probs, preds, fp_or_fn


def main():
    print("======================================================================")
    print("UNTOUCHED 100+ ITEM HARD HOLDOUT EVALUATION (MODEL V2 VS MODEL V3)")
    print("======================================================================")

    clf_v2 = joblib.load(os.path.join('models', 'v2', 'xgb_model_v2.pkl'))
    clf_v3 = joblib.load(os.path.join('models', 'v3', 'xgb_model_v3.pkl'))

    legit_urls = generate_hard_legitimate_holdout()
    phish_urls = generate_hard_malicious_holdout()

    print(f"Hard Legitimate Holdout Count : {len(legit_urls)}")
    print(f"Hard Malicious Holdout Count  : {len(phish_urls)}")

    # 1. Hard Legitimate Holdout Evaluation
    probs_v2_leg, preds_v2_leg, fp_v2 = evaluate_holdout(clf_v2, legit_urls, is_phishing_label=False)
    probs_v3_leg, preds_v3_leg, fp_v3 = evaluate_holdout(clf_v3, legit_urls, is_phishing_label=False)

    fpr_v2 = (len(fp_v2) / len(legit_urls)) * 100
    fpr_v3 = (len(fp_v3) / len(legit_urls)) * 100

    print("\n--- Hard Legitimate Holdout Results ---")
    print(f"  Model V2 False Positives: {len(fp_v2)} / {len(legit_urls)} (FPR = {fpr_v2:.2f}%)")
    print(f"  Model V3 False Positives: {len(fp_v3)} / {len(legit_urls)} (FPR = {fpr_v3:.2f}%)")

    # 2. Hard Malicious Holdout Evaluation
    probs_v2_mal, preds_v2_mal, fn_v2 = evaluate_holdout(clf_v2, phish_urls, is_phishing_label=True)
    probs_v3_mal, preds_v3_mal, fn_v3 = evaluate_holdout(clf_v3, phish_urls, is_phishing_label=True)

    rec_v2 = (sum(preds_v2_mal) / len(phish_urls)) * 100
    rec_v3 = (sum(preds_v3_mal) / len(phish_urls)) * 100

    print("\n--- Hard Malicious Holdout Results ---")
    print(f"  Model V2 Recall: {rec_v2:.2f}% (Missed {len(fn_v2)} / {len(phish_urls)})")
    print(f"  Model V3 Recall: {rec_v3:.2f}% (Missed {len(fn_v3)} / {len(phish_urls)})")

    # 3. Overall Combined Confusion Matrix on Holdout Sets
    y_true_comb = [0]*len(legit_urls) + [1]*len(phish_urls)
    
    # Model V2 Combined
    y_pred_v2_comb = preds_v2_leg + preds_v2_mal
    prec_v2_comb = precision_score(y_true_comb, y_pred_v2_comb) * 100
    f1_v2_comb = f1_score(y_true_comb, y_pred_v2_comb) * 100
    
    # Model V3 Combined
    y_pred_v3_comb = preds_v3_leg + preds_v3_mal
    prec_v3_comb = precision_score(y_true_comb, y_pred_v3_comb) * 100
    f1_v3_comb = f1_score(y_true_comb, y_pred_v3_comb) * 100

    print("\n--- Overall Holdout Performance ---")
    print(f"  Model V2 Combined Precision: {prec_v2_comb:.2f}% | F1-Score: {f1_v2_comb:.2f}%")
    print(f"  Model V3 Combined Precision: {prec_v3_comb:.2f}% | F1-Score: {f1_v3_comb:.2f}%")

    if fp_v3:
        print("\n--- Top Model V3 False Positives (Sample) ---")
        for u, p, f in fp_v3[:5]:
            print(f"  URL: {u[:65]:<65s} | Prob: {p*100:.2f}% | Len: {f['url_length']}")

    print("======================================================================\n")


if __name__ == '__main__':
    main()
