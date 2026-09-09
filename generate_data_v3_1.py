"""
Dataset Generation Script for Web Shield Model V3.1.
Augments dataset_v3.csv with independent domain-based phishing training samples
to resolve coverage gaps (domain-based phishing, valid HTTPS phishing, subdomains, hyphens)
without touching untouched holdout sets or compromising URL-length balance.
"""

import os
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import tldextract
from src.feature_extraction import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

v3_path = os.path.join(DATA_DIR, 'dataset_v3.csv')
v3_1_path = os.path.join(DATA_DIR, 'dataset_v3_1.csv')


def get_registered_domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        if ext.registered_domain:
            return ext.registered_domain
        return ext.domain or "unknown"
    except Exception:
        return "unknown"


def generate_augmented_phishing_pool():
    """
    Independent domain-based phishing training URLs covering structural gaps:
    1. Short domain phishing (url_length 20-35, having_ip=0)
    2. Valid HTTPS domain phishing (ssl_valid=1, having_ip=0)
    3. Subdomain / hyphenated domain phishing (subdomain_count >= 1, hyphen_count >= 1)
    4. Low-entropy / normal-looking path phishing
    
    IMPORTANT: None of these overlap with hard holdout URLs or test domains.
    """
    new_phish = []

    # Category 1: Short & Medium Domain-Based Phishing (Length 22 - 40, having_ip=0, HTTPS & HTTP)
    short_medium_domains = [
        "account-update-portal.com", "secure-login-check.org", "verify-user-credentials.net",
        "auth-session-gateway.info", "billing-security-center.biz", "online-banking-auth.co",
        "user-verification-hub.site", "client-portal-secure.online", "account-recovery-desk.tech",
        "login-verification-page.store", "security-alert-center.live", "auth-identity-check.space",
        "verify-account-status.website", "secure-customer-portal.club", "update-payment-info.link",
        "login-security-service.news", "account-auth-system.work", "verify-billing-details.click",
        "secure-login-session.today", "update-user-profile.online", "auth-portal-direct.tech",
        "customer-verify-login.store", "security-update-now.site", "account-session-verify.live",
        "online-portal-auth.space", "client-security-check.website", "login-auth-gateway.club",
        "verify-identity-today.link", "secure-account-service.news", "update-billing-center.work",
        "auth-login-verification.click", "customer-security-hub.today", "account-portal-check.online",
        "verify-user-session.tech", "secure-billing-login.store", "update-account-details.site",
        "login-portal-service.live", "auth-customer-check.space", "security-verification-now.website",
        "account-login-direct.club", "verify-payment-portal.link", "secure-identity-service.news"
    ]

    for idx, dom in enumerate(short_medium_domains):
        scheme = "https://" if idx % 2 == 0 else "http://"
        sub = "login." if idx % 3 == 0 else ("auth." if idx % 5 == 0 else "www.")
        paths = ["/index.php", "/login.html", "/auth/verify", "/account/signin", "/secure/update", "/session/check"]
        path = paths[idx % len(paths)]
        
        url = f"{scheme}{sub}{dom}{path}"
        new_phish.append({
            'url': url,
            'label': 1,
            'source': 'augmented_domain_phish',
            'registered_domain': get_registered_domain(url)
        })

    # Category 2: Valid HTTPS Phishing on Subdomains & Compromised Domains (ssl_valid=1, having_ip=0)
    https_domains = [
        "secure-cloud-portal-auth.com", "verify-identity-online-gateway.net",
        "account-security-verification-center.org", "login-auth-session-manager.info",
        "customer-support-portal-verify.site", "online-billing-update-service.tech",
        "user-account-recovery-system.online", "secure-login-credentials-check.store",
        "auth-session-validation-desk.live", "client-verification-alert-center.space",
        "update-user-security-settings.website", "login-account-status-service.club",
        "verify-billing-information-now.link", "secure-customer-auth-gateway.news",
        "account-recovery-verification-hub.work", "online-security-alert-service.click",
        "user-login-session-verification.today", "client-auth-portal-direct.online",
        "secure-account-verification-page.tech", "update-payment-security-details.store",
        "auth-identity-verification-center.site", "customer-login-security-check.live",
        "online-portal-account-verify.space", "client-session-security-hub.website",
        "secure-billing-verification-service.club", "update-user-credentials-now.link",
        "auth-portal-security-check.news", "account-verification-alert-system.work",
        "online-login-customer-service.click", "verify-account-security-details.today"
    ]

    for idx, dom in enumerate(https_domains):
        paths = [
            "/assets/app-update.zip", "/downloads/security-patch.exe",
            "/wp-content/plugins/fix.js", "/assets/document-view.pdf.exe",
            "/auth/verify?session_id=89234723", "/login/account-update.php?ref=mail",
            "/secure/gateway/checkpoint.html", "/user/verify-identity-now.php"
        ]
        path = paths[idx % len(paths)]
        url = f"https://{dom}{path}"
        new_phish.append({
            'url': url,
            'label': 1,
            'source': 'augmented_https_phish',
            'registered_domain': get_registered_domain(url)
        })

    # Category 3: Subdomain & Hyphenated Phishing (High subdomain_count, hyphen_count)
    subdomain_phish_domains = [
        "login.auth.security-check-online.com", "verify.account.user-portal-update.net",
        "signin.secure.billing-verification-center.org", "auth.client.account-recovery-gateway.info",
        "update.user.security-alert-hub.site", "portal.online.client-verification-service.tech",
        "checkpoint.secure.account-login-manager.online", "gateway.auth.user-session-verify.store",
        "service.client.security-verification-desk.live", "hub.online.account-recovery-system.space"
    ]
    for idx, dom in enumerate(subdomain_phish_domains):
        url = f"https://{dom}/verify.php?token=9012384"
        new_phish.append({
            'url': url,
            'label': 1,
            'source': 'augmented_subdomain_phish',
            'registered_domain': get_registered_domain(url)
        })

    # Category 4: Typosquatting / Brand Impersonation Training Samples (Independent of Holdouts)
    brand_phish_domains = [
        "paypa1-security-center.com", "g00gle-account-verification.net",
        "micros0ft-online-verify.org", "app1e-id-security-update.info",
        "chase-online-banking-check.site", "wellsfarg0-account-verify.tech",
        "bankofamer1ca-security-portal.online", "netfl1x-account-update-billing.store",
        "spot1fy-premium-verify.live", "dropb0x-shared-file-login.space",
        "amaz0n-order-verification.website", "str1pe-payment-gateway-auth.club"
    ]
    for idx, dom in enumerate(brand_phish_domains):
        scheme = "https://" if idx % 2 == 0 else "http://"
        url = f"{scheme}{dom}/login"
        new_phish.append({
            'url': url,
            'label': 1,
            'source': 'augmented_brand_phish',
            'registered_domain': get_registered_domain(url)
        })

    return new_phish


def process_single_url(item):
    url = item['url']
    label = item['label']
    src = item['source']
    reg_dom = item['registered_domain']

    try:
        from unittest.mock import patch
        ssl_val = 1 if url.lower().startswith('https://') else 0
        with patch('src.feature_extraction.verify_ssl', return_value=ssl_val), \
             patch('src.feature_extraction.get_domain_age_months', return_value=-1):
            feat = extract_features(url)
                
        age = feat.get('domain_age_months', -1)
        
        row = {
            'url': url,
            'label': label,
            'source': src,
            'registered_domain': reg_dom,
            'url_length': feat['url_length'],
            'having_ip': feat['having_ip'],
            'has_at_symbol': feat['has_at_symbol'],
            'redirect_count': feat['redirect_count'],
            'subdomain_count': feat['subdomain_count'],
            'hyphen_count': feat['hyphen_count'],
            'domain_entropy': feat['domain_entropy'],
            'has_suspicious_keyword': feat['has_suspicious_keyword'],
            'ssl_valid': feat['ssl_valid'],
            'domain_age_months_clean': 0 if age == -1 else age,
            'domain_age_known': 0 if age == -1 else 1
        }
        return row
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None


def main():
    df_v3 = pd.read_csv(v3_path)
    print(f"Loaded dataset_v3.csv: {len(df_v3)} rows (Legit: {(df_v3['label']==0).sum()}, Phish: {(df_v3['label']==1).sum()})")

    aug_phish_pool = generate_augmented_phishing_pool()
    print(f"Generated {len(aug_phish_pool)} new domain-based phishing training items.")

    new_rows = []
    for item in aug_phish_pool:
        r = process_single_url(item)
        if r is not None:
            new_rows.append(r)

    df_aug = pd.DataFrame(new_rows)
    print(f"Processed {len(df_aug)} valid augmented rows.")

    # Combine V3 and augmented rows
    df_combined = pd.concat([df_v3, df_aug], ignore_index=True)

    # Feature vector deduplication
    feature_cols = [
        'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
        'subdomain_count', 'hyphen_count', 'domain_entropy',
        'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
    ]
    df_v3_1 = df_combined.drop_duplicates(subset=feature_cols + ['label']).copy()

    df_v3_1.to_csv(v3_1_path, index=False)

    legit_cnt = (df_v3_1['label'] == 0).sum()
    phish_cnt = (df_v3_1['label'] == 1).sum()

    print("\n======================================================================")
    print("DATASET V3.1 AUGMENTATION SUMMARY")
    print("======================================================================")
    print(f"   Output File Path        : {v3_1_path}")
    print(f"   Total Unique V3.1 Rows  : {len(df_v3_1)}")
    print(f"   Legitimate Samples (0)  : {legit_cnt}")
    print(f"   Phishing Samples (1)    : {phish_cnt}")
    print(f"   Scale Pos Weight Ratio  : {legit_cnt / phish_cnt:.4f}")
    print(f"   Unique Registered Domains: {df_v3_1['registered_domain'].nunique()}")
    print("======================================================================\n")


if __name__ == '__main__':
    main()
