"""
Dataset Generation Script for Web Shield Model V2.
Sourcing balanced, diverse, and realistic legitimate and phishing URLs
with rich length overlap, deep paths, query strings, subdomains, and source metadata.
"""

import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
import tldextract
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.feature_extraction import extract_features, get_domain_age_status

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, 'dataset_v2.csv')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) WebShield-V2/1.0'}


def get_registered_domain(url: str) -> str:
    """Extracts registered domain or host string using tldextract."""
    try:
        ext = tldextract.extract(url)
        if ext.registered_domain:
            return ext.registered_domain
        return ext.domain or "unknown"
    except Exception:
        return "unknown"


def generate_legitimate_url_pool():
    """
    Generates a rich, diverse set of legitimate URLs with deep paths,
    query parameters, subdomains, OAuth flows, e-commerce, documentation,
    URL shorteners, and varied lengths (15 to 150+ chars).
    """
    urls_with_metadata = []

    # 1. Base Popular Domains from Tranco Top 1M
    base_domains = []
    try:
        print("--- Fetching Tranco Top 1M for Base Domains ---")
        r = requests.get('https://tranco-list.eu/top-1m.csv.zip', headers=HEADERS, timeout=10)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            df_tranco = pd.read_csv(z.open(csv_name), header=None, names=['rank', 'domain'])
            base_domains = df_tranco['domain'].head(500).tolist()
    except Exception as e:
        print(f"Warning: Tranco download failed ({e}). Using fallback domain list.")

    if len(base_domains) < 100:
        base_domains = [
            "google.com", "github.com", "wikipedia.org", "amazon.com", "microsoft.com",
            "apple.com", "paypal.com", "stripe.com", "chase.com", "wellsfargo.com",
            "bankofamerica.com", "reddit.com", "stackoverflow.com", "medium.com",
            "linkedin.com", "twitter.com", "youtube.com", "facebook.com", "instagram.com",
            "netflix.com", "spotify.com", "dropbox.com", "slack.com", "atlassian.com",
            "cloudflare.com", "adobe.com", "salesforce.com", "zoom.us", "wordpress.com"
        ]

    # Sub-templates to construct realistic diverse legitimate URLs
    path_templates = [
        # Deep paths & articles
        "/wiki/Phishing_detection_techniques",
        "/wiki/Machine_learning_security",
        "/docs/latest/guide/getting-started.html",
        "/docs/v2/api/reference/overview",
        "/questions/12345678/how-to-solve-xgboost-feature-importance-issue",
        "/questions/87654321/python-requests-ssl-verification-failed",
        "/@user/building-a-cybersecurity-dashboard-2026-guide",
        "/@security-engineer/threat-intelligence-analysis-pipeline",
        "/dp/B08N5WRWNW/ref=sr_1_1?dchild=1&keywords=laptop",
        "/products/electronics/smart-home-security-camera-v2",
        "/2026/09/08/technology/cybersecurity-ai-advancements-report.html",
        "/news/security/vulnerability-disclosure-policy-updates",
        # Search & Queries
        "/search?q=machine+learning+phishing+detection+xgboost+shap&hl=en",
        "/search?q=python+tldextract+domain+parser+tutorial&source=hp",
        "/results?search_query=how+to+configure+nginx+ssl+tls+certificates",
        "/watch?v=dQw4w9WgXcQ&feature=emb_title&ab_channel=OfficialVideo",
        # OAuth & SSO Flows
        "/o/oauth2/v2/auth?scope=email%20profile&response_type=code&redirect_uri=https://example.com/callback",
        "/common/oauth2/v2.0/authorize?client_id=12345678-abcd-ef01-2345-6789abcdef01",
        "/auth/authorize?client_id=com.webshield.app&response_type=code&state=xyz123",
        # Payment & Checkout
        "/pay/cs_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
        "/cgi-bin/webscr?cmd=_express-checkout&token=EC-1234567890ABCDEF",
        # Shared hosting & Subdomains
        "/dashboard/settings/security/two-factor-authentication",
        "/user/profile/activity-log?period=last-30-days",
        "/app/v1/workspace/projects/phishing-detection-scanner"
    ]

    # Generate URLs with varied schemes, subdomains, paths, and homepages
    idx = 0
    for dom in base_domains:
        scheme = "https://" if idx % 5 != 0 else "http://"
        sub = "www." if idx % 3 == 0 else ("docs." if idx % 7 == 0 else "")
        base_url = f"{scheme}{sub}{dom}"

        # 1. Bare homepage (1 out of 4)
        urls_with_metadata.append({
            'url': base_url + "/",
            'label': 0,
            'source': 'legit_homepage',
            'registered_domain': get_registered_domain(base_url)
        })

        # 2. Deep Path / Query URL
        path = path_templates[idx % len(path_templates)]
        urls_with_metadata.append({
            'url': base_url + path,
            'label': 0,
            'source': 'legit_deep_path',
            'registered_domain': get_registered_domain(base_url)
        })
        idx += 1

    # Add explicit URL shorteners and hard benign edge cases
    shortener_legit = [
        "https://bit.ly/3xY8zQ",
        "https://tinyurl.com/2p8x9z4n",
        "https://t.co/abc123xyz",
        "https://goo.gl/maps/123456",
        "https://ow.ly/i/12345"
    ]
    for s_url in shortener_legit:
        urls_with_metadata.append({
            'url': s_url,
            'label': 0,
            'source': 'legit_shortener',
            'registered_domain': get_registered_domain(s_url)
        })

    return urls_with_metadata


def generate_phishing_url_pool():
    """
    Generates a rich, diverse set of phishing URLs including real URLhaus/OpenPhish feeds
    plus short phishing URLs, IP-hosted phishing, credential harvesting, typosquatting, etc.
    """
    urls_with_metadata = []

    # Source 1: Real URLhaus Online Feed
    try:
        print("--- Fetching Live Phishing Feed (URLhaus) ---")
        r = requests.get('https://urlhaus.abuse.ch/downloads/text_online/', headers=HEADERS, timeout=10)
        if r.status_code == 200:
            lines = [line.strip() for line in r.text.splitlines() if line and not line.startswith('#')]
            for u in lines[:600]:
                if u.startswith(('http://', 'https://')):
                    urls_with_metadata.append({
                        'url': u,
                        'label': 1,
                        'source': 'urlhaus_live',
                        'registered_domain': get_registered_domain(u)
                    })
            print(f"Fetched {len(urls_with_metadata)} phishing URLs from URLhaus.")
    except Exception as e:
        print(f"Warning: Could not fetch URLhaus feed ({e}).")

    # Source 2: OpenPhish Feed
    try:
        print("--- Fetching Live Phishing Feed (OpenPhish) ---")
        r = requests.get('https://openphish.com/feed.txt', headers=HEADERS, timeout=10)
        if r.status_code == 200:
            lines = [line.strip() for line in r.text.splitlines() if line and not line.startswith('#')]
            existing_urls = {item['url'] for item in urls_with_metadata}
            for u in lines:
                if u.startswith(('http://', 'https://')) and u not in existing_urls:
                    urls_with_metadata.append({
                        'url': u,
                        'label': 1,
                        'source': 'openphish_live',
                        'registered_domain': get_registered_domain(u)
                    })
                    if len(urls_with_metadata) >= 1000:
                        break
            print(f"Total phishing URLs after OpenPhish: {len(urls_with_metadata)}.")
    except Exception as e:
        print(f"Warning: Could not fetch OpenPhish feed ({e}).")

    # Fallback / Short / IP Phishing URLs to guarantee diverse short and long phishing
    diverse_phishing = [
        "http://192.168.1.1/login",
        "http://10.0.0.1/verify",
        "http://192.168.0.100/auth/signin",
        "http://paypa1.com/login",
        "http://g00gle.com/verify",
        "http://short.phish/go",
        "http://bit.ly/phish-account-update",
        "http://tinyurl.com/fake-login-bank",
        "http://secure-update-paypal.com.verify-user-account.info/login.php?session=123",
        "http://banking-secure-auth.net/update-account/login.html",
        "http://verify-microsoft-security-alert.org/signin/",
        "http://chase-bank-online-security-update.com/login",
        "http://wellsfargo-account-verify-alert.net/secure/",
        "http://appleid-verify-account-info.com/login.php",
        "http://netflix-billing-update-account.org/signin"
    ]

    existing_urls = {item['url'] for item in urls_with_metadata}
    for p_url in diverse_phishing:
        if p_url not in existing_urls:
            urls_with_metadata.append({
                'url': p_url,
                'label': 1,
                'source': 'phish_diverse_samples',
                'registered_domain': get_registered_domain(p_url)
            })

    return urls_with_metadata


def process_row_v2(row):
    """Safely extracts features for a single URL entry in dataset V2."""
    url = row['url']
    label = row['label']
    source = row['source']
    reg_domain = row['registered_domain']

    try:
        feat = extract_features(url)
        # Compute V2 domain age clean and known features
        age_months = feat['domain_age_months']
        age_clean = 0 if age_months == -1 else age_months
        age_known = 0 if age_months == -1 else 1

        record = {
            'url': url,
            'label': label,
            'source': source,
            'registered_domain': reg_domain,
            'url_length': feat['url_length'],
            'having_ip': feat['having_ip'],
            'has_at_symbol': feat['has_at_symbol'],
            'redirect_count': feat['redirect_count'],
            'subdomain_count': feat['subdomain_count'],
            'hyphen_count': feat['hyphen_count'],
            'domain_entropy': feat['domain_entropy'],
            'has_suspicious_keyword': feat['has_suspicious_keyword'],
            'ssl_valid': feat['ssl_valid'],
            'domain_age_months': age_months,
            'domain_age_months_clean': age_clean,
            'domain_age_known': age_known
        }
        return record
    except Exception:
        return None


def main():
    legit_items = generate_legitimate_url_pool()
    phish_items = generate_phishing_url_pool()

    all_items = legit_items + phish_items
    total_raw_items = len(all_items)
    print(f"\n--- Total Raw URLs Collected: {total_raw_items} (Legit: {len(legit_items)}, Phish: {len(phish_items)}) ---")

    print("\n--- Starting Parallel Feature Extraction for Dataset V2 (ThreadPoolExecutor max_workers=20) ---")
    processed_records = []
    completed = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_row_v2, item) for item in all_items]
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                processed_records.append(res)
            completed += 1
            if completed % 200 == 0 or completed == total_raw_items:
                print(f"Progress: {completed}/{total_raw_items} URLs processed ({len(processed_records)} valid records)...")

    df_full = pd.DataFrame(processed_records)
    raw_count = len(df_full)

    # DEDUPLICATION
    # 1. Deduplicate by exact URL
    df_dedup_url = df_full.drop_duplicates(subset=['url']).copy()
    url_dedup_count = len(df_dedup_url)

    # 2. Deduplicate by feature vector + label to eliminate redundant identical feature rows
    feature_cols_v2 = [
        'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
        'subdomain_count', 'hyphen_count', 'domain_entropy',
        'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
    ]
    df_final = df_dedup_url.drop_duplicates(subset=feature_cols_v2 + ['label']).copy()
    final_count = len(df_final)

    df_final.to_csv(output_path, index=False)

    print("\n==================================================")
    print("DATASET V2 GENERATION & DEDUPLICATION SUMMARY")
    print("==================================================")
    print(f"   Output File Path        : {output_path}")
    print(f"   Raw Processed Rows      : {raw_count}")
    print(f"   After URL Deduplication  : {url_dedup_count} (Removed {raw_count - url_dedup_count})")
    print(f"   After Feature Dedup     : {final_count} (Removed {url_dedup_count - final_count})")
    print(f"   Total Unique V2 Rows    : {final_count}")
    print(f"   Duplicate Rate          : {(1.0 - final_count / raw_count) * 100:.2f}%")
    print(f"   Legitimate Samples (0)  : {(df_final['label'] == 0).sum()}")
    print(f"   Phishing Samples (1)    : {(df_final['label'] == 1).sum()}")
    print(f"   Unique Hostnames        : {df_final['url'].apply(lambda u: tldextract.extract(u).fqdn).nunique()}")
    print(f"   Unique Registered Domains: {df_final['registered_domain'].nunique()}")
    print("==================================================\n")

    print("Legitimate URL Length Stats in V2:")
    legit_lengths = df_final[df_final['label'] == 0]['url_length']
    print(f"  Mean: {legit_lengths.mean():.2f} | Median: {legit_lengths.median()} | Min: {legit_lengths.min()} | Max: {legit_lengths.max()}")
    print("  Quantiles (25, 50, 75, 90, 95, 99):", np.percentile(legit_lengths, [25, 50, 75, 90, 95, 99]))

    print("\nPhishing URL Length Stats in V2:")
    phish_lengths = df_final[df_final['label'] == 1]['url_length']
    print(f"  Mean: {phish_lengths.mean():.2f} | Median: {phish_lengths.median()} | Min: {phish_lengths.min()} | Max: {phish_lengths.max()}")
    print("  Quantiles (25, 50, 75, 90, 95, 99):", np.percentile(phish_lengths, [25, 50, 75, 90, 95, 99]))


if __name__ == '__main__':
    main()
