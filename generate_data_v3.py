"""
Dataset Generation Script for Web Shield Model V3.
Builds an unbiased, balanced dataset (dataset_v3.csv) specifically eliminating
the URL-length shortcut in the 25-60 character range, with rich legitimate diversity,
multi-source malicious samples, and full source metadata.
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
output_path = os.path.join(DATA_DIR, 'dataset_v3.csv')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) WebShield-V3/1.0'}


def get_registered_domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        if ext.registered_domain:
            return ext.registered_domain
        return ext.domain or "unknown"
    except Exception:
        return "unknown"


def generate_legitimate_url_pool():
    """
    Generates a rich, length-balanced pool of legitimate URLs covering:
    - 15-25 chars (Bare homepages / short paths)
    - 25-35 chars (Medium paths, Wikipedia articles, documentation, pricing)
    - 35-50 chars (Search queries, StackOverflow, Medium, news, product pages)
    - 50-80+ chars (OAuth flows, deep APIs, checkout tokens, cloud consoles)
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
            base_domains = df_tranco['domain'].head(800).tolist()
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

    # Path templates engineered specifically to fill 25-60 character length coverage
    # Short length templates (Length approx 25 - 35 chars)
    short_paths = [
        "/about",
        "/help/faq",
        "/contact-us",
        "/pricing",
        "/login",
        "/signup",
        "/terms",
        "/privacy",
        "/features",
        "/download",
        "/security",
        "/status",
        "/support",
        "/community",
        "/developers",
        "/blog/latest"
    ]

    # Medium length templates (Length approx 35 - 50 chars)
    medium_paths = [
        "/wiki/Phishing",
        "/wiki/Computer_security",
        "/wiki/Machine_learning",
        "/wiki/Cryptography",
        "/docs/getting-started-guide",
        "/docs/v2/api-reference",
        "/questions/1234567/python-xgboost-setup",
        "/@user/cybersecurity-guide-2026",
        "/products/electronics/smart-watch",
        "/2026/09/08/technology/ai-report.html",
        "/news/security/vulnerability-updates",
        "/search?q=machine+learning+security",
        "/user/settings/security/two-factor"
    ]

    # Long length templates (Length approx 55 - 90 chars)
    long_paths = [
        "/o/oauth2/v2/auth?scope=email%20profile&response_type=code&redirect_uri=https://example.com/callback",
        "/common/oauth2/v2.0/authorize?client_id=12345678-abcd-ef01-2345-6789abcdef01",
        "/dp/B08N5WRWNW/ref=sr_1_1?dchild=1&keywords=laptop",
        "/pay/cs_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
        "/cgi-bin/webscr?cmd=_express-checkout&token=EC-1234567890ABCDEF",
        "/search?q=python+tldextract+domain+parser+tutorial&source=hp",
        "/watch?v=dQw4w9WgXcQ&feature=emb_title&ab_channel=OfficialVideo"
    ]

    idx = 0
    for dom in base_domains:
        scheme = "https://" if idx % 4 != 0 else "http://"
        sub = "www." if idx % 3 == 0 else ("docs." if idx % 7 == 0 else "")
        base_url = f"{scheme}{sub}{dom}"

        # 1. Bare homepage (1 out of 4) -> (15-25 chars)
        if idx % 4 == 0:
            urls_with_metadata.append({
                'url': base_url + "/",
                'label': 0,
                'source': 'legit_homepage',
                'registered_domain': get_registered_domain(base_url)
            })

        # 2. Short Path URL -> (25-35 chars)
        path_s = short_paths[idx % len(short_paths)]
        urls_with_metadata.append({
            'url': base_url + path_s,
            'label': 0,
            'source': 'legit_short_path',
            'registered_domain': get_registered_domain(base_url)
        })

        # 3. Medium Path URL -> (35-50 chars)
        path_m = medium_paths[idx % len(medium_paths)]
        urls_with_metadata.append({
            'url': base_url + path_m,
            'label': 0,
            'source': 'legit_medium_path',
            'registered_domain': get_registered_domain(base_url)
        })

        # 4. Long Path URL -> (55-90 chars)
        if idx % 2 == 0:
            path_l = long_paths[idx % len(long_paths)]
            urls_with_metadata.append({
                'url': base_url + path_l,
                'label': 0,
                'source': 'legit_long_path',
                'registered_domain': get_registered_domain(base_url)
            })

        idx += 1

    # Legitimate URL shorteners
    shorteners = ["https://bit.ly/3xY8zQ", "https://tinyurl.com/2p8x9z4n", "https://t.co/abc123xyz", "https://goo.gl/maps/123456"]
    for s_url in shorteners:
        urls_with_metadata.append({
            'url': s_url,
            'label': 0,
            'source': 'legit_shortener',
            'registered_domain': get_registered_domain(s_url)
        })

    return urls_with_metadata


def generate_phishing_url_pool():
    """
    Generates a rich phishing URL pool from live feeds (URLhaus, OpenPhish)
    plus diverse synthetic short, medium, and long phishing URLs.
    """
    urls_with_metadata = []

    # 1. URLhaus Live Feed
    try:
        print("--- Fetching Live Phishing Feed (URLhaus) ---")
        r = requests.get('https://urlhaus.abuse.ch/downloads/text_online/', headers=HEADERS, timeout=10)
        if r.status_code == 200:
            lines = [line.strip() for line in r.text.splitlines() if line and not line.startswith('#')]
            for u in lines[:700]:
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

    # 2. OpenPhish Live Feed
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
                    if len(urls_with_metadata) >= 1200:
                        break
            print(f"Total phishing URLs after OpenPhish: {len(urls_with_metadata)}.")
    except Exception as e:
        print(f"Warning: Could not fetch OpenPhish feed ({e}).")

    # 3. Diverse Short / IP / Typosquatting Phishing URLs
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
        "http://account-verification-chase-online.secure-login-gateway.net/auth.php",
        "http://apple-id-verify.account-security-update.com/signin",
        "http://microsoft-online-365-login.credential-harvest.org/auth/login",
        "http://login.paypal.com.user-security-check.ru/webscr?cmd=_login-run"
    ]
    for d_url in diverse_phishing:
        urls_with_metadata.append({
            'url': d_url,
            'label': 1,
            'source': 'synthetic_phishing',
            'registered_domain': get_registered_domain(d_url)
        })

    return urls_with_metadata


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
    legit_pool = generate_legitimate_url_pool()
    phish_pool = generate_phishing_url_pool()
    all_items = legit_pool + phish_pool

    print(f"\n--- Total Raw URLs Collected: {len(all_items)} (Legit: {len(legit_pool)}, Phish: {len(phish_pool)}) ---")
    print("\n--- Starting Feature Extraction for Dataset V3 (ThreadPoolExecutor max_workers=20) ---")

    valid_records = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single_url, item): item for item in all_items}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                valid_records.append(res)
            if len(valid_records) % 300 == 0:
                print(f"Progress: {len(valid_records)} URLs processed...")

    df_raw = pd.DataFrame(valid_records)

    # 1. Exact URL deduplication
    df_url_dedup = df_raw.drop_duplicates(subset=['url']).copy()
    url_dedup_removed = len(df_raw) - len(df_url_dedup)

    # 2. Feature vector deduplication
    feature_cols = [
        'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
        'subdomain_count', 'hyphen_count', 'domain_entropy',
        'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
    ]
    df_v3 = df_url_dedup.drop_duplicates(subset=feature_cols + ['label']).copy()
    feat_dedup_removed = len(df_url_dedup) - len(df_v3)

    df_v3.to_csv(output_path, index=False)

    dup_rate = ((len(df_raw) - len(df_v3)) / len(df_raw)) * 100
    legit_count = len(df_v3[df_v3['label'] == 0])
    phish_count = len(df_v3[df_v3['label'] == 1])

    print("\n======================================================================")
    print("DATASET V3 GENERATION & DEDUPLICATION SUMMARY")
    print("======================================================================")
    print(f"   Output File Path        : {output_path}")
    print(f"   Raw Processed Rows      : {len(df_raw)}")
    print(f"   After URL Deduplication  : {len(df_url_dedup)} (Removed {url_dedup_removed})")
    print(f"   After Feature Dedup     : {len(df_v3)} (Removed {feat_dedup_removed})")
    print(f"   Total Unique V3 Rows    : {len(df_v3)}")
    print(f"   Duplicate Rate          : {dup_rate:.2f}%")
    print(f"   Legitimate Samples (0)  : {legit_count}")
    print(f"   Phishing Samples (1)    : {phish_count}")
    print(f"   Unique Registered Domains: {df_v3['registered_domain'].nunique()}")
    print("======================================================================\n")

    # Report URL length distribution across required bins (Section 7)
    bins = [0, 20, 25, 30, 35, 40, 45, 50, 60, 80, 120, 1000]
    labels = ['0-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-60', '61-80', '81-120', '121+']
    df_v3['len_bin'] = pd.cut(df_v3['url_length'], bins=bins, labels=labels)

    bin_summary = pd.crosstab(df_v3['len_bin'], df_v3['label'], margins=False)
    bin_summary.columns = ['Legitimate (0)', 'Phishing (1)']
    bin_summary['Total'] = bin_summary['Legitimate (0)'] + bin_summary['Phishing (1)']
    bin_summary['Phishing Ratio'] = (bin_summary['Phishing (1)'] / bin_summary['Total'] * 100).round(2).astype(str) + '%'

    print("=== DATASET V3 URL LENGTH BIN DISTRIBUTION SUMMARY ===")
    print(bin_summary)
    print("======================================================================\n")


if __name__ == '__main__':
    main()
