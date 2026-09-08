import os
import io
import zipfile
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.feature_extraction import extract_features

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
output_path = os.path.join('data', 'dataset.csv')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) WebShield/1.0'}


def fetch_legitimate_urls(target_count=1000):
    """Fetches real legitimate domain names from Tranco Top 1M list."""
    urls = []
    print("--- Fetching Legitimate URLs (Tranco Top 1M List) ---")
    try:
        r = requests.get('https://tranco-list.eu/top-1m.csv.zip', headers=HEADERS, timeout=10)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            df_tranco = pd.read_csv(z.open(csv_name), header=None, names=['rank', 'domain'])
            domains = df_tranco['domain'].head(target_count * 2).tolist()
            for idx, d in enumerate(domains):
                scheme = 'https://' if idx % 4 != 0 else 'http://'
                urls.append(f"{scheme}{d}")
                if len(urls) >= target_count:
                    break
            print(f"Successfully fetched {len(urls)} legitimate URLs from Tranco.")
    except Exception as e:
        print(f"Warning: Could not fetch Tranco list ({e}). Using fallback list.")

    # Fallback if network call retrieved fewer than target_count
    if len(urls) < target_count:
        fallback_domains = [
            "google.com", "youtube.com", "facebook.com", "baidu.com", "wikipedia.org",
            "qq.com", "taobao.com", "yahoo.com", "tmall.com", "amazon.com",
            "twitter.com", "sohu.com", "jd.com", "live.com", "instagram.com",
            "sina.com.cn", "weibo.com", "google.co.in", "reddit.com", "vk.com",
            "blogspot.com", "yandex.ru", "bing.com", "netflix.com", "linkedin.com",
            "microsoft.com", "ebay.com", "alipay.com", "yahoo.co.jp", "twitch.tv",
            "ok.ru", "github.com", "apple.com", "naver.com", "adobe.com",
            "wordpress.com", "chase.com", "wellsfargo.com", "bankofamerica.com", "paypal.com"
        ]
        i = 0
        while len(urls) < target_count:
            base = fallback_domains[i % len(fallback_domains)]
            sub = f"sub{i}." if i >= len(fallback_domains) else ""
            scheme = "https://" if i % 2 == 0 else "http://"
            urls.append(f"{scheme}{sub}{base}")
            i += 1

    return urls[:target_count]


def fetch_phishing_urls(target_count=1000):
    """Fetches real verified online phishing URLs from URLhaus and OpenPhish."""
    urls = []
    print("--- Fetching Phishing URLs (URLhaus & OpenPhish feeds) ---")
    
    # Source 1: URLhaus
    try:
        r = requests.get('https://urlhaus.abuse.ch/downloads/text_online/', headers=HEADERS, timeout=10)
        if r.status_code == 200:
            lines = [line.strip() for line in r.text.splitlines() if line and not line.startswith('#')]
            for u in lines:
                if u.startswith(('http://', 'https://')):
                    urls.append(u)
                    if len(urls) >= target_count:
                        break
            print(f"Fetched {len(urls)} phishing URLs from URLhaus.")
    except Exception as e:
        print(f"Warning: Could not fetch URLhaus feed ({e}).")

    # Source 2: OpenPhish (if needed)
    if len(urls) < target_count:
        try:
            r = requests.get('https://openphish.com/feed.txt', headers=HEADERS, timeout=10)
            if r.status_code == 200:
                lines = [line.strip() for line in r.text.splitlines() if line and not line.startswith('#')]
                for u in lines:
                    if u.startswith(('http://', 'https://')) and u not in urls:
                        urls.append(u)
                        if len(urls) >= target_count:
                            break
                print(f"Total phishing URLs after OpenPhish: {len(urls)}.")
        except Exception as e:
            print(f"Warning: Could not fetch OpenPhish feed ({e}).")

    # Fallback if network feeds retrieve fewer than target_count
    if len(urls) < target_count:
        fallback_phishing = [
            "http://192.168.1.1/login-verify-account/",
            "http://secure-update-paypal.com.verify-user-account.info/login.php",
            "http://banking-secure-auth.net/update-account/login.html",
            "http://verify-microsoft-security-alert.org/signin/",
            "http://chase-bank-online-security-update.com/login",
            "http://wellsfargo-account-verify-alert.net/secure/",
            "http://appleid-verify-account-info.com/login.php",
            "http://netflix-billing-update-account.org/signin"
        ]
        i = 0
        while len(urls) < target_count:
            base = fallback_phishing[i % len(fallback_phishing)]
            urls.append(f"{base}?session={i}&token=abc{i}")
            i += 1

    return urls[:target_count]


def process_url(url, label):
    """Safely extracts features for a single URL."""
    try:
        feat = extract_features(url)
        feat['label'] = label
        return feat
    except Exception:
        return None


def main():
    legit_urls = fetch_legitimate_urls(target_count=1000)
    phish_urls = fetch_phishing_urls(target_count=1000)

    url_pairs = [(u, 0) for u in legit_urls] + [(u, 1) for u in phish_urls]
    total_urls = len(url_pairs)
    print(f"\n--- Starting Parallel Feature Extraction for {total_urls} URLs (ThreadPoolExecutor max_workers=20) ---")

    dataset = []
    completed = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_url, url, label): (url, label) for url, label in url_pairs}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                dataset.append(res)
            completed += 1
            if completed % 200 == 0 or completed == total_urls:
                print(f"Progress: {completed}/{total_urls} URLs processed ({len(dataset)} valid records)...")

    df = pd.DataFrame(dataset)
    df.to_csv(output_path, index=False)

    print("\n==================================================")
    print("DATASET GENERATION SUMMARY")
    print("==================================================")
    print(f"   Output File Path : {output_path}")
    print(f"   Total Rows Saved : {len(df)}")
    print(f"   Class Balance    : Label 0 (Legitimate) = {(df['label'] == 0).sum()}, Label 1 (Phishing) = {(df['label'] == 1).sum()}")
    print(f"   Feature Columns  : {list(df.columns)}")
    print("==================================================\n")

    print("Dataset Sample Head:")
    print(df.head())


if __name__ == '__main__':
    main()