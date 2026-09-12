"""
Enhanced Feature Extraction Module for V4-J / V5 Experimental Candidates.
Implements canonical URL parsing with enhanced structural deception signals:
- Hostname structure & length signals
- Context-aware brand mismatch signals
- Shared hosting & cloud platform context signals
- Query string structural signals
"""

import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlparse
import tldextract

_OFFLINE_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

from generate_data_v4 import calculate_shannon_entropy


BRAND_TOKENS = {
    'paypal', 'microsoft', 'google', 'apple', 'chase', 'wellsfargo',
    'bankofamerica', 'netflix', 'amazon', 'stripe', 'facebook', 'instagram',
    'twitter', 'linkedin', 'github', 'dropbox', 'spotify', 'adobe', 'salesforce'
}

SHARED_HOSTING_PATTERNS = [
    'workers.dev', 'pages.dev', 'r2.dev', 'github.io', 'netlify.app',
    'vercel.app', 'web.app', 'firebaseapp.com', 'storage.googleapis.com',
    's3.amazonaws.com', 'blob.core.windows.net', '000webhostapp.com',
    'weebly.com', 'wixsite.com', 'wordpress.com', 'site123.me'
]


def extract_features_v4j(url: str) -> dict:
    raw_url = url.strip() if url else ""
    if not raw_url.startswith(('http://', 'https://')):
        url_norm = 'http://' + raw_url
    else:
        url_norm = raw_url

    parsed_url = urlparse(url_norm)
    hostname = parsed_url.netloc.split(':')[0] if parsed_url.netloc else ""
    path = parsed_url.path or ""
    query = parsed_url.query or ""
    extracted = _OFFLINE_EXTRACTOR(url_norm)

    url_length = len(raw_url)
    hostname_length = len(hostname)
    path_length = len(path)
    path_depth = len([p for p in path.split('/') if p])

    try:
        ipaddress.ip_address(hostname)
        having_ip = 1
    except ValueError:
        having_ip = 0

    has_at_symbol = 1 if '@' in raw_url else 0

    proto_idx = url_norm.find('://')
    if proto_idx != -1:
        remainder = url_norm[proto_idx + 3:]
        redirect_count = remainder.count('//')
    else:
        redirect_count = 0

    subdomain_str = extracted.subdomain or ""
    if subdomain_str:
        subdomain_count = len([part for part in subdomain_str.split('.') if part])
    else:
        subdomain_count = 0

    domain_str = extracted.domain if extracted.domain else hostname
    hyphen_count = domain_str.count('-')

    registered_domain = extracted.top_domain_under_public_suffix if getattr(extracted, 'top_domain_under_public_suffix', None) else (extracted.domain if extracted.domain else domain_str)
    domain_entropy = calculate_shannon_entropy(registered_domain)
    hostname_entropy = calculate_shannon_entropy(hostname)

    suspicious_tokens = {'login', 'verify', 'update', 'banking', 'secure', 'signin', 'account', 'auth', 'portal', 'payment', 'confirm'}
    
    hostname_lower = hostname.lower()
    keyword_in_hostname = 1 if any(token in hostname_lower for token in suspicious_tokens) else 0

    path_query_lower = (path + " " + query).lower()
    keyword_in_path = 1 if any(token in path_query_lower for token in suspicious_tokens) else 0

    has_suspicious_keyword = 1 if (keyword_in_hostname or keyword_in_path) else 0

    ssl_valid = 1 if url_norm.startswith('https://') else 0
    domain_age_months = -1

    # --- ENHANCED V4-J STRUCTURAL FEATURES ---
    subdomain_length = len(subdomain_str)
    digits = sum(c.isdigit() for c in hostname)
    numeric_ratio = round(digits / len(hostname), 4) if len(hostname) > 0 else 0.0

    registrable_domain_length = len(registered_domain)
    subdomain_depth = subdomain_count
    subdomain_tokens = [t for t in subdomain_str.split('.') if t]
    longest_subdomain_token_length = max([len(t) for t in subdomain_tokens], default=0)

    hyphens_in_hostname = hostname.count('-')
    hostname_hyphen_ratio = round(hyphens_in_hostname / len(hostname), 4) if len(hostname) > 0 else 0.0
    hostname_token_count = len([t for t in hostname.replace('-', '.').split('.') if t])

    # Context-Aware Brand Mismatch Signals
    reg_dom_lower = registered_domain.lower()
    sub_lower = subdomain_str.lower()
    path_lower = path.lower()

    brand_in_reg = 1 if any(b in reg_dom_lower for b in BRAND_TOKENS) else 0
    brand_in_sub = 1 if any(b in sub_lower for b in BRAND_TOKENS) else 0
    brand_in_path = 1 if any(b in path_lower for b in BRAND_TOKENS) else 0

    # Brand Mismatch: Brand token present in subdomain or path BUT NOT in actual registrable domain
    brand_domain_mismatch = 1 if ((brand_in_sub or brand_in_path) and not brand_in_reg) else 0

    # Shared Hosting Platform Context Signal
    is_shared_hosting_platform = 1 if any(p in hostname_lower for p in SHARED_HOSTING_PATTERNS) else 0

    # Query String Structural Signals
    query_length = len(query)

    return {
        'url_length': url_length,
        'hostname_length': hostname_length,
        'path_length': path_length,
        'path_depth': path_depth,
        'having_ip': having_ip,
        'has_at_symbol': has_at_symbol,
        'redirect_count': redirect_count,
        'subdomain_count': subdomain_count,
        'hyphen_count': hyphen_count,
        'domain_entropy': domain_entropy,
        'hostname_entropy': hostname_entropy,
        'keyword_in_hostname': keyword_in_hostname,
        'keyword_in_path': keyword_in_path,
        'has_suspicious_keyword': has_suspicious_keyword,
        'ssl_valid': ssl_valid,
        'domain_age_months_clean': 0 if domain_age_months == -1 else domain_age_months,
        'domain_age_known': 0 if domain_age_months == -1 else 1,
        'subdomain_length': subdomain_length,
        'numeric_ratio': numeric_ratio,
        'brand_token_in_subdomain': brand_in_sub,
        'registrable_domain_length': registrable_domain_length,
        'subdomain_depth': subdomain_depth,
        'longest_subdomain_token_length': longest_subdomain_token_length,
        'hostname_hyphen_ratio': hostname_hyphen_ratio,
        'hostname_token_count': hostname_token_count,
        'brand_token_in_registered_domain': brand_in_reg,
        'brand_token_in_path': brand_in_path,
        'brand_domain_mismatch': brand_domain_mismatch,
        'is_shared_hosting_platform': is_shared_hosting_platform,
        'query_length': query_length
    }
