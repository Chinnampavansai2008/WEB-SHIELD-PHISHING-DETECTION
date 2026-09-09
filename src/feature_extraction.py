import math
import re
import socket
import ssl
import ipaddress
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse
import tldextract  # type: ignore
import whois  # type: ignore


def calculate_shannon_entropy(text: str) -> float:
    """Calculates Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())
    return round(entropy, 4)


def verify_ssl(hostname: str, port: int = 443, timeout: float = 2.0) -> int:
    """Performs a strict socket + TLS handshake check with timeout."""
    if not hostname:
        return 0
    try:
        context = ssl.create_default_context()
        try:
            ipaddress.ip_address(hostname)
            context.check_hostname = False
        except ValueError:
            pass

        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            server_hostname = hostname if context.check_hostname else None
            with context.wrap_socket(sock, server_hostname=server_hostname) as ssock:
                return 1
    except Exception:
        return 0


def get_domain_age_months(domain: str, timeout: float = 3.0) -> int:
    """Queries WHOIS data to calculate domain age in months, defaulting to -1 on failure/timeout."""
    if not domain:
        return -1
    try:
        ipaddress.ip_address(domain)
        return -1
    except ValueError:
        pass

    try:
        socket.setdefaulttimeout(timeout)
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if not creation_date or not isinstance(creation_date, datetime):
            return -1
        now = datetime.now()
        age_days = (now - creation_date).days
        if age_days < 0:
            return -1
        return max(0, age_days // 30)
    except Exception:
        return -1


NEW_DOMAIN_THRESHOLD_MONTHS = 6


def get_domain_age_status(domain_age_months: int) -> str:
    """
    Returns explicit domain age status string:
    - 'unknown': WHOIS lookup failed, unverified, or IP address (-1)
    - 'new': domain age < NEW_DOMAIN_THRESHOLD_MONTHS (0 to 5 months)
    - 'known': domain age >= NEW_DOMAIN_THRESHOLD_MONTHS (6+ months)
    """
    if domain_age_months is None or domain_age_months == -1:
        return "unknown"
    elif domain_age_months < NEW_DOMAIN_THRESHOLD_MONTHS:
        return "new"
    else:
        return "known"


def extract_features(url: str) -> dict:
    """
    Extracts 10 robust ML features from an input URL string.
    Returns a dictionary of feature names and values.
    """
    raw_url = url.strip() if url else ""
    if not raw_url.startswith(('http://', 'https://')):
        url_norm = 'http://' + raw_url
    else:
        url_norm = raw_url

    parsed_url = urlparse(url_norm)
    hostname = parsed_url.netloc.split(':')[0] if parsed_url.netloc else ""
    extracted = tldextract.extract(url_norm)

    # 1. URL Length
    url_length = len(raw_url)

    # 2. IP Address presence in hostname (IPv4 / IPv6)
    try:
        ipaddress.ip_address(hostname)
        having_ip = 1
    except ValueError:
        having_ip = 0

    # 3. '@' Symbol check
    has_at_symbol = 1 if '@' in raw_url else 0

    # 4. Redirect Count ('//' occurrences after protocol)
    proto_idx = url_norm.find('://')
    if proto_idx != -1:
        remainder = url_norm[proto_idx + 3:]
        redirect_count = remainder.count('//')
    else:
        redirect_count = 0

    # 5. Subdomain Count via tldextract
    subdomain_str = extracted.subdomain
    if subdomain_str:
        subdomain_count = len([part for part in subdomain_str.split('.') if part])
    else:
        subdomain_count = 0

    # 6. Hyphen Count in domain string
    domain_str = extracted.domain if extracted.domain else hostname
    hyphen_count = domain_str.count('-')

    # 7. Domain Entropy (Shannon entropy of registered domain)
    registered_domain = extracted.top_domain_under_public_suffix if getattr(extracted, 'top_domain_under_public_suffix', None) else (extracted.domain if extracted.domain else domain_str)
    domain_entropy = calculate_shannon_entropy(registered_domain)

    # 8. Suspicious Keywords check in path and subdomain
    suspicious_tokens = {'login', 'verify', 'update', 'banking', 'secure', 'signin', 'account'}
    path_subdomain_text = (parsed_url.path + " " + extracted.subdomain).lower()
    has_suspicious_keyword = 1 if any(token in path_subdomain_text for token in suspicious_tokens) else 0

    # 9. SSL Validity check (strict 2.0s socket TLS handshake)
    ssl_valid = verify_ssl(hostname) if hostname else 0

    # 10. Domain Age in Months via WHOIS query
    domain_age_months = get_domain_age_months(registered_domain)

    return {
        'url_length': url_length,
        'having_ip': having_ip,
        'has_at_symbol': has_at_symbol,
        'redirect_count': redirect_count,
        'subdomain_count': subdomain_count,
        'hyphen_count': hyphen_count,
        'domain_entropy': domain_entropy,
        'has_suspicious_keyword': has_suspicious_keyword,
        'ssl_valid': ssl_valid,
        'domain_age_months': domain_age_months
    }


if __name__ == '__main__':
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login-verify-account/"
    ]
    for test_url in test_urls:
        print(f"\n--- Extracting features for: {test_url} ---")
        features = extract_features(test_url)
        print(features)