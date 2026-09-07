import re
from urllib.parse import urlparse

def extract_features(url):
    """
    Extracts numerical ML features from an input URL string.
    Matches the feature columns used during model training.
    """
    if not url.startswith(('http://', 'https://')):
        parse_target = 'http://' + url
    else:
        parse_target = url
        
    parsed_url = urlparse(parse_target)
    
    # 1. URL Length
    url_length = len(url)
    
    # 2. IP Address presence in domain
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    having_ip = 1 if re.search(ip_pattern, url) else 0
    
    # 3. '@' Symbol check
    has_at_symbol = 1 if '@' in url else 0
    
    # 4. SSL / HTTPS check
    ssl_valid = 1 if url.startswith('https://') else 0
    
    # 5. Estimated Domain Age
    domain_age_months = 24 if ssl_valid == 1 else 2
    
    # 6. Subdomain Count
    hostname = parsed_url.netloc
    parts = hostname.split('.')
    subdomain_count = max(0, len(parts) - 2)
    
    # 7. Redirect Count
    redirect_count = url.count('//') - 1 if url.count('//') > 1 else 0

    return [
        url_length,
        having_ip,
        has_at_symbol,
        ssl_valid,
        domain_age_months,
        subdomain_count,
        redirect_count
    ]