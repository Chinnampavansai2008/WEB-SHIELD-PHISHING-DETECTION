"""
Homoglyph & Brand Impersonation Detection Module.
Detects Punycode hostnames, mixed unicode scripts, and brand impersonation attacks.
"""

import unicodedata
import tldextract
from typing import Dict, Optional, List, Any


TOP_BRANDS: List[str] = ['google', 'paypal', 'microsoft', 'apple', 'amazon', 'netflix']


def _is_punycode(hostname: str) -> bool:
    """Checks if hostname contains any punycode label (xn--)."""
    if not hostname:
        return False
    parts = hostname.lower().split(".")
    return any(p.startswith("xn--") for p in parts)


def _has_mixed_script(hostname: str) -> bool:
    """
    Checks if hostname contains characters from multiple distinct Unicode scripts
    (e.g., mixing Latin with Cyrillic or Greek).
    """
    if not hostname:
        return False

    decoded = hostname
    if _is_punycode(hostname):
        try:
            decoded = hostname.encode("ascii").decode("idna")
        except Exception:
            decoded = hostname

    scripts = set()
    for char in decoded:
        if not char.isalpha():
            continue
        try:
            script_name = unicodedata.name(char).split()[0]
            if script_name in {"LATIN", "CYRILLIC", "GREEK", "HEBREW", "ARABIC", "ARMENIAN", "GEORGIAN", "THAI"}:
                scripts.add(script_name)
        except (ValueError, KeyError):
            pass

    return len(scripts) > 1


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _get_registered_domain_sld(hostname: str) -> str:
    ext = tldextract.extract(hostname)
    return (ext.domain or "").lower()


def _detect_suspected_brand(hostname: str, brands: List[str] = TOP_BRANDS) -> Optional[str]:
    """
    Checks hostname tokens against top brands for typo-squatting or brand impersonation.
    Legitimate brand subdomains (e.g. google.com, www.google.com, accounts.google.com)
    are recognized as safe brand owners and will NOT be flagged as impersonators.
    """
    if not hostname:
        return None

    decoded = hostname.lower()
    if _is_punycode(hostname):
        try:
            decoded = hostname.encode("ascii").decode("idna").lower()
        except Exception:
            decoded = hostname.lower()

    reg_sld = _get_registered_domain_sld(decoded)
    labels = [l for l in decoded.split(".") if l]
    if not labels:
        return None

    for brand in brands:
        # Legitimate brand owner check (e.g. google.com, www.google.com, mail.google.com)
        if reg_sld == brand:
            continue

        # Check sub-tokens for brand impersonation or typosquatting on third-party domains
        domain_labels = labels[:-1] if len(labels) >= 2 else labels
        all_tokens = []
        for label in domain_labels:
            all_tokens.extend([t for t in label.split("-") if t])

        for token in all_tokens:
            if token == brand:
                return brand

            if brand in token:
                return brand

            dist = _levenshtein_distance(token, brand)
            max_dist = 1 if len(brand) <= 5 else 2
            if 0 < dist <= max_dist:
                return brand

    return None




def analyze_homoglyphs(hostname: str) -> Dict[str, Any]:
    """
    Analyzes a hostname for punycode, mixed unicode scripts, and brand impersonation.

    Args:
        hostname (str): Domain / hostname to analyze.

    Returns:
        dict: {
            'is_punycode': bool,
            'mixed_script': bool,
            'suspected_brand': Optional[str],
            'impersonation_risk': 'low' | 'high'
        }
    """
    if not hostname or not isinstance(hostname, str):
        return {
            'is_punycode': False,
            'mixed_script': False,
            'suspected_brand': None,
            'impersonation_risk': 'low'
        }

    is_puny = _is_punycode(hostname)
    mixed = _has_mixed_script(hostname)
    suspected = _detect_suspected_brand(hostname)

    is_high = is_puny or mixed or (suspected is not None)
    risk_level = 'high' if is_high else 'low'

    return {
        'is_punycode': is_puny,
        'mixed_script': mixed,
        'suspected_brand': suspected,
        'impersonation_risk': risk_level
    }
