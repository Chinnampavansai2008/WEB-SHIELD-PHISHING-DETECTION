"""
IOC (Indicator of Compromise) Generator Module.
Formats deep forensic analysis results into a standardized STIX/JSON IOC object.
"""

import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Dict, Any, List, Set
from src.defang import defang_url, defang_ip


def generate_ioc(target_url: str, forensic_report: Dict[str, Any], shap_risk_factors: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generates a standardized IOC dictionary from deep forensic scan results.

    Args:
        target_url (str): Original target URL scanned.
        forensic_report (dict): Forensic results dictionary from deep_analysis.
        shap_risk_factors (list): Optional list of top SHAP risk factors.

    Returns:
        dict: Standardized IOC schema dictionary.
    """
    scan_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    domains: Set[str] = set()
    ips: Set[str] = set()
    external_destinations: List[Dict[str, str]] = []

    parsed_target = urlparse(target_url)
    if parsed_target.hostname:
        domains.add(parsed_target.hostname.lower())

    for hop in forensic_report.get('redirect_timeline', []):
        raw_url = hop.get('url', '')
        if raw_url:
            clean_host = raw_url.replace("hxxps[://]", "https://").replace("hxxp[://]", "http://").replace("[.]", ".")
            p = urlparse(clean_host)
            if p.hostname:
                domains.add(p.hostname.lower())

    cred_analysis = forensic_report.get('credential_analysis', {})
    for action in cred_analysis.get('form_actions', []):
        act_url = action.get('action_url', '')
        dest_type = action.get('destination_type', '')
        if act_url and dest_type not in ("SAME_ORIGIN", "SAME_REGISTERED_DOMAIN"):
            external_destinations.append({
                'action_url': defang_url(act_url),
                'type': dest_type
            })

    defanged_domains = [d.replace(".", "[.]") for d in sorted(list(domains))]
    defanged_ips = [defang_ip(ip) for ip in sorted(list(ips))]

    return {
        'schema_version': '1.0',
        'scan_id': scan_id,
        'timestamp': timestamp,
        'target': {
            'original_url': target_url,
            'defanged_url': defang_url(target_url)
        },
        'indicators': {
            'domains': defanged_domains,
            'ips': defanged_ips,
            'external_form_destinations': external_destinations,
            'redirect_hops': forensic_report.get('redirect_count', 0)
        },
        'forensics': {
            'title': forensic_report.get('title', ''),
            'meta_description': forensic_report.get('meta_description', ''),
            'redirect_timeline': forensic_report.get('redirect_timeline', []),
            'credential_audit': cred_analysis,
            'homoglyph_analysis': forensic_report.get('homoglyph_analysis', {}),
            'top_risk_factors': shap_risk_factors or [],
            'scan_duration_seconds': forensic_report.get('scan_duration_seconds', 0.0),
            'deadline_exceeded': forensic_report.get('deadline_exceeded', False)
        }
    }
