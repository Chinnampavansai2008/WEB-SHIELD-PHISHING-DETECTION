"""
Hybrid Tier 1 + Tier 2 Triage & Evidence Integration Module.
Implements bounded Tier 2 routing, evidence severity categorization, and combined risk verdicts.
"""

from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import tldextract

_OFFLINE_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

AUTH_TOKENS = {
    'login', 'signin', 'oauth', 'auth', 'sso', 'account', 'verify',
    'security', 'portal', 'payment', 'confirm', 'password', 'credential'
}

SHARED_HOSTING_DOMAINS = [
    'workers.dev', 'pages.dev', 'r2.dev', 'github.io', 'netlify.app',
    'vercel.app', 'web.app', 'firebaseapp.com', 'storage.googleapis.com',
    's3.amazonaws.com', 'blob.core.windows.net', '000webhostapp.com',
    'weebly.com', 'wixsite.com', 'wordpress.com', 'site123.me'
]


def check_auth_context(url: str) -> bool:
    """Checks whether URL path, query, or subdomain contains authentication-related terms."""
    if not url:
        return False
    u_lower = url.lower()
    parsed = urlparse(u_lower if u_lower.startswith(('http://', 'https://')) else 'http://' + u_lower)
    
    extracted = _OFFLINE_EXTRACTOR(url)
    text_to_check = f"{extracted.subdomain} {parsed.path} {parsed.query}"
    return any(token in text_to_check for token in AUTH_TOKENS)


ROUTER_D_CONFIG = {
    'ambiguous_min': 0.35,
    'ambiguous_max': 0.65,
    'auth_weak_min': 0.15,
    'auth_weak_max': 0.35
}

ROUTER_E_CONFIG = {
    'ambiguous_min': 0.40,
    'ambiguous_max': 0.60,
    'auth_weak_min': 0.25,
    'auth_weak_max': 0.35
}


def route_for_tier2(
    ml_probability: float,
    tier1_features: Dict[str, Any],
    url: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    router_version: str = 'E'
) -> Dict[str, Any]:
    """
    Determines whether Tier 2 static forensic analysis is required based on:
    1. ML probability ambiguity band (0.40 <= P <= 0.60 for Router-E, 0.35 <= P <= 0.65 for Router-D).
    2. Targeted HTTPS custom-domain auth escalation (0.25 <= P < 0.35 for Router-E, 0.15 <= P < 0.35 for Router-D).
    3. Shared-hosting platform context.
    4. Contextual brand-domain mismatch.
    5. IP address hostname.
    """
    reasons = []
    url_str = url or (tier1_features.get('url') if isinstance(tier1_features, dict) else "")
    
    cfg = ROUTER_E_CONFIG if str(router_version).upper() == 'E' else ROUTER_D_CONFIG
    
    # 1. Ambiguous ML probability band
    if cfg['ambiguous_min'] <= ml_probability <= cfg['ambiguous_max']:
        reasons.append("ML probability is in ambiguous range.")

    has_auth = check_auth_context(url_str)
    having_ip = tier1_features.get('having_ip', 0) if isinstance(tier1_features, dict) else 0
    is_shared = (any(sh in url_str.lower() for sh in SHARED_HOSTING_DOMAINS) or 
                 (isinstance(tier1_features, dict) and tier1_features.get('is_shared_hosting_platform', 0) == 1))
    brand_mismatch = (isinstance(tier1_features, dict) and tier1_features.get('brand_domain_mismatch', 0) == 1)

    # 2. IP Hostname Escalation
    if having_ip:
        reasons.append("IP address used as hostname requires forensic inspection.")

    # 3. Shared-hosting authentication page
    if is_shared and (has_auth or ml_probability >= 0.20):
        reasons.append("Shared-hosting authentication page requires contextual evidence.")

    # 4. Contextual Brand-Domain Mismatch Escalation
    if brand_mismatch and (has_auth or ml_probability >= 0.15):
        reasons.append("Brand-domain mismatch with authentication context requires forensic verification.")

    # 5. Targeted HTTPS Custom-Domain Auth Escalation
    if has_auth and (cfg['auth_weak_min'] <= ml_probability < cfg['auth_weak_max']):
        reasons.append("HTTPS custom-domain authentication URL matches known Tier-1 weak class.")

    # Deduplicate reasons while preserving order
    seen = set()
    unique_reasons = [r for r in reasons if not (r in seen or seen.add(r))]

    tier2_required = len(unique_reasons) > 0
    primary_reason = unique_reasons[0] if unique_reasons else "High-confidence Tier-1 classification."

    return {
        "tier2_required": tier2_required,
        "routing_reason": primary_reason,
        "routing_reasons": unique_reasons
    }


def evaluate_tier2_evidence_severity(
    tier2_report: Dict[str, Any],
    severity_variant: str = 'C',
    url: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Evaluates severity of static forensic evidence from Tier 2 report supporting variants:
    - Variant A: Legacy baseline logic.
    - Variant B: Registered-domain-aware form logic (SAME_REGISTERED_DOMAIN is MEDIUM, not HIGH).
    - Variant C: Registered-domain-aware + federated auth protocol context.
    """
    evidence = []
    if not tier2_report or tier2_report.get('scan_status') == 'failed':
        return "UNKNOWN", ["Tier 2 analysis unavailable or fetch failed."]

    cred_analysis = tier2_report.get('credential_analysis', {})
    
    sensitive_fields = cred_analysis.get('sensitive_fields', [])
    has_pwd = ('password' in sensitive_fields) or cred_analysis.get('has_login_form', False) or tier2_report.get('has_password_field', False)
    
    form_actions_info = cred_analysis.get('form_actions', [])
    ext_actions = tier2_report.get('external_form_actions', [])
    
    # Analyze form action destination relationships
    external_reg_actions = []
    same_reg_actions = []
    raw_ip_actions = []
    webhook_actions = []
    
    for fa in form_actions_info:
        dest_t = fa.get('destination_type', '')
        act_url = fa.get('action_url', '')
        if dest_t == 'EXTERNAL_REGISTERED_DOMAIN' or dest_t == 'UNKNOWN_EXTERNAL':
            external_reg_actions.append(act_url)
        elif dest_t == 'SAME_REGISTERED_DOMAIN':
            same_reg_actions.append(act_url)
        elif dest_t == 'RAW_IP':
            raw_ip_actions.append(act_url)
        elif dest_t == 'WEBHOOK':
            webhook_actions.append(act_url)

    if not ext_actions and external_reg_actions:
        ext_actions = external_reg_actions
    if not external_reg_actions and ext_actions:
        external_reg_actions = ext_actions

    insecure_action = cred_analysis.get('insecure_form_action', False) or tier2_report.get('insecure_form_action', False)
    
    # Check HTTPS page submitting over HTTP
    final_url = tier2_report.get('final_url', '') or url or ''
    if final_url and final_url.startswith('https://'):
        for fa in form_actions_info:
            act_url = fa.get('action_url', '')
            if act_url.startswith('http://'):
                insecure_action = True

    # Federated Auth Protocol Context Detection
    from src.credential_analysis import check_federated_auth_context
    has_fed_auth = check_federated_auth_context(url or "") or check_federated_auth_context(final_url)

    if severity_variant == 'A':
        # Baseline rules
        unrelated_action = (cred_analysis.get('exfiltration_flag') is True) or len(external_reg_actions) > 0 or len(webhook_actions) > 0
        if has_pwd and len(ext_actions) > 0:
            evidence.append("Password input field submits to external cross-origin domain.")
        if insecure_action and has_pwd:
            evidence.append("HTTPS authentication page submits credentials over unencrypted HTTP.")
        if unrelated_action and has_pwd:
            evidence.append("Credential form posts to completely unrelated third-party domain.")

        if evidence:
            return "HIGH", evidence

        if has_pwd:
            evidence.append("Password input form detected on page.")
        if len(ext_actions) > 0:
            evidence.append(f"External form action detected target domain(s): {', '.join(ext_actions)}.")
        
        timeline = tier2_report.get('redirect_timeline', [])
        actual_redirects = tier2_report.get('redirect_count_actual') or tier2_report.get('redirect_count') or (len(timeline) - 1 if len(timeline) > 1 else 0)
        if actual_redirects > 3:
            evidence.append(f"Suspicious redirect chain detected ({actual_redirects} hops).")

        if evidence:
            return "MEDIUM", evidence
        return "LOW", ["No suspicious static forensic evidence detected."]

    # Variant B & Variant C: Registered-Domain Aware Logic
    # HIGH RISK CONDITIONS
    if raw_ip_actions and has_pwd:
        evidence.append("Credential form submits to raw IP address.")
    if webhook_actions and has_pwd:
        evidence.append("Credential form submits to known exfiltration webhook.")
    if insecure_action and has_pwd:
        evidence.append("HTTPS authentication page submits credentials over unencrypted HTTP.")
    if has_pwd and len(external_reg_actions) > 0:
        if severity_variant == 'C' and has_fed_auth and not (raw_ip_actions or webhook_actions or insecure_action):
            # Federated auth context: external form action is expected in OAuth/SAML redirect flows
            pass
        else:
            evidence.append(f"Password input field submits to external cross-origin domain: {', '.join(external_reg_actions[:2])}.")

    if evidence:
        return "HIGH", evidence

    # MEDIUM RISK CONDITIONS
    if has_pwd and len(external_reg_actions) > 0 and severity_variant == 'C' and has_fed_auth:
        evidence.append(f"Federated authentication form target detected: {', '.join(external_reg_actions[:2])}.")
    if has_pwd and len(same_reg_actions) > 0:
        evidence.append("Password input field submits to sub-domain within same registered domain.")
    elif has_pwd:
        evidence.append("Password input form detected on page.")

    if len(external_reg_actions) > 0 and not has_pwd:
        evidence.append(f"Form action targets external registered domain: {', '.join(external_reg_actions[:2])}.")

    timeline = tier2_report.get('redirect_timeline', [])
    actual_redirects = tier2_report.get('redirect_count_actual') or tier2_report.get('redirect_count') or (len(timeline) - 1 if len(timeline) > 1 else 0)
    if actual_redirects > 3:
        if has_fed_auth:
            evidence.append(f"Federated auth redirect sequence detected ({actual_redirects} hops).")
        else:
            evidence.append(f"Multi-hop redirect chain detected ({actual_redirects} hops).")

    if evidence:
        return "MEDIUM", evidence

    return "LOW", ["No suspicious static forensic evidence detected."]


def compute_hybrid_verdict(
    ml_probability: float,
    ml_is_phishing: bool,
    tier1_features: Dict[str, Any],
    tier2_report: Optional[Dict[str, Any]] = None,
    url: Optional[str] = None,
    severity_variant: str = 'C'
) -> Dict[str, Any]:
    """
    Computes overall evidence-based security verdict combining Tier 1 ML and Tier 2 forensics.
    Supports confidence gating and severity variants (A, B, C).
    Preserves ML probability strictly unfloored and unmodified.
    """
    routing = route_for_tier2(ml_probability, tier1_features, url=url)
    
    # If Tier 2 failed or not executed
    if not tier2_report or tier2_report.get('scan_status') == 'failed':
        if routing['tier2_required']:
            reason_str = "Tier 2 analysis unavailable; verdict based on Tier 1 only."
            scan_st = tier2_report.get('scan_status', 'failed') if tier2_report else "failed"
            analysis_conf = "limited"
            if isinstance(tier1_features, dict) and tier1_features.get('having_ip', 0) == 1:
                risk_verdict = "CRITICAL"
                overall_verdict = "Phishing"
            elif ml_probability >= 0.75:
                risk_verdict = "CRITICAL"
                overall_verdict = "Phishing"
            elif ml_probability >= 0.50:
                risk_verdict = "SUSPICIOUS"
                overall_verdict = "Phishing"
            else:
                risk_verdict = "UNKNOWN"
                overall_verdict = "Legitimate"
        else:
            reason_str = "High-confidence Tier 1 ML assessment."
            scan_st = "not_executed"
            analysis_conf = "high"
            if isinstance(tier1_features, dict) and tier1_features.get('having_ip', 0) == 1:
                risk_verdict = "CRITICAL"
                overall_verdict = "Phishing"
            elif ml_probability >= 0.75:
                risk_verdict = "CRITICAL"
                overall_verdict = "Phishing"
            elif ml_probability >= 0.50:
                risk_verdict = "SUSPICIOUS"
                overall_verdict = "Phishing"
            elif isinstance(tier1_features, dict) and 0 <= tier1_features.get('domain_age_months', -1) < 6:
                risk_verdict = "SUSPICIOUS"
                overall_verdict = "Legitimate"
            elif ml_probability <= 0.30:
                risk_verdict = "SAFE"
                overall_verdict = "Legitimate"
            else:
                risk_verdict = "SAFE"
                overall_verdict = "Legitimate"

        return {
            "ml_probability": float(ml_probability),
            "ml_is_phishing": bool(ml_is_phishing),
            "ml_prediction": "Phishing" if ml_is_phishing else "Legitimate",
            "risk_verdict": risk_verdict,
            "overall_verdict": overall_verdict,
            "tier2_required": routing["tier2_required"],
            "routing_reason": routing["routing_reason"],
            "routing_reasons": routing["routing_reasons"],
            "scan_status": scan_st,
            "analysis_confidence": analysis_conf,
            "forensic_evidence_severity": "UNKNOWN",
            "forensic_evidence": [reason_str]
        }

    severity, evidence_list = evaluate_tier2_evidence_severity(tier2_report, severity_variant=severity_variant, url=url)
    scan_st = tier2_report.get('scan_status', 'completed')
    analysis_conf = tier2_report.get('analysis_confidence', 'high').lower()
    
    # Evidence-based Decision Combination Logic with Confidence Gating
    if severity == "HIGH":
        if ml_probability >= 0.15 or ml_is_phishing:
            risk_verdict = "CRITICAL"
            overall_verdict = "Phishing"
        else:
            risk_verdict = "SUSPICIOUS"
            overall_verdict = "Phishing"
    elif severity == "MEDIUM":
        if ml_probability >= 0.50 or ml_is_phishing:
            risk_verdict = "CRITICAL"
            overall_verdict = "Phishing"
        elif ml_probability >= 0.35:
            risk_verdict = "SUSPICIOUS"
            overall_verdict = "Legitimate"
        else:
            # Confidence Gating: Low ML prob (< 0.35) + MEDIUM evidence -> SUSPICIOUS, Legitimate
            risk_verdict = "SUSPICIOUS"
            overall_verdict = "Legitimate"
    else:  # LOW severity / No suspicious evidence
        if ml_probability >= 0.75:
            risk_verdict = "CRITICAL"
            overall_verdict = "Phishing"
        elif ml_probability >= 0.50:
            risk_verdict = "SUSPICIOUS"
            overall_verdict = "Phishing"
        else:
            risk_verdict = "SAFE"
            overall_verdict = "Legitimate"

    return {
        "ml_probability": float(ml_probability),
        "ml_is_phishing": bool(ml_is_phishing),
        "ml_prediction": "Phishing" if ml_is_phishing else "Legitimate",
        "risk_verdict": risk_verdict,
        "overall_verdict": overall_verdict,
        "tier2_required": routing["tier2_required"],
        "routing_reason": routing["routing_reason"],
        "routing_reasons": routing["routing_reasons"],
        "scan_status": scan_st,
        "analysis_confidence": analysis_conf,
        "forensic_evidence_severity": severity,
        "forensic_evidence": evidence_list
    }

