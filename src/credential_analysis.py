"""
Credential & Form Exfiltration Analysis Module.
Audits HTML DOM forms, fingerprinting sensitive input fields (Password, OTP, Payment)
and classifying form target destinations to detect credential harvesting sinks.
"""

from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import tldextract
import ipaddress
from typing import Dict, List, Any


PASSWORD_KEYWORDS = {"password", "pwd", "pass", "userpass", "secret"}
OTP_KEYWORDS = {"otp", "token", "totp", "verification_code", "verificationcode", "2fa", "mfa", "security_code", "authenticator"}
PAYMENT_KEYWORDS = {"card", "cc_number", "cardnumber", "cvv", "cvc", "exp_month", "exp_year", "creditcard", "pan", "cc_name"}

KNOWN_OAUTH_DOMAINS = {
    "google.com", "accounts.google.com",
    "microsoftonline.com", "login.microsoftonline.com", "login.live.com", "microsoft.com",
    "okta.com", "auth0.com", "pingidentity.com", "keycloak.org"
}

KNOWN_PAYMENT_DOMAINS = {
    "stripe.com", "checkout.stripe.com",
    "paypal.com", "checkout.paypal.com",
    "adyen.com", "braintreegateway.com", "squareup.com"
}

KNOWN_WEBHOOK_DOMAINS = {
    "discord.com", "discordapp.com", "api.telegram.org", "hooks.slack.com",
    "webhook.site", "pipedream.net", "requestcatcher.com"
}


def _get_registered_domain(host: str) -> str:
    ext = tldextract.extract(host)
    if hasattr(ext, 'top_domain_under_public_suffix') and ext.top_domain_under_public_suffix:
        return ext.top_domain_under_public_suffix
    return getattr(ext, 'registered_domain', '') or ''


def _classify_destination(action_url: str, page_url: str) -> str:
    """
    Classifies a form action URL against the current page domain.
    """
    if not action_url or action_url.strip() in ("", "#", "javascript:void(0)"):
        return "SAME_ORIGIN"

    resolved = urljoin(page_url, action_url)
    parsed_action = urlparse(resolved)
    parsed_page = urlparse(page_url)

    action_host = (parsed_action.hostname or "").lower()
    page_host = (parsed_page.hostname or "").lower()

    if not action_host:
        return "SAME_ORIGIN"

    try:
        ipaddress.ip_address(action_host)
        return "RAW_IP"
    except ValueError:
        pass

    if any(wh in action_host or wh in resolved for wh in KNOWN_WEBHOOK_DOMAINS):
        return "WEBHOOK"

    if action_host == page_host and parsed_action.scheme == parsed_page.scheme:
        return "SAME_ORIGIN"

    action_reg = _get_registered_domain(action_host)
    page_reg = _get_registered_domain(page_host)

    if action_reg and action_reg == page_reg:
        return "SAME_REGISTERED_DOMAIN"

    if any(action_reg == _get_registered_domain(d) for d in KNOWN_OAUTH_DOMAINS):
        return "KNOWN_OAUTH"

    if any(action_reg == _get_registered_domain(d) for d in KNOWN_PAYMENT_DOMAINS):
        return "KNOWN_PAYMENT"

    return "UNKNOWN_EXTERNAL"



def analyze_credentials(html_content: str, page_url: str = "http://example.com") -> Dict[str, Any]:
    """
    Audits HTML content for sensitive input fields and exfiltration sinks.

    Args:
        html_content (str): Raw HTML content of the page.
        page_url (str): Current page URL for origin context.

    Returns:
        dict: {
            'status': 'evaluated' | 'not_evaluated',
            'has_login_form': bool,
            'sensitive_fields': list,
            'form_actions': list,
            'sink_risk': 'low' | 'medium' | 'high' | 'critical' | 'unknown',
            'exfiltration_flag': bool | 'not_evaluated'
        }
    """
    if not html_content or not isinstance(html_content, str):
        return {
            'status': 'not_evaluated',
            'reason': 'No HTML content provided',
            'has_login_form': False,
            'sensitive_fields': [],
            'form_actions': [],
            'sink_risk': 'unknown',
            'exfiltration_flag': 'not_evaluated'
        }

    soup = BeautifulSoup(html_content, 'html.parser')
    forms = soup.find_all('form')

    sensitive_fields = set()
    has_login_form = False

    for element in soup.find_all(['input', 'textarea']):
        input_type = (element.get('type') or 'text').lower()

        attr_values = " ".join([
            str(element.get('name') or ''),
            str(element.get('id') or ''),
            str(element.get('placeholder') or ''),
            str(element.get('aria-label') or ''),
            str(element.get('autocomplete') or '')
        ]).lower()

        if input_type == 'password' or any(kw in attr_values for kw in PASSWORD_KEYWORDS):
            sensitive_fields.add('password')
            has_login_form = True

        if any(kw in attr_values for kw in OTP_KEYWORDS):
            sensitive_fields.add('otp')
            has_login_form = True

        if any(kw in attr_values for kw in PAYMENT_KEYWORDS):
            sensitive_fields.add('payment')

    form_actions_info = []
    highest_dest_risk = "SAFE"
    exfiltration_detected = False

    for form in forms:
        action_attr = form.get('action') or ''
        resolved_action = urljoin(page_url, action_attr) if action_attr else page_url
        dest_type = _classify_destination(action_attr, page_url)

        form_actions_info.append({
            'action_url': resolved_action,
            'destination_type': dest_type
        })

        if dest_type in ("WEBHOOK", "RAW_IP"):
            if sensitive_fields:
                exfiltration_detected = True
                highest_dest_risk = "CRITICAL"
        elif dest_type == "UNKNOWN_EXTERNAL":
            if sensitive_fields:
                exfiltration_detected = True
                if highest_dest_risk != "CRITICAL":
                    highest_dest_risk = "CRITICAL"

    if not sensitive_fields:
        sink_risk = "low"
        exfiltration_flag = False
    elif highest_dest_risk == "CRITICAL" or exfiltration_detected:
        sink_risk = "critical"
        exfiltration_flag = True
    elif highest_dest_risk == "HIGH":
        sink_risk = "high"
        exfiltration_flag = True
    else:
        sink_risk = "low"
        exfiltration_flag = False

    return {
        'status': 'evaluated',
        'has_login_form': has_login_form,
        'sensitive_fields': sorted(list(sensitive_fields)),
        'form_actions': form_actions_info,
        'sink_risk': sink_risk,
        'exfiltration_flag': exfiltration_flag
    }

