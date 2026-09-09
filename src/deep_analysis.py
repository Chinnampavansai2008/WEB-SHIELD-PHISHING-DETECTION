"""
Deep Forensic Analysis Engine.
Traces multi-hop redirects (up to 5 hops), performs static DOM credential audit,
checks homoglyph/brand impersonation, and enforces a strict 4.0s global scan deadline.
"""

import time
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import tldextract
from typing import Dict, Any, List

from src.url_normalizer import normalize_url, InvalidURLError
from src.safe_fetcher import safe_fetch, SSRFError, DEFAULT_TIMEOUT
from src.defang import defang_url
from src.credential_analysis import analyze_credentials
from src.homoglyph import analyze_homoglyphs


GLOBAL_SCAN_DEADLINE_SECONDS = 4.0
MAX_REDIRECT_HOPS = 5


def _get_registered_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ext.top_domain_under_public_suffix





def perform_deep_analysis(url: str, timeout_budget: float = GLOBAL_SCAN_DEADLINE_SECONDS) -> Dict[str, Any]:
    """
    Executes a Tier 2 deep forensic scan on the given URL.

    Args:
        url (str): The starting target URL.
        timeout_budget (float): Maximum allowed execution time in seconds (default 4.0s).

    Returns:
        dict: Complete forensic analysis report.
    """
    start_time = time.monotonic()
    deadline = start_time + timeout_budget

    redirect_timeline: List[Dict[str, Any]] = []
    current_url = url
    hop_count = 0
    final_html = ""
    status_code = None
    fetch_succeeded = False
    deadline_exceeded = False
    error_message = None

    try:
        norm = normalize_url(url)
        current_url = norm["canonical_url"]
    except InvalidURLError as e:
        return {
            'scan_status': 'failed',
            'fetch_succeeded': False,
            'http_status': None,
            'fetch_status': 'invalid_url',
            'target_url': defang_url(url),
            'final_url': defang_url(url),
            'redirect_count': 0,
            'redirect_timeline': [],
            'title': '',
            'meta_description': '',
            'credential_analysis': {
                'status': 'not_evaluated',
                'reason': f"URL Normalization error: {e}",
                'has_login_form': False,
                'sensitive_fields': [],
                'form_actions': [],
                'sink_risk': 'unknown',
                'exfiltration_flag': 'not_evaluated'
            },
            'homoglyph_analysis': analyze_homoglyphs(""),
            'scan_duration_seconds': round(time.monotonic() - start_time, 4),
            'deadline_exceeded': False,
            'error': f"URL Normalization error: {e}"
        }

    previous_domain = _get_registered_domain(current_url)

    while hop_count <= MAX_REDIRECT_HOPS:
        time_left = deadline - time.monotonic()
        if time_left <= 0.005:
            deadline_exceeded = True
            error_message = error_message or "Global scan deadline exceeded"
            break

        fetch_timeout = max(0.05, min(time_left, DEFAULT_TIMEOUT))


        try:
            res = safe_fetch(current_url, timeout=fetch_timeout)
            status_code = res["status_code"]
            headers = res["headers"]
            final_html = res["text"]
            fetch_succeeded = True

            curr_domain = _get_registered_domain(current_url)
            cross_domain = bool(previous_domain and curr_domain and previous_domain != curr_domain)

            hop_info = {
                'hop': hop_count,
                'url': defang_url(current_url),
                'status': status_code,
                'cross_domain': cross_domain
            }
            redirect_timeline.append(hop_info)
            previous_domain = curr_domain

            # Check redirect headers
            location_key = next((k for k in headers.keys() if k.lower() == 'location'), None)
            if status_code in (301, 302, 303, 307, 308) and location_key:
                next_url = urljoin(current_url, headers[location_key])
                try:
                    norm_next = normalize_url(next_url)
                    current_url = norm_next["canonical_url"]
                    hop_count += 1
                except InvalidURLError:
                    break
            else:
                break

        except (SSRFError, Exception) as e:
            error_message = str(e)
            hop_info = {
                'hop': hop_count,
                'url': defang_url(current_url),
                'status': status_code if fetch_succeeded else None,
                'error': error_message
            }
            redirect_timeline.append(hop_info)
            break

    # Extract DOM details only if fetch succeeded
    title = ""
    meta_description = ""
    if fetch_succeeded and final_html:
        try:
            soup = BeautifulSoup(final_html, 'html.parser')
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            meta_tag = soup.find('meta', attrs={'name': lambda v: v and v.lower() == 'description'})
            if meta_tag and meta_tag.get('content'):
                meta_description = meta_tag.get('content').strip()
        except Exception:
            pass

        cred_analysis = analyze_credentials(final_html, current_url)
        scan_status = 'completed' if not error_message else 'partial'
    else:
        scan_status = 'failed'
        cred_analysis = {
            'status': 'not_evaluated',
            'reason': f"Final HTML could not be retrieved: {error_message or 'Fetch failed'}",
            'has_login_form': False,
            'sensitive_fields': [],
            'form_actions': [],
            'sink_risk': 'unknown',
            'exfiltration_flag': 'not_evaluated'
        }

    parsed_final = urlparse(current_url)
    homoglyph_analysis = analyze_homoglyphs(parsed_final.hostname or "")

    duration = round(time.monotonic() - start_time, 4)

    return {
        'scan_status': scan_status,
        'fetch_succeeded': fetch_succeeded,
        'http_status': status_code,
        'fetch_status': 'success' if fetch_succeeded else 'request_error',
        'target_url': defang_url(url),
        'final_url': defang_url(current_url),
        'redirect_count': len(redirect_timeline) - 1 if len(redirect_timeline) > 1 and fetch_succeeded else 0,
        'redirect_timeline': redirect_timeline,
        'title': title,
        'meta_description': meta_description,
        'credential_analysis': cred_analysis,
        'homoglyph_analysis': homoglyph_analysis,
        'scan_duration_seconds': duration,
        'deadline_exceeded': deadline_exceeded or (duration >= timeout_budget),
        'error': error_message
    }

