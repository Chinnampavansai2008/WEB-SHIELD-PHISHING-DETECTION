"""
Safe Fetcher Module with SSRF & DNS Rebinding Protection.
Enforces strict IP validation, DNS rebinding mitigation, connection timeouts, and payload size limits.
"""

import socket
import ipaddress
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import connection
from typing import Dict, Any, Set


class SSRFError(ValueError):
    """Raised when a URL targets a restricted IP range, private address, or violates SSRF rules."""
    pass


MAX_RESPONSE_SIZE = 2 * 1024 * 1024  # 2MB limit
DEFAULT_TIMEOUT = 3.0  # 3 seconds timeout


def is_ip_safe(ip_str: str) -> bool:
    """
    Validates if an IP address string is globally routable and not in restricted private,
    loopback, link-local (cloud metadata), CGNAT, or multicast ranges.
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if not ip_obj.is_global:
        return False

    if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
        return False

    if ip_obj.version == 4:
        # CGNAT 100.64.0.0/10
        if ip_obj in ipaddress.ip_network("100.64.0.0/10"):
            return False
        # Link-local / AWS Metadata 169.254.0.0/16
        if ip_obj in ipaddress.ip_network("169.254.0.0/16"):
            return False

    return True


def resolve_and_validate_host(hostname: str, port: int = 80) -> str:
    """
    Resolves hostname to IP using socket.getaddrinfo and verifies all returned IPs are global/safe.
    Returns the validated IP string.

    Raises:
        SSRFError: If any resolved IP is non-global or private.
    """
    if not hostname:
        raise SSRFError("Empty hostname provided.")

    # Check if hostname is an explicit IP string
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if not is_ip_safe(str(ip_obj)):
            raise SSRFError(f"Target IP address '{hostname}' is in a restricted range (SSRF blocked).")
        return str(ip_obj)
    except ValueError:
        pass

    try:
        addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFError(f"DNS resolution failed for hostname '{hostname}': {e}")

    if not addr_info:
        raise SSRFError(f"No IP addresses resolved for hostname '{hostname}'.")

    resolved_ips: Set[str] = set()
    for item in addr_info:
        sock_addr = item[4]
        ip_str = sock_addr[0]
        resolved_ips.add(ip_str)

    # Reject if any resolved IP is unsafe
    for ip_str in resolved_ips:
        if not is_ip_safe(ip_str):
            raise SSRFError(f"Hostname '{hostname}' resolved to restricted IP '{ip_str}' (SSRF blocked).")

    return list(resolved_ips)[0]


class PinnedIPAdapter(HTTPAdapter):
    """
    Custom HTTPAdapter forcing requests to connect directly to the pre-validated target IP.
    """
    def __init__(self, pinned_ip: str, *args, **kwargs):
        self.pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pinned_ip = self.pinned_ip

        def _custom_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, socket_options=None):
            # Connect socket to pinned_ip using specified port address[1]
            return connection.create_connection((pinned_ip, address[1]), timeout, source_address, socket_options)

        # Override default create_connection inside urllib3 pool manager
        pool_kwargs['connection_pool_kw'] = pool_kwargs.get('connection_pool_kw', {})
        pool_kwargs['connection_pool_kw']['create_connection'] = _custom_create_connection
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)


def safe_fetch(url: str, timeout: float = DEFAULT_TIMEOUT, max_size: int = MAX_RESPONSE_SIZE) -> Dict[str, Any]:
    """
    Safely fetches web content enforcing SSRF prevention, IP pinning, timeout, and response size limits.

    Args:
        url (str): Target URL to fetch.
        timeout (float): Request timeout in seconds.
        max_size (int): Max response bytes to download.

    Returns:
        dict: {
            'status_code': int,
            'content': bytes,
            'text': str,
            'headers': dict,
            'final_url': str,
            'resolved_ip': str
        }

    Raises:
        SSRFError: If target URL or resolved IP violates SSRF rules.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFError(f"Unsupported URL scheme '{scheme}'. Only http and https are allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL missing hostname.")

    port = parsed.port if parsed.port else (443 if scheme == "https" else 80)
    target_ip = resolve_and_validate_host(hostname, port)

    session = requests.Session()
    adapter = PinnedIPAdapter(pinned_ip=target_ip)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "WebShield-Phishing-Scanner/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=False  # Do not auto-follow redirects to prevent open-redirect SSRF
        )

        content = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            content.extend(chunk)
            if len(content) > max_size:
                response.close()
                raise SSRFError(f"Response payload size exceeded limit of {max_size} bytes.")

        text_content = ""
        try:
            encoding = response.encoding or "utf-8"
            text_content = content.decode(encoding, errors="replace")
        except Exception:
            text_content = content.decode("utf-8", errors="replace")

        return {
            "status_code": response.status_code,
            "content": bytes(content),
            "text": text_content,
            "headers": dict(response.headers),
            "final_url": url,
            "resolved_ip": target_ip
        }
    except requests.RequestException as e:
        raise SSRFError(f"Failed to safely fetch URL: {e}")
