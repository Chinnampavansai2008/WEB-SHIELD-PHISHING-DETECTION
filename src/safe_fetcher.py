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
from urllib3.connection import HTTPConnection, HTTPSConnection
from typing import Dict, Any, Set, Optional


class SSRFError(ValueError):
    """Raised when a URL targets a restricted IP range, private address, or violates SSRF rules."""
    pass


class FetchError(ValueError):
    """Raised when an HTTP fetch fails due to connection error, timeout, SSL failure, or network error."""
    pass



MAX_RESPONSE_SIZE = 2 * 1024 * 1024  # 2MB limit
DEFAULT_TIMEOUT = 3.0  # 3 seconds timeout


def is_ip_safe(ip_str: str) -> bool:
    """
    Validates if an IP address string is globally routable and not in restricted private,
    loopback, link-local (cloud metadata), CGNAT, multicast, or reserved ranges.
    Handles IPv4, IPv6, IPv4-mapped IPv6, and NAT64 prefixes.
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if ip_obj.version == 4:
        if not ip_obj.is_global or ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
            return False
        # CGNAT 100.64.0.0/10
        if ip_obj in ipaddress.ip_network("100.64.0.0/10"):
            return False
        # Link-local / AWS Metadata 169.254.0.0/16
        if ip_obj in ipaddress.ip_network("169.254.0.0/16"):
            return False
        return True

    elif ip_obj.version == 6:
        # Check IPv4-mapped IPv6 addresses (::ffff:x.x.x.x)
        if ip_obj.ipv4_mapped:
            return is_ip_safe(str(ip_obj.ipv4_mapped))

        # Check NAT64 prefixes (64:ff9b::/96 WKP and 64:ff9b:1::/48 Local)
        nat64_wkp = ipaddress.ip_network("64:ff9b::/96")
        nat64_local = ipaddress.ip_network("64:ff9b:1::/48")
        if ip_obj in nat64_wkp or ip_obj in nat64_local:
            embedded_ipv4 = ipaddress.IPv4Address(ip_obj.packed[-4:])
            return is_ip_safe(str(embedded_ipv4))

        if not ip_obj.is_global or ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
            return False
        return True

    return False


def resolve_and_validate_host(hostname: str, port: int = 80) -> str:
    """
    Resolves hostname to IP using socket.getaddrinfo and verifies all returned IPs are global/safe.
    Returns the selected validated IP string (preferring IPv4 if available among safe IPs).

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

    # Reject if ANY resolved IP is unsafe (prevents DNS rebinding and dual-homed SSRF attacks)
    for ip_str in resolved_ips:
        if not is_ip_safe(ip_str):
            raise SSRFError(f"Hostname '{hostname}' resolved to restricted IP '{ip_str}' (SSRF blocked).")

    # Prefer IPv4 among safe resolved IPs if available
    ipv4_ips = [ip for ip in resolved_ips if ipaddress.ip_address(ip).version == 4]
    if ipv4_ips:
        return sorted(ipv4_ips)[0]
    return sorted(list(resolved_ips))[0]


class PinnedHTTPSConnection(HTTPSConnection):
    """
    urllib3 HTTPSConnection subclass that overrides _new_conn to use pinned_ip for TCP socket creation
    while preserving host for TLS SNI and certificate validation.
    """
    pinned_ip: Optional[str] = None

    def _new_conn(self):
        saved_dns_host = getattr(self, "_dns_host", self.host)
        if self.pinned_ip:
            self._dns_host = self.pinned_ip
        try:
            return super()._new_conn()
        finally:
            self._dns_host = saved_dns_host


class PinnedHTTPConnection(HTTPConnection):
    """
    urllib3 HTTPConnection subclass that overrides _new_conn to use pinned_ip for TCP socket creation.
    """
    pinned_ip: Optional[str] = None

    def _new_conn(self):
        saved_dns_host = getattr(self, "_dns_host", self.host)
        if self.pinned_ip:
            self._dns_host = self.pinned_ip
        try:
            return super()._new_conn()
        finally:
            self._dns_host = saved_dns_host


class PinnedIPAdapter(HTTPAdapter):
    """
    Custom HTTPAdapter forcing requests to connect directly to the pre-validated target IP
    without triggering urllib3 2.x PoolKey errors or breaking TLS SNI/hostname verification.
    """
    def __init__(self, pinned_ip: str, *args, **kwargs):
        self.pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def get_connection_with_tls_context(self, request, verify, cert=None, proxies=None):
        conn = super().get_connection_with_tls_context(request, verify, cert=cert, proxies=proxies)
        pinned_ip = self.pinned_ip

        class CustomHTTPSConnection(PinnedHTTPSConnection):
            pass

        CustomHTTPSConnection.pinned_ip = pinned_ip
        conn.ConnectionCls = CustomHTTPSConnection
        return conn

    def get_connection(self, url, proxies=None):
        conn = super().get_connection(url, proxies=proxies)
        pinned_ip = self.pinned_ip

        class CustomHTTPConnection(PinnedHTTPConnection):
            pass

        CustomHTTPConnection.pinned_ip = pinned_ip
        conn.ConnectionCls = CustomHTTPConnection
        return conn


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
        FetchError: If HTTP fetch fails after security validation.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFError(f"Unsupported URL scheme '{scheme}'. Only http and https are allowed.")

    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        raise SSRFError("URLs containing userinfo (username/password) are rejected.")

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
    except SSRFError:
        raise
    except requests.RequestException as e:
        raise FetchError(f"HTTP request failed: {e}")
    except Exception as e:
        raise FetchError(f"Failed to safely fetch URL: {e}")

