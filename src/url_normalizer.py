"""
URL Normalizer Module.
Provides safe URL parsing, scheme validation, userinfo stripping/rejection, and canonicalization.
"""

from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Dict


class InvalidURLError(ValueError):
    """Raised when a URL is malformed, uses an unsupported scheme, or contains userinfo."""
    pass


ALLOWED_SCHEMES = {"http", "https"}


def normalize_url(url: str) -> Dict[str, str]:
    """
    Parses and canonicalizes a URL string.

    Args:
        url (str): The raw input URL.

    Returns:
        dict: {
            'canonical_url': str,
            'hostname': str,
            'scheme': str
        }

    Raises:
        InvalidURLError: If scheme is not http/https, URL contains userinfo (@), or URL is invalid.
    """
    if not url or not isinstance(url, str):
        raise InvalidURLError("URL must be a non-empty string.")

    url_str = url.strip()

    # Handle protocol-relative or schemeless URLs
    if url_str.startswith("//"):
        url_str = "http:" + url_str
    elif "://" not in url_str:
        url_str = "http://" + url_str

    try:
        parsed = urlparse(url_str)
    except Exception as e:
        raise InvalidURLError(f"Failed to parse URL: {e}")

    try:
        scheme = parsed.scheme.lower()
    except Exception as e:
        raise InvalidURLError(f"Disallowed URL scheme: {e}")

    if scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError(f"Disallowed URL scheme '{scheme}'. Only 'http' and 'https' are allowed.")

    # Reject userinfo (e.g. user:pass@host or user@host)
    if parsed.username or parsed.password:
        raise InvalidURLError("URLs containing user authentication details (userinfo) are rejected for security.")

    netloc = parsed.netloc
    if "@" in netloc:
        raise InvalidURLError("URLs containing '@' in authority component are rejected.")

    hostname = parsed.hostname
    if not hostname:
        raise InvalidURLError("URL missing a valid hostname.")

    hostname = hostname.lower().rstrip(".")

    # Reconstruct port if non-standard
    port_str = ""
    try:
        parsed_port = parsed.port
        if parsed_port:
            if (scheme == "http" and parsed_port != 80) or (scheme == "https" and parsed_port != 443):
                port_str = f":{parsed_port}"
    except ValueError as e:
        raise InvalidURLError(f"Invalid URL port: {e}")


    netloc_clean = f"{hostname}{port_str}"

    path = parsed.path
    if not path:
        path = "/"

    query = parsed.query
    if query:
        try:
            query_params = parse_qsl(query, keep_blank_values=True)
            query = urlencode(sorted(query_params))
        except Exception:
            pass

    canonical_url = urlunparse((scheme, netloc_clean, path, parsed.params, query, parsed.fragment))

    return {
        "canonical_url": canonical_url,
        "hostname": hostname,
        "scheme": scheme,
    }
