"""
URL and IP Defanging Module.
Converts active URLs and IP addresses into safe, non-clickable indicators.
"""

from urllib.parse import urlparse


def defang_ip(ip: str) -> str:
    """
    Defangs an IP address string by replacing dots with [.].

    Example:
        192.168.1.1 -> 192[.]168[.]1[.]1
    """
    if not ip or not isinstance(ip, str):
        return ""
    return ip.strip().replace(".", "[.]")


def defang_url(url: str) -> str:
    """
    Defangs a URL string by sanitizing protocol scheme (http -> hxxp[://], https -> hxxps[://])
    and replacing dots in the domain authority with [.].

    Example:
        https://evil.com/login?u=1 -> hxxps[://]evil[.]com/login?u=1
    """
    if not url or not isinstance(url, str):
        return ""

    url_str = url.strip()

    if url_str.startswith("https://"):
        prefix = "hxxps[://]"
        rest = url_str[8:]
    elif url_str.startswith("http://"):
        prefix = "hxxp[://]"
        rest = url_str[7:]
    else:
        prefix = ""
        rest = url_str

    if "/" in rest:
        netloc, path = rest.split("/", 1)
        path = "/" + path
    elif "?" in rest:
        netloc, path = rest.split("?", 1)
        path = "?" + path
    else:
        netloc = rest
        path = ""

    defanged_netloc = netloc.replace(".", "[.]")

    return f"{prefix}{defanged_netloc}{path}"
