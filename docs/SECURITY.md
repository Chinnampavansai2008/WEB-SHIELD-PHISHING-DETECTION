# Security & Defense Specification — Web Shield

## Overview

Web Shield is engineered to perform threat inspection safely without introducing Server-Side Request Forgery (SSRF), DNS rebinding, or XSS vulnerabilities.

---

## 1. SSRF Protection Architecture (`src/safe_fetcher.py`)

Every outbound network request performed by Tier 2 is subject to strict pre-connection and in-flight IP validation.

### Restricted IP Subnets (Blocked)
1. **Loopback**: `127.0.0.0/8`, `::1`
2. **Private RFC1918**: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
3. **Link-Local / Cloud Metadata**: `169.254.0.0/16`, `fe80::/10`
4. **CGNAT**: `100.64.0.0/10`
5. **NAT64 / IPv4-Mapped IPv6**: IPv6 addresses embedding restricted IPv4 subnets are rejected.

---

## 2. DNS Pinning & SNI Preservation (`PinnedIPAdapter`)
- **DNS Pinning**: Resolves hostname via DNS prior to opening TCP socket. The validated IP address is bound to the socket connection to prevent DNS rebinding attacks between check time and connect time.
- **SNI & TLS Certificate Validation**: Retains the original target hostname in the TLS SNI header to ensure proper virtual hosting while enforcing full certificate chain validation (`CERT_REQUIRED`).

---

## 3. Manual Redirect Chain Tracing
- Automatic HTTP library redirects are disabled (`allow_redirects=False`).
- Each `Location` header is intercepted, normalized, and re-evaluated against SSRF and DNS pinning rules before initiating subsequent requests.
- Maximum redirect depth is capped at 5 hops.

---

## 4. Failed-Fetch Safety Semantics
Network failures (DNS resolution failure, TLS error, connection timeout, connection refused) are treated safely:
- `fetch_succeeded` = `False`
- `HTTP status` = `None`
- `scan_status` = `failed` or `partial`
- `credential_audit` = `NOT_EVALUATED`
- `sink_risk` = `UNKNOWN`

Failed fetches **NEVER** return false `CLEAN` verdicts or fabricated `HTTP 0`.

---

## 5. Defanged Output & XSS Prevention
- Suspicious URLs in exported reports are defanged (`hxxps[://]domain[.]com`).
- Flask Jinja2 auto-escaping is active across all UI templates to prevent stored/reflected XSS from user-controlled URL input.
