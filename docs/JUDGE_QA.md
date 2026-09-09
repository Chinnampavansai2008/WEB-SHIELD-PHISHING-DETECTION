# Web Shield — Hackathon Judge & Technical Q&A

## Frequently Asked Questions

### Q1: What makes Web Shield different from traditional phishing scanners?
**A**: Most phishing scanners either perform simple blacklisting (which fails against brand-new domains) or output an unexplainable 0–100 probability score. Web Shield combines **Tier 1 explainable ML screening** (using XGBoost + TreeSHAP feature attributions) with **Tier 2 security-hardened forensic analysis** (SSRF protection, IP pinning, redirect tracing, static DOM credential audits, and defanged IOC exports).

---

### Q2: How does TreeSHAP explainability work in Web Shield?
**A**: TreeSHAP computes exact Game-Theoretic Shapley values for each feature in the input vector. Web Shield feeds the exact same unscaled feature vector into both prediction and SHAP calculation, and maps mathematical attributions to neutral semantic descriptions (e.g. *"URL Character Count"*) so users can see exactly why a URL was flagged.

---

### Q3: How does Web Shield protect against SSRF and DNS Rebinding?
**A**: Before making any Tier 2 network connection, Web Shield resolves the domain's IP addresses via DNS and checks them against global private IPv4/IPv6 subnets (loopback, RFC1918, link-local, CGNAT, NAT64). The custom `PinnedIPAdapter` binds socket connections directly to pre-validated IP addresses while retaining the target hostname in the TLS SNI header for certificate verification.

---

### Q4: Why is Model V2 used as the production baseline instead of V3 or V3.1?
**A**: Model V2 preserves high baseline phishing recall (94.32% on Set A benchmark). While candidate models V3 and V3.1 significantly reduced false positives on deep paths, their recall on domain-based phishing dropped to 71.72%–74.75%. In a primary phishing detection system, a 20-point drop in phishing recall is unacceptable. Model V2 remains active while future feature expansions are developed.

---

### Q5: What happens when WHOIS or domain age cannot be fetched?
**A**: Web Shield treats missing domain age safely: `domain_age_months_clean = 0` and `domain_age_known = 0`. Domain age status is explicitly marked `UNKNOWN`, preventing WHOIS lookup failures from being misclassified as brand-new domains.

---

### Q6: Can Web Shield analyze JavaScript-rendered phishing pages?
**A**: Tier 2 performs static DOM parsing of HTML forms and inputs. Client-side single-page applications (SPAs) or JS-only rendered forms are a documented limitation of static analysis, which represents a prime area for future dynamic headless browser integration.
