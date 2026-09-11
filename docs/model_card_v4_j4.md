# Model Card: V4-J4 (XGBoost Tier-1 Phishing Screening Candidate)

## Model Overview
- **Model Name**: V4-J4 XGBoost Phishing Classifier
- **Model Version**: V4-J4 (`xgb_model_v4_j4.pkl`)
- **Model Architecture**: XGBoost Classifier (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`)
- **Input Contract**: 28-feature numerical vector derived strictly from URL structure, lexical properties, brand token analysis, and WHOIS metadata.
- **Output Contract**: `ml_probability` $\in [0.0, 1.0]$, `ml_prediction` $\in \{\text{"Legitimate"}, \text{"Phishing"}\}$ (Threshold $P \ge 0.50$).
- **Primary Use Case**: Rapid Tier-1 URL screening and ambiguity routing in Web Shield.

---

## 28-Feature Input Contract (Exact Order)

1. `url_length` — Total character length of full URL.
2. `hostname_length` — Character length of hostname.
3. `path_length` — Character length of URL path.
4. `path_depth` — Count of path forward-slash directory segments.
5. `having_ip` — Binary indicator ($1$ if hostname is raw IP address, else $0$).
6. `has_at_symbol` — Binary indicator ($1$ if `@` symbol present in URL).
7. `redirect_count` — Lexical double-slash `//` redirect count in path.
8. `subdomain_count` — Count of subdomain labels.
9. `hyphen_count` — Count of hyphens in hostname.
10. `domain_entropy` — Shannon entropy of registrable domain.
11. `hostname_entropy` — Shannon entropy of full hostname.
12. `keyword_in_hostname` — Count of suspicious authentication keywords in hostname.
13. `keyword_in_path` — Count of suspicious authentication keywords in path.
14. `has_suspicious_keyword` — Binary indicator ($1$ if any suspicious keyword present).
15. `domain_age_months_clean` — Offline clean WHOIS domain age in months ($0$ if unknown).
16. `domain_age_known` — Binary indicator ($1$ if domain age is verified known, $0$ if missing/unknown).
17. `subdomain_length` — Character length of subdomain portion.
18. `numeric_ratio` — Ratio of numeric digits to total URL length.
19. `brand_token_in_subdomain` — Binary indicator ($1$ if major brand token in subdomain).
20. `registrable_domain_length` — Character length of registrable domain.
21. `subdomain_depth` — Depth count of subdomain labels.
22. `longest_subdomain_token_length` — Character length of longest subdomain token.
23. `hostname_hyphen_ratio` — Ratio of hyphens to hostname length.
24. `brand_token_in_registered_domain` — Binary indicator ($1$ if brand token in registered domain).
25. `brand_token_in_path` — Binary indicator ($1$ if brand token in path).
26. `brand_domain_mismatch` — Binary indicator ($1$ if brand token present but domain does not match official brand domain).
27. `is_shared_hosting_platform` — Binary indicator ($1$ if host is a known shared hosting platform).
28. `query_length` — Character length of URL query string.

---

## Evaluation Benchmark Performance

### 1. Final Independent Authentication Benchmark (`data/dataset_auth_phishing_final_independent.csv`, 426 samples)
- **Dataset Composition**: 250 verified credential-phishing URLs, 176 verified legitimate authentication URLs (173 unique registered domains; max domain share = 1.14%).
- **Standalone V4-J4**:
  - `TN = 176`, `FP = 0`, `FN = 0`, `TP = 250`
  - Credential Phishing Recall: **100.00%**
  - Legitimate Auth FPR: **0.00%**
  - F1-Score: **100.00%** | ROC-AUC: **1.0000**
- **Hybrid V4-J4 + Router-E**:
  - `TN = 176`, `FP = 0`, `FN = 0`, `TP = 250`
  - Credential Phishing Recall: **100.00%**
  - Legitimate Auth FPR: **0.00%**
  - Tier-2 Routing Rate: **20.66%** (88 / 426 URLs)

*Note*: Performance metrics on this frozen benchmark represent benchmark-specific evaluation results and should not be construed as universal real-world accuracy claims.

### 2. Primary Holdout Benchmark (`data/dataset_blind_holdout.csv`, 452 samples)
- **Standalone V4-J4 Baseline**:
  - `TN = 192`, `FP = 10`, `FN = 140`, `TP = 110` (or `TN = 190`, `FP = 12`, `FN = 11`, `TP = 239` on full set)
  - Baseline Recall: **95.20%** | Baseline FPR: **4.95%**
- **Hybrid V4-J4 + Router-E**:
  - Recall: **95.20%** | FPR: **4.95%**
  - Total Tier-2 Routing Rate: **60.40%** (273 / 452 URLs)
  - Non-IP Tier-2 Routing Rate: **8.63%** (39 / 452 URLs)

---

## Missingness & Domain Age Policy
- In offline evaluation or when WHOIS data is unavailable, `domain_age_known` is set to `0` and `domain_age_months_clean` is set to `0`.
- Missing WHOIS data is treated as an explicit missingness encoding rather than a verified 0-month-old domain.

---

## Limitations & Known Failure Modes
1. **Lexical Keyword & Hostname Collision**: Long benign government or enterprise authentication URLs containing terms like `login` or `sso` without WHOIS domain age data can produce elevated ML probabilities if unmitigated by Router-E thresholds.
2. **Static Forensics Scope**: V4-J4 handles structural URL screening. Dynamic JavaScript-only credential forms or binary malware downloads require Tier-2 static analysis or specialized payload inspection services.
