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

## Authoritative Benchmark Metrics

### 1. Auth-426 Holdout Benchmark (`data/dataset_auth_phishing_final_independent.csv`, 426 samples)
- **Confusion Matrix**: `TN = 170`, `FP = 6`, `FN = 0`, `TP = 250`
- **Accuracy**: **98.59%**
- **Precision**: **97.66%**
- **Recall**: **100.00%**
- **F1-Score**: **98.81%**
- **FPR**: **3.41%**
- **Tier-2 Routing Rate**: **20.66%** (88 / 426 URLs)

### 2. Primary-452 Holdout Benchmark (`data/dataset_blind_holdout.csv`, 452 samples)
- **Confusion Matrix**: `TN = 190`, `FP = 12`, `FN = 11`, `TP = 239`
- **Recall**: **95.60%**
- **FPR**: **5.94%**
- **F1-Score**: **95.41%**
- **Total Routing Rate**: **57.74%** (261 / 452 URLs)
- **Non-IP Routing Rate**: **5.97%** (27 / 452 URLs)
- **Offline Tier-2 FN Recovery**: **0**

---

## Missingness & Domain Age Policy
- In offline evaluation or when WHOIS data is unavailable, `domain_age_known` is set to `0` and `domain_age_months_clean` is set to `0`.
- Missing WHOIS data is treated as an explicit missingness encoding rather than a verified 0-month-old domain.

---

## Benchmark Limitations & Disclosure
1. **Auth-426 Structural Collisions**: Auth-426 contains repeated identical 28-feature patterns across distinct real-world URLs. 50 collision groups cover 297 samples; therefore, sample-level metrics do not represent 426 statistically independent feature patterns.
2. **Primary-452 IP-Host Distribution**: Primary-452 contains 234 IP-host phishing samples and 0 legitimate IP-host samples. Its 57.74% total routing rate is dominated by IP-host rules ($234/452 = 51.77\%$) and does not represent real production traffic composition.
3. **Tier-2 Forensic Claim**: Tier 2 provides static forensic evidence enrichment for selected URLs. In the canonical offline Primary-452 benchmark, Tier 2 did not change the Tier-1 confusion matrix.
4. **Shadow Execution Wording**: Shadow inference is isolated from production decision semantics but currently executes synchronously within the request path. Under concurrent load, shadow execution adds request latency. Moving shadow inference to an asynchronous worker/queue is a future production optimization.
