# Model Card — Web Shield Model V2

## Model Overview
- **Model Name**: Web Shield XGBoost Model V2
- **Model Version**: `v2` (Active Production Default)
- **Algorithm**: `XGBoostClassifier` (Tree Gradient Boosting)
- **Artifact Path**: `models/v2/xgb_model_v2.pkl`
- **Artifact SHA256**: `f9ca677c77c0500bf09ec711a7a01a35541ea370bf21f9ae05b46b088e89f6b9`
- **Feature Contract Path**: `models/v2/features_v2.pkl`
- **Scaler Requirement**: **NONE** (Unscaled raw feature input)

---

## Model Inputs (11 Canonical Features)

| # | Feature Name | Description | Data Type | Expected Range |
| :- | :--- | :--- | :---: | :---: |
| 1 | `url_length` | Total character count of canonical URL | Integer | $[1, \infty)$ |
| 2 | `having_ip` | Binary flag for IPv4/IPv6 hostname | Binary | $0$ or $1$ |
| 3 | `has_at_symbol` | Presence of `@` in URL authority/path | Binary | $0$ or $1$ |
| 4 | `redirect_count` | Number of embedded redirect tokens | Integer | $[0, \infty)$ |
| 5 | `subdomain_count` | Number of subdomains under public suffix | Integer | $[0, \infty)$ |
| 6 | `hyphen_count` | Hyphen count in registered domain | Integer | $[0, \infty)$ |
| 7 | `domain_entropy` | Shannon entropy of domain string | Float | $[0.0, 8.0]$ |
| 8 | `has_suspicious_keyword` | Matches in static phishing keyword list | Binary | $0$ or $1$ |
| 9 | `ssl_valid` | Valid TLS certificate availability | Binary | $0$ or $1$ |
| 10 | `domain_age_months_clean` | WHOIS domain age in months ($0$ if unknown) | Integer | $[0, \infty)$ |
| 11 | `domain_age_known` | Binary flag ($1$ if WHOIS age known, $0$ if UNKNOWN) | Binary | $0$ or $1$ |

---

## Output & Label Semantics
- **Classes**: `model.classes_ = [0, 1]`
  - `0`: Legitimate
  - `1`: Phishing
- **Default Decision Threshold**: `0.50`

---

## Evaluation Performance

### 1. SET A: Legacy V2 Grouped Test Partition (245 Samples)
- **Accuracy**: **95.92%**
- **Precision**: **94.32%**
- **Recall**: **94.32%**
- **F1-Score**: **94.32%**
- **ROC-AUC**: **99.02%**
- **FPR**: **3.18%**

### 2. SET B: V3 Length-Balanced Grouped Test Partition (444 Samples)
- **Accuracy**: **66.67%**
- **Recall**: **77.78%**
- **FPR**: **36.52%**

### 3. SET C: Hard Independent Holdout Partition (198 Samples)
- **Accuracy**: **80.30%**
- **Precision**: **80.00%**
- **Recall**: **77.42%**
- **F1-Score**: **78.69%**

---

## Known ML Limitations
- **URL-Length Sensitivity**: Model V2 learned a split boundary near $url\_length \ge 30$, leading to false positive elevations on long legitimate URLs (e.g., Wikipedia articles, GitHub repositories, deep payment paths).
- **Domain-Based Phishing**: High performance on IP-hosted phishing (100% recall), but moderate recall on domain-based phishing (59.26% on Set B domain phishing).
