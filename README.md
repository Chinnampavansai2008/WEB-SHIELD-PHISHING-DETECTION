# Web Shield

> **Explainable Two-Stage Phishing Detection and Forensic Triage Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-114%2F114%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

**Web Shield** is an explainable, two-stage phishing detection and forensic triage platform designed for modern security operations and automated threat analysis. Unlike traditional black-box security tools that return arbitrary risk scores, Web Shield combines rapid machine-learning screening with security-hardened forensic analysis to deliver transparent, actionable threat intelligence.

- **Tier 1 (Rapid ML Screening)**: Performs lightweight, real-time feature extraction across URL structure, hostname entropy, brand token mismatch, TLS state, and domain age, passing canonical features to an XGBoost model. Every prediction is paired with **TreeSHAP attribution**, surfacing exact feature contributions and neutral semantic interpretations.
- **Tier 2 (Forensic Deep Analysis)**: Triggers an isolated, SSRF-protected network transport to inspect HTTP response headers, trace redirect chains step-by-step, conduct static DOM credential sink audits, and export defanged Indicators of Compromise (IOCs).

Web Shield provides a 3-state risk verdict (`SAFE`, `SUSPICIOUS`, `CRITICAL`), SHAP risk explanations, Unicode/Punycode homoglyph analysis, redirect evidence, credential form audits, defanged IOC outputs, and defensive deep-analysis reports.

---

## Production & Candidate System Status

- **Active Production Default**: **Model V2** (`models/xgb_model_v2.pkl`, 11 features) is configured as the active production predictor in `src/predictor.py` and `app.py`.
- **Shadow Candidate Tier 1**: **V4-J4** (`models/v4_variants/v4_j4/xgb_model_v4_j4.pkl`, 28 features) is integrated as the candidate Tier-1 model paired with **Router-E** in shadow mode (`WEBSHIELD_SHADOW_MODE=true`).

> **Shadow Execution Operational Note**: Shadow inference is isolated from production decision semantics but currently executes synchronously within the request path. Under concurrent load, shadow execution adds request latency. Moving shadow inference to an asynchronous worker/queue is a future production optimization.

---

## Benchmark Metrics & Reconciliation Table

Below is the authoritative summary of evaluation metrics across frozen evaluation benchmarks:

| Evaluation Benchmark | Model / Router | Phishing Recall | FPR | Accuracy / F1 | Total Routing | Non-IP Routing | Tier-2 Recovery | Benchmark Context & Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Auth-426 Holdout** | V4-J4 Candidate | **100.00%** (250/250) | **3.41%** (6/176) | **98.59%** / **98.81%** | **20.66%** (88/426) | **20.66%** | N/A | Auth holdout ($TN=170, FP=6, FN=0, TP=250$). |
| **Primary-452 Holdout** | Tier-1 V4-J4 | **95.60%** (239/250) | **5.94%** (12/202) | **94.91%** / **95.41%** | N/A | N/A | N/A | Primary holdout ($TN=190, FP=12, FN=11, TP=239$). |
| **Primary-452 Holdout** | Offline Hybrid Router-E | **95.60%** (239/250) | **5.94%** (12/202) | **94.91%** / **95.41%** | **57.74%** (261/452) | **5.97%** (27/452) | **0** | Tier 2 evidence enrichment. Zero FN recovery. |

### Benchmark Limitations & Disclosure
* **Auth-426 Structural Collisions**: Auth-426 contains repeated identical 28-feature patterns across distinct real-world URLs. 50 collision groups cover 297 samples; therefore, sample-level metrics do not represent 426 statistically independent feature patterns.
* **Primary-452 IP-Host Distribution**: Primary-452 contains 234 IP-host phishing samples and 0 legitimate IP-host samples. Its 57.74% total routing rate is dominated by IP-host rules ($234/452 = 51.77\%$) and does not represent real production traffic composition.
* **Tier-2 Forensic Claim**: Tier 2 provides static forensic evidence enrichment for selected URLs. In the canonical offline Primary-452 benchmark, Tier 2 did not change the Tier-1 confusion matrix.

---

## System Capabilities & Limitations

### Static Forensics Capabilities (Tier 2 Phase 1)
- Static HTML form parsing and input field analysis.
- Password input field detection.
- Registered-domain form destination relationship auditing (`SAME_ORIGIN`, `SAME_REGISTERED_DOMAIN`, `EXTERNAL_REGISTERED_DOMAIN`).
- Cross-origin external credential POST detection.
- HTTPS-to-HTTP unencrypted credential submission (downgrade attack) detection.
- Redirect chain tracing and step-by-step SSRF revalidation.

### Limitations (Phase 1 Scope)
Current Tier 2 static analysis does **NOT** guarantee detection of:
- JavaScript-only rendered credential forms (e.g. single-page apps rendering forms via AJAX).
- Shadow-DOM or iframe-encapsulated forms.
- Executable binary malware, script payloads (`.exe`, `.ps1`, `.js`, `.dll`).
- Archive files (`.zip`, `.rar`, `.7z`) or image steganography.

### Future Roadmap
- **Phase 2 (Isolated Browser Renderer)**: Headless browser environment to execute client-side JavaScript and render dynamic DOMs for single-page application (SPA) credential flows.
- **Phase 3 (Payload Analysis Service)**: Dedicated sandbox service for deep inspection of executable binaries, scripts, and document archives.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION.git
   cd WEB-SHIELD-PHISHING-DETECTION
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Server**:
   ```bash
   python app.py
   ```
   Access the Web Interface at `http://127.0.0.1:5000/`.

---

## API Usage Examples

### 1. Rapid Tier 1 Analysis (`POST /api/v1/analyze`)
```bash
curl -X POST http://127.0.0.1:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://sso.harvard.edu/cas/login"}'
```

### 2. Forensic Deep Analysis (`POST /api/v1/deep-analyze`)
```bash
curl -X POST http://127.0.0.1:5000/api/v1/deep-analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "http://microsoft-online-secure-auth-100.com/login.php"}'
```

---

## Rollback & Configuration

To toggle active model versions or router configurations:
- **Environment Variable Override**:
  ```bash
  export WEBSHIELD_MODEL_VERSION=v2 # Reverts to Model V2 production baseline
  ```
- **Router Configuration Override**:
  Pass `router_version='D'` or `router_version='E'` to `route_for_tier2()`.

---

## Running Tests

Execute the full automated test suite (114 tests):
```bash
pytest tests/
```
