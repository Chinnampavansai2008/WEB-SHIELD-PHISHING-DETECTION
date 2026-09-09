# Web Shield — Live Demonstration & Judge Walkthrough Guide

## Overview

This guide provides a step-by-step walkthrough for demonstrating Web Shield live to hackathon judges, security engineers, or project reviewers.

---

## 1. Environment Setup

Launch the server locally:
```bash
python app.py
```
Open `http://127.0.0.1:5000/` in a web browser.

---

## 2. Test Case Scenarios

### Scenario 1: Standard Legitimate Homepage
- **Input URL**: `https://google.com`
- **Expected Outcome**:
  - **Verdict**: Legitimate
  - **Risk Level**: `SAFE` (Green badge)
  - **Phishing Probability**: $< 10\%$
  - **Top SHAP Signal**: Negative SHAP contribution toward legitimate.

### Scenario 2: IP-Hosted Phishing Target
- **Input URL**: `http://192.168.1.1/login.php`
- **Expected Outcome**:
  - **Verdict**: Phishing
  - **Risk Level**: `CRITICAL` (Red badge)
  - **Phishing Probability**: $> 99\%$
  - **Top SHAP Signal**: Positive SHAP contribution driven by `having_ip = 1`.

### Scenario 3: Complex Legitimate URL (Deep Path)
- **Input URL**: `https://en.wikipedia.org/wiki/Phishing`
- **Expected Outcome**:
  - **Verdict**: Phishing / High Prob *(Demonstrates Model V2 URL-length feature behavior)*
  - **SHAP Explanation**: `url_length` surfaces as top contributor toward phishing.
  - **Judge Talk Track**: Explain how TreeSHAP transparently surfaces feature attribution, showing why the model arrived at this score.

### Scenario 4: Tier 2 Forensic Deep Analysis (URLhaus Reference)
- **Input URL**: `https://urlhaus.abuse.ch/browse/`
- **Action**: Submit URL and click **"Run Tier 2 Deep Analysis"**.
- **Expected Outcome**:
  - **Defanged URL**: `hxxps[://]urlhaus[.]abuse[.]ch/browse/`
  - **Tier 2 Status**: `completed` (HTTP status `200`)
  - **Credential Audit**: `COMPLETED`, `sink_risk = LOW / UNKNOWN`
  - **Export**: JSON IOC download available.

---

## 3. Demonstrating API Integration

You can demonstrate REST API capabilities using `curl` or Postman:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION"}'
```
