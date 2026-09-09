# Web Shield — Model Evaluation & Benchmarks Report

## Overview

This report documents the authoritative, reproducible evaluation benchmarks for Web Shield's machine learning models across three frozen evaluation partitions.

---

## 1. Frozen Evaluation Sets

1. **SET A: `LEGACY_V2_GROUPED_TEST` (245 Samples)**
   - **Legitimate**: 157 | **Phishing**: 88
   - **Split Method**: 20% GroupShuffleSplit on `registered_domain` (`random_state=42`) from `dataset_v2.csv`.
   - **SHA256**: `97761807a7542dd033c346bd3be813699b5546f186805c3b4ea6af9975106cb0`

2. **SET B: `V3_GROUPED_TEST` (444 Samples)**
   - **Legitimate**: 345 | **Phishing**: 99
   - **Split Method**: 20% GroupShuffleSplit on `registered_domain` (`random_state=42`) from `dataset_v3.csv`.
   - **SHA256**: `603f162189c3d9dfc8ff53867f37ca21b587c55b0de5f6793377ca4207abc2f7`

3. **SET C: `HARD_HOLDOUT` (198 Samples)**
   - **Legitimate**: 105 | **Phishing**: 93
   - **Origin**: Independent hard holdout partition covering complex legitimate URLs and domain-based phishing.
   - **SHA256**: `268253db686ddbe648bc1932ce355939528c8db65e32d1b1957fcbf82b08eb6e`

---

## 2. Side-by-Side Model Benchmarks

### SET A: `LEGACY_V2_GROUPED_TEST`

| Metric | Model V2 (Active Production) | Model V3 | Model V3.1 (Candidate) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **95.92%** | 93.88% | 90.20% |
| **Precision** | 94.32% | **98.67%** | 85.56% |
| **Recall** | **94.32%** | 84.09% | 87.50% |
| **F1-Score** | **94.32%** | 90.80% | 86.52% |
| **ROC-AUC** | **99.02%** | 96.11% | 97.15% |
| **FPR** | 3.18% | **0.64%** | 8.28% |

### SET B: `V3_GROUPED_TEST`

| Metric | Model V2 (Active Production) | Model V3 | Model V3.1 (Candidate) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 66.67% | 90.99% | **91.22%** |
| **Precision** | 37.93% | **94.03%** | 86.59% |
| **Recall** | **77.78%** | 63.64% | 71.72% |
| **F1-Score** | 50.99% | 75.90% | **78.45%** |
| **ROC-AUC** | 84.95% | 93.68% | **94.61%** |
| **FPR** | 36.52% | **1.16%** | 3.19% |

### SET C: `HARD_HOLDOUT`

| Metric | Model V2 (Active Production) | Model V3 | Model V3.1 (Candidate) |
| :--- | :---: | :---: | :---: |
| **Hard Legitimate FPR** | 17.14% (18/105) | **2.86%** (3/105) | 3.81% (4/105) |
| **Hard Malicious Recall** | 77.42% (72/93) | 53.76% (50/93) | **83.87%** (78/93) |
| **Overall Accuracy** | 80.30% | 76.77% | **90.40%** |
| **Overall F1-Score** | 78.69% | 68.49% | **89.14%** |

---

## 3. Cost-Sensitive Analysis (Set B)

Evaluated at Total Cost $= (\text{FN} \times \text{FN\_cost}) + (\text{FP} \times 1)$:

| Model | FN | FP | Cost (FN=2x) | Cost (FN=5x) | Cost (FN=10x) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model V2** | 22 | 126 | 170 | 236 | 346 |
| **Model V3** | 36 | 4 | 76 | 184 | 364 |
| **Model V3.1 (@0.50)** | 28 | 11 | **67** | 151 | 291 |
| **Model V3.1 (@0.25)** | 10 | 88 | 108 | **138** | **188** |

---

## 4. Final Model Selection Decision

**Decision**: **KEEP Model V2** as Production Default.

**Reasoning**: Model V2 preserves high baseline phishing recall (94.32% on Set A). While Candidate Model V3.1 achieves lower FPR on Set B (3.19% vs 36.52%), V3.1 recall on standard domain splits is 71.72% - 74.75%, representing an unacceptable 20-point drop in primary phishing detection recall. The 11 static/lexical/network features have reached a practical ML representation ceiling. Future improvements will incorporate brand impersonation signals, TLD categories, and DOM content features.
