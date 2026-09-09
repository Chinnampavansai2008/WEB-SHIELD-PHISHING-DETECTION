"""
Shadow Comparison Utility for Model V1 vs Model V2 Validation.
Enables side-by-side comparison of prediction probabilities and SHAP explanations.
"""

from typing import Dict, Any
from src.url_normalizer import normalize_url
from src.feature_extraction import extract_features
from src.predictor import get_predictor


def run_shadow_comparison(raw_url: str) -> Dict[str, Any]:
    """
    Runs both Model V1 and Model V2 on the same raw URL feature vector
    and returns a structured comparison report.
    """
    norm_res = normalize_url(raw_url)
    canonical_url = norm_res["canonical_url"]
    features_dict = extract_features(canonical_url)

    # Predict with Model V1
    pred_v1 = get_predictor(force_version="v1")
    prob_v1, _, shap_v1 = pred_v1.predict_raw(features_dict)
    verdict_v1 = "Phishing" if prob_v1 >= 0.50 else "Legitimate"

    # Predict with Model V2
    pred_v2 = get_predictor(force_version="v2")
    prob_v2, _, shap_v2 = pred_v2.predict_raw(features_dict)
    verdict_v2 = "Phishing" if prob_v2 >= 0.50 else "Legitimate"

    delta = prob_v2 - prob_v1

    return {
        "url": canonical_url,
        "v1_probability": round(prob_v1, 4),
        "v1_verdict": verdict_v1,
        "v2_probability": round(prob_v2, 4),
        "v2_verdict": verdict_v2,
        "probability_delta": round(delta, 4),
        "v1_top_shap": shap_v1[:2],
        "v2_top_shap": shap_v2[:2]
    }
