"""
SHAP Explainability Module for Phishing Detector.
Provides local model predictions explainability using SHAP TreeExplainer.
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Any


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")


# Human-readable risk factor mappings
FEATURE_RISK_DESCRIPTIONS = {
    'having_ip': 'Raw IP Address used as Hostname',
    'has_at_symbol': "Contains '@' symbol in URL authority",
    'ssl_valid': 'Missing, Invalid, or Unverifiable TLS/SSL Certificate',
    'domain_age_months': 'Recently Registered or Short-Lived Domain Name',
    'url_length': 'Excessively Long URL Structure',
    'domain_entropy': 'High Character Randomness/Entropy in Domain Name',
    'has_suspicious_keyword': 'Contains Suspicious Security or Account Keywords',
    'subdomain_count': 'Excessive Subdomains in URL Structure',
    'hyphen_count': 'Multiple Hyphens in Domain Name',
    'redirect_count': 'Multiple URL Redirects Detected'
}


class ModelExplainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
        if os.path.exists(FEATURES_PATH):
            self.feature_names = joblib.load(FEATURES_PATH)

        if self.model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = shap.Explainer(self.model)

    def get_top_risk_factors(self, feature_dict: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Calculates local SHAP values for a feature dictionary and returns top positive risk drivers.

        Args:
            feature_dict (dict): Dictionary mapping feature names to extracted numeric values.
            top_k (int): Number of top risk factors to return.

        Returns:
            list of dicts: [
                {
                    'feature': str,
                    'description': str,
                    'shap_value': float,
                    'feature_value': Any
                }, ...
            ]
        """
        if self.model is None or self.scaler is None or not self.feature_names or self.explainer is None:
            self._load_artifacts()

        if not self.feature_names:
            return []

        # Create DataFrame matching feature names order
        df = pd.DataFrame([feature_dict])[self.feature_names]
        scaled_array = self.scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_array, columns=self.feature_names)

        # Calculate SHAP values
        raw_shap = self.explainer(scaled_df)
        
        if hasattr(raw_shap, "values"):
            shap_vals = raw_shap.values[0]
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[:, 1]
        else:
            shap_vals = raw_shap[0]

        items = []
        for name, val, s_val in zip(self.feature_names, df.iloc[0].values, shap_vals):
            items.append({
                'feature': name,
                'description': FEATURE_RISK_DESCRIPTIONS.get(name, name.replace("_", " ").title()),
                'shap_value': float(s_val),
                'feature_value': float(val) if isinstance(val, (int, float, np.number)) else val
            })

        positive_risks = [item for item in items if item['shap_value'] > 0]
        positive_risks.sort(key=lambda x: x['shap_value'], reverse=True)

        if not positive_risks:
            items.sort(key=lambda x: x['shap_value'], reverse=True)
            return items[:top_k]

        return positive_risks[:top_k]


# Module singleton
_explainer_instance = None


def get_explainer() -> ModelExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = ModelExplainer()
    return _explainer_instance


def get_top_risk_factors(feature_dict: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    explainer = get_explainer()
    return explainer.get_top_risk_factors(feature_dict, top_k=top_k)
