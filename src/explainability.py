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


# Neutral human-readable feature labels
FEATURE_NEUTRAL_LABELS = {
    'url_length': 'URL Length',
    'having_ip': 'Hostname Type',
    'has_at_symbol': "'@' Symbol Presence",
    'redirect_count': "Lexical '//' Marker Count",
    'subdomain_count': 'Subdomain Count',
    'hyphen_count': 'Domain Hyphen Count',
    'domain_entropy': 'Domain Character Entropy',
    'has_suspicious_keyword': 'Suspicious Keyword Signal',
    'ssl_valid': 'TLS/SSL Status',
    'domain_age_months': 'Domain Age',
    'domain_age_months_clean': 'Domain Age',
    'domain_age_known': 'Domain Age Verification'
}


def get_semantic_interpretation(feature: str, val: Any) -> str:
    """
    Returns a true semantic security interpretation based strictly on whether
    the raw observed feature value satisfies defined threshold conditions.
    """
    if feature == "url_length":
        length = int(val) if isinstance(val, (int, float, np.number)) else 0
        if length > 75:
            return f"Long URL ({length} characters)"
        return f"Normal-length URL ({length} characters)"

    elif feature == "hyphen_count":
        count = int(val) if isinstance(val, (int, float, np.number)) else 0
        if count > 1:
            return f"Multiple hyphens ({count} hyphens)"
        elif count == 1:
            return "Single hyphen"
        return "No hyphens"

    elif feature == "subdomain_count":
        count = int(val) if isinstance(val, (int, float, np.number)) else 0
        if count > 2:
            return f"Deep subdomain structure ({count} subdomains)"
        return f"Normal subdomain depth ({count} subdomains)"

    elif feature == "domain_entropy":
        entropy = float(val) if isinstance(val, (int, float, np.number)) else 0.0
        if entropy > 3.8:
            return f"High character randomness ({entropy:.2f} bits)"
        return f"Normal character entropy ({entropy:.2f} bits)"

    elif feature in ("domain_age_months", "domain_age_months_clean"):
        age = int(val) if isinstance(val, (int, float, np.number)) else -1
        if age == -1:
            return "Domain age unavailable / unverified"
        elif age == 0:
            return "Domain age unverified or zero months"
        elif age < 6:
            return f"Recently registered domain ({age} months)"
        return f"Established domain ({age} months)"

    elif feature == "domain_age_known":
        known = int(val) if isinstance(val, (int, float, np.number)) else 0
        if known == 1:
            return "Domain age verified"
        return "Domain age unverified / unknown"

    elif feature == "ssl_valid":
        status = int(val) if isinstance(val, (int, float, np.number)) else 0
        if status == 1:
            return "Valid TLS/SSL certificate"
        return "Missing or invalid TLS/SSL certificate"

    elif feature == "having_ip":
        is_ip = int(val) if isinstance(val, (int, float, np.number)) else 0
        if is_ip == 1:
            return "Raw IP address used as host"
        return "Standard domain name host"

    elif feature == "has_at_symbol":
        has_at = int(val) if isinstance(val, (int, float, np.number)) else 0
        if has_at == 1:
            return "'@' symbol present in URL"
        return "No '@' symbol in URL"

    elif feature == "has_suspicious_keyword":
        has_kw = int(val) if isinstance(val, (int, float, np.number)) else 0
        if has_kw == 1:
            return "Suspicious keyword present"
        return "No suspicious keywords"

    elif feature == "redirect_count":
        red = int(val) if isinstance(val, (int, float, np.number)) else 0
        if red > 0:
            return f"Multiple redirects ({red} redirections)"
        return "No redirects detected"

    return str(val)


class ModelExplainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.explainer = None
        self.phishing_class_idx = 1
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
        if os.path.exists(FEATURES_PATH):
            self.feature_names = joblib.load(FEATURES_PATH)

        if self.model is not None:
            # Determine class index for Phishing (class 1)
            if hasattr(self.model, "classes_"):
                classes_list = list(self.model.classes_)
                if 1 in classes_list:
                    self.phishing_class_idx = classes_list.index(1)

            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = shap.Explainer(self.model)

    def get_top_risk_factors(self, feature_dict: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Calculates local SHAP values for a feature dictionary and returns top risk drivers.

        Args:
            feature_dict (dict): Dictionary mapping feature names to extracted numeric values.
            top_k (int): Number of top risk factors to return.

        Returns:
            list of dicts: [
                {
                    'feature': str,
                    'label': str,
                    'observed_value': Any,
                    'shap_value': float,
                    'direction': str,
                    'semantic_interpretation': str,
                    'description': str
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
                shap_vals = shap_vals[:, self.phishing_class_idx]
        else:
            shap_vals = raw_shap[0]
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[:, self.phishing_class_idx]

        items = []
        for i, name in enumerate(self.feature_names):
            val = df.iloc[0][name]
            s_val = float(shap_vals[i])

            # Debug assertion enforcing feature alignment
            assert self.feature_names[i] == name, f"Feature index mismatch at {i}: {self.feature_names[i]} != {name}"

            label = FEATURE_NEUTRAL_LABELS.get(name, name.replace("_", " ").title())
            clean_val = float(val) if isinstance(val, (int, float, np.number)) else val
            if isinstance(clean_val, float) and clean_val.is_integer():
                clean_val = int(clean_val)

            direction = "toward_phishing" if s_val > 0 else "toward_legitimate"
            semantic_interp = get_semantic_interpretation(name, clean_val)

            items.append({
                'feature': name,
                'label': label,
                'observed_value': clean_val,
                'feature_value': clean_val,
                'shap_value': round(s_val, 4),
                'direction': direction,
                'semantic_interpretation': semantic_interp,
                'description': label  # Legacy compatibility
            })

        positive_risks = [item for item in items if item['shap_value'] > 0]
        positive_risks.sort(key=lambda x: x['shap_value'], reverse=True)

        if not positive_risks:
            items.sort(key=lambda x: abs(x['shap_value']), reverse=True)
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
