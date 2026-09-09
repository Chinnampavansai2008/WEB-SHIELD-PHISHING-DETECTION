"""
Web Shield Predictor & Model Architecture Abstraction.
Handles versioned model loading (V1 / V2), canonical V2 feature adaptation,
strict input validation, unscaled XGBoost inference, and TreeSHAP explainability.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Tuple

from src.explainability import FEATURE_NEUTRAL_LABELS, get_semantic_interpretation


class ModelInputError(ValueError):
    """Raised when model input features violate expected feature contract."""
    pass


def adapt_features_v2(raw_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapts raw structural feature dictionary to exact V2 feature contract:
    - Replaces domain_age_months with domain_age_months_clean and domain_age_known.
    - Preserves all 9 standard structural features.
    """
    age = raw_features.get('domain_age_months', -1)
    v2_dict = dict(raw_features)
    
    # Missing value transformation matching V2 dataset generation & training
    v2_dict['domain_age_months_clean'] = 0 if age == -1 else age
    v2_dict['domain_age_known'] = 0 if age == -1 else 1

    if 'domain_age_months' in v2_dict:
        del v2_dict['domain_age_months']

    return v2_dict


def validate_v2_input(features_dict: Dict[str, Any], expected_features: List[str]):
    """
    Validates V2 feature vector against strict model input requirements:
    - Exact feature count
    - Exact feature names
    - Numeric types without NaN or Inf
    """
    if len(features_dict) != len(expected_features):
        raise ModelInputError(
            f"Feature count mismatch: expected {len(expected_features)} features for V2, got {len(features_dict)}."
        )

    for name in expected_features:
        if name not in features_dict:
            raise ModelInputError(f"Missing required feature key for Model V2: '{name}'.")
        
        val = features_dict[name]
        if val is None or not isinstance(val, (int, float, np.number, bool)):
            raise ModelInputError(f"Invalid non-numeric feature value for '{name}': {val}")
        
        float_val = float(val)
        if np.isnan(float_val) or np.isinf(float_val):
            raise ModelInputError(f"Feature '{name}' contains NaN or Inf: {val}")


class ModelPredictor:
    """
    Unified Predictor Abstraction managing Model V1 & Model V2 lifecycles.
    """

    def __init__(self, target_version: str = None, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.models_dir = models_dir

        # Determine target version from environment or parameter, default to "v2"
        if target_version is None:
            target_version = os.getenv("WEBSHIELD_MODEL_VERSION", "v2").lower().strip()

        self.requested_version = target_version
        self.version = target_version
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metadata = {}
        self.uses_scaler = False
        self.explainer = None
        self.phishing_class_idx = 1
        
        self.load_model()

    def load_model(self):
        """Loads target model artifacts and initializes TreeSHAP explainer once."""
        if self.requested_version == "v2":
            v2_dir = os.path.join(self.models_dir, "v2")
            model_path = os.path.join(v2_dir, "xgb_model_v2.pkl")
            features_path = os.path.join(v2_dir, "features_v2.pkl")
            metadata_path = os.path.join(v2_dir, "metadata.json")

            if os.path.exists(model_path) and os.path.exists(features_path):
                try:
                    self.model = joblib.load(model_path)
                    self.feature_names = joblib.load(features_path)
                    self.uses_scaler = False
                    self.scaler = None
                    self.version = "v2"

                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path, 'r') as f:
                            self.metadata = json.load(f)

                    # Dynamic phishing class index lookup
                    if hasattr(self.model, "classes_"):
                        classes_list = list(self.model.classes_)
                        if 1 in classes_list:
                            self.phishing_class_idx = classes_list.index(1)

                    # TreeExplainer initialization
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                    except Exception:
                        self.explainer = shap.Explainer(self.model)

                    return
                except Exception as e:
                    print(f"Warning: Failed to load Model V2 ({e}). Falling back to Model V1.")

            # Controlled fallback to V1 if V2 load fails
            self._load_v1(fallback=True)

        else:
            self._load_v1(fallback=False)

    def _load_v1(self, fallback: bool = False):
        model_path = os.path.join(self.models_dir, "xgb_model.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        features_path = os.path.join(self.models_dir, "features.pkl")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
            raise FileNotFoundError("❌ Model V1 artifacts missing in 'models/' directory!")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        self.uses_scaler = True
        self.version = "v1-fallback" if fallback else "v1"

        if hasattr(self.model, "classes_"):
            classes_list = list(self.model.classes_)
            if 1 in classes_list:
                self.phishing_class_idx = classes_list.index(1)

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.Explainer(self.model)

    def predict_raw(self, raw_features: Dict[str, Any]) -> Tuple[float, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Canonical prediction path:
        1. Adapt raw features if V2.
        2. Validate feature contract.
        3. Predict probability using unscaled (V2) or scaled (V1) vector.
        4. Compute SHAP factors using the EXACT SAME input vector.
        """
        if self.version.startswith("v2"):
            # Feature adaptation
            v2_features = adapt_features_v2(raw_features)
            validate_v2_input(v2_features, self.feature_names)
            
            # DataFrame in authoritative feature order
            df = pd.DataFrame([v2_features])[self.feature_names]
            
            # Assert StandardScaler is NOT called for V2
            assert not self.uses_scaler, "Error: StandardScaler must NOT be used for Model V2!"
            assert self.scaler is None, "Error: Scaler instance must be None for Model V2!"

            # Predict probability
            probs = self.model.predict_proba(df)[0]
            phishing_prob = float(probs[self.phishing_class_idx])

            # SHAP computation on exact same unscaled DataFrame
            shap_factors = self._compute_shap(df, raw_features)
            return phishing_prob, df, shap_factors

        else:
            # Model V1 path
            df = pd.DataFrame([raw_features])[self.feature_names]
            scaled_array = self.scaler.transform(df)
            scaled_df = pd.DataFrame(scaled_array, columns=self.feature_names)

            probs = self.model.predict_proba(scaled_df)[0]
            phishing_prob = float(probs[self.phishing_class_idx])

            shap_factors = self._compute_shap(scaled_df, raw_features, is_scaled=True)
            return phishing_prob, df, shap_factors

    def _compute_shap(self, input_df: pd.DataFrame, raw_features: Dict[str, Any], is_scaled: bool = False, top_k: int = 3) -> List[Dict[str, Any]]:
        """Computes SHAP factors cleanly aligning feature values with neutral labels."""
        if self.explainer is None:
            return []

        raw_shap = self.explainer(input_df)
        
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
            val = input_df.iloc[0][name]
            s_val = float(shap_vals[i])

            label = FEATURE_NEUTRAL_LABELS.get(name, name.replace("_", " ").title())
            clean_val = float(val) if isinstance(val, (int, float, np.number)) else val
            if isinstance(clean_val, float) and clean_val.is_integer():
                clean_val = int(clean_val)

            direction = "toward_phishing" if s_val > 0 else "toward_legitimate"
            semantic_interp = get_semantic_interpretation(name, raw_features.get(name, clean_val))

            items.append({
                'feature': name,
                'label': label,
                'observed_value': clean_val,
                'shap_value': round(s_val, 4),
                'direction': direction,
                'semantic_interpretation': semantic_interp,
                'description': label
            })

        positive_risks = [item for item in items if item['shap_value'] > 0]
        positive_risks.sort(key=lambda x: x['shap_value'], reverse=True)

        if not positive_risks:
            items.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            return items[:top_k]

        return positive_risks[:top_k]


# Global singleton instance
_predictor_instance = None


def get_predictor(force_version: str = None) -> ModelPredictor:
    global _predictor_instance
    if _predictor_instance is None or (force_version and _predictor_instance.version != force_version):
        _predictor_instance = ModelPredictor(target_version=force_version)
    return _predictor_instance
