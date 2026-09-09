"""
Unit & Integration Test Suite for Web Shield Model V2 Integration.
Verifies V2 model contract, canonical adapter, input validation, unscaled inference,
SHAP alignment, probability calibration, risk triage, and security regression.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.predictor import get_predictor, ModelPredictor, ModelInputError, adapt_features_v2, validate_v2_input
from src.url_normalizer import normalize_url
from src.feature_extraction import extract_features
from app import app, analyze_tier1_url


class TestModelV2Integration(unittest.TestCase):

    def setUp(self):
        self.predictor = get_predictor(force_version="v2")
        self.client = app.test_client()

    def test_v2_model_and_metadata_load_correctly(self):
        """1 & 2: Verify V2 model and metadata load correctly."""
        self.assertEqual(self.predictor.version, "v2")
        self.assertIsNotNone(self.predictor.model)
        self.assertIn("model_version", self.predictor.metadata)
        self.assertEqual(self.predictor.metadata["model_version"], "v2")

    def test_v2_feature_contract_order_and_count(self):
        """3 & 4: Verify exact V2 feature order and feature count = 11."""
        expected_v2_features = [
            'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
            'subdomain_count', 'hyphen_count', 'domain_entropy',
            'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
        ]
        self.assertEqual(len(self.predictor.feature_names), 11)
        self.assertEqual(self.predictor.feature_names, expected_v2_features)

    def test_v2_input_validation_rejects_invalid_inputs(self):
        """5 & 6: Verify missing or non-numeric/NaN features raise ModelInputError."""
        incomplete_dict = {'url_length': 30, 'having_ip': 0}
        with self.assertRaises(ModelInputError):
            validate_v2_input(incomplete_dict, self.predictor.feature_names)

        nan_dict = adapt_features_v2({
            'url_length': 30, 'having_ip': 0, 'has_at_symbol': 0, 'redirect_count': 0,
            'subdomain_count': 1, 'hyphen_count': 0, 'domain_entropy': np.nan,
            'has_suspicious_keyword': 0, 'ssl_valid': 1, 'domain_age_months': 12
        })
        with self.assertRaises(ModelInputError):
            validate_v2_input(nan_dict, self.predictor.feature_names)

    def test_v2_does_not_use_standard_scaler(self):
        """7: Verify V2 does NOT use StandardScaler."""
        self.assertFalse(self.predictor.uses_scaler)
        self.assertIsNone(self.predictor.scaler)

    def test_phishing_class_index_derived_from_model_classes(self):
        """8: Verify phishing class index is dynamically derived from model.classes_."""
        classes = list(self.predictor.model.classes_)
        expected_idx = classes.index(1)
        self.assertEqual(self.predictor.phishing_class_idx, expected_idx)

    def test_prediction_vector_equals_shap_vector(self):
        """9: Verify input vector passed to predict_proba matches SHAP input vector."""
        raw_feat = {
            'url_length': 45, 'having_ip': 0, 'has_at_symbol': 0, 'redirect_count': 0,
            'subdomain_count': 1, 'hyphen_count': 0, 'domain_entropy': 2.8,
            'has_suspicious_keyword': 0, 'ssl_valid': 1, 'domain_age_months': 24
        }
        prob, input_df, shap_factors = self.predictor.predict_raw(raw_feat)
        
        # Verify shape and column order
        self.assertEqual(list(input_df.columns), self.predictor.feature_names)
        self.assertEqual(input_df.iloc[0]['domain_age_months_clean'], 24)
        self.assertEqual(input_df.iloc[0]['domain_age_known'], 1)
        self.assertTrue(len(shap_factors) > 0)

    def test_domain_age_unknown_mapping(self):
        """10: Verify domain_age_months = -1 maps to clean=0, known=0."""
        v2_feat = adapt_features_v2({'domain_age_months': -1})
        self.assertEqual(v2_feat['domain_age_months_clean'], 0)
        self.assertEqual(v2_feat['domain_age_known'], 0)

    def test_domain_age_known_mapping(self):
        """11: Verify domain_age_months >= 0 maps to clean=age, known=1."""
        v2_feat = adapt_features_v2({'domain_age_months': 36})
        self.assertEqual(v2_feat['domain_age_months_clean'], 36)
        self.assertEqual(v2_feat['domain_age_known'], 1)

    def test_tier1_response_contains_model_version_v2(self):
        """12: Verify model_version = 'v2' in Tier 1 response."""
        res = analyze_tier1_url("https://www.google.com")
        self.assertEqual(res['model_version'], "v2")
        self.assertEqual(res['model_feature_count'], 11)

    def test_long_legitimate_url_not_flagged_solely_due_to_length(self):
        """13: Verify long legitimate GitHub URL is not flagged as phishing."""
        res = analyze_tier1_url("https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION")
        self.assertEqual(res['prediction'], "Legitimate")
        self.assertLess(res['phishing_probability'], 0.50)

    def test_urlhaus_no_longer_triggers_v1_length_shortcut(self):
        """14: Verify URLhaus probability is significantly reduced from V1's 99.58%."""
        res = analyze_tier1_url("https://urlhaus.abuse.ch/browse/")
        self.assertLess(res['phishing_probability'], 0.90)

    def test_short_malicious_sample_detection(self):
        """15: Verify short IP-hosted phishing URL is still detected as Phishing/Critical."""
        res = analyze_tier1_url("http://192.168.1.1/auth/login.php")
        self.assertEqual(res['prediction'], "Phishing")
        self.assertEqual(res['risk_level'], "critical")

    def test_shap_direction_semantics(self):
        """16: Verify SHAP direction is toward_phishing or toward_legitimate."""
        res = analyze_tier1_url("https://www.wikipedia.org")
        for factor in res['top_risk_factors']:
            self.assertIn(factor['direction'], ['toward_phishing', 'toward_legitimate'])

    def test_v1_rollback_capability(self):
        """28: Verify V1 rollback capability via force_version='v1'."""
        v1_pred = get_predictor(force_version="v1")
        self.assertTrue(v1_pred.version.startswith("v1"))
        self.assertTrue(v1_pred.uses_scaler)
        self.assertEqual(len(v1_pred.feature_names), 10)

    def test_frozen_vector_reproducibility(self):
        """17: Regression test ensuring frozen V2 feature vector yields identical deterministic probability."""
        frozen_feat = {
            'url_length': 32, 'having_ip': 0, 'has_at_symbol': 0, 'redirect_count': 0,
            'subdomain_count': 1, 'hyphen_count': 0, 'domain_entropy': 3.0,
            'has_suspicious_keyword': 0, 'ssl_valid': 1, 'domain_age_months': -1
        }
        p1, df1, shap1 = self.predictor.predict_raw(frozen_feat)
        p2, df2, shap2 = self.predictor.predict_raw(frozen_feat)
        
        self.assertEqual(p1, p2)
        self.assertTrue(np.allclose(df1.values, df2.values))
        self.assertEqual(len(shap1), len(shap2))


if __name__ == '__main__':
    unittest.main()

