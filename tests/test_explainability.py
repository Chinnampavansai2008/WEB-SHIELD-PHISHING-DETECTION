"""
Unit Test Suite for SHAP Explainability & Feature Interpretation (Tier 1).
"""

import unittest
import os
import joblib
import pandas as pd
import numpy as np
from src.explainability import (
    ModelExplainer,
    get_top_risk_factors,
    get_semantic_interpretation,
    FEATURE_NEUTRAL_LABELS
)
from src.feature_extraction import extract_features


class TestExplainability(unittest.TestCase):

    def setUp(self):
        self.explainer = ModelExplainer()

    def test_semantic_interpretation_url_length_32(self):
        interp = get_semantic_interpretation("url_length", 32)
        self.assertEqual(interp, "Normal-length URL (32 characters)")
        self.assertNotIn("Excessively Long", interp)

    def test_semantic_interpretation_long_url(self):
        interp = get_semantic_interpretation("url_length", 120)
        self.assertEqual(interp, "Long URL (120 characters)")

    def test_semantic_interpretation_zero_hyphens(self):
        interp = get_semantic_interpretation("hyphen_count", 0)
        self.assertEqual(interp, "No hyphens")
        self.assertNotIn("Multiple Hyphens", interp)

    def test_semantic_interpretation_multiple_hyphens(self):
        interp = get_semantic_interpretation("hyphen_count", 3)
        self.assertEqual(interp, "Multiple hyphens (3 hyphens)")

    def test_semantic_interpretation_low_entropy(self):
        interp = get_semantic_interpretation("domain_entropy", 2.5)
        self.assertEqual(interp, "Normal character entropy (2.50 bits)")
        self.assertNotIn("High Character Randomness", interp)

    def test_semantic_interpretation_high_entropy(self):
        interp = get_semantic_interpretation("domain_entropy", 4.2)
        self.assertEqual(interp, "High character randomness (4.20 bits)")

    def test_semantic_interpretation_unknown_domain_age(self):
        interp = get_semantic_interpretation("domain_age_months", -1)
        self.assertEqual(interp, "Domain age unavailable / unverified")

    def test_semantic_interpretation_established_domain_age(self):
        interp = get_semantic_interpretation("domain_age_months", 24)
        self.assertEqual(interp, "Established domain (24 months)")

    def test_top_risk_factors_structure_and_neutral_labels(self):
        feat_dict = {
            'url_length': 32,
            'having_ip': 0,
            'has_at_symbol': 0,
            'redirect_count': 0,
            'subdomain_count': 1,
            'hyphen_count': 0,
            'domain_entropy': 2.8,
            'has_suspicious_keyword': 0,
            'ssl_valid': 1,
            'domain_age_months': 48
        }
        factors = get_top_risk_factors(feat_dict, top_k=3)
        self.assertGreater(len(factors), 0)

        for factor in factors:
            self.assertIn('feature', factor)
            self.assertIn('label', factor)
            self.assertIn('observed_value', factor)
            self.assertIn('shap_value', factor)
            self.assertIn('direction', factor)
            self.assertIn('semantic_interpretation', factor)

            # Label must be neutral string, never old hardcoded negative description
            self.assertIn(factor['label'], FEATURE_NEUTRAL_LABELS.values())
            self.assertIn(factor['direction'], ['toward_phishing', 'toward_legitimate'])

    def test_feature_ordering_and_vector_alignment(self):
        features_path = os.path.join("models", "features.pkl")
        expected_features = joblib.load(features_path)

        feat_dict = extract_features("https://urlhaus.abuse.ch/browse/")
        df_features = pd.DataFrame([feat_dict])[expected_features]

        # Verify exact column order
        self.assertEqual(list(df_features.columns), expected_features)

        # Verify explainer uses expected_features order
        self.assertEqual(self.explainer.feature_names, expected_features)

    def test_shap_uses_phishing_class_index(self):
        self.assertEqual(self.explainer.phishing_class_idx, 1)


if __name__ == "__main__":
    unittest.main()
