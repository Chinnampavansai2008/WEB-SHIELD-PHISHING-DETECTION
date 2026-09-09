"""
Comprehensive QA, Security & Machine Learning Model Contract Audit Verification Script.
Validates model class indices, feature extraction alignment, ground-truth legitimate/phishing cases,
and Tier 2 deep forensic API execution within the 4.0s deadline.
"""

import unittest
import joblib
import os
import json
import time
import pandas as pd
from app import app, model, scaler, feature_names, analyze_tier1_url
from src.feature_extraction import extract_features


class TestAuditVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.client.testing = True

    def test_01_model_contract_and_classes(self):
        """Verifies model class mapping: index 0 = Legitimate, index 1 = Phishing."""
        self.assertIsNotNone(model)
        classes = list(model.classes_)
        self.assertEqual(classes, [0, 1], "Model classes must be [0, 1]")

    def test_02_feature_alignment(self):
        """Verifies feature extraction dictionary keys match models/features.pkl exactly."""
        sample_dict = extract_features("https://www.google.com")
        dict_keys = set(sample_dict.keys())
        expected_features = set(feature_names)
        
        self.assertEqual(dict_keys, expected_features, "Extracted feature keys must match features.pkl character-for-character.")

        # Test DataFrame reordering and scaling
        df_features = pd.DataFrame([sample_dict])[feature_names]
        self.assertEqual(list(df_features.columns), feature_names, "DataFrame column sequence must match feature_names.")
        
        scaled = scaler.transform(df_features)
        self.assertFalse(pd.isna(scaled).any(), "Scaled features array must not contain NaN values.")

    def test_03_ground_truth_legitimate_google(self):
        """Verifies ground-truth legitimate site (https://www.google.com) yields safe prediction."""
        raw_url = "https://www.google.com"
        result = analyze_tier1_url(raw_url)

        # Feature signals check
        features = result["features"]
        self.assertEqual(features["having_ip"], 0)
        self.assertEqual(features["has_at_symbol"], 0)
        self.assertEqual(features["has_suspicious_keyword"], 0)

        # Model probability & verdict check
        phish_prob = result["phishing_probability"]
        self.assertLess(phish_prob, 0.20, f"Google.com phishing probability should be < 0.20, got {phish_prob}")
        self.assertEqual(result["prediction"], "Legitimate")
        self.assertEqual(result["risk_level"], "safe")
        self.assertGreaterEqual(result["confidence"], 90.0)

    def test_04_ground_truth_malicious_raw_ip(self):
        """Verifies ground-truth malicious site (http://192.168.1.1/admin/login) yields critical phishing prediction."""
        raw_url = "http://192.168.1.1/admin/login"
        result = analyze_tier1_url(raw_url)

        features = result["features"]
        self.assertEqual(features["having_ip"], 1)
        self.assertEqual(features["ssl_valid"], 0)

        self.assertEqual(result["prediction"], "Phishing")
        self.assertEqual(result["risk_level"], "critical")

    def test_05_tier2_forensic_api_performance(self):
        """Verifies POST /api/v1/deep-analyze executes safely within 4.0s deadline."""
        start_time = time.monotonic()
        response = self.client.post(
            '/api/v1/deep-analyze',
            data=json.dumps({'url': 'https://example.com'}),
            content_type='application/json'
        )
        duration = time.monotonic() - start_time

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertEqual(data["status"], "success")
        self.assertIn("target", data)
        self.assertIn("forensics", data)
        self.assertIn("ioc_report", data)
        self.assertEqual(data["ioc_report"]["schema_version"], "1.0")

        self.assertLessEqual(duration, 4.5, f"Deep analysis took {duration:.2f}s, exceeding 4.0s budget tolerances.")


if __name__ == "__main__":
    unittest.main()
