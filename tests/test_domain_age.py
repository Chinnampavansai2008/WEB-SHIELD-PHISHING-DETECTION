"""
Unit Test Suite for Domain Age -1 (UNKNOWN) Semantics & Status Classification.
"""

import unittest
from unittest.mock import patch
from src.feature_extraction import get_domain_age_status, NEW_DOMAIN_THRESHOLD_MONTHS
from src.explainability import get_semantic_interpretation
from app import analyze_tier1_url


class TestDomainAgeSemantics(unittest.TestCase):

    def test_domain_age_status_classification(self):
        # A & D: Test -1, 0, 3, 6, 84
        self.assertEqual(get_domain_age_status(-1), "unknown")
        self.assertEqual(get_domain_age_status(None), "unknown")
        self.assertEqual(get_domain_age_status(0), "new")
        self.assertEqual(get_domain_age_status(3), "new")
        self.assertEqual(get_domain_age_status(6), "known")
        self.assertEqual(get_domain_age_status(84), "known")

    def test_unknown_domain_age_does_not_trigger_new_domain_rule(self):
        # B: Mock feature extraction returning domain_age_months = -1 with benign URL
        with patch("app.extract_features") as mock_extract, \
             patch("app.model.predict_proba") as mock_proba:
            mock_extract.return_value = {
                'url_length': 20,
                'having_ip': 0,
                'has_at_symbol': 0,
                'redirect_count': 0,
                'subdomain_count': 1,
                'hyphen_count': 0,
                'domain_entropy': 2.5,
                'has_suspicious_keyword': 0,
                'ssl_valid': 1,
                'domain_age_months': -1
            }
            # Low phishing probability 0.10
            mock_proba.return_value = [[0.90, 0.10]]

            res = analyze_tier1_url("https://example-unknown-whois.com/")
            
            self.assertEqual(res['domain_age_status'], "unknown")
            self.assertEqual(res['features']['domain_age_months'], -1)
            # Must NOT be promoted to suspicious purely because domain_age is -1
            self.assertEqual(res['risk_level'], "safe")
            self.assertEqual(res['prediction'], "Legitimate")

    def test_new_domain_age_triggers_new_domain_rule(self):
        # C: Mock domain_age_months = 0 (new domain)
        with patch("app.extract_features") as mock_extract, \
             patch("app.model.predict_proba") as mock_proba:
            mock_extract.return_value = {
                'url_length': 20,
                'having_ip': 0,
                'has_at_symbol': 0,
                'redirect_count': 0,
                'subdomain_count': 1,
                'hyphen_count': 0,
                'domain_entropy': 2.5,
                'has_suspicious_keyword': 0,
                'ssl_valid': 1,
                'domain_age_months': 0
            }
            mock_proba.return_value = [[0.80, 0.20]]

            res = analyze_tier1_url("https://brand-new-domain.com/")
            
            self.assertEqual(res['domain_age_status'], "new")
            self.assertEqual(res['risk_level'], "suspicious")

    def test_shap_semantic_interpretation_for_unknown_domain_age(self):
        # E: Verify SHAP interpretation for -1 is unavailable/unverified
        interp = get_semantic_interpretation("domain_age_months", -1)
        self.assertEqual(interp, "Domain age unavailable / unverified")
        self.assertNotIn("Recently registered", interp)
        self.assertNotIn("newly registered", interp.lower())

    def test_api_representation_includes_domain_age_status(self):
        # F: Expose both domain_age_months and domain_age_status in API
        with patch("app.extract_features") as mock_extract, \
             patch("app.model.predict_proba") as mock_proba:
            mock_extract.return_value = {
                'url_length': 20,
                'having_ip': 0,
                'has_at_symbol': 0,
                'redirect_count': 0,
                'subdomain_count': 1,
                'hyphen_count': 0,
                'domain_entropy': 2.5,
                'has_suspicious_keyword': 0,
                'ssl_valid': 1,
                'domain_age_months': -1
            }
            mock_proba.return_value = [[0.90, 0.10]]

            res = analyze_tier1_url("https://test-api-status.com/")
            self.assertIn("domain_age_status", res)
            self.assertEqual(res["domain_age_status"], "unknown")
            self.assertEqual(res["features"]["domain_age_months"], -1)


if __name__ == "__main__":
    unittest.main()
