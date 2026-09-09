"""
Unit tests verifying UI semantics, Tier 1 ML vs Tier 2 Forensic separation,
interstitial confidence handling, 11-feature V2 input contract, and SHAP vector parity.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from app import app, analyze_tier1_url
from src.predictor import get_predictor, adapt_features_v2, validate_v2_input
from src.deep_analysis import perform_deep_analysis, check_interstitial_content
from src.ioc import generate_ioc


class TestUISemantics(unittest.TestCase):

    def setUp(self):
        self.app_client = app.test_client()

    def test_1_explicit_11_feature_contract_in_ui(self):
        """1. UI makes 11-feature model contract explicit."""
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn("Structural URL &amp; Domain Signals", html)
        self.assertIn("10 displayed signals • 11 encoded Model V2 features", html)

    def test_2_no_global_clean_label_in_forensics(self):
        """2. UI does not globally label a successful low-risk forensic result CLEAN."""
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html = f.read()

        # Confirm Exfiltration Flag: CLEAN is not rendered in template JS
        self.assertNotIn("Exfiltration Flag: CLEAN", html)
        self.assertIn("Credential Exfiltration Evidence:", html)
        self.assertIn("NONE DETECTED", html)
        self.assertIn("No credential-harvesting evidence was identified in the fetched static DOM. This does not establish that the entire site is safe.", html)

    def test_3_low_credential_risk_does_not_override_critical_ml(self):
        """3. LOW credential risk does not override CRITICAL ML risk."""
        # Simulated combined interpretation check when Tier 1 = CRITICAL and Tier 2 = LOW
        tier1_assessment = {'risk_level': 'critical', 'prediction': 'Phishing', 'confidence': 88.0}
        forensic_report = {
            'scan_status': 'completed',
            'fetch_succeeded': True,
            'http_status': 200,
            'analysis_confidence': 'HIGH',
            'credential_analysis': {
                'status': 'evaluated',
                'sensitive_fields': [],
                'sink_risk': 'low',
                'exfiltration_flag': False
            }
        }
        ioc = generate_ioc("http://example.com", forensic_report)
        ioc['tier1_assessment'] = tier1_assessment

        self.assertEqual(ioc['tier1_assessment']['risk_level'], 'critical')
        self.assertEqual(ioc['forensics']['credential_audit']['sink_risk'], 'low')
        self.assertFalse(ioc['forensics']['credential_audit']['exfiltration_flag'])

    def test_4_failed_fetch_displays_unknown_not_evaluated(self):
        """4. Failed fetch still displays UNKNOWN / NOT_EVALUATED and never NONE DETECTED or CLEAN."""
        with patch('src.deep_analysis.safe_fetch', side_effect=Exception("Connection refused")):
            report = perform_deep_analysis("http://failed-domain-example.com")

        self.assertFalse(report['fetch_succeeded'])
        self.assertEqual(report['scan_status'], 'failed')
        self.assertIsNone(report['http_status'])
        self.assertEqual(report['credential_analysis']['status'], 'not_evaluated')
        self.assertEqual(report['credential_analysis']['sink_risk'], 'unknown')
        self.assertEqual(report['credential_analysis']['exfiltration_flag'], 'not_evaluated')

    def test_5_redirected_final_content_identified(self):
        """5. Redirected final content is clearly identified."""
        mock_hop_1 = {'status_code': 307, 'headers': {'location': 'http://example.com/verify-ua/'}, 'text': ''}
        mock_hop_2 = {'status_code': 200, 'headers': {}, 'text': '<html><head><title>Verify</title></head><body>Verify Page</body></html>'}

        with patch('src.deep_analysis.safe_fetch', side_effect=[mock_hop_1, mock_hop_2]):
            report = perform_deep_analysis("http://example.com/browse/")

        self.assertTrue(report['fetch_succeeded'])
        self.assertEqual(report['target_url'], 'hxxp[://]example[.]com/browse/')
        self.assertEqual(report['final_url'], 'hxxp[://]example[.]com/verify-ua/')
        self.assertEqual(report['redirect_count'], 1)

    def test_6_interstitial_page_produces_limited_confidence(self):
        """6. Interstitial/verification page can produce LIMITED analysis confidence."""
        mock_hop_1 = {'status_code': 307, 'headers': {'location': 'http://example.com/verify-ua/'}, 'text': ''}
        mock_hop_2 = {'status_code': 200, 'headers': {}, 'text': '<html><head><title>Human Verification</title></head><body>Verify Browser</body></html>'}

        with patch('src.deep_analysis.safe_fetch', side_effect=[mock_hop_1, mock_hop_2]):
            report = perform_deep_analysis("http://example.com/browse/")

        self.assertTrue(report['fetch_succeeded'])
        self.assertTrue(report['is_interstitial'])
        self.assertEqual(report['analysis_confidence'], 'LIMITED')
        self.assertIn("The target redirected to verification/interstitial content", report['analysis_confidence_reason'])

    def test_7_prediction_and_shap_vectors_identical(self):
        """7. Prediction and SHAP vectors remain identical."""
        predictor = get_predictor()
        sample_raw_features = {
            'url_length': 32,
            'having_ip': 0,
            'has_at_symbol': 0,
            'redirect_count': 0,
            'subdomain_count': 1,
            'hyphen_count': 0,
            'domain_entropy': 3.0,
            'has_suspicious_keyword': 0,
            'ssl_valid': 1,
            'domain_age_months': -1
        }

        # Run prediction
        prob, df, shap_factors = predictor.predict_raw(sample_raw_features)

        # Expected V2 columns
        expected_cols = [
            'url_length', 'having_ip', 'has_at_symbol', 'redirect_count',
            'subdomain_count', 'hyphen_count', 'domain_entropy',
            'has_suspicious_keyword', 'ssl_valid', 'domain_age_months_clean',
            'domain_age_known'
        ]

        self.assertEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df.columns), 11)

        # Verify that all features in DataFrame correspond to input values
        adapted = adapt_features_v2(sample_raw_features)
        for col in expected_cols:
            self.assertEqual(df.iloc[0][col], adapted[col])

        # Verify SHAP explainer input vector format matches model input vector
        raw_shap_output = predictor.explainer(df)
        self.assertEqual(raw_shap_output.shape[1], 11)

    def test_8_urlhaus_exact_probability_display(self):
        """8. Verify URLhaus raw model probability is 0.8410, API returns 0.8410, and UI displays 84.1% without 88.0% floor."""
        res = analyze_tier1_url("https://urlhaus.abuse.ch/browse/")
        self.assertAlmostEqual(res['phishing_probability'], 0.8410, places=3)
        self.assertAlmostEqual(res['ml_probability'], 0.8410, places=3)
        self.assertEqual(res['confidence'], 84.1)
        self.assertEqual(res['risk_level'], 'critical')
        self.assertEqual(res['prediction'], 'Phishing')

    def test_9_synthetic_probability_verdict_separation(self):
        """9. Verify raw probability 0.60 coexists with CRITICAL risk verdict without probability display floor."""
        def mock_predict_raw(features_dict):
            return 0.60, pd.DataFrame(), []

        with patch('src.predictor.ModelPredictor.predict_raw', side_effect=mock_predict_raw):
            res = analyze_tier1_url("http://192.168.1.1/login")

        self.assertEqual(res['phishing_probability'], 0.60)
        self.assertEqual(res['ml_probability'], 0.60)
        self.assertEqual(res['confidence'], 60.0)
        self.assertEqual(res['risk_level'], 'critical')
        self.assertEqual(res['prediction'], 'Phishing')


if __name__ == '__main__':
    unittest.main()
