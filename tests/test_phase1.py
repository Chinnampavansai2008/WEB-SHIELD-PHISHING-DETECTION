"""
Phase 1 Unit & Integration Test Suite.
Tests URL Normalization, Homoglyph Detection, Safe Fetcher (SSRF Protection), and SHAP Explainability.
"""

import unittest
from src.url_normalizer import normalize_url, InvalidURLError
from src.homoglyph import analyze_homoglyphs
from src.safe_fetcher import safe_fetch, resolve_and_validate_host, SSRFError
from src.explainability import get_top_risk_factors


class TestURLNormalizer(unittest.TestCase):

    def test_valid_urls(self):
        res1 = normalize_url("https://google.com")
        self.assertEqual(res1["canonical_url"], "https://google.com/")
        self.assertEqual(res1["hostname"], "google.com")
        self.assertEqual(res1["scheme"], "https")

        res2 = normalize_url("http://example.com:8080/test?b=2&a=1")
        self.assertEqual(res2["canonical_url"], "http://example.com:8080/test?a=1&b=2")
        self.assertEqual(res2["hostname"], "example.com")
        self.assertEqual(res2["scheme"], "http")

    def test_userinfo_rejection(self):
        with self.assertRaises(InvalidURLError):
            normalize_url("http://admin:pass@google.com")

        with self.assertRaises(InvalidURLError):
            normalize_url("http://user@google.com")

    def test_disallowed_schemes(self):
        with self.assertRaises(InvalidURLError):
            normalize_url("ftp://files.example.com")

        with self.assertRaises(InvalidURLError):
            normalize_url("javascript:alert(1)")


class TestHomoglyph(unittest.TestCase):

    def test_punycode_detection(self):
        result = analyze_homoglyphs("xn--80akhbyknj4f.com")
        self.assertTrue(result["is_punycode"])
        self.assertEqual(result["impersonation_risk"], "high")

    def test_brand_impersonation(self):
        result1 = analyze_homoglyphs("paypal-login-secure.com")
        self.assertEqual(result1["suspected_brand"], "paypal")
        self.assertEqual(result1["impersonation_risk"], "high")

        result2 = analyze_homoglyphs("g00gle.com")
        self.assertEqual(result2["suspected_brand"], "google")
        self.assertEqual(result2["impersonation_risk"], "high")

    def test_legitimate_domain(self):
        result = analyze_homoglyphs("google.com")
        self.assertFalse(result["is_punycode"])
        self.assertEqual(result["impersonation_risk"], "low")


class TestSafeFetcher(unittest.TestCase):

    def test_ssrf_loopback_rejection(self):
        with self.assertRaises(SSRFError):
            resolve_and_validate_host("127.0.0.1")

        with self.assertRaises(SSRFError):
            safe_fetch("http://127.0.0.1/admin")

    def test_ssrf_private_ip_rejection(self):
        with self.assertRaises(SSRFError):
            resolve_and_validate_host("10.0.0.1")

        with self.assertRaises(SSRFError):
            resolve_and_validate_host("192.168.1.1")

    def test_ssrf_metadata_ip_rejection(self):
        with self.assertRaises(SSRFError):
            resolve_and_validate_host("169.254.169.254")

        with self.assertRaises(SSRFError):
            safe_fetch("http://169.254.169.254/latest/meta-data/")


class TestExplainability(unittest.TestCase):

    def test_shap_generation(self):
        sample_features = {
            'url_length': 120,
            'having_ip': 1,
            'has_at_symbol': 1,
            'redirect_count': 2,
            'subdomain_count': 3,
            'hyphen_count': 2,
            'domain_entropy': 4.5,
            'has_suspicious_keyword': 1,
            'ssl_valid': 0,
            'domain_age_months': 1
        }
        risk_factors = get_top_risk_factors(sample_features, top_k=3)
        
        self.assertEqual(len(risk_factors), 3)
        for factor in risk_factors:
            self.assertIn("feature", factor)
            self.assertIn("description", factor)
            self.assertIn("shap_value", factor)
            self.assertIn("feature_value", factor)
            self.assertIsInstance(factor["shap_value"], float)


if __name__ == "__main__":
    unittest.main()
