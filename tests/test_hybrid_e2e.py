"""
End-to-End API Integration & Regression Tests for Hybrid Tier 1 + Tier 2 System.
"""

import unittest
import json
from app import app


class TestHybridE2E(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_predict_endpoint_form(self):
        response = self.client.post(
            '/predict',
            data={'url': 'https://google.com'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Web Shield", response.data)
        self.assertTrue(b"Lexical" in response.data and b"Marker Count" in response.data)

    def test_api_analyze_safe_domain(self):
        response = self.client.post(
            '/api/v1/analyze',
            data=json.dumps({'url': 'https://google.com'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertEqual(data["status"], "success")
        res = data["data"]
        self.assertIn("ml_probability", res)
        self.assertIn("ml_prediction", res)
        self.assertIn("risk_verdict", res)
        self.assertIn("overall_verdict", res)
        self.assertIn("tier2_required", res)
        self.assertIn("routing_reason", res)
        self.assertIn("top_risk_factors", res)

    def test_api_analyze_gst_regression(self):
        response = self.client.post(
            '/api/v1/analyze',
            data=json.dumps({'url': 'https://services.gst.gov.in/services/login'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        res = data["data"]
        
        self.assertIn("ml_probability", res)
        self.assertIn("tier2_required", res)
        self.assertIn("overall_verdict", res)
        self.assertIn("risk_verdict", res)

    def test_api_deep_analyze_hybrid_integration(self):
        response = self.client.post(
            '/api/v1/deep-analyze',
            data=json.dumps({'url': 'https://example.com'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data["status"], "success")
        self.assertIn("target", data)
        self.assertIn("forensics", data)
        self.assertIn("ioc_report", data)
        
        ioc = data["ioc_report"]
        self.assertIn("tier1_assessment", ioc)
        t1 = ioc["tier1_assessment"]
        self.assertIn("ml_probability", t1)
        self.assertIn("risk_verdict", t1)
        self.assertIn("overall_verdict", t1)


if __name__ == "__main__":
    unittest.main()
