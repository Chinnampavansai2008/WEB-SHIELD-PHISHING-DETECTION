"""
Phase 3 API Integration Test Suite.
Tests Flask REST API endpoints: POST /api/v1/analyze and POST /api/v1/deep-analyze.
"""

import unittest
import json
from app import app


class TestPhase3API(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_api_analyze_safe_domain(self):
        response = self.client.post(
            '/api/v1/analyze',
            data=json.dumps({'url': 'https://google.com'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertEqual(data["status"], "success")
        res_data = data["data"]
        self.assertIn("url", res_data)
        self.assertIn("defanged_url", res_data)
        self.assertIn("risk_level", res_data)
        self.assertIn("top_risk_factors", res_data)
        self.assertIsInstance(res_data["top_risk_factors"], list)

    def test_api_analyze_userinfo_rejection(self):
        response = self.client.post(
            '/api/v1/analyze',
            data=json.dumps({'url': 'http://user:pass@google.com'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")
        self.assertIn("user authentication details", data["message"])

    def test_api_deep_analyze(self):
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
        self.assertEqual(data["ioc_report"]["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
