"""
End-to-End Application Route & Integration Tests for Web Shield.
Tests Flask endpoints /api/v1/analyze, /api/v1/deep-analyze, and /predict.
"""

import json
import unittest
from app import app


class TestE2ERoutes(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_analyze_endpoint_github(self):
        r = self.client.post('/api/v1/analyze', json={'url': 'https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION'})
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['model_version'], 'v2')
        self.assertIn('phishing_probability', data['data'])
        self.assertIn('risk_level', data['data'])

    def test_deep_analyze_endpoint_urlhaus(self):
        r = self.client.post('/api/v1/deep-analyze', json={'url': 'https://urlhaus.abuse.ch/browse/'})
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertIn('hxxps[://]urlhaus[.]abuse[.]ch/browse/', data['target']['defanged_url'])
        self.assertIn(data['forensics']['scan_status'], ['completed', 'partial', 'failed'])

    def test_analyze_endpoint_urlhaus(self):
        r = self.client.post('/api/v1/analyze', json={'url': 'https://urlhaus.abuse.ch/browse/'})
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data.decode('utf-8'))['data']
        self.assertEqual(data['model_version'], 'v2')
        self.assertEqual(data['model_feature_count'], 11)
        self.assertIn('top_risk_factors', data)
        self.assertGreaterEqual(len(data['top_risk_factors']), 1)

    def test_predict_endpoint(self):
        r = self.client.post('/predict', data={'url': 'https://en.wikipedia.org/wiki/Phishing'})
        self.assertEqual(r.status_code, 200)

    def test_analyze_missing_url(self):
        r = self.client.post('/api/v1/analyze', json={})
        self.assertEqual(r.status_code, 400)
        data = json.loads(r.data.decode('utf-8'))
        self.assertEqual(data['status'], 'error')

    def test_analyze_invalid_scheme(self):
        r = self.client.post('/api/v1/analyze', json={'url': 'ftp://example.com/file'})
        self.assertEqual(r.status_code, 400)
        data = json.loads(r.data.decode('utf-8'))
        self.assertEqual(data['status'], 'error')


if __name__ == '__main__':
    unittest.main()
