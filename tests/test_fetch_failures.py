"""
Unit Test Suite for Deep Forensic Fetch Failures and Not-Evaluated DOM States.
Verifies that network/fetch failures do not fabricate HTTP status 0, do not run credential analysis on empty HTML,
and return scan_status='failed' with sink_risk='unknown' and exfiltration_flag='not_evaluated'.
"""

import unittest
from src.deep_analysis import perform_deep_analysis
from src.safe_fetcher import safe_fetch, FetchError, SSRFError


class TestFetchFailures(unittest.TestCase):

    def test_nonexistent_domain_fetch_failure(self):
        url = "http://nonexistent-domain-999999999.invalid"
        report = perform_deep_analysis(url, timeout_budget=2.0)

        # 1. Scan status must be 'failed'
        self.assertEqual(report["scan_status"], "failed")
        self.assertFalse(report["fetch_succeeded"])

        # 2. HTTP status must be None (never 0)
        self.assertIsNone(report["http_status"])
        
        # 3. Redirect timeline hop status must be None (never 0)
        timeline = report.get("redirect_timeline", [])
        self.assertGreater(len(timeline), 0)
        self.assertIsNone(timeline[0]["status"])

        # 4. Credential Analysis must be NOT_EVALUATED and UNKNOWN
        cred = report["credential_analysis"]
        self.assertEqual(cred["status"], "not_evaluated")
        self.assertEqual(cred["sink_risk"], "unknown")
        self.assertEqual(cred["exfiltration_flag"], "not_evaluated")
        self.assertIn("Final HTML could not be retrieved", cred["reason"])

    def test_ssrf_block_fetch_failure(self):
        url = "http://127.0.0.1/admin"
        report = perform_deep_analysis(url, timeout_budget=2.0)

        self.assertEqual(report["scan_status"], "failed")
        self.assertFalse(report["fetch_succeeded"])
        self.assertIsNone(report["http_status"])
        
        cred = report["credential_analysis"]
        self.assertEqual(cred["status"], "not_evaluated")
        self.assertEqual(cred["sink_risk"], "unknown")
        self.assertEqual(cred["exfiltration_flag"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
