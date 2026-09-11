"""
Unit Test Suite for Hybrid Tier 1 + Tier 2 Triage, Routing, and Evidence Aggregation.
"""

import unittest
from src.hybrid_triage import (
    check_auth_context,
    route_for_tier2,
    evaluate_tier2_evidence_severity,
    compute_hybrid_verdict
)


class TestHybridTriage(unittest.TestCase):

    def test_routing_low_ml_benign(self):
        # 1. ml_probability = 0.20, normal benign URL => no Tier 2
        res = route_for_tier2(0.20, {'ssl_valid': 1, 'having_ip': 0, 'has_suspicious_keyword': 0}, url="https://example.com/about")
        self.assertFalse(res['tier2_required'])
        self.assertIn("High-confidence", res['routing_reason'])

    def test_routing_ambiguous_ml(self):
        # 2. ml_probability = 0.45 => Tier 2 required
        res = route_for_tier2(0.45, {'ssl_valid': 1, 'having_ip': 0}, url="https://example.com/page")
        self.assertTrue(res['tier2_required'])
        self.assertIn("ambiguous range", res['routing_reason'])

    def test_routing_high_ml(self):
        # 3. ml_probability = 0.80 => obvious phishing (Tier 2 optional/not required)
        res = route_for_tier2(0.80, {'ssl_valid': 0, 'having_ip': 0}, url="http://example-phish.com/path/index.html")
        self.assertFalse(res['tier2_required'])

    def test_routing_https_auth_escalation(self):
        # 4. ml_probability = 0.25, HTTPS auth-like URL => Tier 2 required
        res = route_for_tier2(0.25, {'ssl_valid': 1, 'having_ip': 0, 'has_suspicious_keyword': 0}, url="https://auth-gate.com/login")
        self.assertTrue(res['tier2_required'])
        self.assertTrue(any("authentication" in r.lower() for r in res['routing_reasons']))

    def test_routing_shared_host_auth_escalation(self):
        # 5. ml_probability = 0.25, shared-host auth URL => Tier 2 required
        res = route_for_tier2(0.25, {'ssl_valid': 1, 'having_ip': 0}, url="https://my-app.workers.dev/login")
        self.assertTrue(res['tier2_required'])
        self.assertTrue(any("shared-hosting" in r.lower() for r in res['routing_reasons']))

    def test_severity_external_credential_post(self):
        # HIGH RISK: password field + external cross-origin form POST
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'external_form_actions': ['https://attacker-domain.xyz/harvest.php'],
            'insecure_form_action': False
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report)
        self.assertEqual(severity, "HIGH")
        self.assertIn("external cross-origin domain", evidence[0])

    def test_severity_https_to_http_credentials(self):
        # HIGH RISK: HTTPS page submitting credentials over HTTP
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'external_form_actions': [],
            'insecure_form_action': True
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report)
        self.assertEqual(severity, "HIGH")
        self.assertIn("unencrypted HTTP", evidence[0])

    def test_same_registered_domain_auth_form(self):
        # same registered domain POST (login.example.com -> auth.example.com) => MEDIUM severity
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'https://auth.example.com/login', 'destination_type': 'SAME_REGISTERED_DOMAIN'}],
                'insecure_form_action': False
            }
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report, severity_variant='C', url="https://login.example.com/signin")
        self.assertEqual(severity, "MEDIUM")

    def test_external_registered_domain_credential_post(self):
        # external registered domain POST (login.example.com -> attacker.net) => HIGH severity
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'https://attacker.net/harvest.php', 'destination_type': 'EXTERNAL_REGISTERED_DOMAIN'}],
                'insecure_form_action': False
            }
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report, severity_variant='C', url="https://login.example.com/signin")
        self.assertEqual(severity, "HIGH")

    def test_oauth_redirect_flow(self):
        # OAuth flow with redirect_uri, client_id => federated auth context prevents false HIGH severity
        oauth_url = "https://idp.company.com/oauth/v2/authorize?client_id=12345&redirect_uri=https://app.partner.com/callback&response_type=code&state=xyz"
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'https://app.partner.com/callback', 'destination_type': 'EXTERNAL_REGISTERED_DOMAIN'}],
                'insecure_form_action': False
            }
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report, severity_variant='C', url=oauth_url)
        self.assertEqual(severity, "MEDIUM")

    def test_saml_form_flow(self):
        # SAML SSO flow with SAMLRequest, RelayState
        saml_url = "https://sso.enterprise.com/idp/profile/SAML2/Redirect/SSO?SAMLRequest=fZJdb4Iw&RelayState=cookie"
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'https://auth.enterprise.com/saml/consume', 'destination_type': 'SAME_REGISTERED_DOMAIN'}],
                'insecure_form_action': False
            }
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report, severity_variant='C', url=saml_url)
        self.assertEqual(severity, "MEDIUM")

    def test_university_shibboleth_flow(self):
        # University Shibboleth SSO
        univ_url = "https://shibboleth.university.edu/idp/profile/SAML2/Redirect/SSO?SAMLRequest=abc&RelayState=xyz"
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'https://shibboleth.university.edu/idp/Authn/UserPassword', 'destination_type': 'SAME_ORIGIN'}],
                'insecure_form_action': False
            }
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report, severity_variant='C', url=univ_url)
        self.assertEqual(severity, "MEDIUM")

    def test_bank_federation_flow(self):
        # Bank OAuth authorization flow
        bank_url = "https://identity.globalbank.com/as/authorization.oauth2?response_type=code&client_id=fintech_app&redirect_uri=https://fintech.com/oauth/callback"
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'https://identity.globalbank.com/as/authorization.oauth2', 'destination_type': 'SAME_ORIGIN'}],
                'insecure_form_action': False
            }
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report, severity_variant='C', url=bank_url)
        self.assertEqual(severity, "MEDIUM")

    def test_https_to_http_credential_post(self):
        # HTTPS -> HTTP credential POST => HIGH severity even with federated parameters
        downgrade_url = "https://secure-login.com/auth?client_id=123&redirect_uri=http://insecure.com/cb"
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'http://insecure.com/post', 'destination_type': 'EXTERNAL_REGISTERED_DOMAIN'}],
                'insecure_form_action': True
            }
        }
        severity, evidence = evaluate_tier2_evidence_severity(t2_report, severity_variant='C', url=downgrade_url)
        self.assertEqual(severity, "HIGH")

    def test_confidence_gating_low_ml_medium_severity(self):
        # Confidence gating: low ML prob (0.18) + MEDIUM severity => SUSPICIOUS, Legitimate
        t2_report = {
            'scan_status': 'completed',
            'has_password_field': True,
            'credential_analysis': {
                'sensitive_fields': ['password'],
                'has_login_form': True,
                'form_actions': [{'action_url': 'https://auth.example.com/login', 'destination_type': 'SAME_REGISTERED_DOMAIN'}],
                'insecure_form_action': False
            }
        }
        verdict = compute_hybrid_verdict(0.18, False, {'ssl_valid': 1}, tier2_report=t2_report, url="https://login.example.com/signin", severity_variant='C')
        self.assertEqual(verdict['risk_verdict'], "SUSPICIOUS")
        self.assertEqual(verdict['overall_verdict'], "Legitimate")

    def test_router_e_vs_router_d_ambiguous_boundaries(self):
        # Router-E ambiguous band is 0.40 <= P <= 0.60
        # P = 0.38: True for Router-D, False for Router-E
        res_d = route_for_tier2(0.38, {'having_ip': 0}, url="https://example.com/page", router_version='D')
        res_e = route_for_tier2(0.38, {'having_ip': 0}, url="https://example.com/page", router_version='E')
        self.assertTrue(res_d['tier2_required'])
        self.assertFalse(res_e['tier2_required'])

        # P = 0.45: True for both Router-D and Router-E
        res_d_45 = route_for_tier2(0.45, {'having_ip': 0}, url="https://example.com/page", router_version='D')
        res_e_45 = route_for_tier2(0.45, {'having_ip': 0}, url="https://example.com/page", router_version='E')
        self.assertTrue(res_d_45['tier2_required'])
        self.assertTrue(res_e_45['tier2_required'])

    def test_router_e_vs_router_d_auth_weak_class(self):
        # Router-E auth weak class is 0.25 <= P < 0.35, Router-D is 0.15 <= P < 0.35
        # P = 0.20 + auth URL: True for Router-D, False for Router-E
        res_d = route_for_tier2(0.20, {'having_ip': 0}, url="https://example-portal.com/login", router_version='D')
        res_e = route_for_tier2(0.20, {'having_ip': 0}, url="https://example-portal.com/login", router_version='E')
        self.assertTrue(res_d['tier2_required'])
        self.assertFalse(res_e['tier2_required'])

        # P = 0.28 + auth URL: True for both Router-D and Router-E
        res_d_28 = route_for_tier2(0.28, {'having_ip': 0}, url="https://example-portal.com/login", router_version='D')
        res_e_28 = route_for_tier2(0.28, {'having_ip': 0}, url="https://example-portal.com/login", router_version='E')
        self.assertTrue(res_d_28['tier2_required'])
        self.assertTrue(res_e_28['tier2_required'])


if __name__ == "__main__":
    unittest.main()


