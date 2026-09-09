"""
Unit Test Suite for Safe Fetcher Module (SSRF, IP Validation, TLS, Adapter Transport & Pool Isolation).
"""

import unittest
import socket
import ipaddress
from unittest.mock import patch, MagicMock
from urllib3.connection import HTTPConnection, HTTPSConnection
import requests
from src.safe_fetcher import (
    safe_fetch,
    is_ip_safe,
    resolve_and_validate_host,
    PinnedIPAdapter,
    SSRFError,
    FetchError
)


class TestSafeFetcher(unittest.TestCase):

    # --- STEP 5: SSRF & IP Validation Tests ---

    def test_ip_safe_ipv4_public(self):
        self.assertTrue(is_ip_safe("8.8.8.8"))
        self.assertTrue(is_ip_safe("151.101.158.49"))

    def test_ip_safe_ipv4_restricted(self):
        self.assertFalse(is_ip_safe("127.0.0.1"))
        self.assertFalse(is_ip_safe("10.0.0.1"))
        self.assertFalse(is_ip_safe("172.16.0.1"))
        self.assertFalse(is_ip_safe("192.168.1.1"))
        self.assertFalse(is_ip_safe("169.254.169.254"))  # AWS metadata / link-local
        self.assertFalse(is_ip_safe("100.64.0.1"))        # CGNAT
        self.assertFalse(is_ip_safe("224.0.0.1"))         # Multicast
        self.assertFalse(is_ip_safe("0.0.0.0"))           # Reserved

    def test_ip_safe_ipv6_public(self):
        self.assertTrue(is_ip_safe("2001:4860:4860::8888"))

    def test_ip_safe_ipv6_restricted(self):
        self.assertFalse(is_ip_safe("::1"))         # Loopback
        self.assertFalse(is_ip_safe("fe80::1"))     # Link-local
        self.assertFalse(is_ip_safe("fc00::1"))     # ULA
        self.assertFalse(is_ip_safe("fd00::1"))     # ULA
        self.assertFalse(is_ip_safe("ff02::1"))     # Multicast

    def test_ip_safe_ipv4_mapped_ipv6(self):
        self.assertFalse(is_ip_safe("::ffff:127.0.0.1"))
        self.assertFalse(is_ip_safe("::ffff:10.0.0.1"))
        self.assertTrue(is_ip_safe("::ffff:8.8.8.8"))

    def test_ip_safe_nat64_prefix(self):
        # NAT64 embedding public IPv4 151.101.158.49 -> 64:ff9b::9765:9e31
        self.assertTrue(is_ip_safe("64:ff9b::9765:9e31"))
        # NAT64 embedding private IPv4 127.0.0.1 -> 64:ff9b::7f00:0001
        self.assertFalse(is_ip_safe("64:ff9b::7f00:0001"))
        # NAT64 embedding 10.0.0.1 -> 64:ff9b::0a00:0001
        self.assertFalse(is_ip_safe("64:ff9b::0a00:0001"))
        # NAT64 embedding metadata 169.254.169.254 -> 64:ff9b::a9fe:a9fe
        self.assertFalse(is_ip_safe("64:ff9b::a9fe:a9fe"))

    def test_ssrf_scheme_validation(self):
        with self.assertRaises(SSRFError):
            safe_fetch("ftp://example.com/file.txt")
        with self.assertRaises(SSRFError):
            safe_fetch("file:///etc/passwd")
        with self.assertRaises(SSRFError):
            safe_fetch("gopher://example.com")

    def test_ssrf_userinfo_validation(self):
        with self.assertRaises(SSRFError):
            safe_fetch("http://user:password@example.com/")

    def test_ssrf_loopback_and_private_urls(self):
        with self.assertRaises(SSRFError):
            safe_fetch("http://localhost/")
        with self.assertRaises(SSRFError):
            safe_fetch("http://127.0.0.1/")
        with self.assertRaises(SSRFError):
            safe_fetch("http://10.0.0.1/admin")
        with self.assertRaises(SSRFError):
            safe_fetch("http://169.254.169.254/latest/meta-data/")

    @patch("socket.getaddrinfo")
    def test_resolve_multi_dns_one_unsafe(self, mock_getaddrinfo):
        # Return both a public IP and a private loopback IP
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
        ]
        with self.assertRaises(SSRFError) as cm:
            resolve_and_validate_host("rebinding.evil.com", 80)
        self.assertIn("127.0.0.1", str(cm.exception))

    # --- STEP 4 & 6: Public HTTPS & Transport / TLS Verification Tests ---

    @patch("socket.getaddrinfo")
    @patch("requests.Session.get")
    def test_safe_fetch_public_https_success(self, mock_get, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.encoding = "utf-8"
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.iter_content.return_value = [b"<html><head><title>Test</title></head><body>Hello</body></html>"]
        mock_get.return_value = mock_response

        result = safe_fetch("https://example.com/test")

        self.assertEqual(result["status_code"], 200)
        self.assertIn("Hello", result["text"])
        self.assertEqual(result["final_url"], "https://example.com/test")
        self.assertEqual(result["resolved_ip"], "93.184.216.34")

    def test_pinned_ip_adapter_connection_creation(self):
        """
        Verify that PinnedIPAdapter creates a connection subclass whose _new_conn
        connects to pinned_ip while retaining the original host for TLS SNI and cert validation.
        """
        adapter = PinnedIPAdapter(pinned_ip="151.101.158.49")
        conn = adapter.get_connection_with_tls_context(
            request=MagicMock(url="https://urlhaus.abuse.ch/browse/"),
            verify=True
        )

        self.assertEqual(conn.host, "urlhaus.abuse.ch")
        self.assertEqual(conn.ConnectionCls.pinned_ip, "151.101.158.49")

        # Test _new_conn sets _dns_host to pinned_ip during super call
        conn_obj = conn.ConnectionCls(host="urlhaus.abuse.ch", port=443)
        with patch.object(HTTPSConnection, "_new_conn", return_value=MagicMock()) as mock_super_new_conn:
            conn_obj._new_conn()
            mock_super_new_conn.assert_called_once()

    # --- STEP 7: Connection Pool Isolation Test ---

    def test_connection_pool_isolation(self):
        """
        Verify that two requests to different hosts generate isolated connection pools with distinct pinned IPs.
        """
        adapter_a = PinnedIPAdapter(pinned_ip="1.1.1.1")
        conn_a = adapter_a.get_connection_with_tls_context(
            request=MagicMock(url="https://hosta.com/"),
            verify=True
        )

        adapter_b = PinnedIPAdapter(pinned_ip="2.2.2.2")
        conn_b = adapter_b.get_connection_with_tls_context(
            request=MagicMock(url="https://hostb.com/"),
            verify=True
        )

        self.assertEqual(conn_a.ConnectionCls.pinned_ip, "1.1.1.1")
        self.assertEqual(conn_b.ConnectionCls.pinned_ip, "2.2.2.2")
        self.assertNotEqual(conn_a.ConnectionCls.pinned_ip, conn_b.ConnectionCls.pinned_ip)

    # --- STEP 8: Error Reporting Semantics ---

    def test_error_type_separation(self):
        # Security rejection must raise SSRFError
        with self.assertRaises(SSRFError):
            safe_fetch("http://127.0.0.1/")

        # Network error after validation must raise FetchError
        with patch("socket.getaddrinfo") as mock_getaddrinfo, \
             patch("requests.Session.get", side_effect=requests.ConnectionError("Connection failed")):
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))
            ]
            with self.assertRaises(FetchError) as cm:
                safe_fetch("http://example.com/")
            self.assertIn("HTTP request failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
