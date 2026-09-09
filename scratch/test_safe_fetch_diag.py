import sys
import os
sys.path.insert(0, os.path.abspath("."))
import traceback
import socket
import ssl
import ipaddress
import requests
from src.safe_fetcher import safe_fetch, resolve_and_validate_host, is_ip_safe

url = 'https://urlhaus.abuse.ch/browse/'
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import socket
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.util import connection

url = 'https://urlhaus.abuse.ch/browse/'
target_ip = '151.101.158.49'

print("Testing PinnedIPAdapter using urllib3 ConnectionPool subclassing...")

class PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    def __init__(self, pinned_ip, *args, **kwargs):
        self.pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def _new_conn(self):
        conn = super()._new_conn()
        # Custom connection wrapping or socket creation
        return conn

class PinnedIPAdapter(HTTPAdapter):
    def __init__(self, pinned_ip: str, **kwargs):
        self.pinned_ip = pinned_ip
        super().__init__(**kwargs)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        conn = super().get_connection_with_tls_context(request, verify, proxies=proxies, cert=cert)
        # Override connection pool's _new_conn or socket creation function:
        pinned_ip = self.pinned_ip
        
        def _pinned_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, socket_options=None):
            # Pass (pinned_ip, address[1]) as socket target!
            return connection.create_connection((pinned_ip, address[1]), timeout, source_address, socket_options)

        # Set connection_pool_kw directly on the HTTPSConnectionPool instance!
        conn.connection_pool_kw = dict(conn.connection_pool_kw) if getattr(conn, 'connection_pool_kw', None) else {}
        conn.connection_pool_kw['create_connection'] = _pinned_create_connection
        return conn

session = requests.Session()
adapter = PinnedIPAdapter(pinned_ip=target_ip)
session.mount("https://", adapter)

try:
    res = session.get(url, headers={"User-Agent": "WebShield-Phishing-Scanner/1.0"}, timeout=5.0, allow_redirects=False)
    print("SUCCESS! Status Code:", res.status_code)
    print("Response text length:", len(res.text))
except Exception as e:
    print("FAILED:", type(e), e)
    import traceback
    traceback.print_exc()





