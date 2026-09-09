"""
End-to-End Application Route & Integration Test Runner.
Executes e2e route tests for /api/v1/analyze, /api/v1/deep-analyze, and /predict.
"""

import sys
import unittest
from tests.test_e2e_routes import TestE2ERoutes

if __name__ == '__main__':
    unittest.main()
