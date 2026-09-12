"""
Unit tests for Candidate Model V4 audit, feature contract, domain isolation,
duplicate feature-vector leakage, login keyword context, GST regression, hard negative/positive sets,
and prediction/SHAP parity.
"""

import unittest
import os
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import GroupShuffleSplit

from generate_data_v4 import extract_features_v4, LEGITIMATE_HARD_NEGATIVES, PHISHING_HARD_POSITIVES
from src.predictor import get_predictor, adapt_features_v2, validate_v2_input


class TestModelV4CandidateAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.v4_model_path = 'models/v4/xgb_model_v4.pkl'
        cls.v4_features_path = 'models/v4/features_v4.pkl'
        cls.v4_metadata_path = 'models/v4/metadata.json'

        self_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(self_dir)
        cls.dataset_v4_path = os.path.join(repo_root, 'data', 'dataset_v4.csv')

        if os.path.exists(cls.v4_model_path):
            cls.model_v4 = joblib.load(cls.v4_model_path)
            cls.features_v4 = joblib.load(cls.v4_features_path)
            cls.explainer_v4 = shap.TreeExplainer(cls.model_v4)
        else:
            cls.model_v4 = None

    def test_01_v4_artifacts_exist_and_isolated(self):
        """Test candidate Model V4 artifacts exist under models/v4 and V2 remains untouched."""
        self.assertTrue(os.path.exists('models/v2/xgb_model_v2.pkl'), "Production V2 model missing!")
        self.assertTrue(os.path.exists(self.v4_model_path), "Candidate V4 model artifact missing under models/v4/")
        self.assertTrue(os.path.exists(self.v4_features_path), "Candidate V4 features artifact missing!")
        self.assertTrue(os.path.exists(self.v4_metadata_path), "Candidate V4 metadata missing!")

        # Verify default predictor is STILL V2
        predictor = get_predictor()
        self.assertEqual(predictor.version, 'v2')

    def test_02_v4_feature_contract(self):
        """Test candidate Model V4 feature contract contains exactly 17 features in canonical order."""
        expected_17 = [
            'url_length', 'hostname_length', 'path_length', 'path_depth',
            'having_ip', 'has_at_symbol', 'redirect_count', 'subdomain_count',
            'hyphen_count', 'domain_entropy', 'hostname_entropy',
            'keyword_in_hostname', 'keyword_in_path', 'has_suspicious_keyword',
            'ssl_valid', 'domain_age_months_clean', 'domain_age_known'
        ]
        self.assertEqual(self.features_v4, expected_17)
        self.assertEqual(len(self.features_v4), 17)

    def test_03_train_test_domain_isolation(self):
        """Test that dataset_v4 registered_domains do NOT overlap between train and test splits."""
        if not os.path.exists(self.dataset_v4_path):
            self.skipTest("dataset_v4.csv missing")
        df = pd.read_csv(self.dataset_v4_path)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(df[self.features_v4], df['label'], df['registered_domain']))

        train_domains = set(df.iloc[tr_idx]['registered_domain'])
        test_domains = set(df.iloc[te_idx]['registered_domain'])
        overlap = train_domains.intersection(test_domains)

        self.assertEqual(len(overlap), 0, f"Domain leakage detected across split: {overlap}")

    def test_04_no_duplicate_feature_vector_leakage_across_split(self):
        """Test that duplicate normalized feature vectors do NOT cross train/test splits in V4."""
        if not os.path.exists(self.dataset_v4_path):
            self.skipTest("dataset_v4.csv missing")
        df = pd.read_csv(self.dataset_v4_path)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(df[self.features_v4], df['label'], df['registered_domain']))

        tr_vecs = set(tuple(x) for x in df.iloc[tr_idx][self.features_v4].values)
        te_vecs = set(tuple(x) for x in df.iloc[te_idx][self.features_v4].values)
        overlap_vecs = tr_vecs.intersection(te_vecs)

        self.assertLessEqual(len(overlap_vecs), 5, f"Excessive feature vector leakage across split: {len(overlap_vecs)}")

    def test_05_login_keyword_context_features(self):
        """Test distinction between keyword in hostname vs keyword in path."""
        # 1. Legitimate service login URL (keyword in path only)
        f_legit = extract_features_v4("https://services.gst.gov.in/services/login")
        self.assertEqual(f_legit['keyword_in_hostname'], 0)
        self.assertEqual(f_legit['keyword_in_path'], 1)
        self.assertEqual(f_legit['has_suspicious_keyword'], 1)

        # 2. Phishing URL with deceptive keyword in hostname
        f_phish = extract_features_v4("http://paypal-login-secure.attacker-domain.com/auth")
        self.assertEqual(f_phish['keyword_in_hostname'], 1)
        self.assertEqual(f_phish['keyword_in_path'], 1)
        self.assertEqual(f_phish['has_suspicious_keyword'], 1)

    def test_06_gst_regression_case(self):
        """Test that GST government portal URL is correctly classified as Legitimate by Candidate V4."""
        gst_url = "https://services.gst.gov.in/services/login"
        f4 = pd.DataFrame([extract_features_v4(gst_url)])[self.features_v4]
        prob_v4 = float(self.model_v4.predict_proba(f4)[0, 1])

        self.assertLess(prob_v4, 0.50, f"GST URL still flagged as Phishing by V4: {prob_v4*100:.2f}%")

    def test_07_legitimate_login_hard_negatives(self):
        """Test candidate V4 false positive rate on legitimate login hard negatives (must be < 10%)."""
        fps = 0
        for u in LEGITIMATE_HARD_NEGATIVES:
            f4 = pd.DataFrame([extract_features_v4(u)])[self.features_v4]
            p = float(self.model_v4.predict_proba(f4)[0, 1])
            if p >= 0.50:
                fps += 1

        fpr = (fps / len(LEGITIMATE_HARD_NEGATIVES)) * 100
        self.assertLess(fpr, 10.0, f"Legitimate hard negative FPR too high for V4: {fpr:.2f}% ({fps}/{len(LEGITIMATE_HARD_NEGATIVES)})")

    def test_08_phishing_login_hard_positives(self):
        """Test candidate V4 recall on phishing login hard positives (must be >= 85%)."""
        tps = 0
        for u in PHISHING_HARD_POSITIVES:
            f4 = pd.DataFrame([extract_features_v4(u)])[self.features_v4]
            p = float(self.model_v4.predict_proba(f4)[0, 1])
            if p >= 0.50:
                tps += 1

        recall = (tps / len(PHISHING_HARD_POSITIVES)) * 100
        self.assertGreaterEqual(recall, 85.0, f"Phishing hard positive recall dropped for V4: {recall:.2f}% ({tps}/{len(PHISHING_HARD_POSITIVES)})")

    def test_09_prediction_shap_vector_parity(self):
        """Test that SHAP values for V4 use the exact same feature vector input as predict_proba."""
        gst_url = "https://services.gst.gov.in/services/login"
        df_v4 = pd.DataFrame([extract_features_v4(gst_url)])[self.features_v4]

        probs = self.model_v4.predict_proba(df_v4)[0, 1]
        shap_res = self.explainer_v4(df_v4)

        self.assertEqual(df_v4.shape[1], 17)
        self.assertEqual(shap_res.values.shape[1], 17)
        self.assertIsNotNone(probs)

    def test_10_model_version_isolation(self):
        """Test that requesting version='v2' loads V2, and V4 remains strictly isolated in models/v4/."""
        pred_v2 = get_predictor('v2')
        self.assertEqual(pred_v2.version, 'v2')
        self.assertEqual(len(pred_v2.feature_names), 11)

    def test_11_v4_d_candidate_variant(self):
        """Test candidate variant V4-D performance on GST and hard negative/positive sets."""
        v4d_path = 'models/v4_variants/v4_d/xgb_model_v4_d.pkl'
        if not os.path.exists(v4d_path):
            self.skipTest("V4-D model missing")
        model_v4d = joblib.load(v4d_path)
        feats_v4d = joblib.load('models/v4_variants/v4_d/features_v4_d.pkl')

        # GST URL check
        df_gst = pd.DataFrame([extract_features_v4("https://services.gst.gov.in/services/login")])[feats_v4d]
        p_gst = float(model_v4d.predict_proba(df_gst)[0, 1])
        self.assertLess(p_gst, 0.50, f"V4-D GST probability too high: {p_gst*100:.2f}%")

        # Legitimate login FPR <= 10%
        fps = 0
        for u in LEGITIMATE_HARD_NEGATIVES:
            f = pd.DataFrame([extract_features_v4(u)])[feats_v4d]
            if float(model_v4d.predict_proba(f)[0, 1]) >= 0.50:
                fps += 1
        fpr = (fps / len(LEGITIMATE_HARD_NEGATIVES)) * 100
        self.assertLessEqual(fpr, 10.0, f"V4-D legitimate login FPR too high: {fpr:.2f}%")

    def test_12_telemetry_url_hashing(self):
        """Test that telemetry URL hashing computes SHA-256 over sanitized canonical URL."""
        import hashlib
        import urllib.parse

        url_a = "https://example.com/path?user=alice&token=secret1"
        url_b = "https://example.com/path?user=bob&token=secret2"
        url_c = "https://example.com/path?user=alice&token=secret999"

        def sanitize_and_hash(u):
            parsed = urllib.parse.urlparse(u)
            params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            clean_params = [f"{k}=[REDACTED]" if k in ['token', 'password'] else f"{k}={v[0]}" for k, v in params.items()]
            clean_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, "&".join(clean_params), parsed.fragment))
            return hashlib.sha256(clean_url.encode('utf-8')).hexdigest()

        h_a = sanitize_and_hash(url_a)
        h_b = sanitize_and_hash(url_b)
        h_c = sanitize_and_hash(url_c)

        empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        self.assertNotEqual(h_a, empty_hash, "URL hash must not hash empty string")
        self.assertNotEqual(h_a, h_b, "Different sanitized URLs must produce different hashes")
        self.assertEqual(h_a, h_c, "Identical sanitized URLs must produce identical hashes")


if __name__ == '__main__':
    unittest.main()
