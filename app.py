"""
Web Shield Phishing Detection & Deep Forensics API.
Unified Flask backend orchestrating Tier 1 triage, SHAP explainability, and Tier 2 deep forensics.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.url_normalizer import normalize_url, InvalidURLError
from src.feature_extraction import extract_features
from src.homoglyph import analyze_homoglyphs
from src.explainability import get_top_risk_factors
from src.deep_analysis import perform_deep_analysis
from src.ioc import generate_ioc
from src.defang import defang_url
from src.safe_fetcher import SSRFError

app = Flask(__name__)

# -------------------------------------------------------------
# 1. LOAD MODEL ARTIFACTS
# -------------------------------------------------------------
model_path = os.path.join('models', 'xgb_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')
features_path = os.path.join('models', 'features.pkl')

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
    raise FileNotFoundError("❌ Model files missing in 'models/' directory! Please run train_model.py first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)


# -------------------------------------------------------------
# 2. UNIFIED TIER 1 SERVICE FUNCTION
# -------------------------------------------------------------
def analyze_tier1_url(raw_url: str) -> dict:
    """
    Unified Tier 1 service function:
    1. Normalizes and validates raw URL.
    2. Extracts 10 feature vector.
    3. Runs XGBoost model probability inference.
    4. Evaluates homoglyph / brand impersonation risk.
    5. Computes top SHAP risk factors.
    6. Determines 3-state risk level ('safe', 'suspicious', 'critical').
    """
    if not raw_url or not isinstance(raw_url, str) or not raw_url.strip():
        raise InvalidURLError("URL parameter cannot be empty.")

    norm_res = normalize_url(raw_url)
    canonical_url = norm_res["canonical_url"]
    hostname = norm_res["hostname"]
    defanged = defang_url(canonical_url)

    # Feature extraction
    features_dict = extract_features(canonical_url)

    # Scaling & XGBoost Model Inference
    df_features = pd.DataFrame([features_dict])[feature_names]
    scaled_features = scaler.transform(df_features)
    probabilities = model.predict_proba(scaled_features)[0]
    phishing_prob = float(probabilities[1])

    # Homoglyph Analysis
    homoglyph_res = analyze_homoglyphs(hostname)
    homoglyph_risk = homoglyph_res.get('impersonation_risk', 'low')
    is_puny = homoglyph_res.get('is_punycode', False)

    # SHAP Explainability
    top_risk_factors = get_top_risk_factors(features_dict, top_k=3)

    # Three-state risk level triage logic
    having_ip = features_dict.get('having_ip', 0)
    domain_age = features_dict.get('domain_age_months', 0)

    if phishing_prob >= 0.85 or homoglyph_risk == 'high' or having_ip == 1:
        risk_level = 'critical'
    elif (0.35 <= phishing_prob < 0.85) or domain_age < 1 or is_puny:
        risk_level = 'suspicious'
    else:
        risk_level = 'safe'

    is_phishing = (phishing_prob >= 0.5) or (risk_level == 'critical')
    verdict = 'Phishing' if is_phishing else 'Legitimate'
    confidence = round((phishing_prob if is_phishing else (1.0 - phishing_prob)) * 100, 1)

    if risk_level == 'critical':
        recommendation = "DANGER: High-risk phishing or homoglyph impersonation site detected! Do NOT interact or enter credentials."
    elif risk_level == 'suspicious':
        recommendation = "WARNING: Suspicious indicators detected. Exercise caution before entering sensitive details."
    else:
        recommendation = "SAFE: No critical phishing or homoglyph risk indicators detected."

    return {
        'url': canonical_url,
        'defanged_url': defanged,
        'hostname': hostname,
        'prediction': verdict,
        'risk_level': risk_level,
        'confidence': confidence,
        'phishing_probability': round(phishing_prob, 4),
        'features': features_dict,
        'homoglyph_analysis': homoglyph_res,
        'top_risk_factors': top_risk_factors,
        'recommendation': recommendation
    }


# -------------------------------------------------------------
# 3. ROUTES
# -------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    raw_url = request.form.get('url', '').strip()
    if not raw_url:
        return render_template('index.html', error="Please enter a valid URL to analyze.")

    try:
        result = analyze_tier1_url(raw_url)
        return render_template(
            'index.html',
            url=result['url'],
            defanged_url=result['defanged_url'],
            prediction=result['prediction'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            features=result['features'],
            homoglyph_analysis=result['homoglyph_analysis'],
            top_risk_factors=result['top_risk_factors'],
            recommendation=result['recommendation'],
            result=result
        )
    except InvalidURLError as e:
        return render_template('index.html', error=str(e), url=raw_url)
    except Exception as e:
        return render_template('index.html', error=f"Analysis Error: {str(e)}", url=raw_url)


@app.route('/api/v1/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(silent=True) or {}
    if not data and request.form.get('url'):
        data = {'url': request.form.get('url')}

    url = data.get('url', '').strip()
    if not url:
        return jsonify({'status': 'error', 'message': 'Missing required parameter: "url"'}), 400

    try:
        result = analyze_tier1_url(url)
        return jsonify({'status': 'success', 'data': result}), 200
    except InvalidURLError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Internal analysis failed: {str(e)}"}), 500


@app.route('/api/v1/deep-analyze', methods=['POST'])
def api_deep_analyze():
    data = request.get_json(silent=True) or {}
    if not data and request.form.get('url'):
        data = {'url': request.form.get('url')}

    url = data.get('url', '').strip()
    if not url:
        return jsonify({'status': 'error', 'message': 'Missing required parameter: "url"'}), 400

    try:
        shap_factors = []
        try:
            tier1_res = analyze_tier1_url(url)
            shap_factors = tier1_res.get('top_risk_factors', [])
        except Exception:
            pass

        forensic_report = perform_deep_analysis(url)
        ioc_report = generate_ioc(url, forensic_report, shap_risk_factors=shap_factors)

        return jsonify({
            'status': 'success',
            'target': ioc_report['target'],
            'forensics': ioc_report['forensics'],
            'ioc_report': ioc_report
        }), 200
    except InvalidURLError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except SSRFError as e:
        return jsonify({'status': 'error', 'message': f"SSRF Blocked: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Deep analysis failed: {str(e)}"}), 500


# -------------------------------------------------------------
# 4. RUN SERVER
# -------------------------------------------------------------
if __name__ == '__main__':
    print("--- Web Shield Server starting on http://127.0.0.1:5000 ---")
    app.run(debug=True, port=5000)