"""
Web Shield Phishing Detection & Deep Forensics API.
Unified Flask backend orchestrating Tier 1 triage, SHAP explainability, and Tier 2 deep forensics.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.url_normalizer import normalize_url, InvalidURLError
from src.safe_fetcher import SSRFError, FetchError
from src.feature_extraction import extract_features, get_domain_age_status
from src.homoglyph import analyze_homoglyphs
from src.explainability import get_top_risk_factors
from src.deep_analysis import perform_deep_analysis
from src.ioc import generate_ioc
from src.defang import defang_url
from src.predictor import get_predictor, ModelPredictor, ModelInputError
from src.hybrid_triage import route_for_tier2, compute_hybrid_verdict

app = Flask(__name__)

# -------------------------------------------------------------
# 1. INITIALIZE MODEL PREDICTOR (MODEL V2 DEFAULT)
# -------------------------------------------------------------
predictor = get_predictor()
model = predictor.model
scaler = predictor.scaler
feature_names = predictor.feature_names


# -------------------------------------------------------------
# 2. UNIFIED TIER 1 SERVICE FUNCTION
# -------------------------------------------------------------
def analyze_tier1_url(raw_url: str, force_model_version: str = None) -> dict:
    """
    Unified Tier 1 service function:
    1. Normalizes and validates raw URL.
    2. Extracts feature vector.
    3. Runs XGBoost model probability inference.
    4. Evaluates homoglyph / brand impersonation risk.
    5. Computes top SHAP risk factors.
    6. Determines Tier 2 hybrid triage routing requirements.
    7. Computes hybrid evidence-based risk verdict.
    """
    if not raw_url or not isinstance(raw_url, str) or not raw_url.strip():
        raise InvalidURLError("URL parameter cannot be empty.")

    norm_res = normalize_url(raw_url)
    canonical_url = norm_res["canonical_url"]
    hostname = norm_res["hostname"]
    defanged = defang_url(canonical_url)

    # Determine predictor instance
    active_predictor = get_predictor(force_version=force_model_version) if force_model_version else predictor

    # Extract features matching the model version
    if active_predictor.version == "v4j4":
        from src.feature_extraction_v4j import extract_features_v4j
        features_dict = extract_features_v4j(canonical_url)
    else:
        features_dict = extract_features(canonical_url)

    # Versioned model prediction and SHAP explainability
    phishing_prob, model_df, top_risk_factors = active_predictor.predict_raw(features_dict)

    # Homoglyph Analysis
    homoglyph_res = analyze_homoglyphs(hostname)

    domain_age = features_dict.get('domain_age_months', -1)
    domain_age_status = get_domain_age_status(domain_age)

    # Pure ML Model Classification (based solely on model decision threshold 0.50)
    MODEL_THRESHOLD = 0.50
    ml_is_phishing = bool(phishing_prob >= MODEL_THRESHOLD)
    ml_prediction = 'Phishing' if ml_is_phishing else 'Legitimate'

    # Compute Hybrid Triage Verdict & Routing
    hybrid_res = compute_hybrid_verdict(phishing_prob, ml_is_phishing, features_dict, tier2_report=None, url=canonical_url)
    routing_info = route_for_tier2(phishing_prob, features_dict, url=canonical_url)

    risk_level = hybrid_res["risk_verdict"].lower()
    risk_verdict = hybrid_res["risk_verdict"]
    overall_verdict = hybrid_res["overall_verdict"]

    # Prediction confidence (confidence in pure ML binary classification)
    if ml_is_phishing:
        prediction_confidence = round(phishing_prob * 100, 2)
    else:
        prediction_confidence = round((1.0 - phishing_prob) * 100, 2)

    if risk_verdict == 'CRITICAL':
        recommendation = "DANGER: High-risk phishing or homoglyph impersonation site detected! Do NOT interact or enter credentials."
    elif risk_verdict == 'SUSPICIOUS':
        recommendation = "WARNING: Suspicious indicators detected. Exercise caution before entering sensitive details."
    else:
        recommendation = "SAFE: No critical phishing or homoglyph risk indicators detected."

    return {
        'url': canonical_url,
        'defanged_url': defanged,
        'hostname': hostname,
        'prediction': overall_verdict,
        'overall_verdict': overall_verdict,
        'ml_prediction': ml_prediction,
        'ml_is_phishing': ml_is_phishing,
        'risk_level': risk_level,
        'risk_verdict': risk_verdict,
        'confidence': prediction_confidence,
        'prediction_confidence': prediction_confidence,
        'phishing_probability': round(phishing_prob, 4),
        'ml_probability': round(phishing_prob, 4),
        'features': features_dict,
        'domain_age_status': domain_age_status,
        'homoglyph_analysis': homoglyph_res,
        'top_risk_factors': top_risk_factors,
        'tier2_required': routing_info['tier2_required'],
        'routing_reason': routing_info['routing_reason'],
        'routing_reasons': routing_info['routing_reasons'],
        'scan_status': hybrid_res['scan_status'],
        'analysis_confidence': hybrid_res['analysis_confidence'],
        'forensic_evidence_severity': hybrid_res['forensic_evidence_severity'],
        'forensic_evidence': hybrid_res['forensic_evidence'],
        'recommendation': recommendation,
        'model_version': active_predictor.version,
        'model_feature_count': len(active_predictor.feature_names)
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
            overall_verdict=result['overall_verdict'],
            ml_prediction=result['ml_prediction'],
            ml_is_phishing=result['ml_is_phishing'],
            risk_level=result['risk_level'],
            risk_verdict=result['risk_verdict'],
            confidence=result['confidence'],
            prediction_confidence=result['prediction_confidence'],
            phishing_probability=result['phishing_probability'],
            ml_probability=result['ml_probability'],
            features=result['features'],
            domain_age_status=result['domain_age_status'],
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
        tier1_res = None
        shap_factors = []
        try:
            tier1_res = analyze_tier1_url(url)
            shap_factors = tier1_res.get('top_risk_factors', [])
        except Exception:
            pass

        forensic_report = perform_deep_analysis(url)
        ioc_report = generate_ioc(url, forensic_report, shap_risk_factors=shap_factors)

        if tier1_res:
            hybrid = compute_hybrid_verdict(
                tier1_res['ml_probability'],
                tier1_res['ml_is_phishing'],
                tier1_res['features'],
                tier2_report=forensic_report,
                url=url
            )
            tier1_summary = {
                'prediction': hybrid.get('overall_verdict'),
                'overall_verdict': hybrid.get('overall_verdict'),
                'ml_prediction': tier1_res.get('ml_prediction'),
                'ml_probability': tier1_res.get('ml_probability'),
                'risk_verdict': hybrid.get('risk_verdict'),
                'risk_level': hybrid.get('risk_verdict').lower(),
                'confidence': tier1_res.get('confidence'),
                'tier2_required': hybrid.get('tier2_required'),
                'routing_reason': hybrid.get('routing_reason'),
                'routing_reasons': hybrid.get('routing_reasons'),
                'scan_status': hybrid.get('scan_status'),
                'analysis_confidence': hybrid.get('analysis_confidence'),
                'forensic_evidence_severity': hybrid.get('forensic_evidence_severity'),
                'forensic_evidence': hybrid.get('forensic_evidence')
            }
            ioc_report['tier1_assessment'] = tier1_summary
            ioc_report['forensics']['tier1_assessment'] = tier1_summary
            ioc_report['hybrid_assessment'] = hybrid

        return jsonify({
            'status': 'success',
            'target': ioc_report['target'],
            'forensics': ioc_report['forensics'],
            'ioc_report': ioc_report
        }), 200
    except InvalidURLError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except (SSRFError, FetchError) as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Deep analysis failed: {str(e)}"}), 500


# -------------------------------------------------------------
# 4. RUN SERVER
# -------------------------------------------------------------
if __name__ == '__main__':
    print("--- Web Shield Server starting on http://127.0.0.1:5000 ---")
    app.run(debug=True, port=5000)