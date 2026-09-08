import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.feature_extraction import extract_features

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


def analyze_url_internal(url: str):
    """Utility function to extract 10 features, scale, and predict on a URL string."""
    clean_url = url.strip()
    if not clean_url:
        raise ValueError("URL cannot be empty.")

    # Extract dictionary of 10 features
    features_dict = extract_features(clean_url)

    # Reorder columns to match trained feature_names exactly
    df_features = pd.DataFrame([features_dict])[feature_names]

    # Scale features
    scaled_features = scaler.transform(df_features)

    # Model inference
    pred_num = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]

    verdict = 'Phishing' if pred_num == 1 else 'Legitimate'
    confidence = round(float(probabilities[pred_num]) * 100, 2)

    return {
        'url': clean_url,
        'prediction': verdict,
        'confidence': confidence,
        'features': features_dict
    }


# -------------------------------------------------------------
# 2. ROUTES
# -------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url', '').strip()

    if not url:
        return render_template('index.html', error="Please enter a valid URL to analyze.")

    try:
        result = analyze_url_internal(url)
        return render_template(
            'index.html',
            url=result['url'],
            prediction=result['prediction'],
            confidence=result['confidence'],
            features=result['features']
        )
    except Exception as e:
        return render_template('index.html', error=f"Analysis Error: {str(e)}", url=url)


@app.route('/api/v1/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        if request.form.get('url'):
            data = {'url': request.form.get('url')}
        else:
            return jsonify({'error': 'Invalid request payload. Expected JSON object with "url" key.'}), 400

    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'Missing required parameter: "url"'}), 400

    try:
        result = analyze_url_internal(url)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f"Internal analysis failed: {str(e)}"}), 500


# -------------------------------------------------------------
# 3. RUN SERVER
# -------------------------------------------------------------
if __name__ == '__main__':
    print("--- Web Shield Server starting on http://127.0.0.1:5000 ---")
    app.run(debug=True, port=5000)