import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from src.feature_extraction import extract_features

# Initialize Flask app
app = Flask(__name__)

# -------------------------------------------------------------
# 1. LOAD MODEL ARTIFACTS
# -------------------------------------------------------------
model_path = os.path.join('models', 'xgb_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')
features_path = os.path.join('models', 'features.pkl')

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
    raise FileNotFoundError("❌ Model files missing in 'models/'! Please run train_model.py first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

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
        return render_template('index.html', error="Please enter a valid URL.")

    try:
        # Extract, align columns, and scale features
        raw_features = extract_features(url)
        df_features = pd.DataFrame([raw_features], columns=feature_names)
        scaled_features = scaler.transform(df_features)
        
        # Predict
        prediction_num = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        label = 'Phishing' if prediction_num == 1 else 'Legitimate'
        confidence = round(float(probabilities[prediction_num]) * 100, 2)
        
        return render_template('index.html', prediction=label, confidence=confidence, url=url)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}", url=url)

# -------------------------------------------------------------
# 3. RUN SERVER
# -------------------------------------------------------------
if __name__ == '__main__':
    print("🚀 Web Shield Server starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)