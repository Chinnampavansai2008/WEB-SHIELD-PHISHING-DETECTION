# 🛡️ Web Shield - Phishing Detection System

**Web Shield** is a Machine Learning powered web application designed to detect phishing URLs in real-time. Using feature extraction techniques and an XGBoost classification model, Web Shield analyzes structural URL heuristics to classify websites as **Legitimate** or **Phishing** with high confidence.

---

## 📌 Project Overview

Phishing attacks remain one of the most prominent cyber threats today. **Web Shield** tackles this challenge by extracting statistical and heuristic features directly from web links—such as SSL validity, domain age estimates, URL length, IP address presence, subdomain count, and redirect counts—and evaluating them through a trained Machine Learning model.

### Key Features
- **Real-Time URL Analysis:** Simple web dashboard allowing users to input URLs for instant classification.
- **XGBoost Classification Engine:** High-performance Gradient Boosted Decision Tree model trained on extracted domain features.
- **Automated Feature Extraction:** Custom parser to inspect URLs for suspicious patterns.
- **Reproducible Data & Training Pipeline:** Automated scripts to generate datasets, train models, and output evaluation metrics alongside confusion matrices.

---

## 📁 Repository Structure

```text
WEB-SHIELD-PHISHING-DETECTION/
├── data/
│   └── dataset.csv              # Synthetic dataset containing URL features & labels
├── models/
│   ├── xgb_model.pkl            # Trained XGBoost Classifier model
│   ├── scaler.pkl               # StandardScaler object for feature scaling
│   ├── features.pkl             # Serialized list of model feature names
│   └── confusion_matrix.png     # Visual evaluation metrics chart
├── src/
│   ├── __init__.py
│   └── feature_extraction.py    # URL feature extraction module
├── templates/
│   └── index.html               # Flask HTML interface
├── app.py                       # Main Flask web application entrypoint
├── generate_data.py             # Script to generate training dataset
├── train_model.py               # Script to train, evaluate, and save ML model
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/WEB-SHIELD-PHISHING-DETECTION.git
cd WEB-SHIELD-PHISHING-DETECTION
```

### 2. Create & Activate Virtual Environment (Optional but Recommended)
- **On Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Step 1: Generate Dataset (Optional)
Generate 1,000 synthetic URL samples matching domain heuristics:
```bash
python generate_data.py
```

### Step 2: Train & Evaluate Model (Optional)
Train the XGBoost model, compute evaluation metrics, and export artifacts:
```bash
python train_model.py
```

### Step 3: Launch Web Interface
Start the Flask web server:
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.

---

## 📊 Model Evaluation & Features

The model evaluates the following 7 URL features:
1. **`url_length`**: Total character count of the URL.
2. **`having_ip`**: Binary indicator if an IP address is used as the domain name.
3. **`has_at_symbol`**: Binary indicator for the `@` symbol in the URL.
4. **`ssl_valid`**: HTTPS protocol presence check.
5. **`domain_age_months`**: Domain age heuristic based on SSL status.
6. **`subdomain_count`**: Number of subdomains detected.
7. **`redirect_count`**: Frequency of `//` redirection markers.

---

## 💻 Tech Stack

- **Backend Framework:** Flask
- **Machine Learning:** XGBoost, Scikit-Learn
- **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
