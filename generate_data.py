import os

import pandas as pd

import numpy as np



# 1. Create 'data' folder if it doesn't exist

os.makedirs('data', exist_ok=True)



# 2. Set seed for reproducible realistic data

np.random.seed(42)

n_samples = 1000



# 3. Generate realistic features for Legitimate (0) vs Phishing (1)

labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])



# Feature rules matching domain heuristics:

url_length = [np.random.randint(15, 45) if l == 0 else np.random.randint(40, 150) for l in labels]

having_ip = [np.random.choice([0, 1], p=[0.98, 0.02]) if l == 0 else np.random.choice([0, 1], p=[0.70, 0.30]) for l in labels]

has_at_symbol = [np.random.choice([0, 1], p=[0.99, 0.01]) if l == 0 else np.random.choice([0, 1], p=[0.80, 0.20]) for l in labels]

ssl_valid = [np.random.choice([0, 1], p=[0.05, 0.95]) if l == 0 else np.random.choice([0, 1], p=[0.85, 0.15]) for l in labels]

domain_age_months = [np.random.randint(24, 360) if l == 0 else np.random.randint(1, 12) for l in labels]

subdomain_count = [np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]) if l == 0 else np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]) for l in labels]

redirect_count = [np.random.choice([0, 1], p=[0.95, 0.05]) if l == 0 else np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]) for l in labels]



# 4. Build DataFrame

df = pd.DataFrame({

    'url_length': url_length,

    'having_ip': having_ip,

    'has_at_symbol': has_at_symbol,

    'ssl_valid': ssl_valid,

    'domain_age_months': domain_age_months,

    'subdomain_count': subdomain_count,

    'redirect_count': redirect_count,

    'label': labels

})



# 5. Save directly to CSV

filepath = os.path.join('data', 'dataset.csv')

df.to_csv(filepath, index=False)



print(f"✅ Dataset successfully generated and saved to: {filepath}")

print(df.head())