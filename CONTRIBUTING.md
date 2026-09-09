# Contributing to Web Shield

Thank you for considering contributing to Web Shield! We welcome community contributions to enhance phishing detection, forensic analysis, and security controls.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/Chinnampavansai2008/WEB-SHIELD-PHISHING-DETECTION.git
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Run the automated test suite:
   ```bash
   python -m unittest discover tests
   ```

## Guidelines

- **Code Quality**: Follow PEP 8 guidelines for Python code.
- **Testing**: Ensure all new features or bug fixes include corresponding unit tests under `tests/`. All 73 tests must pass before submitting pull requests.
- **Security**: Do NOT weaken SSRF protection, IP pinning, or TLS validation.
- **Model Integrity**: Do NOT retrain or alter production Model V2 artifacts without complete evaluation documentation.

## Security Disclosures

If you discover a security vulnerability, please do NOT create a public issue. Report it responsibly to the project maintainers.
