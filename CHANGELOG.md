# Changelog — Web Shield

All notable changes to the Web Shield project will be documented in this file.

## [v2.0.0] - 2026-09-09

### Added
- **Two-Stage Architecture**: Tier 1 rapid ML screening + Tier 2 forensic triage.
- **Model V2 Engine**: Retrained XGBoost Model V2 using 11 canonical unscaled features.
- **TreeSHAP Explainability**: SHAP attribution pipeline with neutral label formatting.
- **SSRF & DNS Security**: `PinnedIPAdapter` enforcing IP subnet validation, IP pinning, NAT64 support, and TLS SNI preservation.
- **Static DOM Credential Auditor**: DOM form analysis for password harvesting, external actions, and OAuth/payment context.
- **Defanged IOC Export**: Automatic URL defanging (`hxxps[://]...[.]...`) and structured JSON export.
- **3-State Risk Triage**: `SAFE`, `SUSPICIOUS`, `CRITICAL` risk classification.
- **API Endpoints**: REST endpoints `/predict`, `/api/v1/analyze`, and `/api/v1/deep-analyze`.
- **Test Suite**: 73 automated tests passing with zero failures.

### Changed
- Refactored `src/predictor.py` with unified `ModelPredictor` abstraction and `WEBSHIELD_MODEL_VERSION` rollback support.
- Updated Jinja2 web interface templates with Cyber-Command dark mode UI.

### Fixed
- Fixed failed-fetch HTTP status `0` bug; network failures return `HTTP status = None`, `scan_status = failed/partial`, `credential_analysis = NOT_EVALUATED`, and `sink_risk = UNKNOWN`.
- Fixed domain age WHOIS fallback semantics to explicitly mark unknown domain age as `UNKNOWN` rather than `NEW`.
