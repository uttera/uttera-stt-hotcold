# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-27

### Added
- Initial production release for high-performance speech-to-text.
- FastAPI-based architecture for low-latency concurrent requests.
- Advanced infrastructure monitoring with comprehensive DEBUG modes.
- Production-grade concurrency handling via Uvicorn.
- Dedicated support for local GPU/CPU offloading.

### Changed
- Sanitized system metadata to ensure local network privacy.
- Simplified installation requirements for Docker and manual deployments.

### Fixed
- Improved handling of multi-part audio stream uploads.
