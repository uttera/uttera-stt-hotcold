# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.3] - 2026-04-03

### Changed
- **`VENV_PYTHON` and `WHISPER_SCRIPT` now configurable via env vars:** Both paths can be overridden in `.env` without modifying source code. Hardcoded values remain as fallback for the canonical sphinx installation.

## [1.3.2] - 2026-04-03

### Fixed
- **Cold Lane timeout:** `run_transcription_slow_lane` refactored from blocking `subprocess.run()` to `asyncio.create_subprocess_exec` + `asyncio.wait_for`. A new `COLD_LANE_TIMEOUT_SECONDS` env var (default: 300s) controls the limit. On timeout the subprocess is killed and the request fails with HTTP 500. Prevents hung subprocesses (OOM, driver crash) from blocking the server indefinitely.

## [1.3.1] - 2026-04-03

### Security
- **Error information leak fixed:** Exceptions in the transcription endpoint no longer expose internal paths, subprocess details, or model locations in HTTP 500 responses. The client receives a generic `"Transcription failed. Check server logs."` message. Full detail is still logged to stdout.

## [1.3.0] - 2026-04-03

### Added
- **`GET /health` endpoint:** Returns server version, model name, and hot worker load state. `hot_worker_error` field is set if the model failed to load at startup. Suitable for proxies and Docker healthchecks.
- **`GET /v1/models` endpoint:** OpenAI-compatible model listing returning `whisper-1`. Required by clients that query the model list before issuing STT requests.
- **`SERVER_VERSION` constant:** Version string centralized in a single constant. `/health` and FastAPI metadata read from it.
- **`hot_worker_error` global:** Captures the exception message if the Whisper model fails to load, and exposes it in `/health`.
- **`setup.sh`:** Unified installation script. Creates venv with Python 3.12 (falls back to python3), installs dependencies, and delegates to `setup_assets.sh`.
- **`setup_assets.sh`:** Provisions Whisper models into the local `assets/` directory. Idempotent — safe to re-run.
- **`.env.example`:** Documents all supported environment variables.
- **`requirements.txt` version pins:** Added minimum version constraints for all dependencies. Added `python-dotenv`.

## [1.2.3] - 2026-02-27

### Fixed
- Strict DEBUG control and shell command printing in slow lane.

## [1.2.2] - 2026-02-27

### Fixed
- Wrapped model loading prints into DEBUG toggle.

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
