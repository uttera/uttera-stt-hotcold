# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.8] - 2026-04-08

### Added
- **`WHISPER_FP16` env var (default `"1"`):** When CUDA is available and `WHISPER_FP16=1`, the hot-worker model is loaded on CPU in fp16 and then moved to GPU, with `LayerNorm` weights kept in fp32. This is required because Whisper's `LayerNorm` calls `x.float()` internally — if the weights are also fp16, PyTorch raises a dtype mismatch at runtime. The result is ~2912 MiB less VRAM reserved for `whisper-medium` (1466 MiB vs 4378 MiB, −66.5%), measured on RTX 5090 with Python 3.12.3 / torch 2.11.0+cu130. Set `WHISPER_FP16=0` to revert to the original fp32 loading path (e.g. for debugging or CPU-only deployments).

## [1.4.7] - 2026-04-07

### Fixed
- **`SyntaxError`: `global _cold_workers_in_flight` used before declaration:** In both `create_transcription` and `create_translation`, a debug `print` f-string inside Branch C read `_cold_workers_in_flight` before the `global` declaration that followed it, causing Python to raise `SyntaxError: name '_cold_workers_in_flight' is used prior to global declaration` on startup. Fixed by moving the `global _cold_workers_in_flight` statement to the very top of the `else:` block, before any read of the variable.
- **Cold lane silent failure on empty/inaudible audio:** When whisper exited with code 0 but produced no output JSON (audio with no detectable speech segments), the server raised an opaque `[Errno 2] No such file or directory`. Added an explicit check: if the output JSON path does not exist after a clean exit, full `stdout` and `stderr` are logged and a descriptive `RuntimeError("Cold Lane produced no output (exit 0, JSON missing)")` is raised instead.

### Validated
- 40-clip Spanish stress test (40/40 OK, sim avg=0.989, WAcc avg=0.986, zero HTTP errors). Four waves: 10 concurrent, 10 staggered @0.2s, 10 concurrent, 10 staggered @0.1s. Cold EMA calibrated at 14-15s, max 3 concurrent cold workers with `MIN_COLD_VRAM_GB=4.0` on 13.5 GB free VRAM. Results fully reproducible across two runs with warm EMA.

## [1.4.6] - 2026-04-06

### Added
- **VRAM pre-check before cold lane dispatch:** Branch C now calls `torch.cuda.mem_get_info()` before spawning a cold subprocess. If effective free VRAM (raw free minus `in_flight × MIN_COLD_VRAM_GB` reserved for already-dispatched workers) is below `MIN_COLD_VRAM_GB` (default `4.0` GB, configurable via `.env`), the request is rerouted to the hot lane queue immediately — avoiding the 8-10s wasted loading the model before OOM. The in-flight reservation prevents burst routing decisions from collectively over-committing memory before any subprocess has actually allocated. Free VRAM, `min_cold_vram_gb`, `cold_workers_in_flight`, and `vram_sufficient_for_cold` added to `GET /health` under `smart_routing`. Set `MIN_COLD_VRAM_GB=0` to disable. Applied to both `/v1/audio/transcriptions` and `/v1/audio/translations`.

## [1.4.5] - 2026-04-06

### Fixed
- **`model_lock` deadlock under client timeout:** Branch B and the Branch C fallback previously used two separate `asyncio.to_thread` calls — one to acquire `model_lock` and one to run transcription. If asyncio cancelled the coroutine (e.g. client timeout during burst load) between the two awaits, `model_lock` was left permanently acquired with no one to release it, deadlocking the server for all subsequent requests. Fixed by introducing `_run_hot_locked()`, which performs acquire + transcribe + release inside a single `asyncio.to_thread` call. Because the entire lock lifecycle is confined to one thread, cancellation of the calling coroutine cannot interrupt it. Applied to both `/v1/audio/transcriptions` and `/v1/audio/translations`. Confirmed by burst-of-40 test: previous version deadlocked after client timeouts; this version drains the queue correctly even after clients disconnect.

## [1.4.4] - 2026-04-06

### Added
- **Auto-Calibration of `COLD_START_TIME_SECONDS`:** The router now measures each successful cold lane completion and maintains an EMA (α = 0.2) of cold lane times in `_cold_ema_start_stt`. Once seeded, `_get_cold_start_time_stt()` returns the live EMA instead of the static `COLD_START_TIME_SECONDS`. `COLD_START_TIME_SECONDS` in `.env` becomes an initial hint / fallback used only before the first successful cold lane completes. `cold_start_calibrated`, `cold_ema_start_seconds`, and `cold_start_configured_seconds` added to `GET /health` under `smart_routing`. Applied to both `/v1/audio/transcriptions` and `/v1/audio/translations`.

## [1.4.3] - 2026-04-06

### Fixed
- **EMA not updated from fallback path:** The cold-lane fallback was calling `_update_hot_ema_stt` with an elapsed time that included the cold-lane failure duration (~`COLD_START_TIME_SECONDS`), inflating `ema_sps` and creating a positive feedback loop. The EMA is now updated only from clean Branch A and Branch B completions. Applied to both `/v1/audio/transcriptions` and `/v1/audio/translations` fallback paths.

## [1.4.2] - 2026-04-06

### Added
- **Startup EMA Warmup:** After the hot worker loads, a 2-second synthetic silence clip is transcribed automatically to seed `_hot_ema_sps` before the first real request arrives. Without this, `EMA=None` at startup caused every concurrent request to go to cold lane (Branch C), triggering CUDA OOM when multiple workers tried to load the model simultaneously. The warmup runs as a FastAPI `startup` event (async, non-blocking) and prints the measured `sps` on the console so the operator can verify hardware throughput at startup. Failure is logged but non-fatal: the server starts in uncalibrated mode rather than refusing to start.

## [1.4.1] - 2026-04-06

### Added
- **Cold-Lane Fallback to Hot Lane:** When a cold lane subprocess exits with a non-zero code (the primary cause being CUDA OOM when multiple cold workers attempt to load the model simultaneously), the request is transparently retried on the hot lane instead of returning HTTP 500. The fallback uses the same Branch-B queuing mechanism — `audio_dur` is added to `_hot_queue_audio_seconds` before waiting so late-arriving requests see the correct queue depth. By the time cold lane fails (~`COLD_START_TIME_SECONDS` in), the hot lane has typically drained significantly and the additional wait is short. Applied to both `/v1/audio/transcriptions` and `/v1/audio/translations`.

## [1.4.0] - 2026-04-06

### Added
- **Smart Hot-Lane Routing:** Three-branch router replaces the previous binary hot/cold decision, mirroring the architecture introduced in coqui-tts-local-server v1.5.0.
  - **Branch A** (hot lane free): use immediately — unchanged from prior behaviour.
  - **Branch B** (hot lane busy, worth waiting): if the estimated queue drain time is below `COLD_START_TIME_SECONDS × HOT_QUEUE_SAFETY_FACTOR`, the request waits for the hot lane via a non-blocking `asyncio.to_thread(model_lock.acquire)` instead of spawning a cold subprocess.
  - **Branch C** (hot lane busy, cold is faster): drain estimate exceeds threshold → spawn cold lane as before.
  - Unlike the TTS server (which uses word count), the STT drain estimate uses **audio duration in seconds** as the queue unit — the natural proxy for Whisper processing time. Duration is read from the WAV header via stdlib `wave`; non-WAV formats (MP3, M4A, etc.) fall back to a byte-size heuristic.
  - EMA (α = 0.2) tracks server-seconds per audio-second (`ema_sps`) and self-calibrates after each successful hot-lane transcription. Falls back to Branch C when not yet calibrated.
  - Applied to both `/v1/audio/transcriptions` and `/v1/audio/translations`.
  - New env vars: `COLD_START_TIME_SECONDS` (default `8.0` s), `HOT_QUEUE_SAFETY_FACTOR` (default `0.8`).
  - Routing stats and live telemetry (`ema_sps`, `hot_queue_audio_seconds`, `hot_queue_drain_estimate_seconds`, `threshold_seconds`) exposed in `GET /health` under `smart_routing`.

## [1.3.8] - 2026-04-04

### Added
- **Docker Support:** Official support for containerized deployment with NVIDIA GPU acceleration.
- **NVIDIA CDI Integration:** Optimized for modern Docker environments (Ubuntu 22.04+) using Container Device Interface for robust GPU passthrough.
- **Documentation:** Added detailed Docker setup instructions and host prerequisites to README.md.

## [1.3.7] - 2026-04-04

### Added
- **API Documentation (`API.md`):** Comprehensive OpenAI-compatible API documentation including transcription and translation endpoints.
- **Project Descriptor (`whisper-stt-local-server.yml`):** YAML metadata for project identification and integration with awesome-selfhosted datasets.

## [1.3.6] - 2026-04-03

### Added
- **`POST /v1/audio/translations` endpoint:** OpenAI-compatible translation endpoint. Accepts audio in any language and returns the transcription translated to English in a single Whisper pass (`task="translate"`). Accepts the same parameters as `/v1/audio/transcriptions` except `language` (output is always English). Fully routed through the Hot/Cold Lane architecture.

## [1.3.5] - 2026-04-03

### Fixed
- **Cold Lane fails with `No such file or directory` on non-sphinx installs:** `VENV_PYTHON` and `WHISPER_SCRIPT` now auto-detect `venv/bin/python` and `venv/bin/whisper` relative to `BASE_DIR` before falling back to the hardcoded sphinx paths. Resolution order: env var → local venv → sphinx fallback.

## [1.3.4] - 2026-04-03

### Fixed
- **`PermissionError` on startup when installing as non-root:** `MODEL_CACHE_DIR` now defaults to `assets/models/whisper` inside the project directory (project-relative, no-sudo). The old default of `/opt/ai/models/speech` required root. Mirrors the `BASE_DIR`/`ASSETS_DIR` pattern from coqui-tts-local-server. Can still be overridden via `XDG_CACHE_HOME`.

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
