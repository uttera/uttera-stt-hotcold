#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Uttera STT Server (Hybrid Model)
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025-2026 Hugo L. Espuny
# Original work created with assistance from Google Gemini and Anthropic Claude
#
# Part of the Uttera voice stack (https://uttera.ai).
# See LICENSE and NOTICE for full terms and attributions.
#
# Package: uttera-stt-hotcold
# Version: 2.5.0
# Maintainer: Hugo L. Espuny
# Description: High-performance STT server with GPU acceleration and concurrency.
#
# CHANGELOG:
# - 2.5.0 (2026-09-24): Robustness sweep + standalone. Five additions, all
#   env-gated and safe by default:
#   1. Optional frozen-model mode: set UTTERA_OFFLINE=1 and the server sets
#      HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / MODELSCOPE_OFFLINE before the ML
#      imports, so a validated model never silently re-downloads or changes on
#      a reboot. Online by default (a fresh install can fetch its model).
#   2. Voice-activity gate (Silero, CPU): returns an empty transcription for
#      clips with no speech, stopping Whisper hallucinating on silence. On any
#      doubt it transcribes. Env: VAD_ENABLED, VAD_THRESHOLD, VAD_JIT,
#      VAD_STRIDE. Needs `silero-vad`; disables itself if absent.
#   3. Engine circuit breaker: N consecutive engine (5xx) failures make /health
#      report 503; a later success clears it. Env: ENGINE_FAIL_THRESHOLD.
#   4. Recovery self-probe (synthetic tone via httpx ASGITransport, no network)
#      auto-clears the breaker when the engine recovers. Env: ENGINE_PROBE_SECONDS.
#   5. Correct HTTP status codes: oversized upload / over-long text -> 413,
#      GPU OOM -> 503 (excluded from the breaker), malformed JSON -> 400.
#      Env: MAX_FILESIZE_MB (default 250).
#   Removed: the optional external self-registration hook and the per-node
#   load metric — this server is now standalone (one process, an OpenAI-
#   compatible API, and a /health). /health dropped its `routing` block and
#   gained metrics.consecutive_engine_failures; it now returns 503 when the
#   engine is not ready or the breaker is open. Dropped redis from
#   requirements; added silero-vad + numpy.
# - 2.4.1 (2026-04-21): Fix module-load-time NameError introduced in
#   2.4.0: the static gauge setters (_BUILD_INFO.labels(...).set(1),
#   _LIBRETRANSLATE_CONFIGURED_GAUGE, _COLD_WORKER_POOL_CAP_GAUGE) were
#   placed above the metric definitions and crashed at import. Moved
#   them to run after the metric-definition block. 2.4.0 never booted.
# - 2.4.0 (2026-04-21): Prometheus /metrics endpoint. Exposes the
#   shared uttera_stt_* HTTP + request-shape metrics (same labels as
#   uttera-stt-vllm v1.4.0), plus this server's hot/cold-specific
#   telemetry: requests_by_route_total{route}, cold_workers_active,
#   cold_workers_loading, cold_workers_spawned_total,
#   cold_worker_ema_start_seconds, work_queue_depth,
#   work_queue_audio_seconds, vram_free_gb,
#   vram_per_cold_worker_gb. Inference duration histogram gets
#   lane-tagged ops: whisper_transcribe_hot vs
#   whisper_transcribe_cold. Additive — all existing endpoints
#   unchanged. Scrape with Telegraf's inputs.prometheus or any
#   OpenMetrics consumer.
# - 2.3.0 (2026-04-18): Default port migrated from 5000 → 9005. The
#   OpenAI spec does not mandate a port for self-hosted compatible
#   servers; we formalise 9005 as the canonical Uttera-stack port for
#   ALL Speech-to-Text backends (hotcold + vllm variants), matching
#   its TTS counterpart on 9004. Rationale: port 5000 has known
#   collisions with macOS AirPlay Receiver (since Monterey) and with
#   Docker Registry v2 (5000 is its default). Range 9000–9099 is IANA
#   "User Ports" without canonical assignment and is collision-free
#   on mainstream systems. All repo artefacts updated: `--port`
#   defaults in main_stt.py + README, Dockerfile EXPOSE, docker-
#   compose port mapping, CI workflow probes, .env.example PORT and
#   NODE_PORT defaults, API.md base URL. Migration for existing
#   deployments: set `PORT=5000` (or the previous value) in your env
#   if you need to preserve the old endpoint, otherwise point your
#   reverse proxy / load balancer at `:9005`. See also uttera-stt-vllm
#   v1.3.0 (sibling sharing the same port).
# - 2.2.1 (2026-04-18): /v1/models `owned_by` field now reports "uttera"
#   instead of a stale pre-rebrand value.
#   Cosmetic only — the field is free-form in the OpenAI spec, so no
#   client compatibility impact.
# - 2.2.0 (2026-04-17): OpenAI-compat polish sweep. Seven endpoint/feature
#   fixes identified in the full-endpoint validation sweep (256-request
#   burst + 19 single-shot tests):
#   1. response_format=srt|vtt|verbose_json now return correctly shaped
#      bodies (SRT / WebVTT timed subtitles and full whisper result) via
#      _render_response() + _segments_to_srt/_segments_to_vtt helpers.
#   2. to_language != "en" with LIBRETRANSLATE_URL unset now returns 400
#      with an explicit message instead of silently falling back to
#      English (contract-surprise).
#   3. Whisper "Unsupported language: XX" now surfaces as HTTP 400 with
#      the actual message (was previously swallowed into a generic 500
#      "Transcription failed. Check server logs.").
#   4. Non-audio / undecodeable file bodies now return HTTP 400 with a
#      typed decode error instead of 500.
#   5. temperature is validated against [0.0, 1.0] (OpenAI spec) and
#      out-of-range values rejected with HTTP 422.
#   6. HEAD /health is now accepted (uptime probes used by load
#      balancers no longer see spurious 405s).
#   7. Opt-in CORSMiddleware registered when CORS_ALLOW_ORIGINS env var
#      is set (comma-separated list, or "*" for all). Default is off,
#      preserving the API-first posture.
#   Also added: X-Translation-Mode: libretranslate response header on
#   the LibreTranslate-mediated translation path, for observability.
# - 2.1.0 (2026-04-17): /v1/audio/translations now works via a
#   Whisper-transcribe → LibreTranslate post-processing pipeline when
#   the new LIBRETRANSLATE_URL env var is set. Supports arbitrary
#   target languages via the `to_language` form field (default "en"
#   for OpenAI-compatibility), not only English. When source ==
#   target, LibreTranslate is skipped. When LibreTranslate fails,
#   the endpoint returns HTTP 502 (no silent fallback to untranslated
#   text). If LIBRETRANSLATE_URL is unset, falls back to the legacy
#   Whisper-native translate task for backward compatibility. New
#   env vars: LIBRETRANSLATE_URL, LIBRETRANSLATE_API_KEY,
#   LIBRETRANSLATE_TIMEOUT_S. New request form field: `to_language`
#   (optional) and explicit `language` (source hint). Requirements
#   gained httpx>=0.27.0.
# - 2.0.0 (2026-04-16): First Uttera-branded release. BREAKING:
#   * Rebranded from "Whisper STT Server" to Uttera. Repository moved to
#     github.com/uttera/uttera-stt-hotcold. README, banner, docs, systemd
#     unit names, FastAPI app title ("Uttera STT Server"), Dockerfile
#     and compose all updated. Personal donation addresses and pre-rebrand
#     artifacts removed.
#   * License changed to Apache-2.0. OpenAI Whisper weights remain under
#     their upstream MIT license.
#   Changed:
#   * SERVER_VERSION re-synced to 2.0.0 — the constant had been stuck at
#     "1.6.3" since v1.6.3, reporting a stale version in /health,
#     /v1/models, and the FastAPI metadata despite inline changelog
#     moves through v1.6.4/v1.6.5/v1.6.6/v1.6.7.
#   Added:
#   * setup_assets.sh now downloads every Whisper model by default
#     (tiny/tiny.en/base/base.en/small/small.en/medium/medium.en/large/
#     large-v2/large-v3/turbo), so swapping via WHISPER_MODEL env var
#     works fully offline. Idempotent — skips already-cached models.
#   * OSS community files: HISTORY.md, AUTHORS.md, CODE_OF_CONDUCT.md,
#     SECURITY.md, CODEOWNERS, CONTRIBUTING updates, etymology/backronym
#     section in the README.
# - 1.6.7 / 1.6.6 (2026-04-10): (superseded) added an optional external
#   self-registration hook and a per-node load metric to /health. Both were
#   removed in 2.5.0 — this server is standalone, see that entry.
# - 1.6.5 (2026-04-10): Align /health schema with coqui-tts-local-server.
#   Renamed work_queue_depth→queue_depth, work_queue_audio_seconds→queue_audio_seconds,
#   work_queue_drain_estimate_seconds→queue_drain_estimate_seconds,
#   cold_workers_in_flight→pool_workers_loading.
# - 1.6.4 (2026-04-10): Fix retried items bouncing between cold pool workers.
#   Cold workers now skip items with retried=True (put back immediately) so only
#   the hot worker processes them, avoiding unnecessary cold re-attempts.
# - 1.6.3 (2026-04-09): Staggered idle timeouts for cold pool wind-down. Workers
#   spawned first receive the longest idle timeout (COLD_WORKER_IDLE_TIMEOUT +
#   COLD_POOL_SIZE * COLD_WORKER_IDLE_STAGGER), workers spawned last receive the
#   base timeout. Prevents the "cliff death" where all workers die simultaneously
#   60 s after burst end; instead they wind down one-by-one over a spread of
#   COLD_POOL_SIZE * COLD_WORKER_IDLE_STAGGER seconds (default: 10 workers × 10 s
#   = 100 s spread).
# - 1.6.2 (2026-04-08): Fix VRAM cap in _optimal_cold_workers. The old cap
#   used int(free_gb/vram_per) as an absolute total, ignoring VRAM already
#   consumed by running workers — with N active workers it returned N as optimal
#   and the manager never spawned more. Removed: VRAM gating is handled solely
#   by _has_vram_for_cold_lane() in the pool manager (correct single source).
#   Result: 400-request burst now spawns 8 workers (0.64 GB free) vs 4 (9.62 GB
#   wasted). Note: GPU serializes inference across workers so throughput is bounded
#   by GPU compute regardless of worker count; benefit is latency distribution.
# - 1.6.1 (2026-04-08): Cold pool worker crash fallback to hot lane. On any pool
#   worker failure (OOM, kill, crash), the WorkItem is re-queued so the hot worker
#   rescues it instead of returning HTTP 500. X-Route reports "COLD-POOL>HOT".
#   Validated with COLD_CRASH_TEST=1: 40/40 OK, 18 requests via fallback path.
# - 1.6.0 (2026-04-08): Shared work queue + dynamic pool sizing. All requests
#   (hot and cold) are dispatched through a single asyncio.Queue. The hot worker
#   and all pool workers consume from this queue, so cold workers spawned mid-burst
#   can serve requests that were queued before they finished loading — eliminating
#   the fundamental limitation of v1.5.x where pool workers could only help new
#   requests arriving after load completed. Pool size is now computed dynamically
#   each tick using the formula N*(N-1) < 2*queue_work_s/cold_ema, which finds the
#   largest N where the last cold worker finishes loading before the burst is done.
#   COLD_POOL_SIZE becomes a safety cap (default 10, effectively uncapped for normal
#   hardware). Branches A/B/C/D replaced by a single enqueue path. X-Route header
#   reports "HOT" or "COLD-POOL" (set by whichever worker served the request).
#   New globals: _work_queue, _work_queue_audio_seconds, _pool_worker_tasks.
#   New functions: _hot_worker_loop, _pool_worker_loop, _optimal_cold_workers.
#   Removed: _cold_idle_pool, _acquire_idle_worker, _return_to_pool,
#   _should_queue_hot_stt, Branch A/B/C/D router logic.
# - 1.5.3 (2026-04-08): Cold pool manager. Background asyncio task (_cold_pool_manager)
#   monitors hot lane queue drain every 0.5 s and spawns cold workers proactively when
#   drain > threshold, pool not full, VRAM sufficient, and no spawn in progress. The router
#   no longer spawns workers (Branch C removed): it only consumes from the pool (Branch 0)
#   or queues hot (HOT-B / HOT-C). Spawning is serial (one at a time via _cold_spawn_lock)
#   to avoid CUDA contention. Pool manager also checks after each spawn whether drain still
#   warrants a second worker (fills pool up to COLD_POOL_SIZE sequentially).
# - 1.5.2 (2026-04-08): Serial cold spawning via _cold_spawn_lock (asyncio.Lock). Branch C
#   now checks _cold_spawn_lock.locked() before spawning: if another worker is already loading,
#   the request goes to HOT-C instead of spawning a concurrent loader. Prevents CUDA contention
#   between simultaneous cold workers whose combined load time far exceeds the calibrated
#   cold_ema (measured with a single uncontested worker). Once loaded, the worker enters the
#   pool and serves subsequent requests at inference-only cost via Branch 0 (COLD-POOL).
# - 1.5.1 (2026-04-08): Cold EMA startup warmup + unified inference EMA. At startup, a cold
#   worker is spawned, a silence clip is transcribed through it to measure total cold start
#   time (load + inference), cold_ema is seeded, VRAM drop measured, then the worker is killed.
#   This eliminates the uncalibrated period where cold_ema=COLD_START_TIME_SECONDS (8s) caused
#   premature cold dispatches before any real cold lane request completed. _cold_inference_ema_stt
#   removed: once a cold worker is loaded, inference time equals hot lane inference time (same
#   model, same GPU), so _hot_ema_sps is the correct estimator for pool worker throughput.
# - 1.5.0 (2026-04-08): Cold worker pool. Persistent subprocesses (cold_worker.py) load the
#   Whisper model once and serve multiple requests via newline-delimited JSON on stdin/stdout,
#   eliminating the ~18-20 s model-load cost on every cold lane request.
# - 1.4.12 (2026-04-08): Auto-calibration of MIN_COLD_VRAM_GB.
# - 1.4.11 (2026-04-08): Fix EMA inflation from queue wait time.
# - 1.4.10 (2026-04-08): Fallback hot lane retry on transient CUDA errors.
# - 1.4.9 (2026-04-08): Cold lane subprocess now passes --fp16 True/False matching WHISPER_FP16.
# - 1.4.8 (2026-04-08): WHISPER_FP16 env var (default "1"). fp16+LN-fp32 on GPU.
# - 1.4.7 (2026-04-07): Fixed SyntaxError in cold lane JSON-missing diagnostic.
# - 1.4.6 (2026-04-06): VRAM pre-check before cold lane dispatch.
# - 1.4.5 (2026-04-06): Fixed model_lock deadlock under client timeout (burst load).
# - 1.4.4 (2026-04-06): Auto-calibration of COLD_START_TIME_SECONDS via EMA.
# - 1.4.3 (2026-04-06): EMA no longer updated from fallback path.
# - 1.4.2 (2026-04-06): Startup EMA Warmup.
# - 1.4.1 (2026-04-06): Cold-Lane Fallback to Hot Lane.
# - 1.4.0 (2026-04-06): Smart Hot-Lane Routing (Branch A/B/C).
# - 1.3.x (2026-04-03): Translations, auto-detect venv, model cache, async cold lane.
# - 1.2.x (2026-02-27): DEBUG control.

import io
import os
import sys
import subprocess

# --- Optional frozen-model (offline) mode ------------------------------------
# ONLINE by default so a fresh install can fetch its model. Set UTTERA_OFFLINE=1
# to pin the engine to the local cache, so a model you have already validated
# can't silently re-download or change on a reboot. Applied BEFORE the ML
# libraries import (they read these variables at import time). You can also set
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / MODELSCOPE_OFFLINE directly.
if os.environ.get("UTTERA_OFFLINE", "0").lower() in ("1", "true", "yes"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")

import wave
import time
import base64
import torch
import uvicorn
import whisper
import tempfile
import asyncio
import threading
import json
import dataclasses
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Set

# Load .env from the project directory or its parent
_base = os.path.dirname(os.path.abspath(__file__))
for _env_path in [os.path.join(_base, ".env"), os.path.join(os.path.dirname(_base), ".env")]:
    if os.path.exists(_env_path):
        from dotenv import load_dotenv
        load_dotenv(_env_path)
        break

# -------------------------------
# 1. Global Config & Logging
# -------------------------------

SERVER_VERSION = "2.5.0"

# Valid response formats per OpenAI spec
SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}

# Valid temperature range per OpenAI spec [0.0, 1.0]
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

def _find_in_venv(rel_path: str) -> str:
    candidate = os.path.join(BASE_DIR, rel_path)
    return candidate if os.path.exists(candidate) else None

VENV_PYTHON = (
    os.environ.get("VENV_PYTHON")
    or _find_in_venv("venv/bin/python")
    or "/usr/local/lib/whisper/bin/python"
)
WHISPER_SCRIPT = (
    os.environ.get("WHISPER_SCRIPT")
    or _find_in_venv("venv/bin/whisper")
    or "/usr/local/lib/whisper/bin/whisper"
)

COLD_WORKER_SCRIPT = os.path.join(BASE_DIR, "cold_worker.py")

# Safety cap on pool workers. Default 10 is effectively uncapped for typical hardware.
# Set to 0 to disable the pool entirely (spawn-and-die behaviour).
# The actual number of active workers is computed dynamically by _optimal_cold_workers().
COLD_POOL_SIZE = int(os.environ.get("COLD_POOL_SIZE", "10"))

# Seconds of inactivity before an idle pool worker exits on its own.
COLD_WORKER_IDLE_TIMEOUT = int(os.environ.get("COLD_WORKER_IDLE_TIMEOUT", "60"))

# Additional idle seconds granted per spawn-order position (0-indexed from last).
# Worker spawned first gets COLD_POOL_SIZE * COLD_WORKER_IDLE_STAGGER extra seconds;
# worker spawned last gets 0 extra. Staggers pool wind-down so workers die one-by-one
# instead of all at once 60 s after the burst ends.
COLD_WORKER_IDLE_STAGGER = int(os.environ.get("COLD_WORKER_IDLE_STAGGER", "10"))

# How often (seconds) the pool manager checks whether to spawn a new cold worker.
COLD_POOL_MANAGER_INTERVAL = float(os.environ.get("COLD_POOL_MANAGER_INTERVAL", "0.5"))

MODEL_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.join(ASSETS_DIR, "models")),
    "whisper"
)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

WHISPER_FP16 = os.environ.get("WHISPER_FP16", "1").lower() in ("1", "true", "yes")

COLD_LANE_TIMEOUT_SECONDS = int(os.environ.get("COLD_LANE_TIMEOUT_SECONDS", "300"))

MIN_COLD_VRAM_GB = float(os.environ.get("MIN_COLD_VRAM_GB", 4.0))

# LibreTranslate post-processing for /v1/audio/translations.
# When this URL is set, /v1/audio/translations first transcribes with Whisper
# (in whatever source language is detected) and then passes the text through
# LibreTranslate to reach the requested `to_language`. Enables targets other
# than English (which Whisper's native translate is limited to) and works
# even with models that handle the native `translate` task poorly.
# If LIBRETRANSLATE_URL is empty, /v1/audio/translations falls back to the
# legacy Whisper-native translate path.
LIBRETRANSLATE_URL = os.environ.get("LIBRETRANSLATE_URL", "").rstrip("/")
LIBRETRANSLATE_API_KEY = os.environ.get("LIBRETRANSLATE_API_KEY", "")
LIBRETRANSLATE_TIMEOUT_S = float(os.environ.get("LIBRETRANSLATE_TIMEOUT_S", "30"))

# Maximum upload size. A body larger than this is rejected with HTTP 413
# (Payload Too Large) before it ever reaches the model. 413 is a client error
# and does NOT count towards the engine circuit breaker.
MAX_FILESIZE_MB = int(os.environ.get("MAX_FILESIZE_MB", "250"))

# Engine circuit breaker: N consecutive engine (5xx) failures mark the server
# not-ready so /health returns 503 instead of handing work to a dead engine. A
# single later success clears it. Only 5xx count — 4xx are the caller's fault.
ENGINE_FAIL_THRESHOLD = int(os.environ.get("ENGINE_FAIL_THRESHOLD", "3"))

# Recovery self-probe: while the breaker is open, every ENGINE_PROBE_SECONDS the
# server sends a tiny in-process request to itself; a 200 clears the breaker
# automatically, without a manual restart.
ENGINE_PROBE_SECONDS = int(os.environ.get("ENGINE_PROBE_SECONDS", "30"))

# Voice-activity gate (VAD). Whisper hallucinates text on silence (it learned
# "Thank you." / "Gracias." after quiet stretches from its subtitle training
# data). The gate returns an empty transcription when the whole clip has no
# speech; it never trims audio, and on any doubt it transcribes.
VAD_ENABLED = os.environ.get("VAD_ENABLED", "1").lower() not in ("0", "false", "no")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
# Path to the Silero JIT model. Empty = look inside the installed package.
VAD_JIT = os.environ.get("VAD_JIT", "")
# Window stride. Silero's window is 512 samples = 32 ms; stride 4 inspects
# 32 ms out of every 128, which does not miss real speech.
VAD_STRIDE = max(1, int(os.environ.get("VAD_STRIDE", "4")))

COLD_START_TIME_SECONDS = float(os.environ.get("COLD_START_TIME_SECONDS", 8.0))

HOT_QUEUE_SAFETY_FACTOR = float(os.environ.get("HOT_QUEUE_SAFETY_FACTOR", 0.8))

_HOT_EMA_ALPHA = 0.2

DEBUG_MODE = os.environ.get("DEBUG", "").lower() == "true"

def log_debug(message: str):
    if DEBUG_MODE:
        print(message)

model_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _work_queue, _cold_spawn_lock, _pool_worker_tasks, _engine_probe_task
    _work_queue = asyncio.Queue()
    _cold_spawn_lock = asyncio.Lock()
    _pool_worker_tasks = set()

    await _warmup_ema()
    await _warmup_cold_ema()

    hot_task = asyncio.create_task(_hot_worker_loop())
    manager_task = asyncio.create_task(_cold_pool_manager())
    # Recovery self-probe: clears the circuit breaker without a manual restart.
    _engine_probe_task = asyncio.create_task(_engine_probe_loop())

    yield

    # Shutdown: cancel all worker tasks
    hot_task.cancel()
    manager_task.cancel()
    if _engine_probe_task:
        _engine_probe_task.cancel()
    for task in list(_pool_worker_tasks):
        task.cancel()
    await asyncio.gather(hot_task, manager_task, _engine_probe_task,
                         *list(_pool_worker_tasks), return_exceptions=True)

    # Drain any remaining queued items (cancel their futures)
    while not _work_queue.empty():
        try:
            item = _work_queue.get_nowait()
            if not item.future.done():
                item.future.cancel()
        except asyncio.QueueEmpty:
            break


app = FastAPI(title="Uttera STT Server", version=SERVER_VERSION, lifespan=_lifespan)


# ── The right status code, not a blanket 500 ────────────────────────────────
# A 500 says "I broke". When the request is at fault, say so with a 4xx — and
# it's not cosmetic: 5xx failures count towards the engine circuit breaker, so
# a caller sending oversized text could otherwise trip it for everyone.
#   · text over the model's context  -> 413 (was 500)
#   · GPU OOM on a busy server        -> 503 (busy, not broken; excluded from
#                                             the breaker, with Retry-After)
import re as _re_err
from fastapi.responses import JSONResponse as _ErrResp

_SIGNS_TOO_LONG = ("max_model_len", "prompt_len", "context length", "too long",
                   "maximum context", "exceeds")
_SIGNS_OOM = ("out of memory", "outofmemoryerror", "cuda error: out of memory",
              "cublas_status_alloc_failed")


def _useful_line(exc) -> str:
    """Pull the line that explains the limit and DROP the traceback (which would
    leak internal paths and package names)."""
    for line in reversed(str(exc).splitlines()):
        low = line.lower()
        if any(s in low for s in _SIGNS_TOO_LONG) and 'file "' not in low:
            return _re_err.sub(r"^[A-Za-z_]+Error:\s*", "", line.strip())[:300]
    return "the request exceeds the model's limit"


def _classify_error(exc):
    """Return (status, detail) by inspecting the error text, or (None, None)."""
    t = str(exc).lower()
    if any(s in t for s in _SIGNS_OOM):
        return 503, "the server has no free memory right now; retry shortly"
    if any(s in t for s in _SIGNS_TOO_LONG):
        return 413, _useful_line(exc)
    return None, None


def _install_error_handlers(app):
    @app.exception_handler(json.JSONDecodeError)
    async def _on_json_error(request, exc):
        return _ErrResp(status_code=400,
                        content={"detail": "malformed JSON body: %s" % exc})

    async def _on_generic_error(request, exc):
        status, detail = _classify_error(exc)
        if status == 503:
            return _ErrResp(status_code=503, headers={"Retry-After": "30"},
                            content={"detail": detail})
        if status:
            return _ErrResp(status_code=status, content={"detail": detail})
        return _ErrResp(status_code=500,
                        content={"detail": "%s: %s" % (type(exc).__name__, str(exc)[:300])})

    app.add_exception_handler(ValueError, _on_generic_error)
    app.add_exception_handler(RuntimeError, _on_generic_error)   # includes OutOfMemoryError


_install_error_handlers(app)


# ── Engine circuit breaker + recovery probe ─────────────────────────────────

def _engine_is_ready() -> bool:
    """The server is ready to serve when the hot worker is loaded, it has no
    error, and the circuit breaker is closed."""
    return whisper_model is not None and hot_worker_error is None and _breaker_error is None


def _engine_failure(http_status: int, exc: Optional[Exception] = None) -> None:
    """Count an engine failure. Open the breaker on reaching the threshold.

    Only 5xx count; once N consecutive engine failures are seen /health serves
    503 until a later success (or the recovery probe) clears it.
    """
    global _consecutive_engine_failures, _breaker_error
    if http_status < 500:
        return                      # caller's fault — the engine is fine
    _consecutive_engine_failures += 1
    if _consecutive_engine_failures >= ENGINE_FAIL_THRESHOLD and _breaker_error is None:
        _breaker_error = ("circuit_breaker: %d consecutive engine failures; last: %s"
                          % (_consecutive_engine_failures,
                             f"{type(exc).__name__}: {exc}" if exc else "unknown"))[:300]
        print("CIRCUIT BREAKER OPEN: %s — /health now reports unavailable" % _breaker_error,
              file=sys.stderr, flush=True)


def _engine_ok() -> None:
    """A success clears the breaker and resets the count."""
    global _consecutive_engine_failures, _breaker_error
    _consecutive_engine_failures = 0
    if _breaker_error is not None:
        _breaker_error = None
        print("CIRCUIT BREAKER CLOSED: the engine responds again",
              file=sys.stderr, flush=True)


def _probe_wav(seconds: float = 1.0, hz: int = 440, sr: int = 16000) -> bytes:
    """A mono 16 kHz WAV holding a tone. A tone, not silence: pure silence can
    make speech models fail, and a failing probe would keep the breaker open."""
    import math
    import struct
    n = int(seconds * sr)
    samples = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * i / sr)))
                       for i in range(n))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples)
    return buf.getvalue()


async def _engine_probe_loop() -> None:
    """While the engine is not ready, send a tiny in-process request every
    ENGINE_PROBE_SECONDS. A 200 makes the endpoint call _engine_ok(), clearing
    the breaker — no manual restart. In-process via httpx ASGITransport (no
    network or open port)."""
    import httpx
    while True:
        try:
            await asyncio.sleep(ENGINE_PROBE_SECONDS)
            if _engine_is_ready():
                continue
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://probe",
                                         timeout=120.0) as cli:
                r = await cli.post("/v1/audio/transcriptions",
                                   files={"file": ("probe.wav", _probe_wav(), "audio/wav")},
                                   data={"model": "whisper-1"})
            if r.status_code == 200:
                print("probe: the engine responds again", file=sys.stderr, flush=True)
            else:
                print("probe: the engine is still down (HTTP %s)" % r.status_code,
                      file=sys.stderr, flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print("probe: failed to probe: %s" % e, file=sys.stderr, flush=True)


# ── Voice-activity gate (VAD) ───────────────────────────────────────────────

def _vad_loaded():
    """The Silero JIT model, loaded by hand without importing `silero_vad`
    (which would drag in onnxruntime). The model carries state and is not
    thread-safe on one instance, hence the lock around every use."""
    global _vad_model, _vad_broken
    if _vad_model is not None or _vad_broken:
        return _vad_model
    with _vad_lock:
        if _vad_model is None and not _vad_broken:
            try:
                path = VAD_JIT
                if not path:
                    import importlib.util as _u
                    path = os.path.join(
                        os.path.dirname(_u.find_spec("silero_vad").origin),
                        "data", "silero_vad.jit")
                _vad_model = torch.jit.load(path)
                _vad_model.eval()
                print("VAD: model loaded from %s (threshold %.2f)" % (path, VAD_THRESHOLD),
                      file=sys.stderr, flush=True)
            except Exception as e:
                _vad_broken = True
                print("VAD: unavailable (%s); transcribing everything, as before" % e,
                      file=sys.stderr, flush=True)
    return _vad_model


def _to_16k(audio_bytes: bytes):
    """The audio as float32 mono at 16 kHz via ffmpeg."""
    import numpy as np
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", "pipe:0",
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "pipe:1"]
    p = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).flatten().astype("float32") / 32768.0


def _no_voice(audio_bytes: bytes) -> bool:
    """True ONLY if the detector asserts there is no speech anywhere. On any
    doubt returns False and the audio is transcribed."""
    if not VAD_ENABLED or _vad_loaded() is None:
        return False
    try:
        pcm = _to_16k(audio_bytes)
        if pcm.size < 512:
            return False
        m = _vad_model
        with _vad_lock:            # a stateful instance: one thread at a time
            m.reset_states()
            with torch.no_grad():
                for i in range(0, len(pcm) - 512, 512 * VAD_STRIDE):
                    chunk = torch.from_numpy(pcm[i:i + 512]).unsqueeze(0)
                    if float(m(chunk, 16000).item()) >= VAD_THRESHOLD:
                        return False          # speech found
        return True
    except Exception as e:
        print("VAD: could not decide (%s); transcribing" % e,
              file=sys.stderr, flush=True)
        return False


# CORS middleware (disabled by default — API-first deployments don't need it).
# Set CORS_ALLOW_ORIGINS to a comma-separated list of origins, or "*" to allow all.
_cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
if _cors_origins_env:
    _cors_origins = ["*"] if _cors_origins_env == "*" else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Route", "X-Translation-Mode"],
    )


# Prometheus middleware — tracks every HTTP request generically.
# Endpoint-specific and lane-specific labels go in the endpoint
# handlers.

class _PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        method = request.method
        if path == "/metrics":
            return await call_next(request)
        endpoint = path if path in _KNOWN_ENDPOINTS else "other"
        t0 = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.monotonic() - t0
            _HTTP_REQUESTS_TOTAL.labels(
                endpoint=endpoint, method=method, status=str(status)
            ).inc()
            _HTTP_REQUEST_DURATION.labels(
                endpoint=endpoint, method=method
            ).observe(elapsed)

app.add_middleware(_PrometheusMiddleware)


# -------------------------------
# 2. Model Loading
# -------------------------------
model_name = os.environ.get("WHISPER_MODEL", "medium")
hot_worker_error: Optional[str] = None

log_debug(f"Loading HOT WORKER model '{model_name}' into memory...")
try:
    _cuda = torch.cuda.is_available()
    _use_fp16 = _cuda and WHISPER_FP16
    whisper_model = whisper.load_model(model_name, device="cpu" if _use_fp16 else None, download_root=MODEL_CACHE_DIR)
    if _use_fp16:
        whisper_model = whisper_model.half()
        for _m in whisper_model.modules():
            if isinstance(_m, torch.nn.LayerNorm):
                _m.float()
        whisper_model = whisper_model.cuda()
    log_debug(f"Model '{model_name}' loaded successfully ({'fp16+LN-fp32 GPU' if _use_fp16 else 'fp32'}).")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model: {e}")
    whisper_model = None
    hot_worker_error = str(e)

class TranscriptionResponse(BaseModel):
    text: str

# -------------------------------
# 2b. Smart Routing Telemetry
# -------------------------------

_hot_ema_sps: Optional[float] = None

# Total audio seconds currently in the shared work queue (pending + being processed).
# Updated on enqueue, decremented on completion so the pool manager sees the live load.
_work_queue_audio_seconds: float = 0.0

_cold_ema_start_stt: Optional[float] = None
_COLD_EMA_ALPHA_STT = 0.2

_cold_workers_in_flight: int = 0

_cold_vram_ema_gb: Optional[float] = None
_COLD_VRAM_EMA_ALPHA = 0.3
_COLD_VRAM_SAFETY_FACTOR = 1.2

# Shared work queue: all requests enqueued here, consumed by hot worker loop and pool workers.
_work_queue: Optional[asyncio.Queue] = None

# Lock ensuring at most one cold worker loads at a time (prevents CUDA contention).
_cold_spawn_lock: Optional[asyncio.Lock] = None

# Set of asyncio Tasks running _pool_worker_loop (one per active pool worker).
_pool_worker_tasks: Set[asyncio.Task] = set()

# Engine circuit breaker state. `_breaker_error` is set once the breaker opens
# and forces /health to report unavailable until a success (or the probe) clears
# it. Readiness = hot worker loaded AND no hot error AND breaker closed.
_consecutive_engine_failures: int = 0
_breaker_error: Optional[str] = None
_engine_probe_task: Optional[asyncio.Task] = None

# VAD state. The Silero model carries internal state and is NOT thread-safe on
# a single instance, so every use is serialised under this lock.
_vad_model = None
_vad_broken: bool = False
_vad_lock = threading.Lock()


# -------------------------------
# Prometheus metrics
# -------------------------------
#
# Naming is shared with uttera-stt-vllm (`uttera_stt_*`) so a dashboard
# that aggregates across both backends can use the same queries. The
# engine label in uttera_stt_build_info differentiates the variant
# ("whisper-hotcold" here, "vllm" on the sibling). Hot/cold-specific
# series (queue depth, cold worker counts, lane routing, VRAM) are
# additive — they only exist on this server.

# --- HTTP-level (shared shape with sibling vllm backend) ---
_HTTP_REQUESTS_TOTAL = Counter(
    "uttera_stt_requests_total",
    "HTTP requests by endpoint, method and status code",
    ["endpoint", "method", "status"],
)
_HTTP_REQUEST_DURATION = Histogram(
    "uttera_stt_request_duration_seconds",
    "HTTP request wall-clock duration in seconds",
    ["endpoint", "method"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
_INFLIGHT_GAUGE = Gauge(
    "uttera_stt_inflight_requests",
    "Requests currently being processed (hot + cold lanes combined)",
)
_ENGINE_READY_GAUGE = Gauge(
    "uttera_stt_engine_ready",
    "1 if the hot worker's Whisper model is loaded and ready, 0 otherwise",
)
_LIBRETRANSLATE_CONFIGURED_GAUGE = Gauge(
    "uttera_stt_libretranslate_configured",
    "1 if LIBRETRANSLATE_URL is set and translations go through LibreTranslate",
)

# --- Shared request-shape counters ---
_TRANSCRIPTIONS_TOTAL = Counter(
    "uttera_stt_transcriptions_total",
    "Transcription requests broken down by requested response_format",
    ["response_format"],
)
_TRANSLATIONS_TOTAL = Counter(
    "uttera_stt_translations_total",
    "Translation requests broken down by post-processing mode and response_format",
    ["mode", "response_format"],
)
_AUDIO_SECONDS_TOTAL = Counter(
    "uttera_stt_audio_seconds_total",
    "Total seconds of audio successfully processed (billing / throughput proxy)",
    ["endpoint", "route"],
)

# --- Inference-duration histogram (per-op) ---
# op values on hot/cold:
#   whisper_transcribe_hot   — the always-resident hot worker handled it
#   whisper_transcribe_cold  — a cold-pool subprocess handled it
#   libretranslate           — LibreTranslate HTTP round-trip
_INFERENCE_DURATION = Histogram(
    "uttera_stt_inference_duration_seconds",
    "Per-call inference latency in seconds, by op",
    ["op"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

_ERRORS_TOTAL = Counter(
    "uttera_stt_errors_total",
    "Errors by type",
    ["type"],   # decode | validation | model | libretranslate | timeout
)

_BUILD_INFO = Gauge(
    "uttera_stt_build_info",
    "Build metadata (label values carry version, engine and served model id)",
    ["version", "engine", "model"],
)

# --- Hot/cold pool specific (additive vs the vllm sibling) ---
_REQUESTS_BY_ROUTE_TOTAL = Counter(
    "uttera_stt_requests_by_route_total",
    "Successful requests broken down by which lane ultimately served them",
    ["route"],   # HOT | COLD-POOL | COLD-POOL>HOT
)
_COLD_WORKERS_ACTIVE_GAUGE = Gauge(
    "uttera_stt_cold_workers_active",
    "Cold worker subprocesses currently alive and consuming from the work queue",
)
_COLD_WORKERS_LOADING_GAUGE = Gauge(
    "uttera_stt_cold_workers_loading",
    "Cold worker subprocesses currently in their spawn/load phase",
)
_COLD_WORKER_POOL_CAP_GAUGE = Gauge(
    "uttera_stt_cold_worker_pool_size_cap",
    "Configured COLD_POOL_SIZE — the ceiling on active + loading cold workers",
)
_COLD_WORKERS_SPAWNED_TOTAL = Counter(
    "uttera_stt_cold_workers_spawned_total",
    "Total cold worker subprocesses ever successfully spawned (monotonic)",
)
_COLD_EMA_START_GAUGE = Gauge(
    "uttera_stt_cold_worker_ema_start_seconds",
    "Rolling EMA of cold worker boot time (seconds from spawn to ready-to-serve)",
)
_WORK_QUEUE_DEPTH_GAUGE = Gauge(
    "uttera_stt_work_queue_depth",
    "Items currently queued waiting for a hot or cold worker",
)
_WORK_QUEUE_AUDIO_SECONDS_GAUGE = Gauge(
    "uttera_stt_work_queue_audio_seconds",
    "Sum of audio durations waiting in the work queue (for drain-estimate latency)",
)
_HOT_EMA_SPS_GAUGE = Gauge(
    "uttera_stt_hot_ema_sps",
    "Rolling EMA of the hot worker's seconds-of-audio per second of wall-time (throughput proxy)",
)
_VRAM_FREE_GB_GAUGE = Gauge(
    "uttera_stt_vram_free_gb",
    "Free VRAM on the serving GPU in GB",
)
_VRAM_PER_COLD_WORKER_GB_GAUGE = Gauge(
    "uttera_stt_vram_per_cold_worker_gb",
    "Rolling EMA of VRAM consumed by each cold worker subprocess, in GB",
)

# Static gauges — set once at module import time, after all metric
# definitions above have registered their series.
_BUILD_INFO.labels(
    version=SERVER_VERSION,
    engine="whisper-hotcold",
    model=os.environ.get("WHISPER_MODEL", "medium"),
).set(1)
_LIBRETRANSLATE_CONFIGURED_GAUGE.set(
    1 if os.environ.get("LIBRETRANSLATE_URL", "").strip() else 0
)
_COLD_WORKER_POOL_CAP_GAUGE.set(COLD_POOL_SIZE)

_KNOWN_ENDPOINTS = {
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/models",
    "/health",
    "/metrics",
}


@dataclasses.dataclass
class _WorkItem:
    """One pending transcription/translation request on the shared work queue."""
    audio_bytes: bytes
    language: Optional[str]
    prompt: Optional[str]
    temperature: float
    task: str           # "transcribe" or "translate"
    audio_dur: float    # estimated audio duration in seconds
    future: asyncio.Future
    route: str = dataclasses.field(default="HOT")  # set by whichever worker serves it
    retried: bool = dataclasses.field(default=False)  # True if re-queued after pool worker failure


def _update_cold_vram_ema(vram_gb: float) -> None:
    global _cold_vram_ema_gb
    if vram_gb <= 0:
        return
    if _cold_vram_ema_gb is None:
        _cold_vram_ema_gb = vram_gb
    else:
        _cold_vram_ema_gb = _COLD_VRAM_EMA_ALPHA * vram_gb + (1.0 - _COLD_VRAM_EMA_ALPHA) * _cold_vram_ema_gb


def _vram_per_cold_worker() -> float:
    if _cold_vram_ema_gb is not None:
        return _cold_vram_ema_gb * _COLD_VRAM_SAFETY_FACTOR
    return MIN_COLD_VRAM_GB


# ── Persistent cold worker class ──────────────────────────────────────────────

class _ColdWorker:
    """
    A persistent cold worker subprocess (cold_worker.py).

    The subprocess loads the Whisper model once on startup, then serves
    successive transcription requests via newline-delimited JSON on stdin/stdout.
    """

    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None
        self.alive: bool = False

    async def spawn(self) -> bool:
        env = os.environ.copy()
        env["WHISPER_CACHE_DIR"] = MODEL_CACHE_DIR
        env["WHISPER_MODEL"] = model_name
        env["WHISPER_FP16"] = "1" if WHISPER_FP16 else "0"
        env["COLD_WORKER_IDLE_TIMEOUT"] = str(COLD_WORKER_IDLE_TIMEOUT)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                VENV_PYTHON, COLD_WORKER_SCRIPT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=90.0)
            msg = json.loads(line)
            if msg.get("ready"):
                self.alive = True
                return True
        except Exception as exc:
            print(f"COLD WORKER: spawn failed: {exc}", flush=True)
            await self.shutdown()
        return False

    def is_alive(self) -> bool:
        return self.alive and self._proc is not None and self._proc.returncode is None

    async def transcribe(self, audio_bytes: bytes, language, prompt, temp: float, task: str) -> dict:
        req = {
            "audio_b64": base64.b64encode(audio_bytes).decode(),
            "language": language,
            "prompt": prompt,
            "temperature": temp,
            "task": task,
        }
        self._proc.stdin.write((json.dumps(req) + "\n").encode())
        await self._proc.stdin.drain()
        resp_line = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=COLD_LANE_TIMEOUT_SECONDS
        )
        resp = json.loads(resp_line)
        if "error" in resp:
            self.alive = False
            raise RuntimeError(f"Cold worker error: {resp['error']}")
        if resp.get("exit"):
            self.alive = False
            raise RuntimeError(f"Cold worker exited unexpectedly: {resp.get('exit')}")
        return resp["result"]

    async def shutdown(self) -> None:
        self.alive = False
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            self._proc.stdin.write((json.dumps({"exit": True}) + "\n").encode())
            await self._proc.stdin.drain()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except Exception:
            pass
        if self._proc.returncode is None:
            self._proc.kill()


def _update_cold_ema_stt(elapsed: float) -> None:
    global _cold_ema_start_stt
    if _cold_ema_start_stt is None:
        _cold_ema_start_stt = elapsed
    else:
        _cold_ema_start_stt = _COLD_EMA_ALPHA_STT * elapsed + (1.0 - _COLD_EMA_ALPHA_STT) * _cold_ema_start_stt


def _get_cold_start_time_stt() -> float:
    return _cold_ema_start_stt if _cold_ema_start_stt is not None else COLD_START_TIME_SECONDS


def _free_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def _has_vram_for_cold_lane() -> bool:
    if MIN_COLD_VRAM_GB <= 0:
        return True
    free = _free_vram_gb()
    if free is None:
        return True
    needed = _vram_per_cold_worker()
    effective_free = free - (_cold_workers_in_flight * needed)
    return effective_free >= needed


def _get_audio_duration(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return len(audio_bytes) / 32000


def _update_hot_ema_stt(elapsed: float, audio_duration: float) -> None:
    global _hot_ema_sps
    sps = elapsed / max(audio_duration, 0.1)
    if _hot_ema_sps is None:
        _hot_ema_sps = sps
    else:
        _hot_ema_sps = _HOT_EMA_ALPHA * sps + (1.0 - _HOT_EMA_ALPHA) * _hot_ema_sps


def _optimal_cold_workers() -> int:
    """
    Compute the optimal number of cold worker tasks given current queue depth.

    With N total workers (1 hot + (N-1) cold) loading serially at cold_ema seconds each,
    the last cold worker contributes only if it finishes loading before the burst ends:

        T_end(N) = total_work_s / N + cold_ema * (N-1) / 2

    The last cold worker (loading at (N-1)*cold_ema) helps iff (N-1)*cold_ema < T_end(N),
    which simplifies to:

        N * (N-1) < 2 * total_work_s / cold_ema
        where total_work_s = _work_queue_audio_seconds * _hot_ema_sps (real seconds, serial)

    Capped by COLD_POOL_SIZE (safety) and available VRAM.
    """
    if _hot_ema_sps is None or _work_queue_audio_seconds <= 0:
        return 0
    cold_start = _get_cold_start_time_stt()
    if cold_start <= 0:
        return 0

    total_work_s = _work_queue_audio_seconds * _hot_ema_sps  # real seconds if processed serially
    limit = 2.0 * total_work_s / cold_start

    # Find largest N_total where N*(N-1) < limit
    N_total = 1
    while N_total * (N_total - 1) < limit:
        N_total += 1
    N_total -= 1  # step back to largest satisfying value
    cold = N_total - 1  # subtract the hot worker

    # Cap by operator safety limit
    if COLD_POOL_SIZE > 0:
        cold = min(cold, COLD_POOL_SIZE)

    return max(0, cold)


# -------------------------------
# 3. Transcription Functions
# -------------------------------

def _run_hot_locked(audio_bytes: bytes, language: Optional[str], prompt: Optional[str], temp: float, task: str = "transcribe") -> tuple:
    """
    Acquire model_lock, transcribe, release — all inside a single thread.
    Returns (result, processing_elapsed).
    """
    model_lock.acquire()
    t_proc = time.monotonic()
    try:
        result = run_transcription_fast_lane(audio_bytes, language, prompt, temp, task)
        return result, time.monotonic() - t_proc
    finally:
        model_lock.release()


def run_transcription_fast_lane(audio_bytes: bytes, language: Optional[str], prompt: Optional[str], temp: float, task: str = "transcribe") -> dict:
    log_debug(f"--- MAIN LANE: Using hot worker (GPU), task={task} ---")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as t:
            t.write(audio_bytes)
            temp_path = t.name
        use_fp16 = torch.cuda.is_available()
        return whisper_model.transcribe(temp_path, language=language, initial_prompt=prompt, temperature=temp, task=task, fp16=use_fp16)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def _spawn_cold_worker_with_vram(task_label: str = "") -> _ColdWorker:
    """Spawn a new _ColdWorker, measure VRAM, return the ready worker."""
    global _cold_workers_in_flight
    v_before = _free_vram_gb() or 0.0
    _cold_workers_in_flight += 1
    worker = _ColdWorker()
    try:
        spawned = await worker.spawn()
        if not spawned:
            raise RuntimeError("Cold worker subprocess failed to start")
        if _cold_workers_in_flight == 1:
            v_after = _free_vram_gb() or 0.0
            drop = v_before - v_after
            if drop > 0:
                _update_cold_vram_ema(drop)
                log_debug(f"--- COLD VRAM: drop={drop:.2f} GB → EMA={_cold_vram_ema_gb:.2f} GB{' ' + task_label if task_label else ''} ---")
        _cold_workers_in_flight -= 1
        return worker
    except Exception:
        _cold_workers_in_flight -= 1
        await worker.shutdown()
        raise


# -------------------------------
# 3b. Worker Loops
# -------------------------------

async def _hot_worker_loop() -> None:
    """
    Persistent asyncio Task that consumes _WorkItem entries from _work_queue
    and processes them through the hot (in-process) Whisper model.
    Exactly one instance runs for the lifetime of the server.
    """
    global _work_queue_audio_seconds
    while True:
        item = await _work_queue.get()
        item.route = "COLD-POOL>HOT" if item.retried else "HOT"
        try:
            result, elapsed = await asyncio.to_thread(
                _run_hot_locked, item.audio_bytes, item.language, item.prompt, item.temperature, item.task
            )
            _update_hot_ema_stt(elapsed, item.audio_dur)
            if not item.future.done():
                item.future.set_result(result)
        except asyncio.CancelledError:
            if not item.future.done():
                item.future.cancel()
            raise  # finally handles decrement
        except Exception as e:
            if not item.future.done():
                item.future.set_exception(e)
        finally:
            _work_queue_audio_seconds -= item.audio_dur


async def _pool_worker_loop(worker: _ColdWorker, idle_timeout: float = float(COLD_WORKER_IDLE_TIMEOUT)) -> None:
    """
    Persistent asyncio Task for one cold pool worker.
    Consumes _WorkItem entries from the same _work_queue as the hot worker.
    Exits on idle timeout or worker failure; pool manager spawns replacements if needed.

    idle_timeout is set at spawn time (staggered by pool manager) so workers wind down
    one-by-one after a burst instead of all dying simultaneously.
    """
    global _work_queue_audio_seconds
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    _work_queue.get(), timeout=idle_timeout
                )
            except asyncio.TimeoutError:
                print(f"--- POOL WORKER: idle timeout ({idle_timeout:.0f}s), exiting ---", flush=True)
                break
            except asyncio.CancelledError:
                break

            # Retried items are reserved for the hot worker — put back and skip.
            if item.retried:
                await _work_queue.put(item)
                continue

            item.route = "COLD-POOL"
            requeued = False
            try:
                result = await worker.transcribe(
                    item.audio_bytes, item.language, item.prompt, item.temperature, item.task
                )
                if not item.future.done():
                    item.future.set_result(result)
            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                raise  # finally handles decrement
            except Exception as e:
                print(f"--- POOL WORKER: transcribe failed ({e}), re-queuing to hot lane ---", flush=True)
                if not item.future.done():
                    # Re-queue for the hot worker to rescue — don't decrement accounting
                    item.retried = True
                    _work_queue_audio_seconds += item.audio_dur  # restore before finally decrements
                    await _work_queue.put(item)
                    requeued = True
                if not worker.is_alive():
                    break
            finally:
                if not requeued:
                    _work_queue_audio_seconds -= item.audio_dur
    finally:
        await worker.shutdown()


# -------------------------------
# 3c. Startup EMA Warmup
# -------------------------------

def _make_silence_wav(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b'\x00' * n_frames * 2)
    return buf.getvalue()


async def _warmup_ema():
    """Seed _hot_ema_sps with a synthetic silence clip at startup."""
    if whisper_model is None:
        return
    print("WARMUP HOT: Seeding EMA with synthetic silence clip...", flush=True)
    try:
        silence = _make_silence_wav(duration_seconds=2.0)
        audio_dur = _get_audio_duration(silence)
        t0 = time.monotonic()
        await asyncio.to_thread(run_transcription_fast_lane, silence, None, None, 0.0)
        elapsed = time.monotonic() - t0
        _update_hot_ema_stt(elapsed, audio_dur)
        print(f"WARMUP HOT: sps={_hot_ema_sps:.4f} (elapsed={elapsed:.2f}s for {audio_dur:.1f}s audio)", flush=True)
    except Exception as e:
        print(f"WARMUP HOT: Failed ({e}). Starting in uncalibrated mode.", flush=True)


async def _warmup_cold_ema():
    """
    Spawn one cold worker at startup, transcribe silence, record total elapsed as
    cold_ema_start_stt, measure VRAM drop, then kill the worker.
    """
    if COLD_POOL_SIZE <= 0:
        return
    print("WARMUP COLD: Spawning cold worker to calibrate cold_ema...", flush=True)
    global _cold_workers_in_flight
    v_before = _free_vram_gb() or 0.0
    _cold_workers_in_flight += 1
    worker = _ColdWorker()
    t_start = time.monotonic()
    try:
        spawned = await worker.spawn()
        if not spawned:
            print("WARMUP COLD: Worker failed to start. cold_ema uncalibrated.", flush=True)
            return
        v_after = _free_vram_gb() or 0.0
        drop = v_before - v_after
        if drop > 0:
            _update_cold_vram_ema(drop)
        _cold_workers_in_flight -= 1
        silence = _make_silence_wav(duration_seconds=2.0)
        await worker.transcribe(silence, None, None, 0.0, "transcribe")
        total_elapsed = time.monotonic() - t_start
        _update_cold_ema_stt(total_elapsed)
        print(
            f"WARMUP COLD: cold_ema={total_elapsed:.1f}s | "
            f"vram_drop={drop:.2f}GB (EMA={_cold_vram_ema_gb:.2f}GB) | "
            f"threshold={_get_cold_start_time_stt() * HOT_QUEUE_SAFETY_FACTOR:.1f}s",
            flush=True
        )
    except Exception as e:
        print(f"WARMUP COLD: Failed ({e}). cold_ema uncalibrated.", flush=True)
        if _cold_workers_in_flight > 0:
            _cold_workers_in_flight -= 1
    finally:
        await worker.shutdown()


# -------------------------------
# 3d. Cold Pool Manager
# -------------------------------

async def _cold_pool_manager() -> None:
    """
    Background task that dynamically spawns cold workers based on current queue load.

    Every COLD_POOL_MANAGER_INTERVAL seconds, computes _optimal_cold_workers() and spawns
    one more worker (serially, via _cold_spawn_lock) if active+loading < optimal.
    Each spawned worker runs as an independent _pool_worker_loop Task consuming from
    _work_queue alongside the hot worker.
    """
    while True:
        await asyncio.sleep(COLD_POOL_MANAGER_INTERVAL)

        target = _optimal_cold_workers()
        active = len(_pool_worker_tasks)
        loading = _cold_workers_in_flight

        if active + loading < target and _cold_spawn_lock and not _cold_spawn_lock.locked() and _has_vram_for_cold_lane():
            drain_s = _work_queue_audio_seconds * (_hot_ema_sps or 0)
            print(
                f"--- POOL MGR: target={target} cold workers (active={active}, loading={loading}) "
                f"| queue={_work_queue_audio_seconds:.1f}s audio ({drain_s:.1f}s drain) → spawning ---",
                flush=True
            )
            try:
                async with _cold_spawn_lock:
                    worker = await _spawn_cold_worker_with_vram()
                # Stagger idle timeouts: first spawned → longest, last spawned → COLD_WORKER_IDLE_TIMEOUT.
                # Ensures workers die one-by-one after a burst rather than simultaneously.
                n_active = len(_pool_worker_tasks)
                stagger_extra = max(0, COLD_POOL_SIZE - n_active) * COLD_WORKER_IDLE_STAGGER
                worker_idle_timeout = float(COLD_WORKER_IDLE_TIMEOUT + stagger_extra)
                task = asyncio.create_task(_pool_worker_loop(worker, worker_idle_timeout))
                _pool_worker_tasks.add(task)
                task.add_done_callback(_pool_worker_tasks.discard)
                _COLD_WORKERS_SPAWNED_TOTAL.inc()
                print(
                    f"--- POOL MGR: pool worker ready, total_active={len(_pool_worker_tasks)}"
                    f", idle_timeout={worker_idle_timeout:.0f}s ---",
                    flush=True
                )
            except Exception as e:
                print(f"--- POOL MGR: spawn failed: {e} ---", flush=True)


# -------------------------------
# 4. Endpoints
# -------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "uttera"},
        ]
    }

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    _free = _free_vram_gb()
    optimal = _optimal_cold_workers()
    ready = _engine_is_ready()
    pool_stats = {
        "ema_sps": round(_hot_ema_sps, 4) if _hot_ema_sps is not None else None,
        "cold_start_calibrated": _cold_ema_start_stt is not None,
        "cold_ema_start_seconds": round(_cold_ema_start_stt, 2) if _cold_ema_start_stt is not None else None,
        "queue_depth": _work_queue.qsize() if _work_queue is not None else 0,
        "queue_audio_seconds": round(_work_queue_audio_seconds, 2),
        "pool_workers_active": len(_pool_worker_tasks),
        "pool_workers_loading": _cold_workers_in_flight,
        "pool_workers_optimal": optimal,
        "pool_size_cap": COLD_POOL_SIZE,
        "vram_free_gb": round(_free, 2) if _free is not None else None,
        "cold_vram_ema_gb": round(_cold_vram_ema_gb, 3) if _cold_vram_ema_gb is not None else None,
        "vram_sufficient_for_cold": _has_vram_for_cold_lane(),
        "cold_start_configured_seconds": COLD_START_TIME_SECONDS,
        "safety_factor": HOT_QUEUE_SAFETY_FACTOR,
        "min_cold_vram_gb": MIN_COLD_VRAM_GB,
        "cold_vram_per_worker_gb": round(_vram_per_cold_worker(), 3),
        "consecutive_engine_failures": _consecutive_engine_failures,
    }
    body = {
        "status": "ok" if ready else "starting",
        "version": SERVER_VERSION,
        "model": model_name,
        "hot_worker_loaded": whisper_model is not None,
        "hot_worker_error": hot_worker_error,
        "engine_error": _breaker_error,
        "limits": {"max_filesize_mb": MAX_FILESIZE_MB},
        "pool": pool_stats,
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


def _refresh_gauges_from_state() -> None:
    """Snapshot the live routing state into Prometheus gauges.

    Called on every `/metrics` scrape so the numbers are always
    current without hooking every state-change site. Mirrors what
    `health_check()` computes.
    """
    _ENGINE_READY_GAUGE.set(1 if _engine_is_ready() else 0)
    _COLD_WORKERS_ACTIVE_GAUGE.set(len(_pool_worker_tasks))
    _COLD_WORKERS_LOADING_GAUGE.set(_cold_workers_in_flight)
    _WORK_QUEUE_DEPTH_GAUGE.set(_work_queue.qsize() if _work_queue is not None else 0)
    _WORK_QUEUE_AUDIO_SECONDS_GAUGE.set(_work_queue_audio_seconds)
    if _hot_ema_sps is not None:
        _HOT_EMA_SPS_GAUGE.set(_hot_ema_sps)
    if _cold_ema_start_stt is not None:
        _COLD_EMA_START_GAUGE.set(_cold_ema_start_stt)
    _free = _free_vram_gb()
    if _free is not None:
        _VRAM_FREE_GB_GAUGE.set(_free)
    _VRAM_PER_COLD_WORKER_GB_GAUGE.set(_vram_per_cold_worker())
    # _INFLIGHT_GAUGE is not refreshed here — it's maintained
    # precisely by the transcription/translation endpoint handlers.


@app.get("/metrics")
async def metrics():
    """Prometheus-format scrape endpoint.

    Exposes both the generic HTTP-level metrics (tracked via the
    middleware) and this server's hot/cold-specific pool telemetry
    (cold workers active/loading/spawned, queue depth, VRAM).
    Cardinality is bounded by design — no per-request-id labels, no
    language labels, no voice labels.

    Scrape with Telegraf's `inputs.prometheus`, Prometheus itself,
    or any OpenMetrics-compatible consumer.
    """
    _refresh_gauges_from_state()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _validate_common_params(response_format: str, temperature: float) -> None:
    """Validate OpenAI-compat request params before doing any work. Raises HTTPException."""
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"response_format '{response_format}' is not supported. "
                f"Valid values: {sorted(SUPPORTED_RESPONSE_FORMATS)}"
            ),
        )
    if not (TEMPERATURE_MIN <= temperature <= TEMPERATURE_MAX):
        raise HTTPException(
            status_code=422,
            detail=(
                f"temperature {temperature} out of range. "
                f"Must be in [{TEMPERATURE_MIN}, {TEMPERATURE_MAX}]."
            ),
        )


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT."""
    return _format_timestamp_srt(seconds).replace(",", ".")


def _segments_to_srt(segments: list) -> str:
    """Convert whisper segments list to SRT."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(seg.get("start", 0.0))
        end = _format_timestamp_srt(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _segments_to_vtt(segments: list) -> str:
    """Convert whisper segments list to WebVTT."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(seg.get("start", 0.0))
        end = _format_timestamp_vtt(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _render_response(res: dict, response_format: str, route: str, extra_headers: dict | None = None) -> Response:
    """Render the whisper result dict as the requested response_format per OpenAI spec.

    - `json`: OpenAI-compact `{"text": "..."}`
    - `text`: plain text body
    - `verbose_json`: full whisper result (text + segments + language + logprobs)
    - `srt`: SRT subtitle format
    - `vtt`: WebVTT subtitle format
    """
    headers = {"X-Route": route, **(extra_headers or {})}
    text = (res.get("text") or "")
    if response_format == "text":
        return PlainTextResponse(content=text, headers=headers)
    if response_format == "srt":
        return PlainTextResponse(
            content=_segments_to_srt(res.get("segments") or []),
            media_type="application/x-subrip",
            headers=headers,
        )
    if response_format == "vtt":
        return PlainTextResponse(
            content=_segments_to_vtt(res.get("segments") or []),
            media_type="text/vtt",
            headers=headers,
        )
    if response_format == "verbose_json":
        # Whisper's raw result IS the verbose format (segments+language+logprobs).
        return JSONResponse(content=res, headers=headers)
    # Default: OpenAI-compact json — only {"text": ...}.
    return JSONResponse(content={"text": text}, headers=headers)


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: float = Form(0.0)
):
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    _validate_common_params(response_format, temperature)
    _TRANSCRIPTIONS_TOTAL.labels(response_format=response_format).inc()
    _INFLIGHT_GAUGE.inc()
    req_t0 = time.monotonic()
    try:
        contents = await file.read()
        # 413, not 400: the file is valid, it just doesn't fit. A 4xx doesn't
        # count towards the circuit breaker.
        _mb = len(contents) / 1048576
        if _mb > MAX_FILESIZE_MB:
            raise HTTPException(status_code=413,
                                detail=f"File is {_mb:.0f} MB; the limit is {MAX_FILESIZE_MB} MB.")
        try:
            audio_dur = _get_audio_duration(contents)
        except Exception as e:
            # ffmpeg / libsndfile failed to decode — client sent a non-audio file
            # or an unsupported codec. 400 is the right answer.
            _ERRORS_TOTAL.labels(type="decode").inc()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode audio: {type(e).__name__}: {str(e)[:200]}",
            )

        # Voice-activity gate, BEFORE occupying a worker: with no speech there
        # is nothing to transcribe and we save the GPU.
        if await asyncio.to_thread(_no_voice, contents):
            print("VAD: no speech; the model is not called", file=sys.stderr, flush=True)
            _engine_ok()
            return _render_response({"text": "", "segments": [], "language": ""},
                                    response_format, "VAD-SKIP")

        global _work_queue_audio_seconds
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _WorkItem(
            audio_bytes=contents,
            language=language,
            prompt=prompt,
            temperature=temperature,
            task="transcribe",
            audio_dur=audio_dur,
            future=future,
        )
        _work_queue_audio_seconds += audio_dur
        await _work_queue.put(item)

        res = await future
        route = item.route
        # Lane-tagged bookkeeping: which lane served this, how much audio
        # it processed, and how long whisper took from queue-dispatch.
        _REQUESTS_BY_ROUTE_TOTAL.labels(route=route).inc()
        _AUDIO_SECONDS_TOTAL.labels(
            endpoint="/v1/audio/transcriptions", route=route
        ).inc(audio_dur)
        op = "whisper_transcribe_cold" if route in ("COLD-POOL",) else "whisper_transcribe_hot"
        _INFERENCE_DURATION.labels(op=op).observe(time.monotonic() - req_t0)
        _engine_ok()
        return _render_response(res, response_format, route)
    except HTTPException:
        raise
    except ValueError as e:
        # Whisper raises ValueError for "Unsupported language: XX" and similar.
        # Surface the actual message instead of a generic 500.
        msg = str(e)
        _ERRORS_TOTAL.labels(type="validation").inc()
        print(f"ERROR in create_transcription (ValueError): {msg}", flush=True)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        # Re-raise known HTTP errors, but surface ValueError from the worker loop too.
        msg = str(e)
        if isinstance(e, ValueError) or "Unsupported language" in msg or ("language" in msg.lower() and "not supported" in msg.lower()):
            _ERRORS_TOTAL.labels(type="validation").inc()
            print(f"ERROR in create_transcription (language): {msg}", flush=True)
            raise HTTPException(status_code=400, detail=msg)
        # Whisper's "Failed to load audio: ..." comes from ffmpeg choking on
        # a non-audio / unsupported-codec body — a client error, not a server
        # fault. Truncate the ffmpeg output since it includes the full build
        # banner on every failure.
        if "Failed to load audio" in msg:
            _ERRORS_TOTAL.labels(type="decode").inc()
            print(f"ERROR in create_transcription (decode): {msg[:200]}", flush=True)
            raise HTTPException(
                status_code=400,
                detail="Failed to decode audio body — not a valid audio stream or unsupported codec.",
            )
        _ERRORS_TOTAL.labels(type="model").inc()
        print(f"ERROR in create_transcription: {e}", flush=True)
        _engine_failure(500, e)
        raise HTTPException(status_code=500, detail="Transcription failed. Check server logs.")
    finally:
        _INFLIGHT_GAUGE.dec()
        await file.close()

# Whisper emits ISO-639-1 language codes (e.g. "zh"); LibreTranslate
# expects different codes for a few Asian languages. Map at the boundary.
_WHISPER_TO_LIBRETRANSLATE_LANG = {
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-tw": "zh-Hant",
}


def _normalise_lang_for_libretranslate(code: str) -> str:
    if not code:
        return code
    code = code.lower()
    return _WHISPER_TO_LIBRETRANSLATE_LANG.get(code, code)


async def _libretranslate(text: str, source: str, target: str) -> str:
    """Call LibreTranslate. Raises on network or HTTP errors; caller maps to 502."""
    import httpx  # imported lazily so servers without LIBRETRANSLATE_URL still start
    src = _normalise_lang_for_libretranslate(source) or "auto"
    tgt = _normalise_lang_for_libretranslate(target)
    payload: dict = {"q": text, "source": src, "target": tgt, "format": "text"}
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY
    async with httpx.AsyncClient(timeout=LIBRETRANSLATE_TIMEOUT_S) as client:
        r = await client.post(f"{LIBRETRANSLATE_URL}/translate", json=payload)
        r.raise_for_status()
        data = r.json()
    out = data.get("translatedText")
    if not isinstance(out, str):
        raise RuntimeError(f"Unexpected LibreTranslate response: {data}")
    return out


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: float = Form(0.0),
    to_language: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """
    OpenAI-compatible translation endpoint.

    When `LIBRETRANSLATE_URL` is configured (recommended), the audio is
    first transcribed with Whisper (in the source language, either
    auto-detected or forced via the `language` form field) and the text
    is then sent to LibreTranslate to reach `to_language` (default
    `"en"` for OpenAI-compatibility). This path supports any target
    language supported by the LibreTranslate instance, not only English.

    When `LIBRETRANSLATE_URL` is empty, the endpoint falls back to the
    legacy Whisper-native `translate` task, which is English-only and
    works poorly on models that were not trained for it (e.g.
    whisper-large-v3-turbo).
    """
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    _validate_common_params(response_format, temperature)
    target_lang = (to_language or "en").lower()
    _mode_label = "libretranslate" if LIBRETRANSLATE_URL else "native"
    _TRANSLATIONS_TOTAL.labels(mode=_mode_label, response_format=response_format).inc()
    _INFLIGHT_GAUGE.inc()
    whisper_t0 = time.monotonic()

    # Contract gate: in legacy Whisper-native mode (no LibreTranslate configured)
    # only target=en is actually supported. Silently falling back to English
    # when the caller explicitly asked for another language is a contract
    # violation, so reject with 400 instead.
    if not LIBRETRANSLATE_URL and target_lang != "en":
        _ERRORS_TOTAL.labels(type="validation").inc()
        _INFLIGHT_GAUGE.dec()
        raise HTTPException(
            status_code=400,
            detail=(
                f"to_language={target_lang!r} requested but LIBRETRANSLATE_URL is not "
                f"configured on this server. Only 'en' is supported in legacy "
                f"Whisper-native translate mode. Configure LIBRETRANSLATE_URL "
                f"to enable arbitrary target languages."
            ),
        )

    try:
        contents = await file.read()
        _mb = len(contents) / 1048576
        if _mb > MAX_FILESIZE_MB:
            raise HTTPException(status_code=413,
                                detail=f"File is {_mb:.0f} MB; the limit is {MAX_FILESIZE_MB} MB.")
        try:
            audio_dur = _get_audio_duration(contents)
        except Exception as e:
            _ERRORS_TOTAL.labels(type="decode").inc()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode audio: {type(e).__name__}: {str(e)[:200]}",
            )

        global _work_queue_audio_seconds
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Route: if LibreTranslate is configured, do a plain transcription
        # and translate the text afterwards. Otherwise, legacy native
        # Whisper translate (English-only).
        item = _WorkItem(
            audio_bytes=contents,
            language=language,
            prompt=prompt,
            temperature=temperature,
            task="transcribe" if LIBRETRANSLATE_URL else "translate",
            audio_dur=audio_dur,
            future=future,
        )
        _work_queue_audio_seconds += audio_dur
        await _work_queue.put(item)

        res = await future
        route = item.route
        # Lane-tagged bookkeeping (shared with /v1/audio/transcriptions).
        _REQUESTS_BY_ROUTE_TOTAL.labels(route=route).inc()
        _AUDIO_SECONDS_TOTAL.labels(
            endpoint="/v1/audio/translations", route=route
        ).inc(audio_dur)
        _op_whisper = "whisper_transcribe_cold" if route in ("COLD-POOL",) else "whisper_transcribe_hot"
        _INFERENCE_DURATION.labels(op=_op_whisper).observe(time.monotonic() - whisper_t0)
        # The transcription (the engine step) succeeded — clear the breaker
        # regardless of what the LibreTranslate step does next.
        _engine_ok()

        # Legacy path: no LibreTranslate, return Whisper native translation.
        if not LIBRETRANSLATE_URL:
            return _render_response(res, response_format, route)

        # LibreTranslate path.
        text = (res.get("text") or "").strip()
        source_lang = (res.get("language") or "").lower()
        log_debug(
            f"[translate] source={source_lang!r} target={target_lang!r} "
            f"route={route} text_preview={text[:80]!r}"
        )

        if not text:
            out_text = text
        elif source_lang and source_lang == target_lang:
            out_text = text  # no-op shortcut
        else:
            try:
                with _INFERENCE_DURATION.labels(op="libretranslate").time():
                    out_text = await _libretranslate(text, source_lang, target_lang)
            except Exception as e:
                _ERRORS_TOTAL.labels(type="libretranslate").inc()
                print(f"ERROR in _libretranslate: {e}", flush=True)
                raise HTTPException(
                    status_code=502,
                    detail=f"Translation backend failure: {type(e).__name__}: {e}",
                )

        # Rebuild a whisper-shaped result dict with the translated text so
        # _render_response can handle every format (including srt/vtt, which
        # need the original segments — we preserve them since timings stay valid).
        out_res = dict(res)
        out_res["text"] = out_text
        extra_headers = {"X-Translation-Mode": "libretranslate"}
        return _render_response(out_res, response_format, route, extra_headers=extra_headers)

    except HTTPException:
        raise
    except ValueError as e:
        msg = str(e)
        _ERRORS_TOTAL.labels(type="validation").inc()
        print(f"ERROR in create_translation (ValueError): {msg}", flush=True)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        msg = str(e)
        if "Unsupported language" in msg or ("language" in msg.lower() and "not supported" in msg.lower()):
            _ERRORS_TOTAL.labels(type="validation").inc()
            print(f"ERROR in create_translation (language): {msg}", flush=True)
            raise HTTPException(status_code=400, detail=msg)
        if "Failed to load audio" in msg:
            _ERRORS_TOTAL.labels(type="decode").inc()
            print(f"ERROR in create_translation (decode): {msg[:200]}", flush=True)
            raise HTTPException(
                status_code=400,
                detail="Failed to decode audio body — not a valid audio stream or unsupported codec.",
            )
        _ERRORS_TOTAL.labels(type="model").inc()
        print(f"ERROR in create_translation: {e}", flush=True)
        _engine_failure(500, e)
        raise HTTPException(status_code=500, detail="Translation failed. Check server logs.")
    finally:
        _INFLIGHT_GAUGE.dec()
        await file.close()

if __name__ == "__main__":
    uvicorn.run("main_stt:app", host="0.0.0.0", port=9005, log_level="info")
