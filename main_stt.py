#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Whisper STT Server (Hybrid Model)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# Package: whisper-stt-server
# Version: 1.5.0
# Maintainer: Uttera, Hugo L. Espuny
# Description: High-performance STT server with GPU acceleration and concurrency.
#
# CHANGELOG:
# - 1.5.0 (2026-04-08): Cold worker pool. Persistent subprocesses (cold_worker.py) load the
#   Whisper model once and serve multiple requests via newline-delimited JSON on stdin/stdout,
#   eliminating the ~18-20 s model-load cost on every cold lane request. New routing branch
#   (Branch 0 / COLD-POOL) fires when an idle pool worker is available and the hot lane is
#   busy — inference-only cost (~5-7 s) vs. waiting for hot or spawning a new cold worker.
#   After any new cold spawn completes, the worker is returned to the pool (up to COLD_POOL_SIZE
#   idle slots; excess workers shut down gracefully). VRAM is now measured directly right after
#   the worker signals ready (model loaded) instead of with a delayed asyncio.sleep sampler.
#   New globals: _cold_inference_ema_stt (inference-only EMA for pool routing), _cold_idle_pool
#   (asyncio.Queue of idle _ColdWorker instances). New env vars: COLD_POOL_SIZE (default 2),
#   COLD_WORKER_IDLE_TIMEOUT (default 60 s). X-Route header now also on translation responses.
#   New route labels: COLD-POOL (pool inference), COLD-POOL→HOT (pool failed, fell back).
# - 1.4.12 (2026-04-08): Auto-calibration of MIN_COLD_VRAM_GB. Samples VRAM 12s after each
#   cold dispatch (model loaded by then) and maintains EMA (_cold_vram_ema_gb × 1.2 safety
#   margin) to replace the static MIN_COLD_VRAM_GB in _has_vram_for_cold_lane(). Fixes the
#   fp16 regression where MIN_COLD_VRAM_GB=4.0 (set for fp32) blocked cold dispatches that
#   would have fit in ~1.4 GB. Exposed as cold_vram_ema_gb and cold_vram_per_worker_gb in
#   GET /health. Falls back to MIN_COLD_VRAM_GB before first cold run completes.
# - 1.4.11 (2026-04-08): Fix EMA inflation from queue wait time. _run_hot_locked now returns
#   (result, proc_elapsed) where proc_elapsed is measured from after lock acquisition, excluding
#   queue wait time. Branch B was previously measuring total time-in-system (wait + process),
#   inflating ema_sps and causing the router to underestimate hot lane throughput, which
#   incorrectly sent requests to cold lane even when hot lane queue drain was well below the
#   cold start threshold.
# - 1.4.10 (2026-04-08): Fallback hot lane retry on transient CUDA errors. When cold lane fails
#   and the hot lane fallback hits a cuDNN/CUBLAS/OOM error (caused by VRAM pressure from dying
#   cold workers not yet released by the driver), retries up to 3 times with exponential backoff
#   (1s, 2s) before giving up. Applied to both transcription and translation endpoints.
# - 1.4.9 (2026-04-08): Cold lane subprocess now passes --fp16 True/False matching WHISPER_FP16.
#   openai-whisper >=20240930 dropped automatic fp16 in the CLI, causing cold workers to load
#   in fp32 (~4.8 GB vs ~1.5 GB in fp16). This restores parity with the hot worker and prevents
#   cascading OOM under concurrent cold lane load.
# - 1.4.8 (2026-04-08): WHISPER_FP16 env var (default "1"). When CUDA is available and
#   WHISPER_FP16=1, the hot-worker model loads on CPU in fp16 then moves to GPU, with
#   LayerNorm weights kept in fp32 to avoid dtype mismatch (whisper LayerNorm does x.float()
#   internally). Saves ~2912 MiB VRAM (−66.5%) vs fp32 on whisper-medium. Set WHISPER_FP16=0
#   to revert to fp32 loading.
# - 1.4.7 (2026-04-07): Fixed SyntaxError: 'global _cold_workers_in_flight' was declared after
#   the variable was first read in a debug f-string inside Branch C of create_transcription and
#   create_translation. Moved global declaration to top of else: block, before any use. Also adds
#   cold lane JSON-missing diagnostic: when whisper exits 0 but produces no output JSON (inaudible
#   or empty audio), full stdout/stderr is logged and a descriptive RuntimeError is raised instead
#   of the opaque [Errno 2] No such file or directory. Validated with 40-clip Spanish stress test
#   (40/40 OK, sim avg=0.989, zero errors, cold EMA 14-15s, max 3 concurrent cold workers).
# - 1.4.6 (2026-04-06): VRAM pre-check before cold lane dispatch. Branch C now queries
#   torch.cuda.mem_get_info() before spawning a cold subprocess. If effective free VRAM
#   (raw free minus in_flight × MIN_COLD_VRAM_GB) is below MIN_COLD_VRAM_GB (default 4.0 GB,
#   configurable via .env), the request is rerouted to the hot lane queue immediately instead of
#   wasting 8-10s on a model load that will OOM mid-way. The in-flight reservation prevents burst
#   routing from over-committing VRAM before any subprocess has allocated. Free VRAM,
#   MIN_COLD_VRAM_GB, cold_workers_in_flight, and vram_sufficient_for_cold exposed in GET /health
#   under smart_routing. Applied to both /v1/audio/transcriptions and /v1/audio/translations.
# - 1.4.5 (2026-04-06): Fixed model_lock deadlock under client timeout (burst load). Branch B and
#   the Branch C fallback previously used two separate asyncio.to_thread calls: one to acquire
#   model_lock and one to run the transcription. If asyncio cancelled the coroutine (client
#   timeout) between the two awaits, model_lock was left permanently acquired, deadlocking the
#   server. Fixed by introducing _run_hot_locked() which performs acquire + transcribe + release
#   inside a single asyncio.to_thread call. Applied to both transcription and translation endpoints.
# - 1.4.4 (2026-04-06): Auto-calibration of COLD_START_TIME_SECONDS. An EMA (alpha=0.2) of
#   measured cold lane completion times now replaces the static COLD_START_TIME_SECONDS as the
#   router threshold once at least one cold lane has completed successfully. COLD_START_TIME_SECONDS
#   in .env becomes an initial hint / fallback. _get_cold_start_time_stt() returns the live EMA
#   or the configured fallback. cold_start_calibrated and cold_ema_start_seconds exposed in
#   GET /health under smart_routing. Applied to both /v1/audio/transcriptions and /v1/audio/translations.
# - 1.4.3 (2026-04-06): EMA no longer updated from fallback path. Fallback elapsed includes
#   cold-lane failure time (~COLD_START_TIME_SECONDS), inflating sps and creating a positive
#   feedback loop (more OOMs → higher EMA → more cold dispatches → more OOMs). EMA is now
#   updated only from clean Branch A and Branch B completions. Applied to both transcription
#   and translation fallback paths.
# - 1.4.2 (2026-04-06): Startup EMA Warmup. After the hot worker loads, a 2-second synthetic
#   silence clip is transcribed automatically to seed _hot_ema_sps before the first real
#   request arrives. Without this, EMA=None at startup caused every concurrent request to
#   go to cold lane (Branch C), triggering CUDA OOM when multiple workers loaded the model
#   simultaneously. The warmup runs as a FastAPI startup event (asyncio, non-blocking) and
#   prints the measured sps so the operator can verify it on startup. Failure is logged but
#   non-fatal: the server still starts in uncalibrated mode.
# - 1.4.1 (2026-04-06): Cold-Lane Fallback to Hot Lane. When a cold lane subprocess exits
#   with a non-zero code (e.g. CUDA OOM caused by too many concurrent cold workers loading
#   the model simultaneously), the request is transparently retried on the hot lane instead
#   of returning HTTP 500. The fallback uses the same Branch-B queuing mechanism: adds
#   audio_dur to _hot_queue_audio_seconds before waiting so late-arriving requests see the
#   correct queue depth. By the time cold lane fails (~cold_start_time seconds in), the hot
#   lane has typically drained significantly, so the additional wait is short. Applied to
#   both /v1/audio/transcriptions and /v1/audio/translations.
# - 1.4.0 (2026-04-06): Smart Hot-Lane Routing. Three-branch router replaces the previous
#   binary hot/cold decision. Branch A: hot lane free → use immediately (unchanged). Branch B:
#   hot lane busy but estimated drain time < COLD_START_TIME_SECONDS * HOT_QUEUE_SAFETY_FACTOR
#   → queue for hot lane (asyncio.to_thread on model_lock.acquire, non-blocking). Branch C:
#   hot lane busy and drain estimate exceeds threshold → spawn cold lane as before. Unlike the
#   TTS server (which uses word count), the STT drain estimate uses audio duration in seconds
#   as the queue unit: _hot_queue_audio_seconds * ema_sps (server-seconds per audio-second,
#   EMA alpha=0.2). Audio duration is read from the WAV header via stdlib 'wave'; non-WAV
#   formats fall back to a byte-size estimate. Falls back to Branch C when EMA not yet
#   calibrated. New env vars: COLD_START_TIME_SECONDS (default 8.0s), HOT_QUEUE_SAFETY_FACTOR
#   (default 0.8). Routing stats exposed in GET /health under 'smart_routing'. Applied to
#   both /v1/audio/transcriptions and /v1/audio/translations.
# - 1.3.6 (2026-04-03): Added POST /v1/audio/translations endpoint (OpenAI spec). Transcribes audio in any language and returns English text. Uses Whisper task="translate" on both Hot and Cold lanes.
# - 1.3.5 (2026-04-03): VENV_PYTHON and WHISPER_SCRIPT auto-detect local venv/bin/ relative to BASE_DIR before falling back to hardcoded sphinx paths.
# - 1.3.4 (2026-04-03): MODEL_CACHE_DIR defaults to project-relative assets/models/whisper (no-sudo, no /opt). Mirrors coqui BASE_DIR pattern.
# - 1.3.3 (2026-04-03): VENV_PYTHON and WHISPER_SCRIPT now read from env vars (VENV_PYTHON, WHISPER_SCRIPT) with hardcoded values as fallback.
# - 1.3.2 (2026-04-03): Cold Lane refactored to asyncio.create_subprocess_exec + asyncio.wait_for. Adds COLD_LANE_TIMEOUT_SECONDS env var (default 300s). Prevents hung subprocesses from blocking indefinitely.
# - 1.3.1 (2026-04-03): Error sanitization: exceptions no longer leak internal paths or subprocess details in HTTP 500 responses. Full detail logged to stdout.
# - 1.3.0 (2026-04-03): Added GET /health and GET /v1/models endpoints. SERVER_VERSION constant introduced. hot_worker_error global tracks model load failures and is exposed in /health.
# - 1.2.3 (2026-02-27): Strict DEBUG control and shell command printing in slow lane.
# - 1.2.2 (2026-02-27): Wrapped model loading prints into DEBUG toggle.

import io
import wave
import time
import base64
import torch
import uvicorn
import whisper
import tempfile
import os
import shutil
import asyncio
import threading
import subprocess
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional

# Load .env from the project directory or its parent
_base = os.path.dirname(os.path.abspath(__file__))
for _env_path in [os.path.join(_base, ".env"), os.path.join(os.path.dirname(_base), ".env")]:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break

# -------------------------------
# 1. Global Config & Logging
# -------------------------------

SERVER_VERSION = "1.5.0"

# BASE_DIR is the directory containing this script. All local paths are relative to it,
# allowing no-sudo installation as any user (mirrors coqui-tts-local-server pattern).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# VENV_PYTHON and WHISPER_SCRIPT: resolution order:
#   1. VENV_PYTHON / WHISPER_SCRIPT env vars (explicit override via .env)
#   2. Auto-detected local venv at BASE_DIR/venv/bin/ (project-relative install)
#   3. Hardcoded fallback for the canonical sphinx installation (/usr/local/lib/whisper)
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

# Path to the persistent cold worker subprocess script.
COLD_WORKER_SCRIPT = os.path.join(BASE_DIR, "cold_worker.py")

# Maximum number of idle (loaded) cold workers kept alive in the pool.
# Each idle worker holds ~1.5 GB VRAM (fp16). Set to 0 to disable the pool (spawn-and-die behaviour).
COLD_POOL_SIZE = int(os.environ.get("COLD_POOL_SIZE", "2"))

# Seconds of inactivity before an idle pool worker exits on its own.
COLD_WORKER_IDLE_TIMEOUT = int(os.environ.get("COLD_WORKER_IDLE_TIMEOUT", "60"))

# MODEL_CACHE_DIR defaults to assets/models/whisper (project-relative, no root needed).
# Can be overridden via XDG_CACHE_HOME env var for installations that share a model cache.
MODEL_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.join(ASSETS_DIR, "models")),
    "whisper"
)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# WHISPER_FP16=1 (default): load model in fp16 on CUDA, saving ~66% VRAM vs fp32.
# Set WHISPER_FP16=0 to revert to fp32 (e.g. for debugging or non-CUDA systems).
WHISPER_FP16 = os.environ.get("WHISPER_FP16", "1").lower() in ("1", "true", "yes")


COLD_LANE_TIMEOUT_SECONDS = int(os.environ.get("COLD_LANE_TIMEOUT_SECONDS", "300"))

# Minimum free VRAM (GB) required to spawn a cold lane worker.
# If free VRAM is below this threshold at dispatch time, Branch C redirects to the hot lane
# queue instead of spawning a subprocess that will OOM mid-load (~8-10s wasted before failure).
# Default: 2.0 GB (whisper-medium ~1.5 GB model + loading overhead).
# Set to 0 to disable the VRAM check (not recommended on memory-constrained hardware).
MIN_COLD_VRAM_GB = float(os.environ.get("MIN_COLD_VRAM_GB", 4.0))

# --- Smart Hot-Lane Routing ---
# Initial hint for cold lane startup time. Used as the routing threshold until the auto-calibrated
# EMA (_cold_ema_start_stt) is seeded by the first successful cold lane completion. After that,
# _get_cold_start_time_stt() returns the live EMA instead. Set in .env only if cold lanes never
# run during startup and you want a specific initial bias.
# Default: 8.0s (typical cold start on a mid-range GPU, shorter than XTTS because Whisper is lighter).
COLD_START_TIME_SECONDS = float(os.environ.get("COLD_START_TIME_SECONDS", 8.0))

# Safety margin: only queue hot if drain_est < COLD_START_TIME * HOT_QUEUE_SAFETY_FACTOR.
# Default 0.8 = queue hot if we expect to finish at least 20% before a cold lane would load.
HOT_QUEUE_SAFETY_FACTOR = float(os.environ.get("HOT_QUEUE_SAFETY_FACTOR", 0.8))

# EMA smoothing factor for the hot-lane server-seconds-per-audio-second estimator.
_HOT_EMA_ALPHA = 0.2

# Strict DEBUG toggle: Only "true" enables extra logging
DEBUG_MODE = os.environ.get("DEBUG", "").lower() == "true"

def log_debug(message: str):
    if DEBUG_MODE:
        print(message)

model_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _cold_idle_pool
    _cold_idle_pool = asyncio.Queue()
    await _warmup_ema()
    yield
    # Shutdown: gracefully drain the idle pool
    while not _cold_idle_pool.empty():
        try:
            w = _cold_idle_pool.get_nowait()
            await w.shutdown()
        except asyncio.QueueEmpty:
            break


app = FastAPI(title="Whisper STT Server", version=SERVER_VERSION, lifespan=_lifespan)

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
    # Critical errors are always printed to stderr
    print(f"CRITICAL ERROR: Could not load model: {e}")
    whisper_model = None
    hot_worker_error = str(e)

class TranscriptionResponse(BaseModel):
    text: str

# -------------------------------
# 2b. Smart Routing Telemetry
# -------------------------------
# All fields accessed exclusively from the asyncio event loop thread — no threading.Lock needed.

# EMA of hot-lane wall-clock seconds per audio-second processed.
# None = not yet calibrated (no successful hot-lane transcription has completed yet).
_hot_ema_sps: Optional[float] = None

# Total audio seconds currently in the hot-lane pipeline (being transcribed + waiting).
# Updated before enqueue, decremented after completion so late-arriving requests see full depth.
_hot_queue_audio_seconds: float = 0.0

# EMA of successful cold lane completion times (seconds). None = not yet calibrated.
# Auto-calibrates COLD_START_TIME_SECONDS so operators don't need to measure it per-hardware.
# Updated after each successful Branch C completion; replaces COLD_START_TIME_SECONDS as the
# router threshold once seeded.
_cold_ema_start_stt: Optional[float] = None
_COLD_EMA_ALPHA_STT = 0.2

# Count of cold workers currently in the model-loading phase (spawned but not yet ready).
# Used by _has_vram_for_cold_lane() to reserve VRAM for workers whose memory hasn't
# appeared yet in torch.cuda.mem_get_info(). Decremented when the worker signals ready.
_cold_workers_in_flight: int = 0

# EMA of measured VRAM consumed per cold worker (GB). None = not yet measured.
# Measured right after each new worker signals ready (model fully loaded on GPU).
# Replaces the static MIN_COLD_VRAM_GB in _has_vram_for_cold_lane() once seeded.
# MIN_COLD_VRAM_GB becomes the fallback used only before the first cold run completes.
_cold_vram_ema_gb: Optional[float] = None
_COLD_VRAM_EMA_ALPHA = 0.3
_COLD_VRAM_SAFETY_FACTOR = 1.2  # 20% headroom above measured EMA

# EMA of inference-only time for pool workers (model already loaded, no spawn overhead).
# Used in health telemetry; routing always prefers an idle pool worker over queuing hot
# when the hot lane is busy.
_cold_inference_ema_stt: Optional[float] = None
_COLD_INFERENCE_EMA_ALPHA = 0.2

# asyncio.Queue of idle _ColdWorker instances. Initialised in _lifespan (requires event loop).
_cold_idle_pool: Optional[asyncio.Queue] = None


def _update_cold_vram_ema(vram_gb: float) -> None:
    """Update the EMA of VRAM consumed per cold worker."""
    global _cold_vram_ema_gb
    if vram_gb <= 0:
        return
    if _cold_vram_ema_gb is None:
        _cold_vram_ema_gb = vram_gb
    else:
        _cold_vram_ema_gb = _COLD_VRAM_EMA_ALPHA * vram_gb + (1.0 - _COLD_VRAM_EMA_ALPHA) * _cold_vram_ema_gb


def _update_cold_inference_ema_stt(elapsed: float) -> None:
    """Update the inference-only EMA for pool workers."""
    global _cold_inference_ema_stt
    if _cold_inference_ema_stt is None:
        _cold_inference_ema_stt = elapsed
    else:
        _cold_inference_ema_stt = _COLD_INFERENCE_EMA_ALPHA * elapsed + (1.0 - _COLD_INFERENCE_EMA_ALPHA) * _cold_inference_ema_stt


def _vram_per_cold_worker() -> float:
    """Return estimated VRAM needed per cold worker with safety margin.
    Uses live EMA once seeded, falls back to MIN_COLD_VRAM_GB."""
    if _cold_vram_ema_gb is not None:
        return _cold_vram_ema_gb * _COLD_VRAM_SAFETY_FACTOR
    return MIN_COLD_VRAM_GB


# ── Persistent cold worker class ──────────────────────────────────────────────

class _ColdWorker:
    """
    A persistent cold worker subprocess (cold_worker.py).

    The subprocess loads the Whisper model once on startup, then serves
    successive transcription requests via newline-delimited JSON on stdin/stdout.
    After each request the caller decides whether to return it to the idle pool
    or shut it down (if the pool is already full).
    """

    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None
        self.alive: bool = False

    async def spawn(self) -> bool:
        """
        Start the subprocess and wait for the {'ready': true} signal.

        Returns True if the worker loaded successfully and is ready to serve requests.
        Returns False (and cleans up) on any failure.
        """
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
            # Wait up to 90 s for the model to load and the ready signal
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
        """Send one request and return the Whisper result dict."""
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
        """Send an exit request and wait for the subprocess to terminate."""
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


async def _acquire_idle_worker() -> Optional[_ColdWorker]:
    """
    Return a live idle worker from the pool, or None if the pool is empty.
    Silently discards workers that died while idle.
    """
    if _cold_idle_pool is None:
        return None
    while not _cold_idle_pool.empty():
        try:
            worker = _cold_idle_pool.get_nowait()
        except asyncio.QueueEmpty:
            break
        if worker.is_alive():
            return worker
        # Died silently while sitting in the pool (idle timeout, OOM, etc.) — discard
    return None


async def _return_to_pool(worker: _ColdWorker) -> None:
    """Return a worker to the idle pool, or shut it down if the pool is already full."""
    if _cold_idle_pool is None or not worker.is_alive():
        await worker.shutdown()
        return
    if _cold_idle_pool.qsize() < COLD_POOL_SIZE:
        await _cold_idle_pool.put(worker)
    else:
        await worker.shutdown()


def _update_cold_ema_stt(elapsed: float) -> None:
    """Update the cold-lane time EMA after a successful cold lane transcription."""
    global _cold_ema_start_stt
    if _cold_ema_start_stt is None:
        _cold_ema_start_stt = elapsed
    else:
        _cold_ema_start_stt = _COLD_EMA_ALPHA_STT * elapsed + (1.0 - _COLD_EMA_ALPHA_STT) * _cold_ema_start_stt


def _get_cold_start_time_stt() -> float:
    """Return the auto-calibrated cold start time EMA, or COLD_START_TIME_SECONDS as fallback."""
    return _cold_ema_start_stt if _cold_ema_start_stt is not None else COLD_START_TIME_SECONDS


def _free_vram_gb() -> Optional[float]:
    """Return current free VRAM in GB, or None if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def _has_vram_for_cold_lane() -> bool:
    """
    Return True if there is enough free VRAM to load one more cold whisper worker.

    Uses auto-calibrated EMA of measured VRAM per cold worker (_cold_vram_ema_gb × 1.2)
    once seeded. Falls back to MIN_COLD_VRAM_GB before first cold run completes.
    Set MIN_COLD_VRAM_GB=0 to disable the check entirely.

    effective_free = gpu_free - (_cold_workers_in_flight × vram_per_worker)
    dispatch cold  if  effective_free >= vram_per_worker
    """
    if MIN_COLD_VRAM_GB <= 0:
        return True  # check disabled via env var
    free = _free_vram_gb()
    if free is None:
        return True  # CPU mode: no VRAM constraint
    needed = _vram_per_cold_worker()
    effective_free = free - (_cold_workers_in_flight * needed)
    return effective_free >= needed


def _get_audio_duration(audio_bytes: bytes) -> float:
    """
    Estimate audio duration in seconds from raw bytes.
    Tries to parse the WAV header via stdlib 'wave' (no extra deps).
    Falls back to a byte-size heuristic for non-WAV formats (MP3, M4A, etc.).
    """
    try:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        # Fallback: assume 16 kHz 16-bit mono PCM equivalent ≈ 32000 bytes/sec.
        # Conservative — better to slightly overestimate duration than underestimate.
        return len(audio_bytes) / 32000


def _update_hot_ema_stt(elapsed: float, audio_duration: float) -> None:
    """Update the server-seconds-per-audio-second EMA after a successful hot-lane transcription."""
    global _hot_ema_sps
    sps = elapsed / max(audio_duration, 0.1)  # guard against zero-length clips
    if _hot_ema_sps is None:
        _hot_ema_sps = sps
    else:
        _hot_ema_sps = _HOT_EMA_ALPHA * sps + (1.0 - _HOT_EMA_ALPHA) * _hot_ema_sps


def _should_queue_hot_stt(audio_duration: float) -> bool:
    """
    Return True if it is cheaper to wait for the hot lane than to start a cold lane.

    Decision formula:
        estimated_drain = _hot_queue_audio_seconds * _hot_ema_sps
        queue_hot  if  estimated_drain < COLD_START_TIME_SECONDS * HOT_QUEUE_SAFETY_FACTOR

    audio_duration of the incoming request is NOT added to the estimate: we ask
    "how long until the hot lane is free?" not "how long will my request take once it starts".
    Returns False when EMA is not yet calibrated.
    """
    if _hot_ema_sps is None:
        return False
    estimated_drain = _hot_queue_audio_seconds * _hot_ema_sps
    return estimated_drain < _get_cold_start_time_stt() * HOT_QUEUE_SAFETY_FACTOR


# -------------------------------
# 3. Transcription Functions
# -------------------------------

def _run_hot_locked(audio_bytes: bytes, language: Optional[str], prompt: Optional[str], temp: float, task: str = "transcribe") -> tuple:
    """
    Acquire model_lock, transcribe, release — all inside a single thread.

    Returns (result, processing_elapsed) where processing_elapsed is the time spent
    actually running the model (after acquiring the lock), excluding queue wait time.
    Callers must unpack: result, proc_elapsed = await asyncio.to_thread(_run_hot_locked, ...)

    This function is always called via asyncio.to_thread(). Keeping acquire and release
    in the same thread call guarantees the lock is released even if the calling coroutine
    is cancelled (e.g. client timeout). With a two-step pattern
    (await to_thread(lock.acquire) + await to_thread(work) + release in coroutine finally),
    a CancelledError between the acquire and the release leaves the lock permanently
    acquired, deadlocking the server.
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
    """
    Spawn a new _ColdWorker, wait for it to load the model, measure VRAM consumption,
    and return the ready worker. Manages _cold_workers_in_flight accounting.

    Raises RuntimeError if the worker fails to start.
    """
    global _cold_workers_in_flight
    v_before = _free_vram_gb() or 0.0
    _cold_workers_in_flight += 1
    worker = _ColdWorker()
    try:
        spawned = await worker.spawn()
        if not spawned:
            raise RuntimeError("Cold worker subprocess failed to start")
        # Model is now loaded: measure VRAM drop (only accurate when we are the sole spawner)
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
# 3b. Startup EMA Warmup
# -------------------------------

def _make_silence_wav(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal valid WAV file containing silence (zero-filled PCM 16-bit mono)."""
    buf = io.BytesIO()
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)      # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b'\x00' * n_frames * 2)
    return buf.getvalue()


async def _warmup_ema():
    """
    Transcribe a short synthetic silence clip through the hot lane at startup.
    Seeds _hot_ema_sps so Branch B is available from the very first concurrent request,
    preventing the EMA=None → cold-lane cascade that causes CUDA OOM under burst load.
    Failure is non-fatal: the server starts in uncalibrated mode with a log warning.
    """
    if whisper_model is None:
        return  # degraded mode — nothing to warm up

    print("WARMUP: Seeding EMA with synthetic silence clip...", flush=True)
    try:
        silence = _make_silence_wav(duration_seconds=2.0)
        audio_dur = _get_audio_duration(silence)
        t0 = time.monotonic()
        await asyncio.to_thread(run_transcription_fast_lane, silence, None, None, 0.0)
        elapsed = time.monotonic() - t0
        _update_hot_ema_stt(elapsed, audio_dur)
        print(f"WARMUP: EMA seeded. sps={_hot_ema_sps:.4f} (elapsed={elapsed:.2f}s for {audio_dur:.1f}s audio)", flush=True)
    except Exception as e:
        print(f"WARMUP: Failed to seed EMA ({e}). Server starting in uncalibrated mode.", flush=True)


# -------------------------------
# 4. Endpoints
# -------------------------------

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing. Returns the standard STT model IDs.
    The 'model' field in transcription requests is accepted for spec compliance
    but ignored internally — all requests are handled by the configured Whisper model.
    """
    return {
        "object": "list",
        "data": [
            {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "uttera-legacy"},
        ]
    }

@app.get("/health")
async def health_check():
    """Returns server liveness, hot worker status, and smart routing telemetry.
    'hot_worker_loaded': false and 'hot_worker_error' set means degraded mode.
    'smart_routing.ema_sps': null until the first hot-lane transcription completes.
    """
    _free = _free_vram_gb()
    routing_stats = {
        "cold_start_time_seconds": round(_get_cold_start_time_stt(), 2),
        "cold_start_calibrated": _cold_ema_start_stt is not None,
        "cold_ema_start_seconds": round(_cold_ema_start_stt, 2) if _cold_ema_start_stt is not None else None,
        "cold_start_configured_seconds": COLD_START_TIME_SECONDS,
        "safety_factor": HOT_QUEUE_SAFETY_FACTOR,
        "threshold_seconds": round(_get_cold_start_time_stt() * HOT_QUEUE_SAFETY_FACTOR, 2),
        "ema_sps": round(_hot_ema_sps, 4) if _hot_ema_sps is not None else None,
        "hot_queue_audio_seconds": round(_hot_queue_audio_seconds, 2),
        "hot_queue_drain_estimate_seconds": round(_hot_queue_audio_seconds * _hot_ema_sps, 2) if _hot_ema_sps else None,
        "vram_free_gb": round(_free, 2) if _free is not None else None,
        "min_cold_vram_gb": MIN_COLD_VRAM_GB,
        "cold_vram_ema_gb": round(_cold_vram_ema_gb, 3) if _cold_vram_ema_gb is not None else None,
        "cold_vram_per_worker_gb": round(_vram_per_cold_worker(), 3),
        "cold_workers_in_flight": _cold_workers_in_flight,
        "cold_pool_idle": _cold_idle_pool.qsize() if _cold_idle_pool is not None else 0,
        "cold_pool_size": COLD_POOL_SIZE,
        "cold_inference_ema_seconds": round(_cold_inference_ema_stt, 2) if _cold_inference_ema_stt is not None else None,
        "vram_sufficient_for_cold": _has_vram_for_cold_lane(),
    }
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "model": model_name,
        "hot_worker_loaded": whisper_model is not None,
        "hot_worker_error": hot_worker_error,
        "smart_routing": routing_stats,
    }

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
    try:
        contents = await file.read()
        audio_dur = _get_audio_duration(contents)
        global _hot_queue_audio_seconds
        route = "HOT-A"

        if model_lock.acquire(blocking=False):
            # ── Branch A: hot lane free → use immediately ────────────────────
            route = "HOT-A"
            log_debug(f"--- ROUTER: Hot lane free. audio={audio_dur:.1f}s ---")
            _hot_queue_audio_seconds += audio_dur
            t0 = time.monotonic()
            try:
                res = await asyncio.to_thread(run_transcription_fast_lane, contents, language, prompt, temperature)
                _update_hot_ema_stt(time.monotonic() - t0, audio_dur)
            finally:
                _hot_queue_audio_seconds -= audio_dur
                model_lock.release()

        else:
            idle_worker = await _acquire_idle_worker()
            if idle_worker is not None:
                # ── Branch 0: idle pool worker available ─────────────────────
                route = "COLD-POOL"
                log_debug(f"--- ROUTER: Pool worker available → COLD-POOL. audio={audio_dur:.1f}s ---")
                t_infer = time.monotonic()
                try:
                    res = await idle_worker.transcribe(contents, language, prompt, temperature, "transcribe")
                    _update_cold_inference_ema_stt(time.monotonic() - t_infer)
                    await _return_to_pool(idle_worker)
                except Exception as pool_err:
                    route = "COLD-POOL→HOT"
                    print(f"--- ROUTER: Pool worker failed ({pool_err}). Falling back to hot lane. audio={audio_dur:.1f}s ---", flush=True)
                    _hot_queue_audio_seconds += audio_dur
                    try:
                        res, _ = await asyncio.to_thread(_run_hot_locked, contents, language, prompt, temperature)
                    finally:
                        _hot_queue_audio_seconds -= audio_dur

            elif _should_queue_hot_stt(audio_dur):
                # ── Branch B: hot lane busy but cheaper to wait ──────────────
                route = "HOT-B"
                drain_est = _hot_queue_audio_seconds * _hot_ema_sps
                threshold = _get_cold_start_time_stt() * HOT_QUEUE_SAFETY_FACTOR
                log_debug(f"--- ROUTER: Queue hot. drain_est={drain_est:.1f}s < threshold={threshold:.1f}s. audio={audio_dur:.1f}s ---")
                _hot_queue_audio_seconds += audio_dur
                try:
                    res, proc_elapsed = await asyncio.to_thread(_run_hot_locked, contents, language, prompt, temperature)
                    _update_hot_ema_stt(proc_elapsed, audio_dur)
                finally:
                    _hot_queue_audio_seconds -= audio_dur

            elif _has_vram_for_cold_lane():
                # ── Branch C: spawn new cold worker ─────────────────────────
                route = "COLD"
                if DEBUG_MODE:
                    free_gb = _free_vram_gb()
                    drain_est = (_hot_queue_audio_seconds * _hot_ema_sps) if _hot_ema_sps else None
                    vram_str = f"{free_gb:.1f} GB free ({_cold_workers_in_flight} in-flight)" if free_gb is not None else "VRAM unknown"
                    if drain_est is not None:
                        print(f"--- ROUTER: Spawn cold. drain_est={drain_est:.1f}s ≥ threshold={_get_cold_start_time_stt() * HOT_QUEUE_SAFETY_FACTOR:.1f}s. {vram_str}. audio={audio_dur:.1f}s ---", flush=True)
                    else:
                        print(f"--- ROUTER: EMA uncalibrated → cold lane. {vram_str}. audio={audio_dur:.1f}s ---", flush=True)
                t_start = time.monotonic()
                try:
                    worker = await _spawn_cold_worker_with_vram()
                    t_infer = time.monotonic()
                    try:
                        res = await worker.transcribe(contents, language, prompt, temperature, "transcribe")
                        inference_elapsed = time.monotonic() - t_infer
                        _update_cold_ema_stt(time.monotonic() - t_start)
                        _update_cold_inference_ema_stt(inference_elapsed)
                        await _return_to_pool(worker)
                    except Exception as infer_err:
                        await worker.shutdown()
                        raise infer_err
                except Exception as cold_err:
                    route = "COLD→HOT"
                    print(f"--- ROUTER: Cold lane failed ({cold_err}). Falling back to hot lane. audio={audio_dur:.1f}s ---", flush=True)
                    _hot_queue_audio_seconds += audio_dur
                    try:
                        _fallback_delay = 1.0
                        for _attempt in range(3):
                            try:
                                res, _ = await asyncio.to_thread(_run_hot_locked, contents, language, prompt, temperature)
                                break
                            except Exception as hot_err:
                                _err_str = str(hot_err)
                                _is_cuda_transient = any(k in _err_str for k in (
                                    "cuDNN", "CUBLAS", "CUDA error", "out of memory"
                                ))
                                if _is_cuda_transient and _attempt < 2:
                                    print(f"--- FALLBACK HOT LANE: transient CUDA error (attempt {_attempt+1}/3), retrying in {_fallback_delay:.1f}s: {hot_err} ---", flush=True)
                                    await asyncio.sleep(_fallback_delay)
                                    _fallback_delay *= 2
                                else:
                                    raise
                    finally:
                        _hot_queue_audio_seconds -= audio_dur

            else:
                # ── Branch D: insufficient VRAM → queue hot ──────────────────
                route = "HOT-C"
                free_gb = _free_vram_gb()
                print(f"--- ROUTER: Insufficient VRAM ({free_gb:.1f} GB free) → queuing hot lane. audio={audio_dur:.1f}s ---", flush=True)
                _hot_queue_audio_seconds += audio_dur
                try:
                    res, proc_elapsed = await asyncio.to_thread(_run_hot_locked, contents, language, prompt, temperature)
                    _update_hot_ema_stt(proc_elapsed, audio_dur)
                finally:
                    _hot_queue_audio_seconds -= audio_dur

        data = res["text"] if response_format == "text" else res
        if response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=data, headers={"X-Route": route})
        return JSONResponse(content=data, headers={"X-Route": route})
    except Exception as e:
        print(f"ERROR in create_transcription: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Transcription failed. Check server logs.")
    finally:
        await file.close()

@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: float = Form(0.0)
):
    """OpenAI-compatible translation endpoint. Transcribes audio in any language and
    returns the result translated to English in a single Whisper pass (task='translate').
    No 'language' parameter — output is always English.
    """
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        contents = await file.read()
        audio_dur = _get_audio_duration(contents)
        global _hot_queue_audio_seconds
        route = "HOT-A"

        if model_lock.acquire(blocking=False):
            # ── Branch A: hot lane free → use immediately ────────────────────
            route = "HOT-A"
            log_debug(f"--- ROUTER: Hot lane free (translation). audio={audio_dur:.1f}s ---")
            _hot_queue_audio_seconds += audio_dur
            t0 = time.monotonic()
            try:
                res = await asyncio.to_thread(run_transcription_fast_lane, contents, None, prompt, temperature, "translate")
                _update_hot_ema_stt(time.monotonic() - t0, audio_dur)
            finally:
                _hot_queue_audio_seconds -= audio_dur
                model_lock.release()

        else:
            idle_worker = await _acquire_idle_worker()
            if idle_worker is not None:
                # ── Branch 0: idle pool worker available ─────────────────────
                route = "COLD-POOL"
                log_debug(f"--- ROUTER: Pool worker available → COLD-POOL (translation). audio={audio_dur:.1f}s ---")
                t_infer = time.monotonic()
                try:
                    res = await idle_worker.transcribe(contents, None, prompt, temperature, "translate")
                    _update_cold_inference_ema_stt(time.monotonic() - t_infer)
                    await _return_to_pool(idle_worker)
                except Exception as pool_err:
                    route = "COLD-POOL→HOT"
                    print(f"--- ROUTER: Pool worker failed ({pool_err}). Falling back to hot lane (translation). audio={audio_dur:.1f}s ---", flush=True)
                    _hot_queue_audio_seconds += audio_dur
                    try:
                        res, _ = await asyncio.to_thread(_run_hot_locked, contents, None, prompt, temperature, "translate")
                    finally:
                        _hot_queue_audio_seconds -= audio_dur

            elif _should_queue_hot_stt(audio_dur):
                # ── Branch B: hot lane busy but cheaper to wait ──────────────
                route = "HOT-B"
                drain_est = _hot_queue_audio_seconds * _hot_ema_sps
                threshold = _get_cold_start_time_stt() * HOT_QUEUE_SAFETY_FACTOR
                log_debug(f"--- ROUTER: Queue hot (translation). drain_est={drain_est:.1f}s < threshold={threshold:.1f}s. audio={audio_dur:.1f}s ---")
                _hot_queue_audio_seconds += audio_dur
                try:
                    res, proc_elapsed = await asyncio.to_thread(_run_hot_locked, contents, None, prompt, temperature, "translate")
                    _update_hot_ema_stt(proc_elapsed, audio_dur)
                finally:
                    _hot_queue_audio_seconds -= audio_dur

            elif _has_vram_for_cold_lane():
                # ── Branch C: spawn new cold worker ─────────────────────────
                route = "COLD"
                if DEBUG_MODE:
                    free_gb = _free_vram_gb()
                    drain_est = (_hot_queue_audio_seconds * _hot_ema_sps) if _hot_ema_sps else None
                    vram_str = f"{free_gb:.1f} GB free ({_cold_workers_in_flight} in-flight)" if free_gb is not None else "VRAM unknown"
                    if drain_est is not None:
                        print(f"--- ROUTER: Spawn cold (translation). drain_est={drain_est:.1f}s ≥ threshold={_get_cold_start_time_stt() * HOT_QUEUE_SAFETY_FACTOR:.1f}s. {vram_str}. audio={audio_dur:.1f}s ---", flush=True)
                    else:
                        print(f"--- ROUTER: EMA uncalibrated → cold lane (translation). {vram_str}. audio={audio_dur:.1f}s ---", flush=True)
                t_start = time.monotonic()
                try:
                    worker = await _spawn_cold_worker_with_vram("(translation)")
                    t_infer = time.monotonic()
                    try:
                        res = await worker.transcribe(contents, None, prompt, temperature, "translate")
                        inference_elapsed = time.monotonic() - t_infer
                        _update_cold_ema_stt(time.monotonic() - t_start)
                        _update_cold_inference_ema_stt(inference_elapsed)
                        await _return_to_pool(worker)
                    except Exception as infer_err:
                        await worker.shutdown()
                        raise infer_err
                except Exception as cold_err:
                    route = "COLD→HOT"
                    print(f"--- ROUTER: Cold lane failed ({cold_err}). Falling back to hot lane (translation). audio={audio_dur:.1f}s ---", flush=True)
                    _hot_queue_audio_seconds += audio_dur
                    try:
                        _fallback_delay = 1.0
                        for _attempt in range(3):
                            try:
                                res, _ = await asyncio.to_thread(_run_hot_locked, contents, None, prompt, temperature, "translate")
                                break
                            except Exception as hot_err:
                                _err_str = str(hot_err)
                                _is_cuda_transient = any(k in _err_str for k in (
                                    "cuDNN", "CUBLAS", "CUDA error", "out of memory"
                                ))
                                if _is_cuda_transient and _attempt < 2:
                                    print(f"--- FALLBACK HOT LANE: transient CUDA error (attempt {_attempt+1}/3), retrying in {_fallback_delay:.1f}s: {hot_err} ---", flush=True)
                                    await asyncio.sleep(_fallback_delay)
                                    _fallback_delay *= 2
                                else:
                                    raise
                    finally:
                        _hot_queue_audio_seconds -= audio_dur

            else:
                # ── Branch D: insufficient VRAM → queue hot ──────────────────
                route = "HOT-C"
                free_gb = _free_vram_gb()
                print(f"--- ROUTER: Insufficient VRAM ({free_gb:.1f} GB free) → queuing hot lane (translation). audio={audio_dur:.1f}s ---", flush=True)
                _hot_queue_audio_seconds += audio_dur
                try:
                    res, proc_elapsed = await asyncio.to_thread(_run_hot_locked, contents, None, prompt, temperature, "translate")
                    _update_hot_ema_stt(proc_elapsed, audio_dur)
                finally:
                    _hot_queue_audio_seconds -= audio_dur

        data = res["text"] if response_format == "text" else res
        if response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=data, headers={"X-Route": route})
        return JSONResponse(content=data, headers={"X-Route": route})
    except Exception as e:
        print(f"ERROR in create_translation: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Translation failed. Check server logs.")
    finally:
        await file.close()

if __name__ == "__main__":
    uvicorn.run("main_stt:app", host="0.0.0.0", port=5000, log_level="info")
