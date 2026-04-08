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
# Version: 1.6.0
# Maintainer: Uttera, Hugo L. Espuny
# Description: High-performance STT server with GPU acceleration and concurrency.
#
# CHANGELOG:
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
import wave
import time
import base64
import torch
import uvicorn
import whisper
import tempfile
import os
import asyncio
import threading
import json
import dataclasses
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
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

SERVER_VERSION = "1.6.2"

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
    global _work_queue, _cold_spawn_lock, _pool_worker_tasks
    _work_queue = asyncio.Queue()
    _cold_spawn_lock = asyncio.Lock()
    _pool_worker_tasks = set()

    await _warmup_ema()
    await _warmup_cold_ema()

    hot_task = asyncio.create_task(_hot_worker_loop())
    manager_task = asyncio.create_task(_cold_pool_manager())

    yield

    # Shutdown: cancel all worker tasks
    hot_task.cancel()
    manager_task.cancel()
    for task in list(_pool_worker_tasks):
        task.cancel()
    await asyncio.gather(hot_task, manager_task, *list(_pool_worker_tasks), return_exceptions=True)

    # Drain any remaining queued items (cancel their futures)
    while not _work_queue.empty():
        try:
            item = _work_queue.get_nowait()
            if not item.future.done():
                item.future.cancel()
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


async def _pool_worker_loop(worker: _ColdWorker) -> None:
    """
    Persistent asyncio Task for one cold pool worker.
    Consumes _WorkItem entries from the same _work_queue as the hot worker.
    Exits on idle timeout or worker failure; pool manager spawns replacements if needed.
    """
    global _work_queue_audio_seconds
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    _work_queue.get(), timeout=float(COLD_WORKER_IDLE_TIMEOUT)
                )
            except asyncio.TimeoutError:
                print(f"--- POOL WORKER: idle timeout ({COLD_WORKER_IDLE_TIMEOUT}s), exiting ---", flush=True)
                break
            except asyncio.CancelledError:
                break

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

        if active + loading >= target:
            continue

        if _cold_spawn_lock is None or _cold_spawn_lock.locked():
            continue

        if not _has_vram_for_cold_lane():
            continue

        drain_s = _work_queue_audio_seconds * (_hot_ema_sps or 0)
        print(
            f"--- POOL MGR: target={target} cold workers (active={active}, loading={loading}) "
            f"| queue={_work_queue_audio_seconds:.1f}s audio ({drain_s:.1f}s drain) → spawning ---",
            flush=True
        )
        try:
            async with _cold_spawn_lock:
                worker = await _spawn_cold_worker_with_vram()
            task = asyncio.create_task(_pool_worker_loop(worker))
            _pool_worker_tasks.add(task)
            task.add_done_callback(_pool_worker_tasks.discard)
            print(
                f"--- POOL MGR: pool worker ready, total_active={len(_pool_worker_tasks)} ---",
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
            {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "uttera-legacy"},
        ]
    }

@app.get("/health")
async def health_check():
    _free = _free_vram_gb()
    optimal = _optimal_cold_workers()
    drain = round(_work_queue_audio_seconds * _hot_ema_sps, 2) if _hot_ema_sps else None
    routing_stats = {
        "cold_start_time_seconds": round(_get_cold_start_time_stt(), 2),
        "cold_start_calibrated": _cold_ema_start_stt is not None,
        "cold_ema_start_seconds": round(_cold_ema_start_stt, 2) if _cold_ema_start_stt is not None else None,
        "cold_start_configured_seconds": COLD_START_TIME_SECONDS,
        "safety_factor": HOT_QUEUE_SAFETY_FACTOR,
        "ema_sps": round(_hot_ema_sps, 4) if _hot_ema_sps is not None else None,
        "work_queue_depth": _work_queue.qsize() if _work_queue is not None else 0,
        "work_queue_audio_seconds": round(_work_queue_audio_seconds, 2),
        "work_queue_drain_estimate_seconds": drain,
        "vram_free_gb": round(_free, 2) if _free is not None else None,
        "min_cold_vram_gb": MIN_COLD_VRAM_GB,
        "cold_vram_ema_gb": round(_cold_vram_ema_gb, 3) if _cold_vram_ema_gb is not None else None,
        "cold_vram_per_worker_gb": round(_vram_per_cold_worker(), 3),
        "cold_workers_in_flight": _cold_workers_in_flight,
        "pool_workers_active": len(_pool_worker_tasks),
        "pool_workers_optimal": optimal,
        "pool_size_cap": COLD_POOL_SIZE,
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
    """OpenAI-compatible translation endpoint. Returns English text."""
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        contents = await file.read()
        audio_dur = _get_audio_duration(contents)

        global _work_queue_audio_seconds
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _WorkItem(
            audio_bytes=contents,
            language=None,
            prompt=prompt,
            temperature=temperature,
            task="translate",
            audio_dur=audio_dur,
            future=future,
        )
        _work_queue_audio_seconds += audio_dur
        await _work_queue.put(item)

        res = await future
        route = item.route

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
