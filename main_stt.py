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
# Version: 1.3.6
# Maintainer: Uttera, Hugo L. Espuny
# Description: High-performance STT server with GPU acceleration and concurrency.
#
# CHANGELOG:
# - 1.3.6 (2026-04-03): Added POST /v1/audio/translations endpoint (OpenAI spec). Transcribes audio in any language and returns English text. Uses Whisper task="translate" on both Hot and Cold lanes.
# - 1.3.5 (2026-04-03): VENV_PYTHON and WHISPER_SCRIPT auto-detect local venv/bin/ relative to BASE_DIR before falling back to hardcoded sphinx paths.
# - 1.3.4 (2026-04-03): MODEL_CACHE_DIR defaults to project-relative assets/models/whisper (no-sudo, no /opt). Mirrors coqui BASE_DIR pattern.
# - 1.3.3 (2026-04-03): VENV_PYTHON and WHISPER_SCRIPT now read from env vars (VENV_PYTHON, WHISPER_SCRIPT) with hardcoded values as fallback.
# - 1.3.2 (2026-04-03): Cold Lane refactored to asyncio.create_subprocess_exec + asyncio.wait_for. Adds COLD_LANE_TIMEOUT_SECONDS env var (default 300s). Prevents hung subprocesses from blocking indefinitely.
# - 1.3.1 (2026-04-03): Error sanitization: exceptions no longer leak internal paths or subprocess details in HTTP 500 responses. Full detail logged to stdout.
# - 1.3.0 (2026-04-03): Added GET /health and GET /v1/models endpoints. SERVER_VERSION constant introduced. hot_worker_error global tracks model load failures and is exposed in /health.
# - 1.2.3 (2026-02-27): Strict DEBUG control and shell command printing in slow lane.
# - 1.2.2 (2026-02-27): Wrapped model loading prints into DEBUG toggle.

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
from pydantic import BaseModel
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

SERVER_VERSION = "1.3.6"

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

# MODEL_CACHE_DIR defaults to assets/models/whisper (project-relative, no root needed).
# Can be overridden via XDG_CACHE_HOME env var for installations that share a model cache.
MODEL_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.join(ASSETS_DIR, "models")),
    "whisper"
)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

COLD_LANE_TIMEOUT_SECONDS = int(os.environ.get("COLD_LANE_TIMEOUT_SECONDS", "300"))

# Strict DEBUG toggle: Only "true" enables extra logging
DEBUG_MODE = os.environ.get("DEBUG", "").lower() == "true"

def log_debug(message: str):
    if DEBUG_MODE:
        print(message)

model_lock = threading.Lock()
app = FastAPI(title="Whisper STT Server", version=SERVER_VERSION)

# -------------------------------
# 2. Model Loading
# -------------------------------
model_name = os.environ.get("WHISPER_MODEL", "medium")
hot_worker_error: Optional[str] = None

log_debug(f"Loading HOT WORKER model '{model_name}' into memory...")
try:
    whisper_model = whisper.load_model(model_name, download_root=MODEL_CACHE_DIR)
    log_debug(f"Model '{model_name}' loaded successfully.")
except Exception as e:
    # Critical errors are always printed to stderr
    print(f"CRITICAL ERROR: Could not load model: {e}")
    whisper_model = None
    hot_worker_error = str(e)

class TranscriptionResponse(BaseModel):
    text: str

# -------------------------------
# 3. Transcription Functions
# -------------------------------

def run_transcription_fast_lane(audio_bytes: bytes, language: Optional[str], prompt: Optional[str], temp: float, task: str = "transcribe") -> dict:
    log_debug(f"--- MAIN LANE: Using hot worker (GPU), task={task} ---")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as t:
            t.write(audio_bytes)
            temp_path = t.name
        return whisper_model.transcribe(temp_path, language=language, initial_prompt=prompt, temperature=temp, task=task)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

async def run_transcription_slow_lane(audio_bytes: bytes, language: Optional[str], prompt: Optional[str], temp: float, task: str = "transcribe") -> dict:
    log_debug(f"--- CHILD LANE: Spawning new cold worker, task={task}... ---")
    t_audio = None
    r_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=MODEL_CACHE_DIR) as t:
            t.write(audio_bytes)
            t_audio = t.name

        sub_env = os.environ.copy()
        sub_env["WHISPER_CACHE_DIR"] = MODEL_CACHE_DIR

        cmd = [
            VENV_PYTHON, WHISPER_SCRIPT, t_audio,
            "--model", model_name,
            "--output_format", "json",
            "--output_dir", MODEL_CACHE_DIR,
            "--temperature", str(temp)
        ]
        if language: cmd.extend(["--language", language])
        if prompt: cmd.extend(["--initial_prompt", prompt])
        if task == "translate": cmd.extend(["--task", "translate"])

        # New in 1.2.3: Print the exact shell command if DEBUG=true
        log_debug(f"DEBUG EXEC: {' '.join(cmd)}")

        # v1.3.2: asyncio.create_subprocess_exec + wait_for timeout replaces blocking subprocess.run().
        # This prevents a hung subprocess (OOM, driver crash) from blocking the server indefinitely.
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sub_env
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=COLD_LANE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            raise RuntimeError(f"Cold Lane subprocess timed out after {COLD_LANE_TIMEOUT_SECONDS}s")

        if process.returncode != 0:
            # v1.3.1: Full detail is logged but NOT sent to the client to avoid path/info leakage.
            print(f"COLD LANE ERROR (exit {process.returncode}): {stderr.decode()}", flush=True)
            raise RuntimeError(f"Cold Lane transcription failed (subprocess exited {process.returncode})")

        if DEBUG_MODE:
            print(f"COLD LANE STDOUT: {stdout.decode()}", flush=True)

        f_base, _ = os.path.splitext(os.path.basename(t_audio))
        r_path = os.path.join(MODEL_CACHE_DIR, f_base + ".json")

        with open(r_path, 'r') as f:
            data = json.load(f)
        return data
    finally:
        if t_audio and os.path.exists(t_audio): os.remove(t_audio)
        if r_path and os.path.exists(r_path): os.remove(r_path)
        log_debug(f"--- CHILD LANE: Cleanup complete. ---")

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
    """Returns server liveness and hot worker status. Suitable for proxies and Docker healthchecks.
    'hot_worker_loaded': false and 'hot_worker_error' set means server is running in degraded mode
    (all requests will fail — cold lane requires the venv python and whisper CLI).
    """
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "model": model_name,
        "hot_worker_loaded": whisper_model is not None,
        "hot_worker_error": hot_worker_error
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
        if model_lock.acquire(blocking=False):
            log_debug("--- ROUTER: Fast lane is free. Sending request. ---")
            try:
                res = await asyncio.to_thread(run_transcription_fast_lane, contents, language, prompt, temperature)
            finally:
                model_lock.release()
        else:
            log_debug("--- ROUTER: Main lane is busy. Rerouting to child lane. ---")
            res = await run_transcription_slow_lane(contents, language, prompt, temperature)

        return res["text"] if response_format == "text" else res
    except Exception as e:
        # v1.3.1: Log full detail but return a generic message to avoid leaking internal paths.
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
        if model_lock.acquire(blocking=False):
            log_debug("--- ROUTER: Fast lane is free. Sending translation request. ---")
            try:
                res = await asyncio.to_thread(run_transcription_fast_lane, contents, None, prompt, temperature, "translate")
            finally:
                model_lock.release()
        else:
            log_debug("--- ROUTER: Main lane is busy. Rerouting translation to child lane. ---")
            res = await run_transcription_slow_lane(contents, None, prompt, temperature, "translate")

        return res["text"] if response_format == "text" else res
    except Exception as e:
        print(f"ERROR in create_translation: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Translation failed. Check server logs.")
    finally:
        await file.close()

if __name__ == "__main__":
    uvicorn.run("main_stt:app", host="0.0.0.0", port=5000, log_level="info")
