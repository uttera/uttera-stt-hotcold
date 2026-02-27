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
# Version: 1.2.3
# Maintainer: Uttera <>, Hugo Espuny <>
# Description: High-performance STT server with GPU acceleration and concurrency.
#
# CHANGELOG:
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
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from pydantic import BaseModel
from typing import Optional

# -------------------------------
# 1. Global Config & Logging
# -------------------------------
VENV_PYTHON = "/usr/local/lib/whisper/bin/python"
WHISPER_SCRIPT = "/usr/local/lib/whisper/bin/whisper"
XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME", "/opt/ai/models/speech")
MODEL_CACHE_DIR = os.path.join(XDG_CACHE_HOME, "whisper")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Strict DEBUG toggle: Only "true" enables extra logging
DEBUG_MODE = os.environ.get("DEBUG", "").lower() == "true"

def log_debug(message: str):
    if DEBUG_MODE:
        print(message)

model_lock = threading.Lock()
app = FastAPI(title="Whisper STT Server", version="1.2.3")

# -------------------------------
# 2. Model Loading
# -------------------------------
model_name = os.environ.get("WHISPER_MODEL", "medium")
log_debug(f"Loading HOT WORKER model '{model_name}' into memory...")
try:
    whisper_model = whisper.load_model(model_name, download_root=MODEL_CACHE_DIR)
    log_debug(f"Model '{model_name}' loaded successfully.")
except Exception as e:
    # Critical errors are always printed to stderr
    print(f"CRITICAL ERROR: Could not load model: {e}")
    whisper_model = None

class TranscriptionResponse(BaseModel):
    text: str

# -------------------------------
# 3. Transcription Functions
# -------------------------------

def run_transcription_fast_lane(audio_bytes: bytes, language: Optional[str], prompt: Optional[str], temp: float) -> dict:
    log_debug("--- MAIN LANE: Using hot worker (GPU) ---")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as t:
            t.write(audio_bytes)
            temp_path = t.name
        return whisper_model.transcribe(temp_path, language=language, initial_prompt=prompt, temperature=temp)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def run_transcription_slow_lane(audio_bytes: bytes, language: Optional[str], prompt: Optional[str], temp: float) -> dict:
    log_debug(f"--- CHILD LANE: Spawning new cold worker... ---")
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
        
        # New in 1.2.3: Print the exact shell command if DEBUG=true
        log_debug(f"DEBUG EXEC: {' '.join(cmd)}")

        subprocess.run(cmd, check=True, capture_output=True, text=True, env=sub_env)
        
        f_base, _ = os.path.splitext(os.path.basename(t_audio))
        r_path = os.path.join(MODEL_CACHE_DIR, f_base + ".json")
        
        with open(r_path, 'r') as f:
            data = json.load(f)
        return data
    finally:
        if t_audio and os.path.exists(t_audio): os.remove(t_audio)
        if r_path and os.path.exists(r_path): os.remove(r_path)
        log_debug(f"--- CHILD LANE: Cleanup complete. ---")

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
            res = await asyncio.to_thread(run_transcription_slow_lane, contents, language, prompt, temperature)
        
        return res["text"] if response_format == "text" else res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")
