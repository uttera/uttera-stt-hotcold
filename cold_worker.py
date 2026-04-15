#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025-2026 Hugo L. Espuny
#
# Part of the Uttera voice stack (https://uttera.ai).
# See LICENSE and NOTICE at the repository root.
"""
cold_worker.py — Persistent Whisper subprocess for the cold worker pool.

This script is spawned by main_stt.py and stays alive to serve multiple transcription
requests without paying the model-load cost each time (~18-20 s on whisper-medium).

Protocol (newline-delimited JSON on stdin/stdout):

  Startup
    Worker writes {"ready": true} when the model is fully loaded and ready.

  Request (parent → worker, one JSON line)
    {"audio_b64": "<base64>", "language": "es|null", "prompt": "...|null",
     "temperature": 0.0, "task": "transcribe|translate"}

  Response (worker → parent, one JSON line)
    {"result": {...}}   — successful transcription (Whisper dict)
    {"error": "..."}    — transcription failed; worker is still alive

  Shutdown
    Parent writes {"exit": true} — clean shutdown
    Parent closes stdin (EOF)   — clean shutdown
    Idle timeout expires        — worker writes {"exit": "idle_timeout"} and exits

Environment variables (set by main_stt.py at spawn time):
  WHISPER_MODEL            — model name (default: medium)
  WHISPER_CACHE_DIR        — path to model cache directory
  WHISPER_FP16             — "1" for fp16+LN-fp32 on CUDA, "0" for fp32
  COLD_WORKER_IDLE_TIMEOUT — seconds before idle exit (default: 60)
"""

import sys
import json
import base64
import tempfile
import os
import select

import torch
import whisper

# ── Config from env ────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("WHISPER_MODEL", "medium")
MODEL_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR", "assets/models/whisper")
_fp16_env = os.environ.get("WHISPER_FP16", "1").lower()
WHISPER_FP16 = _fp16_env in ("1", "true", "yes")
IDLE_TIMEOUT = float(os.environ.get("COLD_WORKER_IDLE_TIMEOUT", "60"))


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ── Model loading ──────────────────────────────────────────────────────────────
_cuda = torch.cuda.is_available()
_use_fp16 = _cuda and WHISPER_FP16

model = whisper.load_model(
    MODEL_NAME,
    device="cpu" if _use_fp16 else None,
    download_root=MODEL_CACHE_DIR,
)

if _use_fp16:
    model = model.half()
    for _m in model.modules():
        if isinstance(_m, torch.nn.LayerNorm):
            _m.float()
    model = model.cuda()

# Signal ready to the parent
_write({"ready": True})

# ── Request loop ───────────────────────────────────────────────────────────────
while True:
    # Wait for input with idle timeout
    ready, _, _ = select.select([sys.stdin], [], [], IDLE_TIMEOUT)
    if not ready:
        _write({"exit": "idle_timeout"})
        break

    line = sys.stdin.readline()
    if not line:  # EOF — parent closed stdin
        break

    req = json.loads(line.strip())
    if req.get("exit"):
        break

    audio_bytes = base64.b64decode(req["audio_b64"])
    language = req.get("language") or None
    prompt = req.get("prompt") or None
    temperature = float(req.get("temperature", 0.0))
    task = req.get("task", "transcribe")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name

        use_fp16 = torch.cuda.is_available()
        result = model.transcribe(
            temp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            task=task,
            fp16=use_fp16,
        )
        _write({"result": result})
    except Exception as exc:
        _write({"error": str(exc)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
