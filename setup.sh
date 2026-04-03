#!/bin/bash
# Uttera STT Unified Setup Script
# Version: 1.0.0
# Description: Orchestrates Python environment setup. Uses Python 3.12 for dependency
#              compatibility. Falls back to python3 if python3.12 is not found.

set -e

echo "🦾 Uttera - Starting STT Installation Protocol..."

# 1. Python Virtual Environment
# Python 3.12 is preferred for wheel availability with torch/openai-whisper.
echo "[*] Initializing Python Virtual Environment..."
if command -v python3.12 &>/dev/null; then
    PYTHON_BIN=python3.12
    echo "    -> Using python3.12"
else
    PYTHON_BIN=python3
    echo "    [!] python3.12 not found, falling back to $(python3 --version). Some dependencies may fail."
fi
$PYTHON_BIN -m venv venv
source venv/bin/activate

# 2. Build-time dependencies
echo "[*] Installing build-time dependencies..."
pip install --upgrade pip setuptools wheel

# 3. Core Dependencies
echo "[*] Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Trigger Asset Provisioning
if [ -f "./setup_assets.sh" ]; then
    echo "[*] Python environment ready. Handing over to setup_assets.sh..."
    chmod +x setup_assets.sh
    ./setup_assets.sh
else
    echo "[!] ERROR: setup_assets.sh not found."
    exit 1
fi

echo "✅ All systems operational."
