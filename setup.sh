#!/bin/bash
# Uttera STT Unified Setup Script
# Version: 1.6.7
# Description: Orchestrates Python environment setup and asset provisioning.

set -e

echo "Whisper STT Server — Starting Installation..."

# 1. Python Virtual Environment
# Uses system default python3 (3.12+ recommended).
echo "[*] Initializing Python Virtual Environment..."
PYTHON_BIN=python3
echo "    -> Using $($PYTHON_BIN --version)"
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

echo "All systems operational."
