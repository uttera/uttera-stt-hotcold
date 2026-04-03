#!/bin/bash
# Uttera STT Asset Provisioning Script
# Version: 1.0.0
# Description: Provisions Whisper models into the local assets directory.

set -e

echo "🦾 Uttera - Provisioning STT Infrastructure Assets..."

# 1. Path Discovery
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ASSETS_DIR="$SCRIPT_DIR/assets"
MODELS_DIR="$ASSETS_DIR/models"
CACHE_DIR="$ASSETS_DIR/cache"

# 2. Create Directory Structure
echo "[*] Creating local directory structure in $ASSETS_DIR..."
mkdir -p "$MODELS_DIR/whisper"
mkdir -p "$CACHE_DIR"

# 3. Environment Variables (directing Whisper to local assets)
export XDG_CACHE_HOME="$MODELS_DIR"

# 4. Use venv python/whisper if available
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
    WHISPER_BIN="$SCRIPT_DIR/venv/bin/whisper"
else
    PYTHON_BIN="python3"
    WHISPER_BIN="whisper"
fi

# 5. Model Provisioning (Idempotent)
# Downloads the default model (medium) by triggering a dummy transcription.
# Whisper downloads on first use; subsequent runs are instant.
echo "[*] Provisioning Whisper model (medium)..."
MODEL_NAME="${WHISPER_MODEL:-medium}"

$PYTHON_BIN - <<EOF
import whisper, os
cache = os.path.join("$MODELS_DIR", "whisper")
model = "$MODEL_NAME"
print(f"    -> Downloading/verifying model '{model}' to {cache}...")
whisper.load_model(model, download_root=cache)
print("    [✓] Model ready.")
EOF

echo "✅ STT Asset Provisioning Complete."
