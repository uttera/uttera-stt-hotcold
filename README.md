# whisper-stt-local-server

<p align="center">
  <img src="docs/img/social-preview.jpg" alt="Whisper STT Local Server" width="800">
</p>

High-performance Whisper STT API server with a hybrid "Hot/Cold" worker architecture.

**Ideal for locally running installations of agents like OpenClaw or Open-WebUI, where the media should not leave the private local domain.**

## 🚀 Key Features

- **Hybrid Concurrency:**
  - **Hot Worker:** Keeps a Whisper model resident in VRAM for sub-second (~0.2s) inference.
  - **Cold Workers:** Spawns on-demand subprocesses when the GPU is busy, ensuring long audio files don't block quick voice commands.
- **GPU Accelerated:** Native support for NVIDIA CUDA, ensuring ultra-fast inference.
- **OpenAI Compatible:** Implements the standard OpenAI STT API (`/v1/audio/transcriptions`, `/v1/audio/translations`). Includes `GET /v1/models` for client autodiscovery.
- **Translation:** `POST /v1/audio/translations` transcribes audio in any language and returns English text in a single Whisper pass.
- **Multilingual:** Supports all languages covered by Whisper (99 languages). Auto-detects language if not specified.
- **Health Endpoint:** `GET /health` exposes server version, model name, and hot worker status for proxies and Docker healthchecks.
- **Privacy First:** 100% local execution. Your audio never leaves your infrastructure.

## 📦 Installation & Setup

### 1. Prerequisites (Debian/Ubuntu)
Install the following system dependencies first:
```bash
sudo apt update && sudo apt install -y ffmpeg python3.12 python3.12-venv
```

> **Python version:** `setup.sh` prefers **Python 3.12**. On systems where Python 3.12 is not the default (e.g. Ubuntu 24.10 with Python 3.14), the package above installs it alongside the system Python. The script falls back to `python3` with a warning if 3.12 is not found.

### 2. Unified Installation
```bash
git clone https://github.com/fakehec/whisper-stt-local-server.git
cd whisper-stt-local-server
chmod +x setup.sh
./setup.sh
```

`setup.sh` creates the virtual environment, installs all dependencies, and downloads the configured Whisper model into `assets/models/`. It is safe to re-run.

### 3. User Permissions & Hardware Acceleration
To run the server without `sudo` privileges and enable GPU acceleration, the user must belong to the `video` and `render` groups:
```bash
sudo usermod -aG video $USER
sudo usermod -aG render $USER
```
*Note: Restart your session for changes to take effect.*

### 4. Network Permissions
The server listens on port `5000` by default. Ensure the user has permissions to open sockets on this port (standard for ports >1024).

## 📡 API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server liveness, version, and hot worker status. |
| `GET` | `/v1/models` | OpenAI-compatible model list (`whisper-1`). |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text (Hot or Cold Lane). |
| `POST` | `/v1/audio/translations` | Transcribe audio in any language → English (Hot or Cold Lane). |

## 🛠 Execution

The server uses direct **Uvicorn** execution for maximum ASGI performance.

### Manual Execution (Console)
```bash
source venv/bin/activate

# Localhost only
uvicorn main_stt:app --host 127.0.0.1 --port 5000

# Expose to local network
uvicorn main_stt:app --host 0.0.0.0 --port 5000
```

### ⚙️ Environment Variables & .env

Copy `.env.example` to `.env` and adjust as needed. All variables are optional.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WHISPER_MODEL` | `medium` | Model to load: `tiny`, `base`, `small`, `medium`, `large`. |
| `COLD_LANE_TIMEOUT_SECONDS` | `300` | Max seconds to wait for a Cold Lane subprocess before HTTP 500. |
| `XDG_CACHE_HOME` | `assets/models` | Override model cache directory (e.g. for shared installs). |
| `VENV_PYTHON` | *(auto-detected)* | Path to venv Python. Auto-detected from `venv/bin/python`. |
| `WHISPER_SCRIPT` | *(auto-detected)* | Path to whisper CLI. Auto-detected from `venv/bin/whisper`. |
| `DEBUG` | `false` | Set to `true` to enable worker routing and subprocess traces. |

### User Service (systemd --user)
1. Create directory if it doesn't exist: `mkdir -p ~/.config/systemd/user`
2. Create: `~/.config/systemd/user/whisper-stt.service`
3. Configuration (environment variables are loaded from your `.env` file):

```ini
[Unit]
Description=Whisper STT Local Server
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/whisper-stt-local-server
ExecStart=%h/whisper-stt-local-server/venv/bin/uvicorn main_stt:app --host 127.0.0.1 --port 5000
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

4. Enable and start:
```bash
systemctl --user daemon-reload
systemctl --user enable --now whisper-stt.service
```

## 🔧 Troubleshooting

### Cold Lane fails with `No such file or directory`
If concurrent requests return HTTP 500 with a path error, the Cold Lane cannot find the Whisper CLI or Python binary. The server auto-detects `venv/bin/python` and `venv/bin/whisper` relative to the project directory. If running from a non-standard location, set the paths explicitly in `.env`:
```env
VENV_PYTHON=/absolute/path/to/venv/bin/python
WHISPER_SCRIPT=/absolute/path/to/venv/bin/whisper
```

### `PermissionError` on startup
The server defaults to `assets/models/whisper` inside the project directory — no root required. If you see a permission error on a path like `/opt/...`, an old `XDG_CACHE_HOME` env var is being inherited from the shell. Either unset it or override it in `.env`:
```env
XDG_CACHE_HOME=assets/models
```

### Cold Lane subprocess times out
If transcription of long audio hangs and eventually returns HTTP 500, increase the timeout in `.env`:
```env
COLD_LANE_TIMEOUT_SECONDS=600
```

## 🐳 Docker

### Host Prerequisites (one-time setup)

Before running `docker compose up` for the first time, the host machine requires two one-time configuration steps to enable GPU passthrough via the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) CDI mode.

> These steps are required because Docker's default legacy GPU mode relies on BPF cgroup device filters, which are not available in cgroup v2 environments (Ubuntu 22.04+). CDI solves this cleanly.

**1. Add the NVIDIA package repository:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

**2. Install the toolkit:**
```bash
sudo apt update && sudo apt install -y nvidia-container-toolkit
```

**3. Generate the CDI spec** (exposes the GPU to containers via a stable device descriptor):
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

**4. Enable CDI in the Docker daemon:**
```bash
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "features": {
    "cdi": true
  }
}
EOF
sudo systemctl restart docker
```

**5. Verify it works:**
```bash
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

> **Note:** Step 3 must be re-run if the NVIDIA driver is updated (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`).

### Running with Docker Compose

```bash
# Build and start
docker compose up -d

# Check server is ready
curl http://localhost:5000/health

# View logs
docker compose logs -f

# Stop
docker compose down
```

The model is persisted in `assets/models/whisper/` (host volume), so it only downloads once.

## 🔒 Security & Network Note
By default, the server binds to **`127.0.0.1`** on port **`5000`**.
- To allow external network access, change `--host` to `0.0.0.0`.
- **WARNING**: This API **does not have authentication**. Exposing it to the network via `0.0.0.0` represents a security risk. Ensure the server is protected by a firewall or operating within a secure VPN/local network.

## 📊 Performance Benchmarks (NVIDIA RTX 5090)

| Task | Hot Lane | Cold Lane |
| :--- | :--- | :--- |
| Short command (2s audio) | **~0.2s** | ~3s |
| Long audio (30s) | **~0.7s** | ~5s |

## 🛡 License

GNU GPL v3. Maintainers: Hugo L. Espuny & Uttera

## ☕ Support

If this project is useful to you, consider supporting its development:

- **Bitcoin (BTC):** `38jJyMomtUqhCjuNJ9VxKpgEyMyx37Zqix`
- **Monero (XMR):** `82bbUZdkMXUPAma4ioTuZNcJgTh8YTv4XNUwPy6T28kYJWCfeGgV79AZb7amCszFXeBaa5u595cQBVjFS4PkBGim56ap7Ej`
