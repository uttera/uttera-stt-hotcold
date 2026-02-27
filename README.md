# whisper-stt-local-server

High-performance Whisper STT API server with a hybrid "Hot/Cold" worker architecture. 

**Ideal for locally running installations of agents like OpenClaw or Open-WebUI, where the media should not leave the private local domain.**

## 🚀 Key Features

- **Hybrid Concurrency:**
  - **Hot Worker:** Keeps a Whisper model resident in VRAM for sub-second (0.2s) inference.
  - **Cold Workers:** Spawns on-demand subprocesses when the GPU is busy, ensuring that long audio files don't block quick voice commands.
- **OpenAI Compatible:** Polimorphic endpoint `/v1/audio/transcriptions` supporting standard parameters (`language`, `prompt`, `temperature`, `response_format`).
- **Hardware Accelerated:** Designed to squeeze maximum performance from NVIDIA GPUs.
- **Privacy First:** 100% local execution. Your data never leaves your infrastructure.

## 📦 Requirements

- **Whisper (Python Implementation):** This server is based on the original [OpenAI Whisper](https://github.com/openai/whisper) repository.
- **FFmpeg:** Required for audio processing.
- **NVIDIA GPU:** For hardware acceleration (CUDA).
- **Python 3.10+**

## 🛠 Installation

### 1. Manual Execution (Console)
To run the server manually for testing or development:

```bash
# Set environment variables (optional)
export WHISPER_MODEL=medium
export DEBUG=false

# Execute using Uvicorn
uvicorn main_stt:app --host 0.0.0.0 --port 5000
```

### 2. System Service (systemd)
To ensure the server runs continuously as a background service:

1. Create a service file: `/etc/systemd/system/whisper-stt.service`
2. Add the following configuration (standard for Debian/Ubuntu):

```ini
[Unit]
Description=Whisper STT Local Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/usr/local/lib/whisper
Environment="WHISPER_MODEL=medium"
Environment="DEBUG=false"
Environment="XDG_CACHE_HOME=/opt/ai/models/speech"
ExecStart=/usr/local/lib/whisper/bin/python -m uvicorn main_stt:app --host 0.0.0.0 --port 5000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable whisper-stt
sudo systemctl start whisper-stt
```

## 📊 Performance Benchmarks (Sphinx GPU)

| Task | Sphinx (GPU Hybrid) | Standard Cloud API |
| :--- | :--- | :--- |
| Short Command (2s) | **0.2s** | ~2.5s |
| Long Strategic Audio (30s) | **0.7s** | ~20s |

## 🛡 License

GNU GPL v3. 
Maintainers: Hugo Espuny (<>) & Uttera (<>)
