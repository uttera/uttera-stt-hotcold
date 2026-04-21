# uttera-stt-hotcold

<p align="center">
  <a href="https://uttera.ai">
    <img src="docs/img/banner.png" alt="uttera.ai — The voice layer for your AI" width="800">
  </a>
</p>

High-performance Whisper STT API server with a hybrid "Hot/Cold" worker architecture.

**Ideal for locally running installations of agents like OpenClaw or Open-WebUI, where the media should not leave the private local domain.**

> **Created and maintained by [Hugo L. Espuny](https://github.com/fakehec).**
> Part of the [Uttera](https://uttera.ai) voice stack.
> Licensed under the [Apache License 2.0](LICENSE).
> See [NOTICE](NOTICE) for third-party attributions.

## 📢 Project history: renamed and transferred

This repository has been **renamed** from `whisper-stt-local-server` to
**`uttera-stt-hotcold`** and **transferred** from its original creator's
personal page ([@fakehec](https://github.com/fakehec)) to the
[Uttera GitHub organization](https://github.com/uttera).

GitHub redirects old URLs automatically, so any existing clones, forks,
bookmarks, and links keep working. If you still have
`fakehec/whisper-stt-local-server` as your `origin`, consider updating:

```bash
git remote set-url origin https://github.com/uttera/uttera-stt-hotcold.git
```

## Positioning

| Use case | This repo | Sibling repo |
|---|---|---|
| Home-lab, personal, small/mid GPU (8–16 GB) | ✅ [uttera-stt-hotcold](https://github.com/uttera/uttera-stt-hotcold) | — |
| Cloud, multi-tenant, large GPU (≥24 GB) | — | [uttera-stt-vllm](https://github.com/uttera/uttera-stt-vllm) |

**Choose `uttera-stt-hotcold` when**:
- You have consumer GPUs (RTX 4070, 4080) and transcribe occasionally.
- Personal or single-user deployment.
- You want to share the GPU with other workloads.
- **You have 8–24 GB of VRAM.** vLLM does not fit comfortably in this
  range: at 8–16 GB the KV cache is too small for continuous batching
  to beat hotcold; at 16–24 GB vLLM works but reserves 11–22 GB
  permanently, wasting the co-location flexibility that is hotcold's
  reason to exist on mid-sized GPUs.

**Choose `uttera-stt-vllm` when**:
- You transcribe hours of audio per day across many concurrent streams.
- You want continuous batching to maximise GPU utilisation.
- You have large-VRAM GPUs dedicated to inference.
- **You have 32 GB+ of VRAM** (vLLM reserves ~22–29 GB at startup
  depending on `gpu_memory_utilization`; below 32 GB total you either
  run out of headroom or lose the batching advantage that justifies
  the reservation).

See [`uttera-benchmarks`](https://github.com/uttera/uttera-benchmarks)
for reproducible head-to-head numbers across four load profiles
(latency, burst up to N=1024, sustained) and two corpora (LibriSpeech
test-clean and an internal Spanish WAV corpus).

## 🚀 Key Features

*Concurrency and engine*
- **Hybrid hot/cold pool:**
  - **Hot worker:** Whisper resident in VRAM for sub-second (~0.2 s)
    inference on short clips.
  - **Cold workers:** on-demand subprocesses spawned on the GPU when
    the hot lane is busy, so long audio files don't block quick voice
    commands. Drains idle after `COLD_WORKER_IDLE_TIMEOUT`.
- **GPU accelerated** via NVIDIA CUDA. fp16 + fp32-LayerNorm by
  default (`WHISPER_FP16=1`) — halves VRAM with no quality loss.

*OpenAI-compatible API*
- Standard endpoints: `POST /v1/audio/transcriptions`,
  `POST /v1/audio/translations`.
- `GET /v1/models` for client autodiscovery (reports `whisper-1`,
  `owned_by: uttera`).
- **All five OpenAI `response_format` values really supported**
  (v2.2.0) — `json`, `text`, `verbose_json`, `srt`, `vtt`. Previously
  `srt` / `vtt` / `verbose_json` silently collapsed to the compact
  JSON form.

*Translation*
- **`POST /v1/audio/translations`** with `to_language` (default `en`).
  With `LIBRETRANSLATE_URL` set: Whisper-transcribe → LibreTranslate
  pipeline to any target language. Without: Whisper native `translate`
  (English only; poor on `turbo`-class models).
- `to_language != "en"` without `LIBRETRANSLATE_URL` → **HTTP 400**
  with the missing-env-var name (previously a silent fallback to
  English — a contract violation).
- **`X-Translation-Mode: libretranslate`** response header whenever
  the LibreTranslate path runs, so clients and observability tooling
  can tell which engine handled the call.

*Validation and observability*
- Strict validation on every knob — out-of-range returns HTTP 422 or
  HTTP 400 with a useful detail body:
  - `response_format` must be one of `json|text|verbose_json|srt|vtt`.
  - `temperature` ∈ `[0.0, 1.0]` (OpenAI spec).
  - Undecodeable / non-audio file bodies → HTTP 400 with the typed
    decode error (was HTTP 500 before v2.2.0).
  - Unsupported Whisper language codes → HTTP 400 with the message
    (was generic HTTP 500 before v2.2.0).
- **`X-Route`** response header — `HOT` / `COLD-POOL` / `COLD-POOL>HOT`
  — tells the client which lane handled the request, exposed to
  browsers via CORS when CORS is enabled.
- Multilingual: 99 languages covered by Whisper. Auto-detects if
  `language` is omitted.

*Operations*
- `GET /health` **and `HEAD /health`** (v2.2.0) expose version,
  model, worker status, queue depth, VRAM — one struct for both
  proxies and Docker healthchecks.
- Opt-in **`CORSMiddleware`** via `CORS_ALLOW_ORIGINS` env var
  (disabled by default — API-first deployments don't need it).
  Exposes `X-Route` and `X-Translation-Mode` to browser clients.
- Canonical Uttera-stack port **`9005`** (STT family). TTS family
  uses `9004`. Swapping `hotcold ↔ vllm` is a backend change, not
  a port change.
- Optional Redis self-registration (`REDIS_URL`) for upstream router
  discovery — same protocol as the sibling `uttera-stt-vllm` and the
  TTS servers.

*Privacy*
- 100% local execution. Your audio never leaves your infrastructure.

## 🧠 Available Models

| Model | Params | VRAM (fp16) | Speed | Languages | Best for |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tiny` | 39M | ~1 GB | Fastest | 99 | Testing, low-resource |
| `tiny.en` | 39M | ~1 GB | Fastest | English only | English-only, low-resource |
| `base` | 74M | ~1 GB | Fast | 99 | Light workloads |
| `base.en` | 74M | ~1 GB | Fast | English only | Light English-only |
| `small` | 244M | ~2 GB | Moderate | 99 | Good accuracy/speed balance |
| `small.en` | 244M | ~2 GB | Moderate | English only | English-only balanced |
| `medium` | 769M | ~5 GB | Slow | 99 | **Default.** High accuracy |
| `medium.en` | 769M | ~5 GB | Slow | English only | English-only high accuracy |
| `large` | 1550M | ~10 GB | Slowest | 99 | Maximum accuracy (v1) |
| `large-v2` | 1550M | ~10 GB | Slowest | 99 | Improved large |
| `large-v3` | 1550M | ~10 GB | Slowest | 99 | Best accuracy overall |
| `turbo` | 809M | ~6 GB | Fast | 99 | **Recommended.** large-v3 distilled, best quality/speed |

Set the model via `WHISPER_MODEL` in `.env`. To download all models at once for offline use:

```bash
source venv/bin/activate
python3 -c "
import whisper
for m in ['tiny','tiny.en','base','base.en','small','small.en',
          'medium','medium.en','large','large-v2','large-v3','turbo']:
    print(f'Downloading {m}...')
    whisper.load_model(m, download_root='assets/models/whisper')
    print(f'  Done: {m}')
"
```

## 📦 Installation & Setup

### 1. Prerequisites (Debian/Ubuntu)
Install the following system dependencies first:
```bash
sudo apt update && sudo apt install -y ffmpeg python3 python3-venv
```

> **Python version:** `setup.sh` uses the system default `python3` (3.12+ recommended). torch is pinned to `>=2.9.0,<2.10.0` to avoid CUDA 13 NPP dependency issues with newer versions.

### 2. Unified Installation
```bash
git clone https://github.com/uttera/uttera-stt-hotcold.git
cd uttera-stt-hotcold
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
The server listens on port `9005` by default. Ensure the user has permissions to open sockets on this port (standard for ports >1024).

## 📡 API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` / `HEAD` | `/health` | Server liveness, version, model, worker status, queue, VRAM. |
| `GET` | `/v1/models` | OpenAI-compatible model list (`whisper-1`, `owned_by: uttera`). |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text (Hot or Cold Lane). Supports `json` / `text` / `verbose_json` / `srt` / `vtt` response formats. |
| `POST` | `/v1/audio/translations` | Transcribe + translate to `to_language` (default `en`). With `LIBRETRANSLATE_URL`: any target language. Without: English only (Whisper native). |

See [API.md](API.md) for full request/response schemas, validation
ranges, `X-Route` / `X-Translation-Mode` semantics, and the error
taxonomy (400 / 422 / 502).

## 🛠 Execution

The server uses direct **Uvicorn** execution for maximum ASGI performance.

### Manual Execution (Console)
```bash
source venv/bin/activate

# Localhost only
uvicorn main_stt:app --host 127.0.0.1 --port 9005

# Expose to local network
uvicorn main_stt:app --host 0.0.0.0 --port 9005
```

### ⚙️ Environment Variables & .env

Copy `.env.example` to `.env` and adjust as needed. All variables are optional.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WHISPER_MODEL` | `medium` | Model to load: `tiny`, `base`, `small`, `medium`, `large`. |
| `WHISPER_FP16` | `1` | fp16+LayerNorm-fp32 (halves VRAM). Set to `0` for fp32. |
| `COLD_POOL_SIZE` | `10` | Max concurrent cold workers (safety cap). |
| `COLD_WORKER_IDLE_TIMEOUT` | `60` | Seconds before idle cold worker exits. |
| `COLD_WORKER_IDLE_STAGGER` | `10` | Stagger per worker slot to avoid mass die-off. |
| `MIN_COLD_VRAM_GB` | `4.0` | Min free VRAM to spawn a cold worker (0=disable). |
| `COLD_LANE_TIMEOUT_SECONDS` | `300` | Max seconds to wait for a Cold Lane subprocess before HTTP 500. |
| `ROUTING_DRAIN_CAP_SECONDS` | `120` | Queue drain time considered 100% load. |
| `REDIS_URL` | *(empty)* | Redis URL for node self-registration (opt-in). |
| `NODE_HOST` | `localhost` | Host advertised to Redis for Gatekeeper routing. |
| `NODE_PORT` | `9005` | Port advertised to Redis for Gatekeeper routing. |
| `DEBUG` | `false` | Set to `true` to enable worker routing and subprocess traces. |
| `VENV_PYTHON` | *(auto-detected)* | Path to venv Python. Auto-detected from `venv/bin/python`. |

*See `.env.example` for the full list of variables and their defaults.*

### User Service (systemd --user)
1. Create directory if it doesn't exist: `mkdir -p ~/.config/systemd/user`
2. Create: `~/.config/systemd/user/uttera-stt.service`
3. Configuration (environment variables are loaded from your `.env` file):

```ini
[Unit]
Description=Uttera STT Hot/Cold Server
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/uttera-stt-hotcold
ExecStart=%h/uttera-stt-hotcold/venv/bin/uvicorn main_stt:app --host 127.0.0.1 --port 9005
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

4. Enable and start:
```bash
systemctl --user daemon-reload
systemctl --user enable --now uttera-stt.service
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
curl http://localhost:9005/health

# View logs
docker compose logs -f

# Stop
docker compose down
```

The model is persisted in `assets/models/whisper/` (host volume), so it only downloads once.

## 📊 Observability (`/metrics`)

`GET /metrics` returns Prometheus-format metrics for direct scraping
by Prometheus, Telegraf's `inputs.prometheus` plugin, or any other
OpenMetrics-compatible consumer. Metrics share the `uttera_stt_*`
namespace with the sibling `uttera-stt-vllm` backend (same names
and label shapes for the common series — the `engine` label in
`uttera_stt_build_info` differentiates the variant), plus this
server's additional hot/cold pool telemetry.

```toml
[[inputs.prometheus]]
  urls = ["http://stt-host:9005/metrics"]
  interval = "15s"
```

Key series:

| Metric | Type | Use |
|---|---|---|
| `uttera_stt_requests_total{endpoint,method,status}` | Counter | Per-endpoint request rate + status mix |
| `uttera_stt_request_duration_seconds{endpoint,method}` | Histogram | HTTP p50/p95/p99 (total RTT) |
| `uttera_stt_inflight_requests` | Gauge | Live load (hot + cold combined) |
| `uttera_stt_requests_by_route_total{route}` | Counter | Lane split — `HOT` / `COLD-POOL` / `COLD-POOL>HOT` |
| `uttera_stt_transcriptions_total{response_format}` | Counter | Traffic mix across the five response formats |
| `uttera_stt_translations_total{mode,response_format}` | Counter | Translation path breakdown |
| `uttera_stt_audio_seconds_total{endpoint,route}` | Counter | Audio processed, lane-tagged — billing / throughput proxy |
| `uttera_stt_inference_duration_seconds{op}` | Histogram | Lane-tagged model latency: `whisper_transcribe_hot` / `whisper_transcribe_cold` / `libretranslate` |
| `uttera_stt_cold_workers_active` | Gauge | Live cold subprocesses |
| `uttera_stt_cold_workers_loading` | Gauge | Cold subprocesses booting |
| `uttera_stt_cold_worker_pool_size_cap` | Gauge | `COLD_POOL_SIZE` |
| `uttera_stt_cold_workers_spawned_total` | Counter | Monotonic spawn count (for cold-worker-churn dashboards) |
| `uttera_stt_cold_worker_ema_start_seconds` | Gauge | Rolling EMA of cold-worker boot time |
| `uttera_stt_work_queue_depth` | Gauge | Items queued |
| `uttera_stt_work_queue_audio_seconds` | Gauge | Audio queued (for drain-time estimate) |
| `uttera_stt_load_score` | Gauge | Saturation signal `[0.0, 1.0]` |
| `uttera_stt_hot_ema_sps` | Gauge | Hot-lane throughput EMA |
| `uttera_stt_vram_free_gb` | Gauge | GPU memory headroom |
| `uttera_stt_vram_per_cold_worker_gb` | Gauge | Rolling EMA of VRAM per cold subprocess |
| `uttera_stt_engine_ready` | Gauge | 1 if hot worker loaded |
| `uttera_stt_libretranslate_configured` | Gauge | 1 if `LIBRETRANSLATE_URL` was set at startup |
| `uttera_stt_errors_total{type}` | Counter | Typed errors (`decode` / `validation` / `model` / `libretranslate`) |
| `uttera_stt_build_info{version,engine,model}` | Gauge | Version + engine + model in the field (value always `1`) |

## 🔒 Security & Network Note
By default, the server binds to **`127.0.0.1`** on port **`9005`**.
- To allow external network access, change `--host` to `0.0.0.0`.
- **WARNING**: This API **does not have authentication**. Exposing it to the network via `0.0.0.0` represents a security risk. Ensure the server is protected by a firewall or operating within a secure VPN/local network.

## 📊 Performance (NVIDIA RTX 5090, fp16, medium model)

| Task | Latency |
| :--- | :--- |
| Short command (2s audio, Hot Lane) | **~0.2s** |
| Long audio (30s, Hot Lane) | **~0.7s** |
| 160 concurrent (Hot + Cold Pool) | Target ~21s total, 0 failures |

## 🛡 License

**Server source code**: [Apache License 2.0](LICENSE). Commercial use permitted.

**Whisper model weights** (OpenAI): released under the MIT License —
commercial use permitted, no restrictions. See [NOTICE](NOTICE) for full
attributions.

Created and maintained by [Hugo L. Espuny](https://github.com/fakehec),
with contributions acknowledged in [AUTHORS.md](AUTHORS.md).

## ☕ Community

If you want to follow the project or get involved:

- ⭐ Star this repo to help discoverability.
- 🐛 Report issues via the [issue tracker](../../issues).
- 💬 Join the conversation in [Discussions](../../discussions).
- 📰 Technical posts at [blog.uttera.ai](https://blog.uttera.ai).
- 🌐 Uttera Cloud: [https://uttera.ai](https://uttera.ai) (EU-hosted,
  solar-powered, subscription flat-rate).

---

*Uttera /ˈʌt.ər.ə/ — from the English verb "to utter" (to speak aloud, to
pronounce, to give audible expression to). Formally, the name is a backronym
of **U**niversal **T**ext **T**ransformer **E**ngine for **R**ealtime **A**udio
— reflecting the project's origin as a STT/TTS server and its underlying
Transformer architecture.*
