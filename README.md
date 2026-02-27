# whisper-hybrid-router

High-performance Whisper STT API server with a hybrid "Hot/Cold" worker architecture. Optimized for single-GPU setups where low latency and high availability are critical.

## 🚀 Key Features

- **Hybrid Concurrency:**
  - **Hot Worker:** Keeps a Whisper model resident in VRAM for sub-second (0.2s) inference.
  - **Cold Workers:** Spawns on-demand subprocesses when the GPU is busy, ensuring that long audio files don't block quick voice commands.
- **OpenAI Compatible:** Polimorphic endpoint `/v1/audio/transcriptions` supporting standard parameters (`language`, `prompt`, `temperature`, `response_format`).
- **Hardware Accelerated:** Designed to squeeze maximum performance from NVIDIA GPUs.
- **Privacy First:** 100% local execution. Your data never leaves your infrastructure.

## 📊 Performance Benchmarks (Sphinx GPU)

| Task | Sphinx (GPU Hybrid) | Standard Cloud API |
| :--- | :--- | :--- |
| Short Command (2s) | **0.2s** | ~2.5s |
| Long Strategic Audio (30s) | **0.7s** | ~20s |

## 🛠 Installation & Usage

(Documentation in progress...)

## 🛡 License

GNU GPL v3. Maintainer: Hugo Espuny & Uttera
