# uttera-stt-hotcold API

The server provides an OpenAI-compatible API for high-performance
speech-to-text transcription and translation, backed by OpenAI
Whisper with a hybrid hot/cold worker pool. Optional LibreTranslate
post-processing extends `/v1/audio/translations` to any target
language.

**Base URL:** `http://localhost:9005`

---

## 1. Transcription Endpoint

### `POST /v1/audio/transcriptions`

Transcribes audio to text in the original language.

#### Headers
- `Content-Type: multipart/form-data`

#### Form-data Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | File | **Required** | The audio file to transcribe. Undecodeable / non-audio bodies → HTTP 400 with a typed decode error. |
| `model` | String | `whisper-1` | Model identifier (OpenAI-compatible). Ignored — the served model is fixed by `WHISPER_MODEL` at startup. |
| `language` | String | `null` | Language of the input audio (ISO-639-1). Unsupported codes → HTTP 400 with the Whisper message. Omit for auto-detection. |
| `prompt` | String | `null` | Optional guide for the model's style (OpenAI spec). |
| `response_format`| String | `json` | One of `json`, `text`, `verbose_json`, `srt`, `vtt`. Any other value → HTTP 422. `srt` and `vtt` emit real subtitle files (`HH:MM:SS,mmm` / `HH:MM:SS.mmm`); `verbose_json` exposes segments, logprobs, and detected language (since v2.2.0). |
| `temperature` | Float | `0.0` | Controls randomness. Valid range `[0.0, 1.0]` (OpenAI spec) — out-of-range → HTTP 422. |

---

## 2. Translation Endpoint

### `POST /v1/audio/translations`

Transcribes audio in any language and translates it to a target language.

- **When `LIBRETRANSLATE_URL` is configured** (recommended): Whisper
  transcribes the audio in the source language (auto-detected or forced
  via the `language` form field), then the text is sent to LibreTranslate
  to reach `to_language` (default `en`). Supports arbitrary target
  languages.
- **When `LIBRETRANSLATE_URL` is empty** (legacy fallback): Whisper native
  `translate` task. English-only. Works poorly on models that were not
  trained for this task (e.g. `whisper-large-v3-turbo`).

#### Headers
- `Content-Type: multipart/form-data`

#### Form-data Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | File | **Required** | The audio file to translate. Undecodeable bodies → HTTP 400. |
| `model` | String | `whisper-1` | Model identifier. Ignored — fixed at startup. |
| `prompt` | String | `null` | Optional guide for the model (OpenAI spec). |
| `response_format`| String | `json` | One of `json`, `text`, `verbose_json`, `srt`, `vtt`. Same validation as the transcription endpoint. |
| `temperature` | Float | `0.0` | Valid range `[0.0, 1.0]` — out-of-range → HTTP 422. |
| `to_language` | String | `en` | ISO-639-1 target (`en`, `es`, `fr`, `de`, …). Any non-`en` value with `LIBRETRANSLATE_URL` unset → HTTP 400 naming the missing env var (since v2.2.0 — was a silent fallback to English before). |
| `language` | String | `null` | Optional **source** language hint for Whisper; omit for auto-detect. |

#### Response headers (translations)

- **`X-Route: HOT` / `COLD-POOL` / `COLD-POOL>HOT`** — which lane handled the underlying Whisper transcription.
- **`X-Translation-Mode: libretranslate`** — emitted when the LibreTranslate post-processing path was used (vs. the legacy Whisper-native translate). Absent on the legacy path.

#### Errors

| HTTP | When |
|---|---|
| 400 | Undecodeable audio (non-audio body, unsupported codec) → `detail: "Failed to decode audio: <ExceptionType>: <msg>"`. Or unsupported source language code. Or `to_language != "en"` with `LIBRETRANSLATE_URL` unset. |
| 422 | `response_format` not in the supported set, or `temperature` out of `[0.0, 1.0]`. |
| 502 | LibreTranslate call failed (network, HTTP error, or malformed response). **No silent fallback to the untranslated transcription** — leaking source-language text under a response schema that promises the target language would be a correctness bug, not a graceful degradation. |
| 503 | Hot worker still loading on startup, or engine crashed and hasn't recovered. |

---

## 3. Utility Endpoints

### `GET /health` and `HEAD /health`

Returns server liveness and hot worker status. Both methods are
accepted — `HEAD` returns the same headers as `GET` with an empty
body, useful for uptime probes that don't want to parse JSON.

**Example Response:**
```json
{
  "status": "ok",
  "version": "2.3.0",
  "model": "medium",
  "hot_worker_loaded": true,
  "hot_worker_error": null,
  "routing": {
    "load_score": 0.0,
    "accepts_requests": true
  },
  "smart_routing": {
    "ema_sps": 12.5,
    "cold_start_calibrated": true,
    "cold_ema_start_seconds": 8.2,
    "queue_depth": 0,
    "queue_audio_seconds": 0.0,
    "queue_drain_estimate_seconds": 0.0,
    "pool_workers_active": 0,
    "pool_workers_loading": 0,
    "pool_workers_optimal": 0,
    "pool_size_cap": 10,
    "vram_free_gb": 28.5,
    "cold_vram_ema_gb": 1.8,
    "vram_sufficient_for_cold": true,
    "cold_start_configured_seconds": 10.0,
    "safety_factor": 1.3,
    "min_cold_vram_gb": 4.0,
    "cold_vram_per_worker_gb": 1.8
  }
}
```

### `GET /v1/models`

OpenAI-compatible model listing. Returns the currently loaded Whisper model.

**Example Request:**
```bash
curl -X GET "http://localhost:9005/v1/models"
```

**Example Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "uttera"}
  ]
}
```

*(`owned_by` was corrected from the pre-rebrand `"uttera-legacy"` to `"uttera"` in v2.2.1.)*

---

## 4. Examples

### Transcription (JSON)
```bash
curl -X POST "http://localhost:9005/v1/audio/transcriptions" \
     -F "file=@/path/to/audio.wav" \
     -F "language=es" \
     -F "response_format=json"
```

### Transcription with subtitles (SRT)
```bash
curl -X POST "http://localhost:9005/v1/audio/transcriptions" \
     -F "file=@/path/to/audio.wav" \
     -F "response_format=srt" \
     -o subtitles.srt
```

### Translation (to English, Whisper native)
```bash
curl -X POST "http://localhost:9005/v1/audio/translations" \
     -F "file=@/path/to/spanish_audio.wav" \
     -F "response_format=text"
```

### Translation to French (requires LIBRETRANSLATE_URL)
```bash
curl -X POST "http://localhost:9005/v1/audio/translations" \
     -F "file=@/path/to/spanish_audio.wav" \
     -F "to_language=fr" \
     -F "response_format=json"
# Response header: X-Translation-Mode: libretranslate
```

---

## 5. CORS

Disabled by default — this server is API-first, typically consumed
by backend-to-backend callers or served through the Uttera
gatekeeper.

To enable browser-origin access, set `CORS_ALLOW_ORIGINS` to a
comma-separated list of origins (or `*` for permissive):

```bash
CORS_ALLOW_ORIGINS="https://app.uttera.ai,https://dev.uttera.ai"
# or:
CORS_ALLOW_ORIGINS="*"
```

Methods, headers, and credentials follow the FastAPI
`CORSMiddleware` defaults (allow all methods, allow all headers,
credentials enabled). The `X-Route` and `X-Translation-Mode`
response headers are explicitly exposed to browser clients so they
can be read from JavaScript.

## 6. Authentication

No authentication in this repo by design. Deploy behind the Uttera
gatekeeper (or any reverse proxy) for API keys, quotas, and rate
limits.
