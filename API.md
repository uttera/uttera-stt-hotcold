# Whisper STT Local Server API Documentation

The server provides an OpenAI-compatible API for high-performance speech-to-text transcription and translation.

**Base URL:** `http://localhost:5000`

---

## 1. Transcription Endpoint

### `POST /v1/audio/transcriptions`

Transcribes audio to text in the original language.

#### Headers
- `Content-Type: multipart/form-data`

#### Form-data Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | File | **Required** | The audio file to transcribe. |
| `model` | String | `whisper-1` | Model identifier (OpenAI compatible). |
| `language` | String | `null` | Language of the input audio (ISO-639-1). |
| `prompt` | String | `null` | Optional guide for the model's style. |
| `response_format`| String | `json` | Output format: `json` or `text`. |
| `temperature` | Float | `0.0` | Controls randomness. |

---

## 2. Translation Endpoint

### `POST /v1/audio/translations`

Transcribes audio in any language and translates it to English in a single pass.

#### Headers
- `Content-Type: multipart/form-data`

#### Form-data Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | File | **Required** | The audio file to translate. |
| `model` | String | `whisper-1` | Model identifier. |
| `prompt` | String | `null` | Optional guide for the model. |
| `response_format`| String | `json` | Output format: `json` or `text`. |
| `temperature` | Float | `0.0` | Controls randomness. |

---

## 3. Utility Endpoints

### `GET /health`

Returns server liveness and hot worker status. Suitable for monitoring and healthchecks.

**Example Response:**
```json
{
  "status": "ok",
  "version": "1.3.6",
  "model": "medium",
  "hot_worker_loaded": true,
  "hot_worker_error": null
}
```

### `GET /v1/models`

OpenAI-compatible model listing. Returns the currently loaded Whisper model.

**Example Request:**
```bash
curl -X GET "http://localhost:5000/v1/models"
```

**Example Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "uttera-legacy"}
  ]
}
```

---

## 4. Examples

### Transcription (JSON)
```bash
curl -X POST "http://localhost:5000/v1/audio/transcriptions" \
     -F "file=@/path/to/audio.wav" \
     -F "language=es" \
     -F "response_format=json"
```

### Translation (to English)
```bash
curl -X POST "http://localhost:5000/v1/audio/translations" \
     -F "file=@/path/to/spanish_audio.wav" \
     -F "response_format=text"
```
