# Windows VPS API

This project now includes a warm HTTP API for Windows VPS deployment.

## Why this path

- the OCR pipeline is loaded once at startup
- later requests reuse the same pipeline instance
- this avoids per-request model startup overhead

## Recommended command

```powershell
python -m ocrreader.api `
  --host 0.0.0.0 `
  --port 8765 `
  --config config/ruhsat_schema_paddle_v29_gpu_lazy.yaml `
  --debug-root output/api_debug
```

## Endpoints

### `GET /health`

Returns service status and runtime information.

### `GET /metrics`

Returns:
- uptime
- request count
- average latency
- runtime backend info

### `POST /ocr`

Multipart form fields:
- `image` required file upload
- `full_output` optional boolean
- `save_debug` optional boolean

Example:

```powershell
curl.exe -X POST `
  -F "image=@testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" `
  -F "full_output=false" `
  -F "save_debug=false" `
  http://127.0.0.1:8765/ocr
```

## Notes

- For CUDA/NVIDIA machines, use `config/ruhsat_schema_paddle_v29_gpu_lazy.yaml`.
- For CPU-only fallback, use `config/ruhsat_schema_paddle_v29_cpu_lazy.yaml`.
- Do not install both `paddlepaddle` and `paddlepaddle-gpu` in the same interpreter.
