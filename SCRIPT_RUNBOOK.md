# Script Runbook

This file lists the runnable commands that remain after removing the dataset generation and benchmark pipeline.

## Before You Start

- Activate your environment.
- Install `requirements.txt`.
- If you use Paddle + GPU, also follow `INSTALL_WINDOWS_GPU.md`.

Example:

```powershell
conda activate ocrreader
python -m pip install -r requirements.txt
```

Recommended bootstrap on a fresh machine:

```powershell
python scripts/bootstrap_runtime.py
```

Preview only:

```powershell
python scripts/bootstrap_runtime.py --dry-run
```

---

## 1) Main OCR CLI

### Fast runtime - `HybridOCREngine`

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29.yaml" ^
  --output "output/result_fast.json" ^
  --debug-dir "output/debug_fast"
```

Uses `PaddleOCR` for full-page word detection and `Tesseract` for per-field crop OCR.

### Slow runtime - `PaddleOCR-VL`

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29_allfields_glm.yaml" ^
  --output "output/result_vl.json" ^
  --debug-dir "output/debug_vl"
```

Uses the experimental full-page `PaddleOCR-VL` fallback path.

### Basic runtime - `Tesseract`

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema.yaml" ^
  --output "output/result_tesseract.json" ^
  --debug-dir "output/debug_tesseract"
```

Uses the plain `Tesseract` pipeline without `PaddleOCR`.

Notes:
- `config/ruhsat_schema_paddle_v29.yaml` selects the faster hybrid path and now uses `engine: auto`.
- `config/ruhsat_schema_paddle_v29_allfields_glm.yaml` selects the slower `PaddleOCR-VL` path and now uses `engine: auto` for the page OCR layer.
- Classic OCR auto engine behavior:
  - `Windows + CUDA Paddle` -> `paddleocr`
  - `Windows + non-CUDA GPU + DML provider` -> `onnxruntime`
  - `macOS + CoreML provider` -> `onnxruntime`
  - otherwise -> `paddleocr`
- `config/ruhsat_schema.yaml` selects the simplest fallback path.
- `--image` is required.
- `--config` defaults to `config/ruhsat_schema.yaml`.
- `--output` is optional.
- `--debug-dir` is optional.

### Warm API service - Windows VPS friendly

Uses one long-lived process, so Paddle stays loaded in memory and later requests avoid cold start.

Recommended GPU config:

```powershell
python -m ocrreader.api ^
  --host 0.0.0.0 ^
  --port 8765 ^
  --config config/ruhsat_schema_paddle_v29_gpu_lazy.yaml ^
  --debug-root output/api_debug
```

Sample health check:

```powershell
curl.exe http://127.0.0.1:8765/health
```

Sample OCR request:

```powershell
curl.exe -X POST ^
  -F "image=@testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  -F "full_output=false" ^
  -F "save_debug=false" ^
  http://127.0.0.1:8765/ocr
```

Helpful endpoints:
- `GET /health`
- `GET /metrics`
- `POST /ocr`

Recommended config choices:
- `config/ruhsat_schema_paddle_v29_gpu_lazy.yaml` for CUDA/NVIDIA machines
- `config/ruhsat_schema_paddle_v29_cpu_lazy.yaml` for CPU fallback

Important:
- `requirements.txt` no longer pins a default Paddle runtime on purpose.
- Install **either** `paddlepaddle` **or** `paddlepaddle-gpu`, not both.

### Experimental backend migration probes

These commands do **not** switch the main OCR API. They only test whether non-CUDA runtimes are available and how far the migration path can go on the current machine.

Optional install set:

```powershell
python -m pip install -r requirements.experimental.txt
```

Probe the migration blockers and available providers:

```powershell
python -m scripts.probe_backend_migrations --attempt-installs
```

Run a tiny runtime microbenchmark for `DirectML` and `OpenVINO`:

```powershell
python -m scripts.benchmark_runtime_providers
```

Outputs:
- `output/backend_migration_probe.json`
- `output/runtime_provider_benchmark.json`

### ONNX + DirectML path

Use this when the deployment target is a Windows machine and you want a non-CUDA GPU path.

Architecture:
- export ONNX on WSL/Linux
- run inference on Windows with `onnxruntime-directml`

Probe the exported OCR ONNX models on Windows:

```powershell
python -m scripts.probe_directml_ocr_models
```

Profile the integrated ONNX + DirectML OCR pipeline:

```powershell
python -m scripts.profile_runtime_mode ^
  --label onnx_directml ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config config/ruhsat_schema_onnx_directml_v29.yaml ^
  --runs 2 ^
  --summary-out output/onnx_directml_profile.json
```

Run the warm API on Windows:

```powershell
python -m ocrreader.api ^
  --host 0.0.0.0 ^
  --port 8766 ^
  --config config/ruhsat_schema_onnx_directml_v29.yaml ^
  --debug-root output/api_onnx_debug
```

Full notes:
- `docs/onnx_directml_path.md`
- `output/directml_ocr_model_probe.json`
- `output/onnx_directml_profile.json`
- `output/api_onnx_health.json`
- `output/api_onnx_result_1.json`
- `output/api_onnx_result_2.json`
- `output/api_onnx_metrics.json`

---

## 2) Anchor / Schema Debugging

### Probe anchor label hits on one image

```powershell
python scripts/anchor_label_probe.py ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config config/ruhsat_schema_paddle_v29.yaml ^
  --out-csv output/anchor_label_probe.csv ^
  --psm 6
```

### Collect anchor templates from local probe outputs

Auto mode:

```powershell
python scripts/collect_anchor_templates.py ^
  --mode auto ^
  --probe-dir output ^
  --image-dir testdata ^
  --out-dir config/anchor_templates ^
  --min-conf 55
```

Manual mode:

```powershell
python scripts/collect_anchor_templates.py ^
  --mode manual ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --out-dir config/anchor_templates
```

### Dump a call trace for one image

```powershell
python scripts/debug_call_trace.py ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config config/ruhsat_schema_paddle_v29.yaml ^
  --pipeline-module ocrreader.pipeline ^
  --output output/result.json ^
  --trace-out output/debug_call_trace.txt ^
  --debug-dir output/debug_trace
```

---

## 3) Output Utilities

### Export OCR JSON outputs into one CSV

```powershell
python scripts/export_results_columns.py ^
  --input-dir output ^
  --output output/results_columns.csv
```

---

## 4) PaddleOCR-VL / GLM Helper Scripts

These are local investigation helpers. They now use `testdata/` only.

For non-NVIDIA local deployment planning, see:

- `docs/paddle_vl_non_nvidia_local_runbook.md`
- `docs/paddle_vl_local_analysis.md`

### Verify the sample image with integrated VL fallback

```powershell
python tests/verify_user_image.py
```

### Run the same helper on the sample image with direct console output

```powershell
python tests/verify_integrated_vl.py
```

### Profile tuned local PaddleOCR-VL on one image

```powershell
python tests/profile_paddle_vl.py ^
  --layout ^
  --image-block-ocr ^
  --runtime-profile native_paddle_vl ^
  --max-side 1600 ^
  --max-new-tokens 512 ^
  --output tests/profile_paddle_vl_tuned_layout_1600.txt ^
  --summary tests/profile_paddle_vl_tuned_layout_1600.json
```

### Profile PaddleOCR-VL through a local vLLM service

```powershell
python tests/profile_paddle_vl.py ^
  --layout ^
  --image-block-ocr ^
  --runtime-profile local_vllm_service ^
  --service-url "http://localhost:8118/v1" ^
  --service-model-name "PaddlePaddle/PaddleOCR-VL-1.5" ^
  --max-side 1600 ^
  --max-new-tokens 512 ^
  --output tests/profile_paddle_vl_vllm_service.txt ^
  --summary tests/profile_paddle_vl_vllm_service.json
```

### Profile PaddleOCR-VL through a local MLX-VLM service

```powershell
python tests/profile_paddle_vl.py ^
  --layout ^
  --image-block-ocr ^
  --runtime-profile local_mlx_vlm_service ^
  --service-url "http://localhost:8111/" ^
  --service-model-name "PaddlePaddle/PaddleOCR-VL-1.5" ^
  --max-side 1600 ^
  --max-new-tokens 512 ^
  --output tests/profile_paddle_vl_mlx_service.txt ^
  --summary tests/profile_paddle_vl_mlx_service.json
```

### Inspect raw VL output structure

```powershell
python tests/inspect_vl_raw.py
```

### Inspect VL result keys

```powershell
python tests/inspect_keys.py
```

### Simple PaddleOCR-VL smoke test

```powershell
python tests/test_vl.py
```

### Layout-enabled VL test

```powershell
python tests/test_vl_layout.py
```

### Batch-run all local sample images

```powershell
python tests/batch_verify_user_images.py
```

### Profile PaddleOCR-VL on one image

```powershell
python tests/profile_paddle_vl.py
```

Example:

```powershell
python tests/profile_paddle_vl.py ^
  --image "testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg" ^
  --layout ^
  --image-block-ocr ^
  --max-side 1600 ^
  --output tests/profile_paddle_vl_output.txt ^
  --summary tests/profile_paddle_vl_summary.json
```

### Profile PaddleOCR-VL on selected ROI crops

```powershell
python tests/profile_paddle_vl_roi.py ^
  --image "testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg" ^
  --result-json output/result_glm_ready.json ^
  --fields type,owner_surname,owner_name,serial_no ^
  --max-side 1600 ^
  --summary tests/profile_paddle_vl_roi_summary.json ^
  --output-dir tests/profile_paddle_vl_roi_outputs
```

---

## 5) General Help Pattern

For any argparse-based script:

```powershell
python <script_path> --help
```

Examples:

```powershell
python scripts/anchor_label_probe.py --help
python scripts/collect_anchor_templates.py --help
python scripts/debug_call_trace.py --help
```

---

## 6) Recommended First Commands

```powershell
python -m ocrreader.cli --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" --config "config/ruhsat_schema_paddle_v29.yaml" --output "output/result.json"
python scripts/anchor_label_probe.py --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" --config config/ruhsat_schema_paddle_v29.yaml
python tests/batch_verify_user_images.py
python tests/profile_paddle_vl.py
python -m ocrreader.api --host 0.0.0.0 --port 8765 --config config/ruhsat_schema_paddle_v29_gpu_lazy.yaml
```
