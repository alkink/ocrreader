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
- `config/ruhsat_schema_paddle_v29.yaml` selects the faster hybrid path.
- `config/ruhsat_schema_paddle_v29_allfields_glm.yaml` selects the slower `PaddleOCR-VL` path.
- `config/ruhsat_schema.yaml` selects the simplest fallback path.
- `--image` is required.
- `--config` defaults to `config/ruhsat_schema.yaml`.
- `--output` is optional.
- `--debug-dir` is optional.

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

### Verify the sample image with integrated VL fallback

```powershell
python tests/verify_user_image.py
```

### Run the same helper on the sample image with direct console output

```powershell
python tests/verify_integrated_vl.py
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
```
