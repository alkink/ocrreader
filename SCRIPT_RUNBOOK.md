# Script Runbook

This file lists the main runnable commands in this repository, in English, with copy-paste examples.

## Before You Start

- Activate your environment first.
- Install Python dependencies from `requirements.txt`.
- If you use the current Paddle + GPU flow, also follow `INSTALL_WINDOWS_GPU.md`.

Example:

```powershell
conda activate ocrreader
python -m pip install -r requirements.txt
```

---

## 1) Main OCR CLI

### Run the OCR pipeline on one image

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29.yaml" ^
  --output "output/result.json" ^
  --debug-dir "output/debug"
```

Notes:
- `--image` is required.
- `--config` defaults to `config/ruhsat_schema.yaml`.
- `--output` is optional.
- `--debug-dir` is optional.

---

## 2) Dataset Generation

### Generate OCR dataset with Mistral OCR API

Requires `MISTRAL_API_KEY`.

```powershell
set MISTRAL_API_KEY=your_key_here
python scripts/generate_mistral_ocr_dataset.py ^
  --photo-dir dataset/photo ^
  --out-dir dataset/generated ^
  --model mistral-ocr-latest ^
  --resume
```

### Check generated dataset consistency

```powershell
python scripts/check_generated_dataset.py ^
  --generated-dir dataset/generated
```

### Build trainable annotations from generated data

```powershell
python scripts/build_training_dataset.py ^
  --generated-dir dataset/generated ^
  --output-csv dataset/generated/train_annotations.csv ^
  --output-jsonl dataset/generated/train_annotations.jsonl
```

### Create train/validation split files

```powershell
python scripts/train_textfield_model.py ^
  --annotations dataset/generated/train_annotations.csv ^
  --out-dir dataset/generated/splits ^
  --val-ratio 0.2 ^
  --seed 42
```

---

## 3) Benchmark and QA

### Run field-level benchmark

```powershell
python scripts/benchmark_pipeline.py ^
  --annotations dataset/generated/train_annotations.csv ^
  --config config/ruhsat_schema_paddle_v29.yaml ^
  --pipeline-module ocrreader.pipeline ^
  --output-dir dataset/generated/qa/benchmark ^
  --workers 1
```

Useful optional flags:
- `--max-images 10`
- `--progress-every 5`
- `--reviewed-only`
- `--review-statuses ok,fixed`
- `--exclude-fields owner_name,owner_surname`
- `--anchor-debug`

### Auto-patch Tesseract path and run benchmark

```powershell
python scripts/setup_and_run_benchmark.py ^
  --config config/ruhsat_schema_paddle_v29.yaml ^
  --annotations dataset/generated/train_annotations.csv ^
  --output-dir dataset/generated/qa/benchmark_tuned_v1
```

Optional:

```powershell
python scripts/setup_and_run_benchmark.py ^
  --config config/ruhsat_schema_paddle_v29.yaml ^
  --annotations dataset/generated/train_annotations.csv ^
  --output-dir dataset/generated/qa/benchmark_tuned_v1 ^
  --tesseract "C:/Program Files/Tesseract-OCR/tesseract.exe" ^
  --python "C:/Users/alkin/miniconda3/envs/ocrreader/python.exe"
```

### Report suspicious annotation rows

```powershell
python scripts/report_dataset_issues.py ^
  --annotations dataset/generated/train_annotations.csv ^
  --out-dir dataset/generated/qa
```

### Audit orientation impact

```powershell
python scripts/orientation_impact_audit.py ^
  --annotations dataset/generated/train_annotations.csv ^
  --config config/ruhsat_schema_paddle_v29.yaml ^
  --output-dir dataset/generated/qa/orientation_audit ^
  --key-fields plate,brand,first_registration_date,registration_date
```

---

## 4) Annotation Review / Merge Tools

### Build manual review queue

```powershell
python scripts/build_manual_review_queue.py ^
  --qa-dir dataset/generated/qa ^
  --out dataset/generated/qa/latest_dataset_manual_review.csv ^
  --low-score-threshold 160
```

### Merge latest dataset into train annotations

```powershell
python scripts/merge_latest_dataset.py ^
  --base dataset/generated/train_annotations.csv ^
  --latest dataset/generated/qa/latest_dataset.csv ^
  --output dataset/generated/train_annotations_merged_latest.csv ^
  --qa-dir dataset/generated/qa ^
  --min-apply-score 80
```

### Clean annotations and build gold annotations

```powershell
python scripts/clean_annotations.py ^
  --base dataset/generated/train_annotations.csv ^
  --review dataset/generated/qa/latest_dataset_manual_review.csv ^
  --output dataset/generated/qa/gold_annotations.csv ^
  --conflict-report dataset/generated/qa/conflict_report.csv
```

### Create gold subset template

```powershell
python scripts/create_gold_subset_template.py ^
  --annotations dataset/generated/train_annotations.csv ^
  --output dataset/generated/qa/gold_subset_template.csv ^
  --max-rows 30
```

### Prepare v27 review-status exclusions

```powershell
python scripts/prepare_v27_review_exclusions.py ^
  --input dataset/generated/train_annotations_v2_corrected.csv ^
  --output dataset/generated/train_annotations_v2_corrected_v27_excluded.csv
```

---

## 5) Anchor / Template Debugging

### Probe anchor label hits on one image

```powershell
python scripts/anchor_label_probe.py ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config config/ruhsat_schema_paddle_v29.yaml ^
  --out-csv dataset/generated/qa/anchor_label_probe.csv ^
  --psm 6
```

### Collect anchor templates

Auto mode:

```powershell
python scripts/collect_anchor_templates.py ^
  --mode auto ^
  --probe-dir dataset/generated/qa ^
  --image-dir dataset/photo ^
  --out-dir config/anchor_templates ^
  --min-conf 55
```

Manual mode:

```powershell
python scripts/collect_anchor_templates.py ^
  --mode manual ^
  --image "dataset/photo/example.jpg" ^
  --out-dir config/anchor_templates
```

### Run nested OCR call trace on one image

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

## 6) Export Utilities

### Export per-image OCR JSON files to one CSV

```powershell
python scripts/export_results_columns.py ^
  --input-dir output ^
  --output output/results_columns.csv
```

---

## 7) PaddleOCR-VL / Experimental Helpers

These are mostly investigation scripts, not the main production path.

### Verify one user image with integrated VL fallback

```powershell
python tests/verify_user_image.py
```

Output:
- `tests/user_image_full_res.txt`

### Verify integrated VL on a hardcoded dataset image

```powershell
python tests/verify_integrated_vl.py
```

### Inspect raw VL output structure

```powershell
python tests/inspect_vl_raw.py
```

Output:
- `tests/vl_raw_res.json`

### Inspect VL result keys / object layout

```powershell
python tests/inspect_keys.py
```

Output:
- `tests/vl_inspection_result.json`

### Simple PaddleOCR-VL smoke test

```powershell
python tests/test_vl.py
```

### VL test with layout enabled

```powershell
python tests/test_vl_layout.py
```

### Batch-run all images under `testdata`

```powershell
python tests/batch_verify_user_images.py
```

Outputs:
- `tests/batch_user_image_results/*.txt`
- `tests/batch_user_image_results/timing_summary.json`

### Profile PaddleOCR-VL on one image

Default run:

```powershell
python tests/profile_paddle_vl.py
```

Example with explicit options:

```powershell
python tests/profile_paddle_vl.py ^
  --image "testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg" ^
  --layout ^
  --image-block-ocr ^
  --max-side 1600 ^
  --output tests/profile_paddle_vl_output.txt ^
  --summary tests/profile_paddle_vl_summary.json
```

### Profile PaddleOCR-VL on selected ROIs only

```powershell
python tests/profile_paddle_vl_roi.py ^
  --image "testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg" ^
  --result-json output/result_glm_ready.json ^
  --fields type,owner_surname,owner_name,serial_no ^
  --max-side 1600 ^
  --summary tests/profile_paddle_vl_roi_summary.json ^
  --output-dir tests/profile_paddle_vl_roi_outputs
```

### Analyze GLM fallback rows inside benchmark detail output

```powershell
python tests/check_glm.py
```

Output:
- `dataset/generated/qa/benchmark/glm_analysis.txt`

---

## 8) PowerShell Speed Test

This script compares multiple benchmark modes by rewriting the config between runs.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_speed_test.ps1
```

Warning:
- This script modifies `config/ruhsat_schema_paddle_v29.yaml` during the test.
- Run it only if you are okay with that temporary config mutation.

---

## 9) General Help Pattern

For any argparse-based script, you can inspect available flags with:

```powershell
python <script_path> --help
```

Examples:

```powershell
python scripts/benchmark_pipeline.py --help
python scripts/anchor_label_probe.py --help
python scripts/debug_call_trace.py --help
```

---

## 10) Recommended First Commands

If you just want the most useful commands first, start with these:

```powershell
python -m ocrreader.cli --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" --config "config/ruhsat_schema_paddle_v29.yaml" --output "output/result.json"
python scripts/benchmark_pipeline.py --config config/ruhsat_schema_paddle_v29.yaml --output-dir dataset/generated/qa/benchmark
python scripts/anchor_label_probe.py --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" --config config/ruhsat_schema_paddle_v29.yaml
python tests/batch_verify_user_images.py
```
