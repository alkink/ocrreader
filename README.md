# OCR Reader - Ruhsat Extraction (Modular)

This project implements a **modular OCR pipeline** for Turkish vehicle registration documents (ruhsat).

## Why fixed ROI fails

Even if the document border is perfectly aligned, printed values can still shift because of printer drift.
So, hardcoded crop boxes alone are fragile.

## Proposed architecture

1. **Document preprocessing**
   - Detect full document quadrilateral
   - Perspective normalize (warp)
   - Optional deskew

2. **Anchor detection**
   - Run OCR once on whole normalized document
   - Detect static labels (anchors) such as `PLAKA`, `MARKASI`, `MOTOR NO`

3. **Dynamic field ROI mapping**
   - Compute field regions from anchor positions
   - Use fallback normalized ROIs if an anchor is missing

4. **Field OCR + cleanup**
   - OCR each field ROI with field-specific PSM
   - Postprocess text (`digits`, `alnum_upper`, etc.)

5. **JSON output + debug artifacts**
   - Final extracted data in JSON
   - Optional debug images with anchor/ROI overlays

---

## Project structure

- `ocrreader/config.py` - schema/config loader
- `ocrreader/preprocess.py` - document detection + perspective + deskew
- `ocrreader/ocr_engine.py` - Tesseract OCR wrapper
- `ocrreader/anchors.py` - anchor matching logic
- `ocrreader/fields.py` - ROI resolution + field extraction
- `ocrreader/pipeline.py` - end-to-end pipeline orchestration
- `ocrreader/cli.py` - command-line entrypoint
- `config/ruhsat_schema.yaml` - editable extraction schema

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Windows + GPU + `PaddleOCR-VL` setup, see `INSTALL_WINDOWS_GPU.md`.

Install Tesseract OCR separately and ensure it is in PATH.

If not in PATH, set in YAML: `ocr.executable: "C:/Program Files/Tesseract-OCR/tesseract.exe"`.

---

## Usage

```bash
python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema.yaml" \
  --output "output/result.json" \
  --debug-dir "output/debug"
```

### Conda quick start (Windows)

```bash
conda create -n ocrreader python=3.11 -y
conda run -n ocrreader python -m pip install -r requirements.txt
conda install -n ocrreader -c conda-forge tesseract -y
conda run -n ocrreader python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema.yaml" \
  --output "output/result.json" \
  --debug-dir "output/debug"
```

### Current extraction status on sample

With the current schema and code, the sample now extracts reliably for:
- plate
- brand
- model_year
- engine_no
- chassis_no
- tax_or_id_no

Still weak/noisy for this sample image quality:
- type
- owner_surname
- owner_name

Those 3 fields are mostly OCR quality limits on this specific image crop/noise, not document-alignment failure.

### Practical tuning checklist

1. For each weak field in [`config/ruhsat_schema.yaml`](config/ruhsat_schema.yaml), try:
   - `value_margin_norm` narrowing/widening by ±0.01..0.03
   - switch `value_from_anchor` between `right` / `below`
   - adjust `force_method` among `anchor_*`, `roi_tesseract_raw`, `roi_tesseract_preprocessed`
2. For alphanumeric fields that include hyphen, use `cleanup: alnum_hyphen_upper`.
3. If OCR still misses, increase normalized output in pipeline (`output_width`/`output_height`) and retest.
4. For production-level robustness, add text detection (CRAFT/DBNet) before recognition and then key-value pairing.

---

## Notes

- The schema is intentionally configurable so you can tune anchors/offsets without changing code.
- Start with current offsets, then calibrate per card layout and camera distance.
- If you later want stronger robustness, add text detection (CRAFT/DBNet) before recognition.

---

## Dataset generation with Mistral OCR

### 1) Generate OCR outputs from images

Input folder expected: [`dataset/photo`](dataset/photo)

Run:

```bash
python -m pip install mistralai
python scripts/generate_mistral_ocr_dataset.py \
  --photo-dir dataset/photo \
  --out-dir dataset/generated \
  --resume
```

Required env var:

```bash
MISTRAL_API_KEY=...
```

Outputs:
- Raw API JSON files: [`dataset/generated/raw`](dataset/generated/raw)
- Markdown OCR files: [`dataset/generated/markdown`](dataset/generated/markdown)
- Manifest: [`dataset/generated/manifest.jsonl`](dataset/generated/manifest.jsonl)

### 2) Check dataset consistency

```bash
python scripts/check_generated_dataset.py --generated-dir dataset/generated
```

QA outputs:
- [`dataset/generated/qa/issues.csv`](dataset/generated/qa/issues.csv)
- [`dataset/generated/qa/coverage.csv`](dataset/generated/qa/coverage.csv)
- [`dataset/generated/qa/summary.json`](dataset/generated/qa/summary.json)

### 3) Build trainable annotations

```bash
python scripts/build_training_dataset.py \
  --generated-dir dataset/generated \
  --output-csv dataset/generated/train_annotations.csv \
  --output-jsonl dataset/generated/train_annotations.jsonl
```

Then create train/val splits:

```bash
python scripts/train_textfield_model.py \
  --annotations dataset/generated/train_annotations.csv \
  --out-dir dataset/generated/splits \
  --val-ratio 0.2 \
  --seed 42
```

Split outputs:
- [`dataset/generated/splits/train.csv`](dataset/generated/splits/train.csv)
- [`dataset/generated/splits/val.csv`](dataset/generated/splits/val.csv)
- [`dataset/generated/splits/split_summary.json`](dataset/generated/splits/split_summary.json)

### 4) Benchmark pipeline vs annotations (field-level metrics)

Run benchmark (uses OCR pipeline predictions vs `train_annotations.csv`):

```bash
python scripts/benchmark_pipeline.py \
  --annotations dataset/generated/train_annotations.csv \
  --config config/ruhsat_schema.yaml \
  --output-dir dataset/generated/qa/benchmark
```

If Tesseract is only available in conda env:

```bash
conda run -n ocrreader python scripts/benchmark_pipeline.py \
  --annotations dataset/generated/train_annotations.csv \
  --config config/ruhsat_schema.yaml \
  --output-dir dataset/generated/qa/benchmark
```

Benchmark outputs:
- [`dataset/generated/qa/benchmark/summary.json`](dataset/generated/qa/benchmark/summary.json)
- [`dataset/generated/qa/benchmark/field_metrics.csv`](dataset/generated/qa/benchmark/field_metrics.csv)
- [`dataset/generated/qa/benchmark/mismatches.csv`](dataset/generated/qa/benchmark/mismatches.csv)
- [`dataset/generated/qa/benchmark/predictions.csv`](dataset/generated/qa/benchmark/predictions.csv)

Suggested regression workflow:
1. Rebuild annotations after parser changes.
2. Run benchmark and compare `field_metrics.csv` with previous run.
3. Prioritize fixes by highest `fn` and lowest recall fields.

