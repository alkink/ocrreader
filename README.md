# OCR Reader - Ruhsat Extraction

This repository is focused on OCR extraction for Turkish vehicle registration documents.

The old dataset generation, annotation, training, and benchmark pipeline has been removed. The repo now keeps only the runtime OCR pipeline and a small set of local helper scripts.

## Why fixed ROI fails

Even if the document border is aligned correctly, printer drift can move the printed values inside the card. Because of that, fixed crop coordinates alone are not reliable.

The current pipeline uses:

1. document normalization,
2. anchor detection,
3. dynamic ROI resolution,
4. field-level OCR,
5. text cleanup and JSON output.

## Project structure

- `ocrreader/config.py` - typed config loader
- `ocrreader/preprocess.py` - document detection, warp, deskew
- `ocrreader/ocr_engine.py` - OCR backends and optional GLM/VL fallback
- `ocrreader/anchors.py` - anchor matching
- `ocrreader/fields.py` - ROI resolution and field extraction
- `ocrreader/pipeline.py` - end-to-end OCR pipeline
- `ocrreader/cli.py` - command-line entrypoint
- `config/` - extraction schemas
- `testdata/` - local sample images for runtime checks

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Windows + GPU + `PaddleOCR-VL` setup, use `INSTALL_WINDOWS_GPU.md`.

Tesseract must still be installed separately if your config uses it for crop-level OCR.

If Tesseract is not in `PATH`, set:

```yaml
ocr:
  executable: C:/Program Files/Tesseract-OCR/tesseract.exe
```

## Usage

### Fast runtime - `HybridOCREngine`

Uses `PaddleOCR` for full-page word detection and `Tesseract` for per-field crop OCR.
The default CLI output is now a flat JSON with only final field values.

```bash
python -m ocrreader.cli \
  --image "testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg" \
  --config "config/ruhsat_schema_paddle_v29.yaml" \
  --output "output/result_fast.json" \
  --debug-dir "output/debug_fast"
```

### Slow runtime - `PaddleOCR-VL`

Uses the experimental `PaddleOCR-VL` fallback path. This is the slower full-page VL route.

```bash
python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema_paddle_v29_allfields_glm.yaml" \
  --output "output/result_vl.json" \
  --debug-dir "output/debug_vl"
```

### Basic runtime - `Tesseract`

Uses the plain `Tesseract` pipeline without `PaddleOCR`.

```bash
python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema.yaml" \
  --output "output/result_tesseract.json" \
  --debug-dir "output/debug_tesseract"
```

### Runtime notes

- `config/ruhsat_schema_paddle_v29.yaml` selects the faster hybrid path.
- `config/ruhsat_schema_paddle_v29_allfields_glm.yaml` selects the slower `PaddleOCR-VL` path.
- `config/ruhsat_schema.yaml` is the simplest fallback path.
- `PaddleOCR`-based commands require the matching `paddleocr` and `paddlepaddle` packages to be installed in the active environment.
- Use `--full-output` only when you want ROI, anchor, and pipeline metadata in the JSON output.

## Useful helper scripts

- `scripts/anchor_label_probe.py` - inspect anchor hits on one image
- `scripts/collect_anchor_templates.py` - build anchor templates from local probe outputs
- `scripts/debug_call_trace.py` - trace the runtime OCR flow on one image
- `scripts/export_results_columns.py` - flatten OCR JSON files into CSV
- `tests/batch_verify_user_images.py` - run the VL helper on all local `testdata/` images
- `tests/profile_paddle_vl.py` - time PaddleOCR-VL on one image
- `tests/profile_paddle_vl_roi.py` - time PaddleOCR-VL on selected ROI crops

## Tuning checklist

1. Adjust `value_margin_norm` for weak fields.
2. Switch `value_from_anchor` between `right` and `below` where needed.
3. Use `force_method` only when the default field selection is unstable.
4. Increase normalized output size if field crops are still too small.

## Notes

- The schema remains configurable so field positions can be tuned without changing pipeline code.
- `testdata/` is the only built-in image workspace kept in the repo.
- For command examples, use `SCRIPT_RUNBOOK.md`.
