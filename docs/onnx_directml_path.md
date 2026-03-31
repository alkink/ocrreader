# ONNX + DirectML Path

This is the path we validated on March 31, 2026.

## Working architecture

- `WSL / Ubuntu 24.04 / Python 3.12` is used for model export
- `Windows host` is used for `ONNX Runtime + DirectML` inference
- `DirectML` stays on the Windows side

Important:
- use WSL to generate the ONNX files
- use Windows to load those ONNX files with `onnxruntime-directml`
- do not plan on running the final `DirectML` inference inside WSL

## Why this split works

The Linux side succeeded with the official Paddle export chain:
- `paddlex --install paddle2onnx -y`
- `paddlex --paddle2onnx --paddle_model_dir ... --onnx_model_dir ...`

The Windows side succeeded with the exported OCR models:
- both detection and recognition ONNX models load with `DmlExecutionProvider`
- dummy inference runs complete successfully

The repo now also contains a working integrated OCR engine:
- config: `config/ruhsat_schema_onnx_directml_v29.yaml`
- page OCR: `OnnxDirectMlEngine`
- crop OCR: `TesseractEngine`
- wrapper: `HybridOCREngine`

## Export flow used here

WSL environment:
- distro: `Ubuntu-24.04`
- conda env: `ocrreader-onnx`
- Python: `3.12`

Packages installed in WSL:
- `paddlepaddle`
- `paddleocr`
- `paddlex`
- `onnx`
- `paddle2onnx`

Commands used:

```bash
~/miniconda3/bin/conda create -y -n ocrreader-onnx python=3.12 pip
~/miniconda3/bin/conda run -n ocrreader-onnx python -m pip install paddlepaddle paddleocr paddlex onnx
~/miniconda3/bin/conda run -n ocrreader-onnx python -m paddlex --install paddle2onnx -y
```

Download the PP-OCRv4 models once:

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
~/miniconda3/bin/conda run --no-capture-output -n ocrreader-onnx python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(
    lang='en',
    ocr_version='PP-OCRv4',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device='cpu',
)
PY
```

Export to ONNX:

```bash
mkdir -p ~/projects/ocrreader/models/onnx/ppocrv4_mobile_det
mkdir -p ~/projects/ocrreader/models/onnx/en_ppocrv4_mobile_rec

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
~/miniconda3/bin/conda run --no-capture-output -n ocrreader-onnx \
  python -m paddlex --paddle2onnx \
  --paddle_model_dir ~/.paddlex/official_models/PP-OCRv4_mobile_det \
  --onnx_model_dir ~/projects/ocrreader/models/onnx/ppocrv4_mobile_det

~/miniconda3/bin/conda run --no-capture-output -n ocrreader-onnx \
  python -m paddlex --paddle2onnx \
  --paddle_model_dir ~/.paddlex/official_models/en_PP-OCRv4_mobile_rec \
  --onnx_model_dir ~/projects/ocrreader/models/onnx/en_ppocrv4_mobile_rec
```

## Windows validation flow

Copy the exported models from the WSL repo if needed, then run:

```powershell
python -m scripts.probe_directml_ocr_models
```

Expected outputs:
- `output/directml_ocr_model_probe.json`
- `models/onnx/ppocrv4_mobile_det/inference.onnx`
- `models/onnx/en_ppocrv4_mobile_rec/inference.onnx`

## End-to-end benchmark used here

Profile command:

```powershell
python -m scripts.profile_runtime_mode ^
  --label onnx_directml ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config config/ruhsat_schema_onnx_directml_v29.yaml ^
  --runs 2 ^
  --summary-out output/onnx_directml_profile.json
```

Latest measured result on March 31, 2026:
- init: `3.326 s`
- run 1: `7.519 s`
- warm run 2: `3.877 s`
- extracted non-empty fields: `12 / 13`
- current gap on this sample: `brand` is still empty, same as the fast Paddle baseline

Output:
- `output/onnx_directml_profile.json`

## Warm API command

Run the Windows API service with the ONNX config:

```powershell
python -m ocrreader.api ^
  --host 0.0.0.0 ^
  --port 8766 ^
  --config config/ruhsat_schema_onnx_directml_v29.yaml ^
  --debug-root output/api_onnx_debug
```

Latest measured API result on March 31, 2026:
- health reports `OnnxDirectMlEngine` with `DmlExecutionProvider`
- request 1: `4628.3 ms`
- warm request 2: `4065.8 ms`
- `serial_no` is recovered as `857563`

Outputs:
- `output/api_onnx_health.json`
- `output/api_onnx_result_1.json`
- `output/api_onnx_result_2.json`
- `output/api_onnx_metrics.json`

## Practical recommendation

If the target machine is a Windows VPS with a non-CUDA GPU:
- keep export on WSL/Linux
- keep runtime on Windows DirectML

If the target machine is a Windows VPS with NVIDIA + CUDA:
- the current warm Paddle path is still the fastest verified end-to-end path
