# OCR Reader

Bu repo, ruhsat gorsellerinden alan cikarmak icin hazirlanmis OCR pipeline'idir.

Amac:
- kur
- komutu calistir
- sonucu al

Detayli benchmark, backend ve platform notlari icin:
- `SCRIPT_RUNBOOK.md`
- `docs/`

## En Kisa Kurulum

Repo kokunde:

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python scripts/bootstrap_runtime.py
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python scripts/bootstrap_runtime.py
```

Not:
- klasik OCR icin `Tesseract` kurulu olmali
- Windows + NVIDIA kullaniyorsan ayrica uygun `paddlepaddle-gpu` kurulumunu yap
- bunun icin `INSTALL_WINDOWS_GPU.md` dosyasina bak

## Tak Calistir

### 1. Klasik OCR

Cogu kullanici icin dogru baslangic budur.

```bash
python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema_paddle_v29.yaml" \
  --output "output/result.json"
```

Sonuc:
- `output/result.json`

### 2. Hangi backend secildigini de gormek istersen

```bash
python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema_paddle_v29.yaml" \
  --runtime-info \
  --output "output/result_runtime.json"
```

Bu cikti icinde sunlar gorunur:
- hangi engine secildi
- ONNX mi Paddle mi kullanildi
- provider ne oldu
- GPU inventory ne goruldu

## Apple Silicon Icin VL

Apple Silicon'da `PaddleOCR-VL` kullanacaksan once local MLX server ac:

Terminal 1:

```bash
source .venv/bin/activate
python scripts/start_mlx_vl_server.py --port 8111
```

Terminal 2:

```bash
source .venv/bin/activate
python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema_paddle_v29_allfields_glm.yaml" \
  --output "output/result_vl.json"
```

Runtime bilgisini de gormek istersen:

```bash
source .venv/bin/activate
python -m ocrreader.cli \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema_paddle_v29_allfields_glm.yaml" \
  --runtime-info \
  --output "output/result_vl_runtime.json"
```

## Dogrudan Hiz Olcmek Icin

Klasik OCR:

```bash
python scripts/profile_runtime_mode.py \
  --label auto \
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
  --config "config/ruhsat_schema_paddle_v29.yaml" \
  --runs 2 \
  --summary-out "output/auto_profile.json"
```

VL:

```bash
python tests/profile_paddle_vl.py \
  --image "testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg" \
  --layout \
  --image-block-ocr \
  --runtime-profile local_mlx_vlm_service \
  --service-url "http://localhost:8111/" \
  --service-model-name "PaddlePaddle/PaddleOCR-VL-1.5" \
  --max-side 1600 \
  --max-new-tokens 512 \
  --output "tests/profile_paddle_vl.txt" \
  --summary "tests/profile_paddle_vl.json"
```

## En Kisa Ozet

Cogu kisi icin sadece bunlar yeterli:

```bash
python scripts/bootstrap_runtime.py
python -m ocrreader.cli --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" --config "config/ruhsat_schema_paddle_v29.yaml" --output "output/result.json"
```

Apple Silicon + VL icin:

```bash
python scripts/start_mlx_vl_server.py --port 8111
python -m ocrreader.cli --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" --config "config/ruhsat_schema_paddle_v29_allfields_glm.yaml" --output "output/result_vl.json"
```
