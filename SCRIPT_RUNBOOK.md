# Script Runbook

Bu dosya production kullanımı için bırakılan minimum komutları içerir.

## Kurulum

```powershell
conda activate ocrreader
python -m pip install -r requirements.txt
```

Yeni Windows makinede hızlı bootstrap:

```powershell
python scripts/bootstrap_runtime.py
```

Python yoksa veya desteklenen sürüm kurulmamışsa:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

## 1. Ana CLI

### Production önerisi: Hybrid

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29.yaml" ^
  --runtime-info ^
  --output "output/result_runtime.json"
```

### Opsiyonel: Full VL

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29_allfields_glm.yaml" ^
  --runtime-info ^
  --output "output/result_vl_runtime.json"
```

Not:

- `config/ruhsat_schema_paddle_v29.yaml` ana production config'idir.
- `config/ruhsat_schema_paddle_v29_allfields_glm.yaml` daha yavaş, opsiyonel kurtarma yoludur.

## 2. Warm API

Önerilen servis:

```powershell
python -m ocrreader.api ^
  --host 0.0.0.0 ^
  --port 8765 ^
  --config config/ruhsat_schema_paddle_v29_gpu_lazy.yaml ^
  --debug-root output/api_debug
```

Sağlık kontrolü:

```powershell
curl.exe http://127.0.0.1:8765/health
```

OCR isteği:

```powershell
curl.exe -X POST ^
  -F "image=@testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  -F "full_output=false" ^
  -F "save_debug=false" ^
  http://127.0.0.1:8765/ocr
```

## 3. Önemli Notlar

- `POST /ocr` varsayılan olarak sade `result` döner.
- `full_output=true` verilirse tam pipeline çıktısı döner.
- Production için mümkünse `review.needs_review` bilgisi de değerlendirilmelidir.
- Ana teslim notları için ayrıca `docs/PRODUCTION_HANDOFF.md` dosyasına bakılmalıdır.
