# Production Handoff

Bu repo, eski `old_hybrid` akışının yerine geçecek güncel ruhsat OCR teslimidir.

## Kullanılacak Modlar

### 1. Ana production modu

- Config: `config/ruhsat_schema_paddle_v29.yaml`
- Motor: `HybridOCREngine`
- Yapı:
  - tam sayfa OCR: `PaddleOCR PP-OCRv5`
  - alan crop OCR: `Tesseract`
- Bu mod şu an en dengeli seçenektir.

CLI örneği:

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29.yaml" ^
  --runtime-info ^
  --output "output/result_runtime.json"
```

### 2. Yavaş ama opsiyonel mod

- Config: `config/ruhsat_schema_paddle_v29_allfields_glm.yaml`
- Amaç: `PaddleOCR-VL` tabanlı ek kurtarma / alternatif tam sayfa okuma
- Not: production ana akışı için önerilen mod bu değildir; daha pahalıdır.

CLI örneği:

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29_allfields_glm.yaml" ^
  --runtime-info ^
  --output "output/result_vl_runtime.json"
```

## API Entegrasyonu

Backend tarafının bağlanacağı servis:

```powershell
python -m ocrreader.api ^
  --host 0.0.0.0 ^
  --port 8765 ^
  --config config/ruhsat_schema_paddle_v29_gpu_lazy.yaml ^
  --debug-root output/api_debug
```

Endpointler:

- `GET /health`
- `GET /metrics`
- `POST /ocr`

Örnek istek:

```powershell
curl.exe -X POST ^
  -F "image=@testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  -F "full_output=false" ^
  -F "save_debug=false" ^
  http://127.0.0.1:8765/ocr
```

`POST /ocr` cevabı:

- `request_id`
- `latency_ms`
- `debug_dir`
- `result`

`full_output=false` ise `result` sade alan sözlüğü döner.  
`full_output=true` ise tüm pipeline çıktısı döner.

## Eski Sürüme Göre Ne Değişti

### OCR ve runtime

- Ana OCR yolu `PP-OCRv5 + Tesseract crop OCR` olacak şekilde güçlendirildi.
- `engine: auto` ile uygun makinede Paddle otomatik seçiliyor.
- CLI ve API artık runtime metadata dönebiliyor.

### Alan çıkarma kalitesi

- `owner_name / owner_surname / owner_title` için kişi-şirket ayrımı güçlendirildi.
- `type / vehicle_type` ayrımı ve rescue kuralları geliştirildi.
- `engine_no` ve `chassis_no` temizleme/normalizasyonu iyileştirildi.
- `serial_no / document_number` türetme ve eşleme mantığı eklendi.
- tarih alanlarında normalize davranış iyileştirildi.
- boş olması beklenen bazı alanlarda false-positive suppression eklendi.

### Fotoğraf profilleri ve review

- Zor / bozuk fotoğraflar için profile bazlı route davranışı eklendi.
- `review.py` ile `needs_review`, `review_score`, `critical_error_fields` gibi karar katmanı eklendi.
- Production entegrasyonunda backend tarafı mümkünse `review.needs_review` alanını dikkate almalıdır.

### VL tarafı

- `ocrreader/full_page_vl_parser.py` ile tam sayfa VL parse katmanı eklendi.
- Bu katman opsiyonel ve daha yavaştır; ana production akışı değildir.

## Backend Tarafının Bilmesi Gereken Minimum Şeyler

1. Ana config `config/ruhsat_schema_paddle_v29.yaml`
2. Servis giriş noktası `ocrreader.api`
3. OCR sonucu sadece alanlardan ibaret değildir; güvenlik için `review` bilgisi de değerlidir
4. `needs_review=true` ise belgeyi manuel review kuyruğuna atmak daha güvenlidir

## Repo İçinde Bilinmesi Gereken Ana Dosyalar

- `ocrreader/api.py`
- `ocrreader/cli.py`
- `ocrreader/pipeline.py`
- `ocrreader/fields.py`
- `ocrreader/field_postprocess.py`
- `ocrreader/review.py`
- `ocrreader/full_page_vl_parser.py`
- `config/ruhsat_schema_paddle_v29.yaml`
- `config/ruhsat_schema_paddle_v29_allfields_glm.yaml`

## Karar

Teslim için önerilen varsayılan:

- API config: `config/ruhsat_schema_paddle_v29_gpu_lazy.yaml`
- Ana çalışma modu: `config/ruhsat_schema_paddle_v29.yaml`
- `allfields_glm` sadece gerektiğinde ikinci seçenek olarak kalsın
