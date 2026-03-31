## Test Run Notes

### 2026-03-10 - CLI output simplification

- `ocrreader.cli` varsayilan cikti formati sadeleştirildi.
- Artik CLI varsayilan olarak sadece son alan degerlerini yazar.
- ROI, anchor ve pipeline metadata gerekirse `--full-output` ile alinabilir.

### 2026-03-10 - `WhatsApp_Image_2026-03-03_at_18.31.01.jpeg`

- Canli test `ocrreader.pipeline_v31` ile calistirildi.
- Config olarak `config/ruhsat_schema.yaml` kullanildi.
- Cikti dosyasi: `output/live_whatsapp_underscore_v31.json`
- Debug goruntuleri:
  - `output/live_debug_whatsapp_underscore/normalized.png`
  - `output/live_debug_whatsapp_underscore/overlay.png`

#### Ozet

- Dogruya yakin cikan alanlar:
  - `plate`: `26ACS145`
  - `engine_no`: `HSHB470D447941`
  - `tax_or_id_no`: `12143472262`
  - `brand`: `RENAULT`
  - `serial_no`: `191377`
- Hatali veya bos cikan alanlar:
  - `chassis_no`: bos
  - `type`: `SINT`
  - `model_year`: `2030`
  - `owner_surname`: bos
  - `owner_name`: `ORBAY CAD NO`
  - `first_registration_date`: bos
  - `registration_date`: bos
  - `inspection_date`: bos

#### Teknik Not

- Bu kosuda `document_quad` tum goruntu cercevesine dustu, yani belge siniri net bulunamadi ve pipeline tam perspektif duzeltme yapamadi.
- Bu durum ozellikle tarih, ad-soyad ve sase no alanlarini bozuyor.
