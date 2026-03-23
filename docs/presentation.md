# OCR Reader — Türk Araç Ruhsatından Alan Çıkarma Sistemi
## Hocaya Kod Düzeyinde Sunum Dökümanı

---

## 1. PROJENİN AMACI VE PROBLEMI

Bu proje, **Türkçe araç ruhsatı** (taşıt tescil belgesi) görüntülerinden otomatik olarak belirli alanları (plaka, marka, şase no, motor no, vergi no, vs.) çıkarmak için geliştirilmiş bir **modüler OCR pipeline**'ıdır.

### Neden zor bir problem?

- Ruhsat fotoğrafları eğik, bozuk veya düşük çözünürlüklü olabilir.
- Yazıcı kayması nedeniyle aynı belgenin farklı baskılarında değerlerin konumu kayabilir.
- OCR motorları Türkçe karakterleri karıştırabilir (İ→I, Ş→S, Ö→O gibi).
- PaddleOCR bazen "MOTOR NO" → "MOTORNO" şeklinde kelimeleri birleştirir.
- Farklı araç markaları için motor no formatları tamamen farklıdır (Ford, Renault, Mercedes).

---

## 2. MİMARİ GENEL BAKIŞ

```
Ham Görüntü (JPEG/PNG)
        │
        ▼
┌─────────────────────┐
│  1. Preprocess      │  preprocess.py
│  - Oryantasyon düzelt│
│  - Perspektif düzelt │
│  - Eğim gider       │
└─────────┬───────────┘
          │ normalized_image (2200×1400 px)
          ▼
┌─────────────────────┐
│  2. OCR Engine      │  ocr_engine.py
│  - Tüm sayfayı tara │
│  - OCRWord listesi  │  → [{text, conf, bbox, block_num, ...}]
└─────────┬───────────┘
          │ words: list[OCRWord]
          ▼
┌─────────────────────┐
│  3. Anchor Detect   │  anchors.py + template_anchor_detector.py
│  - "PLAKA", "MARKASI"│
│  - Fuzzy matching   │
└─────────┬───────────┘
          │ anchors: dict[str, AnchorMatch]
          ▼
┌─────────────────────┐
│  4. ROI Resolve     │  fields.py
│  - Anchor + offset  │
│  - Fallback norm    │
└─────────┬───────────┘
          │ rois: dict[str, Rect]
          ▼
┌─────────────────────┐
│  5. Field Extract   │  fields.py
│  - Çoklu strateji   │
│  - Skor karşılaştır │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  6. Post-Process    │  field_postprocess.py + pipeline_v31.py
│  - v31 rescue       │
│  - VIN fix          │
└─────────┬───────────┘
          │
          ▼
    JSON Çıktısı
```

---

## 3. KATMAN KATMAN KOD ANALİZİ

---

### 3.1 Konfigürasyon Sistemi — `ocrreader/config.py`

Tüm parametreler **YAML dosyasından** okunur. Kod hiç değişmeden sadece YAML düzenleyerek davranış değiştirilebilir.

#### Veri Yapıları (dataclass ile):

```python
@dataclass(frozen=True)
class AnchorConfig:
    aliases: list[str]       # ["PLAKA", "PLAKA NO"]
    min_score: float = 0.72  # fuzzy matching eşiği
    search_region_norm: tuple[float, float, float, float] | None = None
    # (x, y, w, h) — 0.0-1.0 arası normalize koordinatlar
```

```python
@dataclass(frozen=True)
class FieldConfig:
    anchor: str | None = None              # hangi anchor'a bağlı
    offset_from_anchor_norm: ...           # anchor'dan offset (normalize)
    fallback_norm: ...                     # anchor bulunamazsa sabit ROI
    value_from_anchor: str = "below"       # "below" veya "right"
    force_method: str | None = None        # hangi extraction yöntemini zorla
    cleanup: str = "keep"                  # "plate", "digits", "date", "alnum_upper"...
    min_len: int = 0
    confidence_threshold: int = 0
    psm: int | None = None                 # Tesseract PSM modu
```

#### `load_config()` fonksiyonu:
YAML'ı okur, her bölümü (`pipeline`, `ocr`, `anchors`, `fields`) karşılık gelen dataclass'a dönüştürür. `_tuple4()` gibi yardımcılar tip güvenliği sağlar.

---

### 3.2 Görüntü Ön İşleme — `ocrreader/preprocess.py`

```python
def preprocess_document(image: np.ndarray, config: PipelineConfig) -> PreprocessResult:
    oriented = _choose_best_orientation(image)  # Adım 1
    enhanced = _enhance_low_contrast(oriented)  # Adım 2
    quad = detect_document_quad(enhanced, config)  # Adım 3
    normalized = _warp_perspective(enhanced, quad, ...)  # Adım 4
    # Adım 5: eğim düzeltme
    skew = estimate_skew_angle_deg(normalized)
    normalized = _rotate_bound(normalized, -clipped)
```

#### Adım 1 — Oryantasyon Tespiti (`_choose_best_orientation`):
- Görüntü dikey ise (h > w) 0°, 90°, 270° denenir
- Her açı için `HoughLinesP` ile yatay çizgi uzunluğu ölçülür
- En yüksek skor = en fazla yatay satır = doğru oryantasyon

#### Adım 2 — Kontrast İyileştirme (`_enhance_low_contrast`):
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) uygulanır
- LAB renk uzayında L kanalı işlenir
- Sonra `fastNlMeansDenoisingColored` ile gürültü azaltılır

#### Adım 3 — Dörtgen Tespiti (`detect_document_quad`):
İki aşamalı:
1. **Mavi HSV maskesi**: ruhsatın mavi rengi varsa HSV renk filtresi ile direkt bulunur
2. **Kenar tespiti**: `Canny` + kontur analizi ile en büyük dörtgen bulunur

```python
# Mavi maske (HSV aralığı)
blue_mask = cv2.inRange(hsv, (90, 40, 25), (145, 255, 255))
```

#### Adım 4 — Perspektif Dönüşümü:
```python
def _warp_perspective(image, quad, output_w, output_h):
    dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    matrix = cv2.getPerspectiveTransform(ordered_quad, dst)
    return cv2.warpPerspective(image, matrix, (output_w, output_h))
```
Belge çerçevesi her zaman 2200×1400 piksel boyutuna normalize edilir.

#### Adım 5 — Eğim Düzeltme:
- `HoughLinesP` ile düz çizgiler bulunur
- Açıların **medyanı** alınır (gürültüye dayanıklı)
- ±8° sınırı içindeyse rotasyon uygulanır

---

### 3.3 OCR Engine — `ocrreader/ocr_engine.py`

İki farklı OCR motorunu aynı arayüz üzerinden sunar:

```python
class OCREngine(Protocol):
    def iter_words(self, image, psm=None, min_conf=20.0) -> list[OCRWord]: ...
    def read_text(self, image, psm=None, whitelist=None) -> str: ...
```

#### Ortak Veri Yapısı:
```python
@dataclass(frozen=True)
class OCRWord:
    text: str
    conf: float      # güven skoru (0-100)
    bbox: Rect       # (x, y, w, h) piksel koordinatı
    block_num: int   # Tesseract blok numarası
    par_num: int     # paragraf numarası
    line_num: int    # satır numarası
```

#### Tesseract Engine:
```python
data = pytesseract.image_to_data(rgb, output_type=Output.DICT, config="--oem 3 --psm 6")
# Her kelime için OCRWord nesnesi oluşturulur
```

#### PaddleOCR Engine:
PaddleOCR PP-OCRv4/v5 kullanır. Karmaşık kısım: Paddle satır düzeyinde çıktı verir, kelime düzeyine dönüştürme gerekir:

```python
# Eğer text_word_boxes varsa: doğrudan kelime bboxları kullanılır
# Yoksa: satır bbox'u kelime sayısına orantılı bölünür
def _split_line_to_word_boxes(text, bbox) -> list[tuple[str, Rect]]:
    tokens = text.split()
    # Her token'a karakter sayısına orantılı genişlik atanır
```

Önemli mühendislik kararı: PP-OCRv5 Windows CPU'da crash yapabilir. Kod önce v4 dener, başarısız olursa v5'e geçer (veya tam tersi):
```python
self._candidate_versions = ("PP-OCRv4", "PP-OCRv5")
for version in self._candidate_versions:
    try:
        self._ocr = self._build_engine(version)
        break
    except Exception as exc:
        init_errors.append(exc)
```

---

### 3.4 Anchor Tespiti — `ocrreader/anchors.py`

"Anchor" = ruhsat üzerindeki sabit etiket metinleri (PLAKA, MARKASI, MOTOR NO, vb.)

#### Temel Benzerlik Fonksiyonu:
```python
def _token_similarity(a: str, b: str) -> float:
    seq_score = SequenceMatcher(None, a, b).ratio()  # difflib
    dist = _levenshtein(a, b)  # Levenshtein mesafesi
    lev_score = max(0.0, 1.0 - (dist / max_len))
    return max(seq_score, lev_score)  # ikisinin maksimumu
```

#### Arama Süreci:
1. Her anchor için YAML'daki `search_region_norm` ile dokümanda bir bölge tanımlanır
2. O bölgedeki OCR kelimeleri filtrelenir
3. Her alias için kayar pencere ile N-gram eşleştirme yapılır:

```python
for i in range(0, len(words) - n + 1):
    seq_words = words[i : i + n]
    # aynı satırda mı?
    line_keys = {(w.block_num, w.par_num, w.line_num) for w in seq_words}
    if len(line_keys) != 1:
        continue
    scores = [_token_similarity(seq_norm[j], alias_tokens[j]) for j in range(n)]
    score = sum(scores) / n
```

4. PaddleOCR birleşik token durumu ele alınır:
```python
# "MOTOR NO" → "MOTORNO" olabilir, concat kontrolü yapılır
alias_concat = "".join(alias_tokens)
merged = "".join(normalized[i + j] for j in range(win))
score = _token_similarity(merged, alias_concat)
```

---

### 3.5 Alan ROI Çözümleme — `ocrreader/fields.py` (`resolve_field_rois`)

Anchor bulunduysa:
```python
anchor_box = anchors[cfg.anchor].bbox
dx, dy, rw, rh = cfg.offset_from_anchor_norm
roi = Rect(
    x=anchor_box.x + int(dx * doc_w),
    y=anchor_box.y + int(dy * doc_h),
    w=max(1, int(rw * doc_w)),
    h=max(1, int(rh * doc_h)),
)
```

Anchor bulunamadıysa sabit `fallback_norm` koordinatları kullanılır.

**Neden normalize koordinatlar?** Belge her zaman 2200×1400'e normalize edildiğinden koordinatlar her görüntüde tutarlıdır.

---

### 3.6 Alan Çıkarma — `ocrreader/fields.py` (`extract_fields`)

Her alan için **5 farklı kaynak**tan aday üretilir ve en iyi skor kazanır:

| Öncelik | Yöntem | Açıklama |
|---------|--------|----------|
| 6 | semantic_plate_page_words | Plaka için tüm sayfada regex |
| 5 | semantic_owner_roi_words | Sahip adı için ROI kelime tarama |
| 4 | anchor_below_line / anchor_right_line | Anchor'ın altındaki/sağındaki satırı al |
| 3 | roi_words | ROI içindeki tüm kelimeleri birleştir |
| 2 | roi_tesseract_preprocessed | ROI kırpıp Tesseract çalıştır |
| 1 | page_regex | Tüm sayfa metninde regex |

`force_method` varsa o yöntem sonuçları başa alınır.

#### Skor Sistemi (`_score_cleaned`):
```python
if strategy == "plate":
    return len(compact) + (12 if PLATE_PATTERN.fullmatch(compact) else 0)
if strategy == "alnum_upper":
    has_alpha = bool(re.search(r"[A-Z]", cleaned))
    has_digit = bool(re.search(r"\d", cleaned))
    bonus = 8 if has_alpha and has_digit else 0
    return len(cleaned) + bonus
```

#### Temizleme Stratejileri (`cleanup_text`):
- `plate`: Türkçe plaka regex: `\d{2}[A-Z]{1,3}\d{2,4}`
- `digits`: Sadece rakamlar, en uzun grup
- `date`: DD/MM/YYYY formatına normalize
- `alnum_upper`: Harf+rakam karışımı, en uzun
- `alnum_hyphen_upper`: Alnum + tire
- `text_upper`: Büyük harf metin
- `owner_text`: İsim/soyad özel temizleme
- `vehicle_type`: Araç tipi özel temizleme

#### Post-filter (`_post_field_filters`):
Alan bazında ek kurallar:
```python
if field_name == "chassis_no":
    # VIN standardı: I→1, O→0 değişimi (8 konumundan sonra)
    if ch in {"O", "Q"}: chars[idx] = "0"
    elif ch == "I" and idx >= 3: chars[idx] = "1"
```

---

### 3.7 v31 Rescue Sistemi — `ocrreader/pipeline_v31.py`

Standart çıkarma boş döndürdüğünde devreye giren son çare mekanizmaları:

#### `_rescue_engine_no`:
```python
# Motor no'su her zaman sol yarıda, %35-%85 dikey aralığında
if cx > 0.52 or cy < 0.35 or cy > 0.85:
    continue

# Gerçek motor kodu regex'leri
_ENGINE_CODE_RE = re.compile(r"^(?:[A-Z]{1,3}\d{5,12}|K9K[A-Z0-9]{5,9})$")
# Label artifact filtresi (D3TICARIADI, Y2TESCIL gibi OCR hatalarını eler)
_ENGINE_LABEL_ARTIFACT_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z]{3,}")
```

Güç sıralaması (strength):
- K9K... (Renault) → 3
- XX12345 formatı → 2
- XXX12345 formatı → 1

#### `_rescue_serial_no`:
PaddleOCR'ın "Seri CHN664423" → "SeriCHN664423" birleştirme hatasını yakalar:
```python
_SERIAL_MERGED_RE = re.compile(r"(?:Seri?|Sen|Sec|Sc|Se)[A-Za-z]{0,6}(\d{4,8})", re.IGNORECASE)
# 3 geçiş: merged → suffix → pure digit
```

#### `_apply_chassis_vin_fix`:
VIN standardına göre karakter düzeltmeleri:
```python
# L8N → LBN: sayı ortasında 8 → B
if ch == "8" and prev.isalpha(): chars[i] = "B"
# VE1 → VF1: Renault VIN prefix düzeltmesi
if ch == "E" and i == 1 and chars[0] == "V": chars[i] = "F"
```

---

### 3.8 Ana Pipeline — `ocrreader/pipeline_v31.py` (`RuhsatOcrPipeline`)

```python
class RuhsatOcrPipeline:
    def process_path(self, image_path: str, ...) -> dict:
        image = imread_color(image_path)
        
        # 1. Ön işleme
        prep = preprocess_document(image, self.config.pipeline)
        document = prep.normalized_image
        
        # 2. OCR (tüm sayfa)
        words = self.engine.iter_words(document, psm=self.config.ocr.psm, min_conf=0.0)
        
        # 3. Anchor tespiti (hybrid: OCR + template matching)
        anchors = detect_anchors_hybrid(words, self.config.anchors, gray, self.template_detector)
        
        # 4. ROI çözümleme
        rois = resolve_field_rois(document.shape, self.config.fields, anchors)
        
        # 5. Alan çıkarma
        fields = extract_fields(document, rois, self.config.fields, self.engine, ...)
        
        # 6. Second pass (model_year, inspection_date gibi alanlar için)
        _apply_page_second_pass(fields, words, document.shape, rois)
        
        # 7. VIN düzeltme
        _apply_chassis_vin_fix(fields)
        
        # 8. v31 rescue (engine_no, serial_no)
        _apply_v31_rescues(fields, words, document.shape, serial_region)
        
        # 9. Son post-process
        fields = postprocess_fields(fields)
        
        return {
            "image": image_path,
            "pipeline": {...},   # meta: skew_angle, document_quad, vs.
            "anchors": {...},    # bulunan anchor'lar ve skorları
            "fields": fields,    # çıkarılan alanlar
        }
```

---

## 4. KONFİGÜRASYON SİSTEMİ — YAML Örneği

`config/ruhsat_schema_paddle_v29.yaml` dosyasından motor no konfigürasyonu:

```yaml
anchors:
  engine_no_label:
    aliases:
      - MOTOR NO
      - MOTORNO
      - M0T0R NO     # OCR 0/O karışıklığı için varyantlar
      - PSMOTORNO    # PaddleOCR "PS" prefix artifact'ı
    min_score: 0.5
    search_region_norm: [0.0, 0.32, 0.52, 0.28]  # x, y, w, h (normalize)

fields:
  engine_no:
    anchor: engine_no_label
    offset_from_anchor_norm: [0.0, 0.03, 0.22, 0.058]
    fallback_norm: [0.08, 0.44, 0.30, 0.065]
    value_from_anchor: below
    force_method: anchor_below_line
    cleanup: alnum_upper
    psm: 7
    confidence_threshold: 12
    strip_prefixes:
      - MOTOR NO
      - MOTORNO
```

---

## 5. DATASET VE BENCHMARK SİSTEMİ

### 5.1 Dataset Oluşturma Pipeline'ı:

```
Ham fotoğraflar (dataset/photo/)
         │
         ▼ scripts/generate_mistral_ocr_dataset.py
Mistral OCR API
         │
         ▼
dataset/generated/
├── raw/          ← API ham JSON çıktıları
├── markdown/     ← OCR markdown dosyaları
└── manifest.jsonl

         │
         ▼ scripts/build_training_dataset.py
train_annotations.csv  ← GT (ground truth) etiketler

         │
         ▼ scripts/train_textfield_model.py
splits/train.csv + splits/val.csv
```

### 5.2 Benchmark Metrik Sistemi (`scripts/benchmark_pipeline.py`):

Her alan için hesaplanan metrikler:
- **TP** (True Positive): GT=X, Pred=X
- **FP** (False Positive): GT=boş veya farklı, Pred=X
- **FN** (False Negative): GT=X, Pred=boş veya farklı
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2×TP / (2×TP + FP + FN)
- **Token F1**: Kelime düzeyinde kısmi eşleşme skoru

```python
def token_f1(gt: str, pred: str) -> float:
    gt_tokens = re.findall(r"[A-Z0-9]+", gt)
    pred_tokens = re.findall(r"[A-Z0-9]+", pred)
    overlap = sum(min(gt_c[t], pred_c[t]) for t in gt_c.keys() & pred_c.keys())
    p = overlap / sum(pred_c.values())
    r = overlap / sum(gt_c.values())
    return (2 * p * r) / (p + r)
```

### 5.3 Confidence Sweep:
Farklı güven eşiklerinde (0, 8, 12, 16, ...) precision-recall tradeoff'u ölçülür. Bu sayede üretimde ne kadar hassas/kapsamlı davranılacağı ayarlanabilir.

---

## 6. BAĞIMLILIKLAR VE TEKNOLOJİ YIĞINİ

```
requirements.txt:
├── opencv-python >= 4.8.0    # Görüntü işleme (Canny, Hough, warp, CLAHE)
├── numpy >= 1.24.0           # Matris işlemleri
├── pytesseract >= 0.3.10     # Tesseract OCR Python wrapper
├── paddleocr >= 2.8.0        # PaddleOCR PP-OCRv4/v5
├── paddlepaddle >= 3.0.0     # PaddleOCR backend
├── PyYAML >= 6.0.1           # Konfigürasyon okuma
├── pandas >= 2.0.0           # Veri analizi (benchmark)
└── mistralai >= 1.12.4       # Dataset oluşturma için Mistral OCR API
```

---

## 7. TEMEL MÜHENDİSLİK KARARLARI

### 7.1 Normalize Koordinatlar
Tüm anchor arama bölgeleri ve alan ROI'ları 0-1 arasında normalize koordinatlarla tanımlanır. Bu sayede:
- Görüntü çözünürlüğünden bağımsız
- Sadece YAML değiştirerek ince ayar yapılabilir
- Kod yeniden yazılmasına gerek yok

### 7.2 Çoklu Strateji + Skor Bazlı Seçim
Tek bir OCR yöntemi güvenilir değildir. Bunun yerine 5+ yöntem çalıştırılır ve en yüksek skorlu aday seçilir. `force_method` ile ise belirli alanlar için en güvenilir yöntem önceliklendirilir.

### 7.3 OCR Engine Soyutlaması
`OCREngine` Protocol tanımı sayesinde Tesseract ve PaddleOCR değiştirilebilir. Yeni bir motor eklemek için sadece Protocol'ü implement etmek yeterlidir.

### 7.4 v31 Rescue Katmanı
Standart çıkarma başarısız olduğunda devreye giren, alana özgü örüntü eşleştirme kuralları. Yüksek false positive riskini azaltmak için:
- Pozisyon kısıtlaması (sol yarı, belirli yükseklik aralığı)
- Bilinen label kelimelerini blacklist
- Belirli kod formatlarına regex match

### 7.5 Benchmark ile Gerileme Testi
Her değişiklik sonrası `benchmark_pipeline.py` koşulur, `field_metrics.csv` karşılaştırılır. Bu sayede bir alanda yapılan iyileştirmenin başka alanları bozup bozmadığı anında görülür.

---

## 8. CLI KULLANIMI

```bash
# Tekli görüntü işleme
python -m ocrreader.cli \
  --image "testdata/ruhsat.jpeg" \
  --config "config/ruhsat_schema_paddle_v29.yaml" \
  --output "output/result.json" \
  --debug-dir "output/debug"

# Benchmark (tüm dataset üzerinde metrik hesaplama)
python scripts/benchmark_pipeline.py \
  --annotations dataset/generated/train_annotations.csv \
  --config config/ruhsat_schema_paddle_v29.yaml \
  --output-dir dataset/generated/qa/benchmark_v31
```

---

## 9. ÇIKTI YAPISI

```json
{
  "image": "testdata/ruhsat.jpeg",
  "pipeline": {
    "output_width": 1600,
    "output_height": 1000,
    "skew_angle_deg": -1.2,
    "document_quad": [[15, 22], [1585, 18], [1590, 978], [12, 975]]
  },
  "anchors": {
    "plate_label": {
      "alias": "PLAKA",
      "score": 0.9143,
      "bbox": {"x": 45, "y": 112, "w": 68, "h": 24}
    }
  },
  "fields": {
    "plate": {
      "value": "34ABC123",
      "raw": "34 ABC 123",
      "roi": {"x": 45, "y": 124, "w": 304, "h": 75},
      "value_bbox": {"x": 58, "y": 126, "w": 180, "h": 22},
      "method": "anchor_below_line",
      "confidence_score": 24,
      "low_confidence": false
    }
  }
}
```

---

## 10. PROJE EVRİMİ (Versiyon Geçmişi)

| Versiyon | Yenilik |
|----------|---------|
| v1 | Temel Tesseract + sabit ROI |
| v28 | PaddleOCR desteği, VIN char fix |
| v29 | Anchor tabanlı ROI, hybrid detector |
| v31 | Engine no + serial no rescue, second pass, confidence sweep |

---

## 11. PROJE KLASÖRLERİ

```
ocrreader/
├── ocrreader/           ← Ana modül
│   ├── config.py        ← YAML → dataclass dönüşümü
│   ├── preprocess.py    ← Görüntü normalize
│   ├── ocr_engine.py    ← Tesseract + PaddleOCR wrapper
│   ├── anchors.py       ← Fuzzy anchor matching
│   ├── fields.py        ← ROI çözümleme + alan çıkarma
│   ├── pipeline_v31.py  ← Ana orkestratör + rescue
│   ├── field_postprocess.py  ← Son temizleme
│   ├── page_word_extractor.py  ← Second pass
│   └── types.py         ← Rect, vs. veri tipleri
├── config/
│   └── ruhsat_schema_paddle_v29.yaml  ← Tüm parametreler
├── scripts/
│   ├── benchmark_pipeline.py  ← Metrik hesaplama
│   └── generate_mistral_ocr_dataset.py  ← Dataset oluşturma
└── dataset/
    └── generated/
        ├── train_annotations.csv  ← Ground truth
        └── qa/benchmark*/         ← Benchmark sonuçları
```

---

## 12. CANLI DEMO (KOD ÜZERİNDE GEZEREK ANLATIM)

Bu bölüm, sunum sırasında birebir şu akışla anlatman için hazırlandı:

### 12.1 Başlangıç komutu (terminalde göster)

```bash
python -m ocrreader.cli \
  --image testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg \
  --config config/ruhsat_schema_paddle_v29.yaml \
  --output output/result.json
```

Bu komutu çalıştırdıktan sonra aşağıdaki sırayla kodda ilerle.

---

### 12.2 Adım adım fonksiyon akışı (hocaya anlatım metni)

#### Adım 1 — CLI giriş
- Dosya: `ocrreader/cli.py`
- Fonksiyon: `main()`
- Anlatım:
  - Argümanları parse ediyor.
  - `load_config()` ile YAML yükleniyor.
  - `RuhsatOcrPipeline(config)` oluşturuluyor.
  - `process_path()` çağrılıp sonuç JSON yazılıyor.

#### Adım 2 — Konfigürasyon yükleme
- Dosya: `ocrreader/config.py`
- Fonksiyon: `load_config()`
- Anlatım:
  - YAML içindeki `pipeline`, `ocr`, `anchors`, `fields`, `benchmark` blokları tipli dataclass'lara dönüştürülüyor.
  - Böylece kodda her parametre güvenli şekilde erişiliyor.

#### Adım 3 — Pipeline nesnesi oluşturma
- Dosya: `ocrreader/pipeline.py`
- Sınıf: `RuhsatOcrPipeline.__init__`
- Anlatım:
  - OCR motoru `create_ocr_engine()` ile seçiliyor (Tesseract/Paddle).
  - Template tabanlı anchor fallback için `TemplateAnchorDetector` hazırlanıyor.

#### Adım 4 — Asıl iş akışı başlıyor
- Dosya: `ocrreader/pipeline.py`
- Fonksiyon: `process_path()`
- Anlatım:
  1. `imread_color()` ile görüntü okunuyor.
  2. `preprocess_document()` ile normalize görüntü üretiliyor.
  3. `engine.iter_words()` ile tüm sayfadan kelime + bbox + conf alınıyor.
  4. `detect_anchors_hybrid()` ile anchor'lar bulunuyor.
  5. `resolve_field_rois()` ile alan ROI'leri hesaplanıyor.
  6. `extract_fields()` ile her alan için çoklu aday çıkarılıyor.
  7. `_apply_page_second_pass()` ile boş kalan bazı alanlar sayfa geneli kurtarılıyor.
  8. `_apply_chassis_vin_fix()` ile VIN karakter düzeltmeleri uygulanıyor.
  9. `postprocess_fields()` ile son validasyon/abstain yapılıyor.
  10. Sonuç JSON (`image/pipeline/anchors/fields`) döndürülüyor.

#### Adım 5 — Preprocess içinde ne oluyor?
- Dosya: `ocrreader/preprocess.py`
- Fonksiyon: `preprocess_document()`
- Anlatım:
  - Oryantasyon seçimi
  - Kontrast iyileştirme (CLAHE)
  - Belge quad tespiti
  - Perspektif warp
  - Deskew

#### Adım 6 — Anchor nasıl bulunuyor?
- Dosya: `ocrreader/anchors.py`
- Fonksiyon: `detect_anchors()`
- Anlatım:
  - OCR kelimeleri alias listesi ile fuzzy eşleştiriliyor.
  - `SequenceMatcher + Levenshtein` hibrit benzerlik skoru kullanılıyor.
  - `search_region_norm` ile arama alanı daraltılıyor.
  - Paddle birleşik token sorunu (`MOTORNO`) concat kontrolüyle yakalanıyor.

#### Adım 7 — Alan çıkarma çekirdeği
- Dosya: `ocrreader/fields.py`
- Fonksiyon: `extract_fields()`
- Anlatım:
  - Her alan için aday havuzu oluşturuluyor:
    - semantic adaylar
    - anchor line adayları
    - ROI word adayları
    - OCR crop adayları (preprocessed / alt / raw)
  - `score + priority` ile sıralanıyor.
  - `force_method` varsa önce o değerlendiriliyor.
  - `confidence_threshold` ve post-filter sonrası en iyi aday seçiliyor.

#### Adım 8 — Son temizlik / doğrulama
- Dosya: `ocrreader/field_postprocess.py`
- Fonksiyon: `postprocess_fields()`
- Anlatım:
  - Alan bazlı kurallarla düzeltme yapılır.
  - Geçersizse `value=None` (abstain) olarak işaretlenir.

---

### 12.3 Sunumda birebir kullanılacak kısa konuşma şablonu

> "Önce CLI'den pipeline'ı çalıştırıyorum. Burada config yükleniyor ve pipeline nesnesi oluşturuluyor. Sonra process_path içinde görüntü preprocess ediliyor, OCR kelimeleri alınıyor, anchor'lar bulunuyor, ROI'ler hesaplanıyor, extract_fields çoklu yöntemle aday üretiyor, second pass + postprocess ile son JSON çıkıyor. Yani tek bir OCR sonucu değil, kural + skor + fallback + validasyon birleşik bir sistem çalışıyor."

---

### 12.4 PDB ile canlı breakpoint akışı (isteğe bağlı)

```bash
python -m pdb -m ocrreader.cli \
  --image testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg \
  --config config/ruhsat_schema_paddle_v29.yaml \
  --output output/result.json
```

PDB içinde önerilen breakpoint sırası:

```text
b ocrreader/cli.py:24
b ocrreader/config.py:105
b ocrreader/pipeline.py:233
b ocrreader/preprocess.py:248
b ocrreader/anchors.py:128
b ocrreader/fields.py:892
b ocrreader/field_postprocess.py:240
c
```

Breakpoint'te bakılacak değişkenler:

```text
p args.image
p config.ocr.engine
p len(words)
p len(anchors)
p len(rois)
p field_name
p candidates[:3]
```

Bu sayede "hangi fonksiyon ne iş yaptı, bir sonraki adıma ne veri verdi" kısmını canlı olarak gösterebilirsin.
