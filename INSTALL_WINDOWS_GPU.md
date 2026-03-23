# Windows + GPU Kurulum Rehberi

Bu repo için `requirements.txt` **tek başına tam kurulum anlamına gelmez**.

`requirements.txt` şunları kapsar:
- Python paketleri
- `paddleocr[doc-parser]` ile PaddleOCR-VL tarafının Python bağımlılıkları
- `transformers`
- opsiyonel GLM fallback için vendored `GLM-OCR`

Ama şunları **kurmaz**:
- Tesseract ikilisi (`tesseract.exe`)
- NVIDIA driver / CUDA runtime
- CUDA sürümüne uygun `paddlepaddle-gpu`
- GLM fallback kullanacaksan `Ollama` ve `glm-ocr:latest`

Bu yüzden Windows + GPU + PaddleOCR-VL için aşağıdaki adımları izle.

## 1) Conda ortamı oluştur

```powershell
conda create -n ocrreader python=3.11 -y
conda activate ocrreader
python -m pip install --upgrade pip setuptools wheel
```

## 2) Repo bağımlılıklarını kur

```powershell
python -m pip install -r requirements.txt
```

## 3) CPU Paddle yerine GPU Paddle kur

`requirements.txt` içinde güvenli varsayılan olarak `paddlepaddle` bulunuyor.
GPU kullanacaksan bunu **CUDA sürümüne uygun** `paddlepaddle-gpu` ile değiştirmen gerekir.

Önerilen akış:

```powershell
python -m pip uninstall -y paddlepaddle
```

Ardından kendi CUDA sürümüne uygun resmi `paddlepaddle-gpu` kurulum komutunu çalıştır.

Notlar:
- Bu adım makineye göre değişir.
- Yanlış wheel seçersen `PaddleOCRVL` import edilir ama GPU devreye girmeyebilir.
- Repo kodu Windows tarafında CUDA DLL dizinlerini otomatik bulmaya çalışır; yine de doğru wheel şarttır.

## 4) Tesseract kur

Bu projede crop-level OCR için Tesseract hâlâ kullanılıyor.
`config/ruhsat_schema_paddle_v29.yaml` içinde `paddle_crop_engine: tesseract` ayarı aktif.

Seçenekler:

### Seçenek A: Conda ile

```powershell
conda install -n ocrreader -c conda-forge tesseract -y
```

### Seçenek B: Windows installer

Tesseract'ı Windows'a kur ve gerekirse YAML içinde yolu belirt:

```yaml
ocr:
  executable: C:/Program Files/Tesseract-OCR/tesseract.exe
```

## 5) PaddleOCR-VL import kontrolü

```powershell
python -c "import paddle; print('CUDA compiled:', paddle.device.is_compiled_with_cuda())"
python -c "from paddleocr import PaddleOCRVL; print('PaddleOCRVL import ok')"
```

Beklenen durum:
- ilk komut `True` döndürmeli
- ikinci komut hata vermemeli

## 6) Projeyi test et

```powershell
python -m ocrreader.cli ^
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" ^
  --config "config/ruhsat_schema_paddle_v29.yaml" ^
  --output "output/result.json"
```

## 7) Opsiyonel: GLM fallback kullanacaksan

Bu adım sadece `glm_fallback_enabled: true` yapacaksan gerekir.

Bu repodaki varsayılan akış:
- endpoint: `localhost:11434`
- model: `glm-ocr:latest`
- mod: `ollama_generate`

Kurulum sonrası örnek kontrol:

```powershell
ollama pull glm-ocr:latest
ollama list
```

GLM kurulu değilse ana PaddleOCR/PaddleOCR-VL akışı yine çalışır; sadece secondary fallback kapalı kalır.

## Kısa özet

Eğer hedefin **PaddleOCR-VL + GPU** ise:
- `requirements.txt` gerekli Python paketlerini büyük ölçüde içeriyor
- ama **tek başına yeterli değil**
- ayrıca doğru `paddlepaddle-gpu`, Tesseract ve gerekirse Ollama kurulmalı

En güvenli kontrol sırası:
1. `pip install -r requirements.txt`
2. `paddlepaddle` kaldır
3. uygun `paddlepaddle-gpu` kur
4. Tesseract kur
5. `PaddleOCRVL` import testini çalıştır
6. CLI ile gerçek görsel test et
