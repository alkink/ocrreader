# Karşılaştırma Scripti
# Bu scripti PowerShell'de çalıştırarak eski ve yeni sürümlerin hızını tam dataset üzerinde karşılaştırabilirsin.

# 1. ORİJİNAL PADDLEOCR (Sadece GPU, Hybrid kapalı, Tek Worker)
# Bu senin en başta denediğin, her resim için 30 kez GPU round-trip yapan yavaş versiyon.
Write-Host "================= TEST 1: ORİJİNAL PADDLEOCR (ÇOK YAVAŞ) ==================" -ForegroundColor Yellow
# (Config dosyasında geçici olarak hybrid'i kapatarak test ediyoruz)
python -c "
import yaml
with open('config/ruhsat_schema_paddle_v29.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['ocr']['paddle_crop_engine'] = None
with open('config/ruhsat_schema_paddle_v29.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
Measure-Command {
    python scripts/benchmark_pipeline.py --config config/ruhsat_schema_paddle_v29.yaml --pipeline-module ocrreader.pipeline_v31 --workers 1 --max-images 10 --output-dir dataset/generated/qa/bench_test1
}
Write-Host ""


# 2. HYBRID ENGINE (Tek Worker)
# GPU sadece sayfa taramasında kullanılır, küçük alanlar için in-process Tesseract kullanılır.
Write-Host "================= TEST 2: HYBRID ENGINE (TEK WORKER) ==================" -ForegroundColor Yellow
python -c "
import yaml
with open('config/ruhsat_schema_paddle_v29.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['ocr']['paddle_crop_engine'] = 'tesseract'
with open('config/ruhsat_schema_paddle_v29.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
Measure-Command {
    python scripts/benchmark_pipeline.py --config config/ruhsat_schema_paddle_v29.yaml --pipeline-module ocrreader.pipeline_v31 --workers 1 --max-images 10 --output-dir dataset/generated/qa/bench_test2
}
Write-Host ""


# 3. HYBRID ENGINE + MULTIPROCESSING (2 Worker)
# Hem Hybrid Engine aktif, hem de 2 resim aynı anda işlenir. (En hızlı olması beklenen)
Write-Host "================= TEST 3: HYBRID ENGINE + MULTIPROCESSING (2 WORKER) ==================" -ForegroundColor Yellow
Measure-Command {
    python scripts/benchmark_pipeline.py --config config/ruhsat_schema_paddle_v29.yaml --pipeline-module ocrreader.pipeline_v31 --workers 2 --max-images 10 --output-dir dataset/generated/qa/bench_test3
}
Write-Host ""

Write-Host "Tüm testler tamamlandı. Sürelere bakılarak (TotalSeconds) fark net olarak görülebilir." -ForegroundColor Green
