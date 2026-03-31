# Runtime Performance Paths

This note records the three runtime paths we tested on March 27, 2026.

## 1. CPU-only path

Goal:
- keep the current `PaddleOCR + Tesseract` pipeline
- reduce useless crop OCR calls

Implementation:
- added `ocr.crop_ocr_mode`
- added `ocr.crop_ocr_variants`
- added `ocr.crop_ocr_skip_margin`
- new config: `config/ruhsat_schema_paddle_v29_cpu_lazy.yaml`

Command:

```powershell
python -m scripts.profile_runtime_mode `
  --label cpu_lazy `
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" `
  --config config/ruhsat_schema_paddle_v29_cpu_lazy.yaml `
  --runs 1 `
  --summary-out output/cpu_lazy_profile.json
```

Result on local sample:
- baseline CPU: `266.143 s`
- lazy CPU: `256.202 s`
- extracted field values stayed the same on this sample

Conclusion:
- helps a little
- not enough for a `3 s` target on CPU
- main bottleneck stays full-page Paddle inference

## 2. CUDA path

Goal:
- use the already-installed GPU Paddle wheel in the current environment
- combine that with the lazy crop OCR path

What changed in the environment:
- removed the user-site CPU `paddlepaddle` package
- Python now imports `paddlepaddle-gpu`
- `paddle.device.is_compiled_with_cuda()` is now `True`

Configs:
- baseline GPU: `config/ruhsat_schema_paddle_v29.yaml`
- optimized GPU: `config/ruhsat_schema_paddle_v29_gpu_lazy.yaml`

Commands:

```powershell
python -m scripts.profile_runtime_mode `
  --label gpu_baseline `
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" `
  --config config/ruhsat_schema_paddle_v29.yaml `
  --runs 1 `
  --summary-out output/gpu_baseline_profile.json
```

```powershell
python -m scripts.profile_runtime_mode `
  --label gpu_lazy `
  --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" `
  --config config/ruhsat_schema_paddle_v29_gpu_lazy.yaml `
  --runs 2 `
  --summary-out output/gpu_lazy_profile.json
```

Results on local sample:
- GPU baseline: `9.692 s`
- GPU lazy run 1: `3.318 s`
- GPU lazy run 2: `2.996 s`
- extracted field values stayed the same on this sample

Conclusion:
- this is the first path that reaches the `~3 s` target
- the second run shows the value of a warm process that keeps the model loaded

## 3. Non-CUDA GPU path

Goal:
- check whether a CUDA-free GPU acceleration route is available on this Windows machine

Implementation:
- added `scripts/probe_non_cuda_backends.py`
- installed `onnxruntime-directml`

Command:

```powershell
python -m scripts.probe_non_cuda_backends
```

Result:
- OpenCV OpenCL: available
- ONNX Runtime DirectML: available
- providers: `DmlExecutionProvider`, `CPUExecutionProvider`

Output file:
- `output/non_cuda_backend_probe.json`

Conclusion:
- a non-CUDA GPU backend is available at the runtime layer
- but the current OCR pipeline is still Paddle-based, so this is only a probe/POC
- to use this path for real OCR, the page OCR engine would need to be moved to an ONNX/DirectML-compatible backend

## Practical recommendation

If the deployment machine has NVIDIA + CUDA:
- use the GPU lazy path
- keep the process warm

If the deployment machine has no CUDA:
- the current Paddle path is not the right deployment target
- use the DirectML probe as the starting point for a backend migration, not as a drop-in runtime switch
