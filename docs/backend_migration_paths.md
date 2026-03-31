# Backend Migration Paths

This note records what we actually tested on March 27, 2026 for non-CUDA backend options.

## What we tested

- `ONNX Runtime + DirectML`
- `OpenVINO`
- `AMD/ROCm` viability on the current Windows machine

The commands below are reproducible from this repository:

```powershell
python -m scripts.probe_backend_migrations --attempt-installs
python -m scripts.benchmark_runtime_providers
```

## DirectML path

Observed locally:
- `onnxruntime-directml` imports successfully
- available providers include `DmlExecutionProvider`
- runtime-level DirectML is alive on this machine
- provider microbench ran successfully:
  - ONNX Runtime CPU: `1.494 ms`
  - ONNX Runtime DirectML: `4.764 ms`

Important limitation:
- the current OCR codebase is still built around `PaddleOCR`
- `PaddleOCR(enable_hpi=True)` still fails because the required HPI dependency chain is not installed
- official `paddlex --install hpi-cpu` also fails on this Windows setup because `ultra-infer-python` is not available from the packaged HPI links

Meaning:
- `DirectML` is a viable migration target
- it is not a drop-in switch for the current Paddle pipeline
- to use it for real OCR, we would need an ONNX-based page OCR backend

## OpenVINO path

Observed locally:
- `openvino` installs successfully
- `openvino.Core().available_devices` reports `CPU` and `GPU`
- the local runtime can see a GPU device on this machine
- provider microbench ran successfully:
  - OpenVINO CPU: `4.595 ms`
  - OpenVINO GPU: `3.335 ms`

Important limitation:
- the current `PaddleOCR(enable_hpi=True)` path still does not initialize
- `OpenVINO` availability at the runtime layer does not automatically unlock PaddleOCR HPI on Windows

Meaning:
- `OpenVINO` is viable as a backend runtime
- it still requires OCR-engine-level migration work in this repo

## AMD / ROCm path

Observed locally:
- this machine only exposes an NVIDIA GPU
- `hipInfo.exe` and `rocminfo.exe` are not installed

Meaning:
- ROCm cannot be tested end-to-end on this hardware
- for an AMD target machine, this path would require a separate AMD/HIP-capable environment
- this is not a portable fallback we can validate on the current machine

## Important caution about these numbers

- the microbench above is a tiny synthetic ONNX model
- it proves that the runtime providers are alive and measurable
- it does **not** prove that full-page OCR will be faster than the current warm CUDA path
- today, the only end-to-end OCR path we actually verified near the `3 s` target is still the warm CUDA Paddle path

## Practical conclusion

For this repository today:
- NVIDIA + CUDA remains the only path we actually verified end-to-end for full OCR under the existing Paddle stack
- `DirectML` and `OpenVINO` are realistic migration targets, but not small config changes
- `ROCm` is only relevant if the deployment machine is definitely AMD and we can test on AMD hardware
