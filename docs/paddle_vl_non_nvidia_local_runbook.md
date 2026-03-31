# PaddleOCR-VL Non-NVIDIA Local Runbook

This note captures the practical, repo-specific plan for running `PaddleOCR-VL` on non-NVIDIA hardware while keeping `layout on`.

It is intentionally opinionated:
- if you want strict local-only and no service/process split, AMD is the only realistic non-NVIDIA path from the official guidance
- if your target is Intel Arc, the official PaddleOCR-VL path currently requires a local `vLLM` service

## Short decision table

| Target GPU | Officially validated by PaddleOCR-VL docs | Native local `PaddlePaddle` path | Local high-perf path | Works with current repo VL wrapper as-is |
| --- | --- | --- | --- | --- |
| AMD GPU | AMD MI300X | Yes | `vLLM` service in Docker | Closest fit |
| Intel Arc GPU | Intel Arc B60 Pro | No | `vLLM` service in Docker | No, needs a service adapter |

## Shared constraints

- The main PaddleOCR-VL doc says the supported hardware depends on the inference method you choose.
- The same doc says `vLLM`, `SGLang`, and `FastDeploy` do not run natively on Windows and recommends Docker for them.
- In this repo, `PaddleOCRVLEngine` uses the local PaddleOCR Python object directly, so it matches the native `PaddlePaddle` path much better than a service-based path.
- For our ruhsat sample, `layout on` preserved quality, while `layout off` caused a clear drop. Keep `layout on` for quality-sensitive usage.
- This repo now supports explicit VL runtime profiles in config:
  - `auto`
  - `native_paddle_vl`
  - `local_vllm_service`
  - `local_mlx_vlm_service`

Official references:
- Main PaddleOCR-VL usage tutorial:
  - https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- AMD tutorial:
  - https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-AMD-GPU.html
- Intel Arc tutorial:
  - https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-Intel-Arc-GPU.html
- Microsoft WSL GPU compute guide:
  - https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
- AMD ROCm WSL compatibility matrix:
  - https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html

## Recommended repo stance

### Runtime profile fields now supported in repo

```yaml
ocr:
  glm_model: paddle_vl
  paddle_vl_runtime_profile: auto
  paddle_vl_service_url: null
  paddle_vl_service_model_name: null
  paddle_vl_service_api_key: null
  paddle_vl_service_max_concurrency: null
```

Profile meanings:
- `auto`: macOS -> `local_mlx_vlm_service`, Intel Arc -> `local_vllm_service`, others -> `native_paddle_vl`
- `native_paddle_vl`: local PaddleOCR-VL object in-process
- `local_vllm_service`: local PaddleOCR-VL client using a local `vLLM` server URL
- `local_mlx_vlm_service`: local PaddleOCR-VL client using a local `MLX-VLM` server URL

For Apple Silicon in this repo, `scripts/bootstrap_runtime.py` now bootstraps a separate `.venv-mlx` environment automatically when a Python `3.10+` interpreter is available. You can then launch the local server with:

```bash
python scripts/start_mlx_vl_server.py --port 8111
```

### If you need strict local-only and no local API/service

Choose AMD GPU, not Intel Arc.

Reason:
- the AMD tutorial still points to the normal PaddleOCR-VL quick-start flow
- the Intel Arc tutorial explicitly says the `PaddlePaddle` inference method is not supported on Intel Arc and sends you to `vLLM`

### If Intel Arc is non-negotiable

Accept a local service on the same machine.

That still counts as local deployment from an infrastructure point of view, but it is not the same as the current in-process `PaddleOCRVLEngine`.

## AMD GPU path

### What the official docs say

- PaddleOCR-VL AMD validation was done on `AMD MI300X`.
- The tutorial recommends the official Docker image first.
- Manual install is also documented.
- For better performance, the AMD guide uses a local `vLLM` service in Docker.

### Best fit for this repo

Use the native local `PaddlePaddle` path first, because:
- it matches the current repo architecture
- it preserves the "no API" requirement
- it lets us keep using `PaddleOCRVLEngine` directly

### Host recommendation

Preferred order:
1. Native Linux host
2. WSL2 Ubuntu if your exact AMD GPU is listed in AMD's WSL compatibility matrix
3. Windows-native only if you are ready to debug vendor-specific driver/runtime issues yourself

### WSL recommendation

If you use WSL, verify both:
- your Ubuntu version is supported
- your exact GPU is present in AMD's WSL GPU support matrix

Our current WSL distro path is compatible with the documented Ubuntu direction:
- `\\\\wsl.localhost\\Ubuntu-24.04\\home\\alki\\projects\\ocrreader`
- Linux path form: `/home/alki/projects/ocrreader`

### AMD Docker quick-start for this repo

Start a local container from WSL or Linux:

```bash
docker run -it --rm \
  --user root \
  --device /dev:/dev \
  --shm-size 64g \
  --network host \
  -v /home/alki/projects/ocrreader:/workspace \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-amd-gpu \
  /bin/bash
```

Inside the container:

```bash
cd /workspace
paddleocr doc_parser \
  -i testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg \
  --use_layout_detection True
```

### AMD manual install skeleton

The official AMD tutorial documents this minimal setup:

```bash
python -m venv .venv_paddleocr
source .venv_paddleocr/bin/activate
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -U "paddleocr[doc-parser]"
```

For this repo, keep these VL settings:

```yaml
ocr:
  glm_model: paddle_vl
  paddle_vl_runtime_profile: native_paddle_vl
  paddle_vl_use_layout_detection: true
  paddle_vl_use_ocr_for_image_block: true
  paddle_vl_max_side: 1600
  paddle_vl_max_new_tokens: 512
  paddle_vl_use_cache: true
  paddle_vl_use_queues: true
  paddle_vl_prompt_label: ocr
```

### AMD checkpoints before we trust it

1. Confirm the GPU is visible in the target runtime.
2. Run the one-image profile on our sample.
3. Compare output quality against the current NVIDIA baseline.
4. Only then decide whether native local mode is enough or whether local `vLLM` is needed.

## Intel Arc path

### What the official docs say

- PaddleOCR-VL Intel Arc validation was done on `Intel Arc B60 Pro`.
- The Intel Arc tutorial explicitly says:
  - Intel Arc currently does not support inference using the `PaddlePaddle` inference method.
  - use the `vLLM` inference acceleration framework instead.
- The official high-performance path is a local Docker `vLLM` service.

### Practical implication for this repo

This does not match the current `PaddleOCRVLEngine` design.

Our current repo code assumes in-process local inference:
- construct a local PaddleOCR-VL object
- call `.predict(...)`
- parse results in-process

Intel Arc's official path needs a local service boundary instead.

For this repo, that means:

```yaml
ocr:
  glm_model: paddle_vl
  paddle_vl_runtime_profile: local_vllm_service
  paddle_vl_service_url: http://localhost:8118/v1
  paddle_vl_service_model_name: PaddlePaddle/PaddleOCR-VL-1.5
  paddle_vl_use_layout_detection: true
  paddle_vl_use_ocr_for_image_block: true
```

### Intel Arc Docker service command

```bash
docker run -it --rm \
  --name paddleocr_vllm \
  --user root \
  --device /dev:/dev \
  --shm-size 64g \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-intel-gpu \
  paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --host 0.0.0.0 \
    --port 8118 \
    --backend vllm
```

### What "local" means here

This is still local to the machine, but it is not "single-process local inference".

If you keep the hard rule "no API/service at all", Intel Arc is not a good PaddleOCR-VL target today.

### What we would need in this repo for Intel Arc

We would need a new adapter, roughly:
1. add a `PaddleOCRVLServiceEngine`
2. send page images to the local `genai_server`
3. normalize the returned result into the same text/result shape
4. keep the existing ruhsat summary formatting
5. add config fields for service URL, timeout, and backend mode

That is a real engine migration, not a config flip.

## Windows and WSL guidance

### Windows-native

Do not plan around Windows-native `vLLM` for PaddleOCR-VL.

The official main PaddleOCR-VL tutorial says `vLLM`, `SGLang`, and `FastDeploy` do not run natively on Windows and tells users to use Docker images.

### WSL2

WSL is useful as the Linux runtime layer, but you still need the vendor stack to be exposed properly.

- Microsoft documents GPU-accelerated ML in WSL and explicitly calls out DirectML-backed frameworks for AMD, Intel, and NVIDIA GPUs.
- For AMD, AMD also publishes a WSL compatibility matrix with exact supported GPUs and Ubuntu versions.
- For Intel Arc, PaddleOCR-VL's own official guidance still pushes you to the Docker + `vLLM` route.

### Apple Silicon note

The same profile mechanism also gives us a clean Apple Silicon slot:

```yaml
ocr:
  glm_model: paddle_vl
  paddle_vl_runtime_profile: local_mlx_vlm_service
  paddle_vl_service_url: http://localhost:8111/
  paddle_vl_service_model_name: PaddlePaddle/PaddleOCR-VL-1.5
  paddle_vl_use_layout_detection: true
  paddle_vl_use_ocr_for_image_block: true
```

This matches the repo architecture better than inventing a separate Apple-only code path.

## Concrete recommendation

If the next non-NVIDIA target has not been purchased yet:

1. Pick AMD over Intel Arc for PaddleOCR-VL.
2. Use Linux or WSL2 Ubuntu 24.04 with the official AMD-compatible stack.
3. Start with the official AMD PaddleOCR-VL Docker image.
4. Keep `layout on`.
5. Benchmark native local mode first.
6. Only move to local `vLLM` if native local speed is still too slow.

If the target machine is already Intel Arc:

1. Accept a local `vLLM` service as part of the design.
2. Treat it as a local dependency, not a remote API.
3. Plan a repo-level service adapter before promising integration.
