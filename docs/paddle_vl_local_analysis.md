# PaddleOCR-VL Local Analysis

This note captures why the local `PaddleOCR-VL` path was extremely slow in this repo and what we verified on March 31, 2026.

## Short answer

The old local path was not slow because `0.9B` is "supposed" to take 12 to 37 minutes on one page.

The main issues were:
- the local PaddleX `native` backend was being used
- the wrapper did not cap `max_new_tokens`
- PaddleX local `DocVLMPredictor` falls back to `max_new_tokens = 8192` for `PaddleOCR-VL`
- the wrapper did not explicitly pass `use_cache=True`
- the wrapper only looked for a narrow subset of result fields and missed the markdown fallback path

## What the codebase was doing before

The repo's `PaddleOCRVLEngine.read_text()` called:

```python
self._vl.predict(rgb, prompt_label=prompt)
```

That means it relied on all PaddleX defaults.

Relevant upstream defaults we verified:
- pipeline config default backend is `native`
- pipeline config default `use_queues` is `True`
- local `DocVLMPredictor.process()` forces `max_new_tokens = 8192` when no explicit limit is provided

Sources:
- `paddlex/configs/pipelines/PaddleOCR-VL-1.5.yaml`
- `paddlex/inference/models/doc_vlm/predictor.py`

## Why this became pathological

### 1. The local path is native generative inference

`PaddleOCR-VL` is not the same kind of fast static OCR pipeline as PP-OCR detection + recognition.

The local pipeline:
- loads the VLM locally
- preprocesses the page or cropped blocks
- performs autoregressive generation
- formats the result back into OCR / markdown-like output

That is fundamentally slower than classical OCR.

### 2. `max_new_tokens = 8192` was the biggest mistake

In the local predictor, if `max_new_tokens` is not passed, PaddleX uses `8192` for `PaddleOCR-VL`.

That was happening in this repo before the fix.

For local inference this is far too high for the ruhsat image use case and causes a massive decode budget.

### 3. Layout mode multiplies work

When layout detection is on, the pipeline:
- runs `PP-DocLayoutV3`
- crops layout blocks
- sends multiple block images into the VLM

So one page is not one cheap call. It becomes:
- layout detection
- several block-level generation tasks
- optional OCR on image blocks

### 4. Windows local usage is fragile

We also verified a Windows-specific fragility:
- direct raw `PaddleOCRVL(...)` calls can fail with missing CUDA DLL resolution
- our repo wrapper fixed that by calling `_configure_windows_cuda_dlls()`

So the old path was both slow and brittle.

## Measurements from this repo

Old measurements already stored in repo:
- baseline local VL: `2240.008 s`
- no-layout local VL: `764.129 s`
- no-layout `max_side=1600`: `702.049 s`

Files:
- `tests/profile_paddle_vl_baseline.json`
- `tests/profile_paddle_vl_no_layout.json`
- `tests/profile_paddle_vl_no_layout_1600.json`

After tuning the local wrapper:
- no-layout, `max_side=1600`, `max_new_tokens=512`: `24.103 s`
- layout on, `max_side=1600`, `max_new_tokens=512`: `36.247 s`

Files:
- `tests/profile_paddle_vl_tuned_no_layout_1600.json`
- `tests/profile_paddle_vl_tuned_layout_1600.json`

Additional direct experiments during this analysis:
- no-layout, `max_new_tokens=256`: about `15.5-20.3 s`
- no-layout, `max_new_tokens=512`: about `23.6-26.6 s`
- layout on, `max_new_tokens=256`: about `25.7-36.6 s`

## What we changed in repo

The wrapper now supports local VL tuning knobs:
- `paddle_vl_max_new_tokens`
- `paddle_vl_min_pixels`
- `paddle_vl_max_pixels`
- `paddle_vl_use_cache`
- `paddle_vl_use_queues`
- `paddle_vl_prompt_label`

The wrapper now also:
- disables the Paddle model source connectivity check by default
- passes tuned kwargs into `PaddleOCRVL.predict()`
- falls back to `result.markdown["markdown_texts"]` when direct block extraction is missing

## Practical conclusion

The old `700-2200 s` behavior was not "normal model size behavior".

It was mostly a bad local inference configuration.

After fixing the local path, the result is dramatically better, but still not near `3 s`.

On this machine, realistic local-only PaddleOCR-VL numbers were:
- roughly `15-24 s` for no-layout mode
- roughly `25-36 s` for layout mode

So:
- yes, the old behavior was wrong
- yes, we fixed the biggest local bottlenecks
- no, local native PaddleOCR-VL is still not a `3 s` solution here

## Recommended local settings

For local-only testing on ruhsat-like images:

```yaml
ocr:
  glm_model: paddle_vl
  paddle_vl_max_side: 1600
  paddle_vl_max_new_tokens: 512
  paddle_vl_use_cache: true
  paddle_vl_use_queues: true
```

For the best quality on the sample ruhsat page we checked:

```yaml
ocr:
  paddle_vl_use_layout_detection: true
  paddle_vl_use_ocr_for_image_block: true
```

If you only need the fastest local page OCR and can tolerate quality loss:

```yaml
ocr:
  paddle_vl_use_layout_detection: false
  paddle_vl_use_ocr_for_image_block: false
```

That is the fastest local-only configuration we validated in this repo.
