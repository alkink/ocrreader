"""Minimal shim for PaddleX's unconditional `import modelscope`.

This project uses cached/local PaddleOCR models and does not need the
real ModelScope client. The actual `modelscope` package in the current
environment pulls in a broken `torch` stack on import, which prevents
PaddleOCR from initializing at all.
"""

from __future__ import annotations


def snapshot_download(*args, **kwargs):
    raise RuntimeError(
        "ModelScope download is unavailable in this OCRReader environment. "
        "Use cached Paddle models or configure another PaddleX model source."
    )
