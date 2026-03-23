from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Protocol
import yaml

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .text_utils import collapse_spaces, normalize_turkish_ascii

# PaddleOCR 3.x may perform connectivity checks to model hosters on init.
# Keep this disabled by default for deterministic CLI/benchmark behavior.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _configure_windows_cuda_dlls() -> None:
    if os.name != "nt":
        return

    import glob

    candidates: list[Path] = []
    # Search for all bin directories under any 'nvidia' directory in the python path
    for base in map(Path, sys.path):
        nvidia_root = base / "nvidia"
        if not nvidia_root.is_dir():
            continue

        # Find all 'bin' directories (and 'x86_64' subdirs of bin)
        for bin_dir in glob.glob(str(nvidia_root / "**" / "bin"), recursive=True):
            candidates.append(Path(bin_dir))
            # Some packages (like cu13) put DLLs in a 'x86_64' subfolder under bin
            x64_path = Path(bin_dir) / "x86_64"
            if x64_path.is_dir():
                candidates.append(x64_path)

    # Also check AppData specifically
    try:
        appdata_nvidia = (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Python"
            / f"Python{sys.version_info.major}{sys.version_info.minor}"
            / "site-packages"
            / "nvidia"
        )
        if appdata_nvidia.is_dir():
            for bin_dir in glob.glob(str(appdata_nvidia / "**" / "bin"), recursive=True):
                candidates.append(Path(bin_dir))
                x64_path = Path(bin_dir) / "x86_64"
                if x64_path.is_dir():
                    candidates.append(x64_path)
    except Exception:
        pass

    seen: set[str] = set()
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for candidate in candidates:
        resolved = str(candidate.absolute())
        if resolved in seen or not candidate.is_dir():
            continue
        seen.add(resolved)
        try:
            os.add_dll_directory(resolved)
        except (AttributeError, FileNotFoundError, OSError):
            pass
        if resolved not in path_parts:
            path_parts.insert(0, resolved)
    os.environ["PATH"] = os.pathsep.join(path_parts)


# Run configuration on module load to ensure CUDA DLLs are found.
_configure_windows_cuda_dlls()


from .config import OCRConfig
from .types import Rect


@dataclass(frozen=True)
class OCRWord:
    text: str
    conf: float
    bbox: Rect
    block_num: int
    par_num: int
    line_num: int


class OCREngine(Protocol):
    config: OCRConfig

    def iter_words(self, image: np.ndarray, psm: int | None = None, min_conf: float = 20.0) -> list[OCRWord]:
        ...

    def read_text(
        self,
        image: np.ndarray,
        psm: int | None = None,
        whitelist: str | None = None,
    ) -> str:
        ...


class TextReadEngine(Protocol):
    def read_text(
        self,
        image: np.ndarray,
        psm: int | None = None,
        whitelist: str | None = None,
    ) -> str:
        ...


def _prefer_vendored_glmocr_source() -> None:
    project_root = Path(__file__).resolve().parents[1]
    vendor_root = project_root / "vendor" / "GLM-OCR"
    if not vendor_root.is_dir():
        return

    vendor_str = str(vendor_root)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


class TesseractEngine:
    def __init__(self, config: OCRConfig):
        self.config = config
        if config.executable:
            pytesseract.pytesseract.tesseract_cmd = config.executable

    def _build_config(self, psm: int | None = None, whitelist: str | None = None) -> str:
        parts = [f"--oem {self.config.oem}", f"--psm {psm if psm is not None else self.config.psm}"]
        if whitelist:
            parts.append(f"-c tessedit_char_whitelist={whitelist}")
        return " ".join(parts)

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def iter_words(self, image: np.ndarray, psm: int | None = None, min_conf: float = 20.0) -> list[OCRWord]:
        rgb = self._to_rgb(image)
        data = pytesseract.image_to_data(
            rgb,
            output_type=Output.DICT,
            config=self._build_config(psm=psm),
            lang=self.config.language,
        )

        words: list[OCRWord] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue

            try:
                conf = float(data["conf"][i])
            except ValueError:
                conf = -1.0

            if conf < min_conf:
                continue

            bbox = Rect(
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=max(1, int(data["width"][i])),
                h=max(1, int(data["height"][i])),
            )
            words.append(
                OCRWord(
                    text=text,
                    conf=conf,
                    bbox=bbox,
                    block_num=int(data["block_num"][i]),
                    par_num=int(data["par_num"][i]),
                    line_num=int(data["line_num"][i]),
                )
            )
        return words

    def read_text(
        self,
        image: np.ndarray,
        psm: int | None = None,
        whitelist: str | None = None,
    ) -> str:
        rgb = self._to_rgb(image)
        return pytesseract.image_to_string(
            rgb,
            config=self._build_config(psm=psm, whitelist=whitelist),
            lang=self.config.language,
        )


class PaddleOCREngine:
    def __init__(self, config: OCRConfig):
        self.config = config
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PaddleOCR engine selected but dependency is missing. "
                "Install with: pip install paddleocr paddlepaddle"
            ) from exc

        self._paddle_cls = PaddleOCR
        self._lang = self._resolve_lang(config.language)

        # PP-OCRv5 can hard-crash the Python process on some Windows CPU setups
        # (native access violation from backend runtime). Prefer v4 by default for
        # stability and allow explicit override via config or env.
        config_version = str(getattr(config, "paddle_version", "") or "").strip()
        env_version = str(os.environ.get("OCRREADER_PADDLE_OCR_VERSION", "")).strip()
        preferred_version = config_version or env_version
        
        if preferred_version in {"PP-OCRv3", "PP-OCRv4", "PP-OCRv5"}:
            self._candidate_versions = (preferred_version,)
        else:
            self._candidate_versions = ("PP-OCRv4", "PP-OCRv5")

        self._ocr = None
        self._ocr_version: str | None = None
        init_errors: list[Exception] = []
        for version in self._candidate_versions:
            try:
                self._ocr = self._build_engine(version)
                self._ocr_version = version
                break
            except Exception as exc:
                init_errors.append(exc)

        if self._ocr is None:
            msg = "; ".join(f"{type(e).__name__}: {e}" for e in init_errors) or "unknown error"
            raise RuntimeError(f"Failed to initialize PaddleOCR backend. Details: {msg}")

    def _resolve_paddle_device(self) -> str:
        requested = str(getattr(self.config, "paddle_device", "gpu") or "gpu").strip().lower()
        if requested not in {"cpu", "gpu"}:
            requested = "gpu"
        if requested == "cpu":
            return "cpu"

        try:
            import paddle  # type: ignore

            if bool(paddle.device.is_compiled_with_cuda()):
                return "gpu"
        except Exception:
            pass
        return "cpu"

    def _build_engine(self, ocr_version: str):
        # PaddleOCR 3.x common args are parsed by paddleocr._common_args.
        # Keep orientation helpers off and expose runtime knobs from config.
        device = self._resolve_paddle_device()
        kwargs: dict[str, object] = {
            "lang": self._lang,
            "ocr_version": ocr_version,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_hpi": False,
            "device": device,
        }

        if device == "cpu":
            kwargs["enable_mkldnn"] = bool(getattr(self.config, "paddle_enable_mkldnn", False))
            kwargs["cpu_threads"] = max(1, int(getattr(self.config, "paddle_cpu_threads", 1) or 1))

        det_side = getattr(self.config, "paddle_text_det_limit_side_len", None)
        if det_side is not None:
            kwargs["text_det_limit_side_len"] = max(320, int(det_side))

        rec_batch = getattr(self.config, "paddle_text_recognition_batch_size", None)
        if rec_batch is not None:
            kwargs["text_recognition_batch_size"] = max(1, int(rec_batch))

        return self._paddle_cls(
            **kwargs,
        )

    @staticmethod
    def _resolve_lang(language: str) -> str:
        lang_up = (language or "").lower()
        if "tur" in lang_up:
            return "en"
        if "eng" in lang_up:
            return "en"
        return "en"

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def _score_to_percent(scores: object, idx: int) -> float:
        if not isinstance(scores, list) or idx >= len(scores):
            return 0.0
        try:
            raw = float(scores[idx])
        except Exception:
            return 0.0
        return raw * 100.0 if raw <= 1.0 else raw

    def _predict(self, image: np.ndarray, return_word_box: bool) -> list[object]:
        if self._ocr is None:
            return []

        bgr = self._to_bgr(image)
        kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "return_word_box": return_word_box,
        }

        try:
            return self._ocr.predict(bgr, **kwargs)
        except Exception:
            # If MKLDNN path fails on some Windows/Paddle builds, retry safely.
            if bool(getattr(self.config, "paddle_enable_mkldnn", False)):
                self.config = replace(self.config, paddle_enable_mkldnn=False)
                self._ocr = self._build_engine(self._ocr_version or "PP-OCRv4")
                return self._ocr.predict(bgr, **kwargs)
            if self._ocr_version == "PP-OCRv4":
                raise
            # Runtime fallback for PP-OCRv5 incompatibilities on some CPU setups.
            self._ocr = self._build_engine("PP-OCRv4")
            self._ocr_version = "PP-OCRv4"
            return self._ocr.predict(bgr, **kwargs)

    @staticmethod
    def _result_dicts(raw: object) -> list[dict[str, object]]:
        if not isinstance(raw, list):
            return []

        out: list[dict[str, object]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(item)
                continue

            json_payload = getattr(item, "json", None)
            if isinstance(json_payload, dict):
                res = json_payload.get("res")
                if isinstance(res, dict):
                    out.append(res)
        return out

    @staticmethod
    def _bbox_from_word_box(box: object) -> Rect | None:
        # Paddle word boxes may come as [x1, y1, x2, y2].
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in box]
            except Exception:
                return None
            min_x = int(np.floor(min(x1, x2)))
            min_y = int(np.floor(min(y1, y2)))
            max_x = int(np.ceil(max(x1, x2)))
            max_y = int(np.ceil(max(y1, y2)))
            return Rect(
                x=max(0, min_x),
                y=max(0, min_y),
                w=max(1, max_x - min_x),
                h=max(1, max_y - min_y),
            )

        # Fallback: polygon-like [[x,y], ...]
        return PaddleOCREngine._bbox_from_points(box)

    @staticmethod
    def _bbox_from_points(points: object) -> Rect | None:
        try:
            arr = np.array(points, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if arr.size == 0:
            return None
        min_x = int(np.floor(float(np.min(arr[:, 0]))))
        min_y = int(np.floor(float(np.min(arr[:, 1]))))
        max_x = int(np.ceil(float(np.max(arr[:, 0]))))
        max_y = int(np.ceil(float(np.max(arr[:, 1]))))
        return Rect(
            x=max(0, min_x),
            y=max(0, min_y),
            w=max(1, max_x - min_x),
            h=max(1, max_y - min_y),
        )

    @staticmethod
    def _split_line_to_word_boxes(text: str, bbox: Rect) -> list[tuple[str, Rect]]:
        tokens = [t for t in text.strip().split() if t]
        if not tokens:
            return []
        if len(tokens) == 1:
            return [(tokens[0], bbox)]

        total = max(1, sum(len(t) for t in tokens))
        out: list[tuple[str, Rect]] = []
        cursor = bbox.x
        remaining_w = bbox.w
        remaining_chars = total
        for idx, tok in enumerate(tokens):
            if idx == len(tokens) - 1:
                tw = max(1, remaining_w)
            else:
                portion = len(tok) / max(1, remaining_chars)
                tw = max(1, int(round(remaining_w * portion)))
            out.append((tok, Rect(x=cursor, y=bbox.y, w=tw, h=bbox.h)))
            cursor += tw
            remaining_w = max(1, bbox.x + bbox.w - cursor)
            remaining_chars = max(1, remaining_chars - len(tok))
        return out

    def iter_words(self, image: np.ndarray, psm: int | None = None, min_conf: float = 20.0) -> list[OCRWord]:
        _ = psm  # PaddleOCR does not use tesseract-like psm.
        raw = self._predict(image, return_word_box=True)
        results = self._result_dicts(raw)

        words: list[OCRWord] = []
        line_idx = 0

        for result in results:
            rec_texts = result.get("rec_texts")
            rec_scores = result.get("rec_scores")
            rec_polys = result.get("rec_polys")
            text_word = result.get("text_word")
            text_word_boxes = result.get("text_word_boxes")

            if not isinstance(rec_texts, list):
                continue

            for idx, raw_text in enumerate(rec_texts):
                text = str(raw_text or "").strip()
                if not text:
                    continue

                conf = self._score_to_percent(rec_scores, idx)
                if conf < min_conf:
                    continue

                line_idx += 1

                used_word_boxes = False
                if (
                    isinstance(text_word, list)
                    and isinstance(text_word_boxes, list)
                    and idx < len(text_word)
                    and idx < len(text_word_boxes)
                    and isinstance(text_word[idx], list)
                    and isinstance(text_word_boxes[idx], list)
                    and len(text_word[idx]) == len(text_word_boxes[idx])
                ):
                    for tok_raw, box_raw in zip(text_word[idx], text_word_boxes[idx]):
                        tok = str(tok_raw or "").strip()
                        if not tok:
                            continue
                        tok_bbox = self._bbox_from_word_box(box_raw)
                        if tok_bbox is None:
                            continue
                        words.append(
                            OCRWord(
                                text=tok,
                                conf=conf,
                                bbox=tok_bbox,
                                block_num=1,
                                par_num=1,
                                line_num=line_idx,
                            )
                        )
                        used_word_boxes = True

                if used_word_boxes:
                    continue

                line_bbox: Rect | None = None
                if isinstance(rec_polys, list) and idx < len(rec_polys):
                    line_bbox = self._bbox_from_points(rec_polys[idx])
                if line_bbox is None:
                    rec_boxes = result.get("rec_boxes")
                    if isinstance(rec_boxes, list) and idx < len(rec_boxes):
                        line_bbox = self._bbox_from_word_box(rec_boxes[idx])
                if line_bbox is None:
                    continue

                for token, tok_bbox in self._split_line_to_word_boxes(text, line_bbox):
                    words.append(
                        OCRWord(
                            text=token,
                            conf=conf,
                            bbox=tok_bbox,
                            block_num=1,
                            par_num=1,
                            line_num=line_idx,
                        )
                    )
        return words

    def read_text(
        self,
        image: np.ndarray,
        psm: int | None = None,
        whitelist: str | None = None,
    ) -> str:
        _ = psm
        _ = whitelist
        raw = self._predict(image, return_word_box=False)
        results = self._result_dicts(raw)
        text_lines: list[str] = []

        for result in results:
            rec_texts = result.get("rec_texts")
            if not isinstance(rec_texts, list):
                continue

            for raw_text in rec_texts:
                text = str(raw_text or "").strip()
                if text:
                    text_lines.append(text)
        return "\n".join(text_lines)


def _collect_glm_text_parts(payload: object) -> list[str]:
    if payload is None:
        return []

    if isinstance(payload, str):
        text = payload.strip()
        return [text] if text else []

    if isinstance(payload, list):
        out: list[str] = []
        for item in payload:
            out.extend(_collect_glm_text_parts(item))
        return out

    if isinstance(payload, dict):
        out: list[str] = []
        for key in ("content", "text", "markdown", "md"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
        for key, val in payload.items():
            if key in {"bbox", "bbox_2d", "index", "label", "type", "score", "page"}:
                continue
            out.extend(_collect_glm_text_parts(val))
        return out

    return []


def _strip_glm_markup(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^```(?:[A-Za-z0-9_-]+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"\bmarkdown\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _restore_tr_display(text: str) -> str:
    out = str(text or "")
    replacements = {
        "GRI": "GRİ",
        "ESKISEHIR": "ESKİŞEHİR",
        "ESKISEHR": "ESKİŞEHİR",
        "TEPEBASI": "TEPEBAŞI",
        "ULUONDER": "ULUÖNDER",
        "NOTERLIGI": "NOTERLİĞİ",
        "VERILDIGI": "VERİLDİĞİ",
        "KIMLIK": "KİMLİK",
        "OTOMOBIL": "OTOMOBİL",
        "BENZINLI": "BENZİNLİ",
        "TITANYUM": "TİTANYUM",
    }
    for src, dst in replacements.items():
        out = re.sub(rf"\b{src}\b", dst, out)
    return out


class GlmFallbackEngine:
    def __init__(self, config: OCRConfig):
        self.config = config
        _prefer_vendored_glmocr_source()
        try:
            import glmocr  # type: ignore
            from glmocr import GlmOcr  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "GLM OCR fallback is enabled but dependency is missing. "
                "Install with: pip install glmocr"
            ) from exc

        self.source_path = str(getattr(glmocr, "__file__", ""))
        self._config_path: Path | None = None

        glm_mode = str(config.glm_mode or "selfhosted").strip().lower() or "selfhosted"
        ocr_api_host = config.glm_ocr_api_host or ("localhost" if glm_mode == "selfhosted" else None)
        ocr_api_port = int(config.glm_ocr_api_port) if config.glm_ocr_api_port is not None else (11434 if (config.glm_api_mode == "ollama_generate") else (8080 if glm_mode == "selfhosted" else None))
        ocr_api_model = config.glm_model or ("glm-ocr:latest" if (config.glm_api_mode == "ollama_generate") else None)
        ocr_api_path = config.glm_api_path or ("/api/generate" if (config.glm_api_mode == "ollama_generate") else None)
        ocr_api_mode = config.glm_api_mode or "openai"

        if glm_mode == "selfhosted":
            glm_cfg: dict[str, object] = {
                "pipeline": {
                    "maas": {"enabled": False},
                    "enable_layout": bool(getattr(config, "glm_enable_layout", False)),
                    "ocr_api": {
                        "api_host": ocr_api_host or "localhost",
                        "api_port": int(ocr_api_port or 8080),
                        "api_mode": ocr_api_mode,
                    },
                    "page_loader": {
                        "default_prompt": (
                            "Recognize the text in this cropped image patch and output ONLY the raw text. "
                            "Do NOT fabricate or guess content that does not clearly exist in the image. "
                            "If the image is completely blank, illegible, or contains only noise/background, "
                            "you MUST return exactly the word: EMPTY_PATCH"
                        )
                    }
                },
                "logging": {"level": "INFO"},
            }
            ocr_api_cfg = dict(glm_cfg["pipeline"]["ocr_api"])
            if config.glm_api_url:
                ocr_api_cfg["api_url"] = config.glm_api_url
            if ocr_api_model:
                ocr_api_cfg["model"] = ocr_api_model
            if ocr_api_path:
                ocr_api_cfg["api_path"] = ocr_api_path
            if config.glm_api_key:
                ocr_api_cfg["api_key"] = config.glm_api_key
            glm_cfg["pipeline"]["ocr_api"] = ocr_api_cfg

            fd, tmp_name = tempfile.mkstemp(suffix="_glmocr.yaml")
            os.close(fd)
            self._config_path = Path(tmp_name)
            self._config_path.write_text(yaml.safe_dump(glm_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            self._parser = GlmOcr(config_path=str(self._config_path))
            return

        kwargs: dict[str, object] = {
            "enable_layout": bool(getattr(config, "glm_enable_layout", False)),
            "mode": glm_mode,
        }
        if config.glm_api_key:
            kwargs["api_key"] = config.glm_api_key
        if config.glm_api_url:
            kwargs["api_url"] = config.glm_api_url
        if config.glm_model:
            kwargs["model"] = config.glm_model
        if config.glm_timeout is not None:
            kwargs["timeout"] = int(config.glm_timeout)
        if config.glm_env_file:
            kwargs["env_file"] = config.glm_env_file
        if ocr_api_host:
            kwargs["ocr_api_host"] = ocr_api_host
        if ocr_api_port is not None:
            kwargs["ocr_api_port"] = int(ocr_api_port)

        self._parser = GlmOcr(**kwargs)

    def close(self) -> None:
        parser = getattr(self, "_parser", None)
        self._parser = None
        if parser is None:
            parser_closed = True
        else:
            parser_closed = False
            try:
                parser.close()
                parser_closed = True
            except Exception:
                pass
        cfg_path = getattr(self, "_config_path", None)
        self._config_path = None
        if cfg_path is not None:
            try:
                Path(cfg_path).unlink(missing_ok=True)
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _extract_text(result: object) -> str:
        data: dict[str, Any] = {}
        if hasattr(result, "to_dict"):
            try:
                data = result.to_dict()
            except Exception:
                data = {}

        markdown = str(data.get("markdown_result") or "").strip()
        if markdown:
            return _strip_glm_markup(markdown)

        parts = _collect_glm_text_parts(data.get("json_result"))
        deduped: list[str] = []
        seen: set[str] = set()
        for part in parts:
            norm = _strip_glm_markup(part)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(norm)
        return "\n".join(deduped)

    def read_text(
        self,
        image: np.ndarray,
        psm: int | None = None,
        whitelist: str | None = None,
    ) -> str:
        _ = psm
        _ = whitelist

        fd, tmp_name = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            ok = cv2.imwrite(str(tmp_path), image)
            if not ok:
                return ""
            result = self._parser.parse(
                str(tmp_path),
                save_layout_visualization=False,
            )
            return self._extract_text(result)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


class PaddleOCRVLEngine:
    """Experimental fallback engine using PaddleOCR-VL (Vision-Language) model.

    Requires: pip install "paddleocr[doc-parser]" transformers
    """

    def __init__(self, config: OCRConfig):
        self.config = config
        # DLL configuration is run at module load, but we re-verify for VL specifically.
        if os.name == "nt":
            _configure_windows_cuda_dlls()

        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCRVL engine selected but dependency is missing. "
                "Install with: pip install 'paddleocr[doc-parser]' transformers"
            ) from exc

        self._vl = PaddleOCRVL(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=bool(getattr(config, "paddle_vl_use_layout_detection", True)),
            use_ocr_for_image_block=bool(getattr(config, "paddle_vl_use_ocr_for_image_block", True)),
        )

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image

    def _prepare_vl_image(self, image: np.ndarray) -> np.ndarray:
        rgb = self._to_rgb(image)
        max_side = getattr(self.config, "paddle_vl_max_side", None)
        if max_side is None:
            return rgb
        max_side = int(max_side)
        if max_side <= 0:
            return rgb
        h, w = rgb.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return rgb
        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _extract_from_blocks(self, blocks_list: Any) -> str:
        if not isinstance(blocks_list, list):
            return ""
        
        extracted = []
        for item in blocks_list:
            content = ""
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
            else:
                content = str(getattr(item, "content", "") or "").strip()
            
            if content:
                extracted.append(content)
        
        return "\n\n".join(extracted)

    def _maybe_format_ruhsat_html(self, raw_text: str) -> str | None:
        raw = str(raw_text or "")
        if "<table" not in raw.lower():
            return None

        clean = raw.replace("\r", "\n")
        clean = re.sub(r"<img[^>]*>", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"</(?:tr|table|td)>", "\n", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<br\s*/?>", "\n", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = collapse_spaces(clean)
        norm = normalize_turkish_ascii(clean).replace("TESÇIL", "TESCIL").replace("TESÇİL", "TESCIL").replace("N°", "NO")

        def grab(pattern: str) -> str:
            m = re.search(pattern, norm, flags=re.IGNORECASE | re.DOTALL)
            return collapse_spaces(m.group(1)) if m else ""

        place = grab(r"\(Y\.I\)\s*VERILDIGI IL / ILCE\s*(.*?)\s*\(A\)\s*PLAKA")
        plate = grab(r"\(A\)\s*PLAKA\s*(.*?)\s*\(B\)")
        first_reg = grab(r"\(B\)\s*ILK TESCIL TARIHI\s*(.*?)\s*\(Y\.2\)")
        reg_date = grab(r"\(I\)\s*TESCIL TARIHI\s*(.*?)\s*\(D\.1\)")
        brand = grab(r"\(D\.1\)\s*MARKASI\s*(.*?)\s*\(D\.2\)")
        vehicle_type = grab(r"\(D\.3\)\s*TICARI ADI\s*(.*?)\s*\(D\.4\)")
        model_year = grab(r"\(D\.4\)\s*MODEL YILI\s*(.*?)\s*\(J\)")
        cinsi = grab(r"\(D\.5\)\s*CINSI\s*(.*?)\s*\(R\)")
        color = grab(r"\(R\)\s*RENGI\s*(.*?)\s*\(P\.5\)")
        engine_no = grab(r"\(P\.5\)\s*MOTOR NO\s*(.*?)\s*\(E\)")
        vin_no = grab(r"\(E\)\s*SASE NO\s*(.*?)\s*\(G\.1\)")
        silindir = grab(r"\(P\.1\)\s*SILINDIR HACMI\s*(.*?)\s*\(P\.2\)")
        motor_gucu = grab(r"\(P\.2\)\s*MOTOR GUCU\s*(.*?)\s*\(P\.3\)")
        tax_no = grab(r"\(Y\.4\)\s*T\.C\. KIMLIK NO/VERGI NO\s*(.*?)\s*\(C\.1\.1\)")
        surname = grab(r"\(C\.1\.1\)\s*SOYADI/TICARI UNVANI\s*(.*?)\s*\(C\.1\.2\)")
        name = grab(r"\(C\.1\.2\)\s*ADI\s*(.*?)\s*\(C\.1\.3\)")
        address = grab(r"\(C\.1\.3\)\s*ADRESI\s*(.*?)\s*\(Z\.1\)")
        inspection = grab(r"\(Z\.2\)\s*DIGER BILGILER\s*MUA\.GEC\.TRH:\s*(.*?)\s*\(Z\.3\.4\)")
        approver = grab(r"\(Y\.5\)\s*ONAYLAYAN SICIL-IMZA\s*(.*?)\s*BELGE")
        serial = grab(r"BELGE\s*SERI\.?\s*(.*?)\s*$")
        serial = serial.replace("N°", "").replace("NO", "").strip()
        approver = re.sub(r"\s+(S\.N\.Y\.\d+)\s*$", r" (\1)", approver)

        if not plate or not brand or not tax_no:
            return None

        out_lines = [
            "🛡️ Araç Bilgileri",
            f"Plaka: {plate}",
            f"Marka / Ticari Ad: {_restore_tr_display(brand)} / {_restore_tr_display(vehicle_type)}",
            f"Model Yılı: {model_year}",
            f"Şase No (VIN): {vin_no}",
            f"Motor No: {engine_no}",
            f"İlk Tescil Tarihi: {first_reg}",
            f"Tescil Tarihi: {reg_date}",
            f"Cinsi: {_restore_tr_display(cinsi)}",
            f"Rengi: {_restore_tr_display(color)}",
            f"Silindir Hacmi / Gücü: {_restore_tr_display(silindir)} / {_restore_tr_display(motor_gucu)}",
            "👤 Sahip Bilgileri",
            f"T.C. Kimlik / Vergi No: {tax_no}",
            f"Adı Soyadı: {_restore_tr_display(name)} {_restore_tr_display(surname)}".strip(),
            f"Adresi: {_restore_tr_display(address)}",
            "📄 Belge Detayları",
            f"Belge Seri ve No: {_restore_tr_display(serial)}",
            f"Verildiği Yer: {_restore_tr_display(place)}",
            f"Muayene Geçerlilik Tarihi: {inspection}",
            f"Onaylayan: {_restore_tr_display(approver)}",
        ]
        return "\n".join(out_lines)

    def _maybe_format_ruhsat_summary(self, raw_text: str) -> str | None:
        if not raw_text:
            return None

        html_formatted = self._maybe_format_ruhsat_html(raw_text)
        if html_formatted:
            return html_formatted

        htmlish = str(raw_text).replace("\r", "\n")
        htmlish = re.sub(r"<img[^>]*>", " ", htmlish, flags=re.IGNORECASE)
        htmlish = re.sub(r"</(?:tr|table)>", "\n", htmlish, flags=re.IGNORECASE)
        htmlish = re.sub(r"<br\s*/?>", "\n", htmlish, flags=re.IGNORECASE)
        htmlish = re.sub(r"<[^>]+>", " ", htmlish)

        lines = [collapse_spaces(line) for line in htmlish.splitlines()]
        lines = [line for line in lines if line and re.search(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]", line)]
        if not lines:
            return None

        joined = collapse_spaces(" ".join(lines))
        joined = joined.replace("TESÇIL", "TESCIL").replace("TESÇİL", "TESCIL")
        joined = joined.replace("N°", "NO").replace("№", "NO")
        norm = normalize_turkish_ascii(joined)

        if "KIMLIK NO/VERGI NO" not in norm or "TESCIL TARIHI" not in norm:
            return None

        plate_match = re.search(r"\b\d{2}[A-Z]{1,3}\d{2,4}\b", norm)
        first_reg_match = re.search(r"ILK\s+TESCIL\s+TARIHI\s+(\d{2}/\d{2}/\d{4})", norm)
        reg_date_match = re.search(r"TESCIL\s+TARIHI\s+(\d{2}/\d{2}/\d{4})", norm)
        brand_match = re.search(r"MARKASI\s+([A-Z-]+)", norm)
        ticari_adi_match = re.search(r"TICARI\s+ADI\s+(.+?)\s+D\.4\s+MODEL\s+YILI", norm)
        type_match = re.search(r"TIPI\s+(.+?)\s+D\.3\s+TICARI\s+ADI", norm)
        model_year_match = re.search(r"D\.4\s+MODEL\s+YILI\s+(19\d{2}|20\d{2})", norm)
        tax_match = re.search(r"T\.?C\.?\s+KIMLIK\s+NO/?VERGI\s+NO\s+(\d{10,11})", norm)
        engine_match = re.search(r"\b([A-Z0-9]{10,20})\s+(VF[A-Z0-9]{15})\b", norm)
        hacim_guc_match = re.search(r"SILINDIR\s+HACMI\s+(\d+)\s*CM\S*\s+MOTOR\s+GUCU\s+(\d+)\s*KW", norm)
        muayene_match = re.search(r"MUA\.?GEC\.?TRH\s*:?\s*(\d{2}-\d{2}-\d{4})", norm)
        serial_match = re.search(r"BELGE\s+SERI\.?\s*([A-Z]{2})\s+NO\s+(\d{5,7})", norm)

        if not plate_match or not brand_match or not tax_match:
            return None

        place = ""
        prefix = norm.split(plate_match.group(0), 1)[0]
        place = re.sub(r"^.*?VERILDIGI\s+IL\s*/\s*ILCE\s*", "", prefix).strip(" :-/")

        owner_name = ""
        owner_surname = ""
        address = ""
        owner_match = re.search(r"SOYADI/?TICARI\s+UNVANI\s+([A-Z]+)\s+C\.1\.2\s+ADI\s+([A-Z]+)", norm)
        if owner_match:
            owner_surname = owner_match.group(1)
            owner_name = owner_match.group(2)
        addr_match = re.search(r"C\.1\.3\s+ADRESI\s+(.+?)\s+Z\.1\s+ARAC\s+UZERINDE\s+HAK", norm)
        if addr_match:
            address = addr_match.group(1)

        cinsi = ""
        color = ""
        engine_no = ""
        vin_no = ""
        if engine_match:
            engine_no = engine_match.group(1)
            vin_no = engine_match.group(2)
            before_engine = norm.split(engine_no, 1)[0]
            m = re.search(r"ARAC\s+SINIFI\s+[A-Z0-9]+\s+(.+)$", before_engine)
            if m:
                tail = m.group(1).strip()
                color_match = re.search(r"(GRI(?:\s*\([^)]+\))?|BEYAZ(?:\s*\([^)]+\))?|SIYAH(?:\s*\([^)]+\))?|MAVI(?:\s*\([^)]+\))?|KIRMIZI(?:\s*\([^)]+\))?)$", tail)
                if color_match:
                    color = color_match.group(1)
                    cinsi = tail[: color_match.start()].strip()
                else:
                    cinsi = tail

        approver = ""
        approver_match = re.search(r"ONAYLAYAN\s+SICIL-IMZA\s+(.+?)\s+(S\.N\.Y\.\d+)", joined, re.IGNORECASE)
        if approver_match:
            approver = f"{approver_match.group(1).strip()} ({approver_match.group(2).strip()})"

        cinsi_match = re.search(r"D\.5\s+CINSI\s+(.+?)\s+R\s+RENGI", norm)
        renk_match = re.search(r"R\s+RENGI\s+(.+?)\s+P\.5\s+MOTOR\s+NO", norm)
        if cinsi_match:
            cinsi = cinsi_match.group(1).strip()
        if renk_match:
            color = renk_match.group(1).strip()

        out_lines = [
            "🛡️ Araç Bilgileri",
            f"Plaka: {plate_match.group(0)}",
            f"Marka / Ticari Ad: {_restore_tr_display(brand_match.group(1))} / {_restore_tr_display(ticari_adi_match.group(1) if ticari_adi_match else (type_match.group(1) if type_match else ''))}",
            f"Model Yılı: {model_year_match.group(1) if model_year_match else ''}",
            f"Şase No (VIN): {vin_no}",
            f"Motor No: {engine_no}",
            f"İlk Tescil Tarihi: {first_reg_match.group(1) if first_reg_match else ''}",
            f"Tescil Tarihi: {reg_date_match.group(1) if reg_date_match else ''}",
            f"Cinsi: {_restore_tr_display(cinsi)}",
            f"Rengi: {_restore_tr_display(color)}",
            f"Silindir Hacmi / Gücü: {hacim_guc_match.group(1) if hacim_guc_match else ''} cm³ / {hacim_guc_match.group(2) if hacim_guc_match else ''} kw",
            "👤 Sahip Bilgileri",
            f"T.C. Kimlik / Vergi No: {tax_match.group(1)}",
            f"Adı Soyadı: {_restore_tr_display(owner_name)} {_restore_tr_display(owner_surname)}".strip(),
            f"Adresi: {_restore_tr_display(address)}",
            "📄 Belge Detayları",
            f"Belge Seri ve No: {serial_match.group(1)} {serial_match.group(2)}" if serial_match else "Belge Seri ve No: ",
            f"Verildiği Yer: {_restore_tr_display(place)}",
            f"Muayene Geçerlilik Tarihi: {muayene_match.group(1) if muayene_match else ''}",
            f"Onaylayan: {approver}",
        ]
        return "\n".join(out_lines)

    def read_text(
        self,
        image: np.ndarray,
        psm: int | None = None,
        whitelist: str | None = None,
        prompt: str | None = None,
    ) -> str:
        _ = psm
        _ = whitelist
        rgb = self._prepare_vl_image(image)
        try:
            # For PaddleOCR-VL-1.5, prompt_label can be used for VLM tasks.
            # If prompt is None, it uses the default OCR behavior.
            res = self._vl.predict(rgb, prompt_label=prompt)
            if not res or len(res) == 0:
                return ""

            first_res = res[0]
            
            # 1. Try direct attribute access
            parsing_res = getattr(first_res, "parsing_res_list", None)
            val = self._extract_from_blocks(parsing_res)
            if val:
                return self._maybe_format_ruhsat_summary(val) or val

            # 2. Try to_dict() access
            if hasattr(first_res, "to_dict"):
                data = first_res.to_dict()
                val = self._extract_from_blocks(data.get("parsing_res_list"))
                if val:
                    return self._maybe_format_ruhsat_summary(val) or val
                
                # Check for other potential keys
                markdown = str(data.get("markdown") or "").strip()
                if markdown:
                    return self._maybe_format_ruhsat_summary(markdown) or markdown
                text = str(data.get("text") or "").strip()
                if text:
                    return self._maybe_format_ruhsat_summary(text) or text

            # 3. Try dict access (if first_res is already a dict)
            if isinstance(first_res, dict):
                val = self._extract_from_blocks(first_res.get("parsing_res_list"))
                if val:
                    return self._maybe_format_ruhsat_summary(val) or val
                raw = str(first_res.get("markdown") or first_res.get("text") or "").strip()
                return self._maybe_format_ruhsat_summary(raw) or raw
                
            raw = str(first_res).strip()
            return self._maybe_format_ruhsat_summary(raw) or raw
        except Exception as exc:
            import logging
            logging.getLogger("ocrreader").warning("PaddleOCRVL inference failed: %s", exc)
            return ""


def create_glm_fallback_engine(config: OCRConfig) -> GlmFallbackEngine | PaddleOCRVLEngine | None:
    if not bool(getattr(config, "glm_fallback_enabled", False)):
        return None
    
    if getattr(config, "glm_model", "") == "paddle_vl":
        return PaddleOCRVLEngine(config)
        
    return GlmFallbackEngine(config)


class HybridOCREngine:
    """Hybrid engine: PaddleOCR for full-page iter_words, Tesseract for crop read_text.

    Rationale
    ---------
    PaddleOCR excels at full-document scene-text detection and recognition in one
    GPU pass (iter_words).  When called repeatedly on small, already-preprocessed
    field crops, however, each predict() incurs a CPU→GPU tensor-transfer + inference
    + GPU→CPU transfer round-trip that can take hundreds of milliseconds — far slower
    than Tesseract which runs in-process on the CPU.  Using Tesseract solely for the
    ~30 per-image crop read_text() calls typically cuts per-image wall-clock time by
    5-10× while retaining PaddleOCR's superior page-level detection quality.
    """

    def __init__(self, paddle_engine: PaddleOCREngine, tesseract_engine: TesseractEngine):
        self._paddle = paddle_engine
        self._tess = tesseract_engine
        # Expose config from the primary (Paddle) engine so callers that read
        # engine.config still see paddle settings.
        self.config = paddle_engine.config

    def iter_words(self, image, psm=None, min_conf: float = 20.0):
        """Full-page OCR via PaddleOCR GPU."""
        return self._paddle.iter_words(image, psm=psm, min_conf=min_conf)

    def read_text(self, image, psm=None, whitelist=None) -> str:
        """Per-crop OCR via Tesseract (fast, in-process, no GPU overhead)."""
        return self._tess.read_text(image, psm=psm, whitelist=whitelist)


def create_ocr_engine(config: OCRConfig) -> "OCREngine | HybridOCREngine":
    engine_name = str(getattr(config, "engine", "tesseract") or "tesseract").strip().lower()
    if engine_name in {"tesseract", "tess"}:
        return TesseractEngine(config)
    if engine_name in {"paddle", "paddleocr"}:
        paddle_eng = PaddleOCREngine(config)
        crop_eng_name = str(getattr(config, "paddle_crop_engine", None) or "").strip().lower()
        if crop_eng_name in {"tesseract", "tess"}:
            tess_eng = TesseractEngine(config)
            return HybridOCREngine(paddle_eng, tess_eng)
        return paddle_eng
    raise ValueError(f"Unknown OCR engine: {engine_name}. Supported: tesseract, paddle")
