from __future__ import annotations

import re

_TR_TO_ASCII = str.maketrans(
    {
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "I": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
        "Â": "A",
        "Î": "I",
        "Û": "U",
        "ç": "C",
        "ğ": "G",
        "ı": "I",
        "i": "I",
        "ö": "O",
        "ş": "S",
        "ü": "U",
        "â": "A",
        "î": "I",
        "û": "U",
    }
)


def normalize_turkish_ascii(text: str) -> str:
    return text.translate(_TR_TO_ASCII).upper()


def normalize_for_match(text: str) -> str:
    text = normalize_turkish_ascii(text)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

