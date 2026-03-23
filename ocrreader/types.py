from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def clip(self, max_w: int, max_h: int) -> "Rect":
        x = min(max(self.x, 0), max_w - 1)
        y = min(max(self.y, 0), max_h - 1)
        w = min(max(self.w, 1), max_w - x)
        h = min(max(self.h, 1), max_h - y)
        return Rect(x=x, y=y, w=w, h=h)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def union_rects(rects: list[Rect]) -> Rect:
    if not rects:
        return Rect(0, 0, 1, 1)
    x1 = min(r.x for r in rects)
    y1 = min(r.y for r in rects)
    x2 = max(r.x + r.w for r in rects)
    y2 = max(r.y + r.h for r in rects)
    return Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

