from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .config import PipelineConfig
from .types import Rect


@dataclass(frozen=True)
class PreprocessResult:
    original_shape: tuple[int, int, int]
    document_quad: list[list[float]]
    skew_angle_deg: float
    normalized_image: np.ndarray


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _largest_quad_from_contours(
    contours: list[np.ndarray],
    image_w: int,
    image_h: int,
    min_area: float,
) -> np.ndarray | None:
    def touches_border(points: np.ndarray, margin: int = 6) -> bool:
        return bool(
            np.any(points[:, 0] <= margin)
            or np.any(points[:, 1] <= margin)
            or np.any(points[:, 0] >= image_w - margin)
            or np.any(points[:, 1] >= image_h - margin)
        )

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            if touches_border(quad):
                continue
            return quad

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.float32)
        if cv2.contourArea(box) >= min_area and not touches_border(box):
            return box

    return None


def detect_document_quad(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    h, w = image.shape[:2]
    min_area = config.document_detector.min_area_ratio * w * h

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (90, 40, 25), (145, 255, 255))
    kernel = np.ones((5, 5), dtype=np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = _largest_quad_from_contours(contours, w, h, min_area)
    if quad is not None:
        return _order_points(quad)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        gray,
        config.document_detector.canny_threshold1,
        config.document_detector.canny_threshold2,
    )
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = _largest_quad_from_contours(contours, w, h, min_area)
    if quad is not None:
        return _order_points(quad)

    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )


def _warp_perspective(image: np.ndarray, quad: np.ndarray, output_w: int, output_h: int) -> np.ndarray:
    dst = np.array(
        [[0, 0], [output_w - 1, 0], [output_w - 1, output_h - 1], [0, output_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(_order_points(quad), dst)
    return cv2.warpPerspective(image, matrix, (output_w, output_h), flags=cv2.INTER_CUBIC)


def estimate_skew_angle_deg(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    lines = cv2.HoughLinesP(
        bw,
        rho=1,
        theta=np.pi / 180.0,
        threshold=120,
        minLineLength=max(100, image.shape[1] // 6),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -30 <= angle <= 30:
            angles.append(float(angle))

    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image while keeping full bounds (no crop)."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(float(mat[0, 0]))
    sin = abs(float(mat[0, 1]))
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    mat[0, 2] += (new_w / 2.0) - center[0]
    mat[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _enhance_low_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE + light denoise to stabilize low-contrast gray scans."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    return out


def _choose_best_orientation(image: np.ndarray) -> np.ndarray:
    """
    Try 0/90/270 and keep orientation with highest horizontal-line energy.
    Helps portrait/rotated motorcycle documents before quad detection.
    """

    h, w = image.shape[:2]
    # Keep already-landscape pages unchanged (protects standard ruhsat scans).
    if w >= h * 0.9:
        return image

    def score(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        lines = cv2.HoughLinesP(
            bw,
            rho=1,
            theta=np.pi / 180.0,
            threshold=110,
            minLineLength=max(80, img.shape[1] // 6),
            maxLineGap=20,
        )
        if lines is None:
            return 0.0

        total = 0.0
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = l
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if -25.0 <= angle <= 25.0:
                total += float(np.hypot(x2 - x1, y2 - y1))
        return total

    candidates = (
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )
    return max(candidates, key=score)


def _correct_orientation_safe(image: np.ndarray) -> np.ndarray:
    """Optional OSD orientation correction (kept disabled by default)."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        osd = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        rotate = int(float(osd.get("rotate", 0) or 0)) % 360
        orientation_conf = float(osd.get("orientation_conf", 0.0) or 0.0)
    except Exception:
        return image

    if rotate not in {90, 180, 270} or orientation_conf < 4.0:
        return image

    if rotate == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotate == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def preprocess_document(image: np.ndarray, config: PipelineConfig) -> PreprocessResult:
    oriented = _choose_best_orientation(image)
    enhanced = _enhance_low_contrast(oriented)

    quad = detect_document_quad(enhanced, config)
    normalized = _warp_perspective(enhanced, quad, config.output_width, config.output_height)
    if getattr(config, "orientation_osd_enabled", False):
        normalized = _correct_orientation_safe(normalized)
    skew = 0.0

    if config.deskew.enabled:
        skew = estimate_skew_angle_deg(normalized)
        clipped = max(-float(config.deskew.max_correction_deg), min(float(config.deskew.max_correction_deg), float(skew)))
        if abs(clipped) >= 0.3:
            normalized = _rotate_bound(normalized, -clipped)
            normalized = cv2.resize(normalized, (config.output_width, config.output_height), interpolation=cv2.INTER_CUBIC)

    return PreprocessResult(
        original_shape=oriented.shape,
        document_quad=quad.tolist(),
        skew_angle_deg=skew,
        normalized_image=normalized,
    )


def crop(image: np.ndarray, rect: Rect) -> np.ndarray:
    clipped = rect.clip(image.shape[1], image.shape[0])
    return image[clipped.y : clipped.y + clipped.h, clipped.x : clipped.x + clipped.w]

