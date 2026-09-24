"""Independent numeric-card manifest for the observed Chinese My Army row layout.

This checks visible contiguous troop/spell rows, not troop identity, housing space,
siege machines, clan reinforcements, or heroes. Scrolled/clipped rows are incomplete.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
from collections.abc import Sequence

import cv2
import numpy as np

from .locator import scale_box
from .ocr import OCRProvider, OCRText


_BASE = (1280, 720)
_ROWS = {"troops": (545, 188, 1255, 305), "spells": (545, 365, 995, 483)}


def recognize_army_manifest(
    screenshot_path: str | Path, provider: OCRProvider, texts: Sequence[OCRText], *,
    baseline_resolution: tuple[int, int] = _BASE,
) -> dict[str, object]:
    """`texts` and returned card boxes use `baseline_resolution`; provider boxes are native."""
    path = Path(screenshot_path)
    result = {"frame": str(path), "layout": "my_army_rows_v1", "supported_layout": False,
              "complete": False, "troops": [], "spells": [], "unknowns": [],
              "source_resolution": None, "baseline_resolution": list(baseline_resolution),
              "layout_evidence": {"anchors": {}, "rows": {}}}
    unknowns = result["unknowns"]
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        image = None
    if image is None:
        unknowns.append({"reason": "image_unreadable"})
        return result
    height, width = image.shape[:2]
    result["source_resolution"] = [width, height]
    if abs(width / height - 1280 / 720) > .01:
        unknowns.append({"reason": "unsupported_aspect_ratio"})
    image = cv2.resize(image, _BASE, interpolation=cv2.INTER_AREA)
    canonical = [OCRText(item.text, item.confidence, None if item.bbox is None else scale_box(
        item.bbox, from_resolution=baseline_resolution, to_resolution=_BASE)) for item in texts]
    for label, roi in (("我的军队", (210, 55, 310, 100)), ("已保存的配置", (570, 55, 710, 100)),
                       ("精选库", (950, 55, 1045, 100))):
        anchors = [item for item in canonical if item.text.strip() == label and _confident(item)
                   and _inside(item, roi)]
        if len(anchors) != 1:
            unknowns.append({"reason": "missing_or_ambiguous_layout_anchor", "label": label})
        else:
            result["layout_evidence"]["anchors"][label] = asdict(anchors[0])
    result["supported_layout"] = not unknowns
    if not result["supported_layout"]:
        return result

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    foreground = (((hsv[:, :, 0] >= 40) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 100))
                  | ((hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 190))).astype(np.uint8)
    for group, roi in _ROWS.items():
        left, top, right, bottom = roi
        mask = foreground[top:bottom, left:right]
        uncovered = mask.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (96 <= w <= 101 and 95 <= h <= 101):
                continue
            sides = [float(part.mean()) for part in (
                mask[y + 8:y + h - 8, x:x + 4], mask[y + 8:y + h - 8, x + w - 4:x + w],
                mask[y:y + 4, x + 10:x + w - 10], mask[y + h - 4:y + h, x + 10:x + w - 10],
            )]
            if min(sides[0], sides[1], sides[3]) < .7 or sides[2] < .55:
                continue
            boxes.append(((x + left, y + top, x + left + w, y + top + h), sides))
            uncovered[y:y + h, x:x + w] = 0
        boxes.sort(key=lambda item: item[0][0])
        row_unknowns = []
        if not boxes:
            row_unknowns.append("no_complete_card_borders")
        expected_top = 194 if group == "troops" else 373
        for index, (box, _) in enumerate(boxes):
            if abs(box[1] - expected_top) > 3:
                row_unknowns.append("card_vertical_alignment_changed")
            if index == 0 and abs(box[0] - 558) > 3:
                row_unknowns.append("leftmost_card_missing_or_scrolled")
            if index and not 3 <= box[0] - boxes[index - 1][0][2] <= 8:
                row_unknowns.append("card_row_not_contiguous")
        # Every colored/bright border fragment, including a clipped extra card in
        # the right tail, must belong to a complete card independent of its OCR.
        residual = int(np.count_nonzero(uncovered))
        if residual > 8:
            row_unknowns.append("uncovered_card_pixels_or_clipped_tail")
        result["layout_evidence"]["rows"][group] = {
            "roi_at_1280x720": list(roi), "detected_card_count": len(boxes),
            "uncovered_foreground_pixels": residual, "complete": not row_unknowns,
        }
        unknowns.extend({"group": group, "reason": reason} for reason in dict.fromkeys(row_unknowns))
        for box, sides in boxes:
            count, count_evidence = _read_count(path, provider, canonical, box, (width, height), baseline_resolution)
            card_box = list(scale_box(box, from_resolution=_BASE, to_resolution=baseline_resolution))
            result[group].append({"bbox": card_box, "count": count, "evidence": {
                "geometry": {"method": "independent_card_border", "side_support": sides,
                             "bbox_at_1280x720": list(box)}, "count": count_evidence,
            }})
            if count is None:
                unknowns.append({"group": group, "bbox": card_box, "reason": "missing_or_ambiguous_exact_quantity"})
    result["complete"] = not unknowns
    return result


def _confident(item: OCRText) -> bool:
    return math.isfinite(item.confidence) and .9 <= item.confidence <= 1


def _inside(item: OCRText, roi: tuple[int, int, int, int]) -> bool:
    if item.bbox is None:
        return False
    left, top, right, bottom = item.bbox
    return roi[0] <= (left + right) / 2 <= roi[2] and roi[1] <= (top + bottom) / 2 <= roi[3]


def _read_count(path, provider, texts, box, native_resolution, baseline_resolution):
    roi = (box[0], box[1], box[0] + 70, box[1] + 28)
    candidates = [item for item in texts if _confident(item) and _inside(item, roi)
                  and re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())]
    source = "full_ocr"
    if not candidates:
        native_roi = scale_box(roi, from_resolution=_BASE, to_resolution=native_resolution)
        candidates = [OCRText(item.text, item.confidence, None if item.bbox is None else scale_box(
            item.bbox, from_resolution=native_resolution, to_resolution=_BASE))
            for item in provider.recognize_line(path, native_roi)
            if _confident(item) and re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())]
        candidates = [item for item in candidates if _inside(item, roi)]
        source = "line_roi"
    evidence = {"source": source, "roi_at_1280x720": list(roi), "reads": [
        asdict(OCRText(item.text, item.confidence, None if item.bbox is None else scale_box(
            item.bbox, from_resolution=_BASE, to_resolution=baseline_resolution))) for item in candidates]}
    if len(candidates) != 1:
        return None, evidence
    count = int(re.sub(r"\D", "", candidates[0].text))
    return count if count > 0 else None, evidence
