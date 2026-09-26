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
from .unit_catalog import recognize_card_identity, recognize_large_hero_identity


_BASE = (1280, 720)
_ROWS = {"troops": (545, 188, 1255, 305), "spells": (545, 365, 995, 483)}


def recognize_army_heroes(
    screenshot_path: str | Path, capacity: dict[str, object] | None, *,
    baseline_resolution: tuple[int, int] = _BASE, client_version: str | None = None,
) -> dict[str, object]:
    """Read independently bordered large hero cards on the My Army page.

    Card geometry establishes that a card exists. Portrait evidence establishes
    its ID separately; the same hero may appear in any visible column.
    """
    path = Path(screenshot_path)
    result: dict[str, object] = {"frame": str(path), "cards": [], "complete": False,
                                  "unknowns": [], "layout": "my_army_large_heroes_v1"}
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        image = None
    if image is None or abs(image.shape[1] / image.shape[0] - 16 / 9) > .01:
        result["unknowns"].append("hero_image_unreadable_or_aspect_changed")
        return result
    image = cv2.resize(image, _BASE, interpolation=cv2.INTER_AREA)
    if not isinstance(capacity, dict) or type(capacity.get("used")) is not int or type(capacity.get("capacity")) is not int:
        result["unknowns"].append("hero_capacity_unreadable")
        return result
    used, total = capacity["used"], capacity["capacity"]
    if not (0 <= used <= total <= 6):
        result["unknowns"].append("hero_capacity_invalid")
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # The 18.600.7 large-card frame has two long, separate side edges. Locate
    # those edges afresh on every frame; a column number never supplies an ID.
    edge = np.abs(gray[220:650, 1:] - gray[220:650, :-1]).mean(axis=0)
    detected = []
    for group in range(4):
        first = 17 + 130 * group
        candidates = []
        for left in range(first, first + 14):
            for right in range(left + 122, left + 128):
                if right >= edge.size:
                    continue
                left_score, right_score = float(edge[left - 1]), float(edge[right - 1])
                if left_score >= 28 and right_score >= 35:
                    candidates.append((min(left_score, right_score), left, right,
                                       left_score, right_score))
        if not candidates:
            continue
        strongest = max(item[0] for item in candidates)
        # The two pixels of one bevel can have virtually identical edge
        # strength. Anchor the outer edge consistently so pet/equipment icon
        # crops remain aligned across fresh observations.
        _, left, right, left_score, right_score = min(
            (item for item in candidates if item[0] >= strongest - 2),
            key=lambda item: item[1])
        box = (left, 185, right, 668)
        identity = recognize_large_hero_identity(image, box, client_version=client_version)
        card_box = list(scale_box(box, from_resolution=_BASE, to_resolution=baseline_resolution))
        card = {"unit_id": identity["unit_id"], "kind": "hero", "source": "army",
                "count": 1, "level": None, "available": None,
                "confidence": identity["confidence"], "card_bbox": card_box,
                "bbox": card_box,
                "point": [(card_box[0] + card_box[2]) // 2, (card_box[1] + card_box[3]) // 2],
                "evidence": {"geometry": {"method": "observed_long_card_edges",
                                          "left_score": round(left_score, 5),
                                          "right_score": round(right_score, 5)},
                             "identity": identity}}
        detected.append(card)
        if identity["unit_id"] is None:
            result["unknowns"].append("hero_identity_unavailable")
    result["cards"] = detected
    if len(detected) != used:
        result["unknowns"].append("hero_card_count_disagrees_with_capacity")
    ids = [card["unit_id"] for card in detected if card["unit_id"] is not None]
    if len(set(ids)) != len(ids):
        result["unknowns"].append("duplicate_hero_identity")
    result["complete"] = not result["unknowns"]
    return result


def recognize_army_manifest(
    screenshot_path: str | Path, provider: OCRProvider, texts: Sequence[OCRText], *,
    baseline_resolution: tuple[int, int] = _BASE, client_version: str | None = None,
    capacities: dict[str, object] | None = None,
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
    native_image = image
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
        header = capacities.get(group) if isinstance(capacities, dict) else None
        explicit_zero = (isinstance(header, dict) and type(header.get("used")) is int
                         and header["used"] == 0 and type(header.get("capacity")) is int
                         and header["capacity"] >= 0)
        if not boxes and not explicit_zero:
            row_unknowns.append("no_complete_card_borders")
        if boxes and explicit_zero:
            row_unknowns.append("visible_cards_conflict_with_zero_capacity")
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
            "zero_capacity_header": header if explicit_zero else None,
        }
        unknowns.extend({"group": group, "reason": reason} for reason in dict.fromkeys(row_unknowns))
        for box, sides in boxes:
            count, count_evidence = _read_count(path, provider, canonical, box, (width, height), baseline_resolution,
                                                image, native_image)
            card_box = list(scale_box(box, from_resolution=_BASE, to_resolution=baseline_resolution))
            identity = recognize_card_identity(image, box, "troop" if group == "troops" else "spell",
                                               surface="army", client_version=client_version)
            result[group].append({"bbox": card_box, "point": [(card_box[0] + card_box[2]) // 2,
                                                                    (card_box[1] + card_box[3]) // 2],
                                  "kind": "troop" if group == "troops" else "spell", "source": "army",
                                  "unit_id": identity["unit_id"], "identity_confidence": identity["confidence"],
                                  "level": None, "available": None,
                                  "count": count, "evidence": {
                "geometry": {"method": "independent_card_border", "side_support": sides,
                             "bbox_at_1280x720": list(box)}, "count": count_evidence,
                "identity": identity,
            }})
            if count is None:
                unknowns.append({"group": group, "bbox": card_box, "reason": "missing_or_ambiguous_exact_quantity"})
    result["complete"] = not unknowns
    result["identity_complete"] = result["complete"] and all(
        card["unit_id"] is not None for group in ("troops", "spells") for card in result[group])
    return result


def _confident(item: OCRText) -> bool:
    return math.isfinite(item.confidence) and .9 <= item.confidence <= 1


def _inside(item: OCRText, roi: tuple[int, int, int, int]) -> bool:
    if item.bbox is None:
        return False
    left, top, right, bottom = item.bbox
    return roi[0] <= (left + right) / 2 <= roi[2] and roi[1] <= (top + bottom) / 2 <= roi[3]


def _read_count(path, provider, texts, box, native_resolution, baseline_resolution,
                canonical_image, native_image):
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
        # The spell cards' purple/icy background can defeat full-frame and
        # native line OCR. Re-read only the quantity glyphs with several
        # contrast transforms, retaining the exact text and confidence. This
        # never guesses from card order, capacity, or the neighboring cards.
        reader = getattr(provider, "recognize_line_image", None)
        if callable(reader) and len(candidates) == 0:
            left, top = box[:2]
            tiny = canonical_image[top + 1:top + 27, left:left + 40]
            gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
            variants = {"gray": gray, "inverted": 255 - gray,
                        "threshold": cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]}
            native_box = scale_box((left, top + 1, left + 40, top + 27),
                                   from_resolution=_BASE, to_resolution=native_resolution)
            nl, nt, nr, nb = native_box
            native_gray = cv2.cvtColor(native_image[nt:nb, nl:nr], cv2.COLOR_BGR2GRAY)
            variants.update({"native_gray": native_gray, "native_inverted": 255 - native_gray,
                             "native_threshold": cv2.threshold(native_gray, 170, 255, cv2.THRESH_BINARY)[1]})
            reads = []
            for variant, sample in variants.items():
                output = reader(sample)
                if isinstance(output, (list, tuple)):
                    reads.extend({"variant": variant, "text": item.text,
                                  "confidence": item.confidence} for item in output
                                 if isinstance(item, OCRText))
            evidence["contrast_reads"] = reads
            valid = [item for item in reads if math.isfinite(item["confidence"])
                     and item["confidence"] >= .9
                     and re.fullmatch(r"[xX×]\s*[0-9]+", item["text"].strip())]
            values = {int(re.sub(r"\D", "", item["text"])) for item in valid}
            if len(values) == 1 and next(iter(values)) > 0:
                evidence["source"] = "contrast_line_roi"
                return next(iter(values)), evidence
        return None, evidence
    count = int(re.sub(r"\D", "", candidates[0].text))
    return count if count > 0 else None, evidence
