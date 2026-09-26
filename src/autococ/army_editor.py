"""Read-only recognition of the home-village army configuration surfaces.

This module locates navigation controls, but deliberately does not infer a unit
from its position in a saved plan or picker. A control for changing a unit is
only emitted after a named portrait sample and an independent count are read.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from collections.abc import Sequence

import cv2
import numpy as np

from .locator import scale_box
from .ocr import OCRProvider, OCRText
from .unit_catalog import get_unit

_BASE = (1280, 720)
_TITLE = re.compile(r"编辑军队配置\s*([0-9一二三四五六七八九十]+)")
_CAPACITY = re.compile(r"^(\d+)\s*/\s*(\d+)$")


def _inside(item: OCRText, roi: tuple[int, int, int, int]) -> bool:
    if item.bbox is None:
        return False
    l, t, r, b = item.bbox
    return roi[0] <= (l + r) / 2 <= roi[2] and roi[1] <= (t + b) / 2 <= roi[3]


def _point(item: OCRText) -> list[int]:
    assert item.bbox is not None
    l, t, r, b = item.bbox
    return [(l + r) // 2, (t + b) // 2]


def _one(texts: Sequence[OCRText], label: str, roi: tuple[int, int, int, int]) -> OCRText | None:
    matches = [item for item in texts if item.text.strip() == label and item.confidence >= .88
               and _inside(item, roi)]
    return matches[0] if len(matches) == 1 else None


def _control(action: str, item: OCRText, **extra: object) -> dict[str, object]:
    return {"action": action, "point": _point(item), "confidence": item.confidence,
            "enabled": True, "cost_free": True, "evidence": asdict(item), **extra}


def _capacities(texts: Sequence[OCRText], *, edit: bool) -> dict[str, dict[str, int]]:
    # A title is required before these geometry windows are meaningful. These
    # numbers are not used to invent card identity or quantity.
    regions = ({"troop": (550, 55, 705, 195), "spell": (550, 230, 705, 390),
                "siege": (940, 230, 1120, 390)} if edit else
               {"troop": (560, 105, 850, 175), "spell": (560, 345, 850, 415)})
    result = {}
    for kind, roi in regions.items():
        matches = [(_CAPACITY.fullmatch(item.text.strip()), item) for item in texts
                   if item.confidence >= .88 and _inside(item, roi)]
        matches = [(m, item) for m, item in matches if m is not None]
        if len(matches) == 1:
            m, _ = matches[0]
            used, total = int(m.group(1)), int(m.group(2))
            if 0 <= used <= total:
                result[kind] = {"used": used, "total": total}
    return result


def _picker_visible(image: np.ndarray) -> bool:
    # The expanded picker fills both lower rows with bright framed cards while
    # darkening the editor above. Works for coloured troop and gray spell cards.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upper = gray[385:415, 50:1200]
    lower = gray[460:665, 50:1200]
    return float(np.mean(upper)) < 75 and float(np.mean(lower > 160)) > .20


def recognize_army_editor(
    path: str | Path, provider: OCRProvider, texts: Sequence[OCRText], *,
    baseline_resolution: tuple[int, int] = _BASE,
) -> dict[str, object]:
    """Locate the observed page and safe controls in one fresh frame.

    ``texts`` and returned coordinates use ``baseline_resolution``. Missing
    hero, siege, or picker samples remain unknown; ``ready`` only means the
    navigation surface is anchored, not that a recipe can be applied.
    """
    path = Path(path)
    result: dict[str, object] = {"frame": str(path), "surface": "unknown", "ready": False,
                                 "cards": [], "controls": [], "presets": [],
                                 "capacities": {}, "unknowns": [],
                                 "complete": False}
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
    if image is None:
        result["unknowns"].append("image_unreadable")
        return result
    height, width = image.shape[:2]
    if abs(width / height - 16 / 9) > .01:
        result["unknowns"].append("unsupported_aspect_ratio")
        return result
    image = cv2.resize(image, _BASE, interpolation=cv2.INTER_AREA)
    canonical = [OCRText(item.text, item.confidence, None if item.bbox is None else scale_box(
        item.bbox, from_resolution=baseline_resolution, to_resolution=_BASE)) for item in texts]
    edit_titles = [item for item in canonical if item.bbox is not None and
                   _TITLE.search(item.text.replace(" ", "")) and item.confidence >= .88 and
                   _inside(item, (460, 25, 780, 110))]
    has_editor_capacity = bool(_capacities(canonical, edit=True).get("troop"))
    picker = has_editor_capacity and _picker_visible(image)
    if len(edit_titles) == 1 or picker:
        result["surface"] = "picker" if picker else "edit"
        result["ready"] = len(edit_titles) == 1 or (picker and has_editor_capacity)
        result["capacities"] = _capacities(canonical, edit=True)
        if edit_titles:
            result["preset_id"] = _TITLE.search(edit_titles[0].text.replace(" ", "")).group(1)
        # Card and picker identity require independent, surface-specific sample
        # coverage. An unrecognised card must never produce an increment tap.
        result["unknowns"].append("editor_card_identity_samples_unavailable")
        return result
    saved_title = _one(canonical, "已保存的配置", (545, 40, 735, 105))
    army_title = _one(canonical, "我的军队", (185, 40, 340, 105))
    if saved_title is None or army_title is None:
        result["unknowns"].append("army_tabs_unanchored")
        return result
    use = [item for item in canonical if item.text.strip() == "使用" and item.confidence >= .88
           and _inside(item, (1100, 125, 1240, 670))]
    edits = [item for item in canonical if item.text.strip() == "编辑" and item.confidence >= .88
             and _inside(item, (1100, 125, 1240, 670))]
    if use or edits:
        result["surface"] = "saved"
        result["ready"] = True
        if len(use) != len(edits):
            result["unknowns"].append("saved_row_controls_incomplete")
        for index, (use_item, edit_item) in enumerate(zip(sorted(use, key=lambda x: x.bbox[1]),
                                                          sorted(edits, key=lambda x: x.bbox[1]))):
            if abs(_point(use_item)[1] - _point(edit_item)[1]) > 105:
                result["unknowns"].append("saved_row_control_alignment_unknown")
                continue
            result["presets"].append({"row": index, "complete": False, "cards": [],
                                      "use_point": _point(use_item), "edit_point": _point(edit_item),
                                      "evidence": [asdict(use_item), asdict(edit_item)]})
        new = _one(canonical, "新建", (1090, 625, 1250, 715))
        if new is not None:
            result["controls"].append(_control("create_preset", new))
        result["unknowns"].append("saved_preset_manifest_unavailable")
        return result
    result["surface"] = "current"
    result["ready"] = True
    result["capacities"] = _capacities(canonical, edit=False)
    result["controls"].append(_control("open_saved", saved_title))
    return result


def _visual_fingerprint(crop: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, object]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    bits = coefficients > np.median(coefficients.flat[1:])
    value = sum(int(bit) << index for index, bit in enumerate(bits.flat))
    return {"phash": f"{value:016x}", "mean_bgr": [round(float(x), 2) for x in crop.mean((0, 1))],
            "std_bgr": [round(float(x), 2) for x in crop.std((0, 1))], "bbox": list(box)}


def visual_fingerprint_matches(first: dict, second: dict) -> bool:
    """Conservative match of the same icon across two independently captured frames."""
    try:
        a, b = int(first["phash"], 16), int(second["phash"], 16)
        means_a, means_b = first["mean_bgr"], second["mean_bgr"]
        deviations_a, deviations_b = first["std_bgr"], second["std_bgr"]
        return ((a ^ b).bit_count() <= 8 and len(means_a) == len(means_b) == 3
                and len(deviations_a) == len(deviations_b) == 3
                and max(abs(float(x) - float(y)) for x, y in zip(means_a, means_b)) <= 12
                and max(abs(float(x) - float(y)) for x, y in zip(deviations_a, deviations_b)) <= 12)
    except (KeyError, TypeError, ValueError):
        return False


def recognize_hero_loadout(
    path: str | Path, texts: Sequence[OCRText], hero_cards: Sequence[dict], *,
    baseline_resolution: tuple[int, int] = _BASE,
) -> dict[str, object]:
    """Fingerprint pet and equipment icons under independently named hero cards.

    ``hero_cards`` must come from a separate portrait/face recognizer. The
    visual column is used only after that recognizer supplies a unique hero ID,
    card border and high confidence; column order never identifies a hero.
    """
    result: dict[str, object] = {"complete": False, "hero_loadout": {}, "unknowns": [],
                                 "frame": str(path)}
    path = Path(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
    if image is None:
        result["unknowns"].append("image_unreadable")
        return result
    height, width = image.shape[:2]
    if abs(width / height - 16 / 9) > .01:
        result["unknowns"].append("unsupported_aspect_ratio")
        return result
    image = cv2.resize(image, _BASE, interpolation=cv2.INTER_AREA)
    canonical = [OCRText(item.text, item.confidence, None if item.bbox is None else scale_box(
        item.bbox, from_resolution=baseline_resolution, to_resolution=_BASE)) for item in texts]
    saved_actions = [item for item in canonical if item.text.strip() in {"使用", "编辑"}
                     and _inside(item, (1100, 125, 1240, 670))]
    current = (_one(canonical, "我的军队", (185, 40, 340, 105)) is not None
               and not saved_actions and not _picker_visible(image))
    editing = sum(bool(_TITLE.search(item.text.replace(" ", ""))) and
                  _inside(item, (460, 25, 780, 110)) and item.confidence >= .88
                  for item in canonical) == 1
    if not (current or editing):
        result["unknowns"].append("hero_loadout_page_unanchored")
        return result
    if not hero_cards:
        result["unknowns"].append("hero_identity_cards_missing")
        return result
    hero_counts = [(match, item) for item in canonical if item.confidence >= .88
                   and _inside(item, (15, 130, 200, 190))
                   if (match := _CAPACITY.fullmatch(item.text.strip())) is not None]
    if (len(hero_counts) != 1 or int(hero_counts[0][0].group(1)) != len(hero_cards)
            or int(hero_counts[0][0].group(2)) < len(hero_cards)):
        result["unknowns"].append("hero_capacity_or_card_count_unverified")
        return result
    ids: set[str] = set()
    boxes: list[tuple[int, int, int, int]] = []
    for card in hero_cards:
        unit_id = card.get("unit_id")
        raw = card.get("card_bbox", card.get("bbox"))
        unit = get_unit(unit_id) if isinstance(unit_id, str) else None
        if (unit is None or unit.kind != "hero" or card.get("kind") != "hero"
                or card.get("source") != "army" or card.get("count") != 1
                or not isinstance(card.get("confidence"), (int, float))
                or card["confidence"] < .94 or unit_id in ids or
                not isinstance(raw, (list, tuple)) or len(raw) != 4 or
                any(type(number) is not int for number in raw)):
            result["unknowns"].append("hero_card_identity_or_geometry_unverified")
            return result
        l, t, r, b = scale_box(tuple(raw), from_resolution=baseline_resolution,
                               to_resolution=_BASE)
        if not (15 <= l < 540 and 177 <= t <= 200 and 112 <= r-l <= 138
                and 660 <= b <= 685 and 465 <= b-t <= 510):
            result["unknowns"].append("hero_card_layout_changed")
            return result
        if any(max(0, min(r, old[2]) - max(l, old[0])) > 5 for old in boxes):
            result["unknowns"].append("hero_cards_overlap")
            return result
        ids.add(unit_id)
        boxes.append((l, t, r, b))
        icon_boxes = {
            "pet_visual": (l+8, b-135, r-8, b-69),
            "equipment_1_visual": (l+8, b-66, l+59, b-10),
            "equipment_2_visual": (r-59, b-66, r-8, b-10),
        }
        state = {}
        for name, box in icon_boxes.items():
            il, it, ir, ib = box
            if not (0 <= il < ir <= image.shape[1] and 0 <= it < ib <= image.shape[0]):
                result["unknowns"].append("hero_loadout_icon_outside_frame")
                return result
            crop = image[it:ib, il:ir]
            if float(crop.std()) < 15:
                result["unknowns"].append("hero_loadout_icon_obscured")
                return result
            state[name] = _visual_fingerprint(crop, box)
        result["hero_loadout"][unit_id] = state
    result["complete"] = True
    return result
