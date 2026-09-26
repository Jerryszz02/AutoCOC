"""Read-only recognition of the home-village army configuration surfaces.

This module locates navigation controls, but deliberately does not infer a unit
from its position in a saved plan or picker. A control for changing a unit is
only emitted after a named portrait sample and an independent count are read.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import re
from collections.abc import Sequence

import cv2
import numpy as np

from .locator import scale_box
from .ocr import OCRProvider, OCRText
from .unit_catalog import CATALOG_ROOT, get_unit, recognize_card_identity, template_manifest

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


def _scale_point(x: int, y: int, resolution: tuple[int, int]) -> list[int]:
    return [round(x * resolution[0] / _BASE[0]),
            round(y * resolution[1] / _BASE[1])]


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
    bounds = {"troop": 500, "spell": 20, "siege": 10}
    for kind, roi in regions.items():
        matches = [(_CAPACITY.fullmatch(item.text.strip()), item) for item in texts
                   if item.confidence >= .88 and _inside(item, roi)]
        matches = [(m, item) for m, item in matches if m is not None]
        if len(matches) == 1:
            m, _ = matches[0]
            used, total = int(m.group(1)), int(m.group(2))
            if 0 <= used <= total <= bounds[kind]:
                result[kind] = {"used": used, "total": total}
    return result


def _local_capacity(path: Path, provider: OCRProvider | None,
                    native_resolution: tuple[int, int],
                    roi: tuple[int, int, int, int], maximum: int) -> dict[str, int] | None:
    reader = getattr(provider, "recognize_line", None)
    if not callable(reader):
        return None
    native_roi = scale_box(roi, from_resolution=_BASE, to_resolution=native_resolution)
    candidates = [(_CAPACITY.fullmatch(item.text.strip()), item) for item in
                  reader(path, native_roi) if item.confidence >= .65]
    matches = [match for match, _ in candidates if match is not None]
    if len(matches) != 1:
        return None
    used, total = int(matches[0].group(1)), int(matches[0].group(2))
    return {"used": used, "total": total} if 0 <= used <= total <= maximum else None


def _picker_visible(image: np.ndarray) -> bool:
    # The expanded picker fills both lower rows with bright framed cards while
    # darkening the editor above. Works for coloured troop and gray spell cards.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upper = gray[385:415, 50:1200]
    lower = gray[460:665, 50:1200]
    return float(np.mean(upper)) < 75 and float(np.mean(lower > 160)) > .20


def _red_minus(image: np.ndarray, x: int, y: int) -> bool:
    """The observed current picker puts a red remove glyph on each selected card."""
    crop = image[y-14:y+14, x-14:x+14]
    if crop.shape[:2] != (28, 28):
        return False
    b, g, r = cv2.split(crop.astype(np.float32))
    red = (r > 110) & (r > g * 1.45) & (r > b * 1.35)
    return float(red.mean()) >= .28


def _picker_row(image: np.ndarray) -> tuple[str | None, list[tuple[int, int, int, int]]]:
    """Locate top-row selected cards using their own red minus glyphs."""
    rows = {"troop": 194, "spell": 287}
    found = {}
    for kind, top in rows.items():
        boxes = []
        for index in range(6 if kind == "troop" else 4):
            left = 558 + index * 104
            if _red_minus(image, left + 78, top + 16):
                boxes.append((left, top, left + 98, top + 98))
        found[kind] = boxes
    active = [kind for kind, boxes in found.items() if boxes]
    return (active[0], found[active[0]]) if len(active) == 1 else (None, [])


def _picker_count(provider: OCRProvider | None, path: Path, texts: Sequence[OCRText],
                  box: tuple[int, int, int, int],
                  native_resolution: tuple[int, int], image: np.ndarray,
                  client_version: str | None,
                  unit_id: str | None) -> tuple[int | None, dict[str, object]]:
    roi = (box[0], box[1], box[0] + 45, box[1] + 30)
    candidates = [item for item in texts if item.confidence >= .9 and _inside(item, roi)
                  and re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())]
    source = "full_ocr"
    if not candidates and provider is not None:
        reader = getattr(provider, "recognize_line", None)
        if callable(reader):
            native_roi = scale_box(roi, from_resolution=_BASE, to_resolution=native_resolution)
            candidates = [item for item in reader(path, native_roi)
                          if item.confidence >= .9 and
                          re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())]
            source = "line_roi"
    values = {int(re.sub(r"\D", "", item.text)) for item in candidates}
    count = next(iter(values)) if len(values) == 1 and next(iter(values)) > 0 else None
    evidence = {"source": source, "reads": [asdict(item) for item in candidates]}
    votes: dict[int, set[str]] = {}
    if not values:
        image_reader = getattr(provider, "recognize_line_image", None)
        if callable(image_reader):
            left, top = box[:2]
            gray = cv2.cvtColor(image[top:top + 30, left:left + 52],
                                cv2.COLOR_BGR2GRAY)
            variants = {"gray": gray, "inverted": 255 - gray,
                        "threshold": cv2.threshold(gray, 170, 255,
                                                   cv2.THRESH_BINARY)[1]}
            observed: list[dict[str, object]] = []
            for name, crop in variants.items():
                reads = image_reader(crop)
                if not isinstance(reads, (list, tuple)):
                    continue
                for item in reads:
                    if isinstance(item, OCRText):
                        observed.append({"variant": name, "text": item.text,
                                         "confidence": item.confidence})
            evidence["contrast_reads"] = observed
            for item in observed:
                if (item["confidence"] >= .78 and
                        re.fullmatch(r"[xX×]\s*[0-9]+", item["text"].strip())):
                    value = int(re.sub(r"\D", "", item["text"]))
                    if value > 0:
                        votes.setdefault(value, set()).add(item["variant"])
            if len(votes) == 1 and len(next(iter(votes.values()))) >= 2:
                count = next(iter(votes))
                evidence["source"] = "contrast_consensus"
    if not values and not votes and count is None and client_version == template_manifest().get("client"):
        provenance_path = CATALOG_ROOT / "picker_count_provenance.json"
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            provenance = {}
        sample_path = CATALOG_ROOT / "current_picker_x1_18_600_7.png"
        full_path = CATALOG_ROOT / "current_picker_x1_full_18_600_7.png"
        if (provenance.get("client") == client_version and sample_path.is_file()
                and full_path.is_file()):
            sample = cv2.imdecode(np.fromfile(sample_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            full = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            header = cv2.cvtColor(image[box[1]:box[1] + 31, box[0]:box[0] + 55],
                                  cv2.COLOR_BGR2GRAY)
            whole = cv2.cvtColor(image[box[1]:box[1] + 30, box[0] + 1:box[0] + 46],
                                  cv2.COLOR_BGR2GRAY)
            if (sample is not None and full is not None and
                    header.shape[0] >= sample.shape[0] and header.shape[1] >= sample.shape[1]
                    and whole.shape == full.shape):
                score = float(cv2.matchTemplate(header, sample, cv2.TM_CCOEFF_NORMED).max())
                full_score = float(cv2.matchTemplate(whole, full, cv2.TM_CCOEFF_NORMED).max())
                evidence["x1_sample_score"] = round(score, 5)
                evidence["x1_full_score"] = round(full_score, 5)
                if score >= .9 and full_score >= .96:
                    evidence["source"] = "versioned_picker_x1_sample"
                    count = 1
        if (count is None and unit_id == "lightning_spell" and box[:2] == (558, 287)):
            six_path = CATALOG_ROOT / "current_picker_x6_18_600_7.png"
            six = cv2.imdecode(np.fromfile(six_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if six_path.is_file() else None
            header = cv2.cvtColor(image[287:317, 559:604], cv2.COLOR_BGR2GRAY)
            if six is not None and header.shape == six.shape:
                score = float(cv2.matchTemplate(header, six, cv2.TM_CCOEFF_NORMED).max())
                evidence["x6_full_score"] = round(score, 5)
                if score >= .97:
                    evidence["source"] = "versioned_picker_x6_sample"
                    count = 6
    return count, evidence


def _bottom_picker_controls(image: np.ndarray, kind: str,
                            client_version: str | None,
                            baseline_resolution: tuple[int, int],
                            selected: set[str], capacity: dict[str, int] | None
                            ) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Only independently sampled portraits can produce a bottom-card control."""
    controls: list[dict[str, object]] = []
    availability: dict[str, dict[str, object]] = {}
    if client_version != template_manifest().get("client"):
        return controls, availability
    matches: dict[str, list[dict[str, object]]] = {}
    for top in (431, 568):
        for index in range(9):
            left = round(33 + index * 130.25)
            box = (left, top, left + 124, top + 124)
            identity = recognize_card_identity(image, box, kind,
                                               surface="current_picker",
                                               client_version=client_version)
            unit_id = identity["unit_id"]
            if unit_id is None or not identity["evidence"]:
                continue
            matched = identity["evidence"][0]["template"]
            state = ("capacity_gray" if "_capacity_gray_" in matched else
                     "active" if "_active_" in matched else None)
            if state is None:
                continue
            matches.setdefault(unit_id, []).append({"box": box, "identity": identity,
                                                     "state": state})
    for unit_id, hits in matches.items():
        if len(hits) != 1:
            continue
        hit = hits[0]
        box, identity, state = hit["box"], hit["identity"], hit["state"]
        full = (isinstance(capacity, dict) and
                capacity.get("used") == capacity.get("total"))
        if state == "capacity_gray" and not full:
            continue
        enabled = state == "active"
        if enabled or unit_id in selected:
            availability[unit_id] = {"available": True,
                                      "reason": ("active_picker_card" if enabled else
                                                 "present_in_current_army")}
        controls.append({"action": "increment", "unit_id": unit_id,
                         "point": _scale_point(box[0] + 61, box[1] + 57,
                                               baseline_resolution),
                         "confidence": identity["confidence"],
                         "enabled": enabled, "capacity_blocked": not enabled,
                         "identity_verified": True, "cost_free": True,
                         "evidence": {"identity": identity,
                                      "card_bbox": list(scale_box(
                                          box, from_resolution=_BASE,
                                          to_resolution=baseline_resolution)),
                                      "state": state}})
    return controls, availability


def _current_picker(path: Path, image: np.ndarray, provider: OCRProvider | None,
                    texts: Sequence[OCRText], native_resolution: tuple[int, int],
                    client_version: str | None,
                    baseline_resolution: tuple[int, int]) -> dict[str, object]:
    kind, boxes = _picker_row(image)
    capacities = _capacities(texts, edit=False)
    if kind == "spell":
        shifted_troop = [item for item in texts if item.confidence >= .88 and item.bbox is not None
                         and _inside(item, (550, 55, 710, 110))
                         and _CAPACITY.fullmatch(item.text.strip())]
        if len(shifted_troop) == 1:
            match = _CAPACITY.fullmatch(shifted_troop[0].text.strip())
            used, total = int(match.group(1)), int(match.group(2))
            if 0 <= used <= total <= 500:
                capacities["troop"] = {"used": used, "total": total}
        shifted = [item for item in texts if item.confidence >= .88 and item.bbox is not None
                   and _inside(item, (560, 235, 675, 275))
                   and _CAPACITY.fullmatch(item.text.strip())]
        if len(shifted) == 1:
            match = _CAPACITY.fullmatch(shifted[0].text.strip())
            used, total = int(match.group(1)), int(match.group(2))
            if 0 <= used <= total <= 20:
                capacities["spell"] = {"used": used, "total": total}
    rows = ({"troop": ((560, 145, 680, 185), 500),
             "spell": ((560, 325, 670, 365), 20)} if kind == "troop" else
            {"troop": ((560, 55, 680, 100), 500),
             "spell": ((560, 235, 670, 275), 20)})
    for name, (roi, maximum) in rows.items():
        if name not in capacities:
            observed = _local_capacity(path, provider, native_resolution, roi, maximum)
            if observed is not None:
                capacities[name] = observed
    result: dict[str, object] = {"frame": str(path), "surface": "current_picker",
                                 "ready": kind is not None, "editing_kind": kind,
                                 "cards": [], "controls": [], "presets": [],
                                 "capacities": capacities,
                                 "complete_kinds": {"troop": False, "spell": False},
                                 "unit_availability": {}, "unit_housing_space": {},
                                 "unknowns": [], "complete": False}
    if kind is None:
        result["unknowns"].append("picker_editing_kind_unverified")
        return result
    close_point = (800, 320) if kind == "troop" else (1050, 240)
    result["controls"].append({"action": "close_picker",
                               "point": _scale_point(*close_point, baseline_resolution),
                               "confidence": .95, "enabled": True, "cost_free": True,
                               "evidence": {"visual": "observed_dimmed_background_dismissal",
                                            "editing_kind": kind}})
    if boxes != [(558 + i * 104, boxes[0][1], 656 + i * 104, boxes[0][1] + 98)
                 for i in range(len(boxes))]:
        result["unknowns"].append("picker_selected_cards_not_contiguous")
    seen = set()
    for box in boxes:
        identity = recognize_card_identity(image, box, kind, surface="army",
                                           client_version=client_version)
        unit_id = identity["unit_id"]
        count, count_evidence = _picker_count(provider, path, texts, box, native_resolution,
                                              image, client_version, unit_id)
        card = {"kind": kind, "source": "army", "unit_id": unit_id,
                "count": count, "confidence": identity["confidence"],
                "bbox": list(scale_box(box, from_resolution=_BASE,
                                       to_resolution=baseline_resolution)),
                "evidence": {"identity": identity,
                                                 "count": count_evidence,
                                                 "red_minus": True}}
        result["cards"].append(card)
        if count is None or unit_id is None or unit_id in seen:
            result["unknowns"].append("picker_selected_identity_or_count_unverified")
            continue
        seen.add(unit_id)
        unit = get_unit(unit_id)
        if type(unit.housing_space) is int and unit.housing_space > 0:
            result["unit_housing_space"][unit_id] = unit.housing_space
        result["unit_availability"][unit_id] = {"available": True,
                                                  "reason": "present_in_current_army"}
        result["controls"].append({"action": "decrement", "unit_id": unit_id,
                                   "point": _scale_point(box[0] + 78, box[1] + 16,
                                                         baseline_resolution),
                                   "confidence": identity["confidence"],
                                   "enabled": True, "cost_free": True,
                                   "evidence": card["evidence"]})
    capacity = result["capacities"].get(kind)
    if not result["unknowns"] and boxes:
        if (isinstance(capacity, dict) and
                len(result["unit_housing_space"]) == len(boxes) and
                sum(card["count"] * result["unit_housing_space"][card["unit_id"]]
                    for card in result["cards"]) == capacity["used"]):
            result["complete_kinds"][kind] = True
        else:
            result["unknowns"].append("picker_capacity_or_housing_mismatch")
    bottom_controls, availability = _bottom_picker_controls(
        image, kind, client_version, baseline_resolution, seen,
        result["capacities"].get(kind))
    result["controls"].extend(bottom_controls)
    result["unit_availability"].update(availability)
    return result


def _save_current_icon_visible(image: np.ndarray) -> bool:
    """Require the white tray glyph inside the current-army tab before tapping it."""
    hsv = cv2.cvtColor(image[54:100, 388:443], cv2.COLOR_BGR2HSV)
    white = float(np.mean((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 170)))
    outline = float(np.mean(hsv[:, :, 2] < 65))
    return white >= .07 and outline >= .1


def _save_current_dialog(image: np.ndarray, texts: Sequence[OCRText]) -> dict[str, object] | None:
    title = _one(texts, "将军队保存为新配置", (480, 35, 800, 105))
    if title is None:
        return None
    result: dict[str, object] = {"frame": "", "surface": "save_current", "ready": False,
                                 "cards": [], "controls": [], "presets": [],
                                 "capacities": {}, "unknowns": [], "complete": False}
    buttons = sorted((item for item in texts if item.text.strip() == "保存军队"
                      and item.confidence >= .88 and _inside(item, (1100, 120, 1240, 700))),
                     key=lambda item: _point(item)[1])
    if len(buttons) < 2:
        result["unknowns"].append("save_current_rows_unanchored")
        return result
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    empty_active = []
    for item in buttons:
        center_y = _point(item)[1]
        top, bottom = max(160, center_y - 80), min(710, center_y + 30)
        if bottom - top < 60:
            continue
        cards = hsv[top:bottom, 70:1050]
        colorful = float(np.mean((cards[:, :, 0] >= 65) & (cards[:, :, 0] <= 170)
                                  & (cards[:, :, 1] >= 75) & (cards[:, :, 2] >= 110)))
        content_gray = cv2.cvtColor(image[max(160, center_y-70):min(710, center_y+55),
                                              100:1050], cv2.COLOR_BGR2GRAY)
        content_std = float(content_gray.std())
        card_edges = float(np.mean(cv2.Canny(content_gray, 50, 130) > 0))
        count_label = any(item.bbox is not None and re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())
                          and 100 <= _point(item)[0] <= 1050
                          and center_y - 70 <= _point(item)[1] <= center_y + 55
                          for item in texts)
        button = hsv[max(150, center_y - 65):min(710, center_y + 45), 1120:1230]
        brightness = float(np.mean(button[:, :, 2]))
        if (not count_label and colorful < .025 and content_std <= 15
                and card_edges <= .04 and brightness >= 130):
            empty_active.append((item, colorful, brightness, content_std, card_edges))
    if len(empty_active) == 1:
        item, colorful, brightness, content_std, card_edges = empty_active[0]
        result["controls"].append(_control("save_to_empty", item,
                                           empty_color_fraction=colorful,
                                           empty_content_std=content_std,
                                           empty_card_edge_fraction=card_edges,
                                           button_brightness=brightness))
        result["ready"] = True
    elif empty_active:
        result["unknowns"].append("save_current_empty_slot_ambiguous")
    else:
        result["unknowns"].append("save_current_empty_slot_not_visible")
    return result


def recognize_army_editor(
    path: str | Path, provider: OCRProvider, texts: Sequence[OCRText], *,
    baseline_resolution: tuple[int, int] = _BASE,
    client_version: str | None = None,
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
    save_dialog = _save_current_dialog(image, canonical)
    if save_dialog is not None:
        save_dialog["frame"] = str(path)
        return save_dialog
    saved_title = _one(canonical, "已保存的配置", (545, 40, 735, 105))
    army_title = _one(canonical, "我的军队", (185, 40, 340, 105))
    edit_titles = [item for item in canonical if item.bbox is not None and
                   _TITLE.search(item.text.replace(" ", "")) and item.confidence >= .88 and
                   _inside(item, (460, 25, 780, 110))]
    if _picker_visible(image):
        picker_kind, _ = _picker_row(image)
        # Opening the spell tray scrolls the entire page upward by 86 px,
        # clipping both tabs. The fully occupied hero header at its observed
        # shifted location distinguishes this sampled current layout from the
        # saved-plan editor's partially filled hero row.
        shifted_current = (picker_kind == "spell" and army_title is None and
                           _one(canonical, "4/4", (40, 55, 105, 100)) is not None and
                           _one(canonical, "强化军队", (850, 15, 1050, 100)) is not None and
                           _one(canonical, "强化英雄", (1020, 15, 1220, 100)) is not None and
                           not edit_titles)
        if (saved_title is not None and army_title is not None) or shifted_current:
            return _current_picker(path, image, provider, canonical, (width, height),
                                   client_version, baseline_resolution)
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
        result["controls"].append(_control("open_current", army_title))
        if len(use) != len(edits):
            result["unknowns"].append("saved_row_controls_incomplete")
        for index, (use_item, edit_item) in enumerate(zip(sorted(use, key=lambda x: x.bbox[1]),
                                                          sorted(edits, key=lambda x: x.bbox[1]))):
            if abs(_point(use_item)[1] - _point(edit_item)[1]) > 105:
                result["unknowns"].append("saved_row_control_alignment_unknown")
                continue
            labels = [item for item in canonical if item.bbox is not None and
                      re.fullmatch(r"军队配置\s*([0-9]+)", item.text.strip()) and
                      _inside(item, (25, 110, 250, 670)) and
                      25 <= _point(use_item)[1] - _point(item)[1] <= 75]
            result["presets"].append({"row": index, "complete": False, "cards": [],
                                      "preset_id": (re.search(r"[0-9]+", labels[0].text).group()
                                                    if len(labels) == 1 else None),
                                      "use_point": _point(use_item), "edit_point": _point(edit_item),
                                      "evidence": [asdict(use_item), asdict(edit_item)]})
        new = _one(canonical, "新建", (1090, 625, 1250, 715))
        if new is not None:
            result["controls"].append(_control("create_preset", new))
        ids = [item.get("preset_id") for item in result["presets"]]
        first = next((item for item in canonical if item.text.strip() == "军队配置1"
                      and item.bbox is not None and _inside(item, (25, 110, 250, 180))), None)
        result["occupied_preset_ids_complete"] = bool(
            first is not None and new is not None and ids and all(ids)
            and len(set(ids)) == len(ids) and len(use) == len(edits) == len(ids)
            and max(_point(item)[1] for item in use) < _point(new)[1])
        result["unknowns"].append("saved_preset_manifest_unavailable")
        return result
    result["surface"] = "current"
    result["ready"] = True
    result["capacities"] = _capacities(canonical, edit=False)
    for name, roi, maximum in (("troop", (560, 145, 680, 185), 500),
                               ("spell", (560, 325, 670, 365), 20)):
        if name not in result["capacities"]:
            observed = _local_capacity(path, provider, (width, height), roi, maximum)
            if observed is not None:
                result["capacities"][name] = observed
    result["controls"].append(_control("open_saved", saved_title))
    for kind, row_top, capacity_key in (("troop", 194, "troop"),
                                        ("spell", 373, "spell")):
        capacity = result["capacities"].get(capacity_key)
        if (isinstance(capacity, dict) and capacity.get("used", 0) > 0
                and float(cv2.cvtColor(image[row_top:row_top + 98, 558:656],
                                       cv2.COLOR_BGR2GRAY).std()) > 28):
            point = _scale_point(607, row_top + 49, baseline_resolution)
            result["controls"].append({"action": "open_picker", "kind": kind,
                                       "point": point, "confidence": .92,
                                       "enabled": True, "cost_free": True,
                                       "evidence": {"capacity": capacity,
                                                    "first_card_visual_std": round(float(
                                                        cv2.cvtColor(image[row_top:row_top + 98, 558:656],
                                                                       cv2.COLOR_BGR2GRAY).std()), 3)}})
    if _save_current_icon_visible(image):
        result["controls"].append({"action": "save_current", "point": [414, 74],
                                   "confidence": .93, "enabled": True, "cost_free": True,
                                   "evidence": {"icon": "white_down_arrow_and_tray",
                                                "tabs": [asdict(army_title), asdict(saved_title)]}})
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
