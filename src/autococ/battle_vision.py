"""Battle-only fast observations with current-frame evidence and full-OCR fallback."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np

from .images import read_frame
from .locator import scale_box
from .scene import SceneSnapshot


def _preparation_countdown_items(snapshot: SceneSnapshot,
                                 baseline_resolution: tuple[int, int], minimum_confidence: float) -> list[dict]:
    width, height = baseline_resolution
    items = []
    for raw in snapshot.observations.get("ocr", []):
        if not isinstance(raw, dict):
            continue
        box, confidence = raw.get("bbox"), raw.get("confidence")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not minimum_confidence <= confidence <= 1
                or not isinstance(box, (tuple, list)) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or not (width * .25 <= box[0] < box[2] <= width * .75
                        and 0 <= box[1] < box[3] <= height * .20)):
            continue
        items.append({"text": re.sub(r"\s+", "", str(raw.get("text", "")).lower()),
                      "bbox": box, "confidence": confidence})
    return items


def _preparation_seconds(text: str) -> int | None:
    text = text.strip(":：")
    match = re.fullmatch(r"(\d{1,2})(?:秒|s|sec|seconds)?", text)
    if match:
        value = int(match.group(1))
        return value if 1 <= value <= 60 else None
    match = re.fullmatch(r"0:(\d{2})", text)
    if match and 1 <= int(match.group(1)) < 60:
        return int(match.group(1))
    return None


def _preparation_timer_aligned(box: tuple | list, label_box: tuple | list,
                               baseline_resolution: tuple[int, int]) -> bool:
    width, height = baseline_resolution
    # Preserve the same bounded OCR-box overlap and center alignment used by
    # ordinary countdown recognition; neither path may invent a new timer.
    allowed_overlap = min(14 * height / 720, (label_box[3] - label_box[1]) * .40,
                          (box[3] - box[1]) * .25)
    # The current HUD timer is immediately below or slightly overlaps its
    # heading. A base-building level 64px lower was otherwise treated as a
    # second strong timer; keep nearby conflicting numbers inside this band.
    return (-allowed_overlap <= box[1] - label_box[3] <= min(height * .10, 40 * height / 720) and
            abs((box[0] + box[2] - label_box[0] - label_box[2]) / 2) <= width * .07)


def _preparation_countdown_candidates(items: list[dict], baseline_resolution: tuple[int, int]):
    labels = ("战斗开始倒计时", "开战倒计时", "battlestartsin")
    anchors = [(item, prefix) for item in items for prefix in labels if item["text"].startswith(prefix)]
    if len(anchors) != 1:
        return anchors, []
    anchor, prefix = anchors[0]
    candidates = []
    embedded = _preparation_seconds(anchor["text"][len(prefix):])
    if embedded is not None:
        candidates.append((embedded, anchor))
    for item in items:
        if item is anchor:
            continue
        value = _preparation_seconds(item["text"])
        box, label_box = item["bbox"], anchor["bbox"]
        if value is not None and _preparation_timer_aligned(box, label_box, baseline_resolution):
            candidates.append((value, item))
    return anchors, candidates


def preparation_timer_line_recovery_candidate(snapshot: SceneSnapshot,
                                              baseline_resolution: tuple[int, int] = (1280, 720)) -> dict | None:
    """Locate one damaged full-OCR timer for a fresh line read, never infer its value."""
    battle = snapshot.observations.get("battle")
    if (snapshot.scene != "enemy_village" or not isinstance(battle, dict)
            or battle.get("phase") != "scout"
            or battle.get("countdown_seconds") is not None
            or battle.get("countdown_evidence") is not None):
        return None
    items = _preparation_countdown_items(snapshot, baseline_resolution, 0)
    anchors, candidates = _preparation_countdown_candidates(items, baseline_resolution)
    if len(anchors) != 1 or candidates:
        return None
    label, _ = anchors[0]
    label_box = scale_box(tuple(label["bbox"]), from_resolution=baseline_resolution,
                          to_resolution=(1280, 720))
    if (label["text"] != "开战倒计时" or not .9 <= label["confidence"] <= 1
            or not (575 <= label_box[0] <= 590 and 8 <= label_box[1] <= 16
                    and 670 <= label_box[2] <= 690 and 35 <= label_box[3] <= 45)):
        return None
    # The leading dot is evidence of a damaged full OCR line, not punctuation
    # to strip. The complete number must be independently read below.
    malformed = []
    for item in items:
        if item is label:
            continue
        if item["confidence"] >= .9 and _preparation_seconds(item["text"]) is not None:
            return None  # A separate strong valid timer must never be overridden.
        match = re.fullmatch(r"\.(\d{1,2}秒)", item["text"])
        if match is not None:
            box = scale_box(tuple(item["bbox"]), from_resolution=baseline_resolution,
                            to_resolution=(1280, 720))
            if not _preparation_timer_aligned(box, label_box, (1280, 720)):
                continue
            if (not 590 <= box[0] <= 600 or not 29 <= box[1] <= 39
                    or not 680 <= box[2] <= 700 or not 78 <= box[3] <= 95):
                return None
            malformed.append((item, match.group(1)))
    if len(malformed) != 1 or malformed[0][0]["confidence"] < .9:
        return None
    raw, suffix = malformed[0]
    if _preparation_seconds(suffix) is None:
        return None
    roi = scale_box((599, 30, 690, 89), from_resolution=(1280, 720),
                    to_resolution=baseline_resolution)
    return {"reason": "one_strong_heading_one_damaged_timer", "frame": str(snapshot.screenshot_path),
            "raw_label": label, "raw_timer": raw, "expected_full_text": suffix,
            "baseline_roi": roi}


def preparation_countdown_retry_candidate(snapshot: SceneSnapshot,
                                          baseline_resolution: tuple[int, int] = (1280, 720)) -> dict | None:
    """Describe one weak timer that may justify observing again, never a usable countdown."""
    battle = snapshot.observations.get("battle")
    if (snapshot.scene != "enemy_village" or not isinstance(battle, dict)
            or "countdown_seconds" in battle or "countdown_evidence" in battle):
        return None
    items = _preparation_countdown_items(snapshot, baseline_resolution, 0)
    anchors, candidates = _preparation_countdown_candidates(items, baseline_resolution)
    if len(anchors) != 1 or len(candidates) != 1:
        return None
    anchor = anchors[0][0]
    _, timer = candidates[0]
    if (anchor["confidence"] < .9 or timer is anchor
            or not 0 <= timer["confidence"] < .9):
        return None
    # A separate high-confidence timer anywhere in the valid heading region
    # is a contradiction, even if it is not adjacent enough to be accepted.
    if any(item is not anchor and item is not timer and item["confidence"] >= .9
           and _preparation_seconds(item["text"]) is not None for item in items):
        return None
    return {"reason": "unique_weak_timer", "label": anchor, "timer": timer,
            "frame": str(snapshot.screenshot_path)}


def preparation_label_line_recovery_candidate(snapshot: SceneSnapshot,
                                              baseline_resolution: tuple[int, int] = (1280, 720)) -> dict | None:
    """Locate one weak merged heading beside a strong timer; grant no countdown."""
    battle = snapshot.observations.get("battle")
    if (snapshot.scene != "enemy_village" or not isinstance(battle, dict)
            or battle.get("phase") != "scout"):
        return None
    items = _preparation_countdown_items(snapshot, baseline_resolution, 0)
    anchors, candidates = _preparation_countdown_candidates(items, baseline_resolution)
    if len(anchors) != 1 or len(candidates) != 1:
        return None
    label, _ = anchors[0]
    _, timer = candidates[0]
    if (not .8 <= label["confidence"] < .9 or timer is label
            or not .9 <= timer["confidence"] <= 1
            or not label["text"].startswith("开战倒计时")
            or any(item is not label and item is not timer and item["confidence"] >= .9
                   and _preparation_seconds(item["text"]) is not None for item in items)):
        return None
    # Only the measured 18.647.2 heading location has a fixed crop which keeps
    # every title glyph while excluding the upgrade arrow and its level digits.
    box = scale_box(tuple(label["bbox"]), from_resolution=baseline_resolution,
                    to_resolution=(1280, 720))
    if not (575 <= box[0] <= 585 and 8 <= box[1] <= 15
            and 700 <= box[2] <= 720 and 35 <= box[3] <= 45):
        return None
    roi = scale_box((578, 8, 674, 39), from_resolution=(1280, 720),
                    to_resolution=baseline_resolution)
    return {"reason": "one_weak_merged_heading_and_one_strong_timer",
            "frame": str(snapshot.screenshot_path), "raw_label": label,
            "timer": timer, "baseline_roi": roi}


def attach_preparation_countdown(snapshot: SceneSnapshot,
                                 baseline_resolution: tuple[int, int] = (1280, 720)) -> None:
    """Use a unique, high-confidence preparation label and adjacent timer only.

    This reads existing OCR on this frame, never a cached timer or a new image.
    The capture-start timestamp is attached by GameSession, not by inference.
    """
    if snapshot.scene != "enemy_village":
        return
    battle = snapshot.observations.get("battle")
    if not isinstance(battle, dict):
        return
    battle.pop("countdown_seconds", None)
    battle.pop("countdown_evidence", None)
    items = _preparation_countdown_items(snapshot, baseline_resolution, .9)
    anchors, candidates = _preparation_countdown_candidates(items, baseline_resolution)
    if len(anchors) != 1:
        return
    if len(candidates) != 1:
        return
    anchor = anchors[0][0]
    value, timer = candidates[0]
    battle["countdown_seconds"] = value
    battle["countdown_evidence"] = {"frame": str(snapshot.screenshot_path), "label": anchor,
                                      "timer": timer, "confidence": min(anchor["confidence"], timer["confidence"]),
                                      "source": "current_frame_ocr"}


def _static_preparation_label_box(snapshot: SceneSnapshot,
                                  baseline_resolution: tuple[int, int],
                                  label: dict) -> tuple[int, int, int, int] | None:
    """Exclude overlapping timer digits from a source-bound preparation label."""
    battle = snapshot.observations.get("battle")
    if snapshot.scene != "enemy_village" or not isinstance(battle, dict):
        return None
    items = _preparation_countdown_items(snapshot, baseline_resolution, 0)
    anchors, candidates = _preparation_countdown_candidates(items, baseline_resolution)
    if len(anchors) != 1 or len(candidates) != 1:
        return None
    anchor, _ = anchors[0]
    seconds, timer = candidates[0]
    if (anchor["confidence"] < .9 or timer is anchor or timer["confidence"] < .9 or
            tuple(anchor["bbox"]) != tuple(label["bbox"]) or
            anchor["text"] != re.sub(r"\s+", "", str(label["text"]).lower()) or
            any(item is not anchor and item is not timer and
                _preparation_seconds(item["text"]) is not None for item in items)):
        return None
    evidence = battle.get("countdown_evidence")
    if "countdown_evidence" in battle or "countdown_seconds" in battle:
        if (not isinstance(evidence, dict) or
                evidence.get("frame") != str(snapshot.screenshot_path) or
                evidence.get("source") != "current_frame_ocr" or
                battle.get("countdown_seconds") != seconds or
                evidence.get("label") != anchor or evidence.get("timer") != timer):
            return None
    label_box, timer_box = anchor["bbox"], timer["bbox"]
    if not (label_box[1] < timer_box[1] < label_box[3] and
            timer_box[1] - label_box[1] >= .6 * (label_box[3] - label_box[1])):
        return None
    static_box = scale_box((label_box[0], label_box[1], label_box[2], timer_box[1]),
                           from_resolution=baseline_resolution, to_resolution=(1280, 720))
    if static_box[2] - static_box[0] < 40 or static_box[3] - static_box[1] < 12:
        return None
    return static_box


def selected_card_bbox(path, bbox, baseline_resolution=(1280, 720)):
    """Require the closed white selection outline on the current frame."""
    image = cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)
    left, top, right, bottom = scale_box(tuple(bbox), from_resolution=baseline_resolution, to_resolution=(1280, 720))
    x0, x1 = max(0, left - 12), min(1280, right + 12)
    hsv = cv2.cvtColor(image[565:720, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 230]), np.array([180, 35, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (80 <= width <= 108 and 105 <= height <= 135
                and abs(x + x0 + width / 2 - (left + right) / 2) <= 7
                and cv2.contourArea(contour) / (width * height) >= .85):
            return scale_box((x + x0, y + 565, x + x0 + width, y + 565 + height),
                             from_resolution=(1280, 720), to_resolution=baseline_resolution)
    # The live selected-card outline can be bright but discontinuous where the
    # card overlaps the bottom HUD. Its external contour then encloses almost
    # no area despite four visible sides. Require independent support on each
    # side at the selected card's expanded geometry; portrait highlights or an
    # unselected header cannot satisfy all four bands.
    if 12 <= left < right <= 1270 and 575 <= top < bottom <= 718:
        bright = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 95, 255])) > 0

        def vertical(start, stop):
            return max(((float(bright[top + 20 - 565:bottom - 10 - 565,
                                      x - x0 - 1:x - x0 + 2].mean()), x)
                        for x in range(start, stop + 1)
                        if x0 + 1 <= x <= x1 - 2), default=(0.0, 0))

        def horizontal(start, stop):
            return max(((float(bright[y - 566:y - 563,
                                      left + 10 - x0:right - 10 - x0].mean()), y)
                        for y in range(start, stop + 1) if 566 <= y <= 718),
                       default=(0.0, 0))

        left_support, selected_left = vertical(left - 8, left + 2)
        right_support, selected_right = vertical(right - 2, right + 6)
        top_support, selected_top = horizontal(top - 10, top - 2)
        bottom_support, selected_bottom = horizontal(bottom - 5, bottom + 1)
        width, height = selected_right - selected_left + 1, selected_bottom - selected_top + 1
        if (left_support >= .38 and right_support >= .50 and
                top_support >= .75 and bottom_support >= .65 and
                80 <= width <= 108 and 105 <= height <= 135 and
                abs((selected_left + selected_right) / 2 - (left + right) / 2) <= 7):
            return scale_box((selected_left, selected_top, selected_right + 1, selected_bottom + 1),
                             from_resolution=(1280, 720), to_resolution=baseline_resolution)
    return None


def _selected_count_header(path, image, bbox, *, client_version, baseline_resolution):
    """Locate a complete current-client quantity inside a closed selected card.

    The right-hand crop is safe only when all white quantity ink in the blue
    header lies inside it; x10/x100 prefixes cannot be cropped into x1.
    """
    left, top, right, bottom = bbox
    if (client_version != "18.647.2" or
            not (80 <= right - left <= 105 and 590 <= top <= 605 and
                 705 <= bottom <= 715)):
        return None
    # BattleObserver.cards stores 1280x720 boxes even when public slot
    # coordinates use a different configured baseline resolution.
    selected = selected_card_bbox(path, bbox, (1280, 720))
    if selected is None:
        return None
    x0, y0, x1, _y1 = selected
    roi = (x1 - 38, y0 - 2, x1 - 3, y0 + 31)
    scan = (x0 + 6, y0 + 3, x1 - 3, y0 + 31)
    if (not (left <= roi[0] < roi[2] <= right + 3 and
             top - 10 <= roi[1] < roi[3] <= top + 38) or
            not (0 <= scan[0] < scan[2] <= 1280 and 0 <= scan[1] < scan[3] <= 720)):
        return None
    hsv = cv2.cvtColor(image[scan[1]:scan[3], scan[0]:scan[2]], cv2.COLOR_BGR2HSV)
    ink = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 95, 255])) > 0
    ys, xs = np.where(ink)
    ink_box = ([scan[0] + int(xs.min()), scan[1] + int(ys.min()),
                scan[0] + int(xs.max()) + 1, scan[1] + int(ys.max()) + 1]
               if len(xs) else None)
    covered = (ink_box is not None and len(xs) >= 20 and
               ink_box[3] - ink_box[1] >= 8 and
               roi[0] <= ink_box[0] < ink_box[2] <= roi[2] and
               roi[1] <= ink_box[1] < ink_box[3] <= roi[3])
    return {"selected_card_bbox": list(scale_box(selected, from_resolution=(1280, 720),
                                                 to_resolution=baseline_resolution)),
            "selected_header_roi": list(scale_box(roi, from_resolution=(1280, 720),
                                                  to_resolution=baseline_resolution)),
            "selected_header_roi_at_1280x720": list(roi),
            "header_ink_coverage": {"scan_bbox_at_1280x720": list(scan),
                                    "ink_bbox_at_1280x720": ink_box,
                                    "ink_pixels": int(len(xs)), "complete": bool(covered)}}


class BattleObserver:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.anchors = {}
        self.cards = {}
        self.last_count_diagnostic = None

    def remember(self, snapshot):
        if snapshot.scene not in {"enemy_village", "battle"} or snapshot.confidence < .8:
            self.anchors.clear()
            self.cards.clear()
            return
        image = cv2.resize(read_frame(snapshot.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
        texts = snapshot.observations.get("ocr", [])
        timer = [item for item in texts if any(word in item["text"].replace(" ", "") for word in
                 ("战斗开始倒计时", "开战倒计时", "离战斗结束还有", "战斗结束倒计时", "battlestartsin"))
                 and item.get("bbox") and item["confidence"] >= .9]
        buttons = [item for item in snapshot.observations.get("buttons", []) if item["name"] == "end_battle"]
        if len(timer) != 1 or len(buttons) != 1:
            return
        anchors = []
        for item in (timer[0], buttons[0]):
            box = scale_box(tuple(item["bbox"]), from_resolution=self.recognizer.baseline_resolution,
                            to_resolution=(1280, 720))
            if item is timer[0] and snapshot.scene == "enemy_village":
                box = (_static_preparation_label_box(snapshot, self.recognizer.baseline_resolution,
                                                      item) or box)
            x0, y0, x1, y1 = box
            anchors.append((box, image[y0:y1, x0:x1].copy()))
        self.anchors[snapshot.scene] = anchors
        for slot in (snapshot.observations.get("battle") or {}).get("slots", []):
            if slot["kind"] != "troop" or not slot.get("count"):
                continue
            box = scale_box(tuple(slot["bbox"]), from_resolution=self.recognizer.baseline_resolution,
                            to_resolution=(1280, 720))
            left, top, right, bottom = box
            # Portrait interiors exclude quantities, levels and the selected border.
            patch = image[top + 38:min(top + 78, bottom), left + 20:right - 15]
            if patch.size:
                self.cards[round(slot["point"][0])] = (box, cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).copy())

    def guard(self, image):
        for scene, anchors in self.anchors.items():
            scores = []
            for (x0, y0, x1, y1), patch in anchors:
                current = image[y0:y1, x0:x1]
                score = float(cv2.matchTemplate(current, patch, cv2.TM_CCOEFF_NORMED)[0, 0])
                difference = float(np.abs(current.astype(np.int16) - patch).mean())
                # NCC alone tolerates a dimmed modal backdrop; absolute brightness does not.
                if not math.isfinite(score) or score < .94 or difference > 15:
                    break
                scores.append(score)
            if len(scores) == 2:
                return scene, min(scores)
        return None

    def count(self, path, image, slot):
        self.last_count_diagnostic = None
        saved = self.cards.get(round(slot["point"][0]))
        if saved is None:
            return None
        (left, top, right, bottom), patch = saved
        x0, x1 = max(0, left - 12), min(1280, right + 12)
        y0, y1 = max(0, top + 15), min(720, bottom - 20)
        region = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        matches = []
        for scale in (1.0, 1.05, 1.1):
            template = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
                continue
            _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))
            if math.isfinite(score):
                matches.append((score, scale, point))
        if not matches or max(matches)[0] < .92:
            return None
        score, scale, (x, y) = max(matches)
        new_left, new_top = round(x0 + x - 20 * scale), round(y0 + y - 38 * scale)
        new_right = round(new_left + (right - left) * scale)
        roi = (max(0, new_left + 3), max(0, new_top - 2), min(1280, new_right - 2), min(720, new_top + 33))
        # Use the native-resolution crop for small glyphs; no resize/detection round trip.
        source = read_frame(path)
        height, width = source.shape[:2]
        a, b, c, d = scale_box(roi, from_resolution=(1280, 720), to_resolution=(width, height))
        reader = getattr(self.recognizer.provider, "recognize_line_image", None)
        if not callable(reader):
            return None
        crop = source[b:d, a:c]
        reads = reader(crop)
        raw_reads = reads
        def strong_counts(items):
            values = set()
            for item in items:
                match = re.fullmatch(r"[xX×]\s*(\d+)", str(item.text).strip())
                confidence = item.confidence
                if (match is not None and type(confidence) in (int, float) and
                        math.isfinite(confidence) and confidence >= .9):
                    values.add(int(match[1]))
            return values

        numeric_evidence = strong_counts(reads)
        def valid(items):
            return (len(items) == 1 and math.isfinite(items[0].confidence)
                    and max(.9, self.recognizer.ocr_config.confidence_threshold) <= items[0].confidence <= 1
                    and re.fullmatch(r"[xX×]\s*(\d+)", items[0].text.strip()))

        selected = _selected_count_header(
            path, image, (left, top, right, bottom),
            client_version=self.recognizer.client_version,
            baseline_resolution=self.recognizer.baseline_resolution)
        if selected is not None:
            diagnostic = {**selected, "count_reads": []}
            self.last_count_diagnostic = diagnostic
        if len(numeric_evidence) > 1:
            diagnostic = self.last_count_diagnostic or {"count_reads": []}
            diagnostic["count_reads"].append({
                "method": "fast_header_line",
                "roi": list(scale_box(roi, from_resolution=(1280, 720),
                                      to_resolution=self.recognizer.baseline_resolution)),
                "roi_at_1280x720": list(roi),
                "readings": [{"text": item.text, "confidence": item.confidence}
                             for item in reads]})
            diagnostic["strong_counts"] = sorted(numeric_evidence)
            return {"reason": "strong_count_conflict", "counts": sorted(numeric_evidence),
                    "frame": str(path), "slot_bbox": list(slot["bbox"]),
                    "count_diagnostic": diagnostic}
        if selected is not None:
            if selected["header_ink_coverage"]["complete"]:
                selected_roi = tuple(selected["selected_header_roi_at_1280x720"])
                a, b, c, d = scale_box(selected_roi, from_resolution=(1280, 720),
                                       to_resolution=(width, height))
                selected_crop = source[b:d, a:c]
                views = (("fast_header_line", roi, reads),
                         ("fast_header_threshold_line", roi,
                          reader(cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                               120, 255, cv2.THRESH_BINARY)[1])),
                         ("fast_selected_header_line", selected_roi, reader(selected_crop)),
                         ("fast_selected_header_threshold_line", selected_roi,
                          reader(cv2.threshold(cv2.cvtColor(selected_crop, cv2.COLOR_BGR2GRAY),
                                               120, 255, cv2.THRESH_BINARY)[1])))
                proofs = []
                all_strong = set()
                for method, proof_roi, results in views:
                    all_strong.update(strong_counts(results))
                    diagnostic["count_reads"].append({
                        "method": method,
                        "roi": list(scale_box(proof_roi, from_resolution=(1280, 720),
                                              to_resolution=self.recognizer.baseline_resolution)),
                        "roi_at_1280x720": list(proof_roi),
                        "readings": [{"text": item.text, "confidence": item.confidence}
                                     for item in results]})
                    match = valid(results)
                    if match:
                        proofs.append((results[0].confidence, method, proof_roi,
                                       results[0], int(match[1])))
                diagnostic["strong_counts"] = sorted(all_strong)
                if len(all_strong) > 1:
                    return {"reason": "strong_count_conflict", "counts": sorted(all_strong),
                            "frame": str(path), "slot_bbox": list(slot["bbox"]),
                            "selected_header_evidence": diagnostic}
                if proofs:
                    confidence, method, proof_roi, item, count = max(proofs)
                    baseline_roi = scale_box(proof_roi, from_resolution=(1280, 720),
                                             to_resolution=self.recognizer.baseline_resolution)
                    return {"count": count, "confidence": confidence,
                            "frame": str(path), "slot_bbox": list(slot["bbox"]),
                            "roi": list(baseline_roi),
                            "readings": [{**asdict(item), "bbox": list(baseline_roi),
                                          "source": method}],
                            "portrait_score": score, "portrait_scale": scale,
                            "portrait_source": "current_frame_saved_battle_portrait",
                            "selected_header_evidence": diagnostic}
                return None

        match = valid(reads)
        method = "fast_header_line"
        if not match:
            binary = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)[1]
            reads = reader(binary)
            numeric_evidence.update(strong_counts(reads))
            if len(numeric_evidence) > 1:
                baseline_roi = scale_box(roi, from_resolution=(1280, 720),
                                         to_resolution=self.recognizer.baseline_resolution)
                diagnostic = dict(self.last_count_diagnostic or {})
                diagnostic.update({"strong_counts": sorted(numeric_evidence),
                                   "count_reads": [
                                       {"method": name, "roi": list(baseline_roi),
                                        "roi_at_1280x720": list(roi),
                                        "readings": [{"text": item.text,
                                                      "confidence": item.confidence}
                                                     for item in items]}
                                       for name, items in (
                                           ("fast_header_line", raw_reads),
                                           ("fast_header_threshold_line", reads))]})
                return {"reason": "strong_count_conflict", "counts": sorted(numeric_evidence),
                        "frame": str(path), "slot_bbox": list(slot["bbox"]),
                        "count_diagnostic": diagnostic}
            match = valid(reads)
            method = "fast_header_threshold_line"
        if not match:
            return None
        baseline_roi = scale_box(roi, from_resolution=(1280, 720), to_resolution=self.recognizer.baseline_resolution)
        return {"count": int(match[1]), "confidence": reads[0].confidence,
                "frame": str(path), "slot_bbox": list(slot["bbox"]), "roi": list(baseline_roi),
                "readings": [{**asdict(reads[0]), "bbox": list(baseline_roi), "source": method}],
                "portrait_score": score, "portrait_scale": scale,
                "portrait_source": "current_frame_saved_battle_portrait",
                **({"selected_header_evidence": self.last_count_diagnostic}
                   if selected is not None else {})}

    @staticmethod
    def modal_visible(image):
        hsv = cv2.cvtColor(image[140:540, 300:980], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 55, 245]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        return any(width > 220 and height > 90 and area / (width * height) > .65
                   for _, _, width, height, area in stats[1:])

    def observe(self, path: Path, *, purpose: str, slot: dict | None, previous: SceneSnapshot | None):
        started = time.monotonic()
        if previous is not None and previous.observations.get("recognition_scope", "full") == "full":
            self.remember(previous)
        source = read_frame(path)
        height, width = source.shape[:2]
        image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
        decoded = time.monotonic()
        from .terrain import measure_cloud_cover

        cloud = measure_cloud_cover(path)
        modal = self.modal_visible(image)
        guarded = (None if cloud["obscured"] or modal or
                   purpose == "line_boundary_wait" and self.recognizer.client_version != "18.647.2"
                   else self.guard(image))
        guarded_at = time.monotonic()
        reading = None
        self.last_count_diagnostic = None
        selection_wait = purpose == "mixed_selection_wait"
        if guarded is not None and purpose in {"troop_count", "mixed_selection_wait"} and slot is not None:
            reading = self.count(path, image, slot)
        count_at = time.monotonic()
        timings = {"prepare": decoded - started, "guard": guarded_at - decoded,
                   "count": count_at - guarded_at}
        if isinstance(reading, dict) and reading.get("reason") == "strong_count_conflict":
            # A second rendering of the same glyph may not erase a strong
            # contradictory number. Keep this frame unusable for actions.
            return SceneSnapshot("unknown", 0.0, path, {
                "recognition_scope": purpose, "fast_fallback": False,
                "recognition_seconds": time.monotonic() - started,
                "recognition_timings_sec": timings,
                "source_resolution": [width, height],
                "baseline_resolution": list(self.recognizer.baseline_resolution),
                "deployment_count_conflict": reading,
                "deployment_count_reads": [reading], "buttons": [], "ocr": [],
                "battle": {"slots": []}, "matched_reasons": ["strong_count_conflict"],
            })
        if selection_wait and reading is not None and (
                not isinstance(reading, dict) or type(reading.get("count")) is not int or
                reading["count"] < 0 or
                type(reading.get("confidence")) not in (int, float) or
                not math.isfinite(reading["confidence"]) or reading["confidence"] < .9 or
                not isinstance(reading.get("readings"), list) or not reading["readings"]):
            return SceneSnapshot("unknown", 0.0, path, {
                "recognition_scope": purpose, "fast_fallback": False,
                "recognition_seconds": time.monotonic() - started,
                "recognition_timings_sec": timings,
                "source_resolution": [width, height],
                "baseline_resolution": list(self.recognizer.baseline_resolution),
                "buttons": [], "ocr": [], "battle": {"slots": []},
                "matched_reasons": ["mixed_selection_invalid_count"],
            })
        if selection_wait and (guarded is None or slot is None):
            return SceneSnapshot("unknown", 0.0, path, {
                "recognition_scope": purpose, "fast_fallback": False,
                "recognition_seconds": time.monotonic() - started,
                "recognition_timings_sec": timings,
                "source_resolution": [width, height],
                "baseline_resolution": list(self.recognizer.baseline_resolution),
                "cloud_cover": cloud, "modal_visible": modal,
                "buttons": [], "ocr": [], "battle": {"slots": []},
                "matched_reasons": ["mixed_selection_guard_unverified"],
            })
        if purpose == "line_boundary_wait" and (
                guarded is None or guarded[0] != "enemy_village"):
            # A boundary miss is retryable only after the current preview HUD
            # has been proved. Never turn an ambiguous frame into a no-line
            # decision by falling back to full OCR or reusing an older scene.
            return SceneSnapshot("unknown", 0.0, path, {
                "recognition_scope": purpose, "fast_fallback": False,
                "recognition_seconds": time.monotonic() - started,
                "recognition_timings_sec": timings,
                "source_resolution": [width, height],
                "baseline_resolution": list(self.recognizer.baseline_resolution),
                "cloud_cover": cloud, "modal_visible": modal,
                "buttons": [], "ocr": [], "battle": {"slots": []},
                "matched_reasons": ["line_boundary_hud_unverified"],
            })
        if guarded is None or (purpose == "troop_count" and reading is None):
            snapshot = self.recognizer.recognize(path)
            if self.last_count_diagnostic is not None:
                snapshot.observations["deployment_count_diagnostic"] = self.last_count_diagnostic
            if modal and snapshot.scene in {"battle", "enemy_village"}:
                snapshot = SceneSnapshot("unknown", 0.0, path,
                                         {**snapshot.observations, "buttons": [], "modal_visible": True})
            self.remember(snapshot)
            snapshot.observations.update(recognition_scope="full", recognition_requested=purpose,
                                         fast_fallback=True, recognition_seconds=time.monotonic() - started)
            timings["fallback"] = time.monotonic() - count_at
            snapshot.observations["recognition_timings_sec"] = timings
            return snapshot
        scene, confidence = guarded
        snapshot = SceneSnapshot(scene, confidence, path, {
            "recognition_scope": purpose, "fast_fallback": False,
            "recognition_seconds": time.monotonic() - started,
            "recognition_timings_sec": timings,
            "source_resolution": [width, height], "baseline_resolution": list(self.recognizer.baseline_resolution),
            "cloud_cover": cloud, "modal_visible": modal,
            "buttons": [], "ocr": [], "battle": {"slots": []},
            "matched_reasons": ["two_current_hud_anchors"],
        })
        if selection_wait and reading is None:
            # This frame proves only the scene guard. It is not a count, a
            # selected card, or permission to issue any input.
            snapshot.observations["deployment_count_pending"] = {
                "frame": str(path), "slot_bbox": list(slot["bbox"]),
                "purpose": purpose, "reason": "current_frame_count_unavailable"}
            return snapshot
        if reading is not None:
            snapshot.observations["deployment_count_reads"] = [reading]
            snapshot.observations["ocr"] = reading["readings"]
        return snapshot
