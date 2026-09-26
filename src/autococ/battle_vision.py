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
    return None


class BattleObserver:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.anchors = {}
        self.cards = {}

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
        def valid(items):
            return (len(items) == 1 and math.isfinite(items[0].confidence)
                    and max(.9, self.recognizer.ocr_config.confidence_threshold) <= items[0].confidence <= 1
                    and re.fullmatch(r"[xX×]\s*(\d+)", items[0].text.strip()))

        match = valid(reads)
        method = "fast_header_line"
        if not match:
            binary = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)[1]
            reads = reader(binary)
            match = valid(reads)
            method = "fast_header_threshold_line"
        if not match:
            return None
        baseline_roi = scale_box(roi, from_resolution=(1280, 720), to_resolution=self.recognizer.baseline_resolution)
        return {"count": int(match[1]), "confidence": reads[0].confidence,
                "frame": str(path), "slot_bbox": list(slot["bbox"]), "roi": list(baseline_roi),
                "readings": [{**asdict(reads[0]), "bbox": list(baseline_roi), "source": method}],
                "portrait_score": score, "portrait_scale": scale}

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
        guarded = None if cloud["obscured"] or modal else self.guard(image)
        guarded_at = time.monotonic()
        reading = None
        if guarded is not None and purpose == "troop_count" and slot is not None:
            reading = self.count(path, image, slot)
        count_at = time.monotonic()
        timings = {"prepare": decoded - started, "guard": guarded_at - decoded,
                   "count": count_at - guarded_at}
        if guarded is None or (purpose == "troop_count" and reading is None):
            snapshot = self.recognizer.recognize(path)
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
            "cloud_cover": cloud, "buttons": [], "ocr": [], "battle": {"slots": []},
            "matched_reasons": ["two_current_hud_anchors"],
        })
        if reading is not None:
            snapshot.observations["deployment_count_reads"] = [reading]
            snapshot.observations["ocr"] = reading["readings"]
        return snapshot
