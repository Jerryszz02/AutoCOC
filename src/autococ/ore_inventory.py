"""Read the three ores on the 18.647.2 hero-equipment page.

This module does not navigate to the blacksmith or establish a reward delta.
Recognition is read-only; the optional close helper acts only on a current,
fully recognized page supplied by the session recognizer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable

import cv2
import numpy as np

from .device import foreground_package
from .errors import FlowError


CLIENT_VERSION = "18.647.2"
BASELINE = (1280, 720)
_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates"
_ANCHORS = {
    "title": ("英雄装备", (575, 55, 695, 105), .99),
    "hint": ("拖动图标为英雄配置装备", (1020, 370, 1220, 415), .99),
    "close": ("X", (1195, 55, 1250, 110), .90),
}
_ORES = {
    "shiny": {"roi": (425, 615, 555, 655), "icon": (399, 620, 430, 654), "capacity": 50000},
    "glowy": {"roi": (590, 615, 715, 655), "icon": (565, 620, 597, 654), "capacity": 5000},
    "starry": {"roi": (760, 615, 865, 655), "icon": (734, 620, 767, 654), "capacity": 1000},
}
_CLOSE_VISUAL_ROI = (1200, 58, 1247, 105)
_COUNT_RE = re.compile(r"([0-9]{1,6})/([0-9]{1,6})\Z")
_LINE_ROIS = {"starry": (764, 621, 856, 652)}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _inside(box: object, roi: tuple[int, int, int, int]) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value) for value in box)
            and roi[0] <= box[0] < box[2] <= roi[2]
            and roi[1] <= box[1] < box[3] <= roi[3])


def _overlaps(box: object, roi: tuple[int, int, int, int]) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value) for value in box)
            and box[0] < roi[2] and box[2] > roi[0]
            and box[1] < roi[3] and box[3] > roi[1])


def _visual_evidence(image: np.ndarray, name: str, roi: tuple[int, int, int, int]) -> dict:
    template = cv2.imread(str(_TEMPLATES / f"ore_inventory_{name}_18_647_2.png"))
    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]
    if template is None or template.shape != crop.shape:
        return {"ncc": 0.0, "mad": math.inf, "mean_delta": math.inf}
    score = float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)[0, 0])
    # Normalized correlation alone is invariant to global dimming: a modal
    # overlay can leave all four icon shapes and stale OCR visible behind it.
    # Both saved 18.647.2 samples have 0.0 MAD/mean delta for these exact ROIs;
    # the reproduced 65% brightness overlay has 29-43 on each ROI.
    mad = float(np.mean(np.abs(crop.astype(np.float32) - template.astype(np.float32))))
    mean_delta = abs(float(np.mean(crop)) - float(np.mean(template)))
    return {"ncc": score if math.isfinite(score) and 0 <= score <= 1 else 0.0,
            "mad": mad, "mean_delta": mean_delta}


def _unique_text(ocr: list[dict], text: str, roi: tuple[int, int, int, int],
                 minimum: float) -> dict | None:
    # A duplicate or weak candidate is ambiguity, not a reason to choose one.
    matches = [item for item in ocr if item.get("text") == text and _inside(item.get("bbox"), roi)]
    if len(matches) != 1:
        return None
    confidence = matches[0].get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not minimum <= confidence <= 1:
        return None
    return matches[0]


def _canonical_ocr(item: object, roi: tuple[int, int, int, int]) -> dict | None:
    if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not _inside(item.get("bbox"), roi):
        return None
    confidence = item.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    return {"text": item["text"], "confidence": confidence, "bbox": list(item["bbox"])}


def _line_digest(evidence: dict) -> str:
    payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _local_count(frame: Path, frame_sha: str, name: str, zone: list[dict],
                 line_reader: Callable[[Path, tuple[int, int, int, int]], list[dict]] | None,
                 cached_local: dict) -> tuple[dict, dict] | None:
    """Use one same-frame line read only when the full-frame count is invalid.

    A previously bound local result is revalidated without running OCR again.
    Both paths keep the original full-frame text; neither edits or reverses it.
    """
    roi = _LINE_ROIS.get(name)
    if roi is None:
        return None
    full = [_canonical_ocr(item, _ORES[name]["roi"]) for item in zone]
    if any(item is None for item in full):
        return None
    if name in cached_local:
        evidence = cached_local[name]
        if (not isinstance(evidence, dict) or set(evidence) !=
                {"source", "frame", "frame_sha256", "roi", "full_ocr", "readings", "evidence_sha256"}):
            return None
        fields = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        if (fields.get("source") != "same_frame_line_roi"
                or fields.get("frame") != str(frame)
                or fields.get("frame_sha256") != frame_sha
                or fields.get("roi") != list(roi)
                or fields.get("full_ocr") != full):
            return None
        readings = fields.get("readings")
        if (not isinstance(readings, list) or len(readings) != 1
                or _canonical_ocr(readings[0], roi) != readings[0]):
            return None
        if (not isinstance(evidence.get("evidence_sha256"), str)
                or evidence["evidence_sha256"] != _line_digest(fields)):
            return None
    else:
        if line_reader is None:
            return None
        try:
            readings = line_reader(frame, roi)
        except Exception:
            return None
        if not isinstance(readings, list) or len(readings) != 1:
            return None
        reading = _canonical_ocr(readings[0], roi)
        if reading is None:
            return None
        readings = [reading]
        if _sha256(frame) != frame_sha:
            return None
        fields = {"source": "same_frame_line_roi", "frame": str(frame),
                  "frame_sha256": frame_sha, "roi": list(roi),
                  "full_ocr": full, "readings": readings}
        evidence = {**fields, "evidence_sha256": _line_digest(fields)}
    item = _canonical_ocr(readings[0], roi)
    if item is None or item["confidence"] < .99:
        return None
    match = _COUNT_RE.fullmatch(item["text"])
    if match is None:
        return None
    amount, capacity = map(int, match.groups())
    if capacity != _ORES[name]["capacity"] or amount > capacity:
        return None
    # A weak but syntactically valid full-frame value may corroborate the
    # local line; a different value is still a conflict, regardless of score.
    if any(_COUNT_RE.fullmatch(raw["text"]) is not None and raw["text"] != item["text"]
           for raw in full):
        return None
    return item, evidence


def observe_ore_inventory(frame_path: str | Path, ocr: list[dict], *,
                          client_version: str,
                          baseline_resolution: tuple[int, int] = BASELINE,
                          line_reader: Callable[[Path, tuple[int, int, int, int]], list[dict]] | None = None,
                          cached_local: dict | None = None) -> dict | None:
    """Recognize a complete page and exactly one strong count per ore.

    The caller supplies full-frame OCR with boxes in 1280x720 coordinates.
    None means the surface or a number is ambiguous. No value is inferred from
    another frame or from the event reward amount.
    """
    if (client_version != CLIENT_VERSION or
            not isinstance(baseline_resolution, (list, tuple)) or tuple(baseline_resolution) != BASELINE):
        return None
    if not isinstance(ocr, list) or any(not isinstance(item, dict) for item in ocr):
        return None
    cache_supplied = cached_local is not None
    if cached_local is None:
        cached_local = {}
    if not isinstance(cached_local, dict) or set(cached_local) - set(_LINE_ROIS):
        return None
    if any("另一个设备正在连接这个村庄" in str(item.get("text", "")) for item in ocr):
        return None
    frame = Path(frame_path)
    image = cv2.imread(str(frame))
    if image is None or (image.shape[1], image.shape[0]) not in {BASELINE, (2560, 1440)}:
        return None
    image = cv2.resize(image, BASELINE, interpolation=cv2.INTER_AREA)
    anchors: dict[str, dict] = {}
    for name, (text, roi, threshold) in _ANCHORS.items():
        item = _unique_text(ocr, text, roi, threshold)
        if item is None:
            return None
        anchors[name] = {"text": text, "bbox": list(item["bbox"]), "confidence": item["confidence"]}
    visual = {name: _visual_evidence(image, name, spec["icon"]) for name, spec in _ORES.items()}
    visual["close"] = _visual_evidence(image, "close", _CLOSE_VISUAL_ROI)
    if any(item["ncc"] < .97 or item["mad"] > 8 or item["mean_delta"] > 6
           for item in visual.values()):
        return None
    frame_sha = _sha256(frame)
    ores: dict[str, dict] = {}
    local_counts: dict[str, dict] = {}
    used: set[int] = set()
    for name, spec in _ORES.items():
        # Any slash-like OCR in the count zone is a conflict, including malformed
        # or weak readings. A split reading cannot silently become a count.
        zone = [item for item in ocr if _inside(item.get("bbox"), spec["roi"])
                and ("/" in str(item.get("text", "")) or
                     any(char.isdigit() for char in str(item.get("text", ""))))]
        if len(zone) > 1 or any(id(item) in used for item in zone):
            return None
        item = zone[0] if zone else None
        if item is not None:
            used.add(id(item))
        match = (_COUNT_RE.fullmatch(item["text"])
                 if item is not None and isinstance(item.get("text"), str) else None)
        confidence = item.get("confidence") if item is not None else None
        strong = (match is not None and type(confidence) in (int, float)
                  and math.isfinite(confidence) and .99 <= confidence <= 1)
        if strong:
            if name in cached_local:
                return None  # A cached local value must not override a strong full read.
            source = "full_ocr"
        else:
            local = _local_count(frame, frame_sha, name, zone, line_reader, cached_local)
            if local is None:
                return None
            item, evidence = local
            local_counts[name] = evidence
            match = _COUNT_RE.fullmatch(item["text"])
            confidence = item["confidence"]
            source = "same_frame_line_roi"
        amount, capacity = map(int, match.groups())
        if capacity != spec["capacity"] or amount > capacity:
            return None
        ores[name] = {"amount": amount, "capacity": capacity,
                      "text": item["text"], "bbox": list(item["bbox"]),
                      "confidence": confidence, "icon_score": visual[name]["ncc"],
                      "source": source}
    if (cache_supplied and set(local_counts) != set(cached_local)) or _sha256(frame) != frame_sha:
        return None
    # A second complete count in the inventory strip outside the known zones
    # suggests a shifted page or a fourth/duplicated reading.
    if any(id(item) not in used and _overlaps(item.get("bbox"), (380, 605, 900, 665))
           and ("/" in str(item.get("text", "")) or
                any(char.isdigit() for char in str(item.get("text", ""))))
           for item in ocr):
        return None
    close = anchors["close"]["bbox"]
    return {"surface": "hero_equipment_ore_inventory", "client_version": client_version,
            "frame": str(frame), "frame_sha256": frame_sha,
            "anchors": anchors, "visual": visual, "ores": ores,
            "local_counts": local_counts,
            "balances": {name: item["amount"] for name, item in ores.items()},
            "capacities": {name: item["capacity"] for name, item in ores.items()},
            "button": {"name": "close_ore_inventory", "text": "关闭英雄装备", "bbox": close,
                       "point": [(close[0] + close[2]) // 2, (close[1] + close[3]) // 2],
                       "confidence": min(anchors["close"]["confidence"], visual["close"]["ncc"])},
            "account_verified": False, "reward_inventory_verified": False,
            "formal_success": False}


def read_ore_inventory_snapshot(snapshot, *, client_version: str) -> dict | None:
    """Bind a pure page reading to its capture identity for two-frame checks."""
    observations = snapshot.observations
    baseline = observations.get("baseline_resolution", BASELINE)
    if not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
        return None
    bound = observations.get("ore_inventory")
    cached_local = bound.get("local_counts") if isinstance(bound, dict) else None
    result = observe_ore_inventory(snapshot.screenshot_path, observations.get("ocr"),
                                   client_version=client_version,
                                   baseline_resolution=tuple(baseline), cached_local=cached_local)
    if result is None:
        return None
    frame_id = observations.get("frame_id")
    observed = observations.get("observed_at_monotonic")
    if (not isinstance(frame_id, str) or not frame_id or
            type(observed) not in (int, float) or not math.isfinite(observed)):
        return None
    result["frame_id"] = frame_id
    result["observed_at_monotonic"] = observed
    return result


def verify_stable_ore_inventory(first: dict, second: dict, *, max_gap_sec: float = 30) -> dict | None:
    """Require two independent, ordered fresh captures with identical ores."""
    if (not isinstance(first, dict) or not isinstance(second, dict)
            or type(max_gap_sec) not in (int, float) or not math.isfinite(max_gap_sec)
            or not 0 < max_gap_sec <= 30):
        return None
    if any(item.get("surface") != "hero_equipment_ore_inventory" or
           item.get("client_version") != CLIENT_VERSION or
           not isinstance(item.get("frame"), str) or not item["frame"] or
           not isinstance(item.get("frame_sha256"), str) or
           not re.fullmatch(r"[0-9a-f]{64}", item["frame_sha256"]) or
           not isinstance(item.get("frame_id"), str) or not item["frame_id"] or
           type(item.get("observed_at_monotonic")) not in (int, float) or
           not math.isfinite(item["observed_at_monotonic"])
           for item in (first, second)):
        return None
    if (first["frame"] == second["frame"] or first["frame_id"] == second["frame_id"] or
            not 0 < second["observed_at_monotonic"] - first["observed_at_monotonic"] <= max_gap_sec or
            not 0 <= time.monotonic() - second["observed_at_monotonic"] <= 8):
        return None
    expected = tuple(_ORES)
    for item in (first, second):
        ores = item.get("ores")
        if not isinstance(ores, dict) or set(ores) != set(expected):
            return None
        if any(not isinstance(ores[name], dict) or
               type(ores[name].get(key)) is not int
               for name in expected for key in ("amount", "capacity")):
            return None
        if any(not 0 <= ores[name]["amount"] <= ores[name]["capacity"] or
               ores[name]["capacity"] != _ORES[name]["capacity"]
               for name in expected):
            return None
    balances = {name: {"amount": first["ores"][name]["amount"],
                       "capacity": first["ores"][name]["capacity"]} for name in expected}
    if any(balances[name] != {"amount": second["ores"][name]["amount"],
                              "capacity": second["ores"][name]["capacity"]} for name in expected):
        return None
    return {"status": "stable_inventory_read", "client_version": CLIENT_VERSION,
            "ores": balances, "frames": [{"frame": item["frame"], "frame_id": item["frame_id"],
                                          "frame_sha256": item["frame_sha256"]} for item in (first, second)],
            "account_verified": False, "reward_inventory_verified": False,
            "formal_success": False}


def close_ore_inventory(session, snapshot, *, timeout_sec: float = 15) -> dict:
    """Close only a current recognizer-bound page, then require a fresh HOME."""
    if type(timeout_sec) not in (int, float) or not 0 < timeout_sec <= 30:
        raise ValueError("Invalid ore-inventory close timeout")
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    if (session.client_version != CLIENT_VERSION or
            tuple(session.config.game.baseline_resolution) != BASELINE):
        raise FlowError("Unsupported hero-equipment page version")
    session.check_deadline()
    if snapshot is not session.last_snapshot or snapshot.scene != "popup" or snapshot.confidence < .95:
        raise FlowError("Hero-equipment page is not the current recognized popup")
    expected = snapshot.observations.get("ore_inventory")
    actual = read_ore_inventory_snapshot(snapshot, client_version=session.client_version)
    captured = snapshot.observations.get("observed_at_monotonic")
    age = time.monotonic() - captured if type(captured) in (int, float) and math.isfinite(captured) else math.inf
    pure = ({key: value for key, value in actual.items()
             if key not in {"frame_id", "observed_at_monotonic"}} if actual is not None else None)
    if (actual is None or not isinstance(expected, dict) or expected != pure or
            not 0 <= age <= 8 or snapshot.observations.get("account_in_use") is not None):
        raise FlowError("Hero-equipment close source is stale or changed")
    if time.monotonic() >= end or actual["frame_sha256"] != _sha256(snapshot.screenshot_path):
        raise FlowError("Hero-equipment close deadline or frame changed")
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise FlowError("Hero-equipment close deadline expired before foreground check")
    foreground = foreground_package(session.capture.adb, session.capture.serial,
                                    timeout_sec=min(5, session.config.runtime.step_timeout_sec, remaining))
    session.check_deadline()
    if foreground != session.config.game.package_name:
        raise FlowError("Hero-equipment game is not foreground")
    captured_now = snapshot.observations.get("observed_at_monotonic")
    age_now = (time.monotonic() - captured_now if type(captured_now) in (int, float)
               and math.isfinite(captured_now) else math.inf)
    if (time.monotonic() >= end or snapshot is not session.last_snapshot or
            not 0 <= age_now <= 8 or captured_now != actual["observed_at_monotonic"] or
            snapshot.observations.get("frame_id") != actual["frame_id"] or
            snapshot.observations.get("account_in_use") is not None or
            snapshot.observations.get("ore_inventory") != pure or
            actual["frame_sha256"] != _sha256(snapshot.screenshot_path)):
        raise FlowError("Hero-equipment close source changed during foreground check")
    session.tap(snapshot, tuple(actual["button"]["point"]), reason="Close recognized hero-equipment page")
    for index in range(3):
        session.check_deadline()
        if time.monotonic() >= end:
            break
        current = session.observe(f"ore-inventory-close-{index + 1}", purpose="full")
        buttons = {item.get("name") for item in current.observations.get("buttons", [])
                   if isinstance(item, dict)}
        if (current.scene == "village" and current.confidence >= .8 and
                current.observations.get("village_type") == "home" and
                current.observations.get("resource_source") == "village_inventory" and
                {"attack", "shop"} <= buttons):
            return {"status": "home", "frame": str(current.screenshot_path),
                    "reward_inventory_verified": False, "formal_success": False}
    raise FlowError("Hero-equipment close did not return to a confirmed HOME")
