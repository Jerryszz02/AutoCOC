"""Version-bound, read-only navigation to the hero-equipment ore inventory.

The HOME building tap only selects a candidate.  Opening Equipment requires
fresh OCR of the exact selected blacksmith and a matching button icon.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .device import foreground_package
from .errors import FlowError, StopRequested
from .home_evidence import home_navigation_verified
from .ore_inventory import (close_ore_inventory, read_ore_inventory_snapshot,
                            verify_stable_ore_inventory)


CLIENT_VERSION = "18.647.2"
BASELINE = (1280, 720)
_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates"
# Calibrated visible building-map area for this account at the fixed HOME zoom.
# It is not a general full-screen or cross-account blacksmith locator.
_SEARCH = (300, 115, 1000, 470)
_FRONTS = {
    "new": {"file": "ore_navigation_blacksmith_front_new_18_647_2.png",
            "reference_roi": (758, 197, 776, 219), "tap_offset": (1, 10)},
    "old": {"file": "ore_navigation_blacksmith_front_old_18_647_2.png",
            "reference_roi": (754, 202, 772, 224), "tap_offset": (2, 10)},
    # Same verified front at the independently measured ~1.03 HOME map zoom.
    "zoom103": {"file": "ore_navigation_blacksmith_front_zoom103_18_647_2.png",
                "reference_roi": (762, 192, 780, 214), "tap_offset": (1, 10)},
    # Same building selected at (753, 200) in the separately reviewed 123737
    # TRAIN calibration. Keep the old variants for their independently observed
    # map scales; a low-scoring different zoom must still be refused.
    "verified123737": {"file": "ore_navigation_blacksmith_front_verified_123737_18_647_2.png",
                       "reference_roi": (758, 197, 776, 219), "tap_offset": (-5, 3)},
    # The same building at the separately reviewed smaller HOME map scale.
    # Its tap is the calibrated selection point projected through the measured
    # same-village affine; Equipment still requires a fresh exact title/icon.
    "small120645": {"file": "ore_navigation_blacksmith_front_small_verified_120645_18_647_2.png",
                    "reference_roi": (751, 207, 769, 229), "tap_offset": (-5, 3)},
}
_TITLE = "铁匠铺（10级）"
_TITLE_ROI = (535, 450, 740, 510)
_TITLE_LINE_ROI = (535, 452, 740, 508)
_EQUIPMENT_ROI = (680, 580, 740, 616)
_INFO_ROI = (540, 580, 610, 617)
_EQUIPMENT_ICON_ROI = (683, 527, 735, 575)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _ocr_digest(ocr: list[dict]) -> str:
    return hashlib.sha256(json.dumps(ocr, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


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


def _exact(ocr: list[dict], label: str, roi: tuple[int, int, int, int],
           minimum: float) -> dict | None:
    matches = [item for item in ocr if item.get("text") == label and _inside(item.get("bbox"), roi)]
    if len(matches) != 1:
        return None
    confidence = matches[0].get("confidence")
    return (matches[0] if type(confidence) in (int, float) and math.isfinite(confidence)
            and minimum <= confidence <= 1 else None)


def _home(snapshot) -> bool:
    o = snapshot.observations
    return (snapshot.scene == "village" and snapshot.confidence >= .8
            and o.get("village_type") == "home"
            and o.get("resource_source") == "village_inventory"
            and o.get("account_in_use") is None
            and home_navigation_verified(snapshot)
            and isinstance(o.get("ocr"), list)
            and not any("另一个设备正在连接这个村庄" in str(item.get("text", ""))
                        for item in o["ocr"] if isinstance(item, dict)))


def _frame(snapshot, *, client_version: str, baseline_resolution: tuple[int, int]):
    if client_version != CLIENT_VERSION or tuple(baseline_resolution) != BASELINE or not _home(snapshot):
        return None
    o = snapshot.observations
    if (not isinstance(o.get("frame_id"), str) or not o["frame_id"]
            or type(o.get("observed_at_monotonic")) not in (int, float)
            or not math.isfinite(o["observed_at_monotonic"])
            or not 0 <= time.monotonic() - o["observed_at_monotonic"] <= 8
            or any(not isinstance(item, dict) for item in o["ocr"])):
        return None
    path = Path(snapshot.screenshot_path)
    try:
        before = _sha256(path)
        image = cv2.imread(str(path))
        unchanged = _sha256(path) == before
    except OSError:
        return None
    if (image is None or (image.shape[1], image.shape[0]) not in {BASELINE, (2560, 1440)}
            or not unchanged):
        return None
    return path, cv2.resize(image, BASELINE, interpolation=cv2.INTER_AREA), before


def locate_blacksmith(snapshot, *, client_version: str,
                      baseline_resolution: tuple[int, int] = BASELINE) -> dict | None:
    """Find one visible front on a fresh HOME; never infer from old coordinates."""
    frame = _frame(snapshot, client_version=client_version, baseline_resolution=baseline_resolution)
    if frame is None:
        return None
    path, image, frame_sha = frame
    x1, y1, x2, y2 = _SEARCH
    search = image[y1:y2, x1:x2]
    maps = []
    for name, spec in _FRONTS.items():
        template = cv2.imread(str(_TEMPLATES / spec["file"]))
        if template is None or template.shape != (22, 18, 3):
            return None
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, where = cv2.minMaxLoc(scores)
        if not math.isfinite(score):
            return None
        maps.append((name, spec, template, scores, float(score), where))
    name, spec, template, _, best, where = max(maps, key=lambda item: item[4])
    candidate_x, candidate_y = where[0] + x1, where[1] + y1
    crop = image[candidate_y:candidate_y + 22, candidate_x:candidate_x + 18]
    mad = float(np.mean(np.abs(crop.astype(np.float32) - template.astype(np.float32))))
    mean_delta = abs(float(np.mean(crop)) - float(np.mean(template)))
    # The second candidate must be elsewhere, not another template/neighboring
    # pixel alignment for the same blacksmith front.
    runner_up = 0.0
    for _, _, _, scores, _, _ in maps:
        scores = scores.copy()
        left, top = candidate_x - x1, candidate_y - y1
        scores[max(0, top - 25):top + 26, max(0, left - 25):left + 26] = -1
        runner_up = max(runner_up, float(scores.max()))
    point = (candidate_x + spec["tap_offset"][0], candidate_y + spec["tap_offset"][1])
    if (best < .985 or mad > 3 or mean_delta > 3 or best - runner_up < .15
            or not (300 <= point[0] < 1000 and 115 <= point[1] < 470)
            or _sha256(path) != frame_sha):
        return None
    return {"surface": "home_blacksmith_candidate", "frame": str(path),
            "frame_sha256": frame_sha, "frame_id": snapshot.observations["frame_id"],
            "ocr_sha256": _ocr_digest(snapshot.observations["ocr"]),
            "template": spec["file"], "template_sha256": _sha256(_TEMPLATES / spec["file"]),
            "match_roi": [candidate_x, candidate_y, candidate_x + 18, candidate_y + 22],
            "point": list(point), "ncc": best, "mad": mad,
            "mean_delta": mean_delta, "runner_up": runner_up,
            "account_verified": False, "reward_inventory_verified": False,
            "formal_success": False}


def recognize_selected_blacksmith(
        snapshot, *, client_version: str,
        baseline_resolution: tuple[int, int] = BASELINE,
        line_reader: Callable[[Path, tuple[int, int, int, int]], list[dict]] | None = None) -> dict | None:
    """Require the exact selected title, Equipment text, and current anvil icon."""
    frame = _frame(snapshot, client_version=client_version, baseline_resolution=baseline_resolution)
    if frame is None:
        return None
    path, image, frame_sha = frame
    ocr = snapshot.observations["ocr"]
    icon = cv2.imread(str(_TEMPLATES / "ore_navigation_equipment_icon_18_647_2.png"))
    x1, y1, x2, y2 = _EQUIPMENT_ICON_ROI
    crop = image[y1:y2, x1:x2]
    if icon is None or icon.shape != crop.shape:
        return None
    score = float(cv2.matchTemplate(crop, icon, cv2.TM_CCOEFF_NORMED)[0, 0])
    mad = float(np.mean(np.abs(crop.astype(np.float32) - icon.astype(np.float32))))
    mean_delta = abs(float(np.mean(crop)) - float(np.mean(icon)))
    if not math.isfinite(score) or score < .995 or mad > 5 or mean_delta > 5:
        return None
    equipment = _exact(ocr, "装备", _EQUIPMENT_ROI, .99)
    info = _exact(ocr, "信息", _INFO_ROI, .99)
    if equipment is None or info is None:
        return None
    titles = [item for item in ocr if _inside(item.get("bbox"), _TITLE_ROI)
              and item.get("text") == _TITLE]
    if len(titles) != 1 or any(_overlaps(item.get("bbox"), _TITLE_ROI)
                               and item.get("text") != _TITLE and
                               ("级" in str(item.get("text", "")) or "铁匠" in str(item.get("text", "")))
                               for item in ocr):
        return None
    title = _exact(ocr, _TITLE, _TITLE_ROI, .9)
    title_source = "full_ocr"
    local = None
    if title is None:
        weak = titles[0]
        confidence = weak.get("confidence")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not .7 <= confidence < .9 or line_reader is None):
            return None
        try:
            readings = line_reader(path, _TITLE_LINE_ROI)
        except Exception:
            return None
        if not isinstance(readings, list) or len(readings) != 1:
            return None
        local = readings[0]
        if (not isinstance(local, dict) or local.get("text") != _TITLE
                or not _inside(local.get("bbox"), _TITLE_LINE_ROI)
                or type(local.get("confidence")) not in (int, float)
                or not math.isfinite(local["confidence"])
                or not .9 <= local["confidence"] <= 1
                or _sha256(path) != frame_sha):
            return None
        title = local
        title_source = "same_frame_line_roi"
    point = [(equipment["bbox"][0] + equipment["bbox"][2]) // 2,
             equipment["bbox"][1] - 22]
    if not (x1 <= point[0] < x2 and y1 <= point[1] < y2) or _sha256(path) != frame_sha:
        return None
    return {"surface": "selected_blacksmith_equipment", "frame": str(path),
            "frame_sha256": frame_sha, "frame_id": snapshot.observations["frame_id"],
            "ocr_sha256": _ocr_digest(ocr), "title": dict(title),
            "title_source": title_source, "full_title": dict(titles[0]),
            "local_title": dict(local) if local is not None else None,
            "equipment": dict(equipment), "info": dict(info),
            "icon_score": score, "icon_mad": mad, "icon_mean_delta": mean_delta,
            "point": point, "account_verified": False,
            "reward_inventory_verified": False, "formal_success": False}


def _looks_selected(snapshot) -> bool:
    """A different or unreadable selected-building menu must not trigger a tap."""
    ocr = snapshot.observations.get("ocr")
    if not isinstance(ocr, list):
        return False
    return any(isinstance(item, dict) and (
        (item.get("text") == "装备" and _inside(item.get("bbox"), _EQUIPMENT_ROI))
        or (item.get("text") == "信息" and _inside(item.get("bbox"), _INFO_ROI))
        or (_overlaps(item.get("bbox"), _TITLE_ROI) and
            ("级" in str(item.get("text", "")) or "铁匠" in str(item.get("text", "")))))
        for item in ocr)


def _wrong_selected_building(snapshot) -> bool:
    """A readable, different building title is a hard stop during selection."""
    ocr = snapshot.observations.get("ocr")
    return isinstance(ocr, list) and any(
        isinstance(item, dict) and _overlaps(item.get("bbox"), _TITLE_ROI)
        and item.get("text") != _TITLE
        and ("级" in str(item.get("text", "")) or "铁匠" in str(item.get("text", "")))
        for item in ocr)


def _observe_transition(session, label: str, previous, end: float):
    """Accept only a new, ordered capture while waiting for a page transition."""
    session.check_deadline()
    if time.monotonic() >= end:
        raise FlowError("Ore navigation transition deadline exceeded")
    current = session.observe(label, purpose="full")
    session.check_deadline()
    if time.monotonic() >= end:
        raise FlowError("Ore navigation transition deadline exceeded")
    o = current.observations
    prior = previous.observations
    captured = o.get("observed_at_monotonic")
    earlier = prior.get("observed_at_monotonic")
    if (current is not session.last_snapshot
            or session.client_version != CLIENT_VERSION
            or tuple(session.config.game.baseline_resolution) != BASELINE
            or o.get("account_in_use") is not None
            or not isinstance(o.get("ocr"), list)
            or any(isinstance(item, dict) and "另一个设备正在连接这个村庄" in str(item.get("text", ""))
                   for item in o["ocr"])
            or not isinstance(o.get("frame_id"), str) or not o["frame_id"]
            or o["frame_id"] == prior.get("frame_id")
            or str(current.screenshot_path) == str(previous.screenshot_path)
            or type(captured) not in (int, float) or not math.isfinite(captured)
            or type(earlier) not in (int, float) or not math.isfinite(earlier)
            or not earlier < captured <= time.monotonic()
            or time.monotonic() - captured > 8):
        raise FlowError("Ore navigation transition frame is stale, changed or account-blocked")
    return current


def _guard_current(session, snapshot, evidence: dict, end: float) -> None:
    """Check freshness again after the bounded foreground lookup before tap."""
    def current() -> bool:
        o = snapshot.observations
        captured = o.get("observed_at_monotonic")
        return (snapshot is session.last_snapshot and _home(snapshot)
                and time.monotonic() < end
                and type(captured) in (int, float) and math.isfinite(captured)
                and 0 <= time.monotonic() - captured <= 8
                and o.get("frame_id") == evidence.get("frame_id")
                and _ocr_digest(o["ocr"]) == evidence.get("ocr_sha256")
                and str(snapshot.screenshot_path) == evidence.get("frame")
                and _sha256(Path(snapshot.screenshot_path)) == evidence.get("frame_sha256"))

    session.check_deadline()
    if (session.client_version != CLIENT_VERSION or
            tuple(session.config.game.baseline_resolution) != BASELINE or not current()):
        raise FlowError("Blacksmith navigation source is stale or unsupported")
    age = time.monotonic() - snapshot.observations["observed_at_monotonic"]
    remaining = min(end - time.monotonic(), 8 - age)
    if remaining <= 0:
        raise FlowError("Blacksmith navigation freshness expired")
    foreground = foreground_package(session.capture.adb, session.capture.serial,
                                    timeout_sec=min(5, session.config.runtime.step_timeout_sec, remaining))
    session.check_deadline()
    if foreground != session.config.game.package_name or not current():
        raise FlowError("Blacksmith navigation foreground or source changed")


def _guard_unknown_observation(session, snapshot, end: float) -> None:
    """Allow only a fresh, foreground unknown frame to be observed again."""
    o = snapshot.observations
    path = Path(snapshot.screenshot_path)
    captured = o.get("observed_at_monotonic")
    frame_id = o.get("frame_id")
    ocr = o.get("ocr")
    if not isinstance(ocr, list) or not all(isinstance(item, dict) for item in ocr):
        raise FlowError("Unknown blacksmith observation OCR is incomplete")
    ocr_sha = _ocr_digest(ocr)

    def current() -> bool:
        return (snapshot is session.last_snapshot and snapshot.scene == "unknown"
                and session.client_version == CLIENT_VERSION
                and tuple(session.config.game.baseline_resolution) == BASELINE
                and time.monotonic() < end
                and isinstance(frame_id, str) and bool(frame_id)
                and o.get("frame_id") == frame_id
                and type(captured) in (int, float) and math.isfinite(captured)
                and o.get("observed_at_monotonic") == captured
                and 0 <= time.monotonic() - captured <= 8
                and isinstance(o.get("ocr"), list)
                and all(isinstance(item, dict) for item in o["ocr"])
                and _ocr_digest(o["ocr"]) == ocr_sha
                and o.get("account_in_use") is None
                and not any("另一个设备正在连接这个村庄" in str(item.get("text", ""))
                            for item in o["ocr"])
                and path.is_file())

    session.check_deadline()
    if not current():
        raise FlowError("Unknown blacksmith observation is stale, unsupported or account-blocked")
    frame_sha = _sha256(path)
    remaining = min(5, session.config.runtime.step_timeout_sec,
                    end - time.monotonic(), 8 - (time.monotonic() - captured))
    if remaining <= 0:
        raise FlowError("Unknown blacksmith observation freshness expired")
    foreground = foreground_package(session.capture.adb, session.capture.serial,
                                    timeout_sec=remaining)
    session.check_deadline()
    if foreground != session.config.game.package_name or not current() or _sha256(path) != frame_sha:
        raise FlowError("Unknown blacksmith observation foreground or source changed")


def _confirm_return_home(session, close: dict, end: float) -> list[dict]:
    """Require two new ordered, source-bound HOME frames after closing Equipment."""
    previous_frame = close.get("frame")
    previous_id = None
    previous_time = -math.inf
    evidence_frames = []
    for index in range(2):
        session.check_deadline()
        if time.monotonic() >= end:
            raise FlowError("Ore navigation HOME confirmation deadline exceeded")
        current = session.observe(f"ore-navigation-return-home-{index + 1}", purpose="full")
        frame = _frame(current, client_version=session.client_version,
                       baseline_resolution=session.config.game.baseline_resolution)
        if frame is None:
            raise FlowError("Ore navigation return frame is not a fresh HOME")
        path, _, frame_sha = frame
        observed_at = current.observations["observed_at_monotonic"]
        frame_id = current.observations["frame_id"]
        if (str(path) == previous_frame or frame_id == previous_id or
                observed_at <= previous_time):
            raise FlowError("Ore navigation HOME frames are reused or unordered")
        evidence = {"frame": str(path), "frame_sha256": frame_sha,
                    "frame_id": frame_id,
                    "ocr_sha256": _ocr_digest(current.observations["ocr"]),
                    "observed_at_monotonic": observed_at}
        _guard_current(session, current, evidence, end)
        evidence_frames.append(evidence)
        previous_frame, previous_id, previous_time = str(path), frame_id, observed_at
    return evidence_frames


def read_current_ore_inventory(session, *, timeout_sec: float = 90) -> dict:
    """From a fresh HOME/selection, read stable ores and close to confirmed HOME.

    This function does not claim event rewards or establish a before/after delta.
    Unknown screens and unrecognized selected buildings are never clicked.
    """
    if type(timeout_sec) not in (int, float) or not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 180:
        raise ValueError("Invalid ore navigation timeout")
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    if (session.client_version != CLIENT_VERSION or
            tuple(session.config.game.baseline_resolution) != BASELINE):
        raise FlowError("Unsupported blacksmith navigation version or resolution")
    front_end = min(time.monotonic() + 30, end)

    def remaining() -> float:
        session.check_deadline()
        left = end - time.monotonic()
        if left <= 0:
            raise FlowError("Ore navigation deadline exceeded")
        return left

    line_reader = lambda frame, roi: [asdict(item) for item in
                                       session.recognizer.recognize_region(frame, roi, single_line=True)]
    completed = False
    try:
        selected = None
        previous_observation = None
        for attempt in range(8):
            remaining()
            if time.monotonic() >= front_end:
                raise FlowError("Blacksmith front observation deadline exceeded")
            label = f"ore-navigation-current-{attempt + 1}"
            current = (session.observe(label, purpose="full") if previous_observation is None else
                       _observe_transition(session, label, previous_observation, front_end))
            session.check_deadline()
            if time.monotonic() >= front_end:
                raise FlowError("Blacksmith front observation deadline exceeded")
            if not _home(current):
                if current.scene != "unknown":
                    raise FlowError("Blacksmith front observation is not a current HOME")
                _guard_unknown_observation(session, current, front_end)
                previous_observation = current
                if attempt < 7:
                    next_poll = min(time.monotonic() + 1, front_end)
                    while time.monotonic() < next_poll:
                        session.check_deadline()
                        time.sleep(min(.1, max(0, next_poll - time.monotonic())))
                continue  # A transient unknown frame never authorizes a tap.
            selected = recognize_selected_blacksmith(
                current, client_version=session.client_version, line_reader=line_reader)
            if selected is not None:
                break
            if _looks_selected(current):
                raise FlowError("Current selected building is not a verified blacksmith Equipment menu")
            candidate = locate_blacksmith(current, client_version=session.client_version)
            if candidate is None:
                previous_observation = current
                if attempt < 7:
                    next_poll = min(time.monotonic() + 1, front_end)
                    while time.monotonic() < next_poll:
                        session.check_deadline()
                        time.sleep(min(.1, max(0, next_poll - time.monotonic())))
                continue  # The moving hero may uncover the front; never infer its point.
            _guard_current(session, current, candidate, front_end)
            session.tap(current, tuple(candidate["point"]), reason="Select current visible blacksmith front")
            for selected_attempt in range(3):
                remaining()
                current = _observe_transition(session, f"ore-navigation-selected-{selected_attempt + 1}",
                                              current, end)
                if _wrong_selected_building(current):
                    raise FlowError("Selected a different building, not the verified blacksmith")
                selected = recognize_selected_blacksmith(
                    current, client_version=session.client_version, line_reader=line_reader)
                if selected is not None:
                    break
            if selected is None:
                raise FlowError("Selected building was not verified across three fresh captures")
            break
        if selected is None:
            raise FlowError("No unique visible blacksmith front across eight current HOME captures")
        _guard_current(session, current, selected, end)
        session.tap(current, tuple(selected["point"]), reason="Open verified blacksmith Equipment menu")
        first = None
        for ore_attempt in range(3):
            remaining()
            first_snapshot = _observe_transition(session, f"ore-navigation-first-{ore_attempt + 1}",
                                                 current, end)
            current = first_snapshot
            first = read_ore_inventory_snapshot(first_snapshot, client_version=session.client_version)
            if first is not None:
                if first_snapshot.scene != "popup":
                    raise FlowError("Ore inventory evidence is not a popup")
                break
        if first is None:
            raise FlowError("First ore inventory frame was not verified across three fresh captures")
        remaining()
        second_snapshot = session.observe("ore-navigation-second", purpose="full")
        second = read_ore_inventory_snapshot(second_snapshot, client_version=session.client_version)
        stable = verify_stable_ore_inventory(first, second) if second is not None else None
        if stable is None or second_snapshot.scene != "popup":
            raise FlowError("Ore inventory is not stable across fresh captures")
        close = close_ore_inventory(session, second_snapshot, timeout_sec=min(15, remaining()))
        home_evidence = _confirm_return_home(session, close, end)
        completed = True
        return {"status": "stable_inventory_read", "inventory": stable,
                "selected": selected, "close": close, "returned_home": close["status"] == "home",
                "home_evidence": home_evidence,
                "account_verified": False, "reward_inventory_verified": False,
                "formal_success": False}
    except StopRequested:
        raise
    except Exception as exc:
        if not completed and end - time.monotonic() > 1:
            current = getattr(session, "last_snapshot", None)
            if (current is not None and current.scene == "popup" and
                    isinstance(current.observations.get("ore_inventory"), dict)):
                try:
                    close_ore_inventory(session, current,
                                        timeout_sec=min(15, end - time.monotonic()))
                except Exception as close_exc:
                    raise FlowError(f"Ore read failed; safe close also failed: {close_exc}") from exc
        raise
