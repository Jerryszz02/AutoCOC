"""Current-client event reward track and one explicitly enabled free claim.

The track's milestone checkmarks precede claiming.  Only a stable empty
reward pedestal is a UI acknowledgement; ore inventory is not observable here.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np

from .device import foreground_package
from .errors import FlowError, StopRequested


CLIENT_VERSION = "18.647.2"
_BASELINE = (1280, 720)
_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates"
_PAGE_ANCHORS = {
    "title": ("裂隙之乱", (350, 20, 500, 90)),
    "paid": ("购买活动令牌", (850, 95, 1145, 165)),
}
# The 600/95 centers come from the reviewed original fixture. The 25 center
# is bound to the settled post-swipe raw frame, not extrapolated from spacing.
_UPPER_MILESTONES = (
    ("600", "裂隙奖章", 439),
    ("95", "璀璨矿石", 746),
    ("25", "星辉矿石", 1052),
)
_UPPER_AMOUNT_ROI = (175, 205, 1115, 250)
_UPPER_LABEL_ROI = (175, 240, 1115, 285)
_LIST_ICON_ROI = (178, 197, 237, 242)
_TARGET_VIEWPORT = (145, 420, 1145, 670)
_TARGETS = {
    "shiny": {"label": "闪亮矿石", "amount": "1250",
              "text_roi": (670, 455, 810, 530), "claim_roi": (670, 585, 815, 655),
              "button_roi": (675, 580, 800, 660),
              "empty_roi": (682, 526, 785, 640)},
    "glowy": {"label": "璀璨矿石", "amount": "115",
              "text_roi": (990, 455, 1100, 530), "claim_roi": (975, 585, 1125, 655),
              "button_roi": (970, 580, 1125, 660),
              "empty_roi": (992, 526, 1095, 640)},
}
_VISUAL_ROIS = {
    "close": (1092, 26, 1145, 82),
    "track_edge": (132, 172, 145, 683),
    "track_bottom": (138, 682, 1148, 697),
}


class EventPanelEntryUnconfirmed(FlowError):
    """The event list was seen, but no reward-track entry input was attempted."""


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _box_in(box: object, roi: tuple[int, int, int, int]) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
        return False
    return (roi[0] <= box[0] < box[2] <= roi[2]
            and roi[1] <= box[1] < box[3] <= roi[3])


def _exact(ocr: list[dict], label: str, roi: tuple[int, int, int, int]) -> list[dict]:
    normalized = re.sub(r"\s+", "", label)
    return [item for item in ocr if isinstance(item, dict)
            and re.sub(r"\s+", "", str(item.get("text", ""))) == normalized
            and type(item.get("confidence")) in (int, float)
            and math.isfinite(item["confidence"]) and .95 <= item["confidence"] <= 1
            and _box_in(item.get("bbox"), roi)]


def _crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]


def _template_score(image: np.ndarray, name: str, roi: tuple[int, int, int, int]) -> float:
    template = cv2.imread(str(_TEMPLATES / f"event_reward_{name}_18_647_2.png"))
    crop = _crop(image, roi)
    if template is None or template.shape != crop.shape:
        return 0.0
    score = float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score if math.isfinite(score) and 0 <= score <= 1 else 0.0


def _green_fraction(image: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    hsv = cv2.cvtColor(_crop(image, roi), cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85)
                         & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 90)))


def _center_x(box: list[int] | tuple[int, ...]) -> float:
    return (box[0] + box[2]) / 2


def _upper_position(ocr: list[dict]) -> tuple[int | None, list[dict]]:
    """Bind one translation from two independent upper amount+label pairs."""
    pairs = []
    for index, (amount, label, canonical_center) in enumerate(_UPPER_MILESTONES):
        numbers = _exact(ocr, amount, _UPPER_AMOUNT_ROI)
        if len(numbers) != 1:
            continue
        center = _center_x(numbers[0]["bbox"])
        names = [item for item in _exact(ocr, label, _UPPER_LABEL_ROI)
                 if abs(_center_x(item["bbox"]) - center) <= 22]
        if len(names) != 1:
            continue
        pairs.append({"index": index, "amount": amount, "label": label,
                      "amount_bbox": list(numbers[0]["bbox"]),
                      "label_bbox": list(names[0]["bbox"]),
                      "canonical_center_x": canonical_center,
                      "observed_center_x": center,
                      "offset_px": center - canonical_center})
    if (len(pairs) < 2
            or not any(pairs[i + 1]["index"] == pairs[i]["index"] + 1
                       for i in range(len(pairs) - 1))):
        return None, pairs
    offsets = [item["offset_px"] for item in pairs]
    if (max(offsets) - min(offsets) > 6
            or any(pairs[i + 1]["observed_center_x"] - pairs[i]["observed_center_x"] < 295 * (pairs[i + 1]["index"] - pairs[i]["index"])
                   for i in range(len(pairs) - 1))):
        return None, pairs
    translation = round(sum(offsets) / len(offsets))
    if abs(translation) <= 3:
        translation = 0
    if not -650 <= translation <= 330:
        return None, pairs
    if pairs[0]["index"] == 1:
        # The 600 label can be merged with the fixed list button after a
        # scroll. Its independent number still corroborates 95/25 location.
        leading = _exact(ocr, "600", _UPPER_AMOUNT_ROI)
        if (len(leading) != 1
                or abs(_center_x(leading[0]["bbox"]) - (439 + translation)) > 8):
            return None, pairs
    return translation, pairs


def _translated_roi(roi: tuple[int, int, int, int], offset: int) -> tuple[int, int, int, int] | None:
    shifted = (roi[0] + offset, roi[1], roi[2] + offset, roi[3])
    if not _box_in(shifted, _TARGET_VIEWPORT):
        return None
    return shifted


def _translated_empty_score(image: np.ndarray, target: str,
                            roi: tuple[int, int, int, int]) -> float:
    # A tiny search tolerates OCR center jitter; no generic pedestal can bind
    # target identity without two independently named upper milestones.
    scores = []
    for correction in range(-3, 4):
        trial = _translated_roi(roi, correction)
        if trial is not None:
            scores.append(_template_score(image, target + "_empty", trial))
    return max(scores, default=0.0)


def _account_claim(ocr: object) -> bool:
    return isinstance(ocr, list) and any(
        "另一个设备正在连接这个村庄" in re.sub(r"\s+", "", str(item.get("text", "")))
        for item in ocr if isinstance(item, dict))


def observe_reward_track(frame_path: str | Path, ocr: list[dict], *,
                         client_version: str,
                         baseline_resolution: tuple[int, int] = _BASELINE) -> dict | None:
    """Return a versioned complete page or None; target statuses fail closed.

    OCR boxes must already use the shared 1280x720 baseline. A complete page
    does not authorize any claim on its own.
    """
    if client_version != CLIENT_VERSION or tuple(baseline_resolution) != _BASELINE:
        return None
    if not isinstance(ocr, list) or any(not isinstance(item, dict) for item in ocr):
        return None
    if _account_claim(ocr):
        return None
    frame = Path(frame_path)
    image = cv2.imread(str(frame))
    if image is None or (image.shape[1], image.shape[0]) not in {_BASELINE, (2560, 1440)}:
        return None
    image = cv2.resize(image, _BASELINE, interpolation=cv2.INTER_AREA)
    anchors = {}
    for name, (label, roi) in _PAGE_ANCHORS.items():
        matches = _exact(ocr, label, roi)
        if len(matches) != 1:
            return None
        anchors[name] = {"text": label, "bbox": list(matches[0]["bbox"]),
                         "confidence": matches[0]["confidence"]}
    list_matches = _exact(ocr, "奖励清单", (145, 235, 260, 275))
    if len(list_matches) > 1:
        return None
    if list_matches:
        anchors["list"] = {"method": "independent_exact_ocr", "text": "奖励清单",
                           "bbox": list(list_matches[0]["bbox"]),
                           "confidence": list_matches[0]["confidence"]}
    else:
        # This icon remains fixed while reward labels slide behind the button.
        # Without its versioned asset the merged-label page stays unknown.
        icon_score = _template_score(image, "list_icon", _LIST_ICON_ROI)
        if icon_score < .97:
            return None
        icon_asset = _TEMPLATES / "event_reward_list_icon_18_647_2.png"
        anchors["list"] = {"method": "fixed_icon_template", "bbox": list(_LIST_ICON_ROI),
                           "confidence": icon_score, "asset_sha256": _sha256(icon_asset)}
    position_offset, position_basis = _upper_position(ocr)
    position_verified = position_offset is not None
    visual = {name: _template_score(image, name, roi)
              for name, roi in _VISUAL_ROIS.items()}
    if any(score < .97 for score in visual.values()):
        return None
    paid_hsv = cv2.cvtColor(_crop(image, (870, 100, 1140, 160)), cv2.COLOR_BGR2HSV)
    paid_yellow = float(np.mean((paid_hsv[:, :, 0] >= 8) & (paid_hsv[:, :, 0] <= 35)
                                & (paid_hsv[:, :, 1] >= 90) & (paid_hsv[:, :, 2] >= 120)))
    if paid_yellow < .6:
        return None
    targets = {}
    for name, spec in _TARGETS.items():
        rois = ({key: _translated_roi(spec[key], position_offset) for key in
                 ("text_roi", "claim_roi", "button_roi", "empty_roi")}
                if position_verified else {})
        target_visible = position_verified and all(roi is not None for roi in rois.values())
        labels = _exact(ocr, spec["label"], rois["text_roi"]) if target_visible else []
        amounts = _exact(ocr, spec["amount"], rois["text_roi"]) if target_visible else []
        buttons = _exact(ocr, "领取", rois["claim_roi"]) if target_visible else []
        green = _green_fraction(image, rois["button_roi"]) if target_visible else 0.0
        empty_score = (_translated_empty_score(image, name, rois["empty_roi"])
                       if target_visible else 0.0)
        status = "unknown"
        button = None
        if target_visible and len(labels) == len(amounts) == len(buttons) == 1 and green >= .35 and empty_score < .9:
            bbox = list(buttons[0]["bbox"])
            button = {"bbox": bbox, "point": [(bbox[0] + bbox[2]) // 2,
                                                 (bbox[1] + bbox[3]) // 2],
                      "confidence": buttons[0]["confidence"]}
            status = "claimable"
        elif target_visible and not labels and not amounts and not buttons and green <= .05 and empty_score >= .97:
            status = "no_opportunity"
        targets[name] = {"status": status, "label": spec["label"],
                         "amount": int(spec["amount"]), "button": button,
                         "green_fraction": green, "empty_template_score": empty_score,
                         "visible": bool(target_visible), "rois": rois}
    close_box = list(_VISUAL_ROIS["close"])
    return {"surface": "event_reward_track", "client_version": client_version,
            "frame": str(frame), "frame_sha256": _sha256(frame),
            "anchors": anchors, "position_verified": position_verified,
            "position_offset_px": position_offset, "position_basis": position_basis,
            "visual": {**visual, "paid_yellow_fraction": paid_yellow},
            "targets": targets,
            "button": {"name": "close_event_rewards", "text": "X（图像）",
                       "bbox": close_box, "point": [(close_box[0] + close_box[2]) // 2,
                                                      (close_box[1] + close_box[3]) // 2],
                       "confidence": min(visual.values())}}


def _require_current(session, snapshot, target: str, end: float) -> dict:
    session.check_deadline()
    if (time.monotonic() >= end or session.client_version != CLIENT_VERSION
            or tuple(session.config.game.baseline_resolution) != _BASELINE):
        raise FlowError("Event reward deadline or client version changed")
    if snapshot is not session.last_snapshot or snapshot.scene != "popup" or snapshot.confidence < .95:
        raise FlowError("Event reward is not the current recognized popup")
    if snapshot.observations.get("account_in_use") is not None or _account_claim(snapshot.observations.get("ocr")):
        raise FlowError("Account ownership warning blocks event reward input")
    captured = snapshot.observations.get("observed_at_monotonic")
    requested = snapshot.observations.get("requested_at_monotonic")
    frame_id = snapshot.observations.get("frame_id")
    if (type(captured) not in (int, float) or not math.isfinite(captured)
            or type(requested) not in (int, float) or not math.isfinite(requested)
            or not isinstance(frame_id, str) or not frame_id
            or not 0 <= time.monotonic() - captured <= 8):
        raise FlowError("Event reward observation is stale")
    observed = snapshot.observations.get("event_reward_track")
    if not isinstance(observed, dict) or observed.get("frame") != str(snapshot.screenshot_path):
        raise FlowError("Event reward has no bound page evidence")
    if observed.get("frame_sha256") != _sha256(snapshot.screenshot_path):
        raise FlowError("Event reward frame changed")
    result = observe_reward_track(snapshot.screenshot_path, snapshot.observations.get("ocr", []),
                                  client_version=session.client_version)
    if (result is None or result["frame_sha256"] != observed["frame_sha256"]
            or result["button"] != observed.get("button")
            or result["anchors"] != observed.get("anchors")):
        raise FlowError("Event reward current page failed revalidation")
    before = observed.get("targets", {}).get(target, {})
    now = result["targets"][target]
    if (before.get("status") != now["status"] or before.get("button") != now["button"]
            or observed.get("position_offset_px") != result.get("position_offset_px")):
        raise FlowError("Event reward target evidence changed")
    return now


def _require_foreground(session) -> None:
    capture = session.capture
    if foreground_package(capture.adb, capture.serial,
                          timeout_sec=min(5, session.config.runtime.step_timeout_sec)) != session.config.game.package_name:
        raise FlowError("Event reward game is not foreground")


def _bounded_observe(session, label: str, end: float):
    session.check_deadline()
    if time.monotonic() >= end:
        raise FlowError("Event reward navigation deadline expired")
    _require_foreground(session)
    observed = session.observe(label, purpose="full")
    session.check_deadline()
    if time.monotonic() >= end:
        raise FlowError("Event reward navigation deadline expired after observation")
    if (observed.scene == "account_in_use" or observed.observations.get("account_in_use") is not None
            or _account_claim(observed.observations.get("ocr"))):
        raise FlowError("Account is in use during event reward navigation")
    return observed


def _village(snapshot) -> bool:
    controls = {button.get("name") for button in snapshot.observations.get("buttons", [])
                if isinstance(button, dict)}
    return (snapshot.scene == "village" and snapshot.confidence >= .8
            and snapshot.observations.get("village_type") == "home"
            and snapshot.observations.get("resource_source") == "village_inventory"
            and {"attack", "shop"} <= controls)


def _panel_close_matches(ocr: list[dict]) -> list[dict]:
    # The existing 18.647.2 panel recognizer accepts a .90 X; keep that gate.
    return [item for item in ocr if isinstance(item, dict)
            and re.sub(r"\s+", "", str(item.get("text", ""))) == "X"
            and type(item.get("confidence")) in (int, float)
            and math.isfinite(item["confidence"]) and .9 <= item["confidence"] <= 1
            and _box_in(item.get("bbox"), (1015, 20, 1090, 95))]


def _event_panel_anchors(snapshot) -> list[dict] | None:
    """Bind the recognized event list to its current, unique OCR anchors."""
    if snapshot.scene != "popup" or snapshot.confidence < .95:
        return None
    panel = snapshot.observations.get("event_panel")
    ocr = snapshot.observations.get("ocr")
    if not isinstance(panel, dict) or not isinstance(ocr, list):
        return None
    for key, label, roi in (("title", "活动", (590, 20, 690, 90)),
                            ("section", "今日活动", (195, 85, 310, 145))):
        matches = _exact(ocr, label, roi)
        if len(matches) != 1 or panel.get(key) != matches[0]:
            return None
    closes = _panel_close_matches(ocr)
    if len(closes) != 1 or panel.get("close") != closes[0]:
        return None
    return ocr


def _event_card_entry(ocr: list[dict]) -> tuple[str, dict, str] | None:
    """Recognize one of two observed 18.647.2 card layouts, never infer a point."""
    if len(_exact(ocr, "裂隙之乱", (205, 120, 350, 410))) != 1:
        return None
    entries = []
    old_title = _exact(ocr, "裂隙之乱", (205, 335, 315, 395))
    old_buttons = [(label, item) for label in ("领取", "打开")
                   for item in _exact(ocr, label, (850, 530, 1055, 625))]
    if len(old_title) == len(old_buttons) == 1:
        entries.append((*old_buttons[0], "lower"))

    title = _exact(ocr, "裂隙之乱", (205, 125, 325, 185))
    progress = [item for item in ocr if isinstance(item, dict)
                and type(item.get("confidence")) in (int, float)
                and math.isfinite(item["confidence"]) and .95 <= item["confidence"] <= 1
                and _box_in(item.get("bbox"), (850, 250, 1060, 315))
                and re.fullmatch(r"\d{1,7}/\d{1,7}", re.sub(r"\s+", "", str(item.get("text", ""))))]
    upper_buttons = [(label, item) for label in ("领取", "打开")
                     for item in _exact(ocr, label, (875, 330, 1045, 405))]
    if len(title) == len(progress) == len(upper_buttons) == 1:
        label, button = upper_buttons[0]
        title_box, progress_box, button_box = (title[0]["bbox"], progress[0]["bbox"], button["bbox"])
        current, total = map(int, re.sub(r"\s+", "", progress[0]["text"]).split("/"))
        if (total > 0 and (label != "领取" or current >= total)
                and title_box[2] < progress_box[0]
                and 0 < progress_box[1] - title_box[3] <= 140
                and 0 < button_box[1] - progress_box[3] <= 80
                and abs(_center_x(button_box) - _center_x(progress_box)) <= 60):
            entries.append((label, button, "upper"))
    return entries[0] if len(entries) == 1 else None


def open_reward_track(session, *, timeout_sec: float = 20) -> dict:
    """Navigate home -> event card -> track, never claim a reward."""
    if not 0 < timeout_sec <= 30:
        raise ValueError("Invalid event reward navigation timeout")
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    if (session.client_version != CLIENT_VERSION
            or tuple(session.config.game.baseline_resolution) != _BASELINE):
        raise FlowError("Event reward navigation requires the sampled client version")
    home = _bounded_observe(session, "event-reward-home", end)
    if not _village(home):
        raise FlowError("Event reward navigation requires a confirmed home village")
    from .objectives import load_adapter

    adapter = load_adapter(_TEMPLATES.parent / "catalogs" / "rift_event_18_647_2.adapter.toml")
    if adapter.client_versions != (CLIENT_VERSION,):
        raise FlowError("Event reward entry adapter changed")
    home_hash = _sha256(home.screenshot_path)
    _require_foreground(session)
    if time.monotonic() >= end or _sha256(home.screenshot_path) != home_hash:
        raise FlowError("Event reward home source or deadline changed")
    session.click_template(home, adapter.entry_template, roi=adapter.entry_roi)
    panel = None
    confirmed_list = False
    for index in range(3):
        current = _bounded_observe(session, f"event-reward-panel-{index + 1}", end)
        ocr = _event_panel_anchors(current)
        if ocr is not None:
            confirmed_list = True
        if (ocr is not None and
                (_exact(ocr, "裂隙之乱", (205, 335, 315, 395))
                 or _exact(ocr, "裂隙之乱", (205, 125, 325, 185)))):
            panel = current
            break
    if panel is None:
        error = "Current event card was not confirmed; no entry tap issued"
        if confirmed_list:
            raise EventPanelEntryUnconfirmed(error)
        raise FlowError(error)
    matched = _event_card_entry(panel.observations["ocr"])
    if matched is None:
        raise EventPanelEntryUnconfirmed("Current reward-track entry is missing or ambiguous")
    entry_label, entry, layout = matched
    bbox = entry["bbox"]
    panel_hash = _sha256(panel.screenshot_path)
    _require_foreground(session)
    session.check_deadline()
    if (time.monotonic() >= end or panel is not session.last_snapshot
            or _event_panel_anchors(panel) is None
            or _event_card_entry(panel.observations["ocr"]) != (entry_label, entry, layout)
            or _sha256(panel.screenshot_path) != panel_hash):
        raise FlowError("Event reward entry source or deadline changed")
    session.tap(panel, ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                reason="Open confirmed current event reward track")
    for index in range(4):
        current = _bounded_observe(session, f"event-reward-track-{index + 1}", end)
        if current.scene == "popup" and isinstance(current.observations.get("event_reward_track"), dict):
            _require_current(session, current, "shiny", end)
            return {"status": "opened", "frame": str(current.screenshot_path),
                    "frame_sha256": _sha256(current.screenshot_path), "entry_frame": str(panel.screenshot_path),
                    "entry_label": entry_label, "position_verified": current.observations["event_reward_track"]["position_verified"],
                    "claim_attempted": False, "reward_inventory_verified": False, "formal_success": False}
    raise FlowError("Reward track did not become a confirmed current page")


def close_unentered_event_panel(session, *, timeout_sec: float = 20) -> dict:
    """After a pre-entry refusal, close one fresh verified list X and confirm HOME twice."""
    if not 0 < timeout_sec <= 30:
        raise ValueError("Invalid event panel close timeout")
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    if (session.client_version != CLIENT_VERSION
            or tuple(session.config.game.baseline_resolution) != _BASELINE):
        raise FlowError("Event panel close requires the sampled client version")
    prior = getattr(session, "last_snapshot", None)
    panel = _bounded_observe(session, "event-panel-safe-close-before", end)
    ocr = _event_panel_anchors(panel)
    if ocr is None or panel is not session.last_snapshot or prior is None or not _fresh_pair(prior, panel):
        raise FlowError("Event panel close has no current verified list")
    close = _panel_close_matches(ocr)[0]
    panel_hash = _sha256(panel.screenshot_path)
    captured = panel.observations.get("observed_at_monotonic")
    requested = panel.observations.get("requested_at_monotonic")
    frame_id = panel.observations.get("frame_id")
    if (type(captured) not in (int, float) or not math.isfinite(captured)
            or type(requested) not in (int, float) or not math.isfinite(requested)
            or not isinstance(frame_id, str) or not frame_id.strip()
            or not 0 <= time.monotonic() - captured <= 8):
        raise FlowError("Event panel close frame is stale")
    _require_foreground(session)
    session.check_deadline()
    if (time.monotonic() >= end or panel is not session.last_snapshot
            or _event_panel_anchors(panel) is None
            or not 0 <= time.monotonic() - captured <= 8
            or _sha256(panel.screenshot_path) != panel_hash):
        raise FlowError("Event panel close source or deadline changed")
    box = close["bbox"]
    session.tap(panel, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                reason="Close verified unentered event list")
    homes = []
    previous = panel
    for index in range(3):
        current = _bounded_observe(session, f"event-panel-safe-close-home-{index + 1}", end)
        home_captured = current.observations.get("observed_at_monotonic")
        if (current is not session.last_snapshot or not _fresh_pair(previous, current)
                or type(home_captured) not in (int, float) or not math.isfinite(home_captured)
                or not 0 <= time.monotonic() - home_captured <= 8):
            raise FlowError("Event panel close HOME frame is not new")
        if not _village(current):
            if homes:
                raise FlowError("Event panel close lost HOME after first confirmation")
            previous = current
            continue
        homes.append({"frame": str(current.screenshot_path), "frame_id": current.observations["frame_id"],
                      "frame_sha256": _sha256(current.screenshot_path)})
        if len(homes) == 2:
            return {"status": "home", "home_evidence": homes,
                    "frame": str(current.screenshot_path), "formal_success": False}
        previous = current
    raise FlowError("Event panel close did not confirm two new HOME frames")


def close_reward_track(session, *, timeout_sec: float = 20) -> dict:
    """Close only recognized X controls and require a fresh confirmed home."""
    if not 0 < timeout_sec <= 30:
        raise ValueError("Invalid event reward navigation timeout")
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    if (session.client_version != CLIENT_VERSION
            or tuple(session.config.game.baseline_resolution) != _BASELINE):
        raise FlowError("Event reward close requires the sampled client version")
    current = _bounded_observe(session, "event-reward-close-before", end)
    _require_current(session, current, "shiny", end)
    button = current.observations["event_reward_track"]["button"]
    _require_foreground(session)
    _require_current(session, current, "shiny", end)
    session.tap(current, tuple(button["point"]), reason="Close recognized event reward track")
    panel = None
    for index in range(3):
        current = _bounded_observe(session, f"event-reward-close-after-{index + 1}", end)
        if _village(current):
            return {"status": "home", "frame": str(current.screenshot_path),
                    "claim_attempted": False, "reward_inventory_verified": False, "formal_success": False}
        if current.scene == "popup" and isinstance(current.observations.get("event_panel"), dict):
            panel = current
            break
    if panel is None:
        raise FlowError("Event reward close did not reveal a confirmed home or event panel")
    closes = [item for item in panel.observations.get("ocr", [])
              if re.sub(r"\s+", "", str(item.get("text", ""))) == "X"
              and type(item.get("confidence")) in (int, float)
              and math.isfinite(item["confidence"]) and .9 <= item["confidence"] <= 1
              and _box_in(item.get("bbox"), (1015, 20, 1090, 95))]
    if len(closes) != 1:
        raise FlowError("Event panel close X is not uniquely recognized")
    bbox = closes[0]["bbox"]
    panel_hash = _sha256(panel.screenshot_path)
    _require_foreground(session)
    if time.monotonic() >= end or _sha256(panel.screenshot_path) != panel_hash:
        raise FlowError("Event panel close source or deadline changed")
    session.tap(panel, ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                reason="Close recognized event panel after reward track")
    for index in range(3):
        current = _bounded_observe(session, f"event-reward-return-home-{index + 1}", end)
        if _village(current):
            return {"status": "home", "frame": str(current.screenshot_path),
                    "claim_attempted": False, "reward_inventory_verified": False, "formal_success": False}
    raise FlowError("Event reward close did not return to a confirmed home")


def _start_position(ocr: object) -> tuple[float, float] | None:
    """Recognize the one observed 100/800 starting viewport, never guess scroll direction."""
    if not isinstance(ocr, list):
        return None
    centers = []
    for amount, label in (("100", "璀璨矿石"), ("800", "裂隙奖章")):
        numbers = _exact(ocr, amount, _UPPER_AMOUNT_ROI)
        if len(numbers) != 1:
            return None
        center = _center_x(numbers[0]["bbox"])
        labels = [item for item in _exact(ocr, label, _UPPER_LABEL_ROI)
                  if abs(_center_x(item["bbox"]) - center) <= 22]
        if len(labels) != 1:
            return None
        centers.append(center)
    # The third slot is partly covered by the trader; only the two complete
    # amount/name pairs may establish this exact starting viewport.
    if (not 425 <= centers[0] <= 450
            or not 730 <= centers[1] <= 755
            or not 295 <= centers[1] - centers[0] <= 320):
        return None
    return tuple(centers)


def _fresh_pair(first, second) -> bool:
    one, two = first.observations, second.observations
    first_time, second_time = one.get("requested_at_monotonic"), two.get("requested_at_monotonic")
    first_id, second_id = one.get("frame_id"), two.get("frame_id")
    return (type(first_time) in (int, float) and type(second_time) in (int, float)
            and math.isfinite(first_time) and math.isfinite(second_time)
            and first_time > 0 and second_time > first_time
            and isinstance(first_id, str) and isinstance(second_id, str)
            and bool(first_id.strip()) and bool(second_id.strip())
            and first_id != second_id
            and isinstance(first.screenshot_path, Path)
            and isinstance(second.screenshot_path, Path)
            and first.screenshot_path != second.screenshot_path)


def _stable_position(first: dict, second: dict, target: str) -> bool:
    """Allow one-pixel OCR jitter without relaxing same-frame target evidence."""
    a, b = first.get("position_offset_px"), second.get("position_offset_px")
    return (first.get("position_verified") is True and second.get("position_verified") is True
            and type(a) is int and type(b) is int and abs(a - b) <= 1
            and first.get("targets", {}).get(target, {}).get("visible") is True
            and second.get("targets", {}).get(target, {}).get("visible") is True)


def locate_reward_track_position(session, target: str, *, enabled: bool = False,
                                 task_id: str | None = None,
                                 timeout_sec: float = 30) -> dict:
    """At most one guarded `prev` swipe per task from the 100/800 viewport.

    This is a separate opt-in navigation step. It never claims or closes.
    Callers reserve their own close and inventory-read time before invoking it.
    """
    if not enabled:
        raise FlowError("Event track scrolling requires explicit opt-in")
    if target not in _TARGETS or not 0 < timeout_sec <= 30:
        raise ValueError("Unsupported target or navigation timeout")
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
        raise ValueError("Position navigation task id must be nonempty")
    attempt_key = (task_id or target, target)
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    receipt = {"kind": "event_track_position_navigation", "target": target,
               "task_id": task_id,
               "status": "unknown", "input_attempted": False, "input_sent": False,
               "claim_attempted": False, "formal_success": False, "frames": [], "reasons": []}
    try:
        pair = []
        for index in (1, 2):
            frame = _bounded_observe(session, f"event-position-before-{index}", end)
            _require_current(session, frame, target, end)
            receipt["frames"].append({"path": str(frame.screenshot_path),
                                      "sha256": _sha256(frame.screenshot_path),
                                      "frame_id": frame.observations.get("frame_id"),
                                      "requested_at_monotonic": frame.observations.get("requested_at_monotonic"),
                                      "offset": frame.observations["event_reward_track"].get("position_offset_px")})
            pair.append(frame)
        first, second = pair
        if not _fresh_pair(first, second):
            raise FlowError("Position source frames were not independent and newer")
        a = first.observations["event_reward_track"]
        b = second.observations["event_reward_track"]
        if getattr(session, "_event_track_position_uncertain", False):
            receipt["reasons"].append("previous_navigation_transport_uncertain")
            return receipt
        if _stable_position(a, b, target):
            receipt["status"] = "positioned"
            receipt["offset_px"] = b["position_offset_px"]
            return receipt
        attempts = getattr(session, "_event_track_position_swipe_attempts", None)
        if attempts is None:
            attempts = set()
            session._event_track_position_swipe_attempts = attempts
        if attempt_key in attempts:
            receipt["reasons"].append("one_swipe_already_attempted_for_task")
            return receipt
        start_a = _start_position(first.observations.get("ocr"))
        start_b = _start_position(second.observations.get("ocr"))
        if (start_a is None or start_b is None
                or any(abs(x - y) > 4 for x, y in zip(start_a, start_b))):
            receipt["reasons"].append("starting_viewport_not_independently_bound")
            return receipt
        if end - time.monotonic() < 15:
            receipt["reasons"].append("insufficient_budget_for_one_swipe_and_after_frames")
            return receipt
        _require_foreground(session)
        _require_current(session, second, target, end)
        session.check_deadline()
        if time.monotonic() >= end:
            raise FlowError("Position navigation deadline expired before input")
        attempts.add(attempt_key)
        session.event("event_track_position_swipe_reserved", target=target,
                      frame=str(second.screenshot_path), frame_sha256=_sha256(second.screenshot_path),
                      start=[340, 385], finish=[920, 385], duration_ms=550)
        receipt["input_attempted"] = True
        try:
            session._input_attempt_generation += 1
            session.context.swipe(340, 385, 920, 385, 550)
            session.action_count += 1
            receipt["input_sent"] = True
            receipt["input_done_at_monotonic"] = time.monotonic()
        except Exception:
            session._event_track_position_uncertain = True
            raise
        finally:
            session.last_snapshot = None
        previous = None
        for index in range(4):
            current = _bounded_observe(session, f"event-position-after-{index + 1}", end)
            record = {"path": str(current.screenshot_path), "sha256": _sha256(current.screenshot_path),
                      "frame_id": current.observations.get("frame_id"),
                      "requested_at_monotonic": current.observations.get("requested_at_monotonic"),
                      "offset": None}
            receipt["frames"].append(record)
            if (current.scene != "popup"
                    or not isinstance(current.observations.get("event_reward_track"), dict)):
                previous = None
                continue
            _require_current(session, current, target, end)
            state = current.observations["event_reward_track"]
            record["offset"] = state.get("position_offset_px")
            if (previous is not None and _fresh_pair(previous, current)
                    and _stable_position(previous.observations["event_reward_track"], state, target)
                    and record["requested_at_monotonic"] > receipt["input_done_at_monotonic"]
                    and previous.observations["requested_at_monotonic"] > receipt["input_done_at_monotonic"]):
                receipt["status"] = "positioned"
                receipt["offset_px"] = state["position_offset_px"]
                return receipt
            previous = current
        receipt["reasons"].append("stable_target_position_not_established")
        return receipt
    except StopRequested:
        receipt["reasons"].append("StopRequested")
        raise
    except Exception as exc:
        receipt["reasons"].append(f"{type(exc).__name__}: {exc}")
        return receipt
    finally:
        session.event("event_track_position_receipt", receipt=receipt)


def claim_one_free_reward(session, target: str, *, enabled: bool = False,
                          timeout_sec: float = 20) -> dict:
    """Opt-in one-tap claim; return only a UI receipt, never an ore receipt."""
    if not enabled:
        raise FlowError("Free event claiming requires explicit opt-in")
    if target not in _TARGETS or not 0 < timeout_sec <= 30:
        raise ValueError("Unsupported event reward target or timeout")
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    receipt = {"kind": "event_free_claim_ui", "target": target, "client_version": session.client_version,
               "status": "unknown", "input_attempted": False, "input_sent": False,
               "ui_claim_acknowledged": False,
               "reward_inventory_verified": False, "formal_success": False,
               "frames": [], "reasons": []}
    try:
        attempts = getattr(session, "_event_free_claim_attempts", None)
        if attempts is None:
            attempts = set()
            session._event_free_claim_attempts = attempts
        if target in attempts:
            receipt["reasons"].append("already_attempted_in_session")
            return receipt
        _require_foreground(session)
        before = session.observe("event-free-before", purpose="full")
        state = _require_current(session, before, target, end)
        receipt["frames"].append({"path": str(before.screenshot_path),
                                  "sha256": _sha256(before.screenshot_path), "status": state["status"],
                                  "frame_id": before.observations["frame_id"],
                                  "requested_at_monotonic": before.observations["requested_at_monotonic"]})
        if state["status"] == "no_opportunity":
            confirmed = _bounded_observe(session, "event-free-no-opportunity-second", end)
            second = _require_current(session, confirmed, target, end)
            receipt["frames"].append({"path": str(confirmed.screenshot_path),
                                      "sha256": _sha256(confirmed.screenshot_path),
                                      "status": second["status"],
                                      "frame_id": confirmed.observations["frame_id"],
                                      "requested_at_monotonic": confirmed.observations["requested_at_monotonic"]})
            if (_fresh_pair(before, confirmed) and second["status"] == "no_opportunity"
                    and _stable_position(before.observations["event_reward_track"],
                                         confirmed.observations["event_reward_track"], target)):
                receipt["status"] = "no_opportunity"
            else:
                receipt["reasons"].append("no_opportunity_not_stable_in_two_frames")
            return receipt
        if state["status"] != "claimable":
            receipt["reasons"].append("target_not_independently_claimable")
            return receipt
        _require_foreground(session)
        state = _require_current(session, before, target, end)
        session.check_deadline()
        if time.monotonic() >= end:
            raise FlowError("Event reward claim deadline expired before input")
        attempts.add(target)
        session.event("event_free_claim_attempt_reserved", target=target,
                      frame=str(before.screenshot_path), frame_sha256=_sha256(before.screenshot_path))
        receipt["input_attempted"] = True
        receipt["claim_point"] = state["button"]["point"]
        # A transport uncertainty is terminal. Never replay this tap.
        session.tap(before, tuple(state["button"]["point"]),
                    reason=f"Opt-in current free event reward: {target}")
        receipt["input_sent"] = True
        stable = 0
        last_request = before.observations["requested_at_monotonic"]
        seen_ids = {before.observations["frame_id"]}
        for index in range(4):
            session.check_deadline()
            if time.monotonic() >= end:
                break
            _require_foreground(session)
            after = session.observe(f"event-free-after-{index + 1}", purpose="full")
            state = _require_current(session, after, target, end)
            sha = _sha256(after.screenshot_path)
            request = after.observations["requested_at_monotonic"]
            frame_id = after.observations["frame_id"]
            if request <= last_request or frame_id in seen_ids:
                raise FlowError("Event reward post-frame capture is not independent and newer")
            last_request = request
            seen_ids.add(frame_id)
            receipt["frames"].append({"path": str(after.screenshot_path), "sha256": sha,
                                      "status": state["status"], "frame_id": frame_id,
                                      "requested_at_monotonic": request})
            if state["status"] == "no_opportunity":
                stable += 1
            else:
                stable = 0
            if stable >= 2:
                receipt["status"] = "ui_acknowledged"
                receipt["ui_claim_acknowledged"] = True
                return receipt
        receipt["reasons"].append("stable_target_ui_not_established")
        return receipt
    except StopRequested:
        receipt["reasons"].append("StopRequested")
        raise
    except Exception as exc:
        receipt["reasons"].append(f"{type(exc).__name__}: {exc}")
        return receipt
    finally:
        session.event("event_free_claim_receipt", receipt=receipt)
