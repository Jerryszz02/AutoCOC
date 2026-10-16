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
    "list": ("奖励清单", (145, 235, 260, 275)),
    "paid": ("购买活动令牌", (850, 95, 1145, 165)),
}
_POSITION_ANCHORS = {
    # These upper-track milestones bind the observed horizontal page position.
    # An empty generic lower pedestal alone cannot identify a claimed reward.
    "upper_medal_amount": ("600", (400, 210, 475, 250)),
    "upper_medal": ("裂隙奖章", (390, 240, 485, 280)),
    "upper_ore_amount": ("95", (710, 210, 780, 250)),
    "upper_ore": ("璀璨矿石", (690, 240, 790, 280)),
}
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
    position_anchors = {}
    for name, (label, roi) in _POSITION_ANCHORS.items():
        matches = _exact(ocr, label, roi)
        if len(matches) == 1:
            position_anchors[name] = {"text": label, "bbox": list(matches[0]["bbox"]),
                                      "confidence": matches[0]["confidence"]}
    position_verified = len(position_anchors) == len(_POSITION_ANCHORS)
    if position_verified:
        anchors.update(position_anchors)
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
        labels = _exact(ocr, spec["label"], spec["text_roi"])
        amounts = _exact(ocr, spec["amount"], spec["text_roi"])
        buttons = _exact(ocr, "领取", spec["claim_roi"])
        green = _green_fraction(image, spec["button_roi"])
        empty_score = _template_score(image, name + "_empty", spec["empty_roi"])
        status = "unknown"
        button = None
        if position_verified and len(labels) == len(amounts) == len(buttons) == 1 and green >= .35 and empty_score < .9:
            bbox = list(buttons[0]["bbox"])
            button = {"bbox": bbox, "point": [(bbox[0] + bbox[2]) // 2,
                                                 (bbox[1] + bbox[3]) // 2],
                      "confidence": buttons[0]["confidence"]}
            status = "claimable"
        elif position_verified and not labels and not amounts and not buttons and green <= .05 and empty_score >= .97:
            status = "no_opportunity"
        targets[name] = {"status": status, "label": spec["label"],
                         "amount": int(spec["amount"]), "button": button,
                         "green_fraction": green, "empty_template_score": empty_score}
    close_box = list(_VISUAL_ROIS["close"])
    return {"surface": "event_reward_track", "client_version": client_version,
            "frame": str(frame), "frame_sha256": _sha256(frame),
            "anchors": anchors, "position_verified": position_verified,
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
            or result["button"] != observed.get("button")):
        raise FlowError("Event reward current page failed revalidation")
    before = observed.get("targets", {}).get(target, {})
    now = result["targets"][target]
    if before.get("status") != now["status"] or before.get("button") != now["button"]:
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
    for index in range(3):
        current = _bounded_observe(session, f"event-reward-panel-{index + 1}", end)
        if (current.scene == "popup" and isinstance(current.observations.get("event_panel"), dict)
                and len(_exact(current.observations.get("ocr", []), "活动", (590, 20, 690, 90))) == 1
                and len(_exact(current.observations.get("ocr", []), "今日活动", (195, 85, 310, 145))) == 1
                and len(_exact(current.observations.get("ocr", []), "裂隙之乱", (205, 335, 315, 395))) == 1):
            panel = current
            break
    if panel is None:
        raise FlowError("Current event card was not confirmed; no entry tap issued")
    entries = [(label, item) for label in ("领取", "打开")
               for item in _exact(panel.observations.get("ocr", []), label, (850, 530, 1055, 625))]
    if len(entries) != 1:
        raise FlowError("Current reward-track entry is missing or ambiguous")
    entry_label, entry = entries[0]
    bbox = entry["bbox"]
    panel_hash = _sha256(panel.screenshot_path)
    _require_foreground(session)
    session.check_deadline()
    if time.monotonic() >= end or _sha256(panel.screenshot_path) != panel_hash:
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
            receipt["status"] = "no_opportunity"
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
