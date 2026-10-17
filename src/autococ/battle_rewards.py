"""Observed free 18.647.2 battle chest controls after a claim settlement.

The chest receipt is separate from settlement loot.  The captured free skip
confirmation and the original fixed hammer path have independent proof gates.
Unknown pages, prices, transitions, or uncertain input results stop the chain.
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
from .errors import FlowError


CLIENT_VERSION = "18.647.2"
BASELINE = (1280, 720)
_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates"
_CHEST_ROI = (520, 260, 760, 490)
_BACKGROUND_ROI = (30, 220, 180, 360)
_HAMMER_ROIS = ((526, 584, 570, 619), (584, 584, 628, 619),
                (640, 584, 684, 619), (695, 584, 741, 619))
_CHAIN = (("ordinary", 4), ("rare", 3), ("rare", 2), ("rare", 1))
_SKIP_CONFIRM_ROI = (350, 185, 930, 530)
_CURRENCY_NAMES = {"金币": "gold", "圣水": "elixir", "黑油": "dark_elixir"}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _box_in(box: object, roi: tuple[int, int, int, int]) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value) for value in box)
            and roi[0] <= box[0] < box[2] <= roi[2]
            and roi[1] <= box[1] < box[3] <= roi[3])


def _exact(ocr: list[dict], label: str, roi: tuple[int, int, int, int]) -> list[dict]:
    normalized = re.sub(r"\s+", "", label)
    return [item for item in ocr if isinstance(item, dict)
            and re.sub(r"\s+", "", str(item.get("text", ""))) == normalized
            and type(item.get("confidence")) in (int, float)
            and math.isfinite(item["confidence"]) and .95 <= item["confidence"] <= 1
            and _box_in(item.get("bbox"), roi)]


def _blocked(ocr: list[dict]) -> bool:
    return any(any(word in re.sub(r"\s+", "", str(item.get("text", "")))
                   for word in ("另一个设备正在连接这个村庄", "购买", "支付", "价格", "充值", "£", "¥", "$"))
               for item in ocr if isinstance(item, dict))


def _confident_texts(ocr: list[dict]) -> set[str]:
    return {re.sub(r"\s+", "", str(item.get("text", ""))) for item in ocr
            if isinstance(item, dict) and type(item.get("confidence")) in (int, float)
            and math.isfinite(item["confidence"]) and item["confidence"] >= .95}


def _image(path: str | Path, version: str, baseline: tuple[int, int], ocr: object):
    if (version != CLIENT_VERSION or tuple(baseline) != BASELINE
            or not isinstance(ocr, list) or any(not isinstance(item, dict) for item in ocr)
            or _blocked(ocr)):
        return None
    image = cv2.imread(str(path))
    if image is None or (image.shape[1], image.shape[0]) not in {BASELINE, (2560, 1440)}:
        return None
    return cv2.resize(image, BASELINE, interpolation=cv2.INTER_AREA)


def _score(image: np.ndarray, name: str, roi: tuple[int, int, int, int]) -> float:
    template = cv2.imread(str(_TEMPLATES / f"{name}_18_647_2.png"))
    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]
    if template is None or crop.shape != template.shape:
        return 0.0
    score = float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score if math.isfinite(score) and 0 <= score <= 1 else 0.0


def _button(name: str, text: str, bbox: tuple[int, int, int, int], confidence: float) -> dict:
    x1, y1, x2, y2 = bbox
    return {"name": name, "text": text, "bbox": list(bbox),
            "point": [(x1 + x2) // 2, (y1 + y2) // 2], "confidence": confidence}


def observe_battle_reward_claim(frame_path: str | Path, ocr: list[dict], *,
                                client_version: str, baseline_resolution: tuple[int, int] = BASELINE) -> dict | None:
    """Require full real result anchors and the single green chest claim control."""
    image = _image(frame_path, client_version, baseline_resolution, ocr)
    if image is None:
        return None
    results = {name: _exact(ocr, text, (545, 175, 745, 260))
               for name, text in (("victory", "胜利"), ("defeat", "失败"))}
    if sum(len(matches) for matches in results.values()) != 1:
        return None
    result = "victory" if results["victory"] else "defeat"
    anchors = {result: results[result][0]}
    for name, text, roi in (
        ("received", "您得到了", (555, 275, 730, 330)),
        ("losses", "损耗的部队", (550, 450, 735, 495)),
        ("claim", "领取奖励", (525, 580, 760, 660)),
    ):
        matches = _exact(ocr, text, roi)
        if len(matches) != 1:
            return None
        anchors[name] = matches[0]
    percentages = [item for item in ocr if isinstance(item, dict)
                   and type(item.get("confidence")) in (int, float)
                   and .95 <= item["confidence"] <= 1
                   and _box_in(item.get("bbox"), (570, 95, 715, 145))
                   and re.fullmatch(r"\s*\d{1,3}\s*[%％]\s*", str(item.get("text", "")))]
    if len(percentages) != 1 or not 0 <= int(re.sub(r"\D", "", percentages[0]["text"])) <= 100:
        return None
    if (_exact(ocr, "回营", (520, 570, 760, 675)) or
            (result == "defeat" and int(re.sub(r"\D", "", percentages[0]["text"])) >= 50)):
        return None
    score = _score(image, "battle_reward_claim_button", (530, 585, 755, 655))
    if score < .97:
        return None
    claim = anchors["claim"]
    bbox = tuple(claim["bbox"])
    return {"surface": "battle_reward_claim_settlement", "result": result,
            "client_version": client_version,
            "frame": str(frame_path), "frame_sha256": _sha256(Path(frame_path)),
            "anchors": anchors, "percentage": percentages[0], "button_template_score": score,
            "button": _button("claim_battle_reward", "领取奖励", bbox,
                              min(claim["confidence"], score))}


def _hammers(image: np.ndarray) -> tuple[int, list[float]] | None:
    bright = []
    for x1, y1, x2, y2 in _HAMMER_ROIS:
        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        bright.append(float(np.mean(hsv[:, :, 2] > 180)))
    states = [value >= .18 if value >= .18 or value <= .03 else None for value in bright]
    if any(state is None for state in states):
        return None
    count = sum(states)
    return (count, bright) if states == [False] * (4 - count) + [True] * count else None


def observe_battle_chest(frame_path: str | Path, ocr: list[dict], *,
                         client_version: str, baseline_resolution: tuple[int, int] = BASELINE) -> dict | None:
    """Recognize only the five captured closed/result pages; never infer a tap."""
    image = _image(frame_path, client_version, baseline_resolution, ocr)
    if image is None:
        return None
    frame = Path(frame_path)
    frame_hash = _sha256(frame)
    title = _exact(ocr, "跳过开箱动画", (545, 200, 735, 255))
    message = _exact(ocr, "您将获得奖励，但会跳过开箱动画。仍要跳过", (370, 280, 910, 335))
    question = _exact(ocr, "吗？", (600, 320, 680, 370))
    yes = _exact(ocr, "是", (750, 440, 810, 495))
    if title or message or question or yes:
        expected = {"跳过开箱动画", "您将获得奖励，但会跳过开箱动画。仍要跳过", "吗？", "是", "跳过"}
        if (not all(len(matches) == 1 for matches in (title, message, question, yes))
                or _confident_texts(ocr) != expected):
            return None
        score = _score(image, "battle_chest_skip_confirm", _SKIP_CONFIRM_ROI)
        if score < .97:
            return None
        return {"surface": "battle_chest", "stage": "skip_confirmation",
                "client_version": client_version, "frame": str(frame), "frame_sha256": frame_hash,
                "anchors": {"title": title[0], "message": message[0], "question": question[0],
                            "yes": yes[0]},
                "visual": {"dialog_template_score": score, "roi_at_1280x720": list(_SKIP_CONFIRM_ROI)},
                "button": _button("confirm_free_battle_chest_skip", "是", tuple(yes[0]["bbox"]),
                                  min(yes[0]["confidence"], score))}
    potion = _exact(ocr, "英雄药水", (560, 180, 725, 245))
    storage = _exact(ocr, "仓库已满！存放不下的部分已运送到临时仓库。", (420, 495, 855, 565))
    continue_buttons = _exact(ocr, "继续", (555, 610, 735, 690))
    if len(potion) == len(storage) == len(continue_buttons) == 1:
        if _confident_texts(ocr) != {"英雄药水", "仓库已满！存放不下的部分已运送到临时仓库。", "继续"}:
            return None
        item_score = _score(image, "battle_chest_hero_potion", (515, 260, 745, 485))
        button_score = _score(image, "battle_chest_continue", (555, 620, 745, 685))
        if item_score >= .97 and button_score >= .97:
            return {"surface": "battle_chest", "stage": "hero_potion_temporary_storage",
                    "client_version": client_version, "frame": str(frame), "frame_sha256": frame_hash,
                    "reward_item": "hero_potion", "reward_amount": None,
                    "inventory_verified": False, "message": storage[0],
                    "visual": {"item_template_score": item_score, "button_template_score": button_score},
                    "button": _button("continue_battle_reward", "继续", tuple(continue_buttons[0]["bbox"]),
                                      min(continue_buttons[0]["confidence"], item_score, button_score))}
    if potion or storage or continue_buttons:
        return None
    background = _score(image, "battle_chest_background", _BACKGROUND_ROI)
    if background < .97:
        return None
    ordinary = _exact(ocr, "普通", (575, 490, 690, 565))
    rare = _exact(ocr, "稀有", (575, 490, 690, 565))
    if len(ordinary) + len(rare) != 1:
        return None
    rarity = "ordinary" if ordinary else "rare"
    texts = _confident_texts(ocr)
    skip = _exact(ocr, "跳过", (1150, 620, 1230, 685))
    expected = {"点击！", "普通", "跳过"} if skip else {"普通"}
    if ordinary and texts != expected:
        return None
    if rare and texts != {"点击！", "稀有"}:
        return None
    if skip and (len(skip) != 1 or not ordinary):
        return None
    label = ordinary[0] if ordinary else rare[0]
    chest_score = _score(image, "battle_chest_ordinary" if ordinary else "battle_chest_rare", _CHEST_ROI)
    hammers = _hammers(image)
    if chest_score < .97 or hammers is None or (rarity, hammers[0]) not in _CHAIN:
        return None
    return {"surface": "battle_chest", "stage": "closed_chest", "rarity": rarity,
            "hammers_remaining": hammers[0], "client_version": client_version,
            "frame": str(frame), "frame_sha256": frame_hash,
            "label": label, "visual": {"background_score": background,
                                      "chest_score": chest_score, "hammer_bright_fractions": hammers[1]},
            "button": (_button("skip_battle_chest_animation", "跳过", tuple(skip[0]["bbox"]),
                               min(skip[0]["confidence"], background, chest_score)) if skip
                       else _button("strike_battle_chest", "已验证免费宝箱", (600, 345, 680, 435),
                                    min(label["confidence"], background, chest_score)))}


def observe_battle_chest_currency_banner(frame_path: str | Path, ocr: list[dict], *,
                                         client_version: str,
                                         baseline_resolution: tuple[int, int] = BASELINE) -> dict | None:
    """Read one current HOME banner; its currency is separate from battle loot."""
    image = _image(frame_path, client_version, baseline_resolution, ocr)
    if image is None:
        return None
    candidates = []
    signals = []
    for item in ocr:
        if not _box_in(item.get("bbox"), (495, 170, 900, 235)):
            continue
        normalized = re.sub(r"\s+", "", str(item.get("text", "")))
        if "宝箱" in normalized or "获得了" in normalized:
            signals.append(item)
        if (type(item.get("confidence")) not in (int, float) or
                not math.isfinite(item["confidence"]) or not .95 <= item["confidence"] <= 1):
            continue
        match = re.fullmatch(r"一个宝箱并获得了([1-9][0-9]{0,8})[Xx×](金币|圣水|黑油)", normalized)
        if match:
            candidates.append((item, match))
    if len(candidates) != 1 or len(signals) != 1 or candidates[0][0] is not signals[0]:
        return None
    item, match = candidates[0]
    # Require current bright text pixels inside the recognized line as well as
    # the OCR box, so text borrowed from a different frame cannot authorize a
    # reward. The line is against the map rather than a fixed backdrop.
    x1, y1, x2, y2 = map(int, item["bbox"])
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    bright_fraction = float(np.mean((hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 185)))
    if bright_fraction < .08:
        return None
    return {"resource": _CURRENCY_NAMES[match.group(2)], "amount": int(match.group(1)),
            "ocr": item, "bright_text_fraction": bright_fraction,
            "frame": str(frame_path), "frame_sha256": _sha256(Path(frame_path))}


def _require_foreground(session) -> None:
    capture = session.capture
    if foreground_package(capture.adb, capture.serial,
                          timeout_sec=min(5, session.config.runtime.step_timeout_sec)) != session.config.game.package_name:
        raise FlowError("Battle chest game is not foreground")


def _deadline(session, end: float) -> None:
    session.check_deadline()
    if time.monotonic() >= end:
        raise FlowError("Battle chest deadline expired")


def _current(session, snapshot, end: float, stage: str) -> dict:
    _deadline(session, end)
    confidence = snapshot.confidence
    if (session.client_version != CLIENT_VERSION or tuple(session.config.game.baseline_resolution) != BASELINE
            or snapshot is not session.last_snapshot
            or type(confidence) not in (int, float) or not math.isfinite(confidence) or not .95 <= confidence <= 1
            or snapshot.scene != ("settlement" if stage == "claim" else "battle_reward")
            or snapshot.observations.get("account_in_use") is not None):
        raise FlowError("Battle reward input lacks a current supported scene")
    captured = snapshot.observations.get("observed_at_monotonic")
    requested = snapshot.observations.get("requested_at_monotonic")
    frame_id = snapshot.observations.get("frame_id")
    if (type(captured) not in (int, float) or not math.isfinite(captured)
            or type(requested) not in (int, float) or not math.isfinite(requested)
            or not isinstance(frame_id, str) or not frame_id
            or not 0 <= time.monotonic() - captured <= 8):
        raise FlowError("Battle reward observation is stale or lacks capture identity")
    name = "battle_reward_claim" if stage == "claim" else "battle_reward"
    observed = snapshot.observations.get(name)
    if (not isinstance(observed, dict) or observed.get("frame") != str(snapshot.screenshot_path)
            or observed.get("frame_sha256") != _sha256(snapshot.screenshot_path)):
        raise FlowError("Battle reward source frame changed")
    detector = observe_battle_reward_claim if stage == "claim" else observe_battle_chest
    fresh = detector(snapshot.screenshot_path, snapshot.observations.get("ocr", []),
                     client_version=session.client_version,
                     baseline_resolution=session.config.game.baseline_resolution)
    if fresh is None or fresh != observed:
        raise FlowError("Battle reward current pixels or control failed revalidation")
    return fresh


def _read_next(session, expected: tuple[str, int] | str, end: float, evidence: list[Path],
               label: str, pair_out: list | None = None):
    """A single issued input can be followed only by fresh reads, never reissued."""
    previous = None
    previous_snapshot = None
    for index in range(8):
        _deadline(session, end)
        _require_foreground(session)
        current = session.observe(f"battle-reward-{label}-{index + 1}", purpose="full")
        evidence.append(current.screenshot_path)
        _deadline(session, end)
        if current.scene in {"account_in_use", "maintenance", "disconnected"}:
            raise FlowError(f"Game interruption during battle reward: {current.scene}")
        frame_id = current.observations.get("frame_id")
        captured = current.observations.get("observed_at_monotonic")
        requested = current.observations.get("requested_at_monotonic")
        fresh = (isinstance(frame_id, str) and bool(frame_id)
                 and type(captured) in (int, float) and math.isfinite(captured)
                 and type(requested) in (int, float) and math.isfinite(requested)
                 and 0 <= time.monotonic() - captured <= 8
                 and 0 <= time.monotonic() - requested <= 8)
        if expected == "home":
            controls = {button.get("name") for button in current.observations.get("buttons", [])
                        if isinstance(button, dict)}
            resources = current.observations.get("resources")
            complete_resources = (isinstance(resources, dict) and
                                  all(type(resources.get(name)) is int and resources[name] >= 0 for name in
                                      ("gold", "elixir", "dark_elixir", "gems")))
            state = (fresh and current.scene == "village"
                     and type(current.confidence) in (int, float)
                     and math.isfinite(current.confidence) and current.confidence >= .8
                     and current.observations.get("village_type") == "home"
                     and current.observations.get("resource_source") == "village_inventory"
                     and {"attack", "shop"} <= controls and complete_resources)
        else:
            reward = current.observations.get("battle_reward")
            state = (fresh and current.scene == "battle_reward"
                     and type(current.confidence) in (int, float)
                     and math.isfinite(current.confidence) and current.confidence >= .95
                     and isinstance(reward, dict) and
                     (reward.get("stage") == "hero_potion_temporary_storage" if expected == "potion"
                      else reward.get("stage") == "skip_confirmation" if expected == "skip_confirmation"
                      else reward.get("stage") == "closed_chest"
                      and (reward.get("rarity"), reward.get("hammers_remaining")) == expected))
        if state:
            frame_hash = _sha256(current.screenshot_path)
            current_values = (frame_id, frame_hash, captured, requested,
                              tuple(resources[name] for name in ("gold", "elixir", "dark_elixir", "gems"))
                              if expected == "home" else None)
            if (previous is not None and previous[0] != frame_id
                    and previous[2] < captured and previous[3] < requested
                    and (expected != "home" or previous[4] == current_values[4])):
                if pair_out is not None:
                    pair_out[:] = [previous_snapshot, current]
                return current
            previous = current_values
            previous_snapshot = current
        else:
            previous = None
            previous_snapshot = None
        if index < 7:
            time.sleep(min(.35, max(0, end - time.monotonic())))
    raise FlowError(f"Battle reward did not reach two fresh {expected!r} frames")


def finish_claimed_battle_reward(session, settlement, evidence: list[Path], *, timeout_sec: float = 100) -> tuple[object, dict]:
    """Claim one known free chest and return home, with no tap retry on uncertainty."""
    if not 0 < timeout_sec <= 120:
        raise ValueError("Invalid battle reward timeout")
    end = min(time.monotonic() + timeout_sec, session.deadline, session.task_deadline)
    receipt = {"kind": "battle_chest_ui", "client_version": session.client_version,
               "status": "incomplete", "input_attempts": [], "stages": [],
               "reward_item": None, "reward_amount": None,
               "reward_inventory_verified": False, "currency_reward_observed": False,
               "returned_home": False, "formal_acceptance": False}
    claims = getattr(session, "_battle_reward_claim_attempts", None)
    if claims is None:
        claims = set()
        session._battle_reward_claim_attempts = claims
    if getattr(session, "_battle_reward_chain_incomplete", False) or str(settlement.screenshot_path) in claims:
        raise FlowError("Battle reward claim was already attempted or an earlier chain is incomplete")
    session.battle_reward_receipt = receipt
    _require_foreground(session)
    claim = _current(session, settlement, end, "claim")
    attempts = set()

    def click_once(snapshot, action: str, stage: str) -> None:
        _require_foreground(session)
        current = _current(session, snapshot, end, stage)
        buttons = [button for button in snapshot.observations.get("buttons", [])
                   if isinstance(button, dict) and button.get("name") == action]
        if len(buttons) != 1 or buttons[0] != current["button"]:
            raise FlowError("Battle reward button differs from verified current pixels")
        key = (current["frame_sha256"], action)
        if key in attempts:
            raise FlowError("Battle chest input already attempted on this frame")
        attempts.add(key)
        item = {"action": action, "frame": str(snapshot.screenshot_path),
                "frame_sha256": current["frame_sha256"], "frame_id": snapshot.observations["frame_id"],
                "point": current["button"]["point"], "input_attempted": True, "input_returned": False}
        receipt["input_attempts"].append(item)
        session.event("battle_reward_input_reserved", action=action, frame=item["frame"],
                      frame_sha256=item["frame_sha256"])
        _deadline(session, end)
        _require_foreground(session)
        session.tap(snapshot, tuple(current["button"]["point"]),
                    reason=f"Verified free battle reward action: {action}")
        item["input_returned"] = True

    claims.add(str(settlement.screenshot_path))
    session._battle_reward_chain_incomplete = True
    click_once(settlement, "claim_battle_reward", "claim")
    current = _read_next(session, _CHAIN[0], end, evidence, "chest-1")
    first = _current(session, current, end, "chest")
    if first["button"]["name"] == "skip_battle_chest_animation":
        receipt["stages"].append({"rarity": first["rarity"], "hammers_remaining": first["hammers_remaining"],
                                  "frame": first["frame"], "frame_sha256": first["frame_sha256"],
                                  "route": "observed_free_skip"})
        click_once(current, "skip_battle_chest_animation", "chest")
        confirmation = _read_next(session, "skip_confirmation", end, evidence, "skip-confirmation")
        confirm = _current(session, confirmation, end, "chest")
        receipt["stages"].append({"stage": "skip_confirmation", "frame": confirm["frame"],
                                  "frame_sha256": confirm["frame_sha256"]})
        click_once(confirmation, "confirm_free_battle_chest_skip", "chest")
        homes = []
        home = _read_next(session, "home", end, evidence, "home", pair_out=homes)
        if homes[0].screenshot_path == homes[1].screenshot_path:
            raise FlowError("Battle chest HOME captures reused one frame path")
        receipt["home_frames"] = [
            {"frame": str(snapshot.screenshot_path), "frame_sha256": _sha256(snapshot.screenshot_path),
             "frame_id": snapshot.observations["frame_id"],
             "resources": snapshot.observations["resources"]} for snapshot in homes]
        receipt.update(returned_home=True, home_frame=str(home.screenshot_path),
                       home_frame_sha256=_sha256(home.screenshot_path))
        banners = [observe_battle_chest_currency_banner(
            snapshot.screenshot_path, snapshot.observations.get("ocr", []),
            client_version=session.client_version,
            baseline_resolution=session.config.game.baseline_resolution) for snapshot in homes]
        observed = [banner for banner in banners if banner is not None]
        if (not observed or any((banner["resource"], banner["amount"]) !=
                                (observed[0]["resource"], observed[0]["amount"])
                                for banner in observed)):
            raise FlowError("Battle chest currency reward is missing or conflicting")
        reward = observed[0]
        receipt.update(status="ui_complete_home_currency_observed",
                       reward_item=reward["resource"], reward_amount=reward["amount"],
                       currency_reward_observed=True, currency_reward_evidence=reward)
        session._battle_reward_chain_incomplete = False
        return home, receipt
    for index, state in enumerate(_CHAIN):
        if index:
            current = _read_next(session, state, end, evidence, f"chest-{index + 1}")
        stage = _current(session, current, end, "chest")
        receipt["stages"].append({"rarity": stage["rarity"], "hammers_remaining": stage["hammers_remaining"],
                                  "frame": stage["frame"], "frame_sha256": stage["frame_sha256"]})
        click_once(current, "strike_battle_chest", "chest")
    prize = _read_next(session, "potion", end, evidence, "potion")
    prize_state = _current(session, prize, end, "chest")
    receipt.update(reward_item="hero_potion", reward_amount=None,
                   temporary_storage_message=prize_state["message"],
                   prize_frame=prize_state["frame"], prize_frame_sha256=prize_state["frame_sha256"])
    click_once(prize, "continue_battle_reward", "chest")
    home = _read_next(session, "home", end, evidence, "home")
    receipt.update(status="ui_complete_home_verified", returned_home=True,
                   home_frame=str(home.screenshot_path), home_frame_sha256=_sha256(home.screenshot_path))
    session._battle_reward_chain_incomplete = False
    return home, receipt
