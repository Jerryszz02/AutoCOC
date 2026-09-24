"""Clan interactions with observed postconditions, without changing request text."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import re
import time

from .errors import AutoCOCError, FlowError
from .donation import execute_donation
from .reporting import TaskResult
from .scene import SceneSnapshot
from .session import GameSession
from .vision import normalize_label, parse_capacity, parse_resource_number


CHAT_REQUEST_REGION = (365, 635, 489, 719)
SEND_REGION = (300, 400, 1000, 600)
CHAT_MESSAGES_REGION = (0, 60, 490, 635)
MAX_DONATION_SCAN_PAGES = 4


def request_reinforcements(session: GameSession) -> TaskResult:
    started_at, started = datetime.now(), time.monotonic()
    evidence: list[Path] = []
    metrics: dict[str, object] = {"request_submitted": 0, "send_actions": 0, "request_cost_gems": 0}
    status, reason = "failed", "request_postcondition_unverified"
    try:
        current = session.observe("request-start")
        evidence.append(current.screenshot_path)
        _require_scene(current, {"village", "clan_chat", "request"})
        if current.scene != "request":
            current = _open_chat(session, current, evidence)
            current = _latest_chat(session, current, evidence)
            cooldown = _cooldown(current)
            if cooldown is not None and cooldown > 0:
                metrics["cooldown_seconds"] = cooldown
                status, reason = "skipped", "request_cooldown"
                return TaskResult("request", status, reason, started_at, time.monotonic() - started, evidence, metrics)
            session.click_template(current, "chat_request", roi=CHAT_REQUEST_REGION)
            current = session.wait_for({"request"}, timeout_sec=45, label="request-dialog")
            evidence.append(current.screenshot_path)

        _require_scene(current, {"request"})
        gems_before = _gems(current)
        if gems_before is None:
            raise FlowError("Gem inventory unreadable before request")
        body = "".join(item["text"] for item in _ocr_in(current, (350, 280, 920, 430)))
        if not body:
            raise FlowError("Existing request text unreadable; submission cannot be verified")
        metrics.update({"request_text": body, "gems_before": gems_before})
        session.click(current, "send", region=SEND_REGION)
        metrics["send_actions"] = 1
        metrics["request_cost_gems"] = None
        after = session.wait_for({"clan_chat"}, timeout_sec=45, label="request-after-send")
        evidence.append(after.screenshot_path)
        after = _latest_chat(session, after, evidence)
        for attempt in range(3):
            session.check_deadline()
            _require_scene(after, {"clan_chat"})
            if after.screenshot_path == current.screenshot_path:
                raise FlowError("Request verification reused its before screenshot")
            cooldown = _cooldown(after)
            record = _fresh_request_record(session, after, body)
            gems_after = _gems(after)
            metrics.update({"gems_after": gems_after, "cooldown_seconds": cooldown, "request_record": record})
            if gems_after is not None and gems_after != gems_before:
                metrics["request_cost_gems"] = gems_before - gems_after
                raise FlowError("Gem balance changed during request submission")
            if gems_after == gems_before:
                metrics["request_cost_gems"] = 0
            if cooldown is not None and cooldown > 0 and record is not None and gems_after == gems_before:
                metrics["request_submitted"] = 1
                session.event("request_verified", before_frame=str(current.screenshot_path),
                              after_frame=str(after.screenshot_path), cooldown_seconds=cooldown, request_record=record)
                status, reason = "succeeded", "request_record_and_cooldown_verified"
                break
            if attempt < 2:
                time.sleep(min(session.config.runtime.poll_interval_sec, 1.0))
                after = session.observe("request-verify")
                evidence.append(after.screenshot_path)
    except AutoCOCError as exc:
        reason = str(exc)
        _retain_last_frame(session, evidence)
    return TaskResult("request", status, reason, started_at, time.monotonic() - started, evidence, metrics)


def donate_troops(session: GameSession) -> TaskResult:
    """Scan chat and open one request; uncalibrated unit controls cannot be used."""
    started_at, started = datetime.now(), time.monotonic()
    evidence: list[Path] = []
    metrics: dict[str, object] = {"donated_units": 0, "donation_cost_gems": 0,
                                 "donation_cost_gold": 0, "donation_cost_elixir": 0, "donation_cost_dark_elixir": 0,
                                 "scan_scope": "current_clan_chat_and_up_to_three_older_pages",
                                 "scan_region": list(CHAT_MESSAGES_REGION), "scan_max_pages": MAX_DONATION_SCAN_PAGES,
                                 "pages_scanned": 0, "unique_pages_scanned": 0, "scan_pages": [],
                                 "visible_donation_requests": 0, "request_buttons": [], "scan_stop_reason": None}
    status, reason = "failed", "donation_interaction_not_supported"
    try:
        current = session.observe("donate-start")
        evidence.append(current.screenshot_path)
        current = _open_chat(session, current, evidence)
        seen: set[tuple[str, ...]] = set()
        for page_number in range(1, MAX_DONATION_SCAN_PAGES + 1):
            session.check_deadline()
            _require_scene(current, {"clan_chat"})
            identity = _chat_page_identity(current)
            buttons = _donation_buttons(session, current)
            page = {"page": page_number, "frame": str(current.screenshot_path),
                    "identity": list(identity), "donation_buttons": buttons}
            metrics["scan_pages"].append(page)
            metrics["pages_scanned"] = page_number
            metrics["visible_donation_requests"] = len(buttons)
            metrics["request_buttons"] = buttons
            session.event("donation_scan_page", **page)
            repeated = identity in seen
            if identity:
                seen.add(identity)
            metrics["unique_pages_scanned"] = len(seen)
            if buttons:
                metrics["scan_stop_reason"] = "donation_request_found"
                button = min(buttons, key=lambda item: (item["point"][1], item["point"][0]))
                session.tap(current, button["point"], reason="Open recognized clan donation request")
                metrics["opened_request"] = True
                current = session.wait_for({"donation"}, timeout_sec=30, label="donation-dialog")
                evidence.append(current.screenshot_path)
                status, reason = execute_donation(session, current, evidence, metrics)
                break
            if not identity:
                raise FlowError("Clan request area unreadable")
            if repeated or page_number == MAX_DONATION_SCAN_PAGES:
                metrics["scan_stop_reason"] = "repeated_page" if repeated else "page_limit"
                status, reason = "skipped", "no_donation_requests_in_scanned_chat"
                break
            session.swipe(current, (180, 200), (180, 600), duration_ms=500,
                          reason=f"Inspect older clan requests after page {page_number}")
            current = _next_chat_page(session, identity, evidence)
    except AutoCOCError as exc:
        reason = str(exc)
        if not metrics.get("opened_request"):
            metrics["scan_stop_reason"] = "scan_failed"
        _retain_last_frame(session, evidence)
    return TaskResult("donate", status, reason, started_at, time.monotonic() - started, evidence, metrics)


def _next_chat_page(session: GameSession, previous: tuple[str, ...], evidence: list[Path]) -> SceneSnapshot:
    # A completed input command can precede the game's scroll animation. Observe
    # again without issuing another swipe before declaring the page unchanged.
    for _ in range(3):
        session.check_deadline()
        time.sleep(min(max(session.config.runtime.poll_interval_sec, 0.5), 1.0))
        current = session.wait_for({"clan_chat"}, timeout_sec=10, label="donation-scan")
        evidence.append(current.screenshot_path)
        if _donation_buttons(session, current) or _chat_page_identity(current) != previous:
            return current
    return current


def _donation_buttons(session: GameSession, snapshot: SceneSnapshot) -> list[dict]:
    left, top, right, bottom = CHAT_MESSAGES_REGION
    return [button for button in session.buttons(snapshot, "donate")
            if left <= button["point"][0] <= right and top <= button["point"][1] <= bottom
            and normalize_label(button.get("text", "")) in {"捐赠", "donate"}]


def _chat_page_identity(snapshot: SceneSnapshot) -> tuple[str, ...]:
    contents = []
    for item in _ocr_in(snapshot, CHAT_MESSAGES_REGION):
        value = re.sub(r"\s+", "", item["text"].lower())
        # Right-aligned message ages change while the same cards stay on screen.
        if item["bbox"][0] >= 300 and (
                value in {"刚刚", "justnow"} or re.fullmatch(
                    r"(?:\d+(?:天|小时|分钟|秒钟|分|秒|days?|hours?|minutes?|seconds?|d|h|m|s))+(?:前|ago)?", value)):
            continue
        label = normalize_label(value)
        if label:
            contents.append(label)
    # Ignore pixel motion and vertical OCR ordering changes, keeping duplicate
    # text entries so repeated cards still contribute to page identity.
    return tuple(sorted(contents))


def _open_chat(session: GameSession, current: SceneSnapshot, evidence: list[Path]) -> SceneSnapshot:
    _require_scene(current, {"village", "clan_chat"})
    if current.scene == "village":
        session.click_template(current, "hud_chat", roi=(0, 250, 130, 390))
        current = session.wait_for({"clan_chat"}, timeout_sec=30, label="social-chat")
        evidence.append(current.screenshot_path)
    _require_scene(current, {"clan_chat"})
    return current


def _latest_chat(session: GameSession, current: SceneSnapshot, evidence: list[Path]) -> SceneSnapshot:
    """Restore the latest messages after a donation scan or manual scrolling."""
    _require_scene(current, {"clan_chat"})
    if not session.buttons(current, "chat_latest"):
        return current
    session.click(current, "chat_latest", region=(0, 530, 100, 650))
    for _ in range(3):
        current = session.wait_for({"clan_chat"}, timeout_sec=10, label="chat-latest")
        evidence.append(current.screenshot_path)
        if not session.buttons(current, "chat_latest"):
            return current
        time.sleep(min(session.config.runtime.poll_interval_sec, 1.0))
    raise FlowError("Clan chat did not return to the latest messages")


def _require_scene(snapshot: SceneSnapshot, scenes: set[str]) -> None:
    if snapshot.scene not in scenes or not math.isfinite(snapshot.confidence) or snapshot.confidence < 0.8:
        raise FlowError(f"Expected social scene {sorted(scenes)}, observed {snapshot.scene}")


def _retain_last_frame(session: GameSession, evidence: list[Path]) -> None:
    if session.last_snapshot is not None and session.last_snapshot.screenshot_path not in evidence:
        evidence.append(session.last_snapshot.screenshot_path)


def _ocr_in(snapshot: SceneSnapshot, region: tuple[int, int, int, int]) -> list[dict]:
    left, top, right, bottom = region
    items = []
    for item in snapshot.observations.get("ocr", []):
        box = item.get("bbox")
        confidence = item.get("confidence", 0)
        if box is not None and math.isfinite(confidence) and confidence >= 0.8:
            if left <= (box[0] + box[2]) / 2 <= right and top <= (box[1] + box[3]) / 2 <= bottom:
                items.append(item)
    return sorted(items, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _gems(snapshot: SceneSnapshot) -> int | None:
    candidates = [parse_resource_number(item["text"]) for item in _ocr_in(snapshot, (1100, 215, 1230, 270))]
    values = [value for value in candidates if value is not None]
    return values[0] if len(values) == 1 else None


def _duration(text: str) -> int | None:
    normalized = re.sub(r"\s+", "", text.lower())
    if normalized in {"刚刚", "justnow"}:
        return 0
    match = re.fullmatch(r"(?:(\d+)(?:分钟|分|m))?(?:(\d+)(?:秒钟|秒|s))?", normalized)
    if match and any(part is not None for part in match.groups()):
        return int(match[1] or 0) * 60 + int(match[2] or 0)
    return None


def _cooldown(snapshot: SceneSnapshot) -> int | None:
    # The observed button also shows a gem price; bare numbers are never timers.
    timers = [_duration(item["text"]) for item in _ocr_in(snapshot, CHAT_REQUEST_REGION)]
    values = [value for value in timers if value is not None]
    return values[0] if len(values) == 1 else None


def _fresh_request_record(session: GameSession, snapshot: SceneSnapshot, body: str) -> dict[str, object] | None:
    items = _ocr_in(snapshot, (0, 60, 490, 650))
    for marker in items:
        if normalize_label(marker["text"]) not in {"请求", "request"}:
            continue
        y = marker["bbox"][1]
        contents = [item for item in items if 0 < y - item["bbox"][1] < 100
                    and normalize_label(item["text"]) == normalize_label(body)]
        capacities = [item for item in items if y < item["bbox"][1] < min(y + 190, 610)
                      and parse_capacity(item["text"]) is not None]
        if not contents or not capacities:
            continue
        bottom = max(item["bbox"][3] for item in capacities)
        region = (350, bottom, 489, min(bottom + 85, 650))
        readings = [{**item, "source": "frame_ocr"} for item in _ocr_in(snapshot, region)]
        # The small "just now" timestamp was missed by whole-frame OCR in the live fixture.
        if not any(_duration(item["text"]) is not None for item in readings):
            readings += [{"text": item.text, "confidence": item.confidence, "bbox": item.bbox,
                          "source": "region_ocr"}
                         for item in session.recognizer.recognize_region(snapshot.screenshot_path, region)
                         if math.isfinite(item.confidence) and 0.9 <= item.confidence <= 1]
        if not any(_duration(item["text"]) is not None for item in readings):
            # This fixed line is anchored to the observed capacity row of the same card.
            line_region = (420, bottom + 12, 485, min(bottom + 51, 650))
            if line_region[1] < line_region[3]:
                readings += [{"text": item.text, "confidence": item.confidence, "bbox": item.bbox,
                              "source": "timestamp_line_ocr"}
                             for item in session.recognizer.recognize_region(
                                 snapshot.screenshot_path, line_region, single_line=True)
                             if math.isfinite(item.confidence) and 0.9 <= item.confidence <= 1]
        fresh = next((item for item in readings if _duration(item["text"]) is not None
                      and _duration(item["text"]) <= 60), None)
        if fresh is not None:
            return {"text": body, "marker_bbox": marker["bbox"], "capacity_text": [item["text"] for item in capacities],
                    "age_seconds": _duration(fresh["text"]), "timestamp_region": region,
                    "timestamp_evidence": fresh}
    return None
