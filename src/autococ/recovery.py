"""A single, visually verified reconnect attempt with bounded observation."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import time

from .errors import AutoCOCError, FlowError
from .reporting import TaskResult
from .scene import SceneSnapshot
from .session import GameSession


RETRY_REGION = (300, 270, 980, 460)


def welcome_back_point(snapshot: SceneSnapshot) -> list[int] | None:
    """Locate only the observed welcome summary's bottom confirmation."""
    if snapshot.scene != "popup" or not _confident(snapshot.confidence):
        return None
    ocr = snapshot.observations.get("ocr")
    if not isinstance(ocr, list):
        return None

    def anchored(text_test, region, minimum):
        left, top, right, bottom = region
        matches = []
        for item in ocr:
            if not isinstance(item, dict) or not text_test(str(item.get("text", "")).strip()):
                continue
            box, confidence = item.get("bbox"), item.get("confidence")
            if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                    or confidence < minimum or not isinstance(box, (list, tuple))
                    or len(box) != 4 or not all(type(value) is int for value in box)):
                continue
            if left <= box[0] < box[2] <= right and top <= box[1] < box[3] <= bottom:
                matches.append(box)
        return matches[0] if len(matches) == 1 else None

    title = anchored(lambda text: text == "首领，欢迎回来！", (500, 55, 780, 130), .95)
    explanation = anchored(lambda text: text.startswith("您离开村庄的这段时间里"),
                           (490, 125, 800, 190), .9)
    confirms = [item for item in ocr if isinstance(item, dict)
                and str(item.get("text", "")).strip() == "确定"]
    confirm = anchored(lambda text: text == "确定", (600, 560, 710, 640), .95)
    if title is None or explanation is None or len(confirms) != 1 or confirm is None:
        return None
    return [(confirm[0] + confirm[2]) // 2, (confirm[1] + confirm[3]) // 2]


def confirm_welcome_back(session: GameSession, snapshot: SceneSnapshot) -> bool:
    """Dismiss one fresh, narrowly identified welcome summary."""
    point = welcome_back_point(snapshot)
    if point is None:
        return False
    if snapshot is not session.last_snapshot:
        raise FlowError("Welcome confirmation refers to an obsolete observation")
    session.check_deadline()
    session.tap(snapshot, point, reason="Dismiss observed welcome-back summary")
    return True


def recover_connection(session: GameSession, *, initial_snapshot: SceneSnapshot | None = None) -> TaskResult:
    started_at, started = datetime.now(), time.monotonic()
    evidence: list[Path] = []
    metrics: dict[str, object] = {
        "retry_actions": 0, "welcome_confirm_actions": 0, "village_verified": False,
        "before_frame": None, "after_frame": None,
        "gems_before": None, "gems_after": None, "gems_delta": None,
    }
    status, reason = "failed", "connection_recovery_incomplete"
    try:
        before = initial_snapshot if initial_snapshot is not None else session.observe("recovery-before")
        evidence.append(before.screenshot_path)
        metrics.update(before_frame=str(before.screenshot_path), gems_before=_gems(before))
        if before is not session.last_snapshot:
            raise FlowError("Recovery refers to an obsolete observation")
        if before.scene != "disconnected" or not _confident(before.confidence):
            raise FlowError("Recovery requires a confirmed disconnected scene")
        _require_retry(before)
        session.click(before, "retry", region=RETRY_REGION)
        metrics["retry_actions"] = 1
        deadline = min(time.monotonic() + 60, session.deadline, session.task_deadline)
        welcome_confirmed = False
        while True:
            session.check_deadline()
            if time.monotonic() >= deadline:
                raise FlowError("Connection recovery timed out before village verification")
            # wait_for rejects disconnected immediately, including the old frame
            # still visible while this single retry is being processed.
            after = session.observe("recovery-after")
            evidence.append(after.screenshot_path)
            metrics.update(after_frame=str(after.screenshot_path), gems_after=_gems(after),
                           last_scene=after.scene)
            session.check_deadline()
            if time.monotonic() >= deadline:
                raise FlowError("Connection recovery timed out before village verification")
            if after.scene == "maintenance":
                raise FlowError("Game interruption: maintenance")
            if not welcome_confirmed and confirm_welcome_back(session, after):
                welcome_confirmed = True
                metrics["welcome_confirm_actions"] = 1
                continue
            if after.scene == "village" and _confident(after.confidence):
                if any(len(session.buttons(after, name)) != 1 for name in ("attack", "shop")):
                    raise FlowError("Recovered village is missing unique attack/shop controls")
                metrics["village_verified"] = True
                status, reason = "succeeded", "single_retry_and_village_verified"
                break
            time.sleep(min(session.config.runtime.poll_interval_sec, max(0, deadline - time.monotonic())))
    except AutoCOCError as exc:
        reason = str(exc)
        last = session.last_snapshot
        if last is not None and last.screenshot_path not in evidence:
            evidence.append(last.screenshot_path)
    if metrics["gems_before"] is not None and metrics["gems_after"] is not None:
        metrics["gems_delta"] = metrics["gems_after"] - metrics["gems_before"]
    return TaskResult("recover", status, reason, started_at, time.monotonic() - started, evidence, metrics)


def _confident(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0.8 <= value <= 1


def _require_retry(snapshot: SceneSnapshot) -> None:
    left, top, right, bottom = RETRY_REGION
    matches = []
    for button in snapshot.observations.get("buttons", []):
        if button.get("name") != "retry":
            continue
        point = button.get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise FlowError("Retry button has no valid visual point")
        if not all(type(value) in (int, float) and math.isfinite(value) for value in point):
            raise FlowError("Retry button has no valid visual point")
        if left <= point[0] <= right and top <= point[1] <= bottom:
            matches.append(button)
    if len(matches) != 1:
        raise FlowError(f"Expected one central retry button, found {len(matches)}")
    button = matches[0]
    box = button.get("bbox")
    if (not _confident(button.get("confidence")) or not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(type(value) in (int, float) and math.isfinite(value) for value in box)):
        raise FlowError("Retry button lacks reliable visual evidence")
    x, y = button["point"]
    if not (left <= box[0] <= x <= box[2] <= right and top <= box[1] <= y <= box[3] <= bottom
            and box[0] < box[2] and box[1] < box[3]):
        raise FlowError("Retry button is outside the central dialog")


def _gems(snapshot: SceneSnapshot) -> int | None:
    resources = snapshot.observations.get("resources")
    value = resources.get("gems") if isinstance(resources, dict) else None
    return value if type(value) is int and value >= 0 else None
