"""A live screenshot/action session with evidence and bounded execution."""

from __future__ import annotations

from dataclasses import replace
import json
import logging
import math
import re
from pathlib import Path
from threading import Event
import time

from .actions import AutomationContext
from .adb import ADBClient
from .capture import CaptureClient
from .config import AppConfig
from .device import foreground_package, resolve_game_display
from .errors import DeviceConnectionError, FlowError, MuMuDisplayUnavailable, StopRequested
from .locator import find_template
from .mumu import MuMuClient
from .scene import SceneSnapshot
from .vision import ScreenshotRecognizer


class GameSession:
    def __init__(self, config: AppConfig, capture: CaptureClient, recognizer: ScreenshotRecognizer,
                 directory: Path, *, logger: logging.Logger | None = None, native: MuMuClient | None = None,
                 stop_event: Event | None = None) -> None:
        self.config = config
        self.capture = capture
        self.native = native
        self.stop_event = stop_event
        self.recognizer = recognizer
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.events_path = directory / "events.jsonl"
        runtime = replace(config.runtime, screenshot_dir=directory / "frames", report_dir=directory)
        self.context = AutomationContext(capture, runtime, baseline_resolution=config.game.baseline_resolution,
                                         logger=logger, native=native)
        self.deadline = time.monotonic() + config.stop.max_duration_sec
        self.task_deadline = self.deadline
        self.last_snapshot: SceneSnapshot | None = None
        self.frame_number = 0
        self.action_count = 0
        self.logger = logger or logging.getLogger(__name__)
        self.client_version = "unknown"
        self.client_version_code: int | None = None

    @classmethod
    def connect(cls, config: AppConfig, adb: ADBClient, serial: str, directory: Path,
                *, launch: bool, logger: logging.Logger | None = None,
                stop_event: Event | None = None) -> "GameSession":
        def check_stop() -> None:
            if stop_event is not None and stop_event.is_set():
                raise StopRequested("interrupted by user")

        check_stop()
        started = time.monotonic()
        already_foreground = launch and foreground_package(
            adb, serial, timeout_sec=config.runtime.step_timeout_sec) == config.game.package_name
        if launch and not already_foreground:
            check_stop()
            activity = config.game.launch_activity.strip()
            if activity:
                component = activity if "/" in activity else f"{config.game.package_name}/{activity}"
                adb.run(["shell", "am", "start", "-n", component], serial=serial,
                        timeout_sec=config.game.startup_timeout_sec)
            else:
                adb.run(["shell", "monkey", "-p", config.game.package_name,
                         "-c", "android.intent.category.LAUNCHER", "1"], serial=serial,
                        timeout_sec=config.game.startup_timeout_sec)
        deadline = min(started + config.stop.max_duration_sec, time.monotonic() + (
            config.game.startup_timeout_sec if launch else config.runtime.step_timeout_sec))
        native = None
        try:
            pending_error = None
            while True:
                check_stop()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {pending_error}" if pending_error is not None else ""
                    raise DeviceConnectionError("Game display connection deadline exceeded" + detail) from pending_error
                pending_error = None
                try:
                    target = resolve_game_display(adb, serial, config.game.package_name,
                                                   display_id=config.game.display_id,
                                                   timeout_sec=min(config.runtime.step_timeout_sec, remaining))
                except DeviceConnectionError as exc:
                    pending_error = exc
                if pending_error is None and config.mumu is not None:
                    check_stop()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DeviceConnectionError("Game display connection deadline exceeded")
                    try:
                        native = MuMuClient(config.mumu.install_dir, config.mumu.instance_index, serial,
                                            config.game.package_name, target.logical_id,
                                            timeout_sec=min(config.runtime.step_timeout_sec, remaining))
                    except MuMuDisplayUnavailable as exc:
                        pending_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise pending_error or DeviceConnectionError("Game display connection deadline exceeded")
                if pending_error is None:
                    break
                time.sleep(min(1, remaining))
            capture = CaptureClient(adb, serial, config.runtime.step_timeout_sec,
                                    screenshot_display_id=target.physical_id, input_display_id=target.logical_id,
                                    prefer_raw=True, native=native)
            session = cls(config, capture, ScreenshotRecognizer(config.ocr, config.game.baseline_resolution),
                          directory, logger=logger, native=native, stop_event=stop_event)
            check_stop()
            session.deadline = min(session.deadline, started + config.stop.max_duration_sec)
            session.task_deadline = session.deadline
            # Read-only package metadata is provenance, never a connection
            # prerequisite. Do not persist the raw package dump.
            try:
                metadata = adb.run(["shell", "dumpsys", "package", config.game.package_name],
                                   serial=serial, timeout_sec=min(5, config.runtime.step_timeout_sec),
                                   check=False)
                if metadata.returncode == 0:
                    name = re.search(r"(?m)^\s*versionName=([^\s]+)", metadata.stdout)
                    code = re.search(r"(?m)^\s*versionCode=(\d+)", metadata.stdout)
                    if name:
                        session.client_version = name.group(1)
                    if code:
                        session.client_version_code = int(code.group(1))
            except Exception:
                pass
            session.recognizer.client_version = session.client_version
            session.event("session_connected", launch_requested=launch,
                          launch_issued=launch and not already_foreground,
                          reused_foreground=already_foreground,
                          transport="mumu_native" if native is not None else "adb",
                          logical_display_id=target.logical_id, physical_display_id=target.physical_id,
                          client_version=session.client_version,
                          client_version_code=session.client_version_code)
            return session
        except BaseException:
            if native is not None:
                native.close()
            raise

    def close(self) -> None:
        if self.native is not None:
            self.native.close()

    def begin_task(self) -> None:
        self.task_deadline = min(self.deadline, time.monotonic() + self.config.runtime.task_timeout_sec)
        self.check_deadline()

    def check_deadline(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise StopRequested("interrupted by user")
        if time.monotonic() >= min(self.deadline, self.task_deadline):
            raise FlowError("Session or task deadline exceeded")

    def event(self, kind: str, **data: object) -> None:
        with self.events_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"time": time.time(), "kind": kind, **data},
                                    ensure_ascii=False, default=str) + "\n")

    def observe(self, label: str = "observe", *, purpose: str = "full", slot: dict | None = None) -> SceneSnapshot:
        self.check_deadline()
        previous = self.last_snapshot
        self.last_snapshot = None
        self.frame_number += 1
        safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
        path = self.directory / "frames" / f"{self.frame_number:05d}-{safe_label}.png"
        captured_at = time.monotonic()
        artifact = self.capture.capture_screenshot_artifact(path)
        self.context.screen_resolution = (artifact.width, artifact.height)
        recognition_started = time.monotonic()
        if purpose != "full":
            snapshot = self.recognizer.recognize_battle(path, purpose=purpose, slot=slot, previous=previous)
        else:
            snapshot = self.recognizer.recognize(path)
        snapshot.observations["recognition_elapsed_sec"] = time.monotonic() - recognition_started
        snapshot.observations["observation_elapsed_sec"] = time.monotonic() - captured_at
        snapshot.observations["observed_at_monotonic"] = captured_at
        snapshot.observations["capture_method"] = getattr(artifact, "capture_method", "unknown")
        capture_elapsed = getattr(artifact, "capture_elapsed_sec", None)
        snapshot.observations["capture_elapsed_sec"] = artifact.elapsed_sec if capture_elapsed is None else capture_elapsed
        self.last_snapshot = snapshot
        self.event("observation", frame=str(path), scene=snapshot.scene, confidence=snapshot.confidence,
                   observations=snapshot.observations)
        self.logger.info("scene=%s confidence=%.2f capture=%.3fs recognition=%.3fs total=%.3fs frame=%s",
                         snapshot.scene, snapshot.confidence, snapshot.observations["capture_elapsed_sec"],
                         snapshot.observations["recognition_elapsed_sec"],
                         snapshot.observations["observation_elapsed_sec"], path.name)
        return snapshot

    def wait_for(self, scenes: set[str], *, timeout_sec: float = 30, label: str = "wait",
                 purpose: str = "full", poll_interval_sec: float | None = None) -> SceneSnapshot:
        deadline = min(time.monotonic() + timeout_sec, self.deadline, self.task_deadline)
        while True:
            if time.monotonic() >= deadline:
                raise FlowError(f"Expected {sorted(scenes)} before deadline")
            snapshot = self.observe(label, purpose=purpose) if purpose != "full" else self.observe(label)
            if time.monotonic() >= deadline:
                raise FlowError(f"Expected {sorted(scenes)} before deadline; frame={snapshot.screenshot_path}")
            if snapshot.scene in scenes and math.isfinite(snapshot.confidence) and snapshot.confidence >= 0.8:
                return snapshot
            if snapshot.scene in {"maintenance", "disconnected"}:
                raise FlowError(f"Game interruption: {snapshot.scene}")
            interval = self.config.runtime.poll_interval_sec if poll_interval_sec is None else poll_interval_sec
            next_poll = min(time.monotonic() + interval, deadline)
            while time.monotonic() < next_poll:
                self.check_deadline()
                time.sleep(min(.1, max(0, next_poll - time.monotonic())))

    def _validate_snapshot(self, snapshot: SceneSnapshot) -> None:
        self.check_deadline()
        if snapshot is not self.last_snapshot:
            raise FlowError("Action refers to an obsolete observation")
        age = time.monotonic() - float(snapshot.observations.get("observed_at_monotonic", 0))
        if not math.isfinite(age) or age < 0 or age > 30 or snapshot.scene == "unknown" or not math.isfinite(snapshot.confidence) or snapshot.confidence < 0.8:
            raise FlowError("Cannot act on a stale or unrecognized screenshot")

    def tap(self, snapshot: SceneSnapshot, point: tuple[int, int] | list[int], *, reason: str) -> None:
        self._validate_snapshot(snapshot)
        x, y = point
        width, height = self.config.game.baseline_resolution
        if not (0 <= x < width and 0 <= y < height):
            raise FlowError(f"Click is outside the screenshot: {point}")
        self.event("tap", frame=str(snapshot.screenshot_path), scene=snapshot.scene, point=[x, y], reason=reason)
        self.context.tap_xy(x, y, reason=reason)
        self.action_count += 1

    def swipe(self, snapshot: SceneSnapshot, start: tuple[int, int], end: tuple[int, int], *,
              duration_ms: int = 500, reason: str) -> None:
        """Scroll older clan messages without touching navigation or chat input."""
        self._validate_snapshot(snapshot)
        if snapshot.scene != "clan_chat":
            raise FlowError(f"Chat swipe is not permitted in {snapshot.scene}")
        for point in (start, end):
            if (len(point) != 2 or any(type(value) is not int for value in point)
                    or not (120 <= point[0] <= 240 and 150 <= point[1] <= 625)):
                raise FlowError("Swipe must remain inside the clan message area")
        if start[0] != end[0] or end[1] - start[1] < 200:
            raise FlowError("Chat swipe must move vertically toward older messages")
        if type(duration_ms) is not int or not 300 <= duration_ms <= 1000:
            raise FlowError("Chat swipe duration must be between 300 and 1000 ms")
        self.event("swipe", frame=str(snapshot.screenshot_path), scene=snapshot.scene,
                   start=list(start), end=list(end), duration_ms=duration_ms, reason=reason)
        self.context.swipe(*start, *end, duration_ms)
        self.action_count += 1

    def swipe_battle_bar(self, snapshot: SceneSnapshot, *, direction: str, reason: str) -> None:
        """Scroll only the positively observed deployment toolbar.

        Left/right describe the finger gesture. No old card coordinate remains
        actionable after a swipe; the caller must capture the new viewport.
        """
        self._validate_snapshot(snapshot)
        if (snapshot.scene not in {"enemy_village", "battle"} or direction not in {"left", "right"}
                or tuple(self.config.game.baseline_resolution) != (1280, 720)):
            raise FlowError("Battle toolbar scroll requires a recognized battle and supported layout")
        battle = snapshot.observations.get("battle")
        slots = battle.get("slots", []) if isinstance(battle, dict) else []
        anchors = []
        for card in slots:
            if not isinstance(card, dict) or card.get("kind") not in {"troop", "spell", "hero", "siege"}:
                continue
            box, point = card.get("bbox"), card.get("point")
            evidence = card.get("evidence")
            if (not isinstance(box, (tuple, list)) or len(box) != 4
                    or not isinstance(point, (tuple, list)) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in (*box, *point))
                    or not (0 <= box[0] < point[0] < box[2] <= 1280
                            and 575 <= box[1] < point[1] < box[3] <= 720)
                    or not (40 <= box[2] - box[0] <= 180 and 70 <= box[3] - box[1] <= 145)
                    or not isinstance(evidence, dict) or evidence.get("method") != "independent_card_border"):
                continue
            identity = card.get("confidence")
            reading = evidence.get("count")
            quantity = reading.get("confidence") if isinstance(reading, dict) else None
            known_identity = bool(card.get("unit_id")) and type(identity) in (int, float) and .9 <= identity <= 1
            known_count = (type(card.get("count")) is int and card["count"] >= 0
                           and type(quantity) in (int, float) and .9 <= quantity <= 1)
            if known_identity or known_count:
                anchors.append(box)
        if len(anchors) < 2 or len({round(box[0]) for box in anchors}) < 2:
            raise FlowError("Battle toolbar lacks independent current-frame card anchors")
        top, bottom = max(box[1] for box in anchors), min(box[3] for box in anchors)
        if bottom - top < 60:
            raise FlowError("Battle toolbar cards do not share one confirmed row")
        y = round((top + bottom) / 2)
        start, end = ((1060, y), (220, y)) if direction == "left" else ((220, y), (1060, y))
        self.event("swipe_battle_bar", frame=str(snapshot.screenshot_path), scene=snapshot.scene,
                   start=list(start), end=list(end), direction=direction, reason=reason)
        self.context.swipe(*start, *end, 550)
        self.action_count += 1
        self.last_snapshot = None

    def swipe_army_catalog(self, snapshot: SceneSnapshot, *, direction: str, reason: str) -> None:
        """Page the open troop/spell picker, after validating its current frame.

        The gesture stays inside the card grid and never reaches the add/remove
        controls above it or the edge arrow. A new observation is required for
        each page; callers cannot reuse an old editor position.
        """
        self._validate_snapshot(snapshot)
        if snapshot.scene != "training" or direction not in {"next", "previous"}:
            raise FlowError("Army catalog swipe requires an observed training editor and direction")
        texts = snapshot.observations.get("ocr") or []
        if not any("编辑军队配置" in str(item.get("text", ""))
                   and 450 <= (item.get("bbox") or [0])[0] <= 750 for item in texts):
            raise FlowError("Army configuration editor title is not visible")
        import cv2
        import numpy as np
        image = cv2.imread(str(snapshot.screenshot_path))
        if image is None:
            raise FlowError("Army catalog frame cannot be decoded")
        height, width = image.shape[:2]
        x1, x2 = round(250 * width / 1280), round(1100 * width / 1280)
        y1, y2 = round(480 * height / 720), round(555 * height / 720)
        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        # The expanded picker has saturated blue or red/gold super-troop
        # cards here; a collapsed picker leaves the dark panel in this area.
        card_fraction = float(np.mean((hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 125)))
        if card_fraction < 0.22:
            raise FlowError("Expanded army card catalog is not visible")
        start, end = ((980, 500), (330, 500)) if direction == "next" else ((330, 500), (980, 500))
        self.event("swipe_army_catalog", frame=str(snapshot.screenshot_path),
                   scene=snapshot.scene, start=list(start), end=list(end),
                   direction=direction, reason=reason)
        self.context.swipe(*start, *end, 550)
        self.action_count += 1

    def close_army_catalog(self, snapshot: SceneSnapshot, *, reason: str) -> None:
        """Dismiss the open army picker without editing a card or the preset."""
        self.check_deadline()
        if snapshot is not self.last_snapshot:
            raise FlowError("Army picker close refers to an obsolete observation")
        age = time.monotonic() - float(snapshot.observations.get("observed_at_monotonic", 0))
        if not 0 <= age <= 30 or snapshot.scene not in {"training", "unknown"}:
            raise FlowError("Army picker close requires a fresh editor frame")
        texts = snapshot.observations.get("ocr") or []
        has_capacity = any(re.fullmatch(r"\d+/\d+", str(x.get("text", "")))
                           and 450 <= (x.get("bbox") or [0])[0] <= 750
                           and 50 <= (x.get("bbox") or [0, 0])[1] <= 200 for x in texts)
        picker_numbers = sum(bool(re.fullmatch(r"\d{1,2}", str(x.get("text", ""))))
                             and (x.get("bbox") or [0, 0])[1] >= 480 for x in texts)
        if not has_capacity or picker_numbers < 8:
            raise FlowError("Expanded army picker cannot be confirmed in this frame")
        self.event("close_army_catalog", frame=str(snapshot.screenshot_path), reason=reason)
        self.context.back()
        self.action_count += 1

    @staticmethod
    def buttons(snapshot: SceneSnapshot, name: str) -> list[dict]:
        return [button for button in snapshot.observations.get("buttons", []) if button["name"] == name]

    def click(self, snapshot: SceneSnapshot, name: str, *, region: tuple[int, int, int, int] | None = None) -> None:
        matches = self.buttons(snapshot, name)
        if region is not None:
            left, top, right, bottom = region
            matches = [b for b in matches if left <= b["point"][0] <= right and top <= b["point"][1] <= bottom]
        if len(matches) != 1:
            raise FlowError(f"Expected one {name!r} button, found {len(matches)}")
        self.tap(snapshot, matches[0]["point"], reason=f"Recognized {name}: {matches[0]['text']}")

    def click_template(self, snapshot: SceneSnapshot, name: str, *, roi: tuple[int, int, int, int]) -> None:
        self._validate_snapshot(snapshot)
        result = find_template(snapshot.screenshot_path, self.config.vision.template_dir / f"{name}.png",
                               threshold=0.88, roi=roi, roi_base_resolution=self.config.game.baseline_resolution)
        source_width, source_height = self.context.screen_resolution
        width, height = self.config.game.baseline_resolution
        point = (int(result.point[0] * width / source_width), int(result.point[1] * height / source_height))
        self.tap(snapshot, point, reason=f"Matched {name} template, confidence={result.confidence:.3f}")

    def back(self, snapshot: SceneSnapshot, *, reason: str) -> None:
        self._validate_snapshot(snapshot)
        if snapshot.scene not in {"training", "request", "donation", "clan_chat", "search", "popup", "goal_panel"}:
            raise FlowError(f"Back is not permitted in {snapshot.scene}")
        self.event("back", frame=str(snapshot.screenshot_path), reason=reason)
        self.context.back()
        self.action_count += 1
