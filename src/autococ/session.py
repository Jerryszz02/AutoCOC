"""A live screenshot/action session with evidence and bounded execution."""

from __future__ import annotations

from dataclasses import replace
import json
import logging
import math
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
            session.event("session_connected", launch_requested=launch,
                          launch_issued=launch and not already_foreground,
                          reused_foreground=already_foreground,
                          transport="mumu_native" if native is not None else "adb",
                          logical_display_id=target.logical_id, physical_display_id=target.physical_id)
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
        if snapshot.scene not in {"training", "request", "donation", "clan_chat", "search", "popup"}:
            raise FlowError(f"Back is not permitted in {snapshot.scene}")
        self.event("back", frame=str(snapshot.screenshot_path), reason=reason)
        self.context.back()
        self.action_count += 1
