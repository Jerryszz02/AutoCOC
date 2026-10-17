"""A live screenshot/action session with evidence and bounded execution."""

from __future__ import annotations

from dataclasses import replace
import hashlib
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
        self._enrichable_capture: dict[str, object] | None = None
        self.frame_number = 0
        self.action_count = 0
        self._input_attempt_generation = 0
        self._active_resource_capacity: dict[str, object] | None = None
        self.logger = logger or logging.getLogger(__name__)
        self.client_version = "unknown"
        self.client_version_code: int | None = None
        self.vision_model_error: str | None = None
        self.prepared_building_model = None
        from .evidence import EvidenceBudget
        self.evidence_budget = (EvidenceBudget(directory / "frames", config.vision_agent.evidence_limit_mb * 1024 * 1024)
                                if config.vision_agent.enabled else None)

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
                                            timeout_sec=min(config.runtime.step_timeout_sec, remaining),
                                            png_compression_level=config.runtime.png_compression_level)
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
                                    prefer_raw=True, native=native,
                                    png_compression_level=config.runtime.png_compression_level)
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
            if getattr(config, "vision_agent", None) is not None and config.vision_agent.enabled:
                from .model_vision import BuildingModel, SceneModel, ModelUnavailable
                model_dir = Path(config.vision_agent.model_dir)
                if not config.vision_agent.model_dir:
                    session.vision_model_error = "Model directory is not configured"
                else:
                    if not model_dir.is_absolute():
                        model_dir = config.source_path.resolve().parent / model_dir
                    try:
                        scene_model = SceneModel(model_dir / "scene", client_version=session.client_version)
                        building_model = BuildingModel(model_dir / "building", client_version=session.client_version)
                        from .model_release import require_released_model
                        for model in (scene_model, building_model):
                            require_released_model(model)
                        session.recognizer.scene_model = scene_model
                        session.prepared_building_model = building_model
                    except (ModelUnavailable, OSError, ValueError) as exc:
                        session.vision_model_error = str(exc)
                        session.recognizer.scene_model = None
                        session.recognizer.building_model = None
                        session.prepared_building_model = None
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

    def observe(self, label: str = "observe", *, purpose: str = "full", slot: dict | None = None,
                _transient: bool = False) -> SceneSnapshot:
        self.check_deadline()
        previous = self.last_snapshot
        self.last_snapshot = None
        self._enrichable_capture = None
        self.frame_number += 1
        safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
        path = self.directory / "frames" / f"{self.frame_number:05d}-{safe_label}.png"
        captured_at = time.monotonic()
        artifact = (self.evidence_budget.capture(self.capture, path, transient=_transient) if self.evidence_budget is not None
                    else self.capture.capture_screenshot_artifact(path))
        requested_at = getattr(artifact, "requested_at_monotonic", None)
        if isinstance(requested_at, (int, float)) and math.isfinite(requested_at):
            captured_at = min(captured_at, requested_at)
        capture_binding = None
        if purpose == "line_boundary_wait":
            pixels_ready_at = getattr(artifact, "pixels_ready_at_monotonic", None)
            frame_id = getattr(artifact, "frame_id", None)
            requested_utc = getattr(artifact, "requested_at_utc", None)
            if (Path(artifact.path).resolve() != path.resolve()
                    or type(requested_at) not in (int, float) or not math.isfinite(requested_at)
                    or type(pixels_ready_at) not in (int, float) or not math.isfinite(pixels_ready_at)
                    or pixels_ready_at < requested_at or not isinstance(frame_id, str) or not frame_id
                    or not isinstance(requested_utc, str) or not requested_utc):
                raise FlowError("Boundary capture provenance is incomplete")
            with path.open("rb") as source:
                frame_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            capture_binding = {"path": path.resolve(), "sha256": frame_sha256,
                               "frame_id": frame_id, "captured_at": captured_at,
                               "requested_at": requested_at, "pixels_ready_at": pixels_ready_at,
                               "requested_utc": requested_utc, "frame_number": self.frame_number,
                               "action_count": self.action_count,
                               "input_attempt_generation": self._input_attempt_generation}
        self.context.screen_resolution = (artifact.width, artifact.height)
        recognition_started = time.monotonic()
        if purpose != "full":
            snapshot = self.recognizer.recognize_battle(path, purpose=purpose, slot=slot, previous=previous)
        else:
            if self.config.vision_agent.enabled and (
                    getattr(self.recognizer, "scene_model", None) is not None or
                    getattr(self.recognizer, "building_model", None) is not None):
                snapshot = self.recognizer.recognize(path, captured_at=captured_at)
            else:
                snapshot = self.recognizer.recognize(path)
        snapshot.observations["recognition_elapsed_sec"] = time.monotonic() - recognition_started
        snapshot.observations["observation_elapsed_sec"] = time.monotonic() - captured_at
        snapshot.observations["observed_at_monotonic"] = captured_at
        requested_at_utc = getattr(artifact, "requested_at_utc", None)
        if requested_at_utc is not None:
            # This is transport request start, not pixel rendering or log time.
            snapshot.observations["captured_at_utc"] = requested_at_utc
            snapshot.observations["capture_timestamp_basis"] = "capture_request_start"
        snapshot.observations["frame_id"] = getattr(artifact, "frame_id", None) or str(path)
        for name in ("requested_at_monotonic", "pixels_ready_at_monotonic",
                     "payload_ready_at_monotonic", "published_at_monotonic",
                     "pixels_elapsed_sec", "encode_elapsed_sec", "handoff_elapsed_sec"):
            value = getattr(artifact, name, None)
            if value is not None:
                snapshot.observations[name] = value
        snapshot.observations["capture_method"] = getattr(artifact, "capture_method", "unknown")
        capture_elapsed = getattr(artifact, "capture_elapsed_sec", None)
        snapshot.observations["capture_elapsed_sec"] = artifact.elapsed_sec if capture_elapsed is None else capture_elapsed
        self.last_snapshot = snapshot
        if self.evidence_budget is not None:
            snapshot.observations["evidence_retention"] = "transient" if _transient else "retained"
        self.event("observation", frame=str(path), scene=snapshot.scene, confidence=snapshot.confidence,
                   observations=snapshot.observations)
        if capture_binding is not None:
            capture_binding["snapshot"] = snapshot
            self._enrichable_capture = capture_binding
        self.logger.info("scene=%s confidence=%.2f capture=%.3fs recognition=%.3fs total=%.3fs frame=%s",
                         snapshot.scene, snapshot.confidence, snapshot.observations["capture_elapsed_sec"],
                         snapshot.observations["recognition_elapsed_sec"],
                         snapshot.observations["observation_elapsed_sec"], path.name)
        return snapshot

    def enrich_current_boundary_capture(self, snapshot: SceneSnapshot, *, expires_at: float) -> SceneSnapshot:
        """Fully recognize the latest immutable fast frame without another capture."""
        current = self.last_snapshot
        binding = self._enrichable_capture
        # An unsuccessful enrichment must never leave the fast frame actionable.
        self.last_snapshot = None
        self._enrichable_capture = None
        try:
            self.check_deadline()
            if (binding is None or current is not snapshot or binding.get("snapshot") is not snapshot
                    or snapshot.observations.get("recognition_scope") != "line_boundary_wait"
                    or snapshot.observations.get("fast_fallback") is not False
                    or type(expires_at) not in (int, float) or not math.isfinite(expires_at)
                    or self.frame_number != binding["frame_number"]
                    or self.action_count != binding["action_count"]
                    or self._input_attempt_generation != binding["input_attempt_generation"]
                    or snapshot.screenshot_path.resolve() != binding["path"]):
                raise FlowError("Boundary enrichment source is no longer current")
            for key, value in (("frame_id", binding["frame_id"]),
                               ("observed_at_monotonic", binding["captured_at"]),
                               ("requested_at_monotonic", binding["requested_at"]),
                               ("pixels_ready_at_monotonic", binding["pixels_ready_at"]),
                               ("captured_at_utc", binding["requested_utc"]),
                               ("capture_timestamp_basis", "capture_request_start")):
                if snapshot.observations.get(key) != value:
                    raise FlowError("Boundary enrichment capture metadata changed")
            deadline = min(expires_at, self.deadline, self.task_deadline,
                           binding["captured_at"] + 30)
            if time.monotonic() >= deadline:
                raise FlowError("Boundary enrichment deadline expired")

            def unchanged() -> bool:
                if (self.last_snapshot is not None or self.frame_number != binding["frame_number"]
                        or self.action_count != binding["action_count"]
                        or self._input_attempt_generation != binding["input_attempt_generation"]):
                    return False
                try:
                    with snapshot.screenshot_path.open("rb") as source:
                        return hashlib.file_digest(source, "sha256").hexdigest() == binding["sha256"]
                except OSError:
                    return False

            if not unchanged():
                raise FlowError("Boundary enrichment source pixels changed")
            started = time.monotonic()
            full = self.recognizer.recognize(snapshot.screenshot_path, captured_at=binding["captured_at"])
            self.check_deadline()
            if time.monotonic() >= deadline:
                raise FlowError("Boundary enrichment deadline expired")
            if (not unchanged() or full.screenshot_path.resolve() != binding["path"]
                    or full.scene != "enemy_village" or not math.isfinite(full.confidence)
                    or full.confidence < .8 or full.observations.get("recognition_scope", "full") != "full"
                    or full.observations.get("fast_fallback") is True):
                raise FlowError("Boundary enrichment full recognition is unverified")
            for key, value in (("frame_id", binding["frame_id"]),
                               ("observed_at_monotonic", binding["captured_at"]),
                               ("requested_at_monotonic", binding["requested_at"]),
                               ("pixels_ready_at_monotonic", binding["pixels_ready_at"]),
                               ("captured_at_utc", binding["requested_utc"]),
                               ("capture_timestamp_basis", "capture_request_start")):
                if key in full.observations and full.observations[key] != value:
                    raise FlowError("Boundary enrichment full output changed capture metadata")
            observations = dict(full.observations)
            for key in ("observed_at_monotonic", "captured_at_utc", "capture_timestamp_basis",
                        "frame_id", "requested_at_monotonic", "pixels_ready_at_monotonic",
                        "payload_ready_at_monotonic", "published_at_monotonic", "pixels_elapsed_sec",
                        "encode_elapsed_sec", "handoff_elapsed_sec", "capture_method",
                        "capture_elapsed_sec", "evidence_retention"):
                if key in snapshot.observations:
                    observations[key] = snapshot.observations[key]
            observations["recognition_scope"] = "full"
            observations["recognition_elapsed_sec"] = time.monotonic() - started
            observations["observation_elapsed_sec"] = time.monotonic() - binding["captured_at"]
            enriched = replace(full, observations=observations)
            self.event("recognition_enrichment", frame=str(enriched.screenshot_path),
                       frame_id=binding["frame_id"], frame_sha256=binding["sha256"],
                       observed_at_monotonic=binding["captured_at"],
                       requested_at_monotonic=binding["requested_at"],
                       pixels_ready_at_monotonic=binding["pixels_ready_at"],
                       action_count=binding["action_count"], scene=enriched.scene,
                       confidence=enriched.confidence, observations=enriched.observations)
            self.check_deadline()
            if time.monotonic() >= deadline or not unchanged():
                raise FlowError("Boundary enrichment expired before publication")
            self.last_snapshot = enriched
            return enriched
        except BaseException:
            self.last_snapshot = None
            raise

    def wait_for(self, scenes: set[str], *, timeout_sec: float = 30, label: str = "wait",
                 purpose: str = "full", poll_interval_sec: float | None = None) -> SceneSnapshot:
        deadline = min(time.monotonic() + timeout_sec, self.deadline, self.task_deadline)
        if self.evidence_budget is not None:
            try:
                return self._wait_for_bounded_evidence(scenes, deadline, label, purpose, poll_interval_sec)
            finally:
                if self.last_snapshot is not None:
                    self.evidence_budget.retain(self.last_snapshot.screenshot_path)
                    self.last_snapshot.observations["evidence_retention"] = "retained"
                    self.event("wait_evidence_retained", frame=str(self.last_snapshot.screenshot_path))
        while True:
            if time.monotonic() >= deadline:
                raise FlowError(f"Expected {sorted(scenes)} before deadline")
            snapshot = self.observe(label, purpose=purpose) if purpose != "full" else self.observe(label)
            if time.monotonic() >= deadline:
                raise FlowError(f"Expected {sorted(scenes)} before deadline; frame={snapshot.screenshot_path}")
            if snapshot.scene in scenes and math.isfinite(snapshot.confidence) and snapshot.confidence >= 0.8:
                return snapshot
            if snapshot.scene in {"maintenance", "disconnected", "account_in_use"}:
                raise FlowError(f"Game interruption: {snapshot.scene}")
            interval = self.config.runtime.poll_interval_sec if poll_interval_sec is None else poll_interval_sec
            next_poll = min(time.monotonic() + interval, deadline)
            while time.monotonic() < next_poll:
                self.check_deadline()
                time.sleep(min(.1, max(0, next_poll - time.monotonic())))

    def _wait_for_bounded_evidence(self, scenes: set[str], deadline: float, label: str,
                                   purpose: str, poll_interval_sec: float | None) -> SceneSnapshot:
        """Recycle this wait's intermediate polls; retain the result or last failure."""
        while True:
            if time.monotonic() >= deadline:
                raise FlowError(f"Expected {sorted(scenes)} before deadline")
            snapshot = self.observe(label, purpose=purpose, _transient=True)
            if time.monotonic() >= deadline:
                raise FlowError(f"Expected {sorted(scenes)} before deadline; frame={snapshot.screenshot_path}")
            if snapshot.scene in scenes and math.isfinite(snapshot.confidence) and snapshot.confidence >= .8:
                return snapshot
            if snapshot.scene in {"maintenance", "disconnected", "account_in_use"}:
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
        if (not math.isfinite(age) or age < 0 or age > 30
                or snapshot.scene in {"unknown", "account_in_use"}
                or not math.isfinite(snapshot.confidence) or snapshot.confidence < 0.8):
            raise FlowError("Cannot act on a stale or unrecognized screenshot")

    def tap(self, snapshot: SceneSnapshot, point: tuple[int, int] | list[int], *, reason: str) -> None:
        self._validate_snapshot(snapshot)
        x, y = point
        width, height = self.config.game.baseline_resolution
        if not (0 <= x < width and 0 <= y < height):
            raise FlowError(f"Click is outside the screenshot: {point}")
        self.event("tap", frame=str(snapshot.screenshot_path), scene=snapshot.scene, point=[x, y], reason=reason)
        self._input_attempt_generation += 1
        self.context.tap_xy(x, y, reason=reason)
        self.action_count += 1

    def open_resource_capacity_tooltip(self, snapshot: SceneSnapshot, resource: str,
                                       point: tuple[int, int]) -> None:
        """Record the one HOME input that may authorize a narrow unknown-scene close."""
        from .resource_capacity import _LAYOUT

        if (self.client_version != "18.647.2" or self.client_version_code != 180647003
                or resource not in _LAYOUT or point != _LAYOUT[resource][0]
                or snapshot.scene != "village"):
            raise FlowError("Resource tooltip open is not verified")
        self._active_resource_capacity = None
        self.tap(snapshot, point, reason=f"Open {resource} balance detail")
        self._active_resource_capacity = {"resource": resource,
                                          "input_generation": self._input_attempt_generation}

    def close_verified_resource_tooltip(self, snapshot: SceneSnapshot, resource: str,
                                        point: tuple[int, int], proof: dict) -> None:
        """Only dismiss this session's one source-bound, verified HOME HUD detail."""
        from .resource_capacity import _LAYOUT, verified_unknown_tooltip

        self.check_deadline()
        active = self._active_resource_capacity
        age = time.monotonic() - float(snapshot.observations.get("observed_at_monotonic", 0))
        if (self.client_version != "18.647.2" or self.client_version_code != 180647003
                or resource not in _LAYOUT or point != _LAYOUT[resource][0]
                or snapshot is not self.last_snapshot or snapshot.scene != "unknown"
                or not math.isfinite(age) or not 0 <= age <= 30
                or not isinstance(active, dict) or active.get("resource") != resource
                or active.get("input_generation") != self._input_attempt_generation
                or tuple(self.config.game.baseline_resolution) != (1280, 720)
                or self.capture.input_display_id is None or self.context.capture is not self.capture
                or tuple(self.context.screen_resolution) != tuple(snapshot.observations.get("source_resolution", ()))
                or not verified_unknown_tooltip(snapshot, resource, proof, self.recognizer.provider)):
            raise FlowError("Resource tooltip close proof is incomplete or stale")
        self.check_deadline()
        if foreground_package(self.capture.adb, self.capture.serial,
                              timeout_sec=self.config.runtime.step_timeout_sec) != self.config.game.package_name:
            raise FlowError("Game is not foreground before resource tooltip close")
        self.check_deadline()
        final_age = time.monotonic() - float(snapshot.observations.get("observed_at_monotonic", 0))
        if (snapshot is not self.last_snapshot or not math.isfinite(final_age)
                or not 0 <= final_age <= 30 or self._active_resource_capacity is not active
                or active.get("input_generation") != self._input_attempt_generation):
            raise FlowError("Resource tooltip close expired before input")
        self.event("verified_resource_tooltip_close_attempt", frame=str(snapshot.screenshot_path),
                   frame_sha256=proof["frame_sha256"], resource=resource, point=list(point),
                   input_attempt_generation=self._input_attempt_generation)
        self._active_resource_capacity = None
        self._input_attempt_generation += 1
        # Transport uncertainty must never leave the old unknown frame actionable.
        self.last_snapshot = None
        self.context.tap_xy(*point, reason=f"Close verified {resource} balance detail")
        self.action_count += 1

    def frozen_battle_tap(self, point: tuple[int, int], *, timeout_sec: float, reason: str,
                          plan_id: str) -> None:
        """One bounded input for an already frozen battle plan, without observation.

        Only prepared_battle calls this after validating its separate plan
        permission. An uncertain transport result is never retried here.
        """
        self.check_deadline()
        permission = getattr(self, "_active_frozen_plan", None)
        if (not isinstance(permission, tuple) or len(permission) != 3 or
                permission[0] != plan_id or permission[2] != tuple(point) or
                time.monotonic() >= permission[1]):
            raise FlowError("Frozen battle input has no active plan permission")
        x, y = point
        width, height = self.config.game.baseline_resolution
        if (type(x) is not int or type(y) is not int or not 0 <= x < width or
                not 0 <= y < height or not 0 < timeout_sec <= self.config.runtime.step_timeout_sec):
            raise FlowError("Frozen battle input has invalid coordinates or timeout")
        actual_x, actual_y = self.context._scale_point((x, y))
        source_width, source_height = self.context.screen_resolution
        if not (0 <= actual_x < source_width and 0 <= actual_y < source_height):
            raise FlowError("Frozen battle input scales outside the physical display")
        remaining = permission[1] - time.monotonic()
        if remaining <= 0:
            raise FlowError("Frozen battle input permission expired before send")
        timeout_sec = min(timeout_sec, remaining)
        self._input_attempt_generation += 1
        if not self.config.runtime.dry_run:
            if self.native is not None:
                self.native._request("tap", actual_x, actual_y, timeout_sec=timeout_sec)
            else:
                self.capture.adb.run(self.context._input_command("tap", str(actual_x), str(actual_y)),
                                     serial=self.capture.serial, timeout_sec=timeout_sec)
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
        self._input_attempt_generation += 1
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
        self._input_attempt_generation += 1
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
        self._input_attempt_generation += 1
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
        self._input_attempt_generation += 1
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
        self._input_attempt_generation += 1
        self.context.back()
        self.action_count += 1
