"""Safe action helpers for automation flows."""

from __future__ import annotations

from pathlib import Path
import logging
import struct
import time
from typing import TYPE_CHECKING
import zlib

from .capture import CaptureClient
from .config import RuntimeConfig
from .errors import ActionError, AdbError, CaptureError, LocatorError
from .locator import LocatorResult, XMLLocator, find_template, scale_point
from .rules import ActionPlan, ActionStep

if TYPE_CHECKING:
    from .mumu import MuMuClient


class AutomationContext:
    def __init__(
        self,
        capture: CaptureClient,
        runtime: RuntimeConfig,
        *,
        baseline_resolution: tuple[int, int] = (1280, 720),
        screen_resolution: tuple[int, int] | None = None,
        logger: logging.Logger | None = None,
        replay_xml_path: str | Path | None = None,
        replay_screenshot_path: str | Path | None = None,
        native: MuMuClient | None = None,
    ) -> None:
        self.capture = capture
        self.native = native
        self.runtime = runtime
        self.baseline_resolution = baseline_resolution
        self.screen_resolution = screen_resolution or baseline_resolution
        self.logger = logger or logging.getLogger(__name__)
        self.replay_xml_path = Path(replay_xml_path) if replay_xml_path is not None else None
        self.replay_screenshot_path = Path(replay_screenshot_path) if replay_screenshot_path is not None else None
        self.runtime.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.report_dir.mkdir(parents=True, exist_ok=True)

    def tap(self, locator: dict[str, object]) -> LocatorResult:
        result = self.locate(locator)
        x, y = result.point
        self.logger.info("tap via %s at %s,%s", result.method, x, y)
        if not self.runtime.dry_run:
            if self.native is not None:
                self.native.tap(x, y)
            else:
                self.capture.adb.run(
                    self._input_command("tap", str(x), str(y)),
                    serial=self.capture.serial,
                    timeout_sec=self.runtime.step_timeout_sec,
                )
        return result

    def tap_xy(self, x: int, y: int, *, reason: str) -> None:
        if not reason.strip():
            raise ActionError("tap_xy requires an explicit reason")
        actual_x, actual_y = self._scale_point((x, y))
        self.logger.info(
            "tap_xy at %s,%s scaled_to=%s,%s; reason=%s",
            x,
            y,
            actual_x,
            actual_y,
            reason,
        )
        if not self.runtime.dry_run:
            if self.native is not None:
                self.native.tap(actual_x, actual_y)
            else:
                self.capture.adb.run(
                    self._input_command("tap", str(actual_x), str(actual_y)),
                    serial=self.capture.serial,
                    timeout_sec=self.runtime.step_timeout_sec,
                )

    def input_text(self, text: str) -> None:
        self.logger.info("input_text length=%s", len(text))
        if not self.runtime.dry_run:
            self.capture.adb.run(
                self._input_command("text", _escape_adb_text(text)),
                serial=self.capture.serial,
                timeout_sec=self.runtime.step_timeout_sec,
            )

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300) -> None:
        if duration_ms <= 0:
            raise ActionError("duration_ms must be greater than 0")
        actual_start = self._scale_point((start_x, start_y))
        actual_end = self._scale_point((end_x, end_y))
        self.logger.info(
            "swipe from %s,%s to %s,%s scaled_to=%s,%s->%s,%s",
            start_x,
            start_y,
            end_x,
            end_y,
            actual_start[0],
            actual_start[1],
            actual_end[0],
            actual_end[1],
        )
        if not self.runtime.dry_run:
            if self.native is not None:
                self.native.swipe(*actual_start, *actual_end, duration_ms)
                return
            self.capture.adb.run(
                self._input_command(
                    "swipe",
                    str(actual_start[0]),
                    str(actual_start[1]),
                    str(actual_end[0]),
                    str(actual_end[1]),
                    str(duration_ms),
                ),
                serial=self.capture.serial,
                timeout_sec=self.runtime.step_timeout_sec,
            )

    def wait(self, seconds: float) -> None:
        if seconds < 0:
            raise ActionError("seconds must be greater than or equal to 0")
        self.logger.info("wait %.2fs", seconds)
        if not self.runtime.dry_run:
            time.sleep(seconds)

    def wait_until(self, locator: dict[str, object], *, timeout_sec: int | None = None, poll_sec: float = 0.5) -> LocatorResult:
        if self.runtime.dry_run:
            return self.locate(locator)
        deadline = time.monotonic() + (timeout_sec or self.runtime.step_timeout_sec)
        last_error: Exception | None = None
        while time.monotonic() <= deadline:
            try:
                return self.locate(locator)
            except LocatorError as exc:
                last_error = exc
                time.sleep(poll_sec)
        raise LocatorError(f"Timed out waiting for locator {locator!r}: {last_error}")

    def assert_exists(self, locator: dict[str, object]) -> LocatorResult:
        result = self.locate(locator)
        self.logger.info("assert_exists via %s", result.method)
        return result

    def screenshot(self, name: str) -> Path:
        path = self.runtime.screenshot_dir / f"{_safe_name(name)}.png"
        self.logger.info("capture screenshot %s", path)
        if self.runtime.dry_run:
            _write_placeholder_png(path, self.screen_resolution[0], self.screen_resolution[1])
            return path
        artifact = self.capture.capture_screenshot_artifact(path)
        self.screen_resolution = (artifact.width, artifact.height)
        self.logger.info(
            "screenshot dimensions=%sx%s elapsed=%.3fs",
            artifact.width,
            artifact.height,
            artifact.elapsed_sec,
        )
        return artifact.path

    def back(self) -> None:
        self.logger.info("android back")
        if not self.runtime.dry_run:
            if self.native is not None:
                self.native.back()
                return
            self.capture.adb.run(
                self._input_command("keyevent", "4"),
                serial=self.capture.serial,
                timeout_sec=self.runtime.step_timeout_sec,
            )

    def locate(self, locator: dict[str, object]) -> LocatorResult:
        template_path = locator.get("template_path")
        xml_keys = {"resource-id", "content-desc", "text", "class", "bounds"}
        if xml_keys.intersection(locator):
            try:
                if self.runtime.dry_run:
                    if self.replay_xml_path is None:
                        raise LocatorError("Dry-run XML location requires replay_xml_path evidence")
                    xml_path = self.replay_xml_path
                else:
                    xml_path = self.runtime.report_dir / "window-current.xml"
                    self.capture.capture_ui_xml(xml_path)
                result = XMLLocator.from_file(xml_path).find(locator)
                if result:
                    self.logger.info("located by %s at %s", result.method, result.point)
                    return result
            except (AdbError, CaptureError, LocatorError, OSError) as exc:
                if not template_path:
                    raise LocatorError(f"Unable to locate element from UI XML: {exc}") from exc
                self.logger.debug("XML unavailable; using template fallback: %s", exc)

        if template_path:
            if self.runtime.dry_run:
                if self.replay_screenshot_path is None:
                    raise LocatorError("Dry-run template location requires replay_screenshot_path evidence")
                screenshot_path = self.replay_screenshot_path
            else:
                screenshot_path = self.runtime.screenshot_dir / "template-current.png"
                self.capture.capture_screenshot(screenshot_path)
            threshold = float(locator.get("threshold", 0.8))
            roi = locator.get("roi")
            return find_template(
                screenshot_path,
                str(template_path),
                threshold=threshold,
                roi=roi if isinstance(roi, tuple) else None,
                roi_base_resolution=self.baseline_resolution,
            )

        raise LocatorError(f"Unable to locate element safely: {locator!r}")

    def execute_plan(self, plan: ActionPlan) -> str | None:
        self.logger.info("execute plan: %s", plan.reason)
        for step in plan.actions:
            stop_reason = self.execute_step(step)
            if stop_reason:
                return stop_reason
        return None

    def execute_step(self, step: ActionStep) -> str | None:
        self.logger.info("execute step kind=%s reason=%s", step.kind, step.reason)
        if step.kind == "tap":
            if len(step.args) != 1 or not isinstance(step.args[0], dict):
                raise ActionError("tap step requires a locator dictionary")
            self.tap(step.args[0])
        elif step.kind == "tap_xy":
            if len(step.args) != 2:
                raise ActionError("tap_xy step requires x and y")
            self.tap_xy(int(step.args[0]), int(step.args[1]), reason=step.reason)
        elif step.kind == "swipe":
            if len(step.args) not in {4, 5}:
                raise ActionError("swipe step requires start and end coordinates")
            duration = int(step.args[4]) if len(step.args) == 5 else 300
            self.swipe(int(step.args[0]), int(step.args[1]), int(step.args[2]), int(step.args[3]), duration)
        elif step.kind == "wait":
            if len(step.args) != 1:
                raise ActionError("wait step requires seconds")
            self.wait(float(step.args[0]))
        elif step.kind == "screenshot":
            if len(step.args) != 1:
                raise ActionError("screenshot step requires a name")
            self.screenshot(str(step.args[0]))
        elif step.kind == "back":
            self.back()
        elif step.kind == "stop":
            return step.reason
        else:
            raise ActionError(f"Unsupported action step kind: {step.kind}")
        return None

    def _scale_point(self, point: tuple[int, int]) -> tuple[int, int]:
        return scale_point(
            point,
            from_resolution=self.baseline_resolution,
            to_resolution=self.screen_resolution,
        )

    def _input_command(self, *args: str) -> list[str]:
        command = ["shell", "input"]
        if self.capture.input_display_id is not None:
            command.extend(["-d", str(self.capture.input_display_id)])
        return command + list(args)


def _escape_adb_text(text: str) -> str:
    return text.replace(" ", "%s")


def _safe_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in name.strip())
    return safe or "screenshot"


def _write_placeholder_png(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_row = b"\x00" + (b"\xff\xff\xff" * width)
    compressed = zlib.compress(raw_row * height)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)
