"""Screenshot and UI XML capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import time
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

from .adb import ADBClient
from .errors import CaptureError

if TYPE_CHECKING:
    from .mumu import MuMuClient


_SCREENCAP_MULTI_DISPLAY_WARNING = (
    b"[Warning] Multiple displays were found, but no display id was specified! "
    b"Defaulting to the first display found, however this default is not guaranteed "
    b"to be consistent across captures. A display id should be specified.\n"
    b"A display ID can be specified with the [-d display-id] option.\n"
    b'See "dumpsys SurfaceFlinger --display-id" for valid display IDs.\n'
)


@dataclass(frozen=True)
class ScreenshotCapture:
    path: Path
    width: int
    height: int
    elapsed_sec: float
    capture_method: str = "unknown"
    capture_elapsed_sec: float | None = None


@dataclass(frozen=True)
class UICapture:
    path: Path
    elapsed_sec: float


class CaptureClient:
    def __init__(
        self,
        adb: ADBClient,
        serial: str,
        step_timeout_sec: int = 10,
        *,
        screenshot_display_id: str | None = None,
        input_display_id: int | None = None,
        prefer_raw: bool = False,
        native: MuMuClient | None = None,
    ) -> None:
        self.adb = adb
        self.serial = serial
        self.step_timeout_sec = step_timeout_sec
        self.screenshot_display_id = screenshot_display_id
        self.input_display_id = input_display_id
        self.prefer_raw = prefer_raw
        self.native = native

    def capture_ui_xml(self, output_path: str | Path) -> Path:
        return self.capture_ui_xml_artifact(output_path).path

    def capture_ui_xml_artifact(self, output_path: str | Path) -> UICapture:
        started = time.monotonic()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.adb.run(
            ["shell", "uiautomator", "dump", "/sdcard/window.xml"],
            serial=self.serial,
            timeout_sec=self.step_timeout_sec,
        )
        try:
            with TemporaryDirectory(prefix=".capture-", dir=path.parent) as temp_dir:
                fresh_path = Path(temp_dir) / "window.xml"
                self.adb.run(
                    ["pull", "/sdcard/window.xml", str(fresh_path)],
                    serial=self.serial,
                    timeout_sec=self.step_timeout_sec,
                )
                if not fresh_path.exists() or fresh_path.stat().st_size == 0:
                    raise CaptureError(f"UI XML was not written: {path}")
                ET.parse(fresh_path)
                fresh_path.replace(path)
        except (OSError, ET.ParseError) as exc:
            raise CaptureError(f"Unable to save valid UI XML at {path}: {exc}") from exc
        return UICapture(path=path, elapsed_sec=time.monotonic() - started)

    def capture_screenshot(self, output_path: str | Path) -> Path:
        return self.capture_screenshot_artifact(output_path).path

    def capture_screenshot_artifact(self, output_path: str | Path) -> ScreenshotCapture:
        if self.native is not None:
            return self.native.screenshot(Path(output_path), timeout_sec=self.step_timeout_sec)
        started = time.monotonic()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["exec-out", "screencap"]
        if self.screenshot_display_id is not None:
            command.extend(["-d", self.screenshot_display_id])
        converted = None
        method = "png"
        timeout = self.step_timeout_sec
        if self.prefer_raw:
            raw = self.adb.run_bytes(command, serial=self.serial, timeout_sec=timeout)
            if raw.startswith(_SCREENCAP_MULTI_DISPLAY_WARNING):
                raw = raw[len(_SCREENCAP_MULTI_DISPLAY_WARNING):]
            converted = _raw_rgba_png(raw)
            if converted is None:
                timeout = self.step_timeout_sec - (time.monotonic() - started)
                if timeout <= 0:
                    raise CaptureError("Screenshot timeout budget exhausted before PNG fallback")
                method = "png_fallback"
            else:
                method = "raw_rgba"
        if converted is not None:
            data, width, height = converted
        else:
            png_command = [*command[:2], "-p", *command[2:]]
            data = self.adb.run_bytes(png_command, serial=self.serial, timeout_sec=timeout)
            if data.startswith(_SCREENCAP_MULTI_DISPLAY_WARNING):
                data = data[len(_SCREENCAP_MULTI_DISPLAY_WARNING):]
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise CaptureError("ADB screencap did not return PNG data")
            width, height = png_size(data)
        try:
            with TemporaryDirectory(prefix=".capture-", dir=path.parent) as temp_dir:
                fresh_path = Path(temp_dir) / "screen.png"
                fresh_path.write_bytes(data)
                fresh_path.replace(path)
        except OSError as exc:
            raise CaptureError(f"Unable to save screenshot at {path}: {exc}") from exc
        elapsed = time.monotonic() - started
        return ScreenshotCapture(
            path=path,
            width=width,
            height=height,
            elapsed_sec=elapsed,
            capture_method=method,
            capture_elapsed_sec=elapsed,
        )


def _raw_rgba_png(data: bytes) -> tuple[bytes, int, int] | None:
    """Accept only the complete 16-byte-header RGBA_8888 format verified locally."""
    if len(data) < 16:
        return None
    width, height, pixel_format = struct.unpack_from("<III", data)
    if pixel_format != 1 or not (0 < width <= 8192 and 0 < height <= 8192) or width * height > 33554432:
        return None
    # A truncated 16-byte-header frame can look exactly like a 12-byte-header
    # frame. Refuse the older layout rather than shift every pixel by one.
    if len(data) != 16 + width * height * 4:
        return None
    import cv2
    import numpy as np

    pixels = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(height, width, 4)
    try:
        bgra = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGRA)
        success, encoded = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    except cv2.error as exc:
        raise CaptureError("Unable to encode raw RGBA screenshot as PNG") from exc
    if not success:
        raise CaptureError("Unable to encode raw RGBA screenshot as PNG")
    return encoded.tobytes(), width, height


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CaptureError("Invalid PNG data")
    width, height = struct.unpack(">II", data[16:24])
    if data[8:16] != b"\x00\x00\x00\rIHDR":
        raise CaptureError("PNG is missing its IHDR header")
    if width <= 0 or height <= 0:
        raise CaptureError("PNG dimensions must be greater than 0")
    return (width, height)
