"""MuMu renderer transport isolated so a blocked DLL cannot block the runner.

ABI and RGBA orientation: MAA Controller/MumuExtras.cpp (dev-v2).
Use the modern finger API: coordinates are already in screenshot space.
"""

from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from .capture import _validated_png_compression_level
from .errors import DeviceConnectionError, FlowError, MuMuDisplayUnavailable


def verify_instance(root: Path, index: int, serial: str, *, timeout_sec: float) -> Path:
    manager = root / "nx_main/MuMuManager.exe"
    if not manager.is_file():
        raise DeviceConnectionError(f"MuMu manager not found: {manager}")

    def query(*args: str) -> dict:
        try:
            result = subprocess.run([str(manager), *args], cwd=manager.parent, capture_output=True,
                                    timeout=timeout_sec, check=True,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise DeviceConnectionError("Unable to verify MuMu instance/version") from exc

    info = query("info", "--vmindex", str(index))
    if (not isinstance(info, dict) or info.get("error_code") != 0
            or str(info.get("index")) != str(index) or info.get("is_android_started") is not True
            or f"{info.get('adb_host_ip')}:{info.get('adb_port')}" != serial):
        raise DeviceConnectionError("Configured MuMu instance does not match the selected ready ADB device")
    version_info = query("version")
    version = version_info.get("version", "") if isinstance(version_info, dict) else ""
    parts = version.split(".") if isinstance(version, str) else []
    if not 2 <= len(parts) <= 4 or any(not value.isascii() or not value.isdigit() for value in parts):
        raise DeviceConnectionError("Invalid MuMu manager version")
    if tuple(map(int, parts)) + (0,) * (4 - len(parts)) < (6, 3, 2, 0):
        raise DeviceConnectionError("MuMu native input requires manager version >= 6.3.2.0")
    android = info.get("android_version")
    if android not in {"12.0", "15.0"}:
        raise DeviceConnectionError("Unsupported MuMu Android engine")
    for relative in (f"nx_device/{android}/shell/sdk/external_renderer_ipc.dll",
                     "nx_main/sdk/external_renderer_ipc.dll"):
        path = root / relative
        if path.is_file():
            return path
    raise DeviceConnectionError("MuMu renderer SDK not found")


class MuMuClient:
    def __init__(self, root: Path, index: int, serial: str, package: str, display: int,
                 *, timeout_sec: float = 10, png_compression_level: int = 1) -> None:
        png_compression_level = _validated_png_compression_level(png_compression_level)
        root = root.resolve()
        dll = verify_instance(root, index, serial, timeout_sec=timeout_sec)
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker,
                                       args=(child, str(dll), str(root), index, package, display,
                                             png_compression_level),
                                       daemon=True, name="AutoCOC-MuMu")
        self.closed = False
        self.timeout_sec = timeout_sec
        try:
            self.process.start()
            child.close()
            self._receive(timeout_sec)
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout_sec: float):
        if not self.connection.poll(timeout_sec):
            raise FlowError("MuMu operation timed out; result uncertain, no input will be retried")
        try:
            response = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise FlowError("MuMu worker disconnected; no input will be retried") from exc
        if "error" in response:
            if response.get("error_type") == "initial_display_unavailable":
                raise MuMuDisplayUnavailable(response["error"])
            raise FlowError(response["error"])
        return response

    def _request(self, operation: str, *args, timeout_sec: float | None = None):
        if self.closed:
            raise FlowError("MuMu transport is closed")
        try:
            self.connection.send((operation, args))
            return self._receive(self.timeout_sec if timeout_sec is None else timeout_sec)
        except (EOFError, OSError) as exc:
            self.close()
            raise FlowError("MuMu worker disconnected; no input will be retried") from exc
        except BaseException:
            self.close()
            raise

    def screenshot(self, output_path: Path, *, timeout_sec: float):
        from .capture import ScreenshotCapture, png_size
        from .errors import CaptureError

        started = time.monotonic()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The parent owns publication: a late worker cannot overwrite the
        # requested artifact after a timeout or reuse yesterday's screenshot.
        with TemporaryDirectory(prefix=".mumu-", dir=path.parent) as temporary:
            fresh = Path(temporary) / "screen.png"
            response = self._request("capture", str(fresh.resolve()), timeout_sec=timeout_sec)
            received_at = time.monotonic()
            if not fresh.is_file():
                raise FlowError("MuMu worker returned without a fresh screenshot")
            try:
                with fresh.open("rb") as source:
                    dimensions = png_size(source.read(24))
                    source.seek(-12, 2)
                    ending = source.read(12)
            except (OSError, CaptureError) as exc:
                raise FlowError("MuMu worker returned an invalid screenshot") from exc
            if ending != b"\x00\x00\x00\x00IEND\xaeB`\x82":
                raise FlowError("MuMu worker returned an incomplete screenshot")
            if (type(response.get("width")) is not int or type(response.get("height")) is not int or
                    dimensions != (response["width"], response["height"])):
                raise FlowError("MuMu screenshot dimensions do not match worker response")
            pixels_ready = response.get("pixels_ready_at_monotonic", received_at)
            encode_elapsed = response.get("encode_elapsed_sec")
            if (type(pixels_ready) not in (int, float) or not started <= pixels_ready <= received_at or
                    (encode_elapsed is not None and
                     (type(encode_elapsed) not in (int, float) or not 0 <= encode_elapsed <= received_at - pixels_ready))):
                raise FlowError("MuMu screenshot timing is invalid")
            fresh.replace(path)
        elapsed = time.monotonic() - started
        pixel_elapsed = max(0.0, pixels_ready - started)
        return ScreenshotCapture(
            path, response["width"], response["height"], elapsed, "mumu_native", elapsed,
            requested_at_monotonic=started,
            pixels_ready_at_monotonic=pixels_ready,
            payload_ready_at_monotonic=pixels_ready + encode_elapsed if encode_elapsed is not None else None,
            published_at_monotonic=started + elapsed,
            pixels_elapsed_sec=pixel_elapsed,
            encode_elapsed_sec=encode_elapsed,
            handoff_elapsed_sec=max(0.0, elapsed - pixel_elapsed - (encode_elapsed or 0.0)),
            frame_id=uuid4().hex,
        )

    def tap(self, x: int, y: int) -> None:
        self._request("tap", x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        self._request("swipe", x1, y1, x2, y2, duration_ms)

    def back(self) -> None:
        self._request("back")

    def zoom_out(self) -> None:
        self._request("zoom_out")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                self.connection.send(("close", ()))
                self.process.join(.5)
        except (EOFError, OSError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(1)
            self.connection.close()


class _Renderer:
    def __init__(self, dll: str, root: str, index: int, package: str, display: int,
                 png_compression_level: int = 1) -> None:
        self.png_compression_level = _validated_png_compression_level(png_compression_level)
        self.lib = ctypes.CDLL(dll)
        self.package, self.display = package.encode("utf-8"), display
        self.handle = 0
        for name, args, result in (
            ("nemu_connect", [ctypes.c_wchar_p, ctypes.c_int], ctypes.c_int),
            ("nemu_disconnect", [ctypes.c_int], None),
            ("nemu_get_display_id", [ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
            ("nemu_capture_display", [ctypes.c_int, ctypes.c_uint, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_ubyte)], ctypes.c_int),
            ("nemu_input_event_finger_touch_down", [ctypes.c_int] * 5, ctypes.c_int),
            ("nemu_input_event_finger_touch_up", [ctypes.c_int] * 3, ctypes.c_int),
            ("nemu_input_event_key_down", [ctypes.c_int] * 3, ctypes.c_int),
            ("nemu_input_event_key_up", [ctypes.c_int] * 3, ctypes.c_int),
        ):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.handle = self.lib.nemu_connect(root, index)
        if not self.handle:
            raise FlowError("MuMu renderer connection failed")
        try:
            self.check_display()
            width, height = ctypes.c_int(), ctypes.c_int()
            code = self.lib.nemu_capture_display(self.handle, display, 0, ctypes.byref(width),
                                                 ctypes.byref(height), None)
            if code != 0:
                # During cold startup the SDK can expose the display before a
                # first frame exists. Retain the code, without guessing its meaning.
                raise MuMuDisplayUnavailable(f"MuMu initial screenshot unavailable; SDK code {code}")
            self.width, self.height = width.value, height.value
            if not (0 < self.width <= 8192 and 0 < self.height <= 8192 and self.width * self.height <= 33554432):
                raise FlowError("Invalid MuMu screenshot dimensions")
            self.pixels = (ctypes.c_ubyte * (self.width * self.height * 4))()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def checked(code: int) -> None:
        if code != 0:
            raise FlowError(f"MuMu SDK operation failed with code {code}")

    def check_display(self) -> None:
        current = self.lib.nemu_get_display_id(self.handle, self.package, 0)
        if current < 0 or current != self.display:
            raise MuMuDisplayUnavailable("MuMu game display is absent or changed; reconnect before further actions")

    def capture(self, output_path: str) -> dict:
        import cv2
        import numpy as np

        self.check_display()
        pixels_started = time.monotonic()
        width, height = ctypes.c_int(self.width), ctypes.c_int(self.height)
        self.checked(self.lib.nemu_capture_display(self.handle, self.display, len(self.pixels),
                                                   ctypes.byref(width), ctypes.byref(height), self.pixels))
        if (width.value, height.value) != (self.width, self.height):
            raise FlowError("MuMu resolution changed during capture")
        pixels_ready = time.monotonic()
        rgba = np.ctypeslib.as_array(self.pixels).reshape(self.height, self.width, 4)
        bgr = cv2.flip(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), 0)
        okay, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression_level])
        if not okay:
            raise FlowError("Unable to encode MuMu screenshot")
        encoded_at = time.monotonic()
        Path(output_path).write_bytes(encoded.tobytes())
        return {"width": self.width, "height": self.height,
                "pixels_started_at_monotonic": pixels_started,
                "pixels_ready_at_monotonic": pixels_ready,
                "encode_elapsed_sec": encoded_at - pixels_ready}

    def point(self, x: int, y: int) -> None:
        if type(x) is not int or type(y) is not int or not (0 <= x < self.width and 0 <= y < self.height):
            raise FlowError("Native touch point is outside the captured screen")

    def down(self, x: int, y: int) -> None:
        self.checked(self.lib.nemu_input_event_finger_touch_down(self.handle, self.display, 1, x, y))

    def tap(self, x: int, y: int) -> dict:
        self.check_display()
        self.point(x, y)
        try:
            self.down(x, y)
            time.sleep(.06)
        finally:
            self.checked(self.lib.nemu_input_event_finger_touch_up(self.handle, self.display, 1))
        return {}

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> dict:
        self.check_display()
        self.point(x1, y1)
        self.point(x2, y2)
        if type(duration_ms) is not int or not 1 <= duration_ms <= 5000:
            raise FlowError("Native swipe duration must be between 1 and 5000 ms")
        started, duration = time.monotonic(), duration_ms / 1000
        try:
            self.down(x1, y1)
            while (elapsed := time.monotonic() - started) < duration:
                fraction = elapsed / duration
                self.down(round(x1 + (x2-x1) * fraction), round(y1 + (y2-y1) * fraction))
                time.sleep(min(.02, duration-elapsed))
            self.down(x2, y2)
        finally:
            self.checked(self.lib.nemu_input_event_finger_touch_up(self.handle, self.display, 1))
        return {}

    def back(self) -> dict:
        self.check_display()
        # Android KEYCODE_BACK (4) is Linux KEY_BACK (158) in this SDK.
        try:
            self.checked(self.lib.nemu_input_event_key_down(self.handle, self.display, 158))
            time.sleep(.06)
        finally:
            self.checked(self.lib.nemu_input_event_key_up(self.handle, self.display, 158))
        return {}

    def zoom_out(self) -> dict:
        """Pinch both fingers toward the map center using the native pointer IDs."""
        self.check_display()
        paths = [(round(x1 * self.width / 1280), round(350 * self.height / 720),
                  round(x2 * self.width / 1280)) for x1, x2 in ((320, 610), (960, 670))]
        for x1, y, x2 in paths:
            self.point(x1, y)
            self.point(x2, y)
        try:
            for finger, (x1, y, _) in enumerate(paths, 1):
                self.checked(self.lib.nemu_input_event_finger_touch_down(self.handle, self.display, finger, x1, y))
            time.sleep(.35)
            started = time.monotonic()
            while True:
                fraction = min((time.monotonic() - started) / 1.2, 1)
                for finger, (x1, y, x2) in enumerate(paths, 1):
                    self.checked(self.lib.nemu_input_event_finger_touch_down(
                        self.handle, self.display, finger, round(x1 + (x2 - x1) * fraction), y))
                if fraction == 1:
                    break
                time.sleep(.02)
            time.sleep(.15)
        finally:
            codes = [self.lib.nemu_input_event_finger_touch_up(self.handle, self.display, finger)
                     for finger in (1, 2)]
            for code in codes:
                self.checked(code)
        return {}

    def close(self) -> None:
        if self.handle:
            self.lib.nemu_disconnect(self.handle)
            self.handle = 0


def _worker(connection, dll: str, root: str, index: int, package: str, display: int,
            png_compression_level: int = 1) -> None:
    # Vendor DLLs write instance identifiers directly to C stdout/stderr.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    renderer = None
    try:
        renderer = _Renderer(dll, root, index, package, display, png_compression_level)
        connection.send({"ready": True})
        while True:
            operation, args = connection.recv()
            if operation == "close":
                break
            if operation not in {"capture", "tap", "swipe", "back", "zoom_out"}:
                raise FlowError("Unknown MuMu operation")
            connection.send(getattr(renderer, operation)(*args))
    except Exception as exc:
        try:
            response = {"error": f"MuMu worker: {type(exc).__name__}: {exc}"}
            # Only a constructor failure precedes every capture/input operation.
            # A display change during an existing session remains terminal.
            if renderer is None and isinstance(exc, MuMuDisplayUnavailable):
                response["error_type"] = "initial_display_unavailable"
            connection.send(response)
        except (EOFError, OSError):
            pass
    finally:
        if renderer is not None:
            renderer.close()
        connection.close()
