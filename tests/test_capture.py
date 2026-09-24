import struct
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autococ.capture import CaptureClient, png_size
from autococ.errors import AdbError, CaptureError


class CaptureTests(unittest.TestCase):
    def test_native_capture_receives_timeout_without_adb_fallback(self) -> None:
        native, adb = Mock(), Mock()
        path = Path("native.png")
        capture = CaptureClient(adb, "target", step_timeout_sec=4, native=native)
        self.assertIs(capture.capture_screenshot_artifact(path), native.screenshot.return_value)
        native.screenshot.assert_called_once_with(path, timeout_sec=4)
        native.screenshot.side_effect = CaptureError("native failed")
        with self.assertRaisesRegex(CaptureError, "native failed"):
            capture.capture_screenshot_artifact(path)
        adb.run_bytes.assert_not_called()

    def test_reads_png_dimensions(self) -> None:
        data = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 3, 2)

        self.assertEqual(png_size(data), (3, 2))

    def test_rejects_invalid_png(self) -> None:
        with self.assertRaises(CaptureError):
            png_size(b"nope")

    def test_previous_xml_cannot_disguise_failed_pull(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "window.xml"
            path.write_text("<hierarchy />", encoding="utf-8")
            capture = CaptureClient(Mock(), "target", step_timeout_sec=4)
            with self.assertRaisesRegex(CaptureError, "not written"):
                capture.capture_ui_xml(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "<hierarchy />")

    def test_xml_capture_replaces_artifact_only_after_success(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "window.xml"
            path.write_text("<old />", encoding="utf-8")
            adb = Mock()

            def command(args, **kwargs):
                self.assertEqual(kwargs["timeout_sec"], 4)
                if args[0] == "pull":
                    Path(args[-1]).write_text("<hierarchy />", encoding="utf-8")

            adb.run.side_effect = command
            capture = CaptureClient(adb, "target", step_timeout_sec=4)
            self.assertEqual(capture.capture_ui_xml(path), path)
            self.assertEqual(path.read_text(encoding="utf-8"), "<hierarchy />")

    def test_screenshot_uses_selected_physical_display(self) -> None:
        with TemporaryDirectory() as temp:
            adb = Mock()
            adb.run_bytes.return_value = (
                b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 3, 2)
            )
            capture = CaptureClient(adb, "target", screenshot_display_id="4619827767814508545")
            capture.capture_screenshot(Path(temp) / "screen.png")
            self.assertEqual(
                adb.run_bytes.call_args.args[0],
                ["exec-out", "screencap", "-p", "-d", "4619827767814508545"],
            )

    def test_screenshot_strips_only_verified_multi_display_warning(self) -> None:
        warning = (
            b"[Warning] Multiple displays were found, but no display id was specified! Defaulting to the first display found, however this default is not guaranteed to be consistent across captures. A display id should be specified.\n"
            b"A display ID can be specified with the [-d display-id] option.\n"
            b'See "dumpsys SurfaceFlinger --display-id" for valid display IDs.\n'
        )
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 3, 2)
        with TemporaryDirectory() as temp:
            adb = Mock()
            adb.run_bytes.return_value = warning + png
            capture = CaptureClient(adb, "target")
            path = Path(temp) / "screen.png"
            capture.capture_screenshot(path)
            self.assertEqual(path.read_bytes(), png)
            for prefix in (b"corrupt", b"[Warning] Unknown warning\n", warning + b"corrupt"):
                with self.subTest(prefix=prefix):
                    adb.run_bytes.return_value = prefix + png
                    with self.assertRaises(CaptureError):
                        capture.capture_screenshot(path)

    def test_raw_16_byte_header_preserves_rgba_channels_and_native_resolution_losslessly(self) -> None:
        import cv2
        import numpy as np

        rgba = np.array([[[255, 0, 0, 255], [0, 255, 0, 128]],
                         [[0, 0, 255, 255], [42, 73, 99, 17]]], dtype=np.uint8)
        with TemporaryDirectory() as temp:
            header = struct.pack("<IIII", 2, 2, 1, 0)
            adb = Mock()
            adb.run_bytes.return_value = header + rgba.tobytes()
            capture = CaptureClient(adb, "target", prefer_raw=True, screenshot_display_id="physical-two")
            result = capture.capture_screenshot_artifact(Path(temp) / "screen.png")
            decoded = cv2.imdecode(np.frombuffer(result.path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA), rgba)
            self.assertEqual((result.width, result.height), (2, 2))
            self.assertEqual(result.capture_method, "raw_rgba")
            self.assertEqual(result.capture_elapsed_sec, result.elapsed_sec)
            adb.run_bytes.assert_called_once_with(["exec-out", "screencap", "-d", "physical-two"],
                                                   serial="target", timeout_sec=10)

    def test_12_byte_header_and_16_byte_header_truncated_by_four_bytes_use_png(self) -> None:
        import cv2
        import numpy as np

        rgba = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8)
        raw_12 = struct.pack("<III", 2, 1, 1) + rgba.tobytes()
        truncated_16 = (struct.pack("<IIII", 2, 1, 1, 0) + rgba.tobytes())[:-4]
        _, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        png = encoded.tobytes()
        for raw in (raw_12, truncated_16):
            with self.subTest(raw=raw), TemporaryDirectory() as temp:
                path = Path(temp) / "screen.png"
                path.write_bytes(b"previous frame")
                adb = Mock()
                adb.run_bytes.side_effect = [raw, png]
                with patch("cv2.imencode") as encoder:
                    artifact = CaptureClient(adb, "target", prefer_raw=True).capture_screenshot_artifact(path)
                encoder.assert_not_called()
                self.assertEqual(artifact.capture_method, "png_fallback")
                self.assertEqual(adb.run_bytes.call_count, 2)
                self.assertEqual(path.read_bytes(), png)
                decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA), rgba)

    def test_truncated_16_byte_frame_with_failed_fallback_never_overwrites_original(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "screen.png"
            path.write_bytes(b"previous frame")
            adb = Mock()
            raw = struct.pack("<IIII", 2, 1, 1, 0) + bytes([255, 0, 0, 255, 0, 255, 0, 255])
            adb.run_bytes.side_effect = [raw[:-4], b"invalid PNG"]
            with patch("cv2.imencode") as encoder, self.assertRaises(CaptureError):
                CaptureClient(adb, "target", prefer_raw=True).capture_screenshot(path)
            encoder.assert_not_called()
            self.assertEqual(path.read_bytes(), b"previous frame")

    def test_default_capture_remains_png(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 3, 2)
        with TemporaryDirectory() as temp:
            adb = Mock()
            adb.run_bytes.return_value = png
            artifact = CaptureClient(adb, "target").capture_screenshot_artifact(Path(temp) / "screen.png")
            self.assertEqual(artifact.capture_method, "png")
            self.assertEqual(artifact.path.read_bytes(), png)
            adb.run_bytes.assert_called_once_with(["exec-out", "screencap", "-p"], serial="target", timeout_sec=10)

    def test_unsupported_or_incomplete_raw_uses_only_remaining_budget_for_png(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 3, 2)
        valid = struct.pack("<IIII", 2, 2, 1, 0) + bytes(16)
        malformed = [b"short", valid[:-1], valid + b"extra", struct.pack("<III", 2, 2, 2) + bytes(16),
                     struct.pack("<III", 0, 2, 1), struct.pack("<III", 8193, 2, 1),
                     struct.pack("<III", 8192, 8192, 1), struct.pack("<III", 2 ** 32 - 1, 2, 1)]
        for raw in malformed:
            with self.subTest(raw=raw[:12]), TemporaryDirectory() as temp:
                adb = Mock()
                adb.run_bytes.side_effect = [raw, png]
                capture = CaptureClient(adb, "target", step_timeout_sec=5, prefer_raw=True,
                                        screenshot_display_id="physical-two")
                with patch("autococ.capture.time.monotonic", side_effect=[100, 102, 103]):
                    result = capture.capture_screenshot_artifact(Path(temp) / "screen.png")
                self.assertEqual(result.capture_method, "png_fallback")
                self.assertEqual(result.capture_elapsed_sec, 3)
                self.assertEqual(adb.run_bytes.call_count, 2)
                self.assertEqual(adb.run_bytes.call_args_list[0].args[0], ["exec-out", "screencap", "-d", "physical-two"])
                self.assertEqual(adb.run_bytes.call_args.kwargs["timeout_sec"], 3)
                self.assertEqual(adb.run_bytes.call_args.args[0], ["exec-out", "screencap", "-p", "-d", "physical-two"])

    def test_exhausted_raw_budget_never_issues_png_call_and_preserves_old_file(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "screen.png"
            path.write_bytes(b"previous frame")
            adb = Mock()
            adb.run_bytes.return_value = (struct.pack("<IIII", 2, 2, 1, 0) + bytes(16))[:-4]
            capture = CaptureClient(adb, "target", step_timeout_sec=5, prefer_raw=True)
            with patch("autococ.capture.time.monotonic", side_effect=[100, 105]):
                with self.assertRaisesRegex(CaptureError, "budget exhausted"):
                    capture.capture_screenshot(path)
            adb.run_bytes.assert_called_once()
            self.assertEqual(path.read_bytes(), b"previous frame")

    def test_raw_transport_errors_never_trigger_fallback(self) -> None:
        for error in (AdbError("timed out"), OSError("ADB failed"), subprocess.TimeoutExpired("adb", 5)):
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as temp:
                path = Path(temp) / "screen.png"
                path.write_bytes(b"previous frame")
                adb = Mock()
                adb.run_bytes.side_effect = error
                with self.assertRaises(type(error)):
                    CaptureClient(adb, "target", prefer_raw=True).capture_screenshot(path)
                adb.run_bytes.assert_called_once()
                self.assertEqual(path.read_bytes(), b"previous frame")

    def test_failed_fallback_preserves_old_file(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "screen.png"
            path.write_bytes(b"previous frame")
            adb = Mock()
            adb.run_bytes.side_effect = [b"unsupported raw", b"invalid PNG"]
            with self.assertRaises(CaptureError):
                CaptureClient(adb, "target", prefer_raw=True).capture_screenshot(path)
            self.assertEqual(adb.run_bytes.call_count, 2)
            self.assertEqual(path.read_bytes(), b"previous frame")

    def test_partial_local_write_cannot_destroy_previous_screenshot(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "screen.png"
            path.write_bytes(b"previous frame")
            adb = Mock()
            adb.run_bytes.return_value = struct.pack("<IIII", 1, 1, 1, 0) + bytes([255, 0, 0, 255])
            original_write = Path.write_bytes

            def partial_write(target, data):
                original_write(target, data[:1])
                raise OSError("disk full")

            with patch.object(Path, "write_bytes", partial_write):
                with self.assertRaisesRegex(CaptureError, "Unable to save"):
                    CaptureClient(adb, "target", prefer_raw=True).capture_screenshot(path)
            self.assertEqual(path.read_bytes(), b"previous frame")


if __name__ == "__main__":
    unittest.main()
