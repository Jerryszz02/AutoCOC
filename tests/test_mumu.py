import ctypes
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autococ.errors import DeviceConnectionError, FlowError, MuMuDisplayUnavailable
from autococ.mumu import MuMuClient, _Renderer, _worker, verify_instance


class MuMuInstanceTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manager = self.root / "nx_main/MuMuManager.exe"
        self.dll = self.root / "nx_device/15.0/shell/sdk/external_renderer_ipc.dll"
        for path in (self.manager, self.dll):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.info = {"error_code": 0, "index": "1", "is_android_started": True,
                     "adb_host_ip": "127.0.0.1", "adb_port": 16416, "android_version": "15.0"}

    def verify(self, *, version="6.6.4.0"):
        responses = [self.info, {"version": version}]
        with patch("autococ.mumu.subprocess.run", side_effect=[Mock(stdout=json.dumps(item).encode())
                                                               for item in responses]) as command:
            result = verify_instance(self.root, 1, "127.0.0.1:16416", timeout_sec=4)
            self.assertEqual(command.call_args_list[0].args[0][-3:], ["info", "--vmindex", "1"])
            self.assertEqual(command.call_args.kwargs["timeout"], 4)
            return result

    def test_verified_instance_selects_the_matching_android_sdk(self):
        self.assertEqual(self.verify(), self.dll)

    def test_wrong_endpoint_instance_or_stopped_android_is_rejected(self):
        for key, value in (("adb_port", 16384), ("index", "0"), ("is_android_started", False), ("error_code", 1)):
            with self.subTest(key=key):
                old = self.info[key]
                self.info[key] = value
                with self.assertRaisesRegex(DeviceConnectionError, "does not match"):
                    self.verify()
                self.info[key] = old

    def test_unsupported_or_unreadable_input_version_is_rejected(self):
        for version in ("6.3.1.9", "6.3", "bad", "6.6.4.0-extra", None):
            with self.subTest(version=version), self.assertRaises(DeviceConnectionError):
                self.verify(version=version)


class MuMuRendererTests(unittest.TestCase):
    def setUp(self):
        self.lib = Mock()
        self.lib.nemu_connect.return_value = 7
        self.lib.nemu_get_display_id.return_value = 2
        for name in ("nemu_input_event_finger_touch_down", "nemu_input_event_finger_touch_up",
                     "nemu_input_event_key_down", "nemu_input_event_key_up"):
            getattr(self.lib, name).return_value = 0
        self.dimensions = (4, 3)
        self.error = 0

        def capture(handle, display, length, width, height, pixels):
            ctypes.cast(width, ctypes.POINTER(ctypes.c_int))[0] = self.dimensions[0]
            ctypes.cast(height, ctypes.POINTER(ctypes.c_int))[0] = self.dimensions[1]
            if length and self.error == 0:
                for i in range(length):
                    pixels[i] = (i * 13) % 256
            return self.error

        self.lib.nemu_capture_display.side_effect = capture

    def renderer(self):
        with patch("autococ.mumu.ctypes.CDLL", return_value=self.lib):
            return _Renderer("fixture.dll", "root", 1, "target.game", 2)

    def test_capture_flips_rgba_once_and_reuses_verified_dimensions(self):
        import cv2
        import numpy as np

        renderer = self.renderer()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "native.png"
            self.assertEqual(renderer.capture(str(path)), {"width": 4, "height": 3})
            rgba = np.array([(i * 13) % 256 for i in range(48)], dtype=np.uint8).reshape(3, 4, 4)
            np.testing.assert_array_equal(cv2.imread(str(path)), rgba[::-1, :, :3][:, :, ::-1])
        self.assertEqual(self.lib.nemu_capture_display.call_count, 2)
        renderer.close()
        renderer.close()
        self.lib.nemu_disconnect.assert_called_once_with(7)

    def test_missing_or_changed_display_never_falls_back_or_clicks(self):
        renderer = self.renderer()
        for display in (-1, 0, 3):
            with self.subTest(display=display):
                self.lib.nemu_get_display_id.return_value = display
                with self.assertRaisesRegex(FlowError, "absent or changed"):
                    renderer.tap(1, 2)
        self.lib.nemu_input_event_finger_touch_down.assert_not_called()

    def test_invalid_dimensions_disconnect_and_do_not_allocate(self):
        for dimensions in ((0, 3), (9000, 3), (8192, 8192)):
            with self.subTest(dimensions=dimensions):
                self.dimensions = dimensions
                with self.assertRaisesRegex(FlowError, "dimensions"):
                    self.renderer()
        self.assertEqual(self.lib.nemu_disconnect.call_count, 3)

    def test_initial_frame_error_can_wait_but_closes_its_sdk_handle(self):
        self.error = 4
        with self.assertRaisesRegex(MuMuDisplayUnavailable, "initial screenshot unavailable; SDK code 4"):
            self.renderer()
        self.lib.nemu_disconnect.assert_called_once_with(7)
        self.lib.nemu_input_event_finger_touch_down.assert_not_called()

    def test_same_sdk_error_during_later_capture_is_terminal(self):
        renderer = self.renderer()
        self.error = 4
        with TemporaryDirectory() as temp:
            path = Path(temp) / "screen.png"
            with self.assertRaisesRegex(FlowError, "code 4") as failure:
                renderer.capture(str(path))
            self.assertNotIsInstance(failure.exception, MuMuDisplayUnavailable)
            self.assertFalse(path.exists())

    def test_changed_resolution_or_sdk_error_never_publishes_a_png(self):
        renderer = self.renderer()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "native.png"
            self.dimensions = (3, 4)
            with self.assertRaisesRegex(FlowError, "resolution changed"):
                renderer.capture(str(path))
            self.dimensions, self.error = (4, 3), 5
            with self.assertRaisesRegex(FlowError, "code 5"):
                renderer.capture(str(path))
            self.assertFalse(path.exists())

    def test_touch_uses_unrotated_coordinates_and_always_releases_after_error(self):
        renderer = self.renderer()
        with patch("autococ.mumu.time.sleep"):
            renderer.tap(3, 2)
        self.lib.nemu_input_event_finger_touch_down.assert_called_once_with(7, 2, 1, 3, 2)
        self.lib.nemu_input_event_finger_touch_down.return_value = 5
        with self.assertRaisesRegex(FlowError, "code 5"):
            renderer.tap(1, 2)
        self.assertEqual(self.lib.nemu_input_event_finger_touch_down.call_count, 2)
        self.assertEqual(self.lib.nemu_input_event_finger_touch_up.call_count, 2)

    def test_invalid_touch_and_duration_send_no_input(self):
        renderer = self.renderer()
        for point in ((-1, 0), (4, 0), (0, 3), (True, 0)):
            with self.subTest(point=point), self.assertRaises(FlowError):
                renderer.tap(*point)
        for duration in (0, 5001, True):
            with self.subTest(duration=duration), self.assertRaises(FlowError):
                renderer.swipe(0, 0, 3, 2, duration)
        self.lib.nemu_input_event_finger_touch_down.assert_not_called()

    def test_swipe_reaches_both_endpoints_and_back_maps_to_linux_code(self):
        renderer = self.renderer()
        with patch("autococ.mumu.time.monotonic", side_effect=[0, .3, .6]), patch("autococ.mumu.time.sleep"):
            renderer.swipe(0, 0, 3, 2, 600)
            renderer.back()
        calls = self.lib.nemu_input_event_finger_touch_down.call_args_list
        self.assertEqual(calls[0].args, (7, 2, 1, 0, 0))
        self.assertEqual(calls[-1].args, (7, 2, 1, 3, 2))
        self.lib.nemu_input_event_finger_touch_up.assert_called_once_with(7, 2, 1)
        self.lib.nemu_input_event_key_down.assert_called_once_with(7, 2, 158)
        self.lib.nemu_input_event_key_up.assert_called_once_with(7, 2, 158)


class MuMuWorkerTests(unittest.TestCase):
    def client(self):
        client = MuMuClient.__new__(MuMuClient)
        client.connection, client.process = Mock(), Mock()
        client.timeout_sec, client.closed = 4, False
        return client

    def test_timeout_terminates_worker_and_never_retries_input(self):
        client = self.client()
        client.connection.poll.return_value = False
        with self.assertRaisesRegex(FlowError, "timed out"):
            client.tap(10, 20)
        self.assertEqual(client.connection.send.call_args_list[0].args, (("tap", (10, 20)),))
        self.assertEqual(len([call for call in client.connection.send.call_args_list if call.args[0][0] == "tap"]), 1)
        client.process.terminate.assert_called_once_with()
        with self.assertRaisesRegex(FlowError, "closed"):
            client.tap(10, 20)

    def test_timed_out_capture_preserves_previous_artifact(self):
        client = self.client()
        client.connection.poll.return_value = False
        with TemporaryDirectory() as temp:
            path = Path(temp) / "screen.png"
            path.write_bytes(b"previous evidence")
            with self.assertRaisesRegex(FlowError, "timed out"):
                client.screenshot(path, timeout_sec=4)
            self.assertEqual(path.read_bytes(), b"previous evidence")
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_worker_death_is_reported_without_replaying(self):
        client = self.client()
        client.connection.poll.return_value = True
        client.connection.recv.side_effect = EOFError
        with self.assertRaisesRegex(FlowError, "disconnected"):
            client.back()
        self.assertTrue(client.closed)

    def test_initial_missing_display_is_typed_but_never_replays_client_operations(self):
        client = self.client()
        client.connection.recv.return_value = {"error": "display loading",
                                               "error_type": "initial_display_unavailable"}
        with self.assertRaises(MuMuDisplayUnavailable):
            client._receive(1)
        client.connection.send.assert_not_called()
        with self.assertRaises(MuMuDisplayUnavailable):
            client.tap(1, 2)
        self.assertTrue(client.closed)
        self.assertEqual(sum(c.args[0][0] == "tap" for c in client.connection.send.call_args_list), 1)

    def test_only_initial_display_failure_is_marked_retryable(self):
        for during_initialization in (True, False):
            with self.subTest(during_initialization=during_initialization):
                connection = Mock()
                renderer = Mock()
                renderer.capture.side_effect = MuMuDisplayUnavailable("changed")
                connection.recv.return_value = ("capture", ("frame.png",))
                with patch("autococ.mumu.os.dup2"), patch("autococ.mumu._Renderer",
                        side_effect=MuMuDisplayUnavailable("loading") if during_initialization else None,
                        return_value=renderer):
                    _worker(connection, "dll", "root", 1, "game", 2)
                response = connection.send.call_args.args[0]
                self.assertEqual(response.get("error_type"),
                                 "initial_display_unavailable" if during_initialization else None)
                if during_initialization:
                    connection.recv.assert_not_called()
                else:
                    renderer.close.assert_called_once_with()
