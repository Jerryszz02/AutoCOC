from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autococ.adb import ADBResult, DeviceInfo
from autococ.config import load_config
from autococ.device import DeviceManager, DisplayTarget, parse_display_targets, resolve_game_display
from autococ.errors import AdbError, DeviceConnectionError


class DeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "config.toml"
        path.write_text("", encoding="utf-8")
        self.config = load_config(path)
        self.adb = Mock()
        self.adb.raw_devices_output.return_value = "List of devices attached\n"

    def manager(self, serial: str = "") -> DeviceManager:
        config = replace(self.config, adb=replace(self.config.adb, manual_serial=serial))
        with patch("autococ.device.ADBClient", return_value=self.adb):
            return DeviceManager(config)

    def test_manual_serial_wins_over_first_connected_device(self) -> None:
        self.adb.devices.return_value = [DeviceInfo("wrong", "device"), DeviceInfo("wanted", "device")]
        self.assertEqual(self.manager("wanted").connect().serial, "wanted")
        self.adb.connect.assert_not_called()

    def test_missing_manual_serial_never_falls_back(self) -> None:
        self.adb.devices.return_value = [DeviceInfo("wrong", "device")]
        with self.assertRaisesRegex(DeviceConnectionError, "wanted"):
            self.manager("wanted").connect()
        self.adb.connect.assert_not_called()

    def test_manual_network_target_is_tried_before_port_scanning(self) -> None:
        self.adb.devices.side_effect = [
            [DeviceInfo("wrong", "device")],
            [DeviceInfo("wrong", "device"), DeviceInfo("127.0.0.1:1234", "device")],
        ]
        self.assertEqual(self.manager("127.0.0.1:1234").connect().serial, "127.0.0.1:1234")
        self.adb.connect.assert_called_once_with("127.0.0.1:1234", timeout_sec=5)

    def test_offline_manual_target_is_not_selected(self) -> None:
        self.adb.devices.return_value = [DeviceInfo("wanted", "offline"), DeviceInfo("wrong", "device")]
        with self.assertRaises(DeviceConnectionError):
            self.manager("wanted").connect()

    def test_failed_scan_port_does_not_prevent_later_connection(self) -> None:
        self.adb.devices.side_effect = [[], [DeviceInfo("127.0.0.1:16416", "device")]]
        self.adb.connect.side_effect = [AdbError("timed out"), None, None, None, None]
        self.assertEqual(self.manager().connect().serial, "127.0.0.1:16416")

    def test_multiple_ready_targets_require_explicit_selection(self) -> None:
        self.adb.devices.return_value = [DeviceInfo("first", "device"), DeviceInfo("second", "device")]
        with self.assertRaisesRegex(DeviceConnectionError, "manual_serial"):
            self.manager().connect()
        self.adb.connect.assert_not_called()

    def test_multiple_targets_after_scanning_are_not_selected_by_port_order(self) -> None:
        self.adb.devices.side_effect = [[], [DeviceInfo("127.0.0.1:7555", "device"), DeviceInfo("127.0.0.1:16416", "device")]]
        with self.assertRaisesRegex(DeviceConnectionError, "Multiple ADB targets"):
            self.manager().connect()

    def test_one_ready_target_among_offline_targets_can_be_selected(self) -> None:
        self.adb.devices.return_value = [DeviceInfo("offline", "offline"), DeviceInfo("ready", "device")]
        self.assertEqual(self.manager().connect().serial, "ready")


DISPLAY_DUMP = """
  mViewports=[DisplayViewport{type=INTERNAL, valid=true, displayId=0, uniqueId='local:4619827820427265280'}, DisplayViewport{type=INTERNAL, valid=true, displayId=2, uniqueId='local:4619827767814508545'}]
  Display 0:
    mDisplayId=0
    mPrimaryDisplayDevice=内置屏幕(local:4619827820427265280)
  Display 2:
    mDisplayId=2
    mPrimaryDisplayDevice=mumuscreen001(local:4619827767814508545)
"""


class DisplayTests(unittest.TestCase):
    def test_parses_mumu_logical_to_physical_display_mapping(self) -> None:
        self.assertEqual(parse_display_targets(DISPLAY_DUMP), [
            DisplayTarget(0, "4619827820427265280"), DisplayTarget(2, "4619827767814508545"),
        ])

    def test_resolves_target_package_on_second_display(self) -> None:
        adb = Mock()
        adb.run.side_effect = [
            ADBResult((), 0, DISPLAY_DUMP, ""),
            ADBResult((), 0, """
  Display: mDisplayId=0 (organized)
    mCurrentFocus=Window{abc u0 app.lawnchair/.Main}
  Display: mDisplayId=2 (organized)
    mCurrentFocus=Window{def u0 com.supercell.clashofclans/.GameApp}
""", ""),
        ]
        target = resolve_game_display(adb, "serial", "com.supercell.clashofclans")
        self.assertEqual(target, DisplayTarget(2, "4619827767814508545"))

    def test_explicit_display_works_before_game_launch(self) -> None:
        adb = Mock()
        adb.run.return_value = ADBResult((), 0, DISPLAY_DUMP, "")
        self.assertEqual(resolve_game_display(adb, "serial", "game", display_id=0).logical_id, 0)
        self.assertEqual(adb.run.call_count, 1)

    def test_mumu_launch_redirect_uses_actual_display_not_requested_display(self) -> None:
        # Excerpt from activities-launched.txt after am start --display 0.
        activities = """
Display #2 (activities from top to bottom):
    topResumedActivity=ActivityRecord{31601111 u0 com.supercell.clashofclans/com.supercell.titan.GameApp t179}
Display #0 (activities from top to bottom):
      topResumedActivity=ActivityRecord{46952457 u0 app.lawnchair/.LawnchairLauncher t2}
Display #3 (activities from top to bottom):
  ResumedActivity: ActivityRecord{31601111 u0 com.supercell.clashofclans/com.supercell.titan.GameApp t179}
"""
        adb = Mock()
        adb.run.side_effect = [
            ADBResult((), 0, DISPLAY_DUMP, ""),
            ADBResult((), 0, "", ""),
            ADBResult((), 0, activities, ""),
        ]
        self.assertEqual(
            resolve_game_display(adb, "serial", "com.supercell.clashofclans"),
            DisplayTarget(2, "4619827767814508545"),
        )

    def test_absent_package_cannot_silently_choose_first_display(self) -> None:
        adb = Mock()
        adb.run.side_effect = [
            ADBResult((), 0, DISPLAY_DUMP, ""),
            ADBResult((), 0, "Display: mDisplayId=0\n  mCurrentFocus=null", ""),
            ADBResult((), 0, "Display #2\n  ResumedActivity: ActivityRecord{abc u0 game/.Main}", ""),
        ]
        with self.assertRaises(DeviceConnectionError):
            resolve_game_display(adb, "serial", "game")


if __name__ == "__main__":
    unittest.main()
