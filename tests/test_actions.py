from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autococ.actions import AutomationContext
from autococ.config import RuntimeConfig
from autococ.errors import CaptureError, LocatorError
from autococ.locator import LocatorResult


UI_XML = '<hierarchy><node text="Go" bounds="[10,20][30,40]" enabled="true" /></hierarchy>'
MATCH = LocatorResult(node=None, method="template", confidence=1.0, point=(30, 50), bounds=None)


class ActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = Mock()
        self.capture.serial = "target"
        self.capture.input_display_id = None

    def context(self, *, dry_run: bool = False, **kwargs) -> AutomationContext:
        runtime = RuntimeConfig(
            screenshot_dir=self.root / "screenshots", report_dir=self.root / "reports",
            step_timeout_sec=7, dry_run=dry_run,
        )
        return AutomationContext(self.capture, runtime, **kwargs)

    def test_all_input_actions_forward_timeout_and_device(self) -> None:
        context = self.context()
        with patch.object(context, "locate", return_value=MATCH):
            context.tap({"text": "Go"})
        context.tap_xy(3, 4, reason="test")
        context.input_text("hello world")
        context.swipe(1, 2, 3, 4)
        context.back()
        self.assertEqual(self.capture.adb.run.call_count, 5)
        for call in self.capture.adb.run.call_args_list:
            self.assertEqual(call.kwargs, {"serial": "target", "timeout_sec": 7})

    def test_native_inputs_scale_coordinates_once_and_text_stays_display_bound_adb(self) -> None:
        native = Mock()
        context = self.context(native=native, screen_resolution=(2560, 1440))
        context.tap_xy(3, 4, reason="test native")
        native.tap.assert_called_once_with(6, 8)
        context.swipe(1, 2, 3, 4, 600)
        native.swipe.assert_called_once_with(2, 4, 6, 8, 600)
        context.back()
        native.back.assert_called_once_with()
        self.capture.adb.run.assert_not_called()
        context.input_text("hello")
        self.assertEqual(self.capture.adb.run.call_args.args[0], ["shell", "input", "text", "hello"])

    def test_native_failure_is_not_replayed_through_adb(self) -> None:
        native = Mock()
        native.tap.side_effect = RuntimeError("uncertain result")
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            self.context(native=native).tap_xy(1, 2, reason="uncertain native tap")
        native.tap.assert_called_once_with(1, 2)
        self.capture.adb.run.assert_not_called()

    def test_native_dry_run_has_no_transport_calls(self) -> None:
        native = Mock()
        context = self.context(dry_run=True, native=native)
        context.tap_xy(1, 2, reason="dry-run")
        context.swipe(1, 2, 3, 4)
        context.back()
        self.assertEqual(native.mock_calls, [])
        self.capture.adb.run.assert_not_called()

    def test_dry_run_inputs_never_use_adb(self) -> None:
        xml = self.root / "replay.xml"
        xml.write_text(UI_XML, encoding="utf-8")
        context = self.context(dry_run=True, replay_xml_path=xml)
        self.assertEqual(context.tap({"text": "Go"}).point, (20, 30))
        context.tap_xy(3, 4, reason="test")
        context.input_text("hello")
        context.swipe(1, 2, 3, 4)
        context.back()
        self.assertEqual(self.capture.mock_calls, [])

    def test_all_input_actions_target_selected_logical_display(self) -> None:
        self.capture.input_display_id = 2
        context = self.context()
        with patch.object(context, "locate", return_value=MATCH):
            context.tap({"text": "Go"})
        context.tap_xy(3, 4, reason="test")
        context.input_text("hello")
        context.swipe(1, 2, 3, 4)
        context.back()
        for call in self.capture.adb.run.call_args_list:
            self.assertEqual(call.args[0][:4], ["shell", "input", "-d", "2"])

    def test_dry_run_without_replay_evidence_fails_before_adb(self) -> None:
        context = self.context(dry_run=True)
        for locator in ({"text": "Go"}, {"template_path": "go.png"}):
            with self.subTest(locator=locator), self.assertRaisesRegex(LocatorError, "replay_"):
                context.tap(locator)
        self.assertEqual(self.capture.mock_calls, [])

    def test_template_only_locator_skips_xml(self) -> None:
        context = self.context()
        with patch("autococ.actions.find_template", return_value=MATCH):
            self.assertEqual(context.locate({"template_path": "go.png"}), MATCH)
        self.capture.capture_ui_xml.assert_not_called()
        self.capture.capture_screenshot.assert_called_once()

    def test_template_fallback_survives_xml_capture_failure(self) -> None:
        context = self.context()
        self.capture.capture_ui_xml.side_effect = CaptureError("no UI tree")
        with patch("autococ.actions.find_template", return_value=MATCH):
            self.assertEqual(context.locate({"text": "Go", "template_path": "go.png"}), MATCH)
        self.capture.capture_screenshot.assert_called_once()

    def test_dry_run_template_uses_explicit_replay_without_capture(self) -> None:
        replay = self.root / "replay.png"
        context = self.context(dry_run=True, replay_screenshot_path=replay)
        with patch("autococ.actions.find_template", return_value=MATCH) as matcher:
            self.assertEqual(context.locate({"template_path": "go.png"}), MATCH)
            self.assertEqual(matcher.call_args.args[0], replay)
        self.assertEqual(self.capture.mock_calls, [])

    def test_dry_run_wait_does_not_poll_immutable_evidence(self) -> None:
        context = self.context(dry_run=True)
        with patch("autococ.actions.time.sleep") as sleep:
            with self.assertRaises(LocatorError):
                context.wait_until({"text": "Go"}, timeout_sec=30)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
