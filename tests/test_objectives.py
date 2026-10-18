"""Current-client Rift progress evidence from the October 15 live probes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from autococ.errors import CapabilityUnavailable, FlowError
from autococ.locator import find_template
from autococ.objectives import load_adapter, parse_progress, read_progress
from autococ.routine_config import GoalConfig, TaskSpec
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "assets/catalogs/rift_event_18_647_2.adapter.toml"
PROVENANCE = ROOT / "assets/catalogs/rift_event_progress_provenance_18_647_2.json"
TEMPLATE = ROOT / "assets/templates/daily_event_calendar_18_647_2.png"
FIXTURES = ROOT / "tests/fixtures"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_with(snapshot: SceneSnapshot, texts: list[dict]) -> SceneSnapshot:
    return replace(snapshot, observations={**snapshot.observations, "ocr": texts})


class RiftEventProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter(ADAPTER)
        cls.task = TaskSpec("event", "event", goal=GoalConfig(adapter_path=str(ADAPTER), target=16000))
        recognizer = ScreenshotRecognizer()
        recognizer.client_version = "18.647.2"
        cls.calendar = recognizer.recognize(FIXTURES / "rift_event_calendar_18_647_2.png")
        cls.detail = recognizer.recognize(FIXTURES / "rift_event_detail_18_647_2.png")

    def test_masked_live_probe_has_bound_current_card_ratio(self):
        self.assertEqual(self.calendar.scene, "popup")
        value = parse_progress(self.calendar, self.adapter, self.task, client_version="18.647.2")
        self.assertEqual((value.current, value.target, value.reason), (8452, 16000, "in_progress"))
        self.assertEqual(value.values["scoring_condition"], "battle")
        with self.assertRaises(FlowError):
            parse_progress(self.detail, self.adapter, self.task, client_version="18.647.2")

    def test_provenance_and_entry_template_match_real_probe(self):
        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        self.assertEqual(_sha(TEMPLATE), provenance["entry_template_sha256"])
        for item in provenance["frames"]:
            self.assertEqual(_sha(ROOT / item["masked_fixture"]), item["masked_sha256"])
            source = Path(item["source_frame"])
            if source.is_file():
                self.assertEqual(_sha(source), item["source_sha256"])
        match = find_template(FIXTURES / "rift_event_home_entry_18_647_2.png", TEMPLATE,
                              roi=self.adapter.entry_roi, roi_base_resolution=(1280, 720),
                              template_base_resolution=(1280, 720), threshold=.9)
        self.assertGreaterEqual(match.confidence, .9)

    def test_other_event_ratio_expiry_version_and_missing_anchors_are_rejected(self):
        source = self.calendar.observations["ocr"]
        for condition in ("wrong_title", "wrong_foreground", "missing_panel_evidence",
                          "wrong_event", "other_ratio", "expired"):
            with self.subTest(condition=condition):
                texts = [dict(item) for item in source]
                if condition == "wrong_title":
                    texts = [item for item in texts if item["text"] != "活动"]
                elif condition == "wrong_foreground":
                    texts = [item for item in texts if item["text"] != "今日活动"]
                elif condition == "wrong_event":
                    texts = [item for item in texts if item["text"] != "裂隙之乱"]
                elif condition == "other_ratio":
                    next(item for item in texts if item["text"] == "8452/16000")["bbox"] = [898, 140, 1030, 165]
                elif condition == "expired":
                    next(item for item in texts if item["text"] == "9天13小时")["text"] = "0天0小时"
                frame = _frame_with(self.calendar, texts)
                if condition == "missing_panel_evidence":
                    frame.observations["event_panel"] = None
                if condition == "expired":
                    self.assertFalse(parse_progress(frame, self.adapter, self.task,
                                                    client_version="18.647.2").available)
                else:
                    with self.assertRaises(FlowError):
                        parse_progress(frame, self.adapter, self.task, client_version="18.647.2")
        with self.assertRaises(FlowError):
            parse_progress(self.calendar, self.adapter, self.task, client_version="18.600.7")

    def test_read_only_calendar_visit_closes_and_verifies_return_home(self):
        home = SceneSnapshot("village", .99, Path("home.png"), {"buttons": [
            {"name": "attack"}, {"name": "shop"}]})
        returned = replace(home, screenshot_path=Path("returned.png"))
        session = Mock()
        session.client_version = "18.647.2"
        session.config = SimpleNamespace(source_path=ROOT / "config/routine.toml",
                                         vision=SimpleNamespace(template_dir=TEMPLATE.parent))
        session.observe.return_value = self.calendar
        session.wait_for.return_value = returned
        with patch("autococ.flow.return_to_village", side_effect=[home, returned]) as go_home:
            value = read_progress(session, self.task)
        self.assertEqual(value.current, 8452)
        session.click_template.assert_called_once_with(home, "daily_event_calendar_18_647_2",
                                                        roi=(270, 610, 365, 710))
        self.assertEqual(session.tap.call_count, 1)
        self.assertEqual(session.tap.call_args.kwargs["reason"], "Close verified progress panel")
        session.back.assert_not_called()
        self.assertEqual(go_home.call_count, 2)
        self.assertEqual(go_home.call_args.kwargs["initial_snapshot"], returned)

    def test_wrong_version_never_opens_calendar(self):
        home = SceneSnapshot("village", .99, Path("home.png"), {})
        session = Mock()
        session.client_version = "18.600.7"
        session.config = SimpleNamespace(source_path=ROOT / "config/routine.toml",
                                         vision=SimpleNamespace(template_dir=TEMPLATE.parent))
        with patch("autococ.flow.return_to_village", return_value=home), self.assertRaises(CapabilityUnavailable):
            read_progress(session, self.task)
        session.click_template.assert_not_called()
        session.tap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
