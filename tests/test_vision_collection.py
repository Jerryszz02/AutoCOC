"""Collection never deploys troops and preserves uncertain paid-input evidence."""

import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autococ.errors import ConfigError, DeviceConnectionError, FlowError
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.vision_collection import collect_samples
from training.vision.data import validate
from scripts import collect_vision_samples


class FakeSession:
    def __init__(self, menu, army, fresh):
        self.menu, self.army, self.fresh = menu, army, fresh
        self.client_version = "18.647.2"
        self.native = None
        self.last_snapshot = None
        self.clicks = []
        self.observes = []
        self.direct_preview = None
        self.recognizer = None

    def click(self, frame, name):
        self.clicks.append(name)
        self.last_snapshot = frame

    def wait_for(self, scenes, **kwargs):
        result = (self.menu if "search" in scenes else
                  self.direct_preview if self.direct_preview is not None else self.army)
        self.last_snapshot = result
        return result

    def observe(self, label):
        self.observes.append(label)
        self.last_snapshot = self.fresh
        return self.fresh


class FakeRecognizer:
    def __init__(self, frames):
        self.frames = frames

    def recognize_region(self, path, roi, *, single_line=False):
        assert single_line
        frame = self.frames[path]
        value = frame.observations["independent_price"]
        if value is None:
            return []
        bbox = frame.observations["search_cost_evidence"]["bbox"]
        reads = [OCRText(str(value), frame.observations.get("independent_confidence", .99), tuple(bbox))]
        if frame.observations.get("independent_extra") is not None:
            reads.append(OCRText(str(frame.observations["independent_extra"]), .99, tuple(bbox)))
        return reads


class VisionCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = time.monotonic()
        self.frames_by_path = {}
        self.home_before = self.frame("home-before", "village", gold=100)
        self.home_after = self.frame("home-after", "village", gold=90)
        self.menu = self.frame("menu", "search", fee=10)
        self.army = self.frame("army", "training")
        self.army.observations["army"] = {"troops": {"used": 1, "capacity": 1},
            "spells": {"used": 0, "capacity": 0},
            "manifest": {"complete": True, "supported_layout": True,
                         "frame": str(self.army.screenshot_path)}}
        self.scout1 = self.frame("scout1", "enemy_village", fee=20, enemy_gold=300)
        self.scout2 = self.frame("scout2", "enemy_village", fee=20, enemy_gold=400)
        self.exit_frame = self.frame("exit", "enemy_village", fee=20, enemy_gold=400)
        self.session = FakeSession(self.menu, self.army, self.exit_frame)
        self.session.recognizer = FakeRecognizer(self.frames_by_path)

    def frame(self, name, scene, *, gold=None, fee=None, enemy_gold=None, countdown=30):
        path = self.root / f"{name}.png"
        cv2.imencode(".png", np.full((24, 32, 3), len(name), dtype=np.uint8))[1].tofile(path)
        observations = {"observed_at_monotonic": self.now}
        if gold is not None:
            observations.update(resources={"gold": gold}, resource_source="village_inventory")
        if fee is not None:
            observations["search_cost_gold"] = fee
            roi = ([145, 530, 245, 573] if scene == "search" else [1098, 515, 1250, 560])
            bbox = ([160, 532, 235, 568] if scene == "search" else [1110, 520, 1210, 550])
            item = {"text": str(fee), "confidence": .99, "bbox": bbox}
            observations["ocr"] = [item.copy()]
            observations["baseline_resolution"] = [1280, 720]
            observations["independent_price"] = fee
            observations["search_cost_evidence"] = {**item, "frame": str(path),
                "frame_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_roi_at_1280x720": roi, "candidate_count": 1}
        if enemy_gold is not None:
            observations.update(resources={"gold": enemy_gold, "elixir": 200},
                                resource_source="enemy_available")
            observations["battle"] = {"countdown_seconds": countdown,
                "countdown_evidence": {"frame": str(path), "source": "current_frame_ocr",
                                       "confidence": .99}}
        snapshot = SceneSnapshot(scene, .99, path, observations)
        self.frames_by_path[path] = snapshot
        return snapshot

    def collect(self, *, max_samples=1, budget=30, candidates=None, after=None):
        output = self.root / "collected"
        home_after = after or self.home_after
        with patch("autococ.vision_collection.return_to_village",
                   side_effect=[self.home_before, home_after]) as home, \
                patch("autococ.vision_collection._next_candidate",
                      side_effect=candidates or [self.scout1]) as next_candidate, \
                patch("autococ.vision_collection._leave_scout") as leave:
            report = collect_samples(self.session, output, max_samples=max_samples,
                                     max_search_cost_gold=budget)
        return report, output, home, next_candidate, leave

    def test_one_candidate_records_unreviewed_hashes_and_real_home_delta(self):
        report, output, _, _, leave = self.collect()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["returned_home"])
        self.assertEqual(report["actual_search_cost_gold"], 10)
        self.assertEqual(report["quoted_confirmed_gold"], 10)
        self.assertEqual(report["attempts"][0]["fee_evidence"]["independent_ocr"]["text"], "10")
        self.assertEqual(self.session.clicks, ["attack", "find_match_regular", "attack"])
        self.assertEqual(leave.call_count, 1)
        self.assertEqual(leave.call_args.args[1].screenshot_path, self.exit_frame.screenshot_path)
        rows = [json.loads(line) for line in (output / "unreviewed-manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_status"], "pending")
        self.assertEqual(rows[0]["split"], "unassigned")
        self.assertEqual(rows[0]["client_version"], "18.647.2")
        self.assertFalse(rows[0]["zoom_evidence"]["verified"])
        self.assertTrue(validate(output, rows, [])["valid"])

    def test_direct_preview_from_menu_reserves_one_fee_without_army_attack(self):
        self.session.direct_preview = self.scout1
        report, _, _, _, _ = self.collect()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["quoted_attempted_gold"], 10)
        self.assertEqual(report["attempts"][0]["action_history"], ["find_match_regular"])
        self.assertEqual(self.session.clicks, ["attack", "find_match_regular"])

    def test_menu_transition_uncertain_never_replays_paid_input(self):
        def wait_for(scenes, **kwargs):
            if "search" in scenes:
                return self.menu
            raise FlowError("transition uncertain")
        self.session.wait_for = wait_for
        report, _, _, next_candidate, leave = self.collect()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["quoted_attempted_gold"], 10)
        self.assertEqual(self.session.clicks, ["attack", "find_match_regular"])
        self.assertFalse(report["returned_home"])
        self.assertFalse(next_candidate.called)
        self.assertFalse(leave.called)

    def test_next_candidate_rechecks_budget_before_click(self):
        report, _, _, next_candidate, _ = self.collect(max_samples=2, budget=25)
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["quoted_attempted_gold"], 10)
        self.assertEqual(self.session.clicks.count("next"), 0)
        self.assertEqual(next_candidate.call_count, 1)
        self.assertIn("exceeds remaining", report["reason"])
        self.assertTrue(report["returned_home"])

    def test_two_distinct_candidates_share_conservative_session_group(self):
        self.home_after = self.frame("home-after-30", "village", gold=70)
        report, output, _, _, _ = self.collect(max_samples=2, budget=30,
                                               candidates=[self.scout1, self.scout2])
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["actual_search_cost_gold"], 30)
        self.assertEqual(self.session.clicks.count("next"), 1)
        rows = [json.loads(line) for line in (output / "unreviewed-manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len({row["source_session"] for row in rows}), 1)
        self.assertEqual(len({row["layout_group_id"] for row in rows}), 1)
        self.assertEqual(len({row["leakage_group_id"] for row in rows}), 1)

    def test_uncertain_paid_transition_keeps_attempt_and_never_blind_exits(self):
        report, output, _, _, leave = self.collect(candidates=[FlowError("transition timed out")])
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["returned_home"])
        self.assertIsNone(report["actual_search_cost_gold"])
        self.assertEqual(report["quoted_attempted_gold"], 10)
        self.assertEqual(report["attempts"][0]["state"], "input_uncertain")
        self.assertEqual(report["samples"], 0)
        self.assertFalse(leave.called)
        self.assertEqual(json.loads((output / "collection.json").read_text())["status"], "failed")

    def test_sample_write_failure_recovers_confirmed_preview_with_fresh_frame(self):
        with patch("autococ.vision_collection._save_sample", side_effect=OSError("disk full")):
            report, output, home, _, leave = self.collect()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason"], "disk full")
        self.assertTrue(report["returned_home"])
        self.assertEqual(report["actual_search_cost_gold"], 10)
        self.assertEqual(home.call_count, 2)
        leave.assert_called_once_with(self.session, self.exit_frame)
        self.assertEqual(json.loads((output / "collection.json").read_text()),
                         json.loads(json.dumps(report)))

    def test_recovery_rejects_fresh_battle_frame(self):
        self.session.fresh = self.frame("battle", "battle")
        with patch("autococ.vision_collection._save_sample", side_effect=OSError("disk full")):
            report, _, home, _, leave = self.collect()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason"], "disk full")
        self.assertIn("observed battle", report["recovery_error"])
        self.assertFalse(report["returned_home"])
        self.assertEqual(home.call_count, 1)
        leave.assert_not_called()

    def test_recovery_observation_failure_preserves_original_error(self):
        with patch("autococ.vision_collection._save_sample", side_effect=OSError("disk full")), \
                patch.object(self.session, "observe", side_effect=OSError("capture failed")):
            report, _, home, _, leave = self.collect()
        self.assertEqual(report["reason"], "disk full")
        self.assertEqual(report["recovery_error"], "capture failed")
        self.assertFalse(report["returned_home"])
        self.assertEqual(home.call_count, 1)
        leave.assert_not_called()

    def test_second_paid_transition_cannot_recover_using_previous_preview(self):
        report, _, home, _, leave = self.collect(
            max_samples=2, candidates=[self.scout1, FlowError("next uncertain")])
        self.assertEqual(report["attempts"][-1]["state"], "input_uncertain")
        self.assertFalse(report["returned_home"])
        self.assertEqual(home.call_count, 1)
        leave.assert_not_called()

    def test_exit_failure_is_not_retried(self):
        with patch("autococ.vision_collection._safe_exit", side_effect=FlowError("exit uncertain")) as safe_exit:
            report, _, home, _, _ = self.collect()
        self.assertEqual(report["reason"], "exit uncertain")
        self.assertFalse(report["returned_home"])
        self.assertEqual(home.call_count, 1)
        safe_exit.assert_called_once_with(self.session)

    def test_short_preview_stops_next_search_and_uses_fresh_exit_frame(self):
        self.scout1 = self.frame("short-scout", "enemy_village", fee=20,
                                 enemy_gold=300, countdown=4)
        report, _, _, next_candidate, leave = self.collect(max_samples=2, budget=40)
        self.assertEqual(report["samples"], 1)
        self.assertEqual(next_candidate.call_count, 1)
        self.assertEqual(self.session.clicks.count("next"), 0)
        self.assertTrue(report["returned_home"])
        self.assertEqual(leave.call_args.args[1].screenshot_path, self.exit_frame.screenshot_path)

    def test_twenty_seconds_left_preserves_original_without_zoom(self):
        self.session.native = object()
        self.scout1 = self.frame("scout-20", "enemy_village", fee=20,
                                 enemy_gold=300, countdown=20)
        with patch("autococ.vision_collection._prepare_building_view") as prepare:
            report, output, _, _, _ = self.collect()
        self.assertEqual(report["samples"], 1)
        self.assertFalse(prepare.called)
        row = json.loads((output / "unreviewed-manifest.jsonl").read_text().splitlines()[0])
        self.assertFalse(row["zoom_evidence"]["verified"])

    def test_fee_over_budget_before_paid_input_returns_from_search(self):
        report, _, _, next_candidate, _ = self.collect(
            budget=9, after=self.frame("home-no-cost", "village", gold=100))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["attempts"], [])
        self.assertEqual(self.session.clicks, ["attack"])
        self.assertFalse(next_candidate.called)
        self.assertTrue(report["returned_home"])
        self.assertEqual(report["actual_search_cost_gold"], 0)

    def test_low_confidence_or_ambiguous_quote_never_starts_paid_search(self):
        self.menu.observations["search_cost_evidence"]["confidence"] = .8
        self.menu.observations["ocr"][0]["confidence"] = .8
        report, _, _, next_candidate, _ = self.collect(
            budget=10, after=self.frame("home-no-cost", "village", gold=100))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["quoted_attempted_gold"], 0)
        self.assertEqual(self.session.clicks, ["attack"])
        self.assertFalse(next_candidate.called)
        self.assertTrue(report["returned_home"])

    def test_ambiguous_price_roi_cannot_use_smaller_budget_quote(self):
        self.menu.observations["search_cost_gold"] = 1
        self.menu.observations["search_cost_evidence"].update(text="1", candidate_count=2)
        self.menu.observations["ocr"] = [
            {"text": "1", "confidence": .99, "bbox": [160, 532, 235, 568]},
            {"text": "10", "confidence": .97, "bbox": [160, 532, 235, 568]},
        ]
        report, _, _, next_candidate, _ = self.collect(
            budget=1, after=self.frame("home-no-cost", "village", gold=100))
        self.assertEqual(report["quoted_attempted_gold"], 0)
        self.assertEqual(self.session.clicks, ["attack"])
        self.assertFalse(next_candidate.called)

    def test_underread_full_ocr_disagrees_with_independent_same_frame_read(self):
        self.menu.observations["search_cost_gold"] = 1
        self.menu.observations["search_cost_evidence"]["text"] = "1"
        self.menu.observations["ocr"][0]["text"] = "1"
        report, _, _, next_candidate, _ = self.collect(
            budget=1, after=self.frame("home-no-cost", "village", gold=100))
        self.assertIn("disagrees", report["reason"])
        self.assertEqual(report["quoted_attempted_gold"], 0)
        self.assertEqual(self.session.clicks, ["attack"])
        self.assertFalse(next_candidate.called)

    def test_underread_next_fee_stops_before_second_paid_click(self):
        self.scout1.observations["search_cost_gold"] = 2
        self.scout1.observations["search_cost_evidence"]["text"] = "2"
        self.scout1.observations["ocr"][0]["text"] = "2"
        report, _, _, next_candidate, _ = self.collect(max_samples=2, budget=12)
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["quoted_attempted_gold"], 10)
        self.assertEqual(self.session.clicks.count("next"), 0)
        self.assertEqual(next_candidate.call_count, 1)
        self.assertIn("disagrees", report["reason"])

    def test_stale_frame_hash_or_wrong_roi_quote_never_clicks(self):
        self.menu.observations["search_cost_evidence"]["frame_sha256"] = "0" * 64
        report, _, _, _, _ = self.collect(
            budget=10, after=self.frame("home-no-cost", "village", gold=100))
        self.assertEqual(report["quoted_attempted_gold"], 0)
        self.assertEqual(self.session.clicks, ["attack"])

    def test_ambiguous_independent_roi_read_stops_next_click(self):
        self.scout1.observations["independent_extra"] = 200
        report, _, _, _, _ = self.collect(max_samples=2, budget=30)
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["quoted_attempted_gold"], 10)
        self.assertEqual(self.session.clicks.count("next"), 0)
        self.assertIn("Independent", report["reason"])

    def test_limits_and_existing_output_are_rejected(self):
        with self.assertRaises(ValueError):
            collect_samples(self.session, self.root / "invalid", max_samples=11,
                            max_search_cost_gold=1)
        with self.assertRaises(ValueError):
            collect_samples(self.session, self.root, max_samples=1,
                            max_search_cost_gold=1)


class VisionCollectionCliTests(unittest.TestCase):
    def test_framework_setup_failures_return_structured_json(self):
        for stage, error in (("config", ConfigError("missing config")),
                             ("device", DeviceConnectionError("emulator unavailable")),
                             ("session", FlowError("session unavailable"))):
            with self.subTest(stage=stage), TemporaryDirectory() as directory, \
                    patch.object(collect_vision_samples, "load_config") as config, \
                    patch.object(collect_vision_samples, "DeviceManager") as manager, \
                    patch.object(collect_vision_samples.GameSession, "connect") as connect, \
                    redirect_stdout(io.StringIO()) as stdout:
                target = {"config": config, "device": manager.return_value.connect,
                          "session": connect}[stage]
                target.side_effect = error
                result = collect_vision_samples.main([
                    "--output", str(Path(directory) / "samples"),
                    "--max-samples", "1", "--max-search-cost-gold", "10"])
                self.assertEqual(result, 1)
                self.assertEqual(json.loads(stdout.getvalue()),
                                 {"status": "failed", "reason": str(error)})


if __name__ == "__main__":
    unittest.main()
