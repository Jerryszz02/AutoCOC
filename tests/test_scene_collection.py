"""No-device checks for the fixed scene collection input boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from autococ.scene_collection import (SceneCollectionStopped, _fresh, _unique_chat,
                                       capture_route, collect_scene_samples)


RESOURCES = {"gold": 100, "elixir": 200, "dark_elixir": 30, "gems": 40}


class FakeSession:
    def __init__(self, root: Path):
        self.root = root
        self.client_version = "18.647.2"
        self.client_version_code = 180647003
        self.config = SimpleNamespace(game=SimpleNamespace(
            baseline_resolution=(1280, 720), package_name="com.supercell.clashofclans"))
        self.action_count = 0
        self.scene = "village"
        self.serial = 0
        self.actions = []
        self.army_change = False
        self.resource_change = False
        self.duplicate_button = False
        self.stale = False
        self.conflict = False

    def check_deadline(self):
        return None

    def frame(self):
        if not self.stale:
            self.serial += 1
        path = self.root / f"{self.serial:05d}.png"
        path.write_bytes(b"raw-frame" + str(self.serial).encode())
        now = time.monotonic()
        resources = dict(RESOURCES)
        if self.resource_change and self.action_count >= 6:
            resources["gems"] += 1
        buttons = ([{"name": "attack"}, {"name": "shop"}]
                   if self.scene == "village" else [])
        if self.duplicate_button and self.scene == "village":
            buttons.append({"name": "attack"})
        return SimpleNamespace(
            scene="account_in_use" if self.conflict else self.scene,
            confidence=.95, screenshot_path=path,
            observations={"observed_at_monotonic": now,
                          "requested_at_monotonic": now,
                          "frame_id": f"frame-{self.serial}", "buttons": buttons,
                          "village_type": "home", "resource_source": "village_inventory",
                          "resources": resources})

    def observe(self, _label):
        return self.frame()

    def wait_for(self, scenes, *, timeout_sec, label):
        assert self.scene in scenes
        return self.frame()

    def click(self, _shot, button):
        if button == "attack" and self.scene == "village":
            self.scene = "search"
        elif button == "edit_army" and self.scene == "search":
            self.scene = "training"
        else:
            raise AssertionError("unsafe or unexpected click")
        self.actions.append(button)
        self.action_count += 1

    def click_template(self, _shot, template, *, roi):
        assert (self.scene, template, roi) == ("village", "hud_chat_18_647_2",
                                                (35, 330, 115, 410))
        self.actions.append("open_chat")
        self.action_count += 1
        self.scene = "clan_chat"

    def back(self, _shot, *, reason):
        assert self.scene in {"training", "search", "clan_chat"}
        self.actions.append("back")
        self.action_count += 1
        self.scene = {"training": "search", "search": "village",
                      "clan_chat": "village"}[self.scene]


def fake_open_army(session, home, evidence):
    session.click(home, "attack")
    search = session.wait_for({"search"}, timeout_sec=20, label="search")
    evidence.append(search.screenshot_path)
    session.click(search, "edit_army")
    return session.wait_for({"training"}, timeout_sec=20, label="training")


def fake_open_chat(session, home, evidence):
    session.click_template(home, "hud_chat_18_647_2", roi=(35, 330, 115, 410))
    chat = session.wait_for({"clan_chat"}, timeout_sec=20, label="chat")
    evidence.append(chat.screenshot_path)
    return chat


class SceneRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = FakeSession(Path(self.temp.name))

    def run_route(self, *, foreground=lambda: True, stop=lambda: False,
                  chat_unique=lambda shot, session: {"score": .96}):
        return capture_route(self.session, foreground=foreground, stop=stop,
                             open_army=fake_open_army, open_chat=fake_open_chat,
                             chat_unique=chat_unique,
                             army_state=lambda shot: ("full-roster", 2 if self.session.army_change
                                                      and self.session.serial >= 6 else 1))

    def test_success_uses_exactly_six_read_only_inputs_and_candidate_evidence(self):
        result = self.run_route()
        self.assertEqual(self.session.actions,
                         ["attack", "edit_army", "back", "back", "open_chat", "back"])
        self.assertEqual(result["action_count"], 6)
        self.assertTrue(result["army_unchanged_while_on_training_page"])
        self.assertTrue(result["home_resources_and_gems_unchanged"])
        self.assertEqual(result["army_comparison_scope"],
                         "two_independent_training_page_frames_only")
        self.assertGreaterEqual(len(result["frames"]), 10)
        self.assertTrue(all(item["sha256"] for item in result["frames"]))

    def test_default_chat_reuses_unique_match_without_a_second_match(self):
        matched = {"score": .96, "point_at_1280x720": [82, 371]}
        taps = []
        def tap(shot, point, *, reason):
            self.assertEqual(self.session.scene, "village")
            taps.append(point)
            self.session.actions.append("open_chat")
            self.session.action_count += 1
            self.session.scene = "clan_chat"
        self.session.tap = tap
        with patch.object(self.session, "click_template",
                          side_effect=AssertionError("must not repeat template matching")):
            result = capture_route(
                self.session, foreground=lambda: True, stop=lambda: False,
                open_army=fake_open_army, chat_unique=lambda shot, session: matched,
                army_state=lambda shot: ("full-roster", 1))
        self.assertEqual(taps, [[82, 371]])
        self.assertEqual(result["action_count"], 6)

    def test_default_chat_rechecks_foreground_after_the_only_match(self):
        state = {"front": True}
        def match(shot, session):
            state["front"] = False
            return {"score": .96, "point_at_1280x720": [82, 371]}
        with patch.object(self.session, "tap", create=True) as tap:
            with self.assertRaisesRegex(SceneCollectionStopped, "foreground"):
                capture_route(
                    self.session, foreground=lambda: state["front"], stop=lambda: False,
                    open_army=fake_open_army, chat_unique=match,
                    army_state=lambda shot: ("full-roster", 1))
            tap.assert_not_called()
        self.assertEqual(self.session.actions, ["attack", "edit_army", "back", "back"])

    def test_wrong_version_has_zero_input(self):
        self.session.client_version_code = 1
        with self.assertRaisesRegex(SceneCollectionStopped, "client_version"):
            self.run_route()
        self.assertEqual(self.session.actions, [])

    def test_not_foreground_has_zero_input(self):
        with self.assertRaisesRegex(SceneCollectionStopped, "foreground"):
            self.run_route(foreground=lambda: False)
        self.assertEqual(self.session.actions, [])

    def test_duplicate_attack_button_has_zero_input(self):
        self.session.duplicate_button = True
        with self.assertRaisesRegex(SceneCollectionStopped, "attack_not_unique"):
            self.run_route()
        self.assertEqual(self.session.actions, [])

    def test_stale_pair_has_zero_input(self):
        self.session.stale = True
        with self.assertRaisesRegex(SceneCollectionStopped, "frame_not_independent"):
            self.run_route()
        self.assertEqual(self.session.actions, [])

    def test_account_conflict_has_zero_input(self):
        self.session.conflict = True
        with self.assertRaisesRegex(SceneCollectionStopped, "expected_village"):
            self.run_route()
        self.assertEqual(self.session.actions, [])

    def test_chat_ambiguity_stops_after_return_home(self):
        with self.assertRaisesRegex(SceneCollectionStopped, "chat_hud_not_unique"):
            self.run_route(chat_unique=lambda shot, session: (_ for _ in ()).throw(
                SceneCollectionStopped("chat_hud_not_unique")))
        self.assertEqual(self.session.actions,
                         ["attack", "edit_army", "back", "back"])

    def test_foreground_lost_during_chat_match_sends_no_chat_input(self):
        state = {"front": True}
        def match(shot, session):
            state["front"] = False
            return {"score": .96}
        with self.assertRaisesRegex(SceneCollectionStopped, "foreground"):
            self.run_route(foreground=lambda: state["front"], chat_unique=match)
        self.assertEqual(self.session.actions,
                         ["attack", "edit_army", "back", "back"])

    def test_stop_during_chat_match_sends_no_chat_input(self):
        from autococ.errors import StopRequested
        state = {"stop": False}
        def match(shot, session):
            state["stop"] = True
            return {"score": .96}
        with self.assertRaises(StopRequested):
            self.run_route(stop=lambda: state["stop"], chat_unique=match)
        self.assertEqual(self.session.actions,
                         ["attack", "edit_army", "back", "back"])

    def test_final_resource_change_is_failure(self):
        self.session.resource_change = True
        with self.assertRaisesRegex(SceneCollectionStopped, "resources_changed"):
            self.run_route()
        self.assertEqual(len(self.session.actions), 6)

    def test_stop_before_input_has_zero_input(self):
        from autococ.errors import StopRequested
        with self.assertRaises(StopRequested):
            self.run_route(stop=lambda: True)
        self.assertEqual(self.session.actions, [])

    def test_army_difference_is_failure_before_any_back(self):
        self.session.army_change = True
        with self.assertRaisesRegex(SceneCollectionStopped, "army_changed"):
            self.run_route()
        self.assertEqual(self.session.actions, ["attack", "edit_army"])

    def test_nonfinite_provenance_rejected(self):
        shot = self.session.frame()
        shot.observations["requested_at_monotonic"] = float("nan")
        with self.assertRaisesRegex(SceneCollectionStopped, "capture_time"):
            _fresh(shot)

    def _native_chat_frame(self, positions):
        import cv2
        import numpy as np
        root = Path(self.temp.name)
        templates = root / "templates"
        templates.mkdir(exist_ok=True)
        template = np.random.default_rng(7).integers(20, 235, size=(43, 44, 3), dtype=np.uint8)
        template_path = templates / "hud_chat_18_647_2.png"
        ok, encoded = cv2.imencode(".png", template)
        self.assertTrue(ok)
        encoded.tofile(str(template_path))
        scaled = cv2.resize(template, (88, 86), interpolation=cv2.INTER_LINEAR)
        image = np.full((1440, 2560, 3), 50, dtype=np.uint8)
        for x, y in positions:
            image[y:y + 86, x:x + 88] = scaled
        shot = self.session.frame()
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        encoded.tofile(str(shot.screenshot_path))
        self.session.config.vision = SimpleNamespace(template_dir=templates)
        self.session.context = SimpleNamespace(screen_resolution=(2560, 1440))
        self.session._validate_snapshot = lambda current: None
        return shot

    def test_chat_unique_current_native_2560_frame(self):
        shot = self._native_chat_frame([(100, 690)])
        proof = _unique_chat(shot, self.session)
        self.assertEqual(proof["native_resolution"], [2560, 1440])
        self.assertEqual(proof["scaled_roi"], [70, 660, 230, 820])
        self.assertEqual(proof["scaled_template_size"], [88, 86])
        self.assertEqual(proof["point_at_1280x720"], [72, 366])
        self.assertGreaterEqual(proof["score"], .88)

    def test_chat_two_native_2560_peaks_rejected(self):
        shot = self._native_chat_frame([(70, 660), (142, 734)])
        with self.assertRaisesRegex(SceneCollectionStopped, "chat_hud_not_unique"):
            _unique_chat(shot, self.session)

    def test_chat_native_frame_context_mismatch_rejected(self):
        shot = self._native_chat_frame([(100, 690)])
        self.session.context.screen_resolution = (1280, 720)
        with self.assertRaisesRegex(SceneCollectionStopped, "resolution_mismatch"):
            _unique_chat(shot, self.session)

    def test_invalid_split_or_existing_output_prevents_device_work(self):
        root = Path(self.temp.name)
        repo = root / "repo"
        repo.mkdir()
        config = root / "config.toml"
        config.write_text("", encoding="utf-8")
        output = root / "output"
        with self.assertRaises(ValueError):
            collect_scene_samples(config, output, split="TEST", session_id="S1",
                                  group_id="G1", repo_root=repo)
        output.mkdir()
        with self.assertRaises(ValueError):
            collect_scene_samples(config, output, split="TRAIN", session_id="S1",
                                  group_id="G1", repo_root=repo)

    def test_dry_run_config_refused_before_device_connect_or_input(self):
        root = Path(self.temp.name)
        repo = root / "repo"
        repo.mkdir()
        config_path = root / "config.toml"
        config_path.write_text("", encoding="utf-8")
        output = root / "dry-run-candidate"
        config = SimpleNamespace(
            game=SimpleNamespace(baseline_resolution=(1280, 720),
                                 package_name="com.supercell.clashofclans"),
            runtime=SimpleNamespace(dry_run=True))
        def git_value(repo_arg, *args):
            return "" if args == ("status", "--porcelain") else "head123"
        with (patch("autococ.scene_collection._git", side_effect=git_value),
              patch("autococ.scene_collection._sha", return_value="sha"),
              patch("autococ.config.load_config", return_value=config),
              patch("autococ.device.DeviceManager") as manager,
              patch("autococ.session.GameSession.connect") as connect,
              patch("autococ.scene_collection.capture_route") as route):
            result = collect_scene_samples(config_path, output, split="TRAIN",
                                           session_id="dry-run-session", group_id="dry-run-group",
                                           repo_root=repo)
        self.assertEqual(result["status"], "failed")
        self.assertIn("dry_run_cannot_produce_live_scene_candidates", result["reason"])
        self.assertEqual(result["action_count"], 0)
        self.assertEqual(result["raw_frames"], [])
        self.assertTrue((output / "prospective.json").is_file())
        self.assertTrue((output / "summary.json").is_file())
        manager.assert_not_called()
        connect.assert_not_called()
        route.assert_not_called()

    def test_prospective_split_and_group_exist_before_first_capture(self):
        root = Path(self.temp.name)
        repo = root / "repo"
        template_dir = repo / "assets" / "templates"
        template_dir.mkdir(parents=True)
        (template_dir / "hud_chat_18_647_2.png").write_bytes(b"template")
        config_path = root / "config.toml"
        config_path.write_text("", encoding="utf-8")
        output = root / "candidate-1"
        config = SimpleNamespace(
            game=SimpleNamespace(baseline_resolution=(1280, 720),
                                 package_name="com.supercell.clashofclans"),
            vision=SimpleNamespace(template_dir=template_dir),
            runtime=SimpleNamespace(step_timeout_sec=5, dry_run=False))
        manager = SimpleNamespace(adb=object(), connect=lambda: SimpleNamespace(serial="fake-device"))
        session = SimpleNamespace(client_version="18.647.2", client_version_code=180647003,
                                  action_count=6, deadline=time.monotonic() + 180,
                                  task_deadline=time.monotonic() + 180,
                                  begin_task=lambda: None, close=lambda: None)
        def fake_capture(current, *, foreground, stop):
            prospective = __import__("json").loads((output / "prospective.json").read_text(encoding="utf-8"))
            self.assertEqual((prospective["split"], prospective["candidate_leakage_group_id"]),
                             ("VAL", "new-group-candidate"))
            self.assertIsNone(prospective["labels"])
            self.assertFalse(prospective["leakage_independence_verified"])
            self.assertTrue(foreground())
            frames = output / "frames"
            frames.mkdir()
            frame = frames / "00001-home.png"
            frame.write_bytes(b"raw")
            (output / "events.jsonl").write_text("{}\n", encoding="utf-8")
            return {"fixed_route_mock": True,
                    "frames": [{"path": str(frame), "sha256": "sha"}],
                    "search_menu_frames": []}
        def git_value(repo_arg, *args):
            return "" if args == ("status", "--porcelain") else "head123"
        with (patch("autococ.scene_collection._git", side_effect=git_value),
              patch("autococ.scene_collection._sha", return_value="sha"),
              patch("autococ.scene_collection._loaded_source_paths", return_value={}),
              patch("autococ.config.load_config", return_value=config),
              patch("autococ.device.DeviceManager", return_value=manager),
              patch("autococ.device.foreground_package", return_value="com.supercell.clashofclans"),
              patch("autococ.session.GameSession.connect", return_value=session),
              patch("autococ.scene_collection.capture_route", side_effect=fake_capture)):
            result = collect_scene_samples(config_path, output, split="VAL", session_id="new-session",
                                           group_id="new-group-candidate", repo_root=repo)
        self.assertEqual(result["status"], "candidate_captured", result["reason"])
        self.assertTrue((output / "summary.json").is_file())
        self.assertTrue((output.parent / ".scene_collection_groups").is_dir())


if __name__ == "__main__":
    unittest.main()
