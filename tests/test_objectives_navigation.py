from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from autococ.errors import ConfigError, FlowError
from autococ.objectives import NavigationStep, ProgressAdapter, load_adapter, read_progress
from autococ.routine_config import GoalConfig, TaskSpec
from autococ.scene import SceneSnapshot


def frame(name, texts, scene="unknown"):
    return SceneSnapshot(scene, .99, Path(name), {"baseline_resolution": [1280, 720],
        "ocr": [{"text": text, "confidence": .99, "bbox": [100, 30+i*55, 280, 60+i*55]}
                for i, text in enumerate(texts)]})


def frame_at(name, items, scene="unknown"):
    return SceneSnapshot(scene, .99, Path(name), {"baseline_resolution": [1280, 720],
        "ocr": [{"text": text, "confidence": .99, "bbox": box} for text, box in items]})


class ProgressNavigationTests(unittest.TestCase):
    def setUp(self):
        self.home = frame("home.png", ["当前活动"], "village")
        self.after = frame("home-after.png", [], "village")
        self.adapter = ProgressAdapter("current-event", "event", "活动甲", (0, 0, 600, 600),
            entry_text="当前活动", active_label="进行中", scoring_condition="battle", unavailable_label="已结束")
        self.task = TaskSpec("event", "event", goal=GoalConfig(adapter_path="current.toml", target=50))
        self.session = Mock()
        self.session.config = SimpleNamespace(source_path=Path("config.toml"))
        self.session.last_snapshot = self.after

    def test_real_page_progress_is_read_then_verified_home(self):
        page = frame("event-page.png", ["活动甲", "进行中", "50 / 100"])
        self.session.observe.return_value = page
        with patch("autococ.flow.return_to_village", side_effect=[self.home, self.after]) as home, \
                patch("autococ.objectives.load_adapter", return_value=self.adapter):
            progress = read_progress(self.session, self.task)
        self.assertTrue(progress.completed)
        self.assertEqual((progress.current, progress.target), (50, 50))
        self.assertEqual(progress.evidence, (page.screenshot_path,))
        self.assertEqual(self.session.tap.call_args.args[0], self.home)
        self.assertEqual(self.session.back.call_args.args[0].scene, "goal_panel")
        self.assertEqual(home.call_count, 2)

    def test_expired_activity_returns_home_before_known_no_op_result(self):
        self.session.observe.return_value = frame("expired.png", ["活动甲", "已结束", "30 / 100"])
        with patch("autococ.flow.return_to_village", side_effect=[self.home, self.after]) as home, \
                patch("autococ.objectives.load_adapter", return_value=self.adapter):
            progress = read_progress(self.session, self.task)
        self.assertFalse(progress.available)
        self.assertEqual(progress.reason, "event_inactive")
        self.assertEqual(home.call_count, 2)
        self.session.back.assert_called_once()
        self.session.click.assert_not_called()

    def test_unknown_page_does_not_receive_unverified_back_or_confirm(self):
        self.session.observe.return_value = frame("different-page.png", ["购买", "50 / 100"])
        with patch("autococ.flow.return_to_village", return_value=self.home), \
                patch("autococ.objectives.load_adapter", return_value=self.adapter), \
                self.assertRaises(FlowError):
            read_progress(self.session, self.task)
        self.assertEqual(self.session.observe.call_count, 3)
        self.session.back.assert_not_called()
        self.session.click.assert_not_called()

    def _clan_navigation(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "daily_event_calendar.png").write_bytes(b"fixture")
        self.session.config.vision = SimpleNamespace(template_dir=folder)
        adapter = ProgressAdapter("clan-air-defense", "clan_games", "部落竞赛", (400, 250, 900, 600),
            entry_template="daily_event_calendar", entry_roi=(270, 610, 365, 720),
            accepted_label="已接受防空火箭任务", building_type="air_defense",
            navigation=(NavigationStep("活动", "打开", (850, 470, 1040, 550),
                                       section_label="部落竞赛", section_roi=(600, 430, 1050, 560)),))
        task = TaskSpec("clan_games", "clan_games", goal=GoalConfig(adapter_path="clan.toml", building_type="air_defense"))
        calendar = frame_at("calendar.png", [("活动", [500, 40, 590, 75]),
            ("部落竞赛", [650, 460, 820, 495]), ("打开", [875, 485, 950, 525])])
        challenge = frame_at("challenge.png", [("部落竞赛", [500, 40, 650, 75]),
            ("已接受防空火箭任务", [150, 160, 430, 205]), ("2 / 5", [500, 330, 610, 365])])
        return adapter, task, calendar, challenge

    def test_calendar_to_clan_page_unwinds_each_anchored_page_once(self):
        adapter, task, calendar, challenge = self._clan_navigation()
        self.session.observe.side_effect = [calendar, challenge, challenge, calendar]
        self.session.wait_for.return_value = self.after
        with patch("autococ.flow.return_to_village", side_effect=[self.home, self.after]) as home, \
                patch("autococ.objectives.load_adapter", return_value=adapter):
            progress = read_progress(self.session, task)
        self.assertEqual((progress.current, progress.target), (2, 5))
        self.assertFalse(progress.completed)
        self.session.click_template.assert_called_once_with(self.home, "daily_event_calendar", roi=(270, 610, 365, 720))
        self.session.tap.assert_called_once()
        self.assertEqual(self.session.tap.call_args.kwargs["reason"], "Open verified progress page: clan-air-defense, step 1")
        self.assertEqual(self.session.back.call_count, 2)
        self.assertEqual([call.args[0].screenshot_path.name for call in self.session.back.call_args_list],
                         ["challenge.png", "calendar.png"])
        self.assertEqual(self.session.observe.call_count, 4)
        self.assertEqual(home.call_count, 2)

    def test_two_navigation_steps_return_in_reverse_page_order(self):
        adapter, task, calendar, challenge = self._clan_navigation()
        adapter = ProgressAdapter(**{**adapter.__dict__, "navigation": adapter.navigation + (
            NavigationStep("竞赛主页", "查看", (850, 470, 1040, 550)),)})
        hub = frame_at("clan-hub.png", [("竞赛主页", [500, 40, 650, 75]),
            ("查看", [875, 485, 950, 525])])
        self.session.observe.side_effect = [calendar, hub, challenge, hub, calendar]
        self.session.wait_for.return_value = self.after
        with patch("autococ.flow.return_to_village", side_effect=[self.home, self.after]), \
                patch("autococ.objectives.load_adapter", return_value=adapter):
            progress = read_progress(self.session, task)
        self.assertEqual(progress.current, 2)
        self.assertEqual(self.session.tap.call_count, 2)
        self.assertEqual([call.args[0].screenshot_path.name for call in self.session.back.call_args_list],
                         ["challenge.png", "clan-hub.png", "calendar.png"])

    def test_android_back_can_return_directly_to_verified_village(self):
        adapter, task, calendar, challenge = self._clan_navigation()
        village = frame_at("back-home.png", [], "village")
        village.observations["buttons"] = [{"name": "attack"}, {"name": "shop"}]
        self.session.observe.side_effect = [calendar, challenge, village]
        with patch("autococ.flow.return_to_village", side_effect=[self.home, village]) as home, \
                patch("autococ.objectives.load_adapter", return_value=adapter):
            progress = read_progress(self.session, task)
        self.assertEqual(progress.current, 2)
        self.session.back.assert_called_once()
        self.session.wait_for.assert_not_called()
        self.assertEqual(home.call_args.kwargs["initial_snapshot"], village)

    def test_unverified_village_frame_is_not_a_return_anchor(self):
        adapter, task, calendar, challenge = self._clan_navigation()
        uncertain = frame_at("uncertain-home.png", [], "village")
        self.session.observe.side_effect = [calendar, challenge, uncertain, uncertain, uncertain]
        with patch("autococ.flow.return_to_village", return_value=self.home), \
                patch("autococ.objectives.load_adapter", return_value=adapter), self.assertRaises(FlowError):
            read_progress(self.session, task)
        self.session.back.assert_called_once()
        self.session.wait_for.assert_not_called()

    def test_personal_points_without_accepted_task_are_not_counted(self):
        adapter, task, calendar, challenge = self._clan_navigation()
        personal = frame_at("personal-points.png", [("部落竞赛", [500, 40, 650, 75]),
            ("3600 / 10000", [500, 330, 690, 365])])
        self.session.observe.side_effect = [calendar, personal, calendar]
        self.session.wait_for.return_value = self.after
        with patch("autococ.flow.return_to_village", side_effect=[self.home, self.after]), \
                patch("autococ.objectives.load_adapter", return_value=adapter), self.assertRaises(FlowError):
            read_progress(self.session, task)
        self.assertEqual(self.session.back.call_count, 2)

    def test_ambiguous_open_button_or_missing_section_never_clicks(self):
        adapter, task, calendar, _ = self._clan_navigation()
        ambiguous = frame_at("calendar.png", [("活动", [500, 40, 590, 75]),
            ("部落竞赛", [650, 460, 820, 495]),
            ("打开", [875, 485, 930, 525]), ("打开", [960, 485, 1020, 525])])
        for page in (ambiguous, frame_at("calendar.png", [("活动", [500, 40, 590, 75]),
                     ("打开", [875, 485, 950, 525])])):
            with self.subTest(page=page.observations["ocr"]):
                self.session.reset_mock()
                self.session.observe.return_value = page
                with patch("autococ.flow.return_to_village", return_value=self.home), \
                        patch("autococ.objectives.load_adapter", return_value=adapter), self.assertRaises(FlowError):
                    read_progress(self.session, task)
                self.session.tap.assert_not_called()
                self.session.back.assert_not_called()

    def test_transition_stuck_on_final_page_does_not_repeat_back(self):
        adapter, task, calendar, challenge = self._clan_navigation()
        self.session.observe.side_effect = [calendar, challenge, challenge, challenge, challenge]
        with patch("autococ.flow.return_to_village", return_value=self.home), \
                patch("autococ.objectives.load_adapter", return_value=adapter), self.assertRaises(FlowError):
            read_progress(self.session, task)
        self.session.back.assert_called_once()
        self.session.wait_for.assert_not_called()

    def test_navigation_schema_is_bounded_and_read_only(self):
        adapter, _, _, _ = self._clan_navigation()
        adapter.validate()
        with self.assertRaises(ConfigError):
            NavigationStep("活动", "接取", (850, 470, 1040, 550)).validate()
        with self.assertRaises(ConfigError):
            NavigationStep("活动", "打开", (850, 470, 1040, 550),
                           section_label="部落竞赛", section_roi=(0, 0, 300, 300)).validate()
        with self.assertRaises(ConfigError):
            ProgressAdapter(**{**adapter.__dict__, "navigation": adapter.navigation * 4}).validate()
        for unsafe in ({"entry_text": "领取奖励", "entry_template": ""},
                       {"entry_template": "buy_offer"}, {"close_text": "接取"}):
            with self.subTest(unsafe=unsafe), self.assertRaises(ConfigError):
                ProgressAdapter(**{**adapter.__dict__, **unsafe}).validate()

    def test_load_adapter_parses_navigation_tables(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "clan.toml"
        path.write_text('id = "clan"\nkind = "clan_games"\ntitle = "部落竞赛"\n'
                        'progress_roi = [400, 250, 900, 600]\nentry_template = "daily_event_calendar"\n'
                        'accepted_label = "已接受任务"\nbuilding_type = "air_defense"\n'
                        '[[navigation]]\npage_title = "活动"\nopen_text = "打开"\n'
                        'open_roi = [850, 470, 1040, 550]\nsection_label = "部落竞赛"\n'
                        'section_roi = [600, 430, 1050, 560]\n', encoding="utf-8")
        adapter = load_adapter(path)
        self.assertEqual(adapter.navigation[0].page_title, "活动")
        self.assertEqual(adapter.navigation[0].open_text, "打开")

    def test_resource_inventory_waits_for_stable_fresh_values(self):
        task = TaskSpec("resources", "resources", goal=GoalConfig(resource_targets={"gold": 500}))
        def inventory(name, amount):
            result = frame_at(name, [], "village")
            result.observations.update(resource_source="village_inventory", resources={"gold": amount})
            return result
        frames = [inventory("animating.png", 0), inventory("still-zero.png", 0),
                  inventory("settled-a.png", 500), inventory("settled-b.png", 500),
                  inventory("settled-c.png", 500)]
        with patch("autococ.flow.return_to_village", side_effect=frames) as home, \
                patch("autococ.objectives.time.sleep"):
            progress = read_progress(self.session, task)
        self.assertTrue(progress.completed)
        self.assertEqual(progress.current, {"gold": 500})
        self.assertEqual(home.call_count, 5)

    def test_resource_inventory_never_stabilizes_or_remains_unknown(self):
        task = TaskSpec("resources", "resources", goal=GoalConfig(resource_targets={"gold": 500}))
        for values in ([0, 100, 200, 300, 400], [None] * 5):
            with self.subTest(values=values):
                frames = []
                for index, value in enumerate(values):
                    result = frame_at(f"inventory-{index}.png", [], "village")
                    result.observations.update(resource_source="village_inventory", resources={"gold": value})
                    frames.append(result)
                with patch("autococ.flow.return_to_village", side_effect=frames) as home, \
                        patch("autococ.objectives.time.sleep"), self.assertRaises(FlowError):
                    read_progress(self.session, task)
                self.assertEqual(home.call_count, 5)

    def test_full_storage_waits_for_capacity_and_inventory_to_stabilize(self):
        task = TaskSpec("resources", "resources", goal=GoalConfig(full_resources=("gold",)))
        frames = []
        for index, (amount, capacity) in enumerate(((0, None), (500, None), (500, 500),
                                                     (500, 500), (500, 500))):
            result = frame_at(f"stock-{index}.png", [], "village")
            result.observations.update(resource_source="village_inventory", resources={"gold": amount},
                                       resource_capacities={"gold": capacity})
            frames.append(result)
        with patch("autococ.flow.return_to_village", side_effect=frames), \
                patch("autococ.objectives.time.sleep"):
            progress = read_progress(self.session, task)
        self.assertTrue(progress.completed)
        self.assertEqual(progress.target, {"gold": 500})

    def test_battle_limit_only_resource_task_does_not_wait_for_stock(self):
        task = TaskSpec("resources", "resources")
        home_frame = frame_at("home.png", [], "village")
        home_frame.observations.update(resource_source="village_inventory", resources={"gold": 0})
        with patch("autococ.flow.return_to_village", return_value=home_frame) as home:
            progress = read_progress(self.session, task)
        self.assertEqual(progress.reason, "bounded_battles_only")
        home.assert_called_once()

    def test_reused_inventory_frame_is_not_stability_evidence(self):
        task = TaskSpec("resources", "resources", goal=GoalConfig(resource_targets={"gold": 500}))
        home_frame = frame_at("home.png", [], "village")
        home_frame.observations.update(resource_source="village_inventory", resources={"gold": 500})
        with patch("autococ.flow.return_to_village", return_value=home_frame) as home, \
                patch("autococ.objectives.time.sleep"), self.assertRaises(FlowError):
            read_progress(self.session, task)
        self.assertEqual(home.call_count, 2)


if __name__ == "__main__":
    unittest.main()
