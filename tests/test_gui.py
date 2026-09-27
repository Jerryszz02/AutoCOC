from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch

from autococ.gui import AutoCOCApp
from autococ.desktop import load_options, settings_path
from autococ.config import load_config
from autococ.routine_config import GoalConfig, RoutineConfig, TaskSpec


class GUITests(unittest.TestCase):
    def test_vision_agent_settings_reach_worker_and_persist(self):
        self.app.agent_enabled.set(True)
        self.app.agent_values["model_dir"].set("models/current")
        self.app.agent_values["layout_profile"].set("profiles/calibrated.json")
        self.app.agent_values["preparation_reserve_sec"].set("6")
        self.app._save()
        loaded = load_options(settings_path(self.path), load_config(self.path))
        self.assertTrue(loaded.vision_agent.enabled)
        self.assertEqual(loaded.vision_agent.preparation_reserve_sec, 6)
        self.app.start()
        options = self.controller.start.call_args.args[1]
        self.assertEqual(options.vision_agent.model_dir, "models/current")
        self.assertTrue(options.dry_run)

    @classmethod
    def setUpClass(cls):
        cls.interpreter = tk.Tk()
        cls.interpreter.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.interpreter.destroy()

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.toml"
        self.path.write_text("", encoding="utf-8")
        self.root = tk.Toplevel(self.interpreter)
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.controller = Mock()
        self.controller.running = False
        self.app = AutoCOCApp(self.root, self.path, controller=self.controller)

    def test_opening_window_does_not_start_or_connect(self):
        self.controller.start.assert_not_called()
        self.assertEqual(self.app.mode.get(), "离线预演")
        self.assertEqual(self.app.start_button.cget("text"), "开始预演")

    def test_selected_tasks_are_passed_to_background_worker(self):
        for var in self.app.task_vars.values():
            var.set(False)
        self.app.task_vars["resources"].set(True)
        self.app.task_vars["event"].set(True)
        self.app.start()
        options = self.controller.start.call_args.args[1]
        self.assertEqual(options.tasks, ("launch", "battle"))
        self.assertEqual([task.id for task in options.routine.tasks if task.enabled], ["resources", "event"])
        self.assertTrue(options.dry_run)
        self.assertTrue(self.app.start_button.instate(["disabled"]))
        self.app.stop()
        self.controller.stop.assert_called_once_with()

    def test_invalid_settings_never_start_worker(self):
        self.app.values["minutes"].set("nan")
        with patch("autococ.gui.messagebox.showerror") as error:
            self.app.start()
        error.assert_called_once()
        self.controller.start.assert_not_called()

    def test_no_checked_task_never_starts_worker(self):
        for var in self.app.task_vars.values():
            var.set(False)
        with patch("autococ.gui.messagebox.showerror") as error:
            self.app.start()
        error.assert_called_once()
        self.controller.start.assert_not_called()

    def test_two_edge_strategy_reaches_worker(self):
        self.app.task_fields["resources"]["strategy"].set("two_edge")
        self.app.start()
        self.assertEqual(self.controller.start.call_args.args[1].strategy, "two_edge")

    def test_edrag_strategy_reaches_worker(self):
        self.app.task_fields["resources"]["strategy"].set("edrag_line")
        self.app.start()
        self.assertEqual(self.controller.start.call_args.args[1].strategy, "edrag_line")

    def test_configuration_page_and_independent_battle_settings(self):
        self.app._open_task_config("event")
        self.assertEqual(self.app.pages.index("current"), 1)
        self.app.task_fields["resources"]["gold"].set("800000")
        self.app.task_fields["resources"]["target_gold"].set("25000000")
        self.app.task_fields["event"]["adapter_path"].set("events/current.toml")
        self.app.task_fields["event"]["target"].set("100")
        self.app.task_fields["clan_games"]["building_type"].set("air_defense")
        self.app.task_fields["clan_games"]["target"].set("5")
        options = self.app._options()
        tasks = {task.id: task for task in options.routine.tasks}
        self.assertEqual(tasks["resources"].resource_filter.min_gold, 800000)
        self.assertEqual(tasks["resources"].goal.resource_targets["gold"], 25000000)
        self.assertEqual((tasks["event"].goal.adapter_path, tasks["event"].goal.target), ("events/current.toml", 100))
        self.assertEqual((tasks["clan_games"].goal.building_type, tasks["clan_games"].goal.target), ("air_defense", 5))

    def test_localized_goal_controls_store_internal_values(self):
        self.app.selected_task.set("resources")
        self.assertEqual(self.app.selected_task_label.get(), "刷资源")
        self.assertIn("刷活动", self.app.task_choice.cget("values"))
        self.app.task_fields["resources"]["filter"].set("1")
        self.app.full_flags["resources"]["gold"].set(True)
        self.app._sync_full_resources("resources")
        self.assertEqual(self.app._options().routine.tasks[3].goal.full_resources, ("gold",))
        self.app.selected_task.set("clan_games")
        goal_children = [child for widget in self.app.task_config.winfo_children() for child in widget.winfo_children()]
        goal_labels = [child.cget("text") for child in goal_children if isinstance(child, ttk.Label)]
        self.assertIn("目标建筑", goal_labels)
        self.assertNotIn("金币库存", goal_labels)
        building = next(child for child in goal_children if isinstance(child, ttk.Combobox)
                        and "防空火箭" in child.cget("values"))
        building.set("法术工厂")
        building.event_generate("<<ComboboxSelected>>")
        self.assertEqual(self.app._options().routine.tasks[-1].goal.building_type, "spell_factory")

    def test_task_order_and_battle_progress_are_separate_from_task_success(self):
        self.app.selected_task.set("resources")
        self.app._move_task(-1)
        self.assertEqual(self.app.task_order[2:4], ["resources", "donate"])
        self.app._event({"kind": "battle_progress", "task_id": "resources", "phase": "筛选对手",
                         "battle_number": 3, "battles_completed": 2, "battles_won": 1,
                         "resources": {"gold": 750000}, "goal_current": 2, "goal_target": 5})
        self.assertEqual(self.app.counters["battles_completed"].get(), "2")
        self.assertEqual(self.app.counters["battles_won"].get(), "1")
        self.assertIn("第 3 场", self.app.run_label.get())
        self.assertIn("目标 2 / 5", self.app.progress_label.get())

    def test_unsaved_task_edits_survive_switch_and_order_is_persisted(self):
        self.app.selected_task.set("resources")
        self.app.task_fields["resources"]["gold"].set("500000")
        self.app.strategy_text.delete("1.0", "end")
        self.app.strategy_text.insert("1.0", "# resources draft")
        self.app.selected_task.set("event")
        self.app.task_fields["event"]["target"].set("250")
        self.app.strategy_text.delete("1.0", "end")
        self.app.strategy_text.insert("1.0", "# event draft")
        self.app.selected_task.set("resources")
        self.assertEqual(self.app.task_fields["resources"]["gold"].get(), "500000")
        self.assertEqual(self.app.strategy_text.get("1.0", "end-1c"), "# resources draft")
        self.app._move_task(-1)
        self.app._save()
        loaded = load_options(settings_path(self.path), load_config(self.path))
        self.assertEqual([task.id for task in loaded.routine.tasks][:4], ["collect", "request", "resources", "donate"])
        values = {task.id: task for task in loaded.routine.tasks}
        self.assertEqual(values["resources"].resource_filter.min_gold, 500000)
        self.assertEqual(values["event"].goal.target, 250)

    def test_custom_task_ids_survive_gui_round_trip(self):
        routine = RoutineConfig((TaskSpec("gold_run", "resources", goal=GoalConfig(resource_targets={"gold": 1000000})),
                                 TaskSpec("event_run", "event", enabled=False)))
        options = replace(self.app._options(), routine=routine)
        self.app._apply(options)
        self.assertEqual(self.app.task_order[:2], ["gold_run", "event_run"])
        rebuilt = {task.id: task for task in self.app._options().routine.tasks}
        self.assertEqual(rebuilt["gold_run"].goal.resource_targets["gold"], 1000000)
        self.assertFalse(rebuilt["event_run"].enabled)

    def test_multiple_custom_battle_ids_keep_order_settings_and_distinct_run_labels(self):
        routine = RoutineConfig((
            TaskSpec("farm_high", "resources", max_battles=2,
                     goal=GoalConfig(resource_targets={"gold": 1200000})),
            TaskSpec("farm_low", "resources", max_battles=7,
                     goal=GoalConfig(resource_targets={"elixir": 900000})),
            TaskSpec("event_run", "event", enabled=False)),
            maintenance_interval_sec=600)
        self.app._apply(replace(self.app._options(), routine=routine))
        self.app.selected_task.set("farm_low")
        self.app._move_task(-1)
        self.app._save()
        self.app._load()
        self.assertEqual(self.app.task_order[:3], ["farm_low", "farm_high", "event_run"])
        rebuilt = {task.id: task for task in self.app._options().routine.tasks}
        self.assertEqual(rebuilt["farm_high"].max_battles, 2)
        self.assertEqual(rebuilt["farm_high"].goal.resource_targets, {"gold": 1200000})
        self.assertEqual(rebuilt["farm_low"].max_battles, 7)
        self.assertEqual(rebuilt["farm_low"].goal.resource_targets, {"elixir": 900000})
        self.assertEqual(self.app._options().routine.maintenance_interval_sec, 600)
        self.app.start()
        self.assertEqual(self.app.task_table.item("farm_low", "values")[0], "刷资源（farm_low）")
        self.app._event({"kind": "task_started", "task": "farm_high"})
        self.app._event({"kind": "battle_progress", "task_id": "farm_low", "phase": "筛选对手",
                         "battle_number": 2})
        self.assertIn("刷资源（farm_high）", self.app.log.get("1.0", "end"))
        self.assertIn("刷资源（farm_low）", self.app.run_label.get())

    def test_dictionary_goal_progress_is_readable(self):
        self.app._event({"kind": "battle_progress", "task_id": "resources", "phase": "结算",
                         "battle_number": 2, "resources": {"gold": None},
                         "resource_gains": {"gold": 1000, "elixir": None},
                         "goal_current": {"gold": 300}, "goal_target": {"gold": 500}})
        self.assertIn("刷资源", self.app.run_label.get())
        self.assertIn("金币: 300 / 500", self.app.progress_label.get())
        self.assertIn("库存 金币: 未知", self.app.progress_label.get())
        self.assertIn("已核验收益 金币: 1000 / 圣水: 未知", self.app.progress_label.get())
        self.app._event({"kind": "battle_progress", "task_id": "resources", "phase": "执行打法"})
        self.assertIn("金币: 300 / 500", self.app.progress_label.get())
        self.app._event({"kind": "task_result", "task": "resources", "status": "succeeded",
                         "reason": "goal_reached", "metrics": {"goals_completed": 1}})
        self.assertEqual(self.app.counters["goals_completed"].get(), "1")
        self.app._event({"kind": "battle_progress", "task_id": "event", "phase": "检查目标"})
        self.assertNotIn("300 / 500", self.app.progress_label.get())

    def test_strategy_editor_validates_before_replacing_file(self):
        strategy_path = self.path.parent / "battle.toml"
        self.app.selected_task.set("resources")
        self.app.task_fields["resources"]["strategy_file"].set("battle.toml")
        valid = 'id = "single_edge"\nlabel = "单边"\narmy_mode = "captured"\n[[steps]]\naction = "deploy_troop"\nunit_id = "*"\ncount = "all"\nedge = "west"\n'
        self.app.strategy_text.delete("1.0", "end")
        self.app.strategy_text.insert("1.0", valid)
        self.app._save_strategy_file()
        self.assertEqual(strategy_path.read_text(encoding="utf-8"), valid)
        self.app.strategy_text.delete("1.0", "end")
        self.app.strategy_text.insert("1.0", 'id = "bad"\nlabel = "无效"\n[[steps]]\naction = "unknown"\n')
        with patch("autococ.gui.messagebox.showerror") as error:
            self.app._save_strategy_file()
        error.assert_called_once()
        self.assertEqual(strategy_path.read_text(encoding="utf-8"), valid)
        self.assertFalse(strategy_path.with_name(strategy_path.name + ".tmp").exists())

    def test_strategy_preset_lists_and_loads_relative_to_config(self):
        folder = self.path.parent / "strategies"
        folder.mkdir()
        preset = folder / "single_edge.toml"
        preset.write_text('id = "single_edge"\nlabel = "单边"\narmy_mode = "captured"\n[[steps]]\naction = "deploy_troop"\nunit_id = "*"\ncount = "all"\nedge = "first"\n', encoding="utf-8")
        self.app._strategy_file_choices()
        self.assertIn("single_edge.toml", self.app.strategy_presets.cget("values"))
        self.app.selected_task.set("resources")
        self.app.strategy_preset.set("single_edge.toml")
        self.app._choose_strategy_preset()
        self.assertEqual(self.app.task_fields["resources"]["strategy_file"].get(), str(Path("strategies") / "single_edge.toml"))
        self.assertIn('id = "single_edge"', self.app.strategy_text.get("1.0", "end"))

    def test_gui_completion_distinguishes_simulation_from_success(self):
        self.app._event({"kind": "finished", "report": None,
                         "summary": {"mode": "dry-run", "successes": 0, "failures": 0,
                                     "skipped": 0, "simulated": 5, "stop_reason": "dry-run planning complete"}})
        self.assertEqual(self.app.counters["successes"].get(), "0")
        self.assertEqual(self.app.counters["simulated"].get(), "5")
        self.assertIn("报告未保存", self.app.status.get())

    def test_close_requests_stop_and_keeps_window_until_worker_exits(self):
        self.controller.running = True
        self.app.close()
        self.controller.stop.assert_called_once_with()
        self.assertTrue(self.app.closing)
        self.assertTrue(self.root.winfo_exists())

    def test_finished_interruption_does_not_leave_task_running(self):
        self.app.start()
        self.app._event({"kind": "task_started", "task": "resources"})
        self.app._event({"kind": "finished", "report": None, "summary": {
            "mode": "live", "successes": 0, "failures": 1, "stop_reason": "interrupted by user",
            "task_results": [{"task": "resources", "status": "failed", "reason": "interrupted by user"}]}})
        self.assertEqual(self.app.task_table.item("resources", "values")[1], "失败")
        self.assertEqual(self.app.task_table.item("collect", "values")[1], "未执行")
