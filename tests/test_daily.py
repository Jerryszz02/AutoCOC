from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from autococ.battle import BattleTarget, score_target
from autococ.config import BattleConfig, load_config
from autococ.errors import CapabilityUnavailable, ConfigError, FlowError
from autococ.flow import FlowRunner
from autococ.objectives import GoalProgress, ProgressAdapter, parse_progress, resource_progress
from autococ.reporting import RunStats, TaskResult, summarize_run
from autococ.routine_config import (ResourceFilter, GoalConfig, TaskSpec, RoutineConfig,
                                    routine_from_dict, routine_to_dict, default_routine)
from autococ.scene import SceneSnapshot
from autococ.session import GameSession


class ResourceFilteringTests(unittest.TestCase):
    def test_strategy_does_not_bypass_explicit_and_thresholds(self):
        rule = ResourceFilter(min_gold=200, min_elixir=100, min_dark_elixir=30, min_total=350)
        for strategy in ("two_edge", "edrag_line", "verified"):
            config = BattleConfig(strategy=strategy, resource_filter=rule, max_searches=2)
            for target, accepted in ((BattleTarget(200,150,30),True), (BattleTarget(199,500,30),False),
                                     (BattleTarget(300,99,30),False), (BattleTarget(200,100,30),False),
                                     (BattleTarget(200,150,None),False), (BattleTarget(None,900,40),False)):
                with self.subTest(strategy=strategy,target=target):
                    self.assertEqual(score_target(target, config, searches=2).should_attack, accepted)
                    self.assertFalse(score_target(target, config, searches=2).can_search_next)

    def test_legacy_migration_preserves_threshold_bypass(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"config.toml"
            for strategy in ("two_edge", "edrag_line", "verified"):
                path.write_text(f'[battle]\nstrategy="{strategy}"\nmin_expected_resources=765432\n')
                routine=default_routine(load_config(path))
                rule=next(t for t in routine.tasks if t.kind=="resources").resource_filter
                self.assertEqual(rule.enabled,strategy=="verified")
                self.assertEqual(rule.min_total,765432)

    def test_disabled_filter_accepts_unknown_but_budget_still_applies(self):
        config=BattleConfig(resource_filter=ResourceFilter(enabled=False),max_searches=1)
        self.assertTrue(score_target(BattleTarget(),config).should_attack)
        self.assertFalse(score_target(BattleTarget(),config,searches=2).should_attack)

    def test_explicit_filter_and_strategy_file_survive_desktop_migration(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\nstrategy="two_edge"\nstrategy_file="custom.toml"\n'
                            '[battle.resource_filter]\nenabled=true\nmin_dark_elixir=9000\n', encoding="utf-8")
            config = load_config(path)
            migrated = next(t for t in default_routine(config).tasks if t.kind == "resources")
            self.assertEqual(migrated.strategy_file, "custom.toml")
            self.assertEqual(migrated.resource_filter, config.battle.resource_filter)

    def test_toml_single_resource_floor_does_not_enable_hidden_total_floor(self):
        routine = routine_from_dict({"tasks": [{"id": "oil", "kind": "resources",
            "resource_filter": {"enabled": True, "min_dark_elixir": 9000}}]})
        rule = routine.tasks[0].resource_filter
        self.assertIsNone(rule.min_total)
        config = BattleConfig(resource_filter=rule)
        self.assertTrue(score_target(BattleTarget(dark_elixir=9001), config).should_attack)
        self.assertFalse(score_target(BattleTarget(dark_elixir=None), config).should_attack)


class GoalTests(unittest.TestCase):
    def frame(self, texts=(), **observations):
        return SceneSnapshot("village",.99,Path("frame.png"),{"baseline_resolution":[1280,720],
            "resource_source": "village_inventory",
            "ocr":[{"text":t,"confidence":.99,"bbox":[20,20+i*50,120,40+i*50]} for i,t in enumerate(texts)],**observations})

    def test_stock_goal_needs_every_requested_resource(self):
        task=TaskSpec("farm","resources",goal=GoalConfig(resource_targets={"gold":100,"elixir":200}))
        self.assertFalse(resource_progress(self.frame(resources={"gold":100,"elixir":199}),task).completed)
        self.assertTrue(resource_progress(self.frame(resources={"gold":101,"elixir":201}),task).completed)
        with self.assertRaises(FlowError):
            resource_progress(self.frame(resources={"gold":100}),task)

    def test_enemy_amount_never_counts_as_achieved_stock(self):
        task = TaskSpec("farm", "resources", goal=GoalConfig(resource_targets={"gold": 100}))
        with self.assertRaises(FlowError):
            resource_progress(self.frame(resources={"gold": 9999}, resource_source="enemy_available"), task)

    def test_event_adapter_requires_explicit_known_scoring_conditions(self):
        adapter = ProgressAdapter("event", "event", "活动", (0, 0, 300, 300),
                                  entry_text="活动", active_label="进行中")
        with self.assertRaises(ConfigError):
            adapter.validate()
        replace(adapter, scoring_condition="battle").validate()
        with self.assertRaises(ConfigError):
            replace(adapter, scoring_condition="deploy_units").validate()
        replace(adapter, scoring_condition="deploy_units", scoring_units=("electro_dragon",)).validate()

    def test_full_storage_never_infers_capacity(self):
        task=TaskSpec("farm","resources",goal=GoalConfig(full_resources=("gold",)))
        with self.assertRaises(CapabilityUnavailable):
            resource_progress(self.frame(resources={"gold":99999999}),task)
        self.assertTrue(resource_progress(self.frame(resources={"gold":500},resource_capacities={"gold":500}),task).completed)

    def test_event_progress_requires_exact_active_page(self):
        adapter=ProgressAdapter("event-x","event","活动甲",(0,0,300,300),entry_text="活动",active_label="进行中", unavailable_label="已结束")
        task=TaskSpec("event","event",goal=GoalConfig(target=50))
        progress=parse_progress(self.frame(("活动甲","进行中","50 / 100")),adapter,task)
        self.assertTrue(progress.completed)
        self.assertFalse(parse_progress(self.frame(("活动甲","已结束","50 / 100")),adapter,task).available)
        with self.assertRaises(FlowError):
            parse_progress(self.frame(("活动甲","看不清","50 / 100")),adapter,task)
        with self.assertRaises(FlowError):
            parse_progress(self.frame(("活动乙","进行中","50 / 100")),adapter,task)

    def test_clan_task_uses_game_denominator_not_custom_partial_count(self):
        adapter=ProgressAdapter("clan-air","clan_games","防空火箭任务",(0,0,300,300),
                                entry_text="部落竞赛",accepted_label="进行中",building_type="air_defense")
        task=TaskSpec("clan","clan_games",goal=GoalConfig(building_type="air_defense"))
        self.assertFalse(parse_progress(self.frame(("防空火箭任务","进行中","1 / 5")),adapter,task).completed)
        with self.assertRaises(ConfigError):
            parse_progress(self.frame(("防空火箭任务","进行中","1 / 5")),adapter,replace(task,goal=replace(task.goal,target=1)))


class DailyRunnerTests(unittest.TestCase):
    def test_cancelled_vision_battle_retains_partial_receipt(self):
        receipt = {"plan_id": "frozen", "input_count": 5, "attempted_placements": 1,
                   "input_sent": False, "verified": False, "actions": [{"status": "input_sent"}]}
        self.session.prepared_battle_receipt = receipt
        with patch("autococ.daily.read_progress", return_value=GoalProgress(False)), \
                patch("autococ.combat.run_battle", side_effect=KeyboardInterrupt):
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("farm", "resources"),)))
        result = stats.task_results[-1]
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.metrics["deployment"], receipt)
        self.assertFalse(result.metrics["vision_agent"]["consumption_verified"])
        self.assertEqual(stats.battles_completed, 0)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);path=root/"config.toml";path.write_text("")
        config=load_config(path)
        self.config=replace(config,runtime=replace(config.runtime,report_dir=root/"reports"))
        self.frame=SceneSnapshot("village",.99,root/"home.png",{
            "buttons":[{"name":"attack"},{"name":"shop"}],"resources":{"gold":100},"village_type":"home"})
        self.session=Mock(spec=GameSession)
        self.session.config=self.config
        self.session.observe.return_value=self.frame
        self.session.last_snapshot=self.frame
        self.session.events_path=root/"events.jsonl"
        self.factory=Mock(return_value=self.session)
        self.events=[]

    def runner(self, **options):
        return FlowRunner(options.pop("config",self.config),Mock(),"local",session_factory=self.factory,
                          progress=self.events.append,**options)

    def success(self):
        return TaskResult("battle","succeeded","verified",evidence=[self.frame.screenshot_path],
                          metrics={"returned_home":True,"rounds_completed":1,"victory":False})

    def test_building_event_requires_targeted_strategy_before_battle(self):
        routine = RoutineConfig((TaskSpec("building-event", "event", max_battles=1),))
        progress = GoalProgress(False, values={"scoring_condition": "destroy_building",
                                               "building_type": "air_defense"})
        with patch("autococ.daily.read_progress", return_value=progress), \
                patch("autococ.combat.run_battle") as battle:
            stats = self.runner().run_routine(routine)
        battle.assert_not_called()
        self.assertEqual(stats.battles_completed, 0)
        self.assertTrue(any(r.status == "not_supported" and "targeted strategy" in r.reason
                            for r in stats.task_results))

    def test_multiple_goals_run_sequentially_with_independent_options(self):
        routine=RoutineConfig((TaskSpec("gold","resources",strategy="two_edge",max_battles=1),
                               TaskSpec("oil","resources",strategy="edrag_line",max_battles=1,
                                        resource_filter=ResourceFilter(min_dark_elixir=10,min_total=None))))
        seen=[]
        def battle(session):
            seen.append(session.config.battle);return self.success()
        with patch("autococ.daily.read_progress",return_value=GoalProgress(False)),patch("autococ.combat.run_battle",side_effect=battle):
            stats=self.runner().run_routine(routine)
        self.assertEqual([x.strategy for x in seen],["two_edge","edrag_line"])
        self.assertEqual(seen[1].resource_filter.min_dark_elixir,10)
        self.assertEqual(stats.battles_completed,2)
        self.assertEqual(stats.battles_won,0)
        self.assertEqual(stats.goals_completed,0)
        receipts=[r for r in stats.task_results if r.metrics.get("battle_id")]
        self.assertEqual(len(set(r.metrics["battle_id"] for r in receipts)),2)
        self.assertEqual([r.task for r in stats.task_results if r.status=="limited"],["gold","oil"])

    def test_goal_already_complete_never_searches(self):
        with patch("autococ.daily.read_progress",return_value=GoalProgress(True,100,100,evidence=(self.frame.screenshot_path,))),patch("autococ.combat.run_battle") as battle:
            stats=self.runner().run_routine(RoutineConfig((TaskSpec("gold","resources"),)))
        battle.assert_not_called();self.assertEqual(stats.goals_completed,1)
        self.assertEqual(stats.battles_completed,0)

    def test_periodic_collection_rechecks_goal_before_starting_battle(self):
        import time
        clock = [time.monotonic()]
        goals = iter((False, True))
        def read(session, task):
            clock[0] += 601
            return GoalProgress(next(goals), evidence=(self.frame.screenshot_path,))
        collect = Mock(return_value=TaskResult("collect", "succeeded", "verified",
                                               evidence=[self.frame.screenshot_path]))
        runner = self.runner()
        routine = RoutineConfig((TaskSpec("collect", "collect"), TaskSpec("farm", "resources")), 600)
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
                patch("autococ.daily.read_progress", side_effect=read), \
                patch.object(runner, "_handler", return_value=collect), \
                patch("autococ.combat.run_battle") as battle:
            stats = runner.run_routine(routine)
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(stats.goals_completed, 1)
        battle.assert_not_called()

    def test_goal_receipt_links_before_after_and_gains_never_use_inventory(self):
        before = GoalProgress(False, values={"gold": 100}, evidence=(self.frame.screenshot_path,))
        after = GoalProgress(True, values={"gold": 150}, evidence=(self.frame.screenshot_path,))
        result = self.success()
        result.metrics.update(loot_gold=20, bonus_gold=5, resources_after={"gold": 150})
        with patch("autococ.daily.read_progress", side_effect=[before, after]), patch("autococ.combat.run_battle", return_value=result):
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("gold", "resources"),)))
        battles = [r for r in stats.task_results if r.metrics.get("receipt_kind") == "battle"]
        progress = next(r for r in stats.task_results if r.metrics.get("receipt_kind") == "goal_progress")
        self.assertEqual(stats.battles_completed, 1)
        self.assertEqual(stats.goals_completed, 1)
        self.assertEqual(progress.metrics["battle_id"], battles[0].metrics["battle_id"])
        self.assertEqual(progress.metrics["before"]["values"]["gold"], 100)
        self.assertEqual(progress.metrics["after"]["values"]["gold"], 150)
        returned = next(e for e in self.events if e.get("phase") == "已回村")
        self.assertEqual(returned["resource_gains"]["gold"], 25)
        self.assertIsNone(returned["resource_gains"]["elixir"])

    def test_builder_base_is_rejected_before_any_task_input(self):
        self.frame.observations["village_type"]="builder_base"
        with patch("autococ.combat.run_battle") as battle:
            stats=self.runner().run_routine(RoutineConfig((TaskSpec("gold","resources"),)))
        battle.assert_not_called();self.assertEqual(stats.battles_completed,0)
        self.assertTrue(any(r.status=="not_supported" for r in stats.task_results))

    def test_failed_startup_recovery_stops_even_if_last_frame_looks_like_home(self):
        failed = TaskResult("launch", "failed", "Connection recovery timed out",
                            evidence=[self.frame.screenshot_path])
        with patch("autococ.flow.verify_home", return_value=failed), \
                patch("autococ.daily.read_progress") as goal, \
                patch("autococ.combat.run_battle") as battle:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("farm", "resources"),)))
        goal.assert_not_called()
        battle.assert_not_called()
        self.assertEqual(stats.failures, 1)
        self.assertEqual(stats.task_results[0].task, "launch")
        self.assertIn("Connection recovery timed out", stats.stop_reason)

    def test_event_write_failure_still_preserves_final_report(self):
        self.session.event.side_effect=OSError("disk full")
        stats=self.runner().run_routine(RoutineConfig((TaskSpec("gold","resources"),)))
        self.assertGreater(stats.failures,0)
        self.assertTrue((self.config.runtime.report_dir/f"run-{stats.run_id}.json").exists())

    def test_missing_adapter_is_visible_and_safe_next_task_runs(self):
        with patch("autococ.daily.read_progress",side_effect=[CapabilityUnavailable("not sampled"),GoalProgress(False),GoalProgress(False)]),patch("autococ.combat.run_battle",return_value=self.success()):
            stats=self.runner().run_routine(RoutineConfig((TaskSpec("event","event"),TaskSpec("farm","resources",max_battles=1))))
        self.assertTrue(any(r.task=="event" and r.status=="not_supported" for r in stats.task_results))
        self.assertEqual(stats.battles_completed,1)

    def test_failed_battle_does_not_run_later_tasks(self):
        result=TaskResult("battle","failed","unknown scene",evidence=[self.frame.screenshot_path])
        with patch("autococ.daily.read_progress",return_value=GoalProgress(False)),patch("autococ.combat.run_battle",return_value=result) as battle:
            stats=self.runner().run_routine(RoutineConfig((TaskSpec("first","resources"),TaskSpec("second","resources"))))
        self.assertEqual(battle.call_count,1);self.assertEqual(stats.battles_completed,0)
        self.assertEqual(stats.failures, 1)

    def test_unavailable_army_skips_task_without_counting_a_battle(self):
        skipped = TaskResult("battle", "skipped", "unit_locked:barbarian",
                             metrics={"returned_home": True, "search_count": 0})
        with patch("autococ.daily.read_progress", return_value=GoalProgress(False)), \
                patch("autococ.combat.run_battle", side_effect=[skipped, self.success()]) as battle:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("first", "resources"),
                                                           TaskSpec("second", "resources", max_battles=1))))
        self.assertEqual(battle.call_count, 2)
        self.assertEqual(stats.failures, 0)
        self.assertEqual(stats.battles_completed, 1)
        self.assertEqual(stats.goals_completed, 0)
        self.assertTrue(any(result.task == "first" and result.status == "skipped"
                            for result in stats.task_results))

    def test_unavailable_army_never_switches_task_if_safe_home_is_uncertain(self):
        skipped = TaskResult("battle", "skipped", "unit_locked:barbarian")
        launched = TaskResult("launch", "succeeded", "home verified", evidence=[self.frame.screenshot_path])
        with patch("autococ.flow.verify_home", return_value=launched), \
                patch("autococ.flow.return_to_village", side_effect=FlowError("Home unknown")), \
                patch("autococ.daily.read_progress", return_value=GoalProgress(False)), \
                patch("autococ.combat.run_battle", return_value=skipped) as battle:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("first", "resources"),
                                                           TaskSpec("second", "resources"))))
        self.assertEqual(battle.call_count, 1)
        self.assertEqual(stats.battles_completed, 0)
        self.assertEqual(stats.failures, 1)
        self.assertIn("Home unknown", stats.stop_reason)

    def test_settled_partial_battle_counts_once_then_stops_queue(self):
        result = replace(self.success(), status="failed", reason="Last action unverified",
                         metrics={**self.success().metrics, "victory": True})
        with patch("autococ.daily.read_progress", return_value=GoalProgress(False)), \
                patch("autococ.combat.run_battle", return_value=result) as battle:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("first", "resources"),
                                                            TaskSpec("second", "resources"))))
        self.assertEqual(battle.call_count, 1)
        self.assertEqual((stats.battles_completed, stats.battles_won, stats.failures), (1, 1, 1))
        self.assertEqual(stats.goals_completed, 0)
        returned = [event for event in self.events if event.get("phase") == "已回村"]
        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0]["battles_completed"], 1)

    def test_completed_battle_count_survives_event_log_failure(self):
        runner = self.runner()
        with patch("autococ.daily.read_progress", return_value=GoalProgress(False)), \
                patch("autococ.combat.run_battle", return_value=self.success()), \
                patch.object(runner, "_record_event", side_effect=[True, False]):
            stats = runner.run_routine(RoutineConfig((TaskSpec("first", "resources"),)))
        self.assertEqual(stats.battles_completed, 1)
        self.assertEqual(stats.failures, 1)
        self.assertTrue((self.config.runtime.report_dir / f"run-{stats.run_id}.json").exists())

    def test_returning_from_search_does_not_count_as_completed_battle(self):
        result = TaskResult("battle", "limited", "No qualifying opponent",
                            metrics={"returned_home": True})
        with patch("autococ.daily.read_progress", return_value=GoalProgress(False)), \
                patch("autococ.combat.run_battle", return_value=result):
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("first", "resources"),)))
        self.assertEqual(stats.battles_completed, 0)

    def test_dry_run_and_preexisting_stop_never_connect(self):
        routine=RoutineConfig((TaskSpec("farm","resources"),))
        config=replace(self.config,runtime=replace(self.config.runtime,dry_run=True))
        stats=self.runner(config=config).run_routine(routine)
        self.factory.assert_not_called();self.assertEqual(stats.simulated,1)
        self.assertEqual(stats.battles_completed,0)
        event=Event();event.set()
        stats=self.runner(stop_event=event).run_routine(routine)
        self.factory.assert_not_called()
        self.assertEqual(stats.task_results[-1].status,"cancelled")

    def test_roundtrip_and_duplicate_validation(self):
        routine=RoutineConfig((TaskSpec("farm","resources",goal=GoalConfig(full_resources=("gold",))),),600)
        self.assertEqual(routine_from_dict(routine_to_dict(routine)),routine)
        with self.assertRaises(ConfigError):
            RoutineConfig((routine.tasks[0],routine.tasks[0])).validate()

    def test_dry_report_never_exposes_live_counters(self):
        stats=RunStats(mode="dry-run",battles_completed=8,battles_won=3,goals_completed=2)
        payload=summarize_run(stats)
        self.assertEqual([payload[k] for k in ("battles_completed","battles_won","goals_completed")],[0,0,0])


if __name__=="__main__":
    unittest.main()
