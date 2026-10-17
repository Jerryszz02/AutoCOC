from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

from autococ.battle import BattleTarget, score_target
from autococ.config import BattleConfig, load_config
from autococ.daily import _event_reward_task
from autococ.errors import CapabilityUnavailable, ConfigError, FlowError, StopRequested
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
    def event_ore_read(self, phase, *, shiny=15662, glowy=1225, starry=77,
                       gold=100, elixir=200, dark_elixir=30, gems=4):
        root = self.frame.screenshot_path.parent
        frame = root / f"{phase}-home-2.png"
        frame_id = f"{phase}-home-2"
        observed_at = time.monotonic()
        home = SceneSnapshot("village", .99, frame, {
            "village_type": "home", "resource_source": "village_inventory",
            "buttons": [{"name": "attack"}, {"name": "shop"}], "ocr": [],
            "frame_id": frame_id,
            "observed_at_monotonic": observed_at,
            "resources": {"gold": gold, "elixir": elixir,
                          "dark_elixir": dark_elixir, "gems": gems}})
        capacities = {"shiny": 50000, "glowy": 5000, "starry": 1000}
        balances = {"shiny": shiny, "glowy": glowy, "starry": starry}
        receipt = {"status": "stable_inventory_read", "returned_home": True,
                   "inventory": {"status": "stable_inventory_read",
                                 "ores": {name: {"amount": amount,
                                                 "capacity": capacities[name]}
                                          for name, amount in balances.items()},
                                 "frames": [{"frame": str(root / f"{phase}-ore-{index}.png"),
                                             "frame_id": f"{phase}-ore-{index}",
                                             "frame_sha256": f"{index + 1:064x}"}
                                            for index in (1, 2)]},
                   "selected": {"frame": str(root / f"{phase}-selected.png")},
                   "close": {"status": "home", "frame": str(root / f"{phase}-close.png")},
                   "home_evidence": [
                       {"frame": str(root / f"{phase}-home-1.png"),
                        "frame_id": f"{phase}-home-1", "frame_sha256": "a" * 64,
                        "observed_at_monotonic": observed_at - .01},
                       {"frame": str(frame), "frame_id": frame_id,
                        "frame_sha256": "b" * 64,
                        "observed_at_monotonic": observed_at}]}
        return home, receipt

    def event_ore_reader(self, *reads):
        steps = iter(reads)
        def read(_session, *, timeout_sec):
            home, receipt = next(steps)
            self.session.last_snapshot = home
            return receipt
        return read

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
        position_patcher = patch("autococ.event_rewards.locate_reward_track_position",
                                 return_value={"status": "positioned", "frames": []})
        self.position_mock = position_patcher.start()
        self.addCleanup(position_patcher.stop)
        root=Path(self.temp.name);path=root/"config.toml";path.write_text("")
        config=load_config(path)
        self.config=replace(config,runtime=replace(config.runtime,report_dir=root/"reports"))
        self.frame=SceneSnapshot("village",.99,root/"home.png",{
            "buttons":[{"name":"attack"},{"name":"shop"}],"resources":{"gold":100},"village_type":"home"})
        self.session=Mock(spec=GameSession)
        self.session.config=self.config
        self.session.observe.return_value=self.frame
        self.session.last_snapshot=self.frame
        self.session._input_attempt_generation=0
        self.session.deadline=time.monotonic()+self.config.stop.max_duration_sec
        self.session.task_deadline=self.session.deadline
        self.session.events_path=root/"events.jsonl"
        self.factory=Mock(return_value=self.session)
        self.events=[]

    def runner(self, **options):
        return FlowRunner(options.pop("config",self.config),Mock(),"local",session_factory=self.factory,
                          progress=self.events.append,**options)

    def success(self):
        return TaskResult("battle","succeeded","verified",evidence=[self.frame.screenshot_path],
                          metrics={"returned_home":True,"rounds_completed":1,"victory":False})

    def test_disabled_event_reward_does_not_open_or_claim(self):
        routine = RoutineConfig((TaskSpec("reward", "event_reward", enabled=False),
                                 TaskSpec("collect", "collect")))
        runner = self.runner()
        collected = TaskResult("collect", "succeeded", "verified", evidence=[self.frame.screenshot_path])
        with patch.object(runner, "_handler", return_value=Mock(return_value=collected)), \
             patch("autococ.event_rewards.open_reward_track") as opened, \
             patch("autococ.event_rewards.claim_one_free_reward") as claimed:
            stats = runner.run_routine(routine)
        opened.assert_not_called()
        claimed.assert_not_called()
        self.assertFalse(any(item.task == "reward" for item in stats.task_results))

    def test_disabled_ore_inventory_does_not_navigate(self):
        routine = RoutineConfig((TaskSpec("ores", "ore_inventory", enabled=False),
                                 TaskSpec("collect", "collect")))
        runner = self.runner()
        collected = TaskResult("collect", "succeeded", "verified",
                               evidence=[self.frame.screenshot_path])
        with patch.object(runner, "_handler", return_value=Mock(return_value=collected)), \
             patch("autococ.ore_navigation.read_current_ore_inventory") as read:
            stats = runner.run_routine(routine)
        read.assert_not_called()
        self.assertFalse(any(item.task == "ores" for item in stats.task_results))

    def test_ore_inventory_read_once_records_balances_without_reward_claim(self):
        receipt = {"status": "stable_inventory_read", "returned_home": True,
                   "inventory": {"status": "stable_inventory_read",
                                 "ores": {"shiny": {"amount": 15662, "capacity": 50000}},
                                 "frames": [{"frame": str(self.frame.screenshot_path)}]},
                   "selected": {"frame": str(self.frame.screenshot_path)},
                   "close": {"status": "home", "frame": str(self.frame.screenshot_path)},
                   "home_evidence": [{"frame": "home-a.png"}, {"frame": "home-b.png"}],
                   "account_verified": False, "reward_inventory_verified": False,
                   "formal_success": False}
        with patch("autococ.ore_navigation.read_current_ore_inventory", return_value=receipt) as read, \
             patch("autococ.event_rewards.claim_one_free_reward") as claim:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("ores", "ore_inventory"),)))
        read.assert_called_once()
        claim.assert_not_called()
        result = next(item for item in stats.task_results if item.task == "ores")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["inventory"]["ores"]["shiny"]["amount"], 15662)
        self.assertEqual(len(result.metrics["home_evidence"]), 2)
        self.assertTrue(result.metrics["inventory_read_verified"])
        self.assertFalse(result.metrics["account_verified"])
        self.assertFalse(result.metrics["reward_inventory_verified"])
        self.assertFalse(result.metrics["formal_success"])

    def test_ore_inventory_failure_stops_queue_without_blind_recovery(self):
        routine = RoutineConfig((TaskSpec("ores", "ore_inventory"), TaskSpec("collect", "collect")))
        runner = self.runner()
        with patch("autococ.flow.return_to_village", return_value=self.frame) as navigate, \
             patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=FlowError("unrecognized selected building")) as read, \
             patch.object(runner, "_handler") as handler:
            stats = runner.run_routine(routine)
        read.assert_called_once()
        handler.assert_not_called()
        result = next(item for item in stats.task_results if item.task == "ores")
        self.assertEqual(result.status, "failed")
        self.assertIn("unrecognized selected building", result.reason)
        self.assertFalse(result.metrics["returned_home"])
        self.assertEqual(navigate.call_count, 2)  # Startup and pre-task only.

    def test_ore_inventory_is_not_periodic_maintenance(self):
        clock = [time.monotonic()]
        goals = iter((False, True))
        def read_progress(_session, _task):
            clock[0] += 601
            return GoalProgress(next(goals), evidence=(self.frame.screenshot_path,))
        receipt = {"status": "stable_inventory_read", "returned_home": True,
                   "inventory": {"status": "stable_inventory_read", "frames": []},
                   "home_evidence": [{"frame": "home-a.png"}, {"frame": "home-b.png"}]}
        routine = RoutineConfig((TaskSpec("ores", "ore_inventory"), TaskSpec("farm", "resources")), 600)
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.daily.read_progress", side_effect=read_progress), \
             patch("autococ.ore_navigation.read_current_ore_inventory", return_value=receipt) as read, \
             patch("autococ.combat.run_battle") as battle:
            stats = self.runner().run_routine(routine)
        read.assert_called_once()
        battle.assert_not_called()
        self.assertEqual(stats.goals_completed, 1)

    def test_event_reward_claims_once_and_records_delta_without_account_verification(self):
        routine = RoutineConfig((TaskSpec("reward", "event_reward", free_reward_target="shiny"),))
        receipt = {"status": "ui_acknowledged", "ui_claim_acknowledged": True,
                   "input_attempted": True, "input_sent": True,
                   "reward_inventory_verified": False,
                   "frames": [{"path": str(self.frame.screenshot_path)}]}
        before = self.event_ore_read("before")
        after = self.event_ore_read("after", shiny=16912)
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before, after)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened", "frame": str(self.frame.screenshot_path)}) as opened, \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=receipt) as claimed, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home", "frame": str(self.frame.screenshot_path)}) as closed:
            stats = self.runner().run_routine(routine)
        reward = next(item for item in stats.task_results if item.task == "reward")
        opened.assert_called_once()
        self.position_mock.assert_called_once()
        self.assertEqual(self.position_mock.call_args.args[1], "shiny")
        self.assertTrue(self.position_mock.call_args.kwargs["enabled"])
        self.assertEqual(self.position_mock.call_args.kwargs["task_id"], "reward")
        claimed.assert_called_once()
        closed.assert_called_once()
        self.assertEqual(read.call_count, 2)
        self.assertEqual(claimed.call_args.kwargs["enabled"], True)
        self.assertEqual(claimed.call_args.args[1], "shiny")
        self.assertEqual(reward.status, "limited")
        self.assertTrue(reward.metrics["ui_claim_acknowledged"])
        self.assertEqual(reward.metrics["ore_delta"], {"shiny": 1250, "glowy": 0, "starry": 0})
        self.assertTrue(reward.metrics["reward_inventory_verified"])
        self.assertFalse(reward.metrics["account_verified"])
        self.assertFalse(reward.metrics["formal_success"])
        self.assertEqual(stats.successes, 1)  # Verified launch only.

    def test_event_reward_no_opportunity_skips_without_claim_success(self):
        routine = RoutineConfig((TaskSpec("reward", "event_reward", free_reward_target="glowy"),))
        receipt = {"status": "no_opportunity", "ui_claim_acknowledged": False,
                   "frames": [{"path": str(self.frame.screenshot_path)}]}
        before = self.event_ore_read("before")
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=receipt), \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home", "frame": str(self.frame.screenshot_path)}):
            stats = self.runner().run_routine(routine)
        reward = next(item for item in stats.task_results if item.task == "reward")
        self.assertEqual(reward.status, "skipped")
        read.assert_called_once()
        self.assertFalse(reward.metrics["reward_inventory_verified"])

    def test_event_reward_unknown_closes_known_track_and_stops_queue(self):
        routine = RoutineConfig((TaskSpec("reward", "event_reward", free_reward_target="shiny"),
                                 TaskSpec("collect", "collect")))
        runner = self.runner()
        receipt = {"status": "unknown", "ui_claim_acknowledged": False,
                   "input_attempted": True, "input_sent": False,
                   "frames": [{"path": str(self.frame.screenshot_path)}],
                   "reasons": ["transport result uncertain"]}
        before = self.event_ore_read("before")
        after = self.event_ore_read("after")
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before, after)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=receipt) as claimed, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home", "frame": str(self.frame.screenshot_path)}) as closed, \
             patch.object(runner, "_handler") as handler:
            stats = runner.run_routine(routine)
        claimed.assert_called_once()
        closed.assert_called_once()
        self.assertEqual(read.call_count, 2)
        handler.assert_not_called()
        reward = next(item for item in stats.task_results if item.task == "reward")
        self.assertEqual(reward.status, "failed")
        self.assertIs(reward.metrics["claim"], receipt)
        self.assertTrue(reward.metrics["returned_home"])

    def test_event_reward_close_failure_keeps_receipt_and_never_blind_recovers(self):
        routine = RoutineConfig((TaskSpec("reward", "event_reward", free_reward_target="shiny"),))
        receipt = {"status": "unknown", "ui_claim_acknowledged": False,
                   "frames": [{"path": str(self.frame.screenshot_path)}]}
        before = self.event_ore_read("before")
        with patch("autococ.flow.return_to_village", return_value=self.frame) as navigate, \
             patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=receipt), \
             patch("autococ.event_rewards.close_reward_track", side_effect=FlowError("unrecognized page")):
            stats = self.runner().run_routine(routine)
        reward = next(item for item in stats.task_results if item.task == "reward")
        self.assertEqual(reward.status, "failed")
        self.assertIs(reward.metrics["claim"], receipt)
        self.assertIn("unrecognized page", reward.metrics["close_failure"])
        self.assertFalse(reward.metrics["returned_home"])
        read.assert_called_once()
        self.assertEqual(navigate.call_count, 2)  # Startup and pre-task; no recovery after the failed close.

    def test_event_reward_insufficient_total_budget_never_enters_inventory_or_track(self):
        self.session.deadline = time.monotonic() + 260
        self.session.task_deadline = self.session.deadline
        self.session.last_snapshot = self.event_ore_read("initial")[0]
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        with patch("autococ.ore_navigation.read_current_ore_inventory") as read, \
             patch("autococ.event_rewards.open_reward_track") as opened, \
             patch("autococ.event_rewards.claim_one_free_reward") as claim:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "limited")
        self.assertTrue(result.metrics["returned_home"])
        read.assert_not_called()
        opened.assert_not_called()
        claim.assert_not_called()

    def test_event_reward_glowy_exact_delta_has_inventory_evidence_but_not_formal_success(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="glowy")
        before = self.event_ore_read("before")
        after = self.event_ore_read("after", glowy=1340)
        claim_receipt = {"status": "ui_acknowledged", "ui_claim_acknowledged": True,
                         "input_attempted": True, "input_sent": True, "frames": []}
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before, after)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=claim_receipt) as claim, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home"}):
            result = _event_reward_task(self.session, spec)
        self.assertEqual(read.call_count, 2)
        claim.assert_called_once()
        self.assertEqual(result.metrics["ore_delta"], {"shiny": 0, "glowy": 115, "starry": 0})
        self.assertTrue(result.metrics["reward_inventory_verified"])
        self.assertFalse(result.metrics["account_verified"])
        self.assertFalse(result.metrics["formal_success"])

    def test_event_reward_claim_budget_shortfall_closes_without_claiming(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny",
                        max_duration_sec=300)
        before = self.event_ore_read("before")
        clock = [time.monotonic()]
        def slow_open(*_args, **_kwargs):
            clock[0] += 171
            return {"status": "opened"}
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", side_effect=slow_open), \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home"}) as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "failed")
        read.assert_called_once()
        self.position_mock.assert_not_called()
        claim.assert_not_called()
        close.assert_called_once()

    def test_event_reward_unknown_position_closes_without_claim_or_after_read(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        before = self.event_ore_read("before")
        receipt = {"status": "unknown", "input_attempted": True,
                   "input_sent": False,
                   "frames": [{"path": str(self.frame.screenshot_path)}],
                   "reasons": ["transport uncertain"]}
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.locate_reward_track_position", return_value=receipt) as locate, \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home"}) as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "event_reward_position_unknown")
        self.assertIs(result.metrics["position_navigation"], receipt)
        self.assertIn(self.frame.screenshot_path, result.evidence)
        locate.assert_called_once()
        claim.assert_not_called()
        close.assert_called_once()
        read.assert_called_once()

    def test_event_reward_position_overrun_preserves_claim_and_after_read_budget(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny",
                        max_duration_sec=300)
        before = self.event_ore_read("before")
        clock = [time.monotonic()]
        def slow_position(*_args, **_kwargs):
            clock[0] += 171
            return {"status": "positioned", "frames": []}
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.locate_reward_track_position",
                   side_effect=slow_position) as locate, \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home"}) as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "event_reward_claim_and_after_read_budget_insufficient")
        locate.assert_called_once()
        claim.assert_not_called()
        close.assert_called_once()
        read.assert_called_once()

    def test_event_reward_position_stop_and_account_block_send_no_close(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        before = self.event_ore_read("before")
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)), \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.locate_reward_track_position",
                   side_effect=StopRequested("stop")), \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track") as close:
            with self.assertRaises(StopRequested):
                _event_reward_task(self.session, spec)
        claim.assert_not_called()
        close.assert_not_called()
        occupied = SceneSnapshot("account_in_use", .99, self.frame.screenshot_path,
                                 {"account_in_use": True, "ocr": []})
        def occupied_position(*_args, **_kwargs):
            self.session.last_snapshot = occupied
            return {"status": "unknown", "frames": []}
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)), \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.locate_reward_track_position",
                   side_effect=occupied_position), \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track") as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.reason, "event_reward_account_in_use")
        claim.assert_not_called()
        close.assert_not_called()

    def test_event_reward_near_budget_stop_or_account_block_never_closes(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny",
                        max_duration_sec=300)
        before = self.event_ore_read("before")
        clock = [time.monotonic()]
        def stop_during_open(*_args, **_kwargs):
            clock[0] += 171
            raise StopRequested("stop")
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)), \
             patch("autococ.event_rewards.open_reward_track", side_effect=stop_during_open), \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track") as close:
            with self.assertRaises(StopRequested):
                _event_reward_task(self.session, spec)
        claim.assert_not_called()
        close.assert_not_called()

        occupied = SceneSnapshot("account_in_use", .99, self.frame.screenshot_path,
                                 {"account_in_use": True, "ocr": []})
        clock = [time.monotonic()]
        def occupied_during_open(*_args, **_kwargs):
            clock[0] += 171
            self.session.last_snapshot = occupied
            return {"status": "opened"}
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)), \
             patch("autococ.event_rewards.open_reward_track", side_effect=occupied_during_open), \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track") as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.reason, "event_reward_account_in_use")
        self.assertFalse(result.metrics["returned_home"])
        claim.assert_not_called()
        close.assert_not_called()

    def test_event_reward_uncertain_open_input_cannot_reuse_earlier_home(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        before = self.event_ore_read("before")
        def uncertain_open(*_args, **_kwargs):
            self.session._input_attempt_generation += 1
            raise FlowError("event panel input uncertain")
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", side_effect=uncertain_open), \
             patch("autococ.event_rewards.claim_one_free_reward") as claim, \
             patch("autococ.event_rewards.close_reward_track") as close:
            result = _event_reward_task(self.session, spec)
        read.assert_called_once()
        claim.assert_not_called()
        close.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.metrics["returned_home"])
        self.assertNotIn("input_attempt_generation", self.session.last_snapshot.observations)
        self.assertEqual(self.session._input_attempt_generation, 1)

    def test_event_reward_uncertain_after_ore_input_cannot_reuse_close_home(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        before = self.event_ore_read("before")
        closed_home = self.event_ore_read("closed")[0]
        reads = [0]
        def read(_session, *, timeout_sec):
            reads[0] += 1
            if reads[0] == 1:
                self.session.last_snapshot = before[0]
                return before[1]
            self.session._input_attempt_generation += 1
            raise FlowError("blacksmith input uncertain")
        def close_track(*_args, **_kwargs):
            self.session.last_snapshot = closed_home
            return {"status": "home", "frame": str(closed_home.screenshot_path)}
        claim_receipt = {"status": "ui_acknowledged", "ui_claim_acknowledged": True,
                         "input_attempted": True, "input_sent": True, "frames": []}
        with patch("autococ.ore_navigation.read_current_ore_inventory", side_effect=read), \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=claim_receipt) as claim, \
             patch("autococ.event_rewards.close_reward_track", side_effect=close_track) as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(reads[0], 2)
        claim.assert_called_once()
        close.assert_called_once()
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.metrics["returned_home"])
        self.assertNotIn("input_attempt_generation", closed_home.observations)
        self.assertEqual(self.session._input_attempt_generation, 1)

    def test_event_reward_none_evidence_lists_keep_partial_receipts(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        before_home, before_receipt = self.event_ore_read("before")
        before_receipt["inventory"]["frames"] = None
        before_receipt["home_evidence"] = None
        def invalid_read(_session, *, timeout_sec):
            self.session.last_snapshot = before_home
            return before_receipt
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=invalid_read), \
             patch("autococ.event_rewards.open_reward_track") as opened:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "failed")
        self.assertIs(result.metrics["before_ore_read"], before_receipt)
        self.assertIn(Path(before_receipt["selected"]["frame"]), result.evidence)
        opened.assert_not_called()

        normal = self.event_ore_read("normal")
        unknown = {"status": "unknown", "input_attempted": False,
                   "ui_claim_acknowledged": False, "frames": None}
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(normal)), \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=unknown), \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home"}):
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "failed")
        self.assertIs(result.metrics["claim"], unknown)

    def test_event_reward_before_read_failure_or_capacity_never_claims(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=FlowError("before page unreadable")) as read, \
             patch("autococ.event_rewards.open_reward_track") as opened, \
             patch("autococ.event_rewards.claim_one_free_reward") as claim:
            failed = _event_reward_task(self.session, spec)
        self.assertEqual(failed.status, "failed")
        read.assert_called_once()
        opened.assert_not_called()
        claim.assert_not_called()
        full = self.event_ore_read("before", shiny=49000)
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(full)) as read, \
             patch("autococ.event_rewards.open_reward_track") as opened, \
             patch("autococ.event_rewards.claim_one_free_reward") as claim:
            skipped = _event_reward_task(self.session, spec)
        self.assertEqual(skipped.status, "skipped")
        self.assertEqual(skipped.reason, "event_reward_target_capacity_insufficient")
        self.assertTrue(skipped.metrics["returned_home"])
        read.assert_called_once()
        opened.assert_not_called()
        claim.assert_not_called()

    def test_event_reward_after_read_failure_keeps_claim_and_close_evidence(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="glowy")
        before_home, before_receipt = self.event_ore_read("before")
        calls = [0]
        def read(_session, *, timeout_sec):
            calls[0] += 1
            if calls[0] == 1:
                self.session.last_snapshot = before_home
                return before_receipt
            self.session.last_snapshot = SceneSnapshot("popup", .99,
                                                       self.frame.screenshot_path,
                                                       {"ore_inventory": None})
            raise FlowError("after page unreadable")
        claim_receipt = {"status": "ui_acknowledged", "ui_claim_acknowledged": True,
                         "input_attempted": True, "input_sent": True,
                         "frames": [{"path": "claim.png"}]}
        with patch("autococ.ore_navigation.read_current_ore_inventory", side_effect=read), \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=claim_receipt) as claim, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home", "frame": "close.png"}) as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "failed")
        self.assertEqual(calls[0], 2)
        claim.assert_called_once()
        close.assert_called_once()
        self.assertIs(result.metrics["claim"], claim_receipt)
        self.assertIs(result.metrics["before_ore_read"], before_receipt)
        self.assertIsNone(result.metrics["after_ore_read"])
        self.assertFalse(result.metrics["returned_home"])
        self.assertFalse(result.metrics["reward_inventory_verified"])

    def test_event_reward_wrong_delta_or_resource_change_is_unverified(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="glowy")
        claim_receipt = {"status": "ui_acknowledged", "ui_claim_acknowledged": True,
                         "input_attempted": True, "input_sent": True, "frames": []}
        for after in (self.event_ore_read("after-wrong", glowy=1339),
                      self.event_ore_read("after-gem", glowy=1340, gems=3),
                      self.event_ore_read("before", glowy=1340)):
            with self.subTest(after=after[0].screenshot_path):
                before = self.event_ore_read("before")
                with patch("autococ.ore_navigation.read_current_ore_inventory",
                           side_effect=self.event_ore_reader(before, after)), \
                     patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
                     patch("autococ.event_rewards.claim_one_free_reward", return_value=claim_receipt), \
                     patch("autococ.event_rewards.close_reward_track", return_value={"status": "home"}):
                    result = _event_reward_task(self.session, spec)
                self.assertEqual(result.status, "failed")
                self.assertFalse(result.metrics["reward_inventory_verified"])
                self.assertFalse(result.metrics["formal_success"])

    def test_event_reward_duplicate_unknown_and_account_block_never_retry(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        before = self.event_ore_read("before")
        duplicate = {"status": "unknown", "input_attempted": False,
                     "ui_claim_acknowledged": False,
                     "reasons": ["already_attempted_in_session"], "frames": []}
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", return_value=duplicate) as claim, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home"}) as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.status, "failed")
        read.assert_called_once()
        claim.assert_called_once()
        close.assert_called_once()

        occupied = SceneSnapshot("account_in_use", .99, self.frame.screenshot_path,
                                 {"account_in_use": True, "ocr": []})
        def become_occupied(*_args, **_kwargs):
            self.session.last_snapshot = occupied
            return {"status": "unknown", "input_attempted": True,
                    "input_sent": False, "frames": []}
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)) as read, \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward", side_effect=become_occupied) as claim, \
             patch("autococ.event_rewards.close_reward_track") as close:
            result = _event_reward_task(self.session, spec)
        self.assertEqual(result.reason, "event_reward_account_in_use")
        read.assert_called_once()
        claim.assert_called_once()
        close.assert_not_called()

    def test_event_reward_stop_does_not_issue_cleanup_input(self):
        spec = TaskSpec("reward", "event_reward", free_reward_target="shiny")
        before = self.event_ore_read("before")
        with patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)), \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}), \
             patch("autococ.event_rewards.claim_one_free_reward",
                   side_effect=StopRequested("stop")) as claim, \
             patch("autococ.event_rewards.close_reward_track") as close:
            with self.assertRaises(StopRequested):
                _event_reward_task(self.session, spec)
        claim.assert_called_once()
        close.assert_not_called()

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

    def test_periodic_maintenance_never_reopens_event_reward(self):
        clock = [time.monotonic()]
        goals = iter((False, True))
        def read(_session, _task):
            clock[0] += 601
            return GoalProgress(next(goals), evidence=(self.frame.screenshot_path,))
        reward = TaskSpec("reward", "event_reward", free_reward_target="glowy")
        routine = RoutineConfig((reward, TaskSpec("farm", "resources")), 600)
        before = self.event_ore_read("before")
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.daily.read_progress", side_effect=read), \
             patch("autococ.ore_navigation.read_current_ore_inventory",
                   side_effect=self.event_ore_reader(before)), \
             patch("autococ.event_rewards.open_reward_track", return_value={"status": "opened"}) as opened, \
             patch("autococ.event_rewards.claim_one_free_reward", return_value={"status": "no_opportunity", "frames": []}) as claimed, \
             patch("autococ.event_rewards.close_reward_track", return_value={"status": "home", "frame": str(self.frame.screenshot_path)}), \
             patch("autococ.combat.run_battle") as battle:
            stats = self.runner().run_routine(routine)
        opened.assert_called_once()
        claimed.assert_called_once()
        battle.assert_not_called()
        self.assertEqual(stats.goals_completed, 1)

    def test_front_tasks_cannot_start_battle_with_less_than_full_global_window(self):
        clock = [time.monotonic()]
        self.config = replace(self.config, runtime=replace(self.config.runtime, task_timeout_sec=600))
        self.session.deadline = clock[0] + 1050
        def collected(_session):
            clock[0] += 500
            return TaskResult("collect", "succeeded", "verified", evidence=[self.frame.screenshot_path])
        runner = self.runner()
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(runner, "_handler", return_value=collected), \
             patch("autococ.daily.read_progress") as goal, \
             patch("autococ.combat.run_battle") as battle:
            stats = runner.run_routine(RoutineConfig((TaskSpec("collect", "collect"),
                                                     TaskSpec("farm", "resources"))))
        goal.assert_not_called()
        battle.assert_not_called()
        limit = next(r for r in stats.task_results if r.reason == "global_battle_budget_insufficient")
        self.assertEqual(limit.metrics["phase"], "before_progress")
        self.assertEqual(limit.metrics["completed_battles"], 0)
        self.assertTrue(limit.metrics["returned_home"])
        self.assertEqual(stats.battles_completed, 0)

    def test_initial_maintenance_needs_full_task_budget_before_any_handler(self):
        clock = [time.monotonic()]
        self.config = replace(self.config, runtime=replace(self.config.runtime, task_timeout_sec=600))
        self.session.deadline = clock[0] + 550
        runner = self.runner()
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(runner, "_handler") as handler, \
             patch("autococ.daily.read_progress") as goal, \
             patch("autococ.combat.run_battle") as battle:
            stats = runner.run_routine(RoutineConfig((TaskSpec("collect", "collect"),
                                                     TaskSpec("farm", "resources"))))
        handler.assert_not_called()
        goal.assert_not_called()
        battle.assert_not_called()
        limit = next(r for r in stats.task_results if r.reason == "global_maintenance_budget_insufficient")
        self.assertEqual(limit.task, "collect")
        self.assertEqual(limit.metrics["phase"], "before_initial_maintenance")
        self.assertEqual(limit.metrics["global_remaining_sec"], 550)
        self.assertEqual(limit.metrics["required_task_sec"], 600)
        self.assertTrue(limit.metrics["returned_home"])
        self.assertEqual(stats.battles_completed, 0)
        self.assertIn("limited by global maintenance budget", stats.stop_reason)

    def test_stop_after_launch_preempts_initial_maintenance_budget(self):
        event = Event()
        self.session.deadline = time.monotonic() + 1
        launch = TaskResult("launch", "succeeded", "home verified",
                            evidence=[self.frame.screenshot_path])
        def verify(_session):
            event.set()
            return launch
        runner = self.runner(stop_event=event)
        with patch("autococ.flow.verify_home", side_effect=verify), \
             patch.object(runner, "_handler") as handler:
            stats = runner.run_routine(RoutineConfig((TaskSpec("collect", "collect"),)))
        handler.assert_not_called()
        self.assertEqual(stats.task_results[-1].status, "cancelled")
        self.assertFalse(any(r.reason == "global_maintenance_budget_insufficient"
                             for r in stats.task_results))

    def test_read_progress_can_exhaust_battle_budget_before_search(self):
        clock = [time.monotonic()]
        self.config = replace(self.config, runtime=replace(self.config.runtime, task_timeout_sec=600))
        self.session.deadline = clock[0] + 700
        task_beginnings = []
        def begin():
            self.session.task_deadline = min(self.session.deadline, clock[0] + 600)
            task_beginnings.append(clock[0])
        self.session.begin_task.side_effect = begin
        def read(_session, _task):
            clock[0] += 150
            return GoalProgress(False, evidence=(self.frame.screenshot_path,))
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.daily.read_progress", side_effect=read), \
             patch("autococ.combat.run_battle") as battle:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("farm", "resources"),)))
        battle.assert_not_called()
        limit = next(r for r in stats.task_results if r.reason == "global_battle_budget_insufficient")
        self.assertEqual(limit.metrics["phase"], "before_battle")
        self.assertEqual(limit.metrics["global_remaining_sec"], 550)
        self.assertTrue(limit.metrics["returned_home"])
        self.assertEqual(task_beginnings[-1], clock[0])
        self.assertEqual(self.session.task_deadline, self.session.deadline)

    def test_admitted_battle_gets_fresh_full_task_window_and_finishes_home(self):
        clock = [time.monotonic()]
        self.config = replace(self.config, runtime=replace(self.config.runtime, task_timeout_sec=600))
        self.session.deadline = clock[0] + 650
        task_beginnings = []
        def begin():
            self.session.task_deadline = min(self.session.deadline, clock[0] + 600)
            task_beginnings.append(clock[0])
        self.session.begin_task.side_effect = begin
        def read(_session, _task):
            clock[0] += 30
            return GoalProgress(False, evidence=(self.frame.screenshot_path,))
        def battle(_session):
            self.assertEqual(self.session.task_deadline, task_beginnings[-1] + 600)
            clock[0] += 500
            return self.success()
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.daily.read_progress", side_effect=read), \
             patch("autococ.combat.run_battle", side_effect=battle) as called:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("farm", "resources"),)))
        called.assert_called_once()
        self.assertEqual(stats.battles_completed, 1)
        self.assertEqual(task_beginnings[-1] - task_beginnings[-2], 30)
        self.assertEqual([r.reason for r in stats.task_results if r.status == "limited"],
                         ["global_battle_budget_insufficient"])

    def test_resource_window_ends_after_completed_battle_without_next_search(self):
        clock = [time.monotonic()]
        self.config = replace(self.config, runtime=replace(self.config.runtime, task_timeout_sec=600))
        self.session.deadline = clock[0] + 5000
        def battle(_session):
            clock[0] += 600
            return self.success()
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch("autococ.daily.read_progress", return_value=GoalProgress(False)), \
             patch("autococ.combat.run_battle", side_effect=battle) as called:
            stats = self.runner().run_routine(RoutineConfig((TaskSpec("farm", "resources",
                                                           max_duration_sec=600),)))
        called.assert_called_once()
        self.assertEqual(stats.battles_completed, 1)
        self.assertEqual([r.reason for r in stats.task_results if r.status == "limited"],
                         ["task_limit_reached"])

    def test_periodic_chain_stops_at_budget_boundary_before_another_maintenance(self):
        clock = [time.monotonic()]
        self.config = replace(self.config, runtime=replace(self.config.runtime, task_timeout_sec=600))
        self.session.deadline = clock[0] + 1900
        calls = []
        def handler(kind):
            def execute(_session):
                calls.append(kind)
                if len(calls) > 3:
                    clock[0] += 590
                return TaskResult(kind, "succeeded", "verified", evidence=[self.frame.screenshot_path])
            return execute
        def read(_session, _task):
            clock[0] += 601
            return GoalProgress(False, evidence=(self.frame.screenshot_path,))
        routine = RoutineConfig((TaskSpec("collect", "collect"), TaskSpec("request", "request"),
                                 TaskSpec("train", "train"), TaskSpec("farm", "resources")), 600)
        runner = self.runner()
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(runner, "_handler", side_effect=handler), \
             patch("autococ.daily.read_progress", side_effect=read), \
             patch("autococ.combat.run_battle") as battle:
            stats = runner.run_routine(routine)
        self.assertEqual(calls, ["collect", "request", "train", "collect", "request"])
        battle.assert_not_called()
        limit = next(r for r in stats.task_results if r.reason == "global_battle_budget_insufficient")
        self.assertEqual(limit.metrics["phase"], "before_maintenance")
        self.assertTrue(limit.metrics["returned_home"])

    def test_periodic_chain_stops_when_resource_window_expires_mid_chain(self):
        clock = [time.monotonic()]
        self.session.deadline = clock[0] + 5000
        calls = []
        def handler(kind):
            def execute(_session):
                calls.append(kind)
                if len(calls) > 3:
                    clock[0] += 49
                return TaskResult(kind, "succeeded", "verified", evidence=[self.frame.screenshot_path])
            return execute
        def read(_session, _task):
            clock[0] += 601
            return GoalProgress(False, evidence=(self.frame.screenshot_path,))
        routine = RoutineConfig((TaskSpec("collect", "collect"), TaskSpec("request", "request"),
                                 TaskSpec("train", "train"),
                                 TaskSpec("farm", "resources", max_duration_sec=650)), 600)
        runner = self.runner()
        with patch("autococ.daily.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(runner, "_handler", side_effect=handler), \
             patch("autococ.daily.read_progress", side_effect=read), \
             patch("autococ.combat.run_battle") as battle:
            stats = runner.run_routine(routine)
        self.assertEqual(calls, ["collect", "request", "train", "collect"])
        battle.assert_not_called()
        self.assertEqual([r.reason for r in stats.task_results if r.status == "limited"],
                         ["task_limit_reached"])

    def test_stop_request_during_progress_preempts_budget_result(self):
        event = Event()
        def read(_session, _task):
            event.set()
            return GoalProgress(False, evidence=(self.frame.screenshot_path,))
        with patch("autococ.daily.read_progress", side_effect=read), \
             patch("autococ.combat.run_battle") as battle:
            stats = self.runner(stop_event=event).run_routine(RoutineConfig((TaskSpec("farm", "resources"),)))
        battle.assert_not_called()
        self.assertEqual(stats.task_results[-1].status, "cancelled")
        self.assertFalse(any(r.reason == "global_battle_budget_insufficient" for r in stats.task_results))

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
