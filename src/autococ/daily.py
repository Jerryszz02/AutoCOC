"""Sequential daily tasks, independent battle goals, and verified live counters."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import time
from uuid import uuid4

from .config import ProfileConfig
from .errors import CapabilityUnavailable, ConfigError, FlowError, StopRequested
from .objectives import load_adapter, read_progress
from .reporting import RunStats, TaskResult, capture_run_provenance, resource_metrics
from .routine_config import BATTLE_KINDS, RoutineConfig


class _RecordedTaskFailure(FlowError):
    """Abort the queue without recording the same task failure twice."""


def _ore_inventory_task(session, spec) -> TaskResult:
    """Read current ore balances once; this does not verify a reward delta."""
    from .ore_navigation import read_current_ore_inventory

    receipt = None
    failure = None
    try:
        receipt = read_current_ore_inventory(session, timeout_sec=min(180, spec.max_duration_sec))
    except StopRequested:
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    frames = []
    if isinstance(receipt, dict):
        selected = receipt.get("selected")
        inventory = receipt.get("inventory")
        closed = receipt.get("close")
        if isinstance(selected, dict) and selected.get("frame"):
            frames.append(selected["frame"])
        if isinstance(inventory, dict):
            frames.extend(item["frame"] for item in (inventory.get("frames") or [])
                          if isinstance(item, dict) and item.get("frame"))
        if isinstance(closed, dict) and closed.get("frame"):
            frames.append(closed["frame"])
        frames.extend(item["frame"] for item in (receipt.get("home_evidence") or [])
                      if isinstance(item, dict) and item.get("frame"))
    current = getattr(session, "last_snapshot", None)
    if not frames and current is not None:
        frames.append(str(current.screenshot_path))
    evidence = [Path(path) for path in dict.fromkeys(frames)]
    inventory = receipt.get("inventory") if isinstance(receipt, dict) else None
    homes = receipt.get("home_evidence") if isinstance(receipt, dict) else None
    verified = (isinstance(receipt, dict) and receipt.get("status") == "stable_inventory_read"
                and receipt.get("returned_home") is True
                and isinstance(inventory, dict) and inventory.get("status") == "stable_inventory_read"
                and isinstance(homes, list) and len(homes) == 2)
    metrics = {"receipt_kind": "ore_inventory", "receipt": receipt,
               "inventory": inventory, "home_evidence": homes,
               "inventory_read_verified": bool(verified),
               "returned_home": bool(verified), "account_verified": False,
               "reward_inventory_verified": False, "formal_success": False,
               "failure": failure}
    if not verified:
        return TaskResult(spec.id, "failed", failure or "ore_inventory_read_unverified",
                          evidence=evidence, metrics=metrics)
    return TaskResult(spec.id, "succeeded", "ore_inventory_stable_read_returned_home",
                      evidence=evidence, metrics=metrics)


def _event_reward_task(session, spec) -> TaskResult:
    """Reconcile one free claim against ore balances in the same task session."""
    from .event_rewards import (close_reward_track, claim_one_free_reward,
                                locate_reward_track_position, open_reward_track)
    from .ore_navigation import read_current_ore_inventory

    end = min(session.deadline, session.task_deadline,
              time.monotonic() + spec.max_duration_sec)
    amounts = {"shiny": 1250, "glowy": 115}
    expected_amount = amounts[spec.free_reward_target]
    before = None
    after = None
    before_home = None
    after_home = None
    opened = None
    located = None
    claim = None
    closed = None
    failure = None
    close_failure = None
    delta = None
    verified = False
    known_home = False

    def remaining() -> float:
        session.check_deadline()
        return end - time.monotonic()

    def account_blocked() -> bool:
        current = getattr(session, "last_snapshot", None)
        if current is None:
            return False
        observations = current.observations
        ocr = observations.get("ocr")
        if current.scene == "account_in_use" or observations.get("account_in_use") is not None:
            return True
        return isinstance(ocr, list) and any(
            "另一个设备正在连接这个村庄" in str(item.get("text", ""))
            for item in ocr if isinstance(item, dict))

    def stable_ores(receipt: dict) -> dict[str, dict[str, int]]:
        inventory = receipt.get("inventory") if isinstance(receipt, dict) else None
        homes = receipt.get("home_evidence") if isinstance(receipt, dict) else None
        if (not isinstance(receipt, dict)
                or receipt.get("status") != "stable_inventory_read"
                or receipt.get("returned_home") is not True
                or not isinstance(inventory, dict)
                or inventory.get("status") != "stable_inventory_read"
                or not isinstance(inventory.get("frames"), list)
                or len(inventory["frames"]) != 2
                or not isinstance(homes, list) or len(homes) != 2):
            raise FlowError("Event reward ore inventory read is incomplete")
        ores = inventory.get("ores")
        if not isinstance(ores, dict) or set(ores) != {"shiny", "glowy", "starry"}:
            raise FlowError("Event reward ore balances are incomplete")
        capacities = {"shiny": 50000, "glowy": 5000, "starry": 1000}
        for name, capacity in capacities.items():
            value = ores[name]
            if (not isinstance(value, dict) or type(value.get("amount")) is not int
                    or type(value.get("capacity")) is not int
                    or value["capacity"] != capacity
                    or not 0 <= value["amount"] <= capacity):
                raise FlowError("Event reward ore balance or capacity is invalid")
        return ores

    def home_resources(receipt: dict) -> dict:
        current = getattr(session, "last_snapshot", None)
        last = receipt["home_evidence"][-1]
        if (current is None or current.scene != "village"
                or current.observations.get("village_type") != "home"
                or current.observations.get("resource_source") != "village_inventory"
                or account_blocked() or not isinstance(last, dict)
                or last.get("frame") != str(current.screenshot_path)
                or last.get("frame_id") != current.observations.get("frame_id")):
            raise FlowError("Event reward ore read did not finish at its bound HOME")
        resources = current.observations.get("resources")
        names = ("gold", "elixir", "dark_elixir", "gems")
        if (not isinstance(resources, dict)
                or any(type(resources.get(name)) is not int or resources[name] < 0
                       for name in names)):
            raise FlowError("Event reward HOME resources are incomplete")
        observed_at = last.get("observed_at_monotonic")
        if (type(observed_at) not in (int, float) or not math.isfinite(observed_at)
                or current.observations.get("observed_at_monotonic") != observed_at):
            raise FlowError("Event reward HOME capture time is missing")
        return {"frame": last["frame"], "frame_id": last["frame_id"],
                "frame_sha256": last.get("frame_sha256"),
                "observed_at_monotonic": observed_at,
                "resources": {name: resources[name] for name in names}}

    def finish(status: str, reason: str) -> TaskResult:
        frames = []
        for receipt in (before, after):
            if not isinstance(receipt, dict):
                continue
            selected = receipt.get("selected")
            inventory = receipt.get("inventory")
            close = receipt.get("close")
            if isinstance(selected, dict) and selected.get("frame"):
                frames.append(selected["frame"])
            if isinstance(inventory, dict):
                frames.extend(item["frame"] for item in (inventory.get("frames") or [])
                              if isinstance(item, dict) and item.get("frame"))
            if isinstance(close, dict) and close.get("frame"):
                frames.append(close["frame"])
            frames.extend(item["frame"] for item in (receipt.get("home_evidence") or [])
                          if isinstance(item, dict) and item.get("frame"))
        if isinstance(opened, dict):
            frames.extend(opened[key] for key in ("entry_frame", "frame") if opened.get(key))
        if isinstance(located, dict):
            frames.extend(item["path"] for item in (located.get("frames") or [])
                          if isinstance(item, dict) and item.get("path"))
        if isinstance(claim, dict):
            frames.extend(item["path"] for item in (claim.get("frames") or [])
                          if isinstance(item, dict) and item.get("path"))
        if isinstance(closed, dict) and closed.get("frame"):
            frames.append(closed["frame"])
        current = getattr(session, "last_snapshot", None)
        if current is not None and not frames:
            frames.append(str(current.screenshot_path))
        captured = (current.observations.get("observed_at_monotonic")
                    if current is not None else None)
        current_home = (current is not None and current.scene == "village"
                        and current.observations.get("village_type") == "home"
                        and current.observations.get("resource_source") == "village_inventory"
                        and type(captured) in (int, float) and math.isfinite(captured)
                        and 0 <= time.monotonic() - captured <= 8
                        and not account_blocked())
        returned_home = bool(known_home and current_home)
        metrics = {"receipt_kind": "event_reward", "target": spec.free_reward_target,
                   "before_ore_read": before, "after_ore_read": after,
                   "before_home": before_home, "after_home": after_home,
                   "opened": opened, "position_navigation": located,
                   "claim": claim, "closed": closed,
                   "ore_delta": delta, "ui_claim_acknowledged": bool(
                       isinstance(claim, dict) and claim.get("ui_claim_acknowledged") is True),
                   "reward_inventory_verified": verified,
                   "account_verified": False, "formal_success": False,
                   "returned_home": returned_home,
                   "failure": failure, "close_failure": close_failure}
        return TaskResult(spec.id, status, reason,
                          evidence=[Path(path) for path in dict.fromkeys(frames)], metrics=metrics)

    # Two complete 90s inventory reads, three bounded 20s UI stages, and
    # one bounded 30s positioning step share the same task/global deadline.
    # The task/global deadlines are shared; no phase can extend either one.
    current = getattr(session, "last_snapshot", None)
    captured = current.observations.get("observed_at_monotonic") if current is not None else None
    known_home = (current is not None and current.scene == "village"
                  and current.observations.get("village_type") == "home"
                  and current.observations.get("resource_source") == "village_inventory"
                  and type(captured) in (int, float) and math.isfinite(captured)
                  and 0 <= time.monotonic() - captured <= 8 and not account_blocked())
    if remaining() < 270:
        return finish("limited", "event_reward_full_reconciliation_budget_insufficient")

    known_home = False
    try:
        before = read_current_ore_inventory(session, timeout_sec=90)
        before_ores = stable_ores(before)
        before_home = home_resources(before)
        known_home = True
    except StopRequested:
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        return finish("failed", failure)

    current_amount = before_ores[spec.free_reward_target]["amount"]
    capacity = before_ores[spec.free_reward_target]["capacity"]
    if capacity - current_amount < expected_amount:
        return finish("skipped", "event_reward_target_capacity_insufficient")
    if remaining() < 180:
        return finish("limited", "event_reward_post_read_budget_insufficient")

    known_home = False
    try:
        opened = open_reward_track(session, timeout_sec=20)
        if remaining() < 160:
            failure = "event_reward_position_and_after_read_budget_insufficient"
        else:
            located = locate_reward_track_position(session, spec.free_reward_target,
                                                   enabled=True, task_id=spec.id,
                                                   timeout_sec=30)
        if failure is None and (not isinstance(located, dict)
                                or located.get("status") != "positioned"):
            failure = "event_reward_position_unknown"
        if failure is None and remaining() < 130:
            failure = "event_reward_claim_and_after_read_budget_insufficient"
        if failure is None:
            claim = claim_one_free_reward(session, spec.free_reward_target, enabled=True,
                                          timeout_sec=20)
    except StopRequested:
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    # Only a recognized current track is safe to close. An occupied account is
    # terminal and must not turn an uncertain claim into another input.
    current = getattr(session, "last_snapshot", None)
    current_track = (current is not None and current.scene == "popup"
                     and isinstance(current.observations.get("event_reward_track"), dict))
    if account_blocked():
        failure = "event_reward_account_in_use"
        return finish("failed", failure)
    if opened is not None or current_track:
        try:
            available = remaining()
            if available <= 0:
                raise FlowError("Event reward has no time to close its track")
            closed = close_reward_track(session, timeout_sec=min(20, available))
            known_home = closed.get("status") == "home"
        except StopRequested:
            raise
        except Exception as exc:
            close_failure = f"{type(exc).__name__}: {exc}"

    if close_failure or (opened is not None and not (closed and closed.get("status") == "home")):
        return finish("failed", "event_reward_close_unverified")
    if failure:
        return finish("failed", failure)
    if not isinstance(claim, dict):
        return finish("failed", "event_reward_claim_unknown")
    if claim.get("status") == "no_opportunity":
        return finish("skipped", "event_reward_no_opportunity")
    if not claim.get("input_attempted"):
        return finish("failed", "event_reward_claim_unknown")

    known_home = False
    try:
        if remaining() <= 0:
            raise FlowError("Event reward has no time for the after inventory read")
        after = read_current_ore_inventory(session, timeout_sec=min(90, remaining()))
        after_ores = stable_ores(after)
        after_home = home_resources(after)
        known_home = True
        delta = {name: after_ores[name]["amount"] - before_ores[name]["amount"]
                 for name in ("shiny", "glowy", "starry")}
        expected = {name: expected_amount if name == spec.free_reward_target else 0
                    for name in delta}
        verified = (claim.get("status") == "ui_acknowledged"
                    and claim.get("ui_claim_acknowledged") is True
                    and claim.get("input_sent") is True
                    and delta == expected
                    and before_home["observed_at_monotonic"] < after_home["observed_at_monotonic"]
                    and before_home["frame"] != after_home["frame"]
                    and before_home["frame_id"] != after_home["frame_id"]
                    and not ({item["frame"] for item in before["inventory"]["frames"]}
                             & {item["frame"] for item in after["inventory"]["frames"]})
                    and before_home["resources"] == after_home["resources"])
    except StopRequested:
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        return finish("failed", failure)
    if verified:
        return finish("limited", "event_reward_delta_verified_account_unverified")
    return finish("failed", "event_reward_claim_or_delta_unverified")


def validate_routine_files(config, routine):
    """Validate syntax and referenced files before connecting or changing the army."""
    routine.validate()
    if not any(task.enabled for task in routine.tasks):
        raise ConfigError("Select at least one daily task")
    base = config.source_path.resolve().parent
    for task in routine.tasks:
        if not task.enabled:
            continue
        if task.strategy_file:
            from .strategy_config import load_strategy
            path = Path(task.strategy_file)
            load_strategy(path if path.is_absolute() else base / path)
        if task.goal.adapter_path:
            path = Path(task.goal.adapter_path)
            adapter = load_adapter(path if path.is_absolute() else base / path)
            if adapter.kind != task.kind:
                raise ConfigError(f"Adapter kind does not match task {task.id}")


def run_routine(runner, routine: RoutineConfig) -> RunStats:
    from .flow import return_to_village, verify_home
    from .combat import run_battle
    validate_routine_files(runner.config, routine)
    enabled = tuple(t for t in routine.tasks if t.enabled)
    if not enabled:
        raise ConfigError("Select at least one daily task")
    config = replace(runner.config, routine=routine,
                     profiles={"daily": ProfileConfig(("launch",) + tuple(t.kind for t in enabled))})
    stats = RunStats(profile="daily", device_serial=runner.serial,
                     mode="dry-run" if config.runtime.dry_run else "live",
                     provenance=capture_run_provenance(config, "daily"))
    runner._notify("run_started", run_id=stats.run_id, mode=stats.mode, tasks=tuple(t.id for t in enabled))
    session = None
    active = None
    seen_battles = set()
    task_started_at = time.monotonic()
    last_maintenance = time.monotonic()
    event_log_failed = False

    def record(result, *, notify=True):
        nonlocal event_log_failed
        stats.record_task(result)
        if session is not None:
            runner._record_resolution(session, stats)
            if not event_log_failed and not runner._record_event(session, stats, result):
                event_log_failed = True
                raise FlowError("Unable to record verified result")
        if notify:
            runner._notify("task_result", task=result.task, status=result.status, reason=result.reason,
                           evidence=result.evidence, metrics=result.metrics)

    def maintenance(spec, *, periodic=False):
        session.begin_task()
        return_to_village(session)
        if not periodic:
            runner._notify("task_started", task=spec.id, cycle=1)
        result = (_event_reward_task(session, spec) if spec.kind == "event_reward"
                  else _ore_inventory_task(session, spec) if spec.kind == "ore_inventory"
                  else runner._handler(spec.kind)(session))
        # Unsupported donation remains an explicit capability result, not an empty chat.
        if result.status == "failed" and result.reason == "donation_interaction_not_supported":
            result = replace(result, status="not_supported")
        result = replace(result, task=spec.id,
                         metrics={**result.metrics, "receipt_kind": spec.kind, "periodic": periodic})
        record(result, notify=not periodic)
        if result.status == "failed":
            raise _RecordedTaskFailure(f"{spec.id}: {result.reason}")
        if spec.kind not in {"event_reward", "ore_inventory"}:
            return return_to_village(session)

    def confirmed_home(known_home=None):
        # A just-verified Home frame remains valid when no input followed it.
        if known_home is not None and session.last_snapshot is known_home:
            return known_home
        # A prior progress or maintenance task may have used its own deadline.
        # Refresh that task allowance only for safe navigation, never global.
        session.begin_task()
        return return_to_village(session)

    def limit_for_battle_budget(spec, completed, *, phase, progress=None, known_home=None):
        """Leave Home without starting a battle that cannot use its full task window."""
        runner._check_stop()
        remaining = session.deadline - time.monotonic()
        required = config.runtime.task_timeout_sec
        if remaining >= required:
            return False
        home = confirmed_home(known_home)
        evidence = list(progress.evidence) if progress is not None else []
        evidence.append(home.screenshot_path)
        record(TaskResult(spec.id, "limited", "global_battle_budget_insufficient",
                          evidence=list(dict.fromkeys(evidence)),
                          metrics={"receipt_kind": "battle_budget", "phase": phase,
                                   "completed_battles": completed,
                                   "global_remaining_sec": max(0.0, remaining),
                                   "required_battle_task_sec": required,
                                   "returned_home": True, "formal_success": False}))
        return True

    try:
        runner._check_stop()
        if config.runtime.dry_run:
            for task in enabled:
                runner._check_stop()
                runner._notify("task_started", task=task.id, cycle=1)
                record(TaskResult(task.id, "simulated", "Planning only: no game action or outcome was verified"))
            stats.stop_reason = "dry-run planning complete; real success count remains zero"
            return stats
        options = {"stop_event": runner.stop_event} if runner.stop_event is not None else {}
        session = runner.session_factory(config, runner.adb, runner.serial, config.runtime.report_dir / stats.run_id,
                                         launch=True, logger=runner.logger, **options)
        version = getattr(session, "client_version", None)
        if isinstance(version, str) and version:
            stats.provenance["client_version"] = version
        stats.events_path = session.events_path
        session.begin_task()
        launched = verify_home(session)
        record(launched, notify=False)
        if launched.status != "succeeded":
            raise _RecordedTaskFailure(f"launch: {launched.reason}")
        if session.last_snapshot.observations.get("village_type") != "home":
            raise CapabilityUnavailable("Cannot positively identify the main Home Village")
        last_verified_home = session.last_snapshot
        for task in enabled:
            runner._check_stop()
            active = task
            task_started_at = time.monotonic()
            if task.kind not in BATTLE_KINDS:
                runner._check_stop()
                remaining = session.deadline - time.monotonic()
                required = config.runtime.task_timeout_sec
                if remaining < required:
                    home = confirmed_home(last_verified_home)
                    record(TaskResult(task.id, "limited", "global_maintenance_budget_insufficient",
                                      evidence=[home.screenshot_path],
                                      metrics={"receipt_kind": "maintenance_budget",
                                               "phase": "before_initial_maintenance",
                                               "global_remaining_sec": max(0.0, remaining),
                                               "required_task_sec": required,
                                               "returned_home": True, "formal_success": False}))
                    stats.stop_reason = "daily task queue limited by global maintenance budget"
                    active = None
                    break
                last_verified_home = maintenance(task)
                if last_verified_home is None:
                    current = session.last_snapshot
                    if (current is not None and current.scene == "village"
                            and current.observations.get("village_type") == "home"):
                        last_verified_home = current
                active = None
                continue
            runner._notify("task_started", task=task.id, cycle=1)
            session.config = replace(config, battle=replace(config.battle, objective=task.kind,
                strategy=task.strategy, strategy_file=task.strategy_file, target_building=task.goal.building_type,
                resource_filter=task.resource_filter, max_searches=task.max_searches))
            completed = 0
            last_progress = None
            previous_goal = None
            previous_battle_id = None
            while True:
                runner._check_stop()
                if completed >= task.max_battles or time.monotonic() - task_started_at >= task.max_duration_sec:
                    home = confirmed_home(last_verified_home)
                    record(TaskResult(task.id, "limited", "task_limit_reached",
                                      evidence=[home.screenshot_path],
                                      metrics={"completed_battles": completed,
                                               "returned_home": True}))
                    break
                if limit_for_battle_budget(task, completed, phase="before_progress",
                                           known_home=last_verified_home):
                    break
                session.begin_task()
                last_verified_home = None
                try:
                    last_progress = read_progress(session, task)
                except CapabilityUnavailable as exc:
                    # The reader has either issued no input or returned from an anchored panel.
                    home = return_to_village(session)
                    record(TaskResult(task.id, "not_supported", str(exc), evidence=[home.screenshot_path]))
                    break
                if not last_progress.available:
                    record(TaskResult(task.id, "skipped", last_progress.reason, evidence=list(last_progress.evidence)))
                    break
                current_goal = {"current": last_progress.current, "target": last_progress.target,
                                "completed": last_progress.completed, "values": last_progress.values}
                session.goal_progress = current_goal
                if previous_battle_id and last_progress.evidence:
                    record(TaskResult(task.id, "succeeded", "goal_progress_observed",
                        evidence=list(last_progress.evidence), metrics={"receipt_kind": "goal_progress",
                        "battle_id": previous_battle_id, "before": previous_goal, "after": current_goal}), notify=False)
                    previous_battle_id = None
                runner._notify("battle_progress", task_id=task.id, phase="检查目标", battle_number=completed + 1,
                    battles_completed=stats.battles_completed, battles_won=stats.battles_won,
                    goal_current=last_progress.current, goal_target=last_progress.target,
                    resources=last_progress.values if task.kind == "resources" else None)
                if last_progress.completed:
                    stats.goals_completed += 1
                    record(TaskResult(task.id, "succeeded", "goal_reached", evidence=list(last_progress.evidence),
                                      metrics={"goal_current": last_progress.current, "goal_target": last_progress.target,
                                               "goals_completed": stats.goals_completed}))
                    break
                if completed >= task.max_battles or time.monotonic() - task_started_at >= task.max_duration_sec:
                    record(TaskResult(task.id, "limited", "task_limit_reached", evidence=list(last_progress.evidence),
                                      metrics={"completed_battles": completed, "goal_current": last_progress.current,
                                               "goal_target": last_progress.target}))
                    break
                if routine.maintenance_interval_sec and time.monotonic() - last_maintenance >= routine.maintenance_interval_sec:
                    maintenance_limit = None
                    for daily_task in enabled:
                        if daily_task.kind not in BATTLE_KINDS and daily_task.kind not in {"event_reward", "ore_inventory"}:
                            runner._check_stop()
                            if time.monotonic() - task_started_at >= task.max_duration_sec:
                                maintenance_limit = "task_limit_reached"
                                break
                            if limit_for_battle_budget(task, completed, phase="before_maintenance",
                                                       progress=last_progress,
                                                       known_home=last_verified_home):
                                maintenance_limit = "global_battle_budget_insufficient"
                                break
                            last_verified_home = maintenance(daily_task, periodic=True)
                    if maintenance_limit:
                        # A budget result was already recorded when global time was short.
                        if maintenance_limit == "task_limit_reached":
                            home = confirmed_home(last_verified_home)
                            record(TaskResult(task.id, "limited", "task_limit_reached",
                                              evidence=[home.screenshot_path],
                                              metrics={"completed_battles": completed,
                                                       "returned_home": True}))
                        break
                    last_maintenance = time.monotonic()
                    # Collection can itself reach a stock goal. Refresh the goal
                    # before spending another search or starting another battle.
                    continue
                if task.kind == "clan_games" and not task.strategy_file:
                    record(TaskResult(task.id, "not_supported", "Clan building tasks require a configured targeted strategy",
                                      evidence=list(last_progress.evidence)))
                    break
                if task.kind == "clan_games" and not session.config.battle.target_building:
                    session.config = replace(session.config, battle=replace(session.config.battle,
                        target_building=last_progress.values.get("building_type", "")))
                if task.kind == "event":
                    scoring = last_progress.values.get("scoring_condition")
                    if scoring == "destroy_building":
                        if not task.strategy_file:
                            record(TaskResult(task.id, "not_supported", "Event building tasks require a configured targeted strategy",
                                              evidence=list(last_progress.evidence)))
                            break
                        session.config = replace(session.config, battle=replace(session.config.battle,
                            target_building=last_progress.values.get("building_type", "")))
                    if scoring == "deploy_units":
                        from .strategy_config import load_strategy
                        path = Path(task.strategy_file)
                        definition = load_strategy(path if path.is_absolute() else config.source_path.resolve().parent / path) if task.strategy_file else None
                        deployed = {step.unit_id for step in definition.steps
                                    if step.action in {"deploy_troop", "deploy_hero", "deploy_siege", "cast_spell"}} if definition else set()
                        required = set(last_progress.values.get("scoring_units", ()))
                        if not definition or definition.planner or not required or not required <= deployed:
                            record(TaskResult(task.id, "not_supported", "Event scoring units need explicit matching strategy actions",
                                              evidence=list(last_progress.evidence)))
                            break
                if limit_for_battle_budget(task, completed, phase="before_battle",
                                           progress=last_progress):
                    break
                # read_progress and any periodic work used a different task window.
                # Grant the admitted battle its own full window, capped by global.
                session.begin_task()
                battle_id = uuid4().hex
                battle_number = completed + 1

                def progress(phase, **data):
                    gains = resource_metrics(stats)["resources"]
                    runner._notify("battle_progress", task_id=task.id, battle_id=battle_id, phase=phase,
                        battle_number=battle_number, battles_completed=stats.battles_completed,
                        battles_won=stats.battles_won,
                        resource_gains={key: value["battle_gross"] for key, value in gains.items()}, **data)

                # GameSession reports semantic stages, keeping UI updates independent of logging.
                session.progress_callback = progress
                progress("准备军队")
                result = run_battle(session)
                result = replace(result, task=task.id, metrics={**result.metrics,
                    "receipt_kind": "battle", "battle_id": battle_id, "objective": task.kind,
                    "goal_before": current_goal})
                settled_home = (result.metrics.get("returned_home") is True and
                                type(result.metrics.get("rounds_completed")) is int and
                                result.metrics["rounds_completed"] == 1)
                if settled_home and battle_id not in seen_battles:
                    seen_battles.add(battle_id)
                    stats.battles_completed += 1
                    stats.battles_won += int(result.metrics.get("victory") is True)
                    completed += 1
                    stats.cycles += 1
                    current = session.last_snapshot
                    if (current is not None and current.scene == "village"
                            and current.observations.get("village_type") == "home"):
                        last_verified_home = current
                record(result, notify=False)
                if settled_home:
                    progress("已回村", resources=result.metrics.get("resources_after"))
                if result.status in {"not_supported", "skipped"}:
                    home = return_to_village(session)
                    runner._notify("task_result", task=task.id, status=result.status, reason=result.reason,
                                   evidence=[home.screenshot_path], metrics=result.metrics)
                    break
                if result.status == "limited":
                    return_to_village(session)
                    runner._notify("task_result", task=task.id, status="limited", reason=result.reason, metrics=result.metrics)
                    break
                if result.status == "failed":
                    raise _RecordedTaskFailure(f"{task.id}: {result.reason}")
                if result.status != "succeeded" or not settled_home:
                    raise FlowError(f"{task.id}: {result.reason}")
                previous_goal, previous_battle_id = current_goal, battle_id
            session.progress_callback = None
            session.config = config
            active = None
        if not stats.stop_reason:
            stats.stop_reason = "daily task queue finished"
    except (StopRequested, KeyboardInterrupt):
        from .reporting import interrupted_battle_metrics
        frame = getattr(session, "last_snapshot", None)
        record(TaskResult(active.id if active else "daily", "cancelled", "interrupted by user",
                          evidence=[frame.screenshot_path] if frame else [],
                          metrics=interrupted_battle_metrics(session) if active and active.kind in BATTLE_KINDS else {}))
        stats.stop_reason = "interrupted by user"
    except CapabilityUnavailable as exc:
        # Capability issues outside the progress reader still require a verified safe home.
        try:
            home = return_to_village(session) if session else None
            record(TaskResult(active.id if active else "daily", "not_supported", str(exc),
                              evidence=[home.screenshot_path] if home else []))
            stats.stop_reason = str(exc)
        except Exception as recovery_error:
            stats.stop_reason = f"Cannot verify home after unsupported task: {recovery_error}"
            record(TaskResult(active.id if active else "daily", "failed", stats.stop_reason))
    except Exception as exc:
        frame = getattr(session, "last_snapshot", None)
        stats.stop_reason = f"daily task failed: {exc}"
        if not isinstance(exc, _RecordedTaskFailure):
            record(TaskResult(active.id if active else "initialization", "failed", str(exc),
                              elapsed_sec=time.monotonic() - task_started_at,
                              evidence=[frame.screenshot_path] if frame else []))
        runner.logger.exception("Daily task failed")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                stats.record_task(TaskResult("transport_cleanup", "failed", str(exc)))
                stats.stop_reason += f"; transport cleanup failed: {exc}"
        runner._finish_report(stats)
    return stats
