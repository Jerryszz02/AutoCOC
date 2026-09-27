"""Sequential daily tasks, independent battle goals, and verified live counters."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from uuid import uuid4

from .config import ProfileConfig
from .errors import CapabilityUnavailable, ConfigError, FlowError, StopRequested
from .objectives import load_adapter, read_progress
from .reporting import RunStats, TaskResult, capture_run_provenance, resource_metrics
from .routine_config import BATTLE_KINDS, RoutineConfig


class _RecordedTaskFailure(FlowError):
    """Abort the queue without recording the same task failure twice."""


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
        result = runner._handler(spec.kind)(session)
        # Unsupported donation remains an explicit capability result, not an empty chat.
        if result.status == "failed" and result.reason == "donation_interaction_not_supported":
            result = replace(result, status="not_supported")
        result = replace(result, task=spec.id,
                         metrics={**result.metrics, "receipt_kind": spec.kind, "periodic": periodic})
        record(result, notify=not periodic)
        if result.status == "failed":
            raise _RecordedTaskFailure(f"{spec.id}: {result.reason}")
        return_to_village(session)

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
        for task in enabled:
            runner._check_stop()
            active = task
            task_started_at = time.monotonic()
            if task.kind not in BATTLE_KINDS:
                maintenance(task)
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
                session.begin_task()
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
                    for daily_task in enabled:
                        if daily_task.kind not in BATTLE_KINDS:
                            maintenance(daily_task, periodic=True)
                    last_maintenance = time.monotonic()
                    session.begin_task()
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
