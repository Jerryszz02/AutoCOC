"""Run selected tasks and count only verified game outcomes."""

from collections.abc import Callable
from dataclasses import replace
import logging
import math
from threading import Event
import time
from uuid import uuid4

from .actions import AutomationContext
from .adb import ADBClient
from .config import AppConfig
from .errors import FlowError, StopRequested
from .recovery import confirm_welcome_back, recover_connection, welcome_back_point
from .reporting import RunStats, TaskResult, capture_run_provenance, write_report
from .scene import SceneSnapshot
from .session import GameSession
from .stop_rules import StopController
from .village import collect_resources

FlowFunction = Callable[[AutomationContext], None]


def return_to_village(session: GameSession, *, initial_snapshot: SceneSnapshot | None = None):
    exit_scenes = {"training", "request", "donation", "clan_chat", "search", "popup"}
    snapshot = initial_snapshot if initial_snapshot is not None else session.observe("home-check")
    welcome_confirmed = False
    for exits in range(5):
        session.check_deadline()
        if snapshot.scene == "village":
            break
        if exits == 4:
            raise FlowError("Village recovery exceeded four verified scene exits")
        if snapshot.scene == "popup" and welcome_back_point(snapshot) is not None:
            if not welcome_confirmed:
                confirm_welcome_back(session, snapshot)
                welcome_confirmed = True
            snapshot = session.wait_for((exit_scenes | {"village", "settlement"}) - {"popup"},
                                        timeout_sec=20, label="welcome-home")
            continue
        if snapshot.scene in exit_scenes:
            session.back(snapshot, reason=f"Return from {snapshot.scene} before selected task")
            snapshot = session.wait_for((exit_scenes | {"village", "settlement"}) - {snapshot.scene},
                                        timeout_sec=20, label="returning-home")
        elif snapshot.scene == "settlement":
            session.click(snapshot, "return_home")
            snapshot = session.wait_for({"village"} | exit_scenes, timeout_sec=30, label="settlement-home")
        else:
            snapshot = session.wait_for({"village"} | exit_scenes,
                                        timeout_sec=session.config.game.startup_timeout_sec, label="starting-home")
    buttons = snapshot.observations.get("buttons", [])
    if (not math.isfinite(snapshot.confidence) or snapshot.confidence < 0.8
            or not isinstance(buttons, list)
            or any(sum(button.get("name") == name for button in buttons if isinstance(button, dict)) != 1
                   for name in ("attack", "shop"))):
        raise FlowError("Village is not verified by confident scene and both navigation controls")
    return snapshot


def verify_home(session: GameSession) -> TaskResult:
    initial = session.observe("launch-home-check")
    if initial.scene == "disconnected":
        recovery = recover_connection(session, initial_snapshot=initial)
        return replace(recovery, task="launch", metrics={**recovery.metrics, "startup_recovery": True})
    snapshot = return_to_village(session, initial_snapshot=initial)
    return TaskResult("launch", "succeeded", "Game village and navigation controls recognized",
                      evidence=[snapshot.screenshot_path])


class FlowRunner:
    def __init__(self, config: AppConfig, adb: ADBClient, serial: str,
                 *, logger: logging.Logger | None = None, session_factory=None,
                 stop_event: Event | None = None,
                 progress: Callable[[dict[str, object]], None] | None = None) -> None:
        self.config = config
        self.adb = adb
        self.serial = serial
        self.logger = logger or logging.getLogger(__name__)
        self.session_factory = session_factory or GameSession.connect
        self.stop_event = stop_event
        self.progress = progress

    def _check_stop(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise StopRequested("interrupted by user")

    def run_routine(self, routine) -> RunStats:
        from .daily import run_routine
        return run_routine(self, routine)

    def _notify(self, kind: str, **data: object) -> None:
        if self.progress is not None:
            try:
                self.progress({"kind": kind, **data})
            except Exception:
                self.logger.exception("Unable to deliver progress update")

    def _handler(self, task: str):
        if task == "launch":
            return verify_home
        if task == "recover":
            return recover_connection
        if task == "collect":
            return collect_resources
        if task in {"request", "donate"}:
            from .social import request_reinforcements, donate_troops
            return request_reinforcements if task == "request" else donate_troops
        if task in {"battle", "train"}:
            from .combat import run_battle, inspect_army
            return run_battle if task == "battle" else inspect_army
        if task == "settle":
            raise FlowError("Settlement belongs to the battle task; remove standalone settle from profile")
        raise FlowError(f"No task handler for {task!r}")

    def run_profile(self, profile_name: str) -> RunStats:
        if profile_name not in self.config.profiles:
            raise FlowError(f"Unknown profile {profile_name!r}")
        profile = self.config.profiles[profile_name]
        stats = RunStats(profile=profile_name, device_serial=self.serial,
                         mode="dry-run" if self.config.runtime.dry_run else "live",
                         provenance=capture_run_provenance(self.config, profile_name))
        self._notify("run_started", run_id=stats.run_id, mode=stats.mode, tasks=profile.enabled_tasks)
        if self.stop_event is not None and self.stop_event.is_set():
            stats.stop_reason = "interrupted by user before initialization"
            self._finish_report(stats)
            return stats
        if self.config.runtime.dry_run:
            for task in profile.enabled_tasks:
                self._notify("task_started", task=task, cycle=1)
                stats.record_task(TaskResult(task, "simulated", "Planning only: no game action or outcome was verified"))
                self._notify("task_result", task=task, status="simulated", reason=stats.task_results[-1].reason)
            stats.stop_reason = "dry-run planning complete; real success count remains zero"
            self._finish_report(stats)
            return stats

        stops = StopController(self.config.stop)
        session = None
        active_task = None
        started = time.monotonic()
        try:
            self._check_stop()
            handlers = [(task, self._handler(task)) for task in profile.enabled_tasks]
            connection_options = {"stop_event": self.stop_event} if self.stop_event is not None else {}
            session = self.session_factory(self.config, self.adb, self.serial,
                                           self.config.runtime.report_dir / stats.run_id,
                                           launch="launch" in profile.enabled_tasks, logger=self.logger,
                                           **connection_options)
            stats.events_path = session.events_path
            version = getattr(session, "client_version", None)
            if isinstance(version, str) and version:
                stats.provenance["client_version"] = version
            while True:
                self._check_stop()
                stop = stops.check()
                if stop.should_stop:
                    stats.stop_reason = stop.reason
                    break
                cycle_failed = False
                for task, handler in handlers:
                    self._check_stop()
                    active_task = task
                    started = time.monotonic()
                    self._notify("task_started", task=task, cycle=stats.cycles + 1)
                    try:
                        session.begin_task()
                        result = handler(session)
                        if not isinstance(result, TaskResult):
                            raise FlowError(f"Task {task} did not return a TaskResult receipt")
                        result = replace(result, task=task, elapsed_sec=time.monotonic() - started)
                    except Exception as exc:
                        self.logger.exception("Task %s failed", task)
                        frame = session.last_snapshot
                        result = TaskResult(task, "failed", str(exc), elapsed_sec=time.monotonic() - started,
                                            evidence=[frame.screenshot_path] if frame else [])
                    if task == "battle":
                        result = replace(result, metrics={**result.metrics,
                                         "receipt_kind": "battle", "battle_id": uuid4().hex})
                        if (result.metrics.get("returned_home") is True and
                                type(result.metrics.get("rounds_completed")) is int and
                                result.metrics["rounds_completed"] == 1):
                            stats.battles_completed += 1
                            stats.battles_won += int(result.metrics.get("victory") is True)
                    stats.record_task(result)
                    active_task = None
                    self._record_resolution(session, stats)
                    event_saved = self._record_event(session, stats, result)
                    self._notify("task_result", task=task, status=result.status, reason=result.reason,
                                 evidence=result.evidence, metrics=result.metrics)
                    if result.status == "failed":
                        cycle_failed = True
                        stats.failure_screenshots.extend(result.evidence[-1:])
                        break
                    if result.status == "succeeded" and self.config.reporting.save_success_screenshots:
                        stats.success_screenshots.extend(result.evidence[-1:])
                    if not event_saved:
                        cycle_failed = True
                        break
                stats.cycles += 1
                if cycle_failed:
                    stops.record_failure()
                else:
                    stops.record_success()
                if cycle_failed:
                    if not stats.stop_reason:
                        stats.stop_reason = f"task failed: {result.task}: {result.reason}"
                    break
        except KeyboardInterrupt:
            if active_task is not None:
                frame = session.last_snapshot if session is not None else None
                stats.record_task(TaskResult(active_task, "failed", "interrupted by user before verification completed",
                                            elapsed_sec=time.monotonic() - started,
                                            evidence=[frame.screenshot_path] if frame else [],
                                            metrics={"interrupted": True}))
            stats.stop_reason = "interrupted by user"
        except Exception as exc:
            stats.record_task(TaskResult("initialization", "failed", str(exc)))
            stats.stop_reason = f"initialization failed: {exc}"
            self.logger.exception("Run initialization failed")
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as exc:
                    stats.record_task(TaskResult("transport_cleanup", "failed", str(exc)))
                    stats.stop_reason = f"{stats.stop_reason}; transport cleanup failed"
            self._finish_report(stats)
        return stats

    def run(self, flow: FlowFunction) -> RunStats:
        raise FlowError("Unverified custom flows cannot count as game successes; use a task profile")

    def _write_report(self, stats: RunStats) -> None:
        if self.config.reporting.write_markdown:
            path = write_report(self.config.runtime.report_dir, stats,
                                save_decision_trace=self.config.reporting.save_decision_trace)
            self.logger.info("Report: %s", path)

    def _finish_report(self, stats: RunStats) -> None:
        try:
            self._write_report(stats)
        except (Exception, KeyboardInterrupt) as exc:
            reason = "interrupted by user while writing report" if isinstance(exc, KeyboardInterrupt) else str(exc)
            stats.record_task(TaskResult("report", "failed", reason))
            stats.stop_reason = f"{stats.stop_reason}; report failed: {reason}".strip("; ")
            self.logger.error("Unable to save run report; existing task results are retained: %s", reason)

    def _record_resolution(self, session: GameSession, stats: RunStats) -> None:
        if session.last_snapshot is None:
            return
        resolution = session.last_snapshot.observations.get("source_resolution")
        if isinstance(resolution, (tuple, list)) and len(resolution) == 2 and all(type(value) is int and value > 0 for value in resolution):
            stats.screenshot_resolution = tuple(resolution)
        else:
            self.logger.warning("Latest observation has no valid source resolution; task receipt is preserved")

    def _record_event(self, session: GameSession, stats: RunStats, result: TaskResult) -> bool:
        try:
            session.event("task_result", task=result.task, status=result.status, reason=result.reason,
                          evidence=result.evidence, metrics=result.metrics)
        except Exception as exc:
            reason = f"Unable to log {result.task} result: {exc}"
            stats.record_task(TaskResult("event_log", "failed", reason, evidence=result.evidence))
            stats.stop_reason = reason
            self.logger.exception("Task result event could not be saved")
            return False
        return True


def sample_flow(context: AutomationContext) -> None:
    """Compatibility capture utility, never a completed game task."""
    context.screenshot("start")
    context.screenshot("done")
