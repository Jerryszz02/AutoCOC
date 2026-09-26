"""Desktop settings, report browsing and a cooperative background runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import logging
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from uuid import uuid4

from .config import AppConfig, ProfileConfig, load_config
from .device import DeviceManager
from .strategies import STRATEGIES
from .errors import ConfigError, StopRequested
from .flow import FlowRunner
from .reporting import RunStats, TaskResult, summarize_run, write_report
from .routine_config import RoutineConfig, default_routine, routine_from_dict, routine_to_dict


TASK_LABELS = {"launch": "连接与启动", "collect": "收集资源", "request": "请求增援",
               "donate": "部落捐兵", "train": "检查军队", "battle": "自动对战", "recover": "恢复连接",
               "resources": "刷资源", "event": "刷活动", "clan_games": "部落竞赛"}
STATUS_LABELS = {"succeeded": "成功", "failed": "失败", "skipped": "跳过", "simulated": "预演",
                 "running": "进行中", "pending": "等待", "not_supported": "未支持",
                 "cancelled": "已停止", "limited": "达到限额"}
STRATEGY_LABELS = {name: entry.label for name, entry in STRATEGIES.items()}


def display_reason(reason: str) -> str:
    return {"Planning only: no game action or outcome was verified": "预演完成，未操作游戏",
            "dry-run planning complete; real success count remains zero": "离线预演完成，真实成功数为 0",
            "interrupted by user": "已按用户要求停止",
            "interrupted by user before verification completed": "用户停止，当前任务结果尚未核验",
            "request_record_and_cooldown_verified": "请求记录与冷却已核验",
            "request_cooldown": "增援请求仍在冷却",
            "no_donation_requests_in_scanned_chat": "已扫描范围内没有可捐请求",
            "donation_interaction_not_supported": "捐兵窗口尚未完成实机适配",
            "goal_reached": "游戏中的目标已达成",
            "goal_progress_observed": "已读取游戏中的目标进度",
            "task_limit_reached": "已达到场数或时长上限，目标尚未达成",
            "search_limit_reached_without_target": "搜索上限内未找到符合条件的对手",
            "daily task queue finished": "本次任务已全部处理",
            "event_inactive": "当前活动已结束或不适用",
            "no_accepted_clan_challenge": "尚未接受支持的部落竞赛任务",
            "No calibrated progress adapter selected for this task": "请先配置当前活动或竞赛的进度识别",
            "strategy_actions_settlement_and_return_verified": "打法动作、结算及回村已确认",
            "edrag_line_troops_settlement_and_return_verified": "单边部队、英雄投放、结算及回村已确认",
            "two_edge_troops_settlement_and_return_verified": "两边部队投放、结算及回村已确认",
            "Game village and navigation controls recognized": "村庄及导航入口已确认"}.get(reason, reason)


@dataclass(frozen=True)
class RunOptions:
    tasks: tuple[str, ...]
    max_runs: int
    max_duration_sec: int
    min_expected_resources: int
    max_searches: int
    serial: str
    dry_run: bool = True
    strategy: str = "verified"
    routine: RoutineConfig | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "RunOptions":
        profile = config.profiles.get("core-loop", next(iter(config.profiles.values())))
        return cls(profile.enabled_tasks, config.stop.max_runs, config.stop.max_duration_sec,
                   config.battle.min_expected_resources, config.battle.max_searches, config.adb.manual_serial,
                   strategy=config.battle.strategy, routine=config.routine)

    def validate(self) -> None:
        if not self.tasks or self.tasks[0] != "launch":
            raise ConfigError("任务必须以连接与启动开始")
        if any(task not in TASK_LABELS for task in self.tasks) or len(set(self.tasks)) != len(self.tasks):
            raise ConfigError("任务包含未支持的名称或重复项")
        for name, value, minimum in (("轮次", self.max_runs, 1), ("运行时长", self.max_duration_sec, 1),
                                     ("最低资源", self.min_expected_resources, 0), ("搜索上限", self.max_searches, 1)):
            if type(value) is not int or value < minimum:
                raise ConfigError(f"{name}必须为不小于 {minimum} 的整数")
        if type(self.dry_run) is not bool or not isinstance(self.serial, str):
            raise ConfigError("运行模式或设备地址无效")
        if self.strategy not in STRATEGY_LABELS:
            raise ConfigError("对战策略无效")
        if self.routine is not None:
            self.routine.validate()


def desktop_config(path: Path, options: RunOptions) -> AppConfig:
    options.validate()
    config = load_config(path.resolve())
    base = config.source_path.parent

    def absolute(value: Path) -> Path:
        return value if value.is_absolute() else (base / value).resolve()

    return replace(config, adb=replace(config.adb, manual_serial=options.serial.strip()),
                   runtime=replace(config.runtime, dry_run=options.dry_run,
                                   screenshot_dir=absolute(config.runtime.screenshot_dir),
                                   report_dir=absolute(config.runtime.report_dir)),
                   vision=replace(config.vision, template_dir=absolute(config.vision.template_dir)),
                   stop=replace(config.stop, max_runs=options.max_runs, max_duration_sec=options.max_duration_sec),
                   battle=replace(config.battle, min_expected_resources=options.min_expected_resources,
                                  max_searches=options.max_searches, strategy=options.strategy),
                   reporting=replace(config.reporting, write_markdown=True),
                   profiles={"desktop": ProfileConfig(tuple(task for task in options.tasks if task in
                                                             {"launch", "collect", "request", "donate", "train", "battle", "recover"}))},
                   routine=options.routine)


def settings_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + ".desktop.json")


def save_options(path: Path, options: RunOptions) -> None:
    options.validate()
    # Opening the application must never restore a previous live-run choice.
    payload = asdict(replace(options, dry_run=True))
    if options.routine is not None:
        payload["routine"] = routine_to_dict(options.routine)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_options(path: Path, config: AppConfig) -> RunOptions:
    if not path.exists():
        return RunOptions.from_config(config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            raise ValueError("方案格式无效")
        payload["tasks"] = tuple(payload["tasks"])
        payload["dry_run"] = True
        if payload.get("routine") is not None:
            payload["routine"] = routine_from_dict(payload["routine"])
        options = RunOptions(**payload)
        options.validate()
        if options.routine is None:
            options = replace(options, routine=default_routine(desktop_config(config.source_path, options)))
        return options
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"无法读取界面方案：{exc}") from exc


def report_history(directory: Path, limit: int = 100) -> tuple[list[tuple[Path, dict]], list[str]]:
    reports, errors = [], []
    for path in sorted(directory.glob("run-*.json"), reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("mode") not in {"live", "dry-run"}:
                raise ValueError("不是有效的运行报告")
            if not isinstance(payload.get("task_results"), list):
                raise ValueError("缺少任务结果")
            for result in payload["task_results"]:
                if not isinstance(result, dict) or not isinstance(result.get("evidence", []), list):
                    raise ValueError("任务结果格式无效")
                if any(not isinstance(name, str) for name in result.get("evidence", [])):
                    raise ValueError("截图路径格式无效")
            if payload.get("resource_metrics") is not None and not isinstance(payload["resource_metrics"], dict):
                raise ValueError("收益字段格式无效")
            reports.append((path, payload))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    return reports, errors


def report_text(payload: dict) -> str:
    mode = "离线预演 · 不计真实成功与收益" if payload.get("mode") == "dry-run" else "实机运行 · 是否验收以证据为准"
    lines = [mode, f"运行：{payload.get('run_id', '未知')}", f"开始：{payload.get('started_at', '未知')}",
             f"停止原因：{display_reason(str(payload.get('stop_reason', '未知')))}", ""]
    for result in payload.get("task_results", []):
        if isinstance(result, dict):
            lines.extend([f"{TASK_LABELS.get(result.get('task'), result.get('task'))} · "
                          f"{STATUS_LABELS.get(result.get('status'), result.get('status'))}",
                          display_reason(str(result.get("reason", "未知"))), ""])
    if payload.get("mode") == "live":
        lines.append("战斗：完成 {battles_completed} / 胜利 {battles_won} / 目标达成 {goals_completed}".format(
            battles_completed=payload.get("battles_completed", 0),
            battles_won=payload.get("battles_won", 0),
            goals_completed=payload.get("goals_completed", 0)))
    metrics = payload.get("resource_metrics") or {}
    for name, field in (("金币与圣水战斗毛产出 / 小时", "battle_gold_elixir_per_hour"),
                        ("金币与圣水经营净产出 / 小时", "net_gold_elixir_per_hour")):
        value = metrics.get(field)
        display = "不适用（离线预演）" if payload.get("mode") != "live" else "未知" if value is None else str(value)
        lines.append(f"{name}：{display}")
    return "\n".join(lines)


class _QueueLogHandler(logging.Handler):
    def __init__(self, events: Queue) -> None:
        super().__init__()
        self.events = events
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.events.put({"kind": "log", "text": self.format(record)})


class DesktopController:
    def __init__(self) -> None:
        self.events: Queue[dict] = Queue()
        self.stop_event = Event()
        self.thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, path: Path, options: RunOptions) -> None:
        if self.running:
            raise RuntimeError("已有任务正在运行")
        config = desktop_config(path, options)
        if config.routine is not None:
            from .daily import validate_routine_files
            validate_routine_files(config, config.routine)
            if not any(task.enabled for task in config.routine.tasks):
                raise ConfigError("请至少勾选一项本次任务")
        self.stop_event.clear()
        self.thread = Thread(target=self._run, args=(config,), name="AutoCOC-runner", daemon=False)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self, config: AppConfig) -> None:
        logger = logging.getLogger("autococ.desktop." + uuid4().hex)
        logger.setLevel(config.runtime.log_level)
        logger.propagate = False
        logger.addHandler(_QueueLogHandler(self.events))
        stats = None
        report = None
        try:
            config.runtime.report_dir.mkdir(parents=True, exist_ok=True)
            log_path = config.runtime.report_dir / f"desktop-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
            if self.stop_event.is_set():
                raise StopRequested()
            if config.runtime.dry_run:
                adb, serial = None, "dry-run"
            else:
                self.events.put({"kind": "connecting"})
                manager = DeviceManager(config, logger=logger)
                device = manager.connect()
                adb, serial = manager.adb, device.serial
            runner = FlowRunner(config, adb, serial, logger=logger, stop_event=self.stop_event,
                                progress=self.events.put)
            stats = runner.run_routine(config.routine) if config.routine is not None else runner.run_profile("desktop")
            report = config.runtime.report_dir / f"run-{stats.run_id}.json"
        except (Exception, KeyboardInterrupt) as exc:
            stats = RunStats(profile="desktop", mode="dry-run" if config.runtime.dry_run else "live")
            interrupted = isinstance(exc, KeyboardInterrupt)
            stats.stop_reason = "interrupted by user" if interrupted else f"initialization failed: {exc}"
            stats.record_task(TaskResult("initialization", "cancelled" if interrupted else "failed", stats.stop_reason,
                                         metrics={"interrupted": interrupted}))
            try:
                report = write_report(config.runtime.report_dir, stats).with_suffix(".json")
            except OSError as report_error:
                self.events.put({"kind": "error", "text": f"报告保存失败：{report_error}"})
            if not interrupted:
                self.events.put({"kind": "error", "text": str(exc)})
        finally:
            if stats is not None:
                self.events.put({"kind": "finished", "summary": summarize_run(stats),
                                 "report": report if report is not None and report.is_file() else None})
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
