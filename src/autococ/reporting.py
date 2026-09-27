"""Logging and report output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
import platform
import time
import tomllib
from typing import Literal
from uuid import uuid4

from .config import AppConfig


@dataclass(frozen=True)
class DecisionTrace:
    attempt: int
    task: str
    scene: str
    confidence: float
    plan_reason: str
    actions: tuple[str, ...]
    expected_next_scene: str | None


@dataclass
class TaskResult:
    task: str
    status: Literal["succeeded", "skipped", "failed", "simulated", "not_supported", "cancelled", "limited"]
    reason: str
    started_at: datetime = field(default_factory=datetime.now)
    elapsed_sec: float = 0.0
    evidence: list[Path] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "skipped", "failed", "simulated", "not_supported", "cancelled", "limited"}:
            raise ValueError(f"Unknown task status: {self.status}")
        if self.status == "succeeded" and not self.evidence:
            raise ValueError("A succeeded task requires verification evidence")


def _new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid4().hex[:8]}"


@dataclass
class RunStats:
    started_at: datetime = field(default_factory=datetime.now)
    started_monotonic: float = field(default_factory=time.monotonic)
    profile: str = ""
    device_serial: str = ""
    screenshot_resolution: tuple[int, int] | None = None
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    stop_reason: str = ""
    decision_traces: list[DecisionTrace] = field(default_factory=list)
    success_screenshots: list[Path] = field(default_factory=list)
    failure_screenshots: list[Path] = field(default_factory=list)
    failure_xml: list[Path] = field(default_factory=list)
    mode: Literal["live", "dry-run"] = "live"
    cycles: int = 0
    skipped: int = 0
    simulated: int = 0
    task_results: list[TaskResult] = field(default_factory=list)
    events_path: Path | None = None
    run_id: str = field(default_factory=_new_run_id)
    provenance: dict[str, object] = field(default_factory=dict)
    battles_completed: int = 0
    battles_won: int = 0
    goals_completed: int = 0

    def record_task(self, result: TaskResult) -> None:
        if self.mode != "live" and result.status == "succeeded":
            result = replace(result, status="simulated")
        self.task_results.append(result)
        if result.status == "succeeded":
            self.successes += 1
            self.attempts += 1
        elif result.status == "failed":
            self.failures += 1
            self.attempts += 1
        elif result.status == "skipped":
            self.skipped += 1
        elif result.status == "simulated":
            self.simulated += 1

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_monotonic


def capture_run_provenance(config: AppConfig, profile_name: str) -> dict[str, object]:
    """Snapshot the on-disk inputs before session creation; never disclose config values."""
    root = Path(__file__).resolve().parents[2]
    files: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    for relative, suffix in (("src/autococ", ".py"), ("assets/templates", None)):
        try:
            paths = sorted(path for path in (root / relative).iterdir()
                           if path.is_file() and (suffix is None or path.suffix == suffix))
            if not paths:
                errors[relative] = "no matching files"
            for path in paths:
                name = path.relative_to(root).as_posix()
                try:
                    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    files[name] = None
                    errors[name] = type(exc).__name__
        except OSError as exc:
            errors[relative] = type(exc).__name__
    # Catalog samples and declarative actions change what can be clicked, too.
    # These optional directories also keep provenance compatible with old installs.
    for relative in ("assets/catalogs", "strategies"):
        for path in sorted((root / relative).rglob("*")):
            if path.is_file() and path.suffix in {".json", ".toml", ".png", ".py"}:
                name = path.relative_to(root).as_posix()
                try:
                    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    files[name] = None
                    errors[name] = type(exc).__name__
    configured_paths = [config.battle.strategy_file]
    if config.routine:
        for task in config.routine.tasks:
            if task.enabled:
                configured_paths.extend((task.strategy_file, task.goal.adapter_path))
    pending = [(config.source_path.resolve().parent, path) for path in configured_paths if path]
    visited = set()
    while pending:
        base, configured = pending.pop()
        path = (base / configured).resolve()
        if path in visited:
            continue
        visited.add(path)
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else f"configured/{path.as_posix()}"
        try:
            raw = path.read_bytes()
            files[name] = hashlib.sha256(raw).hexdigest()
            if path.suffix == ".toml":
                fields = tomllib.loads(raw.decode("utf-8"))
                if isinstance(fields.get("planner"), str) and fields["planner"]:
                    pending.append((path.parent, fields["planner"]))
                if isinstance(fields.get("entry_template"), str) and fields["entry_template"]:
                    pending.append((config.vision.template_dir, f"{fields['entry_template']}.png"))
        except (OSError, ValueError) as exc:
            files[name] = None
            errors[name] = type(exc).__name__
    vision_artifacts: dict[str, str | None] = {"model_manifest_sha256": None,
                                               "layout_sha256": None, "guide_sha256": None}
    if config.vision_agent.enabled:
        base = config.source_path.resolve().parent

        def add_artifact(name: str, configured: str) -> str | None:
            if not configured:
                errors[name] = "not configured"
                return None
            path = Path(configured)
            path = path if path.is_absolute() else base / path
            try:
                if path.is_symlink() or not path.is_file():
                    raise OSError("artifact is not a regular file")
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[name] = digest.hexdigest()
                return files[name]
            except OSError as exc:
                files[name] = None
                errors[name] = type(exc).__name__
                return None

        vision_artifacts["layout_sha256"] = add_artifact(
            "configured/vision_agent/layout_profile", config.vision_agent.layout_profile)
        if config.vision_agent.guide_file:
            vision_artifacts["guide_sha256"] = add_artifact(
                "configured/vision_agent/guide_file", config.vision_agent.guide_file)
        model_name = "configured/vision_agent/model_dir"
        if not config.vision_agent.model_dir:
            errors[model_name] = "not configured"
        else:
            model_dir = Path(config.vision_agent.model_dir)
            model_dir = model_dir if model_dir.is_absolute() else base / model_dir
            try:
                if model_dir.is_symlink() or not model_dir.is_dir():
                    raise OSError("model directory unavailable")
                model_files = sorted(path for path in model_dir.rglob("*") if path.is_file())
                if not model_files:
                    errors[model_name] = "no files"
                else:
                    for path in model_files:
                        relative = path.relative_to(model_dir).as_posix()
                        add_artifact(f"{model_name}/{relative}", str(path))
                    model_hashes = {name: value for name, value in files.items()
                                    if name.startswith(model_name + "/")}
                    if all(value is not None for value in model_hashes.values()):
                        vision_artifacts["model_manifest_sha256"] = hashlib.sha256(
                            json.dumps(model_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
            except OSError as exc:
                errors[model_name] = type(exc).__name__
    # Hash the sorted path/hash manifest, so adding or removing a file also
    # changes the fingerprint. An incomplete manifest has no aggregate hash.
    fingerprint = (hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                   if files and not errors else None)
    config_hash = None
    try:
        effective = asdict(config)
        effective.pop("source_path", None)  # The config filename is not an effective setting.
        serialized = json.dumps(effective, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, default=_json_value, allow_nan=False)
        config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        errors["effective_config"] = type(exc).__name__
    dependencies = {}
    for package in ("opencv-python", "rapidocr", "onnxruntime", "numpy"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "captured_at": datetime.now().isoformat(timespec="microseconds"),
        "scope": "on-disk files at run start; not loaded module bytecode",
        "file_scope": ["src/autococ/*.py", "assets/templates/* (files only)",
                       "assets/catalogs/**", "strategies/**", "configured strategy, planner and adapter files",
                       "configured vision model, layout and guide files when enabled"],
        "source_fingerprint_sha256": fingerprint,
        "files_sha256": files,
        "effective_config_sha256": config_hash,
        "vision_agent_artifacts": vision_artifacts,
        "package_name": config.game.package_name,
        "client_version": "unknown",
        "python_version": platform.python_version(),
        "dependencies": dependencies,
        "baseline_resolution": list(config.game.baseline_resolution),
        "profile": profile_name,
        "tasks": list(config.profiles[profile_name].enabled_tasks),
        "errors": errors,
    }


def setup_logging(log_level: str, report_dir: Path, *, run_id: str | None = None) -> logging.Logger:
    report_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("autococ")
    logger.setLevel(getattr(logging, log_level))
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(getattr(logging, log_level))
    logger.addHandler(stream_handler)

    log_path = report_dir / f"run-{run_id or _new_run_id()}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level))
    logger.addHandler(file_handler)
    return logger


def write_report(report_dir: Path, stats: RunStats, *, save_decision_trace: bool = True) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"run-{stats.run_id}.md"
    payload = summarize_run(stats, save_decision_trace=save_decision_trace)
    lines = [
        "# AutoCOC Run Report",
        "",
        f"- Profile: {stats.profile or 'not set'}",
        f"- Run ID: {stats.run_id}",
        f"- Mode: {stats.mode}",
        f"- Started: {stats.started_at.isoformat(timespec='seconds')}",
        f"- Device serial: {stats.device_serial or 'not set'}",
        f"- Screenshot resolution: {_format_resolution(stats.screenshot_resolution)}",
        f"- Attempts: {payload['attempts']}",
        f"- Profile cycles: {stats.cycles}",
        f"- Successes: {payload['successes']}",
        f"- Verified battles completed: {payload['battles_completed']}",
        f"- Verified battles won: {payload['battles_won']}",
        f"- Goals completed: {payload['goals_completed']}",
        f"- Failures: {payload['failures']}",
        f"- Skipped: {payload['skipped']}",
        f"- Unsupported: {payload['not_supported']}",
        f"- Reached limits: {payload['limited']}",
        f"- Cancelled: {payload['cancelled']}",
        f"- Simulated: {payload['simulated']}",
        f"- Elapsed seconds: {payload['elapsed_sec']:.2f}",
        f"- Stop reason: {stats.stop_reason or 'not set'}",
        f"- Events: {stats.events_path or 'not recorded'}",
        f"- Machine-readable report: {report_path.with_suffix('.json').name}",
        "",
        "## Run Provenance",
        "",
        f"- Source fingerprint (SHA256): {stats.provenance.get('source_fingerprint_sha256') or 'unknown'}",
        f"- Effective config (SHA256): {stats.provenance.get('effective_config_sha256') or 'unknown'}",
        f"- Vision model manifest (SHA256): {(stats.provenance.get('vision_agent_artifacts') or {}).get('model_manifest_sha256') or 'unknown'}",
        f"- Vision layout (SHA256): {(stats.provenance.get('vision_agent_artifacts') or {}).get('layout_sha256') or 'unknown'}",
        f"- Snapshot scope: {stats.provenance.get('scope', 'unknown')}",
        f"- Snapshot time: {stats.provenance.get('captured_at', 'unknown')}",
        f"- Package: {stats.provenance.get('package_name', 'unknown')}",
        f"- Client version: {stats.provenance.get('client_version', 'unknown')}",
        f"- Baseline resolution: {_format_resolution(stats.provenance.get('baseline_resolution'))}",
        f"- Profile tasks: {', '.join(stats.provenance.get('tasks', [])) or 'unknown'}",
        f"- Snapshot errors: {json.dumps(stats.provenance.get('errors', {}), ensure_ascii=False)}",
        "- Per-file SHA256 values are in the machine-readable report's provenance.files_sha256.",
        "",
        "## Task Results",
        "",
    ]
    for result in payload["task_results"]:
        lines.extend([
            f"### {result['task']}: {result['status']}", "",
            f"- Reason: {result['reason']}",
            f"- Elapsed seconds: {result['elapsed_sec']:.2f}",
            "- Evidence: " + (", ".join(str(path) for path in result["evidence"]) or "none"), "",
        ])
    if not stats.task_results:
        lines.extend(["- No verified task results recorded.", ""])

    vision = payload["vision_agent_metrics"]
    if vision["attempts"]:
        lines.extend(["## Vision Agent Deployment", "",
                      f"- Attempts: {vision['attempts']}",
                      f"- Complete input batches: {vision['complete_input_batches']}",
                      f"- Consumption verified: {vision['consumption_verified']}",
                      f"- Full batches within frozen limit: {vision['within_burst_limit']}",
                      f"- Incomplete or over-limit attempts: {vision['not_within_burst_limit']}",
                      "- Input sent, consumption verified, and battle completed are separate evidence states.", ""])
        for attempt in vision["details"]:
            lines.append("- " + f"{attempt['task']} ({attempt['task_status']}, {attempt['mode']}): "
                         f"plan={_format_state(attempt['plan_ready'])}, "
                         f"input={_format_state(attempt['input_sent'])}, "
                         f"consumption={_format_state(attempt['consumption_verified'])}, "
                         f"burst={_format_number(attempt['burst_elapsed_sec'])}/"
                         f"{_format_number(attempt['burst_limit_sec'])} s, "
                         f"inputs={_format_number(attempt['input_count'])}, "
                         f"speed={attempt['speed_status']}")
        lines.append("")

    revenue = payload["resource_metrics"]
    lines.extend([
        "## Resource Ledger", "",
        "Unknown values are not zero. Rates use the entire run wall-clock duration.", "",
        "| Resource | Battle loot | Bonus | Collected | Donation spend | Search spend | Battle gross / h | Operating net / h |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for resource, values in revenue["resources"].items():
        columns = ("battle_loot", "battle_bonus", "collected_credited", "donation_spend", "search_spend", "battle_gross_per_hour", "operating_net_per_hour")
        lines.append(f"| {resource} | " + " | ".join(_format_number(values[key]) for key in columns) + " |")
    lines.extend([
        "",
        f"- Gold + elixir battle gross / hour: {_format_number(revenue['battle_gold_elixir_per_hour'])}",
        f"- Gold + elixir operating net / hour: {_format_number(revenue['net_gold_elixir_per_hour'])}",
        "- Full acceptance remains subject to the evidence requirements in docs/ACCEPTANCE.md.", "",
    ])

    if save_decision_trace:
        lines.extend(["## Decision Trace", ""])
    if save_decision_trace and stats.decision_traces:
        for trace in stats.decision_traces:
            lines.extend(
                [
                    f"### Attempt {trace.attempt}: {trace.task}",
                    "",
                    f"- Scene: {trace.scene} ({trace.confidence:.2f})",
                    f"- Plan: {trace.plan_reason}",
                    f"- Expected next scene: {trace.expected_next_scene or 'none'}",
                    f"- Actions: {', '.join(trace.actions) or 'none'}",
                    "",
                ]
            )
    elif save_decision_trace:
        lines.append("- none")
        lines.append("")

    lines.extend(["## Success Screenshots", ""])
    lines.extend(f"- {path}" for path in stats.success_screenshots)
    if not stats.success_screenshots:
        lines.append("- none")

    lines.extend(["", "## Failure Screenshots", ""])
    lines.extend(f"- {path}" for path in stats.failure_screenshots)
    if not stats.failure_screenshots:
        lines.append("- none")

    lines.extend(["", "## Failure UI XML", ""])
    lines.extend(f"- {path}" for path in stats.failure_xml)
    if not stats.failure_xml:
        lines.append("- none")

    serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value, allow_nan=False)
    with report_path.open("x", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")
    with report_path.with_suffix(".json").open("x", encoding="utf-8") as report:
        report.write(serialized + "\n")
    return report_path


def summarize_run(stats: RunStats, *, save_decision_trace: bool = True) -> dict[str, object]:
    elapsed_sec = stats.elapsed_sec
    results = [
        replace(result, status="simulated") if stats.mode != "live" and result.status == "succeeded" else result
        for result in stats.task_results
    ]
    counts = {status: sum(result.status == status for result in results) for status in ("succeeded", "failed", "skipped", "simulated")}
    payload = asdict(stats)
    payload.pop("started_monotonic")
    payload.update({
        "elapsed_sec": elapsed_sec,
        "attempts": counts["succeeded"] + counts["failed"],
        "successes": counts["succeeded"],
        "failures": counts["failed"],
        "skipped": counts["skipped"],
        "simulated": counts["simulated"],
        "task_results": [asdict(result) for result in results],
        "resource_metrics": resource_metrics(stats, elapsed_sec=elapsed_sec),
        "vision_agent_metrics": vision_agent_metrics(stats),
        "battles_completed": stats.battles_completed if stats.mode == "live" else 0,
        "battles_won": stats.battles_won if stats.mode == "live" else 0,
        "goals_completed": stats.goals_completed if stats.mode == "live" else 0,
        "not_supported": sum(r.status == "not_supported" for r in results),
        "cancelled": sum(r.status == "cancelled" for r in results),
        "limited": sum(r.status == "limited" for r in results),
    })
    if not save_decision_trace:
        payload.pop("decision_traces")
    return payload


def interrupted_battle_metrics(session) -> dict[str, object]:
    """Preserve issued inputs when cooperative stop prevents further observation."""
    receipt = getattr(session, "prepared_battle_receipt", None)
    if not isinstance(receipt, dict):
        return {}
    return {"deployment": receipt, "vision_agent": {
        "mode": "continuous", "status": "cancelled",
        "plan_ready": bool(receipt.get("plan_id")),
        "input_sent": receipt.get("input_sent") is True,
        "consumption_verified": receipt.get("verified") is True,
        "burst_pass": receipt.get("burst_pass") is True,
        "input_count": receipt.get("input_count"),
        "burst_elapsed_sec": receipt.get("burst_elapsed_sec"),
        "burst_limit_sec": receipt.get("burst_limit_sec"),
        "attempted_placements": receipt.get("attempted_placements", 0)}}


def vision_agent_metrics(stats: RunStats) -> dict[str, object]:
    """Keep incomplete deployment attempts and unknown evidence visible."""
    details: list[dict[str, object]] = []
    for result in stats.task_results:
        raw = result.metrics.get("vision_agent")
        if not isinstance(raw, dict):
            continue
        mode = raw.get("mode") if raw.get("mode") in {"continuous", "enhanced"} else "unknown"
        stages = {name: raw.get(name) if type(raw.get(name)) is bool else None
                  for name in ("plan_ready", "input_sent", "consumption_verified")}
        elapsed = raw.get("burst_elapsed_sec")
        elapsed = elapsed if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0 else None
        limit = raw.get("burst_limit_sec")
        limit = limit if type(limit) in (int, float) and math.isfinite(limit) and limit > 0 else None
        count = raw.get("input_count")
        count = count if type(count) is int and count > 0 else None
        if mode != "continuous":
            speed_status = "not_applicable"
        elif stats.mode != "live":
            speed_status = "simulated"
        elif any(value is False for value in stages.values()):
            speed_status = "failed"
        elif None in stages.values() or elapsed is None or limit is None or count is None:
            speed_status = "unknown"
        elif limit > min(15.0, 1.0 + .25 * count) + 1e-9 or elapsed > limit:
            speed_status = "failed"
        else:
            speed_status = "passed"
        details.append({"task": result.task, "task_status": result.status,
                        "mode": mode, "status": raw.get("status") if isinstance(raw.get("status"), str) else None,
                        **stages, "burst_elapsed_sec": elapsed, "burst_limit_sec": limit,
                        "input_count": count, "speed_status": speed_status})
    applicable = [item for item in details if item["mode"] == "continuous" and stats.mode == "live"]
    return {"attempts": len(details),
            "plan_ready": sum(item["plan_ready"] is True for item in details),
            "complete_input_batches": sum(item["input_sent"] is True for item in details),
            "consumption_verified": sum(item["consumption_verified"] is True for item in details),
            "within_burst_limit": sum(item["speed_status"] == "passed" for item in applicable),
            "not_within_burst_limit": sum(item["speed_status"] != "passed" for item in applicable),
            "details": details}


def resource_metrics(stats: RunStats, *, elapsed_sec: float | None = None) -> dict[str, object]:
    elapsed = stats.elapsed_sec if elapsed_sec is None else elapsed_sec
    live = stats.task_results if stats.mode == "live" else []
    battles = [result for result in live if
               (result.status == "succeeded" and result.task == "battle") or
               (result.metrics.get("receipt_kind") == "battle" and
                result.metrics.get("returned_home") is True and
                type(result.metrics.get("rounds_completed")) is int and
                result.metrics["rounds_completed"] == 1)]
    collections = [result for result in live if (result.task == "collect" or result.metrics.get("receipt_kind") == "collect")
                   and result.status in {"succeeded", "failed"}]
    donations = [result for result in live if (result.task in {"donate", "donation"} or result.metrics.get("receipt_kind") == "donate")
                 and result.status in {"succeeded", "failed"}]
    searches = [result for result in live if (result.task == "battle" or result.metrics.get("receipt_kind") == "battle")
                and result.status in {"succeeded", "failed", "limited"}]
    resources: dict[str, dict[str, int | float | None]] = {}
    for resource in ("gold", "elixir", "dark_elixir"):
        values = {
            "battle_loot": _sum_metric(battles, f"loot_{resource}"),
            "battle_bonus": _sum_metric(battles, f"bonus_{resource}"),
            "collected_credited": _collection_credit(collections, resource),
            "donation_spend": _sum_metric(donations, f"donation_cost_{resource}", empty=0),
            "search_spend": _sum_metric(searches, f"search_cost_{resource}", empty=0),
        }
        gross = _sum_known(values["battle_loot"], values["battle_bonus"])
        net_in = _sum_known(gross, values["collected_credited"])
        spend = _sum_known(values["donation_spend"], values["search_spend"])
        net = net_in - spend if net_in is not None and spend is not None else None
        values.update({
            "battle_gross": gross,
            "operating_net": net,
            "battle_loot_per_hour": _hourly(values["battle_loot"], elapsed),
            "battle_gross_per_hour": _hourly(gross, elapsed),
            "operating_net_per_hour": _hourly(net, elapsed),
        })
        resources[resource] = values
    return {
        "elapsed_sec": elapsed,
        "resources": resources,
        "battle_gold_elixir_per_hour": _sum_known(resources["gold"]["battle_gross_per_hour"], resources["elixir"]["battle_gross_per_hour"]),
        "net_gold_elixir_per_hour": _sum_known(resources["gold"]["operating_net_per_hour"], resources["elixir"]["operating_net_per_hour"]),
    }


def _sum_metric(results: list[TaskResult], key: str, *, empty: int | None = None) -> int | float | None:
    return _sum_amounts([result.metrics.get(key) for result in results], empty=empty)


def _collection_credit(results: list[TaskResult], resource: str) -> int | float | None:
    amounts: list[object] = []
    for result in results:
        if result.status == "succeeded":
            amounts.append(result.metrics.get(f"collected_{resource}"))
            continue
        events = result.metrics.get("verified_collections")
        if not isinstance(events, list) or any(
            not isinstance(event, dict) or event.get("resource") not in {"gold", "elixir", "dark_elixir"}
            for event in events
        ):
            amounts.append(None)
            continue
        # A failed task may have later unverified changes; only its confirmed events count.
        amounts.append(_sum_amounts([event.get("increase") for event in events if event["resource"] == resource], empty=0))
    return _sum_amounts(amounts)


def _sum_amounts(values: list[object], *, empty: int | None = None) -> int | float | None:
    if not values:
        return empty
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
        return None
    return sum(values)


def _sum_known(*values: int | float | None) -> int | float | None:
    return sum(values) if all(value is not None for value in values) else None


def _hourly(value: int | float | None, elapsed_sec: float) -> float | None:
    return value * 3600 / elapsed_sec if value is not None and elapsed_sec > 0 else None


def _format_number(value: int | float | None) -> str:
    return "unknown" if value is None else f"{value:,.2f}"


def _format_state(value: bool | None) -> str:
    return "unknown" if value is None else "yes" if value else "no"


def _json_value(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unable to serialize report value {type(value).__name__}")


def _format_resolution(value: tuple[int, int] | None) -> str:
    if value is None:
        return "not set"
    return f"{value[0]}x{value[1]}"
