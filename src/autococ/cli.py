"""Command-line entry point."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import tomllib
from uuid import uuid4

from .capture import CaptureClient
from .config import load_config
from .device import DeviceManager, resolve_game_display
from .errors import AutoCOCError
from .flow import FlowRunner
from .mumu import MuMuClient
from .reporting import RunStats, setup_logging
from .scene import SceneSnapshot
from .strategies import STRATEGIES
from .vision import ScreenshotRecognizer
from .routine_config import routine_from_dict, BATTLE_KINDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autococ")
    parser.add_argument("command", nargs="?", choices=["check", "capture", "inspect", "run", "sample", "gui", "capabilities"], default="check")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--profile", default="core-loop", help="Profile name for the run command")
    parser.add_argument("--image", type=Path, help="Inspect an existing screenshot without connecting to ADB")
    parser.add_argument("--once", action="store_true", help="Run one complete profile cycle")
    parser.add_argument("--dry-run", action="store_true", help="Plan the run without connecting to or operating the game")
    parser.add_argument("--strategy", choices=tuple(STRATEGIES), help="Override battle strategy for run")
    parser.add_argument("--strategy-file", type=Path, help="Local TOML deployment strategy for run")
    parser.add_argument("--routine", type=Path, help="Daily tasks in a TOML or JSON file (run only)")
    args = parser.parse_args(argv)
    if args.image is not None and args.command != "inspect":
        parser.error("--image is only valid with inspect")
    if (args.once or args.dry_run) and args.command != "run":
        parser.error("--once and --dry-run are only valid with run")
    if args.strategy is not None and args.command != "run":
        parser.error("--strategy is only valid with run")
    if (args.routine or args.strategy_file) and args.command != "run":
        parser.error("--routine and --strategy-file are only valid with run")
    if args.strategy and args.strategy_file:
        parser.error("Choose --strategy or --strategy-file")
    if args.command == "gui":
        from .gui import main as gui_main
        return gui_main(["--config", args.config])
    if args.command == "sample":
        print("sample is deprecated; capturing one game screenshot. This is not a completed game task.", file=sys.stderr)
        args.command = "capture"

    try:
        config = load_config(args.config)
        if args.command == "capabilities":
            from .unit_catalog import coverage, template_manifest
            from .building_vision import building_coverage, building_manifest
            unit_samples = template_manifest()
            units = coverage()
            for unit_id, item in units.items():
                samples = [entry for entry in unit_samples["templates"] if entry.get("unit_id") == unit_id]
                item["sampled_surfaces"] = sorted({entry["surface"] for entry in samples
                                                  if isinstance(entry.get("surface"), str)})
                item["sampled_states"] = sorted({entry["state"] for entry in samples
                                                if isinstance(entry.get("state"), str)})
            print(json.dumps({"sample_client_version": unit_samples.get("client", "unknown"),
                              "sample_version_code": unit_samples.get("version_code"),
                              "building_sample_client_version": building_manifest().get("client_version", "unknown"),
                              "coverage_note": "Samples on disk do not imply full roster or live acceptance",
                              "units": units, "buildings": building_coverage(),
                                "donation": "live window calibration pending"}, ensure_ascii=False, indent=2))
            return 0
        if args.routine:
            path = args.routine.resolve()
            payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else tomllib.loads(path.read_text(encoding="utf-8"))
            routine = routine_from_dict(payload.get("routine", payload) if isinstance(payload, dict) else payload)
            # All routine file references are made absolute at import, retaining their own file base.
            tasks = []
            for task in routine.tasks:
                def absolute(value):
                    return str((path.parent / value).resolve()) if value and not Path(value).is_absolute() else value
                tasks.append(replace(task, strategy_file=absolute(task.strategy_file),
                                     goal=replace(task.goal, adapter_path=absolute(task.goal.adapter_path))))
            config = replace(config, routine=replace(routine, tasks=tuple(tasks)))
        if args.strategy is not None:
            config = replace(config, battle=replace(config.battle, strategy=args.strategy, strategy_file=""))
        if args.strategy_file:
            from .strategy_config import load_strategy
            load_strategy(args.strategy_file)
            config = replace(config, battle=replace(config.battle, strategy_file=str(args.strategy_file.resolve())))
        if config.routine and (args.strategy or args.strategy_file or args.once):
            config = replace(config, routine=replace(config.routine, tasks=tuple(
                replace(task, strategy=args.strategy or task.strategy,
                        strategy_file=str(args.strategy_file.resolve()) if args.strategy_file else "" if args.strategy else task.strategy_file,
                        max_battles=1 if args.once else task.max_battles) if task.kind in BATTLE_KINDS else task
                for task in config.routine.tasks)))
        if args.once:
            config = replace(config, stop=replace(config.stop, max_runs=1))
        if args.dry_run:
            config = replace(config, runtime=replace(config.runtime, dry_run=True))
        if config.routine and args.command == "run":
            from .daily import validate_routine_files
            validate_routine_files(config, config.routine)
        elif args.command == "run" and config.battle.strategy_file:
            from .strategy_config import load_strategy
            configured = Path(config.battle.strategy_file)
            load_strategy(configured if configured.is_absolute() else config.source_path.resolve().parent / configured)
        logger = setup_logging(config.runtime.log_level, config.runtime.report_dir)
        logger.info("Loaded config: %s", Path(args.config))
        logger.info("ADB path: %s", config.adb.adb_path)

        if args.command == "inspect" and args.image is not None:
            if not args.image.is_file():
                raise FileNotFoundError(f"Screenshot does not exist: {args.image}")
            snapshot = ScreenshotRecognizer(config.ocr, config.game.baseline_resolution).recognize(args.image)
            _print_snapshot(snapshot)
            return 0

        if args.command == "run" and config.runtime.dry_run:
            runner = FlowRunner(config, _DryRunADB(), "dry-run", logger=logger)  # type: ignore[arg-type]
            return _print_run(runner.run_routine(config.routine) if config.routine else runner.run_profile(args.profile))

        manager = DeviceManager(config, logger=logger)
        device = manager.connect()
        if args.command == "run":
            print(f"Connected device: {device.serial}")
            runner = FlowRunner(config, manager.adb, device.serial, logger=logger)
            stats = runner.run_routine(config.routine) if config.routine else runner.run_profile(args.profile)
            return _print_run(stats)

        target = resolve_game_display(manager.adb, device.serial, config.game.package_name,
                                      display_id=config.game.display_id, timeout_sec=config.runtime.step_timeout_sec)
        native = None
        try:
            if config.mumu is not None:
                native = MuMuClient(config.mumu.install_dir, config.mumu.instance_index, device.serial,
                                    config.game.package_name, target.logical_id,
                                    timeout_sec=config.runtime.step_timeout_sec)
            capture = CaptureClient(manager.adb, device.serial, config.runtime.step_timeout_sec,
                                    screenshot_display_id=target.physical_id, input_display_id=target.logical_id,
                                    native=native)
            stamp = f"{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid4().hex[:8]}"
            screenshot = capture.capture_screenshot_artifact(config.runtime.screenshot_dir / f"{args.command}-{stamp}.png")
        finally:
            if native is not None:
                native.close()
        if args.command == "inspect":
            snapshot = ScreenshotRecognizer(config.ocr, config.game.baseline_resolution).recognize(screenshot.path)
            _print_snapshot(snapshot, device_serial=device.serial,
                            logical_display_id=target.logical_id, physical_display_id=target.physical_id)
        else:
            print(f"Connected device: {device.serial}")
            print(f"Game package: {config.game.package_name}")
            print(f"Logical display: {target.logical_id}; physical display: {target.physical_id}")
            print(f"Screenshot: {screenshot.path}")
            print(f"Screenshot resolution: {screenshot.width}x{screenshot.height}")
            print(f"Screenshot elapsed: {screenshot.elapsed_sec:.3f}s")
        return 0
    except (AutoCOCError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130


def _print_snapshot(snapshot: SceneSnapshot, **metadata: object) -> None:
    print(json.dumps({"scene": snapshot.scene, "confidence": snapshot.confidence,
                      "screenshot_path": str(snapshot.screenshot_path),
                      "observations": snapshot.observations, **metadata},
                     ensure_ascii=False, indent=2, default=str, allow_nan=False))


def _print_run(stats: RunStats) -> int:
    print(f"Profile: {stats.profile}; mode: {stats.mode}; run: {stats.run_id}")
    print(f"Cycles: {stats.cycles}; attempts: {stats.attempts}; successes: {stats.successes}; "
          f"failures: {stats.failures}; skipped: {stats.skipped}; simulated: {stats.simulated}")
    print(f"Stop reason: {stats.stop_reason or 'not set'}")
    if stats.profile == "daily":
        print(f"Battles completed: {stats.battles_completed}; won: {stats.battles_won}; goals completed: {stats.goals_completed}")
    if "interrupted" in stats.stop_reason.lower():
        return 130
    if stats.failures or any(result.status in {"failed", "not_supported"} for result in stats.task_results):
        return 1
    if stats.mode == "dry-run":
        return 0
    return 0 if any(result.status in {"succeeded", "skipped"} for result in stats.task_results) else 1


class _DryRunADB:
    def run(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("dry-run ADB should not execute commands")

    def run_bytes(self, *args: object, **kwargs: object) -> bytes:
        raise RuntimeError("dry-run ADB should not execute commands")


if __name__ == "__main__":
    raise SystemExit(main())
