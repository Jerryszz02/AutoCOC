"""Measure the configured live capture and recognition path without game input."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from autococ.config import load_config
from autococ.device import DeviceManager
from autococ.session import GameSession
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer
from autococ.deployment import _remaining
from autococ.images import _frame


def replay(directory: Path, output: Path, repeats: int) -> None:
    """Compare both paths on existing, receipt-labelled frames without a device."""
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    observations = {Path(row["frame"]).name: row for row in events if row.get("kind") == "observation"}
    deployments = [row for row in events if row.get("kind") == "line_deployment_consumed"]
    if not deployments:
        raise ValueError("Replay needs line_deployment_consumed events")
    first = Path(deployments[0]["before_frame"]).name
    reference_name = max(name for name, row in observations.items() if name < first
                         and (row["observations"].get("battle") or {}).get("slots"))
    reference = observations[reference_name]
    initial = SceneSnapshot(reference["scene"], reference["confidence"], directory / "frames" / reference_name,
                            reference["observations"])
    slots = initial.observations["battle"]["slots"]
    cases = []
    for event in deployments:
        slot = next(slot for slot in slots if slot["point"] == event["slot"])
        for name, count in (("before_frame", "before_count"), ("after_frame", "after_count")):
            cases.append((Path(event[name]).name, slot, event[count], "troop_count"))
    waits = sorted(name for name in observations if "battle-settlement" in name)
    for name in waits[::max(1, len(waits) // 4)]:
        cases.append((name, None, None, "settlement"))
    if waits and cases[-1][0] != waits[-1]:
        cases.append((waits[-1], None, None, "settlement"))
    started = perf_counter()
    reader = ScreenshotRecognizer()
    initialization = perf_counter() - started
    baseline = ScreenshotRecognizer(provider=reader.provider)
    rows = []
    for repeat in range(repeats):
        reader.battle_observer.anchors.clear()
        reader.battle_observer.cards.clear()
        previous = initial
        for name, slot, expected, purpose in cases:
            path = directory / "frames" / name
            _frame.cache_clear()
            started = perf_counter()
            full = baseline.recognize(path)
            full_count = _remaining(full, slot, recognizer=baseline) if slot else None
            full_seconds = perf_counter() - started
            # A live capture produces a new file. Do not let the preceding full
            # measurement give the fast path an already decoded target frame.
            _frame.cache_clear()
            started = perf_counter()
            fast = reader.recognize_battle(path, purpose=purpose, slot=slot, previous=previous)
            fast_count = _remaining(fast, slot, recognizer=reader) if slot else None
            fast_seconds = perf_counter() - started
            previous = fast
            row = {"repeat": repeat, "frame": name, "purpose": purpose, "expected_count": expected,
                   "full_count": full_count, "fast_count": fast_count, "full_scene": full.scene, "fast_scene": fast.scene,
                   "correct": full_count == fast_count == expected and full.scene == fast.scene,
                   "full_sec": full_seconds, "fast_sec": fast_seconds, "fallback": fast.observations["fast_fallback"],
                   "stages": fast.observations.get("recognition_timings_sec", {})}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    def summarize(samples):
        return {"samples": len(samples), "all_correct": all(row["correct"] for row in samples),
                "fallback_count": sum(row["fallback"] for row in samples),
                **{key: {"median": median(row[key] for row in samples),
                         "p95": sorted(row[key] for row in samples)[math.ceil(len(samples) * .95) - 1]}
                   for key in ("full_sec", "fast_sec")}}
    hot = [row for row in rows if row["repeat"] > 0] or rows
    result = {"source": str(directory), "fresh_frame_decode_per_path": True,
              "provider_initialization_sec": initialization,
              "first_pass": summarize([row for row in rows if row["repeat"] == 0]), "hot": summarize(hot),
              "hot_troop_count": summarize([row for row in hot if row["purpose"] == "troop_count"]), "frames": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--replay", type=Path, help="Existing run directory; never connects to a device")
    parser.add_argument("--output", type=Path, default=Path("reports/recognition-replay.json"))
    args = parser.parse_args()
    if not 1 <= args.frames <= 20:
        parser.error("--frames must be between 1 and 20")
    if args.replay:
        replay(args.replay, args.output, args.frames)
        return
    config = load_config(args.config)
    directory = config.runtime.report_dir / f"benchmark-{datetime.now():%Y%m%d-%H%M%S-%f}"
    manager = DeviceManager(config)
    device = manager.connect()
    session = GameSession.connect(config, manager.adb, device.serial, directory, launch=False)
    rows = []
    try:
        for index in range(args.frames):
            frame = session.observe("benchmark")
            row = {"index": index + 1, "scene": frame.scene, "frame": str(frame.screenshot_path)}
            row.update({key: frame.observations[key] for key in (
                "capture_method", "capture_elapsed_sec", "recognition_elapsed_sec", "observation_elapsed_sec")})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        session.close()
    result = {"frames": rows, "medians": {
        key: median(row[key] for row in rows) for key in (
            "capture_elapsed_sec", "recognition_elapsed_sec", "observation_elapsed_sec")}}
    path = directory / "summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
