"""Bounded current-client evidence for the first building-tactics pilot.

Only the observed HOME -> search menu -> army page -> HOME navigation is used.
This does not choose an opponent, deploy units or change an army preset.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
from threading import Event, Thread
import time
from typing import Callable

from .errors import StopRequested
from .picker_collection import SamplingStopped, _army_counts, _roster_signature
from .scene_collection import _fresh, _home, _scene


SUPPORTED_CLIENT_VERSIONS = {
    "18.647.2": 180647003,
    "18.647.5": 180647006,
}
PACKAGE = "com.supercell.clashofclans"
BASE = (1280, 720)
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "src/autococ/pilot_preflight.py", "scripts/capture_pilot_baseline.py",
    "src/autococ/session.py", "src/autococ/scene_collection.py",
    "src/autococ/picker_collection.py", "src/autococ/army_navigation.py",
    "src/autococ/vision.py", "src/autococ/device.py",
    "src/autococ/config.py", "src/autococ/home_evidence.py",
    "src/autococ/scene.py", "src/autococ/locator.py",
    "src/autococ/capture.py", "src/autococ/adb.py",
    "src/autococ/army_manifest.py", "src/autococ/unit_catalog.py",
    "src/autococ/army_editor.py", "src/autococ/combat.py",
    "src/autococ/ocr.py", "src/autococ/images.py",
)
ASSET_DIR = REPO_ROOT / "assets" / "templates"
CATALOG_DIR = REPO_ROOT / "assets" / "catalogs"


class PilotPreflightStopped(RuntimeError):
    """A current-client evidence gate failed; do not retry a game input."""


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _asset_hashes() -> dict[str, str]:
    return {path.name: _sha(path) for path in sorted(ASSET_DIR.glob("*")) if path.is_file()}


def _catalog_hashes() -> dict[str, str]:
    return {str(path.relative_to(CATALOG_DIR)).replace("\\", "/"): _sha(path)
            for path in sorted(CATALOG_DIR.rglob("*")) if path.is_file()}


def _supported_version(version: str, code: int | None) -> bool:
    return (version in SUPPORTED_CLIENT_VERSIONS and type(code) is int
            and SUPPORTED_CLIENT_VERSIONS[version] == code)


def _bound_modules(repo: Path) -> dict[str, str]:
    if Path(__file__).resolve() != (repo / SOURCE_FILES[0]).resolve():
        raise PilotPreflightStopped("pilot_module_not_loaded_from_requested_checkout")
    paths = {}
    for relative in SOURCE_FILES:
        if relative.startswith("scripts/"):
            continue
        module_name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        loaded = Path(importlib.import_module(module_name).__file__).resolve()
        expected = (repo / relative).resolve()
        if loaded != expected:
            raise PilotPreflightStopped(f"module_not_loaded_from_requested_checkout:{module_name}")
        paths[module_name] = str(loaded)
    return paths


def _army(shot) -> dict:
    _scene(shot, "training")
    counts = _army_counts(shot)
    roster = _roster_signature(shot)
    return {"counts": [list(item) for item in counts],
            "roster_signature": [[name, list(items)] for name, items in roster]}


def _hero_animation_partial(shot, client_version: str) -> dict:
    """Accept only one below-threshold animated portrait as a retryable read."""
    from .combat import _verified_army_identities

    army = shot.observations.get("army")
    if not isinstance(army, dict):
        raise PilotPreflightStopped("hero_animation_partial_unverified")
    counts = _army_counts(shot)
    manifest = army.get("manifest")
    frame = str(shot.screenshot_path)
    resolution = shot.observations.get("source_resolution")
    editor = shot.observations.get("army_editor")
    if (not isinstance(manifest, dict) or manifest.get("frame") != frame
            or manifest.get("layout") != "my_army_rows_v1"
            or manifest.get("supported_layout") is not True
            or manifest.get("complete") is not True
            or manifest.get("identity_complete") is not True
            or manifest.get("unknowns") != []
            or manifest.get("complete_kinds") != {"troop": True, "spell": True}
            or manifest.get("source_resolution") != resolution
            or manifest.get("baseline_resolution") != [1280, 720]
            or shot.observations.get("baseline_resolution") != [1280, 720]
            or shot.observations.get("account_in_use") not in (None, False)
            or not isinstance(editor, dict) or editor.get("surface") != "current"
            or editor.get("ready") is not True
            or not isinstance(resolution, list) or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)):
        raise PilotPreflightStopped("hero_animation_partial_unverified")
    rows = []
    for name, kind in (("troops", "troop"), ("spells", "spell")):
        cards = manifest.get(name)
        used = army["troops" if kind == "troop" else "spells"]["used"]
        if not (cards == [] and used == 0) and not _verified_army_identities(cards, kind, client_version):
            raise PilotPreflightStopped("hero_animation_nonhero_roster_unverified")
        row = []
        for card in cards:
            box = card.get("bbox")
            count = card.get("count")
            if (type(count) is not int or count < 1 or not isinstance(box, list)
                    or len(box) != 4 or any(type(value) is not int for value in box)):
                raise PilotPreflightStopped("hero_animation_nonhero_roster_unverified")
            row.append((tuple(box), card["unit_id"], count))
        rows.append((name, tuple(row)))
    coverage = army.get("identity_coverage")
    heroes = army.get("hero_evidence")
    siege = army.get("siege_evidence")
    identities = army.get("identity_cards")
    if (not isinstance(coverage, dict) or coverage.get("hero") is not False
            or coverage.get("siege") is not True
            or not isinstance(heroes, dict) or heroes.get("frame") != frame
            or heroes.get("layout") != "my_army_large_heroes_v1"
            or heroes.get("complete") is not False
            or heroes.get("unknowns") != ["hero_identity_unavailable"]
            or not isinstance(siege, dict) or siege.get("frame") != frame
            or siege.get("complete") is not True or siege.get("unknowns") != []
            or not isinstance(heroes.get("cards"), list)
            or not isinstance(siege.get("cards"), list)
            or not isinstance(identities, list)
            or identities != heroes.get("cards") + siege.get("cards")):
        raise PilotPreflightStopped("hero_animation_partial_unverified")
    hero_cards, siege_cards = heroes["cards"], siege["cards"]
    if (not isinstance(hero_cards, list) or len(hero_cards) != army["heroes"]["used"]
            or len(hero_cards) != 4 or not ((siege_cards == [] and army["siege"]["used"] == 0)
                                           or _verified_army_identities(siege_cards, "siege", client_version))):
        raise PilotPreflightStopped("hero_animation_cards_unverified")
    slots, known, missing = [], {}, []
    for card in hero_cards:
        box = card.get("bbox") if isinstance(card, dict) else None
        identity = card.get("evidence", {}).get("identity") if isinstance(card, dict) else None
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) is not int for value in box)
                or card.get("kind") != "hero" or card.get("source") != "army"
                or card.get("count") != 1 or not isinstance(identity, dict)):
            raise PilotPreflightStopped("hero_animation_cards_unverified")
        slot = tuple(box)
        slots.append(slot)
        if card.get("unit_id") is None:
            if (identity.get("unit_id") is not None
                    or identity.get("reason") != "below_identity_threshold"
                    or identity.get("version_verified") is not True
                    or identity.get("template_client_version") != client_version):
                raise PilotPreflightStopped("hero_animation_reason_unverified")
            missing.append(slot)
        else:
            if not _verified_army_identities([card], "hero", client_version):
                raise PilotPreflightStopped("hero_animation_known_id_unverified")
            known[slot] = card["unit_id"]
    if len(set(slots)) != 4 or len(missing) != 1 or len(set(known.values())) != 3:
        raise PilotPreflightStopped("hero_animation_cards_unverified")
    siege_row = []
    for card in siege_cards:
        box = card.get("bbox")
        if (type(card.get("count")) is not int or card["count"] < 1
                or not isinstance(box, list) or len(box) != 4
                or any(type(value) is not int for value in box)):
            raise PilotPreflightStopped("hero_animation_siege_unverified")
        siege_row.append((tuple(box), card["unit_id"], card["count"]))
    if sum(card[2] for card in siege_row) != army["siege"]["used"]:
        raise PilotPreflightStopped("hero_animation_siege_unverified")
    return {"counts": counts, "rows": tuple(rows), "siege": tuple(siege_row),
            "slots": tuple(slots), "known": known}


def _hero_slot_mapping(reference: dict, current: dict) -> dict | None:
    """Map observed hero boxes to the first frame; never accumulate drift."""
    if not all(reference[key] == current[key] for key in ("counts", "rows", "siege")):
        return None
    anchors = reference["slots"]
    observed = current["slots"]
    if len(anchors) != len(observed) or len(set(observed)) != len(observed):
        return None
    mapping = {}
    for box in observed:
        matches = [anchor for anchor in anchors
                   if all(abs(a - b) <= 2 for a, b in zip(anchor, box))]
        if len(matches) != 1:
            return None
        mapping[box] = matches[0]
    if len(set(mapping.values())) != len(anchors):
        return None
    if any(mapping[box] != anchors[index] for index, box in enumerate(observed)):
        return None
    for box, unit_id in current["known"].items():
        anchor = mapping.get(box)
        if (anchor is None or reference["known"].get(anchor, unit_id) != unit_id
                or (unit_id in reference["known"].values()
                    and reference["known"].get(anchor) != unit_id)):
            return None
    return mapping


def _complete_army_structure(shot) -> dict:
    counts = _army_counts(shot)
    roster = _roster_signature(shot)
    identities = roster[2][1]
    return {"counts": counts, "rows": roster[:2],
            "siege": tuple((item[3], item[1], item[2]) for item in identities if item[0] == "siege"),
            "slots": tuple(item[3] for item in identities if item[0] == "hero"),
            "known": {item[3]: item[1] for item in identities if item[0] == "hero"}}


def capture_route(session, *, foreground: Callable[[], bool],
                  stop: Callable[[], bool] = lambda: False,
                  evidence: dict | None = None) -> dict:
    """Capture current HOME and complete army twice, then return to HOME."""
    started_actions = session.action_count
    result: dict = evidence if evidence is not None else {}
    result.update(frames=[], inputs=[], army=None, resources_before=None, resources_after=None,
                  camera_state="unverified", zoom="unverified",
                  army_unchanged=False, returned_home=False)
    last = None

    def gate(expected_actions: int, reserve_sec: float) -> None:
        session.check_deadline()
        if stop():
            raise StopRequested("interrupted by user")
        if not _supported_version(session.client_version, session.client_version_code):
            raise PilotPreflightStopped("client_version_changed_or_unverified")
        if not foreground():
            raise PilotPreflightStopped("game_not_foreground")
        if session.action_count != started_actions + expected_actions:
            raise PilotPreflightStopped("unexpected_input_count")
        if min(session.deadline, session.task_deadline) - time.monotonic() < reserve_sec:
            raise PilotPreflightStopped("insufficient_time_for_safe_return")

    def record(shot, expected_scene: str, *, after_input: float | None = None) -> None:
        nonlocal last
        result["frames"].append(_fresh(shot, last, after_input=after_input))
        _scene(shot, expected_scene)
        last = shot

    def input_attempt(name: str, shot, action: Callable[[], None]) -> float:
        entry = {"name": name, "source_frame": str(shot.screenshot_path),
                 "source_sha256": _sha(shot.screenshot_path),
                 "attempted_at_monotonic": time.monotonic(),
                 "completed_at_monotonic": None, "state": "attempted"}
        result["inputs"].append(entry)
        action()
        entry["completed_at_monotonic"] = time.monotonic()
        entry["state"] = "completed"
        return entry["completed_at_monotonic"]

    gate(0, 65)
    home1 = session.observe("pilot-home-1")
    record(home1, "village")
    baseline = _home(home1)
    result["resources_before"] = baseline
    home2 = session.observe("pilot-home-2")
    record(home2, "village")
    if _home(home2) != baseline:
        raise PilotPreflightStopped("home_inventory_unstable")
    gate(0, 55)
    after_input = input_attempt("attack_menu", home2, lambda: session.click(home2, "attack"))
    menu = session.wait_for({"search"}, timeout_sec=20, label="pilot-army-menu")
    record(menu, "search", after_input=after_input)
    gate(1, 40)
    after_input = input_attempt("open_army", menu, lambda: session.click(menu, "edit_army"))
    training = session.wait_for({"training"}, timeout_sec=20, label="pilot-army-open")
    record(training, "training", after_input=after_input)
    gate(2, 40)
    observation_deadline = min(time.monotonic() + 30,
                               session.deadline - 30, session.task_deadline - 30)
    result["army_observations"] = []
    stable, first, reference, last_army = 0, None, None, None
    action_attempts = getattr(session, "_input_attempt_generation", None)
    if type(action_attempts) is not int:
        raise PilotPreflightStopped("army_observation_input_attempt_unverified")
    for attempt in range(6):
        gate(2, 30)
        if getattr(session, "_input_attempt_generation", None) != action_attempts:
            raise PilotPreflightStopped("army_observation_input_attempt_changed")
        if time.monotonic() >= observation_deadline:
            raise PilotPreflightStopped("army_observation_window_exhausted")
        shot = session.observe(f"pilot-army-{attempt + 1}")
        record(shot, "training")
        entry = {"frame": str(shot.screenshot_path), "state": "unverified", "reason": None}
        result["army_observations"].append(entry)
        gate(2, 30)
        if getattr(session, "_input_attempt_generation", None) != action_attempts:
            raise PilotPreflightStopped("army_observation_input_attempt_changed")
        if time.monotonic() > observation_deadline:
            raise PilotPreflightStopped("army_observation_window_exhausted")
        try:
            current = _army(shot)
        except SamplingStopped as exc:
            entry["reason"] = str(exc)
            if str(exc) != "current_hero_siege_coverage_incomplete":
                raise
            partial = _hero_animation_partial(shot, session.client_version)
            if reference is None:
                reference = partial
            else:
                mapping = _hero_slot_mapping(reference, partial)
                if mapping is None:
                    raise PilotPreflightStopped("army_partial_structure_changed")
                reference["known"].update({mapping[box]: unit_id
                                           for box, unit_id in partial["known"].items()})
            entry["state"] = "hero_animation_below_threshold"
            stable = 0
        else:
            complete = _complete_army_structure(shot)
            if reference is None:
                reference = complete
            else:
                mapping = _hero_slot_mapping(reference, complete)
                if mapping is None:
                    raise PilotPreflightStopped("army_readings_disagree")
                reference["known"].update({mapping[box]: unit_id
                                           for box, unit_id in complete["known"].items()})
            if first is None:
                first = current
            result["army"] = first
            last_army = shot
            stable += 1
            entry["state"] = "complete"
            if stable == 2:
                result["army_unchanged"] = True
                break
        if attempt < 5:
            time.sleep(min(.35, max(0, observation_deadline - time.monotonic())))
    if not result["army_unchanged"]:
        raise PilotPreflightStopped("army_observation_attempts_exhausted")
    gate(2, 30)
    after_input = input_attempt("back_from_army", last_army,
        lambda: session.back(last_army, reason="pilot baseline: leave army without edits"))
    menu2 = session.wait_for({"search"}, timeout_sec=15, label="pilot-back-menu")
    record(menu2, "search", after_input=after_input)
    gate(3, 12)
    after_input = input_attempt("back_from_menu", menu2,
        lambda: session.back(menu2, reason="pilot baseline: leave free menu"))
    home3 = session.wait_for({"village"}, timeout_sec=10, label="pilot-back-home")
    record(home3, "village", after_input=after_input)
    if _home(home3) != baseline:
        raise PilotPreflightStopped("home_inventory_changed")
    gate(4, 0)
    result["resources_after"] = _home(home3)
    result["returned_home"] = True
    return result


def capture_pilot_baseline(config_path: Path, output: Path, *, limit_sec: int = 150,
                           stop: Callable[[], bool] = lambda: False,
                           repo_root: Path | None = None) -> dict:
    """Save a source-bound development snapshot in a fresh directory."""
    repo = REPO_ROOT.resolve()
    if repo_root is not None and repo_root.resolve() != repo:
        raise ValueError("repo_root_must_match_loaded_checkout")
    raw_output = output
    config_path, output = config_path.resolve(), output.resolve()
    if (type(limit_sec) is not int or not 90 <= limit_sec <= 240
            or output.exists() or output.is_relative_to(repo) or raw_output.is_symlink()
            or not config_path.is_file()):
        raise ValueError("invalid_budget_config_or_output")
    started = time.monotonic()
    deadline = started + limit_sec
    modules = _bound_modules(repo)
    hashes_before = {name: _sha(repo / name) for name in SOURCE_FILES}
    assets_before = _asset_hashes()
    catalogs_before = _catalog_hashes()
    config_before = _sha(config_path)
    output.mkdir(parents=True, exist_ok=False)
    result: dict = {"kind": "batch_0_tactical_pilot_baseline_v1",
                    "status": "failed", "reason": "not_started",
                    "prototype_only": True, "counts_toward_acceptance": False,
                    "authorized_inputs": ["attack_menu", "open_army", "back_from_army", "back_from_menu"],
                    "economic_result": "pending_route_and_home_delta_check",
                    "repo_root": str(repo), "source_module_paths": modules,
                    "source_sha256_before": hashes_before,
                    "template_asset_sha256_before": assets_before,
                    "catalog_asset_sha256_before": catalogs_before,
                    "config_path": str(config_path), "config_sha256_before": config_before,
                    "device_serial": None, "client_version": None,
                    "client_version_code": None, "screen_resolution": None,
                    "action_count": 0, "route": None,
                    "recovery": "No automatic input after an uncertain state; inspect raw frames separately."}
    session = None
    stop_event, monitor_done = Event(), Event()

    def check_budget(stage: str, reserve_sec: float) -> None:
        if stop_event.is_set() or stop():
            raise StopRequested("interrupted by user")
        if deadline - time.monotonic() < reserve_sec:
            raise PilotPreflightStopped(f"budget_exhausted_before_{stage}")


    def watch_stop() -> None:
        while not monitor_done.wait(.1):
            try:
                if stop():
                    stop_event.set()
                    return
            except Exception:
                stop_event.set()
                return

    monitor = Thread(target=watch_stop, name="pilot-baseline-stop", daemon=True)
    monitor.start()
    try:
        from .config import load_config
        from .device import DeviceManager, foreground_package
        from .session import GameSession

        config = load_config(config_path)
        check_budget("device_config", 65)
        if (config.game.package_name != PACKAGE or tuple(config.game.baseline_resolution) != BASE
                or config.runtime.dry_run is not False
                or config.vision_agent.enabled is not False
                or config.vision_agent.jev_enabled is not False
                or config.vision.template_dir.resolve() != ASSET_DIR.resolve()):
            raise PilotPreflightStopped("config_package_resolution_or_runtime_unverified")
        if (not config.adb.manual_serial.strip() or config.adb.auto_scan_ports is not False
                or config.adb.connect_timeout_sec > 10 or config.runtime.step_timeout_sec > 10):
            raise PilotPreflightStopped("manual_device_or_bounded_timeouts_required")
        # Capture into the requested directory, never the user's standard report paths.
        config = replace(config, runtime=replace(config.runtime,
            screenshot_dir=output / "frames", report_dir=output),
            stop=replace(config.stop, max_duration_sec=min(config.stop.max_duration_sec, limit_sec)))
        check_budget("device_manager", 65)
        manager = DeviceManager(config)
        check_budget("device_connect", 65)
        device = manager.connect()
        result["device_serial"] = device.serial
        if device.serial != config.adb.manual_serial.strip():
            raise PilotPreflightStopped("connected_device_differs_from_manual_serial")
        check_budget("foreground_check", 55)

        def foreground() -> bool:
            return foreground_package(manager.adb, device.serial,
                                      timeout_sec=config.runtime.step_timeout_sec) == PACKAGE

        if not foreground():
            raise PilotPreflightStopped("stopped_or_game_not_foreground")
        check_budget("session_connect", 80)
        session = GameSession.connect(config, manager.adb, device.serial, output,
                                      launch=False, stop_event=stop_event)
        check_budget("version_metadata", 65)
        result["client_version"] = session.client_version
        result["client_version_code"] = session.client_version_code
        if not _supported_version(session.client_version, session.client_version_code):
            raise PilotPreflightStopped("client_version_unverified")
        session.begin_task()
        session.deadline = min(session.deadline, deadline)
        session.task_deadline = min(session.task_deadline, session.deadline)
        result["route"] = {}
        capture_route(session, foreground=foreground, stop=stop, evidence=result["route"])
        result["screen_resolution"] = list(session.context.screen_resolution)
        if session.action_count != 4:
            raise PilotPreflightStopped("unexpected_total_input_count")
        result["economic_result"] = "home_gold_unchanged_and_no_paid_search_input_in_route"
        result["status"], result["reason"] = "candidate_captured", "current_client_army_read_twice_and_returned_home"
    except KeyboardInterrupt as exc:
        result["status"] = "interrupted"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor_done.set()
        monitor.join(timeout=1)
        if session is not None:
            result["action_count"] = session.action_count
            result["screen_resolution"] = list(session.context.screen_resolution)
            try:
                session.close()
            except Exception as exc:
                result["status"] = "failed"
                result["reason"] = f"session_close_failed: {type(exc).__name__}: {exc}"
        result["elapsed_sec"] = round(time.monotonic() - started, 3)
        result["raw_frames"] = [{"path": str(p), "sha256": _sha(p)}
                                for p in sorted((output / "frames").glob("*.png"))]
        events = output / "events.jsonl"
        result["events_path"] = str(events)
        result["events_sha256"] = _sha(events) if events.is_file() else None
        result["source_sha256_after"] = {name: _sha(repo / name) for name in SOURCE_FILES}
        result["template_asset_sha256_after"] = _asset_hashes()
        result["catalog_asset_sha256_after"] = _catalog_hashes()
        result["config_sha256_after"] = _sha(config_path)
        if (result["source_sha256_after"] != hashes_before
                or result["template_asset_sha256_after"] != assets_before
                or result["catalog_asset_sha256_after"] != catalogs_before
                or result["config_sha256_after"] != config_before):
            result["status"], result["reason"] = "failed", "source_or_config_changed_during_capture"
        if result["status"] == "candidate_captured":
            hashes = {Path(item["path"]).resolve(): item["sha256"] for item in result["raw_frames"]}
            if (not result["events_sha256"] or
                    any(hashes.get(Path(item["path"]).resolve()) != item["sha256"]
                        for item in result["route"]["frames"])):
                result["status"], result["reason"] = "failed", "required_raw_evidence_missing_or_changed"
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return result
