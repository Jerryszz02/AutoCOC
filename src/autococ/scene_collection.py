"""Candidate scene captures on one fixed, free, read-only navigation route.

The only permitted inputs are attack, edit_army, three Back presses and the
version-bound clan-chat HUD. No sample is labelled or admitted to a split here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
from threading import Event, Thread
import time
from typing import Callable

from .home_evidence import home_navigation_verified


VERSION = "18.647.2"
VERSION_CODE = 180647003
PACKAGE = "com.supercell.clashofclans"
BASE = (1280, 720)
CHAT_TEMPLATE = "hud_chat_18_647_2.png"
CHAT_ROI = (35, 330, 115, 410)
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "src/autococ/scene_collection.py", "scripts/collect_scene_samples.py",
    "src/autococ/session.py", "src/autococ/scene.py", "src/autococ/vision.py",
    "src/autococ/home_evidence.py",
    "src/autococ/picker_collection.py", "src/autococ/social.py",
    "src/autococ/locator.py", "src/autococ/device.py", "src/autococ/config.py",
)
RESOURCE_NAMES = ("gold", "elixir", "dark_elixir", "gems")


class SceneCollectionStopped(RuntimeError):
    """A gate failed; no navigation retry is allowed."""


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   encoding="utf-8").strip()


def _loaded_source_paths(repo: Path) -> dict[str, str]:
    if Path(__file__).resolve() != (repo / SOURCE_FILES[0]).resolve():
        raise SceneCollectionStopped("collector_loaded_from_unbound_path")
    found = {}
    for relative in SOURCE_FILES:
        if relative.startswith("scripts/"):
            continue
        module_name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module = importlib.import_module(module_name)
        actual = Path(module.__file__).resolve()
        if actual != (repo / relative).resolve():
            raise SceneCollectionStopped(f"module_loaded_from_unbound_path:{module_name}")
        found[module_name] = str(actual)
    return found


def _finite(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SceneCollectionStopped("capture_time_unverified")
    return float(value)


def _fresh(shot, previous=None, *, after_input: float | None = None) -> dict:
    obs = shot.observations
    when = _finite(obs.get("observed_at_monotonic"))
    requested = _finite(obs.get("requested_at_monotonic"))
    frame_id = obs.get("frame_id")
    path = Path(shot.screenshot_path).resolve()
    if not isinstance(frame_id, str) or not frame_id or not path.is_file():
        raise SceneCollectionStopped("frame_provenance_unverified")
    if previous is not None:
        old = previous.observations
        if (frame_id == old.get("frame_id") or path == Path(previous.screenshot_path).resolve()
                or when <= _finite(old.get("observed_at_monotonic"))
                or requested <= _finite(old.get("requested_at_monotonic"))):
            raise SceneCollectionStopped("frame_not_independent_or_fresh")
    if after_input is not None and requested <= after_input:
        raise SceneCollectionStopped("frame_precedes_input_completion")
    return {"path": str(path), "sha256": _sha(path), "frame_id": frame_id,
            "observed_at_monotonic": when, "requested_at_monotonic": requested,
            "scene": shot.scene, "confidence": shot.confidence}


def _scene(shot, expected: str) -> None:
    if shot.scene != expected or _finite(shot.confidence) < .8:
        raise SceneCollectionStopped(f"expected_{expected}_got_{shot.scene}")


def _unique_button(shot, name: str) -> None:
    buttons = shot.observations.get("buttons")
    if (not isinstance(buttons, list)
            or sum(isinstance(item, dict) and item.get("name") == name for item in buttons) != 1):
        raise SceneCollectionStopped(f"{name}_not_unique")


def _home(shot) -> dict[str, int]:
    _scene(shot, "village")
    obs = shot.observations
    if obs.get("village_type") != "home" or obs.get("resource_source") != "village_inventory":
        raise SceneCollectionStopped("home_identity_or_inventory_source_unverified")
    _unique_button(shot, "attack")
    if not home_navigation_verified(shot):
        raise SceneCollectionStopped("home_navigation_unverified")
    resources = obs.get("resources")
    if not isinstance(resources, dict) or any(type(resources.get(k)) is not int or resources[k] < 0
                                              for k in RESOURCE_NAMES):
        raise SceneCollectionStopped("home_inventory_incomplete")
    return {k: resources[k] for k in RESOURCE_NAMES}


def _army_state(shot) -> tuple:
    # The existing current-page recognizer requires a frame-bound complete
    # manifest, all four capacities and complete hero/siege identities.
    from .picker_collection import _army_counts, _roster_signature
    _scene(shot, "training")
    return _army_counts(shot), _roster_signature(shot)


def _unique_chat(shot, session) -> dict:
    """Prove one chat-HUD peak cluster in this very screenshot before click."""
    import cv2
    import numpy as np
    from .locator import Bounds, _clip_bounds

    session._validate_snapshot(shot)
    template_path = session.config.vision.template_dir / CHAT_TEMPLATE
    # Match the same BGR pixels/method as locator.find_template, so the
    # uniqueness proof cannot select a different peak from click_template.
    template = cv2.imdecode(np.fromfile(str(template_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.imdecode(np.fromfile(str(shot.screenshot_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or template is None:
        raise SceneCollectionStopped("chat_template_or_frame_unreadable")
    height, width = image.shape[:2]
    if (width, height) != tuple(session.context.screen_resolution):
        raise SceneCollectionStopped("chat_frame_context_resolution_mismatch")
    template = cv2.resize(
        template,
        (max(1, round(template.shape[1] * width / BASE[0])),
         max(1, round(template.shape[0] * height / BASE[1]))),
        interpolation=cv2.INTER_LINEAR,
    )
    bounds = _clip_bounds(Bounds(*CHAT_ROI).scale(BASE, (width, height)), width, height)
    x1, y1, x2, y2 = bounds.left, bounds.top, bounds.right, bounds.bottom
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        raise SceneCollectionStopped("chat_roi_or_template_invalid")
    scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(scores).all():
        raise SceneCollectionStopped("chat_match_nonfinite")
    good = (scores >= .88).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(good, connectivity=8)
    components = [stats[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 0]
    cluster_width = max(6, round(6 * width / BASE[0]))
    cluster_height = max(6, round(6 * height / BASE[1]))
    if (len(components) != 1 or components[0][cv2.CC_STAT_WIDTH] > cluster_width
            or components[0][cv2.CC_STAT_HEIGHT] > cluster_height):
        raise SceneCollectionStopped("chat_hud_not_unique")
    _, best, _, point = cv2.minMaxLoc(scores)
    return {"template": str(template_path), "template_sha256": _sha(template_path),
            "score": float(best), "peak": [x1 + point[0], y1 + point[1]],
            "point_at_1280x720": [
                int((x1 + point[0] + template.shape[1] // 2) * BASE[0] / width),
                int((y1 + point[1] + template.shape[0] // 2) * BASE[1] / height)],
            "cluster_area": int(components[0][cv2.CC_STAT_AREA]),
            "native_resolution": [width, height], "scaled_roi": [x1, y1, x2, y2],
            "scaled_template_size": [template.shape[1], template.shape[0]]}


def capture_route(session, *, foreground: Callable[[], bool], stop: Callable[[], bool],
                  open_army: Callable | None = None, open_chat: Callable | None = None,
                  chat_unique: Callable | None = None, army_state: Callable | None = None) -> dict:
    """One fixed route; six inputs maximum and zero mutations by construction."""
    from .errors import StopRequested

    if session.client_version != VERSION or session.client_version_code != VERSION_CODE:
        raise SceneCollectionStopped("client_version_unverified")
    if (tuple(session.config.game.baseline_resolution) != BASE
            or session.config.game.package_name != PACKAGE):
        raise SceneCollectionStopped("game_config_unverified")
    if open_army is None:
        from .picker_collection import _open_army_checked
        open_army = lambda s, home, evidence: _open_army_checked(
            s, home, evidence, foreground=foreground, stop=stop)
    chat_unique = chat_unique or _unique_chat
    army_state = army_state or _army_state
    start_actions = session.action_count
    result = {"kind": "read_only_scene_candidate_route_v1", "frames": [], "inputs": [],
              "army_comparison_scope": "two_independent_training_page_frames_only",
              "army_unchanged_while_on_training_page": False,
              "home_resources_and_gems_unchanged": False,
              "chat_hud_evidence": None}
    last = None

    def gate(expected_actions: int) -> None:
        session.check_deadline()
        if stop():
            raise StopRequested("interrupted by user")
        if session.action_count != start_actions + expected_actions or expected_actions > 6:
            raise SceneCollectionStopped("unexpected_input_count")
        if not foreground():
            raise SceneCollectionStopped("game_not_foreground")
        if hasattr(session, "deadline") and hasattr(session, "task_deadline"):
            remaining = min(_finite(session.deadline), _finite(session.task_deadline)) - time.monotonic()
            reserve = {0: 65, 2: 45, 3: 30, 4: 35, 5: 15, 6: 5}[expected_actions]
            if remaining < reserve:
                raise SceneCollectionStopped("insufficient_time_for_safe_return")

    def capture(label: str, scene: str, *, after_input: float | None = None):
        nonlocal last
        session.check_deadline()
        if stop():
            raise StopRequested("interrupted by user")
        shot = session.observe(label)
        result["frames"].append(_fresh(shot, last, after_input=after_input))
        _scene(shot, scene)
        last = shot
        return shot

    def returned(shot, scene: str, before, count: int,
                 after_input: float | None = None) -> None:
        nonlocal last
        evidence_frame = _fresh(shot, before, after_input=after_input)
        _scene(shot, scene)
        if session.action_count != start_actions + count:
            raise SceneCollectionStopped("unexpected_input_count")
        result["frames"].append(evidence_frame)
        last = shot

    gate(0)
    first = capture("scene-initial-home-1", "village")
    baseline = _home(first)
    second = capture("scene-initial-home-2", "village")
    if _home(second) != baseline:
        raise SceneCollectionStopped("initial_home_inventory_unstable")
    gate(0)
    evidence: list[Path] = []
    before = last
    training = open_army(session, second, evidence)
    returned(training, "training", before, 2)
    input_done = time.monotonic()
    result["inputs"].extend(["attack", "edit_army"])
    result["search_menu_frames"] = [str(path) for path in evidence]
    gate(2)
    army_first = capture("scene-army-1", "training", after_input=input_done)
    army_baseline = army_state(army_first)
    army_second = capture("scene-army-2", "training")
    if army_state(army_second) != army_baseline:
        raise SceneCollectionStopped("army_changed_during_training_page_capture")
    result["army_unchanged_while_on_training_page"] = True
    gate(2)
    before = last
    session.back(army_second, reason="scene collection: leave army without edits")
    input_done = time.monotonic()
    result["inputs"].append("back_from_training")
    search = session.wait_for({"search"}, timeout_sec=20, label="scene-army-to-search")
    returned(search, "search", before, 3, after_input=input_done)
    gate(3)
    before = last
    session.back(search, reason="scene collection: leave free search menu")
    input_done = time.monotonic()
    result["inputs"].append("back_from_search")
    home = session.wait_for({"village"}, timeout_sec=20, label="scene-search-to-home")
    returned(home, "village", before, 4, after_input=input_done)
    if _home(home) != baseline:
        raise SceneCollectionStopped("resources_changed_after_army_visit")
    gate(4)
    result["chat_hud_evidence"] = chat_unique(home, session)
    before = last
    if open_chat is None:
        # Reuse this frame's unique match. The general social helper would
        # repeat image matching after the foreground check and delay the tap.
        point = result["chat_hud_evidence"]["point_at_1280x720"]
        gate(4)
        session.tap(home, point, reason="scene collection: open uniquely matched chat HUD")
        chat = session.wait_for({"clan_chat"}, timeout_sec=30, label="scene-open-chat")
        evidence.append(chat.screenshot_path)
    else:
        gate(4)
        chat = open_chat(session, home, evidence)
    returned(chat, "clan_chat", before, 5)
    input_done = time.monotonic()
    result["inputs"].append("open_clan_chat_hud")
    gate(5)
    before = last
    session.back(chat, reason="scene collection: leave chat without sending")
    input_done = time.monotonic()
    result["inputs"].append("back_from_chat")
    home = session.wait_for({"village"}, timeout_sec=20, label="scene-chat-to-home")
    returned(home, "village", before, 6, after_input=input_done)
    if _home(home) != baseline:
        raise SceneCollectionStopped("resources_changed_after_chat_visit")
    gate(6)
    tail1 = capture("scene-final-home-1", "village", after_input=input_done)
    tail2 = capture("scene-final-home-2", "village")
    if _home(tail1) != baseline or _home(tail2) != baseline:
        raise SceneCollectionStopped("final_home_inventory_changed_or_unstable")
    result["home_resources_and_gems_unchanged"] = True
    result["action_count"] = session.action_count - start_actions
    result["resources_before"] = baseline
    result["resources_after"] = _home(tail2)
    return result


def collect_scene_samples(config_path: Path, output: Path, *, split: str, session_id: str,
                          group_id: str, limit_sec: int = 180,
                          stop: Callable[[], bool] = lambda: False,
                          repo_root: Path | None = None) -> dict:
    """Reserve split/group before any screenshot and save immutable raw evidence."""
    repo = (repo_root or REPO_ROOT).resolve()
    config_path, output = config_path.resolve(), output.resolve()
    if (split not in {"TRAIN", "VAL"} or not session_id or not group_id
            or not 90 <= limit_sec <= 240 or output.exists()
            or output.is_relative_to(repo) or not config_path.is_file()):
        raise ValueError("Invalid split, identity, budget, config, or output")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--porcelain") or _git(repo, "rev-parse", "@{u}") != head:
        raise SceneCollectionStopped("source_must_be_clean_and_pushed")
    source_before = {name: _sha(repo / name) for name in SOURCE_FILES}
    config_before = _sha(config_path)
    if stop():
        from .errors import StopRequested
        raise StopRequested("interrupted by user")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic reservation forbids group reuse, even if an earlier run failed.
    registry = output.parent / ".scene_collection_groups"
    registry.mkdir(exist_ok=True)
    group_token = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    group_marker = registry / f"{group_token}.json"
    with group_marker.open("x", encoding="utf-8") as stream:
        json.dump({"group_id": group_id, "split": split, "session_id": session_id,
                   "intended_output": str(output)}, stream, ensure_ascii=False)
    session_registry = output.parent / ".scene_collection_sessions"
    session_registry.mkdir(exist_ok=True)
    session_token = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    with (session_registry / f"{session_token}.json").open("x", encoding="utf-8") as stream:
        json.dump({"session_id": session_id, "group_id": group_id, "split": split,
                   "intended_output": str(output)}, stream, ensure_ascii=False)
    output.mkdir(exist_ok=False)
    prospective = {"kind": "scene_candidate_prospective_v1", "split": split,
                   "session_id": session_id, "candidate_leakage_group_id": group_id,
                   "leakage_independence_verified": False, "labels": None,
                   "admitted_to_training": False, "human_reviewed": False,
                   "formal_acceptance": False, "registered_before_screenshot": True,
                   "head_before": head, "source_sha256_before": source_before,
                   "config_path": str(config_path), "config_sha256_before": config_before}
    with (output / "prospective.json").open("x", encoding="utf-8") as stream:
        json.dump(prospective, stream, ensure_ascii=False, indent=2)
    result = {**prospective, "status": "failed", "reason": "not_started", "route": None,
              "action_count": 0, "device_serial": None, "client_version": None,
              "source_module_paths": None, "chat_template_sha256_before": None,
              "recovery": {"attempted": False,
                           "reason": "Any uncertain state requires a separate reviewed recovery; never reclassify this attempt"}}
    session = None
    started = time.monotonic()
    stop_event = Event()
    monitor_done = Event()
    def monitor_stop() -> None:
        while not monitor_done.wait(.1):
            try:
                if stop():
                    stop_event.set()
                    return
            except Exception:
                stop_event.set()
                return
    monitor = Thread(target=monitor_stop, name="scene-collection-stop", daemon=True)
    monitor.start()
    try:
        from .config import load_config
        from .device import DeviceManager, foreground_package
        from .session import GameSession

        config = load_config(config_path)
        if (tuple(config.game.baseline_resolution) != BASE or config.game.package_name != PACKAGE):
            raise SceneCollectionStopped("config_resolution_or_package_unverified")
        if config.runtime.dry_run is not False:
            raise SceneCollectionStopped("dry_run_cannot_produce_live_scene_candidates")
        result["source_module_paths"] = _loaded_source_paths(repo)
        template_path = (config.vision.template_dir / CHAT_TEMPLATE).resolve()
        if (template_path != (repo / "assets" / "templates" / CHAT_TEMPLATE).resolve()
                or not template_path.is_file()):
            raise SceneCollectionStopped("chat_template_not_bound_to_loaded_checkout")
        result["chat_template_path"] = str(template_path)
        result["chat_template_sha256_before"] = _sha(template_path)
        manager = DeviceManager(config)
        device = manager.connect()
        result["device_serial"] = device.serial
        def foreground() -> bool:
            return foreground_package(manager.adb, device.serial,
                                      timeout_sec=config.runtime.step_timeout_sec) == PACKAGE
        if not foreground() or stop():
            raise SceneCollectionStopped("stopped_or_not_foreground")
        session = GameSession.connect(config, manager.adb, device.serial, output,
                                      launch=False, stop_event=stop_event)
        result["client_version"] = session.client_version
        result["client_version_code"] = session.client_version_code
        if session.client_version != VERSION or session.client_version_code != VERSION_CODE:
            raise SceneCollectionStopped("client_version_unverified")
        session.begin_task()
        session.deadline = min(session.deadline, started + limit_sec)
        session.task_deadline = min(session.task_deadline, session.deadline)
        result["route"] = capture_route(session, foreground=foreground, stop=stop)
        if session.action_count != 6:
            raise SceneCollectionStopped("unexpected_total_input_count")
        result["status"], result["reason"] = "candidate_captured", "fixed_read_only_route_completed"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor_done.set()
        monitor.join(timeout=1)
        if session is not None:
            result["action_count"] = session.action_count
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
        if result["status"] == "candidate_captured":
            raw_by_path = {Path(item["path"]).resolve(): item["sha256"]
                           for item in result["raw_frames"]}
            expected = result["route"]["frames"]
            menu = result["route"].get("search_menu_frames", [])
            if (not result["events_sha256"]
                    or any(raw_by_path.get(Path(item["path"]).resolve()) != item["sha256"]
                           for item in expected)
                    or any(Path(path).resolve() not in raw_by_path for path in menu)):
                result["status"], result["reason"] = "failed", "required_raw_evidence_missing_or_changed"
        result["head_after"] = _git(repo, "rev-parse", "HEAD")
        result["source_status_after"] = _git(repo, "status", "--porcelain")
        result["source_sha256_after"] = {name: _sha(repo / name) for name in SOURCE_FILES}
        result["config_sha256_after"] = _sha(config_path)
        result["chat_template_sha256_after"] = (_sha(Path(result["chat_template_path"]))
                                                 if result.get("chat_template_path") else None)
        if (result["head_after"] != head or result["source_status_after"]
                or result["source_sha256_after"] != source_before
                or result["config_sha256_after"] != config_before
                or result["chat_template_sha256_after"] != result["chat_template_sha256_before"]):
            result["status"], result["reason"] = "failed", "source_or_config_changed_during_capture"
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return result
