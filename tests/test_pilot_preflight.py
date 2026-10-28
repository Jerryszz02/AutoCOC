"""The pilot snapshot must stop at uncertainty without searching or editing."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from autococ import pilot_preflight as pilot
from autococ.errors import StopRequested
from autococ.picker_collection import SamplingStopped


@dataclass
class Shot:
    scene: str
    screenshot_path: Path
    observations: dict
    confidence: float = 1.0


class Session:
    client_version = "18.647.2"
    client_version_code = 180647003

    def __init__(self, tmp_path: Path, *, army_values=("same", "same"),
                 budget_sec: float = 100):
        self.action_count = 0
        self.deadline = time.monotonic() + budget_sec
        self.task_deadline = self.deadline
        self.calls = []
        self._army_values = iter(army_values)
        self._input_attempt_generation = 0
        self._index = 0
        self.tmp_path = tmp_path

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise RuntimeError("deadline")

    def _shot(self, scene, label):
        self._index += 1
        path = self.tmp_path / f"{self._index:02d}-{label}.png"
        path.write_bytes(b"fake png bytes")
        now = time.monotonic()
        observations = {"frame_id": f"frame-{self._index}",
                        "observed_at_monotonic": now,
                        "requested_at_monotonic": now,
                        "army": next(self._army_values) if label.startswith("pilot-army-")
                        and label.removeprefix("pilot-army-").isdigit() else None}
        return Shot(scene, path, observations)

    def observe(self, label):
        scene = "training" if label.startswith("pilot-army-") else "village"
        self.calls.append(("observe", label))
        return self._shot(scene, label)

    def wait_for(self, scenes, *, timeout_sec, label):
        self.calls.append(("wait_for", label))
        return self._shot(next(iter(scenes)), label)

    def click(self, shot, name):
        self.calls.append(("click", name))
        assert (shot.scene, name) in {("village", "attack"), ("search", "edit_army")}
        self.action_count += 1
        self._input_attempt_generation += 1

    def back(self, shot, *, reason):
        self.calls.append(("back", shot.scene))
        assert shot.scene in {"training", "search"}
        self.action_count += 1
        self._input_attempt_generation += 1


@pytest.fixture(autouse=True)
def simple_proofs(monkeypatch):
    monkeypatch.setattr(pilot, "_home", lambda shot: {"gold": 100} if shot.scene == "village" else None)
    monkeypatch.setattr(pilot, "_army", lambda shot: {"signature": shot.observations["army"]})
    monkeypatch.setattr(pilot, "_complete_army_structure", lambda shot: {
        "counts": ("fixed",), "rows": ("fixed",), "siege": ("fixed",),
        "slots": ((0, 0, 10, 10),), "known": {(0, 0, 10, 10): "hero"}})


def test_route_reads_army_twice_and_returns_without_paid_or_edit_input(tmp_path):
    session = Session(tmp_path)
    result = pilot.capture_route(session, foreground=lambda: True)
    assert result["army_unchanged"] and result["returned_home"]
    assert [item["name"] for item in result["inputs"]] == [
        "attack_menu", "open_army", "back_from_army", "back_from_menu"]
    assert all(item["state"] == "completed" and item["completed_at_monotonic"]
               for item in result["inputs"])
    assert session.action_count == 4
    assert [name for action, name in session.calls if action == "click"] == ["attack", "edit_army"]
    assert len(result["frames"]) == 8


def test_route_stops_on_ambiguous_army_without_back_or_search(tmp_path, monkeypatch):
    original = pilot._complete_army_structure
    def changed(shot):
        structure = original(shot)
        if shot.observations["army"] == "different":
            structure["known"] = {(0, 0, 10, 10): "different_hero"}
        return structure
    monkeypatch.setattr(pilot, "_complete_army_structure", changed)
    session = Session(tmp_path, army_values=("first", "different"))
    partial = {}
    with pytest.raises(pilot.PilotPreflightStopped, match="army_readings_disagree"):
        pilot.capture_route(session, foreground=lambda: True, evidence=partial)
    assert session.action_count == 2
    assert len(partial["frames"]) == 6
    assert len(partial["inputs"]) == 2
    assert partial["army"] == {"signature": "first"}
    assert not any(action == "back" for action, _ in session.calls)
    assert not any(name == "search_opponent" for _, name in session.calls)


def _animation_setup(monkeypatch, *, partial_structure=None):
    base = {"counts": ("fixed",), "rows": ("fixed",), "siege": ("fixed",),
            "slots": ((0, 0, 10, 10),), "known": {(0, 0, 10, 10): "hero"}}
    monkeypatch.setattr(pilot, "_hero_animation_partial",
                        lambda shot, version: dict(partial_structure or base))
    def read(shot):
        value = shot.observations["army"]
        if value.startswith("lost"):
            raise SamplingStopped("current_hero_siege_coverage_incomplete")
        if value == "bad-roster":
            raise SamplingStopped("current_roster_manifest_incomplete")
        return {"signature": value}
    monkeypatch.setattr(pilot, "_army", read)


def test_animation_gap_requires_two_consecutive_complete_reads(tmp_path, monkeypatch):
    _animation_setup(monkeypatch)
    session = Session(tmp_path, army_values=("lost", "same", "same"))
    result = pilot.capture_route(session, foreground=lambda: True)
    assert result["returned_home"]
    assert [item["state"] for item in result["army_observations"]] == [
        "hero_animation_below_threshold", "complete", "complete"]
    assert result["army_observations"][0]["reason"] == "current_hero_siege_coverage_incomplete"
    assert session.action_count == 4


def test_animation_gap_between_complete_reads_resets_candidate(tmp_path, monkeypatch):
    _animation_setup(monkeypatch)
    session = Session(tmp_path, army_values=("same", "lost", "same", "same"))
    result = pilot.capture_route(session, foreground=lambda: True)
    assert [item["state"] for item in result["army_observations"]] == [
        "complete", "hero_animation_below_threshold", "complete", "complete"]


def test_animation_retries_have_frame_and_time_limits(tmp_path, monkeypatch):
    _animation_setup(monkeypatch)
    session = Session(tmp_path, army_values=("lost",) * 6)
    partial = {}
    with pytest.raises(pilot.PilotPreflightStopped, match="attempts_exhausted"):
        pilot.capture_route(session, foreground=lambda: True, evidence=partial)
    assert len(partial["army_observations"]) == 6
    assert session.action_count == 2

    class Clock:
        now = 0.0
        def monotonic(self):
            return self.now
        def sleep(self, duration):
            self.now += 31
    monkeypatch.setattr(pilot, "time", Clock())
    timed = Session(tmp_path, army_values=("lost", "lost"))
    timed.deadline = timed.task_deadline = 150
    timed.check_deadline = lambda: None
    with pytest.raises(pilot.PilotPreflightStopped, match="window_exhausted"):
        pilot.capture_route(timed, foreground=lambda: True)
    assert timed.action_count == 2


@pytest.mark.parametrize("values,error", [
    (("lost", "bad-roster"), "current_roster_manifest_incomplete"),
    (("lost", "different"), "army_readings_disagree"),
])
def test_animation_does_not_hide_other_roster_errors(tmp_path, monkeypatch, values, error):
    _animation_setup(monkeypatch)
    if values[1] == "different":
        original = pilot._complete_army_structure
        def changed(shot):
            structure = original(shot)
            structure["known"] = {(0, 0, 10, 10): "different_hero"}
            return structure
        monkeypatch.setattr(pilot, "_complete_army_structure", changed)
    session = Session(tmp_path, army_values=values)
    with pytest.raises((SamplingStopped, pilot.PilotPreflightStopped), match=error):
        pilot.capture_route(session, foreground=lambda: True)
    assert session.action_count == 2


@pytest.mark.parametrize("change,error", [
    ("stop", "interrupted by user"),
    ("foreground", "game_not_foreground"),
    ("input", "army_observation_input_attempt_changed"),
    ("action", "unexpected_input_count"),
    ("version", "client_version_changed_or_unverified"),
])
def test_animation_wait_stops_on_environment_change(tmp_path, monkeypatch, change, error):
    _animation_setup(monkeypatch)
    session = Session(tmp_path, army_values=("lost", "same", "same"))
    state = {"after_first_army": False}
    original = session.observe
    def observed(label):
        shot = original(label)
        if label == "pilot-army-1":
            state["after_first_army"] = True
            if change == "input":
                session._input_attempt_generation += 1
            if change == "action":
                session.action_count += 1
            if change == "version":
                session.client_version_code += 1
        return shot
    session.observe = observed
    stop = lambda: change == "stop" and state["after_first_army"]
    foreground = lambda: not (change == "foreground" and state["after_first_army"])
    with pytest.raises((StopRequested, pilot.PilotPreflightStopped), match=error):
        pilot.capture_route(session, stop=stop, foreground=foreground)
    assert session.action_count == (3 if change == "action" else 2)


def _partial_hero_shot(tmp_path):
    frame = tmp_path / "partial.png"
    version = "18.647.5"
    def identity(unit_id, *, reason="portrait_template"):
        return {"unit_id": unit_id, "reason": reason, "version_verified": True,
                "template_client_version": version, "confidence": .95}
    def card(unit_id, kind, box, *, reason="portrait_template"):
        return {"unit_id": unit_id, "kind": kind, "source": "army", "count": 1,
                "bbox": box, "evidence": {"identity": identity(unit_id, reason=reason)}}
    troops = [card("archer", "troop", [10, 10, 30, 30])]
    spells = [card("rage", "spell", [40, 10, 60, 30])]
    heroes = [card("archer_queen", "hero", [i * 100, 100, i * 100 + 90, 600])
              for i in range(4)]
    heroes[0]["unit_id"] = "grand_warden"
    heroes[0]["evidence"]["identity"] = identity("grand_warden")
    heroes[1]["unit_id"] = "royal_champion"
    heroes[1]["evidence"]["identity"] = identity("royal_champion")
    heroes[2]["unit_id"] = "minion_prince"
    heroes[2]["evidence"]["identity"] = identity("minion_prince")
    heroes[3]["unit_id"] = None
    heroes[3]["evidence"]["identity"] = identity(None, reason="below_identity_threshold")
    sieges = [card("battle_blimp", "siege", [500, 100, 590, 200])]
    army = {
        "heroes": {"used": 4, "capacity": 4}, "troops": {"used": 1, "capacity": 320},
        "spells": {"used": 1, "capacity": 11}, "siege": {"used": 1, "capacity": 3},
        "manifest": {"frame": str(frame), "layout": "my_army_rows_v1",
                     "supported_layout": True, "complete": True, "identity_complete": True,
                     "unknowns": [], "complete_kinds": {"troop": True, "spell": True},
                     "source_resolution": [2560, 1440], "baseline_resolution": [1280, 720],
                     "troops": troops, "spells": spells},
        "identity_coverage": {"hero": False, "siege": True},
        "hero_evidence": {"frame": str(frame), "layout": "my_army_large_heroes_v1",
                          "complete": False, "unknowns": ["hero_identity_unavailable"],
                          "cards": heroes},
        "siege_evidence": {"frame": str(frame), "complete": True,
                           "unknowns": [], "cards": sieges},
        "identity_cards": heroes + sieges,
    }
    return Shot("training", frame, {"army": army, "source_resolution": [2560, 1440],
                                    "baseline_resolution": [1280, 720],
                                    "army_editor": {"surface": "current", "ready": True}})


def test_hero_animation_partial_accepts_only_single_below_threshold_gap(tmp_path):
    shot = _partial_hero_shot(tmp_path)
    partial = pilot._hero_animation_partial(shot, "18.647.5")
    assert len(partial["known"]) == 3
    assert partial["counts"][0] == ("heroes", 4, 4)
    bad = deepcopy(shot)
    bad.observations["army"]["hero_evidence"]["cards"][3]["evidence"]["identity"]["reason"] = "ambiguous_identity"
    with pytest.raises(pilot.PilotPreflightStopped, match="reason_unverified"):
        pilot._hero_animation_partial(bad, "18.647.5")
    bad = deepcopy(shot)
    bad.observations["army"]["siege_evidence"]["unknowns"] = ["siege_identity_unavailable"]
    with pytest.raises(pilot.PilotPreflightStopped, match="partial_unverified"):
        pilot._hero_animation_partial(bad, "18.647.5")
    bad = deepcopy(shot)
    bad.observations["army"]["manifest"]["troops"][0]["source"] = "foreign"
    with pytest.raises(pilot.PilotPreflightStopped, match="nonhero_roster_unverified"):
        pilot._hero_animation_partial(bad, "18.647.5")


def test_hero_box_mapping_is_bounded_against_first_frame():
    boxes = ((0, 10, 20, 40), (100, 10, 120, 40),
             (200, 10, 220, 40), (300, 10, 320, 40))
    reference = {"counts": ("same",), "rows": ("same",), "siege": ("same",),
                 "slots": boxes, "known": {boxes[0]: "grand_warden"}}
    shifted = deepcopy(reference)
    shifted["slots"] = ((2, 10, 22, 40),) + boxes[1:]
    shifted["known"] = {shifted["slots"][0]: "grand_warden"}
    assert pilot._hero_slot_mapping(reference, shifted)[shifted["slots"][0]] == boxes[0]
    shifted["slots"] = ((3, 10, 23, 40),) + boxes[1:]
    shifted["known"] = {shifted["slots"][0]: "grand_warden"}
    assert pilot._hero_slot_mapping(reference, shifted) is None
    # A later frame cannot drift another two pixels from the preceding frame.
    shifted["slots"] = ((4, 10, 24, 40),) + boxes[1:]
    shifted["known"] = {shifted["slots"][0]: "grand_warden"}
    assert pilot._hero_slot_mapping(reference, shifted) is None
    changed = deepcopy(reference)
    changed["known"][boxes[0]] = "royal_champion"
    assert pilot._hero_slot_mapping(reference, changed) is None
    changed = deepcopy(reference)
    changed["rows"] = ("changed",)
    assert pilot._hero_slot_mapping(reference, changed) is None
    crossed = deepcopy(reference)
    crossed["slots"] = (boxes[1], boxes[0], boxes[2], boxes[3])
    crossed["known"] = {boxes[1]: "grand_warden"}
    assert pilot._hero_slot_mapping(reference, crossed) is None


def test_two_complete_frames_allow_only_hero_box_pixel_jitter(tmp_path, monkeypatch):
    session = Session(tmp_path, army_values=("first", "second"))
    boxes = ((0, 10, 20, 40), (100, 10, 120, 40),
             (200, 10, 220, 40), (300, 10, 320, 40))
    def structure(shot):
        shifted = shot.observations["army"] == "second"
        slots = (((2, 10, 22, 40),) + boxes[1:]) if shifted else boxes
        return {"counts": ("same",), "rows": ("same",), "siege": ("same",),
                "slots": slots, "known": {slots[0]: "grand_warden"}}
    monkeypatch.setattr(pilot, "_complete_army_structure", structure)
    result = pilot.capture_route(session, foreground=lambda: True)
    assert result["returned_home"]
    assert result["army"] == {"signature": "first"}
    assert [item["state"] for item in result["army_observations"]] == ["complete", "complete"]


def test_route_rejects_insufficient_return_budget_before_first_input(tmp_path):
    session = Session(tmp_path, budget_sec=64)
    with pytest.raises(pilot.PilotPreflightStopped, match="insufficient_time"):
        pilot.capture_route(session, foreground=lambda: True)
    assert session.action_count == 0


def test_route_rejects_stop_and_version_change_without_new_input(tmp_path):
    stopped = Session(tmp_path)
    with pytest.raises(StopRequested, match="interrupted by user"):
        pilot.capture_route(stopped, foreground=lambda: True, stop=lambda: True)
    assert stopped.action_count == 0
    changed = Session(tmp_path)
    changed.client_version = "18.647.3"
    with pytest.raises(pilot.PilotPreflightStopped, match="client_version_changed"):
        pilot.capture_route(changed, foreground=lambda: True)
    assert changed.action_count == 0


def test_version_pairs_are_exact_and_new_client_keeps_army_identity_gate(tmp_path):
    from autococ.picker_collection import _roster_signature
    from autococ.picker_collection import SamplingStopped
    assert pilot._supported_version("18.647.2", 180647003)
    assert pilot._supported_version("18.647.5", 180647006)
    assert not pilot._supported_version("18.647.5", 180647003)
    assert not pilot._supported_version("18.647.2", 180647006)
    assert not pilot._supported_version("18.647.6", 180647006)
    new_client = Session(tmp_path)
    new_client.client_version = "18.647.5"
    new_client.client_version_code = 180647006
    assert pilot.capture_route(new_client, foreground=lambda: True)["returned_home"]
    incomplete = Shot("training", tmp_path / "unverified.png", {"army": {"manifest": None}})
    with pytest.raises(SamplingStopped, match="current_roster_manifest_incomplete"):
        _roster_signature(incomplete)


@pytest.mark.parametrize("version,code", [
    ("unknown", None), ("18.647.6", None), ("18.647.5", None),
    ("18.647.5", 180647006.0),
])
def test_route_rejects_missing_or_untyped_version_before_observation(tmp_path, version, code):
    session = Session(tmp_path)
    session.client_version = version
    session.client_version_code = code
    with pytest.raises(pilot.PilotPreflightStopped, match="client_version_changed_or_unverified"):
        pilot.capture_route(session, foreground=lambda: True)
    assert session.calls == []
    assert session.action_count == 0


def test_route_stops_on_navigation_failure_without_further_input(tmp_path):
    session = Session(tmp_path)
    original = session.wait_for

    def wrong_scene(scenes, *, timeout_sec, label):
        if label == "pilot-army-menu":
            return session._shot("unknown", label)
        return original(scenes, timeout_sec=timeout_sec, label=label)

    session.wait_for = wrong_scene
    with pytest.raises(Exception):
        pilot.capture_route(session, foreground=lambda: True)
    assert session.action_count == 1
    assert [name for action, name in session.calls if action == "click"] == ["attack"]


def test_input_attempt_and_stale_post_input_frame_are_preserved(tmp_path):
    session = Session(tmp_path)
    partial = {}
    original = session.wait_for

    def stale(scenes, *, timeout_sec, label):
        shot = original(scenes, timeout_sec=timeout_sec, label=label)
        if label == "pilot-army-menu":
            shot.observations["requested_at_monotonic"] -= 5
        return shot

    session.wait_for = stale
    with pytest.raises(Exception, match="frame_not_independent_or_fresh|frame_precedes_input_completion"):
        pilot.capture_route(session, foreground=lambda: True, evidence=partial)
    assert session.action_count == 1
    assert partial["inputs"][0]["state"] == "completed"
    assert len(partial["frames"]) == 2


def test_uncertain_input_retains_attempt_without_claiming_completion(tmp_path):
    session = Session(tmp_path)
    partial = {}

    def uncertain_click(shot, name):
        assert name == "attack"
        raise RuntimeError("transport uncertain")

    session.click = uncertain_click
    with pytest.raises(RuntimeError, match="transport uncertain"):
        pilot.capture_route(session, foreground=lambda: True, evidence=partial)
    assert session.action_count == 0
    assert len(partial["frames"]) == 2
    assert partial["inputs"][0]["state"] == "attempted"
    assert partial["inputs"][0]["completed_at_monotonic"] is None


def test_output_must_be_fresh_and_outside_checkout(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="invalid_budget"):
        pilot.capture_pilot_baseline(config, existing)
    with pytest.raises(ValueError, match="repo_root_must_match"):
        pilot.capture_pilot_baseline(config, tmp_path / "new", repo_root=tmp_path)


def test_loaded_source_must_match_requested_checkout(tmp_path, monkeypatch):
    from autococ import scene_collection
    original = scene_collection.__file__
    monkeypatch.setattr(scene_collection, "__file__", str(tmp_path / "foreign.py"))
    with pytest.raises(pilot.PilotPreflightStopped, match="module_not_loaded_from_requested_checkout"):
        pilot._bound_modules(pilot.REPO_ROOT)
    monkeypatch.setattr(scene_collection, "__file__", original)


def test_current_client_version_is_required_before_any_capture(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text("pilot config", encoding="utf-8")
    output = tmp_path / "result"
    from autococ.config import AdbConfig, GameConfig, RuntimeConfig, StopConfig

    @dataclass
    class Config:
        game: GameConfig
        runtime: RuntimeConfig
        vision_agent: object
        vision: object
        adb: AdbConfig
        stop: StopConfig

    fake_config = Config(
        game=GameConfig(package_name=pilot.PACKAGE, baseline_resolution=pilot.BASE),
        runtime=RuntimeConfig(dry_run=False, step_timeout_sec=1,
                              screenshot_dir=tmp_path, report_dir=tmp_path),
        vision_agent=SimpleNamespace(enabled=False, jev_enabled=False),
        vision=SimpleNamespace(template_dir=pilot.ASSET_DIR),
        adb=AdbConfig(adb_path=tmp_path, manual_serial="device", auto_scan_ports=False),
        stop=StopConfig())
    fake_session = SimpleNamespace(client_version="other", client_version_code=1,
                                   action_count=0, context=SimpleNamespace(screen_resolution=(1280, 720)),
                                   close=lambda: None)
    from autococ import config, device, session
    monkeypatch.setattr(config, "load_config", lambda _: fake_config)
    monkeypatch.setattr(device, "DeviceManager", lambda _: SimpleNamespace(
        adb=object(), connect=lambda: SimpleNamespace(serial="device")))
    monkeypatch.setattr(device, "foreground_package", lambda *args, **kwargs: pilot.PACKAGE)
    monkeypatch.setattr(session.GameSession, "connect", lambda *args, **kwargs: fake_session)
    monkeypatch.setattr(pilot, "capture_route", lambda *args, **kwargs: pytest.fail("captured on wrong client"))
    result = pilot.capture_pilot_baseline(config_path, output)
    assert result["status"] == "failed"
    assert "client_version_unverified" in result["reason"]
    assert result["action_count"] == 0
    assert (output / "summary.json").is_file()


def test_interrupt_persists_summary_and_cli_returns_130(tmp_path, monkeypatch, capsys):
    import json
    from autococ import config, device, session
    from autococ.config import AdbConfig, AppConfig, GameConfig, RuntimeConfig, StopConfig, VisionConfig
    config_path = tmp_path / "config.toml"
    config_path.write_text("pilot config", encoding="utf-8")
    output = tmp_path / "interrupted"

    # Use a real dataclass config so the production replacements run unchanged.
    app = object.__new__(AppConfig)
    from dataclasses import fields
    values = {field.name: None for field in fields(AppConfig)}
    values.update(adb=AdbConfig(adb_path=tmp_path, manual_serial="device", auto_scan_ports=False),
                  game=GameConfig(package_name=pilot.PACKAGE, baseline_resolution=pilot.BASE),
                  runtime=RuntimeConfig(dry_run=False, step_timeout_sec=1),
                  stop=StopConfig(), vision=VisionConfig(template_dir=pilot.ASSET_DIR),
                  vision_agent=SimpleNamespace(enabled=False, jev_enabled=False))
    app = AppConfig(**values)
    monkeypatch.setattr(config, "load_config", lambda _: app)
    monkeypatch.setattr(device, "DeviceManager", lambda _: SimpleNamespace(
        adb=object(), connect=lambda: SimpleNamespace(serial="device")))
    monkeypatch.setattr(device, "foreground_package", lambda *args, **kwargs: pilot.PACKAGE)
    monkeypatch.setattr(session.GameSession, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    from scripts.capture_pilot_baseline import main
    assert main(["--config", str(config_path), "--output", str(output)]) == 130
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["reason"].startswith("KeyboardInterrupt:")
    assert report["action_count"] == 0
    assert report["source_sha256_before"] == report["source_sha256_after"]
    assert report["config_sha256_before"] == report["config_sha256_after"]
    assert json.loads(capsys.readouterr().out)["status"] == "interrupted"


def test_auto_scan_is_rejected_before_device_manager(tmp_path, monkeypatch):
    import json
    from autococ import config, device
    from autococ.config import AdbConfig, GameConfig, RuntimeConfig, StopConfig

    @dataclass
    class Config:
        adb: AdbConfig
        game: GameConfig
        runtime: RuntimeConfig
        stop: StopConfig
        vision: object
        vision_agent: object

    app = Config(
        adb=AdbConfig(adb_path=tmp_path, manual_serial="device", auto_scan_ports=True),
        game=GameConfig(package_name=pilot.PACKAGE, baseline_resolution=pilot.BASE),
        runtime=RuntimeConfig(dry_run=False, step_timeout_sec=1), stop=StopConfig(),
        vision=SimpleNamespace(template_dir=pilot.ASSET_DIR),
        vision_agent=SimpleNamespace(enabled=False, jev_enabled=False))
    config_path = tmp_path / "config.toml"
    config_path.write_text("pilot config", encoding="utf-8")
    output = tmp_path / "scan-rejected"
    monkeypatch.setattr(config, "load_config", lambda _: app)
    monkeypatch.setattr(device, "DeviceManager", lambda _: pytest.fail("DeviceManager must not be reached"))
    result = pilot.capture_pilot_baseline(config_path, output)
    assert result["status"] == "failed"
    assert "manual_device_or_bounded_timeouts_required" in result["reason"]
    assert result["action_count"] == 0
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["source_sha256_before"]


def test_device_prelude_budget_stops_before_session_connect(tmp_path, monkeypatch):
    from autococ import config, device, session
    from autococ.config import AdbConfig, GameConfig, RuntimeConfig, StopConfig

    @dataclass
    class Config:
        adb: AdbConfig
        game: GameConfig
        runtime: RuntimeConfig
        stop: StopConfig
        vision: object
        vision_agent: object

    app = Config(
        adb=AdbConfig(adb_path=tmp_path, manual_serial="device", auto_scan_ports=False),
        game=GameConfig(package_name=pilot.PACKAGE, baseline_resolution=pilot.BASE),
        runtime=RuntimeConfig(dry_run=False, step_timeout_sec=1), stop=StopConfig(),
        vision=SimpleNamespace(template_dir=pilot.ASSET_DIR),
        vision_agent=SimpleNamespace(enabled=False, jev_enabled=False))
    config_path = tmp_path / "config.toml"
    config_path.write_text("pilot config", encoding="utf-8")
    clock = {"now": 100.0}
    monkeypatch.setattr(pilot, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    monkeypatch.setattr(config, "load_config", lambda _: app)

    def connect():
        clock["now"] = 145.0
        return SimpleNamespace(serial="device")

    monkeypatch.setattr(device, "DeviceManager", lambda _: SimpleNamespace(adb=object(), connect=connect))
    monkeypatch.setattr(session.GameSession, "connect", lambda *args, **kwargs: pytest.fail("Session must not connect"))
    result = pilot.capture_pilot_baseline(config_path, tmp_path / "timed-out", limit_sec=90)
    assert result["status"] == "failed"
    assert "budget_exhausted_before_foreground_check" in result["reason"]
    assert result["action_count"] == 0
