"""A portrait animation may be reread, but no earlier hero ID may be reused."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from autococ.army_control import (_inspect_mixed_archer_current,
                                  ensure_mixed_archer_army)
from autococ.errors import FlowError, StopRequested
from autococ.scene import SceneSnapshot


HEROES = ("grand_warden", "dragon_duke", "minion_prince", "archer_queen")
HERO_BOXES = ((23, 185, 147, 668), (153, 185, 277, 668),
              (283, 185, 407, 668), (413, 185, 537, 668))


def _identity(unit_id):
    return {"unit_id": unit_id, "confidence": .99, "version_verified": True,
            "template_client_version": "18.647.2"}


def army_frame(name, *, gap=True, captured=None):
    path = Path(name)
    troop_counts = (("electro_dragon", 10), ("dragon", 1), ("furnace", 1),
                    ("balloon", 1), ("archer", 3))
    troops = [{"kind": "troop", "source": "army", "unit_id": unit, "count": count,
               "evidence": {"identity": _identity(unit)}} for unit, count in troop_counts]
    spells = [{"kind": "spell", "source": "army", "unit_id": unit, "count": count,
               "evidence": {"identity": _identity(unit)}} for unit, count in
              (("lightning_spell", 6), ("totem_spell", 5))]
    hero_cards = [{"kind": "hero", "source": "army", "unit_id": None if gap and i == 2 else unit,
                   "count": 1, "bbox": list(box),
                   "evidence": {"identity": {} if gap and i == 2 else _identity(unit)}}
                  for i, (unit, box) in enumerate(zip(HEROES, HERO_BOXES))]
    siege = {"kind": "siege", "source": "army", "unit_id": "siege_barracks",
             "count": 1, "evidence": {"identity": _identity("siege_barracks")}}
    ratio = {"text": "346/346", "confidence": .999,
             "bbox": [578, 156, 663, 180], "source": "full_ocr"}
    army = {"manifest": {"frame": str(path), "layout": "my_army_rows_v1",
                         "supported_layout": True, "complete": True,
                         "identity_complete": True, "unknowns": [],
                         "source_resolution": [2560, 1440],
                         "baseline_resolution": [1280, 720],
                         "troops": troops, "spells": spells},
            "troops": {"used": 346, "capacity": 346,
                       "meaning": "displayed_capacity_ratio", "evidence": ratio},
            "identity_cards": [*hero_cards, siege],
            "identity_coverage": {"hero": not gap, "siege": True},
            "hero_evidence": {"frame": str(path), "layout": "my_army_large_heroes_v1",
                              "complete": not gap, "cards": hero_cards,
                              "unknowns": ["hero_identity_unavailable"] if gap else []},
            "siege_evidence": {"frame": str(path), "complete": True,
                               "unknowns": [], "cards": [siege]},
            "hero_loadout_complete": not gap,
            "hero_loadout": {} if gap else {unit: {"pet_id": "owl",
                                                    "equipment_ids": ["book", "gem"]}
                                            for unit in HEROES}}
    o = {"army": army, "army_editor": {"surface": "current", "ready": True},
         "baseline_resolution": [1280, 720], "source_resolution": [2560, 1440],
         "ocr": [{k: v for k, v in ratio.items() if k != "source"}],
         "observed_at_monotonic": time.monotonic() - 4 if captured is None else captured,
         "frame_id": str(path)}
    return SceneSnapshot("training", .99, path, o)


class NoInputSession:
    client_version = "18.647.2"

    def __init__(self, frames, *, deadline_after=None):
        self.frames = iter(frames)
        self.observed = []
        self.last_snapshot = None
        self.action_count = 0
        self._input_attempt_generation = 0
        self.taps = []
        self.deadline_checks = 0
        self.deadline_after = deadline_after
        self.capture = SimpleNamespace(adb=object(), serial="test")
        self.config = SimpleNamespace(game=SimpleNamespace(package_name="game.test"),
                                      runtime=SimpleNamespace(step_timeout_sec=5))

    def check_deadline(self):
        self.deadline_checks += 1
        if self.deadline_after is not None and self.deadline_checks > self.deadline_after:
            raise FlowError("deadline reached")

    def observe(self, label):
        shot = next(self.frames)
        self.observed.append((label, shot.screenshot_path))
        self.last_snapshot = shot
        return shot

    def tap(self, *_args, **_kwargs):
        self.taps.append(True)
        raise AssertionError("An identity reread must not tap")


def test_original_frame10_cache_still_rejects_without_reusing_frame9():
    events = Path(r"E:\projects\AutoCOC\reports\completion-20261015\live"
                  r"\20261017-021753-311126-d7771699\events.jsonl")
    if not events.exists():
        pytest.skip("Local live cache is unavailable")
    row = next(json.loads(line) for line in events.read_text(encoding="utf8").splitlines()
               if '00010-mixed-army-preparation.png' in line and '"kind": "observation"' in line)
    shot = SceneSnapshot(row["scene"], row["confidence"], Path(row["frame"]),
                         row["observations"])
    assert _inspect_mixed_archer_current(shot, "18.647.2")[1] == \
        "hero_or_siege_identity_coverage_missing"


@pytest.mark.parametrize("gaps", (1, 2))
def test_only_last_independent_complete_frame_can_succeed(monkeypatch, gaps):
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_args, **_kw: "game.test")
    origin = time.monotonic() - 5
    frames = [army_frame(f"gap-{i}.png", captured=origin + i) for i in range(gaps)]
    frames.append(army_frame("fresh-full.png", gap=False, captured=origin + gaps))
    session = NoInputSession(frames)
    result = ensure_mixed_archer_army(session)
    assert result.status == "succeeded"
    assert result.metrics["recipe_mutations"] == 0
    assert result.metrics["final_frame"] == "fresh-full.png"
    assert len(result.evidence) == gaps + 1 <= 3
    assert not session.taps


def test_three_gap_frames_exhaust_without_input(monkeypatch):
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_args, **_kw: "game.test")
    origin = time.monotonic() - 5
    session = NoInputSession([army_frame(f"gap-{i}.png", captured=origin + i)
                              for i in range(3)])
    result = ensure_mixed_archer_army(session)
    assert result.status == "not_supported"
    assert result.reason == "hero_or_siege_identity_coverage_missing"
    assert result.metrics["recipe_mutations"] == 0
    assert len(result.evidence) == 3 and not session.taps


def test_third_frame_cannot_reuse_initial_identity_or_path(monkeypatch):
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_args, **_kw: "game.test")
    origin = time.monotonic() - 5
    for reused in ("path", "id"):
        first = army_frame("initial.png", captured=origin)
        middle = army_frame("middle.png", captured=origin + 1)
        final = army_frame("final.png", gap=False, captured=origin + 2)
        if reused == "path":
            final = replace(final, screenshot_path=first.screenshot_path)
        else:
            final.observations["frame_id"] = first.observations["frame_id"]
        session = NoInputSession([first, middle, final])
        result = ensure_mixed_archer_army(session)
        assert result.status == "not_supported"
        assert result.metrics["recipe_mutations"] == 0
        assert len(session.observed) == 3 and not session.taps


@pytest.mark.parametrize("defect", ("weak_capacity", "stale_capture", "wrong_page",
                                    "weak_scene",
                                    "account_overlay", "missing_siege"))
def test_incomplete_or_untrusted_initial_frame_does_not_reread(monkeypatch, defect):
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_args, **_kw: "game.test")
    first = army_frame("initial.png")
    if defect == "weak_capacity":
        first.observations["army"]["troops"]["evidence"]["confidence"] = .8
    elif defect == "stale_capture":
        first.observations["observed_at_monotonic"] = time.monotonic() - 20
    elif defect == "wrong_page":
        first.observations["army_editor"]["surface"] = "picker"
    elif defect == "weak_scene":
        first = replace(first, confidence=.7)
    elif defect == "account_overlay":
        first.observations["account_in_use"] = True
    elif defect == "missing_siege":
        first.observations["army"]["identity_coverage"]["siege"] = False
    session = NoInputSession([first])
    result = ensure_mixed_archer_army(session)
    assert result.status == "not_supported"
    assert result.metrics["recipe_mutations"] == 0
    assert len(session.observed) == 1 and not session.taps


@pytest.mark.parametrize("defect", ("same_path", "old_time", "capacity", "roster",
                                    "known_hero", "account", "foreground", "weak_capacity"))
def test_retry_rejects_any_stale_changed_or_interrupted_current_frame(monkeypatch, defect):
    monkeypatch.setattr("autococ.device.foreground_package",
                        lambda *_args, **_kw: "other.app" if defect == "foreground" else "game.test")
    origin = time.monotonic() - 5
    first = army_frame("initial.png", captured=origin)
    second = army_frame("next.png", gap=False, captured=origin + 1)
    if defect == "same_path":
        second = replace(second, screenshot_path=first.screenshot_path)
    elif defect == "old_time":
        second.observations["observed_at_monotonic"] = origin
    elif defect == "capacity":
        second.observations["army"]["troops"]["capacity"] = 347
    elif defect == "roster":
        second.observations["army"]["manifest"]["troops"][1]["count"] = 2
    elif defect == "known_hero":
        second.observations["army"]["identity_cards"][0]["unit_id"] = "barbarian_king"
        second.observations["army"]["identity_cards"][0]["evidence"]["identity"] = \
            _identity("barbarian_king")
    elif defect == "account":
        second = replace(second, scene="account_in_use")
    elif defect == "weak_capacity":
        second.observations["army"]["troops"]["evidence"]["confidence"] = .8
    session = NoInputSession([first, second])
    result = ensure_mixed_archer_army(session)
    assert result.status == "not_supported"
    assert result.metrics["recipe_mutations"] == 0
    assert len(result.evidence) <= 2 and not session.taps


def test_deadline_stops_before_reread(monkeypatch):
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_args, **_kw: "game.test")
    session = NoInputSession([army_frame("initial.png")], deadline_after=1)
    with pytest.raises(FlowError, match="deadline reached"):
        ensure_mixed_archer_army(session)
    assert len(session.observed) == 1 and not session.taps


def test_stop_request_stops_before_reread(monkeypatch):
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_args, **_kw: "game.test")
    session = NoInputSession([army_frame("initial.png")])
    original = session.check_deadline

    def check():
        original()
        if session.deadline_checks > 1:
            raise StopRequested("user stop")

    session.check_deadline = check
    with pytest.raises(StopRequested, match="user stop"):
        ensure_mixed_archer_army(session)
    assert len(session.observed) == 1 and not session.taps


def test_input_generation_or_client_version_drift_stops(monkeypatch):
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_args, **_kw: "game.test")
    for drift in ("input", "attempt", "version"):
        session = NoInputSession([army_frame("initial.png"), army_frame("next.png", gap=False)])
        original = session.observe

        def changed(label):
            shot = original(label)
            if len(session.observed) == 2:
                if drift == "input":
                    session.action_count += 1
                elif drift == "attempt":
                    session._input_attempt_generation += 1
                else:
                    session.client_version = "unknown"
            return shot

        session.observe = changed
        result = ensure_mixed_archer_army(session)
        assert result.status == "not_supported" and result.metrics["recipe_mutations"] == 0
        assert len(result.evidence) == 2 and not session.taps
