from pathlib import Path
from types import SimpleNamespace

import pytest

from autococ.errors import ConfigError, DeploymentError, FlowError, StopRequested
from autococ.scene import SceneSnapshot
from autococ.strategy_config import StrategyDefinition, StrategyStep, load_strategy, planned_steps
from autococ.strategy_execution import (_battle_bar_viewport, _end_battle_confirmation,
                                         declared_building_targets,
                                         _find_named_card, _match_card, _points,
                                         _same_battle_bar_view,
                                         _terrain, _wait_step, execute_strategy)
from autococ.unit_catalog import ArmyRecipe, ArmyRequirement


ROOT = Path(__file__).resolve().parents[1]


def test_builtin_strategies_have_validated_action_budget():
    for name in ("single_edge", "two_edge", "two_edge_heroes", "lightning_snipe"):
        definition = load_strategy(ROOT / "strategies" / f"{name}.toml")
        assert definition.id == name
        assert definition.steps
    lightning = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    assert lightning.recipe.units[0].unit_id == "lightning_spell"
    assert lightning.steps[-1].target == "target_destroyed"


def test_invalid_strategy_is_rejected_before_input(tmp_path):
    strategy = tmp_path / "bad.toml"
    strategy.write_text('id="bad"\nlabel="bad"\narmy_mode="captured"\n'
                        '[[steps]]\naction="cast_spell"\nunit_id="lightning_spell"\n'
                        'count=0\ntarget="objective"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="bounded count"):
        load_strategy(strategy)


@pytest.mark.parametrize("action,unit_id,target", [
    ("deploy_hero", "grand_warden", ""),
    ("deploy_siege", "wall_wrecker", ""),
    ("activate_ability", "grand_warden", ""),
    ("wait", "", "battle"),
    ("end_battle", "", ""),
])
def test_boolean_count_is_never_one_action(tmp_path, action, unit_id, target):
    strategy = tmp_path / "boolean-count.toml"
    strategy.write_text(f'id="bad"\nlabel="bad"\narmy_mode="captured"\n'
                        f'[[steps]]\naction="{action}"\nunit_id="{unit_id}"\n'
                        f'target="{target}"\ncount=true\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_strategy(strategy)


def _frame(name, *, scene="battle", count=2, building_state="alive"):
    path = Path(f"{name}.png")
    card = {"unit_id": "lightning_spell", "kind": "spell", "source": "army",
            "count": count, "point": [110, 650], "bbox": [70, 600, 155, 710],
            "confidence": .99}
    building = {"building_id": "spell_factory:500:300", "type": "spell_factory",
                "point": [500, 300], "bbox": [470, 275, 530, 330],
                "confidence": .97, "state": building_state,
                "frame": str(path), "evidence": {"template": "fixture"}}
    return SceneSnapshot(scene, .95, path, {"battle": {"slots": [card]},
                                             "buildings": [building], "count": count})


class _Session:
    def __init__(self, frames):
        self.client_version = "18.600.7"
        self.frames = list(frames)
        self.taps = []
        self.clicks = []
        self.swipes = []
        self.last_snapshot = None
        self.config = SimpleNamespace(battle=SimpleNamespace(deploy_timeout_sec=30,
                                                               target_building="spell_factory"),
                                      runtime=SimpleNamespace(poll_interval_sec=0),
                                      game=SimpleNamespace(baseline_resolution=(1280, 720)))
        self.recognizer = None

    def check_deadline(self):
        pass

    def _validate_snapshot(self, frame):
        pass

    def tap(self, frame, point, *, reason):
        self.taps.append((frame.screenshot_path.name, tuple(point), reason))

    def click(self, frame, name):
        self.clicks.append(name)

    def swipe_battle_bar(self, frame, *, direction, reason):
        self.swipes.append((frame.screenshot_path.name, direction, reason))
        self.last_snapshot = None

    def observe(self, label, **kwargs):
        self.last_snapshot = self.frames.pop(0)
        return self.last_snapshot

    def wait_for(self, scenes, **kwargs):
        frame = self.observe("wait")
        assert frame.scene in scenes
        return frame


def test_pure_spell_battle_requires_fresh_positive_destruction(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["count"])
    strategy = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    scout = _frame("scout")
    session = _Session([_frame("selected"), _frame("consumed", count=1),
                        _frame("destroyed", count=1, building_state="destroyed"),
                        SceneSnapshot("settlement", .95, Path("settled.png"), {})])
    result = execute_strategy(session, scout, strategy)
    receipt = result.observations["deployment"]
    assert receipt["completed"] is True
    assert receipt["spells_used"] == receipt["offensive_actions"] == 1
    assert receipt["deployed_units"] == 0
    assert receipt["target_building_destroyed"] is True
    assert session.clicks == ["end_battle"]
    assert len(session.taps) == 2


def test_missing_building_evidence_does_not_issue_any_input():
    strategy = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    scout = _frame("scout")
    scout.observations["buildings"] = []
    session = _Session([])
    with pytest.raises(DeploymentError) as exc:
        execute_strategy(session, scout, strategy)
    assert session.taps == []
    assert session.clicks == []
    assert exc.value.partial_receipt["offensive_actions"] == 0


def test_multiple_same_type_buildings_bind_one_stable_id(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["count"])
    strategy = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    scout = _frame("scout")
    right = dict(scout.observations["buildings"][0],
                 building_id="spell_factory:700:300", point=[700, 300])
    scout.observations["buildings"].append(right)
    session = _Session([_frame("selected"), _frame("consumed", count=1),
                        _frame("destroyed", count=1, building_state="destroyed"),
                        SceneSnapshot("settlement", .95, Path("settled.png"), {})])
    result = execute_strategy(session, scout, strategy)
    assert result.observations["deployment"]["target_building_id"] == "spell_factory:500:300"
    assert session.taps[1][1] == (500, 300)


def test_disappearing_building_never_triggers_another_spell_or_early_end(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["count"])
    strategy = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    missing = [_frame(f"missing-{index}", count=1) for index in range(4)]
    for frame in missing:
        frame.observations["buildings"] = []
    session = _Session([_frame("selected"), _frame("consumed", count=1), *missing])
    with pytest.raises(DeploymentError) as exc:
        execute_strategy(session, _frame("scout"), strategy)
    assert len(session.taps) == 2  # Select once, cast once.
    assert session.clicks == []
    assert exc.value.partial_receipt["spells_used"] == 1
    assert exc.value.partial_receipt["completed"] is False


def test_unknown_consumption_never_reissues_placement(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["count"])
    strategy = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    session = _Session([_frame("selected"), *(_frame(f"unchanged-{i}") for i in range(3))])
    with pytest.raises(DeploymentError) as exc:
        execute_strategy(session, _frame("scout"), strategy)
    assert len(session.taps) == 2
    assert session.clicks == []
    assert exc.value.partial_receipt["issued_placements"] == 1
    assert exc.value.partial_receipt["spells_used"] == 0


def test_one_step_timeout_preserves_first_consumption_without_second_tap(monkeypatch):
    class Clock:
        after_consumption = False
        reads_after_consumption = 0

        def monotonic(self):
            if not self.after_consumption:
                return 0.0
            self.reads_after_consumption += 1
            return 0.8 if self.reads_after_consumption == 1 else 1.1

    clock = Clock()
    monkeypatch.setattr("autococ.strategy_execution.time",
                        SimpleNamespace(monotonic=clock.monotonic, sleep=lambda _: None))

    def remaining(frame, card, recognizer=None):
        if frame.screenshot_path.name == "consumed-first.png":
            clock.after_consumption = True
        return frame.observations["counts"][card["unit_id"]]

    monkeypatch.setattr("autococ.strategy_execution._remaining", remaining)
    troop = dict(_troop("barbarian", (100, 650)), count=2)
    scout = _troop_frame("start", [troop], {"barbarian": 2}, terrain=True)
    session = _Session([_troop_frame("selected", [troop], {"barbarian": 2}),
                        _troop_frame("consumed-first", [troop], {"barbarian": 1})])
    session.config.battle.deploy_timeout_sec = 1
    definition = StrategyDefinition("bounded", "bounded",
        ArmyRecipe((ArmyRequirement("barbarian", 2),)),
        (StrategyStep("deploy_troop", "barbarian", count=2),))

    with pytest.raises(DeploymentError, match="Strategy deployment deadline exceeded") as exc:
        execute_strategy(session, scout, definition)
    assert len(session.taps) == 2  # Card selection and exactly one placement.
    receipt = exc.value.partial_receipt
    assert receipt["issued_placements"] == receipt["deployed_units"] == 1
    assert receipt["actions"][0]["consumed"] == 1
    assert receipt["completed"] is False


def test_observation_crossing_deadline_blocks_next_hero_input(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("autococ.strategy_execution.time",
                        SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda _: None))
    import autococ.hero_state as hero_state
    monkeypatch.setattr(hero_state, "recognize_hero_state",
                        lambda *args, **kwargs: {"selected": True})
    definition = StrategyDefinition("hero", "hero",
        ArmyRecipe((ArmyRequirement("grand_warden", 1),)),
        (StrategyStep("deploy_hero", "grand_warden", edge="midpoint"),))
    session = _Session([_hero_frame("selected")])
    session.config.battle.deploy_timeout_sec = 1
    observe = session.observe

    def delayed_observe(label, **kwargs):
        frame = observe(label, **kwargs)
        clock.now = 1.1
        return frame

    session.observe = delayed_observe
    with pytest.raises(DeploymentError, match="Strategy deployment deadline exceeded") as exc:
        execute_strategy(session, _hero_frame("hero-start", terrain=True), definition)
    assert len(session.taps) == 1
    assert exc.value.partial_receipt["actions"][0]["clicks"] == 1
    assert "selected.png" in exc.value.partial_receipt["evidence"]


def test_shared_terrain_preparer_checks_deadline_before_each_native_input(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("autococ.strategy_execution.time",
                        SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda _: None))
    session = _Session([])
    session.action_count = 0
    zooms = []

    def zoom_out():
        zooms.append(True)
        clock.now = 1.1

    session.native = SimpleNamespace(zoom_out=zoom_out)

    def prepare(guarded, frame, receipt):
        guarded.native.zoom_out()
        guarded.native.zoom_out()

    monkeypatch.setattr("autococ.strategy_execution._prepare_two_edge_view", prepare)
    with pytest.raises(Exception, match="Strategy deployment deadline exceeded"):
        _terrain(session, _hero_frame("start"), {"evidence": []}, {0}, 1.0)
    assert len(zooms) == 1


def _troop_frame(name, slots, counts, *, terrain=False, manifest=False):
    path = Path(f"{name}.png")
    observed = {"battle": {"slots": slots}, "counts": counts}
    if terrain:
        observed["terrain"] = [
            {"edge": edge, "point": point, "frame": str(path), "verified": True}
            for edge, points in ((0, ([250, 150], [250, 250])),
                                 (1, ([250, 350], [250, 450]))) for point in points]
    if manifest:
        observed["expected_army_manifest"] = {"complete": True,
                                                "troops": [{"count": 1}, {"count": 1}]}
    return SceneSnapshot("battle", .95, path, observed)


def _troop(unit_id, point):
    return {"unit_id": unit_id, "kind": "troop", "source": "army", "count": 1,
            "confidence": .99, "point": list(point), "bbox": [point[0]-40, 600, point[0]+40, 710]}


def _prepared(observed):
    return {"status": "succeeded", "frame": "army-verified.png", "observed": observed}


def test_verified_absent_optional_spell_skips_target_preflight_and_records_receipt(monkeypatch):
    scout = _troop_frame("optional-spell", [_troop("barbarian", (100, 650))], {}, terrain=True)
    definition = StrategyDefinition("optional", "optional", ArmyRecipe((
        ArmyRequirement("barbarian", 1), ArmyRequirement("lightning_spell", 1, optional=True))),
        (StrategyStep("cast_spell", "lightning_spell", target="spell_factory"),
         StrategyStep("deploy_troop", "barbarian")))
    used = []

    def consume(session, frame, card, points, receipt, deadline, *, kind):
        used.append(card["unit_id"])
        receipt["deployed_units"] += 1
        receipt["offensive_actions"] += 1
        receipt["verified"] = True
        return frame

    monkeypatch.setattr("autococ.strategy_execution._consume", consume)
    session = _Session([])
    receipt = execute_strategy(session, scout, definition,
        prepared_army=_prepared({"troop": {"barbarian": 1}, "spell": {}})).observations["deployment"]
    assert used == ["barbarian"]
    assert receipt["completed"] is True
    assert receipt["actions"][0] == {"action": "cast_spell", "unit_id": "lightning_spell",
        "status": "skipped", "reason": "optional_unit_absent_in_verified_army",
        "army_frame": "army-verified.png", "verified": True}
    assert session.taps == []


def test_declared_building_requirement_returns_when_battle_card_contradicts_absence():
    definition = StrategyDefinition("optional", "optional", ArmyRecipe((
        ArmyRequirement("barbarian", 1), ArmyRequirement("lightning_spell", 1, optional=True))),
        (StrategyStep("cast_spell", "lightning_spell", target="spell_factory"),
         StrategyStep("deploy_troop", "barbarian")))
    proof = _prepared({"troop": {"barbarian": 1}, "spell": {}})
    assert declared_building_targets(definition, "", prepared_army=proof) == set()
    spell = {"unit_id": "lightning_spell", "kind": "spell", "source": "army",
             "count": 1, "confidence": .99, "point": [200, 650]}
    scout = _troop_frame("contrary-spell", [_troop("barbarian", (100, 650)), spell], {})
    assert declared_building_targets(definition, "", prepared_army=proof,
                                     scout=scout) == {"spell_factory"}
    anonymous = dict(spell, unit_id=None)
    scout = _troop_frame("unknown-spell", [_troop("barbarian", (100, 650)), anonymous], {})
    assert declared_building_targets(definition, "", prepared_army=proof,
                                     scout=scout) == {"spell_factory"}


def test_planner_static_building_capability_stays_declared_when_optional_card_absent():
    definition = StrategyDefinition("planned", "planned", ArmyRecipe((
        ArmyRequirement("barbarian", 1), ArmyRequirement("lightning_spell", 1, optional=True))),
        (StrategyStep("cast_spell", "lightning_spell", target="spell_factory"),),
        planner=Path("local-planner.py"))
    proof = _prepared({"troop": {"barbarian": 1}, "spell": {}})
    assert declared_building_targets(definition, "", prepared_army=proof) == {"spell_factory"}


def test_verified_absent_optional_hero_skips_deploy_wait_and_ability(monkeypatch):
    scout = _troop_frame("optional-hero", [_troop("barbarian", (100, 650))], {}, terrain=True)
    definition = StrategyDefinition("optional-hero", "optional-hero", ArmyRecipe((
        ArmyRequirement("barbarian", 1), ArmyRequirement("grand_warden", 1, optional=True))),
        (StrategyStep("deploy_hero", "grand_warden"),
         StrategyStep("wait", "grand_warden", target="hero_ready"),
         StrategyStep("activate_ability", "grand_warden"),
         StrategyStep("deploy_troop", "barbarian")))

    def consume(session, frame, card, points, receipt, deadline, *, kind):
        receipt["deployed_units"] += 1
        receipt["offensive_actions"] += 1
        receipt["verified"] = True
        return frame

    monkeypatch.setattr("autococ.strategy_execution._consume", consume)
    receipt = execute_strategy(_Session([]), scout, definition,
        prepared_army=_prepared({"troop": {"barbarian": 1}, "hero": {}})).observations["deployment"]
    assert receipt["completed"] is True
    assert [action["action"] for action in receipt["actions"]] == [
        "deploy_hero", "wait", "activate_ability"]
    assert all(action["status"] == "skipped" for action in receipt["actions"])
    assert receipt["hero_abilities"] == 0


@pytest.mark.parametrize("scenario", ["unknown_card", "required_unit", "clan_source",
                                      "no_proof", "unknown_army_kind"])
def test_optional_absence_never_hides_unknown_required_or_clan_card(monkeypatch, scenario):
    required = scenario == "required_unit"
    recipe = ArmyRecipe((ArmyRequirement("barbarian", 1),
                         ArmyRequirement("archer", 1, optional=not required)))
    source = "clan_reinforcement" if scenario == "clan_source" else "auto"
    definition = StrategyDefinition("guard", "guard", recipe,
        (StrategyStep("deploy_troop", "archer", source=source),))
    cards = [_troop("barbarian", (100, 650))]
    if scenario == "unknown_card":
        cards.append(dict(_troop("archer", (200, 650)), unit_id=None))
    scout = _troop_frame("guard", cards, {}, terrain=True)
    checked = []

    def find(*args, **kwargs):
        checked.append(args[2])
        raise FlowError("Named card identity not independently verified")

    monkeypatch.setattr("autococ.strategy_execution._find_named_card", find)
    session = _Session([])
    proof = None if scenario == "no_proof" else _prepared(
        {} if scenario == "unknown_army_kind" else {"troop": {"barbarian": 1}})
    with pytest.raises(DeploymentError, match="Named card identity not independently verified"):
        execute_strategy(session, scout, definition, prepared_army=proof)
    assert checked == ["archer"]
    assert session.taps == [] and session.swipes == []


def test_visible_optional_card_contradicts_preparation_absence(monkeypatch):
    scout = _troop_frame("visible-optional", [_troop("archer", (100, 650))], {}, terrain=True)
    definition = StrategyDefinition("present", "present", ArmyRecipe((
        ArmyRequirement("barbarian", 1), ArmyRequirement("archer", 1, optional=True))),
        (StrategyStep("deploy_troop", "archer"),))
    used = []

    def consume(session, frame, card, points, receipt, deadline, *, kind):
        used.append(card["unit_id"])
        receipt["deployed_units"] += 1
        receipt["offensive_actions"] += 1
        receipt["verified"] = True
        return frame

    monkeypatch.setattr("autococ.strategy_execution._consume", consume)
    receipt = execute_strategy(_Session([]), scout, definition,
        prepared_army=_prepared({"troop": {"barbarian": 1}})).observations["deployment"]
    assert used == ["archer"]
    assert receipt["skipped_actions"] == []


def test_all_optional_offensive_actions_absent_never_complete(monkeypatch):
    scout = _troop_frame("none", [], {}, terrain=True)
    definition = StrategyDefinition("none", "none", ArmyRecipe((
        ArmyRequirement("barbarian", 1), ArmyRequirement("archer", 1, optional=True))),
        (StrategyStep("deploy_troop", "archer"),))
    session = _Session([])
    with pytest.raises(DeploymentError, match="no deployable offensive action") as exc:
        execute_strategy(session, scout, definition,
            prepared_army=_prepared({"troop": {"barbarian": 1}}))
    assert exc.value.partial_receipt["completed"] is False
    assert exc.value.partial_receipt["skipped_actions"][0]["unit_id"] == "archer"
    assert session.taps == [] and session.swipes == []


def test_named_cards_rebind_after_reordering(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["counts"][card["unit_id"]])
    first, second = _troop("barbarian", (100, 650)), _troop("archer", (200, 650))
    moved = [_troop("archer", (110, 650)), _troop("barbarian", (210, 650))]
    recipe = ArmyRecipe((ArmyRequirement("barbarian", 1), ArmyRequirement("archer", 1)))
    definition = StrategyDefinition("reorder", "reorder", recipe,
        (StrategyStep("deploy_troop", "barbarian", 1), StrategyStep("deploy_troop", "archer", 1)))
    scout = _troop_frame("first", [first, second], {"barbarian": 1, "archer": 1}, terrain=True)
    session = _Session([
        _troop_frame("selected-first", [], {"barbarian": 1, "archer": 1}),
        _troop_frame("consumed-first", [], {"barbarian": 0, "archer": 1}),
        _troop_frame("moved", moved, {"barbarian": 0, "archer": 1}),
        _troop_frame("selected-second", [], {"barbarian": 0, "archer": 1}),
        _troop_frame("consumed-second", [], {"barbarian": 0, "archer": 0}),
        _troop_frame("done", moved, {"barbarian": 0, "archer": 0}),
    ])
    result = execute_strategy(session, scout, definition)
    assert result.observations["deployment"]["deployed_units"] == 2
    assert session.taps[0][1] == (100, 650)
    assert session.taps[2][1] == (110, 650)


def test_captured_line_uses_independent_manifest(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["counts"][card["unit_id"]])
    first, second = _troop("barbarian", (100, 650)), _troop("archer", (200, 650))
    definition = load_strategy(ROOT / "strategies/two_edge.toml")
    scout = _troop_frame("captured", [first, second], {"barbarian": 1, "archer": 1},
                         terrain=True, manifest=True)
    session = _Session([
        _troop_frame("selected-first", [], {"barbarian": 1, "archer": 1}),
        _troop_frame("consumed-first", [], {"barbarian": 0, "archer": 1}),
        _troop_frame("after-first", [first, second], {"barbarian": 0, "archer": 1}),
        _troop_frame("selected-second", [], {"barbarian": 0, "archer": 1}),
        _troop_frame("consumed-second", [], {"barbarian": 0, "archer": 0}),
        _troop_frame("after-second", [first, second], {"barbarian": 0, "archer": 0}),
    ])
    result = execute_strategy(session, scout, definition)
    assert result.observations["deployment"]["completed"] is True
    assert result.observations["deployment"]["issued_placements"] == 2


def test_captured_line_rejects_stacks_missing_after_boundary_reveal(monkeypatch):
    cards = [dict(_troop("barbarian", (x, 650)), count=count) for x, count in
             ((60, 8), (156, 2), (253, 1), (350, 1))]
    scout = _troop_frame("four-stacks", cards, {}, terrain=True)
    scout.observations["expected_army_manifest"] = {
        "complete": True, "troops": [{"count": count} for count in (8, 2, 1, 1)]}
    shrunk = _troop_frame("two-stacks", [cards[0], cards[3]], {}, terrain=True)
    monkeypatch.setattr("autococ.strategy_execution._terrain",
                        lambda *args: (shrunk, scout.observations["terrain"]))
    session = _Session([])
    with pytest.raises(DeploymentError, match="stack missing after boundary reveal") as exc:
        execute_strategy(session, scout, load_strategy(ROOT / "strategies/two_edge.toml"))
    assert session.taps == []
    assert exc.value.partial_receipt["issued_placements"] == 0


def test_captured_event_stack_is_counted_but_same_unit_clan_card_is_excluded(monkeypatch):
    own = dict(_troop("barbarian", (60, 650)), count=1)
    event = dict(_troop("event_super_pekka", (156, 650)), count=40, source="event")
    clan = dict(_troop("event_super_pekka", (253, 650)), count=5,
                source="clan_reinforcement")
    scout = _troop_frame("event-stacks", [own, event, clan], {}, terrain=True)
    scout.observations["expected_army_manifest"] = {
        "complete": True, "troops": [{"count": 1}, {"count": 40}]}
    selected = []

    def consume(session, frame, card, points, receipt, deadline, *, kind):
        selected.append((card["source"], card["count"], card["point"]))
        receipt["offensive_actions"] += 1
        receipt["deployed_units"] += 1
        receipt["verified"] = True
        return frame

    monkeypatch.setattr("autococ.strategy_execution._consume", consume)
    definition = StrategyDefinition("captured", "captured", None,
        (StrategyStep("deploy_troop", "*", count=1, edge="two"),), "captured")
    receipt = execute_strategy(_Session([]), scout, definition).observations["deployment"]
    assert selected == [("army", 1, [60, 650]), ("event", 40, [156, 650])]
    assert receipt["offensive_actions"] == 2


def _hero_frame(name, *, terrain=False):
    path = Path(f"{name}.png")
    hero = {"unit_id": "grand_warden", "kind": "hero", "source": "army", "count": None,
            "confidence": .99, "point": [100, 650], "bbox": [60, 580, 155, 710]}
    observed = {"battle": {"slots": [hero]}}
    if terrain:
        observed["terrain"] = [{"edge": 0, "point": point, "frame": str(path),
                                "verified": True} for point in ([250, 150], [250, 250])]
    return SceneSnapshot("battle", .95, path, observed)


def test_hero_deploy_and_ability_are_separate_verified_actions(monkeypatch):
    import autococ.hero_state as hero_state
    def state(path, bbox, **kwargs):
        name = Path(path).stem
        return {"selected": "selected" in name,
                "deployed": name in {"hero-deployed", "hero-ready", "ability-used"},
                "ability_ready": name == "hero-ready",
                "ability_used": name == "ability-used"}
    monkeypatch.setattr(hero_state, "recognize_hero_state", state)
    definition = StrategyDefinition("hero", "hero", ArmyRecipe((ArmyRequirement("grand_warden", 1),)),
        (StrategyStep("deploy_hero", "grand_warden", edge="midpoint"),
         StrategyStep("activate_ability", "grand_warden")))
    session = _Session([_hero_frame(name) for name in
                        ("hero-selected", "hero-deployed", "hero-ready", "ability-used", "hero-final")])
    result = execute_strategy(session, _hero_frame("hero-start", terrain=True), definition)
    assert result.observations["deployment"]["hero_abilities"] == 1
    assert result.observations["deployment"]["offensive_actions"] == 1
    assert len(session.taps) == 3


def test_planner_output_is_validated(tmp_path):
    planner = tmp_path / "planner.py"
    planner.write_text('def plan(context):\n    return [{"action": "cast_spell", '
                       '"unit_id": "lightning_spell", "count": 0, "target": "objective"}]\n',
                       encoding="utf-8")
    definition = StrategyDefinition("custom", "custom", None,
        (StrategyStep("wait", target="battle"),), "captured", planner)
    with pytest.raises(ConfigError, match="bounded count"):
        planned_steps(definition, {})


def test_planner_cannot_add_unannounced_building_before_game_input(tmp_path):
    planner = tmp_path / "planner.py"
    planner.write_text('def plan(context):\n    return [{"action": "cast_spell", '
                       '"unit_id": "lightning_spell", "target": "air_defense"}]\n',
                       encoding="utf-8")
    definition = StrategyDefinition("target", "target",
        ArmyRecipe((ArmyRequirement("lightning_spell", 2),)),
        (StrategyStep("cast_spell", "lightning_spell", target="objective"),),
        planner=planner)
    session = _Session([])
    with pytest.raises(DeploymentError, match="undeclared building target"):
        execute_strategy(session, _frame("planner-scout"), definition)
    assert session.taps == []


def test_planner_can_reorder_wait_and_adjust_declared_action_budget(tmp_path):
    planner = tmp_path / "planner.py"
    planner.write_text('def plan(context):\n    return [\n'
                       ' {"action": "wait", "target": "battle", "delay_sec": 0.2},\n'
                       ' {"action": "cast_spell", "unit_id": "lightning_spell", '
                       '"count": 1, "target": "spell_factory:500:300"},\n'
                       ' {"action": "deploy_troop", "unit_id": "barbarian", '
                       '"count": 2, "edge": "two"}]\n', encoding="utf-8")
    definition = StrategyDefinition("flexible", "flexible", ArmyRecipe((
        ArmyRequirement("barbarian", 2), ArmyRequirement("lightning_spell", 2))),
        (StrategyStep("deploy_troop", "barbarian", edge="first"),
         StrategyStep("cast_spell", "lightning_spell", 2, target="objective")),
        planner=planner)
    steps = planned_steps(definition, {}, objective_target="spell_factory")
    assert [step.action for step in steps] == ["wait", "cast_spell", "deploy_troop"]
    assert steps[-1].count == 2 and steps[-1].edge == "two"


def test_planner_cannot_add_an_unannounced_click_action(tmp_path):
    planner = tmp_path / "planner.py"
    planner.write_text('def plan(context):\n    return [{"action": "end_battle"}]\n',
                       encoding="utf-8")
    definition = StrategyDefinition("no_surrender", "no_surrender", None,
        (StrategyStep("wait", target="battle"),), "captured", planner)
    with pytest.raises(ConfigError, match="undeclared action end_battle"):
        planned_steps(definition, {})


@pytest.mark.parametrize("static_target,dynamic_target,expected", [
    ("objective", "relative:center", "undeclared relative spell target"),
    ("relative:center", "spell_factory", "undeclared building target"),
    ("objective", "air_defense", "undeclared building target"),
])
def test_planner_target_category_cannot_exceed_static_declaration(
        tmp_path, static_target, dynamic_target, expected):
    planner = tmp_path / "planner.py"
    planner.write_text('def plan(context):\n    return [{"action": "cast_spell", '
                       f'"unit_id": "lightning_spell", "target": "{dynamic_target}"}}]\n',
                       encoding="utf-8")
    definition = StrategyDefinition("bounded", "bounded",
        ArmyRecipe((ArmyRequirement("lightning_spell", 2),)),
        (StrategyStep("cast_spell", "lightning_spell", target=static_target),),
        planner=planner)
    with pytest.raises(ConfigError, match=expected):
        planned_steps(definition, {}, objective_target="spell_factory")


def test_stop_during_wait_propagates_without_replay():
    definition = StrategyDefinition("stop", "stop", None,
        (StrategyStep("wait", target="settlement"),), "captured")
    scout = _troop_frame("wait", [_troop("barbarian", (100, 650)),
                                  _troop("archer", (200, 650))],
                         {"barbarian": 1, "archer": 1}, manifest=True)
    session = _Session([])
    session.wait_for = lambda *args, **kwargs: (_ for _ in ()).throw(StopRequested())
    with pytest.raises(StopRequested):
        execute_strategy(session, scout, definition)
    assert session.taps == []


def test_spell_depletion_stops_before_exceeding_visible_budget(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["count"])
    strategy = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    frames = [_frame("selected-one"), _frame("consumed-one", count=1),
              _frame("alive-one", count=1), _frame("selected-two", count=1),
              _frame("consumed-two", count=0), _frame("alive-two", count=0)]
    session = _Session(frames)
    with pytest.raises(DeploymentError) as exc:
        execute_strategy(session, _frame("scout"), strategy)
    assert exc.value.partial_receipt["spells_used"] == 2
    assert exc.value.partial_receipt["issued_placements"] == 2
    assert len(session.taps) == 4
    assert session.clicks == []


def test_unknown_hero_ability_state_does_not_click_again(monkeypatch):
    import autococ.hero_state as hero_state
    def state(path, bbox, **kwargs):
        name = Path(path).stem
        return {"selected": name == "hero-selected",
                "deployed": name != "hero-selected",
                "ability_ready": name == "hero-ready",
                "ability_used": None}
    monkeypatch.setattr(hero_state, "recognize_hero_state", state)
    definition = StrategyDefinition("hero", "hero", ArmyRecipe((ArmyRequirement("grand_warden", 1),)),
        (StrategyStep("deploy_hero", "grand_warden", edge="midpoint"),
         StrategyStep("activate_ability", "grand_warden")))
    session = _Session([_hero_frame(name) for name in ("hero-selected", "hero-deployed",
                        "hero-ready", "unknown-one", "unknown-two", "unknown-three")])
    with pytest.raises(DeploymentError) as exc:
        execute_strategy(session, _hero_frame("hero-start", terrain=True), definition)
    assert len(session.taps) == 3
    assert exc.value.partial_receipt["hero_abilities"] == 0
    assert exc.value.partial_receipt["actions"][-1]["clicks"] == 1


def test_source_selector_disambiguates_own_and_clan_cards():
    own = _troop("barbarian", (100, 650))
    clan = dict(_troop("barbarian", (200, 650)), source="clan_reinforcement")
    frame = _troop_frame("same-unit", [own, clan], {"barbarian": 1})
    assert _match_card(frame, "barbarian", "troop")["point"] == [100, 650]
    assert _match_card(frame, "barbarian", "troop", source="army")["point"] == [100, 650]
    assert _match_card(frame, "barbarian", "troop", source="clan_reinforcement")["point"] == [200, 650]


def test_named_preflight_finds_fresh_card_across_battle_bar_and_ignores_clan(monkeypatch):
    scout = _troop_frame("bar-start", [_troop("barbarian", (100, 650)),
        dict(_troop("archer", (200, 650)), source="clan_reinforcement")], {}, terrain=True)
    own = _troop("archer", (430, 650))
    page_cards = [dict(_troop("archer", (300, 650)), source="clan_reinforcement"), own]
    first = _troop_frame("bar-moving", page_cards, {}, terrain=True)
    settled = _troop_frame("bar-settled", page_cards, {}, terrain=True)
    session = _Session([first, settled])
    selected = []

    def consume(session, frame, card, points, receipt, deadline, *, kind):
        selected.append((frame.screenshot_path.name, card["point"], card["source"]))
        receipt["deployed_units"] += 1
        receipt["offensive_actions"] += 1
        receipt["verified"] = True
        return frame

    monkeypatch.setattr("autococ.strategy_execution._consume", consume)
    definition = StrategyDefinition("scroll", "scroll",
        ArmyRecipe((ArmyRequirement("archer", 1),)),
        (StrategyStep("deploy_troop", "archer"),))
    result = execute_strategy(session, scout, definition)
    assert result.observations["deployment"]["completed"] is True
    assert selected == [("bar-settled.png", [430, 650], "army")]
    assert len(session.swipes) == 1 and session.swipes[0][1] == "left"
    assert session.taps == []  # The fake consumption records the bound card.


def test_battle_bar_repeated_viewport_stops_without_repeating_same_swipe():
    page = [_troop("barbarian", (100, 650)), _troop("wall_breaker", (200, 650))]
    session = _Session([_troop_frame(f"repeat-{index}", page, {}) for index in range(8)])
    receipt = {"evidence": []}
    with pytest.raises(FlowError, match="not identified in bounded battle bar search"):
        _find_named_card(session, _troop_frame("start", page, {}), "archer", "troop",
                         "auto", receipt, float("inf"))
    assert [direction for _, direction, _ in session.swipes] == ["left", "right"]
    assert session.taps == []


def test_battle_bar_waits_for_old_then_two_stable_new_views():
    old = [_troop("barbarian", (100, 650)), _troop("wall_breaker", (200, 650))]
    new = [_troop("archer", (420, 650)), _troop("wall_breaker", (320, 650))]
    session = _Session([_troop_frame("old-animation", old, {}),
                        _troop_frame("new-animation", new, {}),
                        _troop_frame("new-stable", new, {})])
    frame, card = _find_named_card(session, _troop_frame("start", old, {}), "archer", "troop",
                                   "auto", {"evidence": []}, float("inf"))
    assert frame.screenshot_path.name == "new-stable.png"
    assert card["point"] == [420, 650]
    assert len(session.swipes) == 1


def test_battle_bar_waits_through_realistic_inertial_rebound_without_second_swipe():
    initial = [_troop("barbarian", (100, 650)), _troop("wall_breaker", (200, 650))]
    moving = [_troop("barbarian", (40, 650)), _troop("wall_breaker", (140, 650)),
              _troop("archer", (400, 650))]
    rebound = [_troop("barbarian", (70, 650)), _troop("wall_breaker", (170, 650)),
               _troop("archer", (430, 650))]
    settled = [_troop("barbarian", (72, 651)), _troop("wall_breaker", (171, 650)),
               _troop("archer", (431, 650))]
    session = _Session([_troop_frame("old-after-swipe", initial, {}),
                        _troop_frame("moving", moving, {}),
                        _troop_frame("rebound", rebound, {}),
                        _troop_frame("settled", settled, {})])
    frame, card = _find_named_card(session, _troop_frame("start", initial, {}),
                                   "archer", "troop", "auto", {"evidence": []}, float("inf"))
    assert frame.screenshot_path.name == "settled.png"
    assert card["point"] == [431, 650]
    assert len(session.swipes) == 1


def test_spell_smoke_sanitized_bar_shows_old_view_then_moving_rebound():
    # Original 18-21 battle frames are ignored runtime data. These tracked
    # samples keep only the bottom toolbar at native recognition scale.
    import cv2
    from autococ.images import read_frame
    from autococ.ocr import filter_ocr_results
    from autococ.vision import ScreenshotRecognizer
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.600.7"
    frames = []
    for number in (18, 19, 20, 21):
        path = ROOT / "tests/fixtures" / f"spell_smoke1_bar_{number:02d}.png"
        image = read_frame(path, cv2.IMREAD_COLOR)
        assert image.shape == (720, 1280, 3)
        assert not image[:585].any()  # Opponent village and account data removed.
        texts = filter_ocr_results(recognizer.provider.recognize(path),
                                   recognizer.ocr_config.confidence_threshold)
        battle = recognizer._battle_observation(path, texts, "enemy_village")
        frames.append(SceneSnapshot("enemy_village", .95, path, {"battle": battle}))
    views = [_battle_bar_viewport(frame) for frame in frames]
    assert _same_battle_bar_view(views[0], views[1])
    assert not _same_battle_bar_view(views[1], views[2])
    assert not _same_battle_bar_view(views[2], views[3])
    assert any(card.get("unit_id") == "totem_spell" for card in frames[2].observations["battle"]["slots"])


def test_battle_bar_scans_at_most_three_times_each_direction():
    frames = []
    for page in range(1, 7):
        cards = [_troop("barbarian", (100 + 16 * page, 650)),
                 _troop("wall_breaker", (200 + 16 * page, 650))]
        frames.extend(_troop_frame(f"page-{page}-{sample}", cards, {}) for sample in (1, 2))
    session = _Session(frames)
    with pytest.raises(FlowError, match="not identified in bounded battle bar search"):
        _find_named_card(session, _troop_frame("start", [_troop("barbarian", (100, 650)),
                                                 _troop("wall_breaker", (200, 650))], {}),
                         "archer", "troop", "auto", {"evidence": []}, float("inf"))
    assert [direction for _, direction, _ in session.swipes] == ["left"] * 3 + ["right"] * 3


def test_battle_bar_unknown_scene_stops_before_another_swipe():
    session = _Session([SceneSnapshot("unknown", 0, Path(f"unknown-{i}.png"), {}) for i in range(6)])
    with pytest.raises(FlowError, match="did not stabilize"):
        _find_named_card(session, _troop_frame("start", [_troop("barbarian", (100, 650))], {}),
                         "archer", "troop", "auto", {"evidence": []}, float("inf"))
    assert len(session.swipes) == 1
    assert session.taps == []


def test_battle_bar_stop_after_swipe_precedes_reobservation():
    session = _Session([])
    session.check_deadline = lambda: (_ for _ in ()).throw(StopRequested()) if session.swipes else None
    with pytest.raises(StopRequested):
        _find_named_card(session, _troop_frame("start", [_troop("barbarian", (100, 650))], {}),
                         "archer", "troop", "auto", {"evidence": []}, float("inf"))
    assert len(session.swipes) == 1
    assert session.taps == []


def test_captured_card_rebinds_named_unit_across_small_border_jitter():
    prior = dict(_troop("archer", (350, 651)), count=1)
    own = dict(_troop("archer", (349, 650)), count=1, confidence=.99)
    clan = dict(_troop("archer", (350, 651)), count=1,
                source="clan_reinforcement", confidence=.99)
    frame = _troop_frame("jitter", [own, clan], {})
    assert _match_card(frame, "*", "troop", prior=prior) is own
    own["confidence"] = .89
    with pytest.raises(FlowError, match="Expected one identified"):
        _match_card(frame, "*", "troop", prior=prior)


def test_anonymous_captured_card_requires_fresh_portrait_and_count():
    prior = dict(_troop("archer", (350, 651)), unit_id=None, confidence=None, count=1)
    current = dict(prior, point=[349, 650], evidence={"portrait_score": .97,
        "count": {"frame": "anonymous.png", "count": 1, "confidence": .96}})
    frame = _troop_frame("anonymous", [current], {})
    assert _match_card(frame, "*", "troop", prior=prior) is current
    current["evidence"]["count"]["frame"] = "old.png"
    with pytest.raises(FlowError, match="Expected one identified"):
        _match_card(frame, "*", "troop", prior=prior)


def test_smoke4_archer_rebinds_from_real_fresh_frame():
    folder = ROOT / "reports/daily-smoke-4/20260926-170821-024736-71e481cd/frames"
    if not folder.is_dir():
        pytest.skip("Optional smoke4 frames are absent")
    from autococ.vision import ScreenshotRecognizer
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.600.7"
    before = recognizer.recognize(next(folder.glob("00031-*.png")))
    after = recognizer.recognize(next(folder.glob("00061-*.png")))
    prior = next(card for card in before.observations["battle"]["slots"]
                 if card.get("unit_id") == "archer")
    matched = _match_card(after, "*", "troop", prior=prior)
    assert matched["unit_id"] == "archer"
    assert matched["source"] == "army"
    assert matched["count"] == 1
    assert matched["point"] == [349, 650]


def test_first_second_edges_are_both_observed_western_flanks():
    terrain = [{"edge": edge, "point": point} for edge, points in
               ((0, ([200, 100], [220, 200])), (1, ([200, 300], [220, 400])))
               for point in points]
    assert _points(terrain, "first", 1) == [[210, 150]]
    assert _points(terrain, "second", 1) == [[210, 350]]
    assert _points(terrain, "two", 2) == [[210, 150], [210, 350]]


def test_siege_deployment_requires_selected_and_deployed_evidence(monkeypatch):
    import autococ.battle_vision as battle_vision
    import autococ.hero_state as hero_state
    monkeypatch.setattr(battle_vision, "selected_card_bbox", lambda *a, **kw: [60, 580, 155, 710])
    monkeypatch.setattr(hero_state, "recognize_siege_state",
                        lambda path, box, **kw: {"deployed": Path(path).stem == "siege-deployed"})
    siege = {"unit_id": "wall_wrecker", "kind": "siege", "source": "army", "count": None,
             "confidence": .99, "point": [100, 650], "bbox": [60, 580, 155, 710]}
    scout = _troop_frame("siege-start", [siege], {}, terrain=True)
    recipe = ArmyRecipe((ArmyRequirement("wall_wrecker", 1),))
    definition = StrategyDefinition("siege", "siege", recipe,
                                    (StrategyStep("deploy_siege", "wall_wrecker", edge="first"),))
    session = _Session([_troop_frame("siege-selected", [siege], {}),
                        _troop_frame("siege-deployed", [siege], {})])
    receipt = execute_strategy(session, scout, definition).observations["deployment"]
    assert receipt["siege_deployed"] == receipt["offensive_actions"] == 1
    assert receipt["actions"][0]["verified"] is True
    assert len(session.taps) == 2


def test_relative_spell_uses_verified_terrain_not_building(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["count"])
    scout = _frame("relative-start", count=2)
    scout.observations["buildings"] = []
    scout.observations["battle"]["slots"][0]["unit_id"] = "rage_spell"
    scout.observations["terrain"] = [{"edge": edge, "point": point,
        "frame": "relative-start.png", "verified": True} for edge, points in
        ((0, ([250, 150], [250, 250])), (1, ([250, 350], [250, 450]))) for point in points]
    recipe = ArmyRecipe((ArmyRequirement("rage_spell", 2),))
    definition = StrategyDefinition("rage", "rage", recipe,
                                    (StrategyStep("cast_spell", "rage_spell", 2,
                                                  target="relative:center"),))
    frames = [_frame("selected-one", count=2), _frame("consumed-one", count=1),
              _frame("after-one", count=1), _frame("selected-two", count=1),
              _frame("consumed-two", count=0), _frame("after-two", count=0)]
    for frame in frames:
        frame.observations["buildings"] = []
        frame.observations["battle"]["slots"][0]["unit_id"] = "rage_spell"
    session = _Session(frames)
    receipt = execute_strategy(session, scout, definition).observations["deployment"]
    assert receipt["spells_used"] == 2
    assert session.taps[1][1] == session.taps[3][1] == (250, 300)


def test_wait_for_hero_ready_observes_without_input(monkeypatch):
    import autococ.hero_state as hero_state
    monkeypatch.setattr(hero_state, "recognize_hero_state",
                        lambda path, box, **kw: {"deployed": True,
                                                 "ability_ready": Path(path).stem == "ready"})
    hero = _hero_frame("not-ready")
    session = _Session([_hero_frame("ready")])
    receipt = {"evidence": [], "actions": [], "hero_states": []}
    final = _wait_step(session, hero, StrategyStep("wait", "grand_warden",
                       target="hero_ready", timeout_sec=2), receipt)
    assert final.screenshot_path.name == "ready.png"
    assert session.taps == []
    assert receipt["actions"][0]["verified"] is True


def test_future_unsupported_target_rejected_before_prior_deployment():
    troop = _troop("barbarian", (100, 650))
    spell = {"unit_id": "lightning_spell", "kind": "spell", "source": "army",
             "count": 1, "confidence": .99, "point": [200, 650], "bbox": [160, 600, 240, 710]}
    scout = _troop_frame("preflight", [troop, spell], {"barbarian": 1,
                              "lightning_spell": 1}, terrain=True)
    definition = StrategyDefinition("future", "future", ArmyRecipe((
        ArmyRequirement("barbarian", 1), ArmyRequirement("lightning_spell", 1))),
        (StrategyStep("deploy_troop", "barbarian"),
         StrategyStep("cast_spell", "lightning_spell", target="spell_factory")))
    session = _Session([])
    with pytest.raises(DeploymentError):
        execute_strategy(session, scout, definition)
    assert session.taps == []


def test_siege_selection_unknown_does_not_attempt_placement(monkeypatch):
    import autococ.battle_vision as battle_vision
    monkeypatch.setattr(battle_vision, "selected_card_bbox", lambda *a, **kw: None)
    siege = {"unit_id": "wall_wrecker", "kind": "siege", "source": "army", "count": None,
             "confidence": .99, "point": [100, 650], "bbox": [60, 580, 155, 710]}
    scout = _troop_frame("siege-start", [siege], {}, terrain=True)
    definition = StrategyDefinition("siege", "siege", ArmyRecipe((ArmyRequirement("wall_wrecker", 1),)),
                                    (StrategyStep("deploy_siege", "wall_wrecker"),))
    session = _Session([_troop_frame(f"unselected-{i}", [siege], {}) for i in range(3)])
    with pytest.raises(DeploymentError) as exc:
        execute_strategy(session, scout, definition)
    assert len(session.taps) == 1
    assert exc.value.partial_receipt["siege_deployed"] == 0


def test_wait_delay_reobserves_without_taps():
    session = _Session([_hero_frame("after-delay")])
    receipt = {"evidence": [], "actions": [], "hero_states": []}
    result = _wait_step(session, _hero_frame("before-delay"),
                        StrategyStep("wait", target="battle", timeout_sec=2, delay_sec=.01), receipt)
    assert result.screenshot_path.name == "after-delay.png"
    assert receipt["actions"][0]["delay_sec"] == .01
    assert session.taps == []


def test_planner_cannot_exceed_recipe_budget(tmp_path):
    planner = tmp_path / "planner.py"
    planner.write_text('def plan(context):\n    return [{"action": "deploy_troop", '
                       '"unit_id": "barbarian", "count": 2}]\n', encoding="utf-8")
    definition = StrategyDefinition("budget", "budget", ArmyRecipe((ArmyRequirement("barbarian", 1),)),
        (StrategyStep("deploy_troop", "barbarian"),), planner=planner)
    with pytest.raises(ConfigError, match="exceeds army recipe"):
        planned_steps(definition, {})


def test_east_edge_is_rejected_as_misleading_name(tmp_path):
    strategy = tmp_path / "east.toml"
    strategy.write_text('id="east"\nlabel="east"\narmy_mode="captured"\n'
                        '[[steps]]\naction="deploy_troop"\nunit_id="*"\ncount="all"\nedge="east"\n',
                        encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid edge"):
        load_strategy(strategy)


def test_surrender_popup_needs_central_prompt_and_unique_confirm():
    popup = SceneSnapshot("popup", .95, Path("popup.png"), {
        "ocr": [{"text": "确定要结束战斗吗？", "confidence": .97,
                 "bbox": [430, 250, 780, 300]}],
        "buttons": [{"name": "confirm", "point": [700, 500]}],
    })
    assert _end_battle_confirmation(popup) == (300, 200, 980, 650)
    generic = SceneSnapshot("popup", .95, Path("generic.png"), {
        "ocr": [{"text": "确定领取奖励吗？", "confidence": .97,
                 "bbox": [430, 250, 780, 300]}],
        "buttons": [{"name": "confirm", "point": [700, 500]}],
    })
    with pytest.raises(Exception, match="no positive surrender prompt"):
        _end_battle_confirmation(generic)
    outside = SceneSnapshot("popup", .95, Path("outside.png"), {
        "ocr": [{"text": "确定要结束战斗吗？", "confidence": .97,
                 "bbox": [0, 500, 180, 590]}],
        "buttons": [{"name": "confirm", "point": [700, 500]}],
    })
    with pytest.raises(Exception, match="no positive surrender prompt"):
        _end_battle_confirmation(outside)


def test_unrelated_popup_after_end_battle_is_not_confirmed(monkeypatch):
    monkeypatch.setattr("autococ.strategy_execution._remaining",
                        lambda frame, card, recognizer=None: frame.observations["count"])
    strategy = load_strategy(ROOT / "strategies/lightning_snipe.toml")
    popup = SceneSnapshot("popup", .95, Path("reward-popup.png"), {
        "ocr": [{"text": "确认领取奖励吗？", "confidence": .98,
                 "bbox": [440, 250, 780, 300]}],
        "buttons": [{"name": "confirm", "point": [700, 500]}],
    })
    session = _Session([_frame("selected"), _frame("consumed", count=1),
                        _frame("destroyed", count=1, building_state="destroyed"), popup])
    with pytest.raises(DeploymentError, match="no positive surrender prompt"):
        execute_strategy(session, _frame("scout"), strategy)
    assert session.clicks == ["end_battle"]
