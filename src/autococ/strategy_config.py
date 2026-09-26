"""Validated, declarative battle strategies.

Files are local configuration. Optional Python planners may return steps, but
never receive a session or the game input interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import importlib.util
import re
import tomllib

from .errors import ConfigError
from .unit_catalog import ArmyRecipe, ArmyRequirement, get_unit


@dataclass(frozen=True)
class StrategyStep:
    action: str
    unit_id: str = ""
    count: int | str = 1
    edge: str = "first"
    target: str = ""
    timeout_sec: int = 15
    source: str = "auto"
    delay_sec: float = 0.0


@dataclass(frozen=True)
class StrategyDefinition:
    id: str
    label: str
    recipe: ArmyRecipe | None
    steps: tuple[StrategyStep, ...]
    army_mode: str = "recipe"
    planner: Path | None = None


_ACTIONS = {"deploy_troop", "deploy_hero", "deploy_siege", "activate_ability", "cast_spell", "wait", "end_battle"}
_STEP_KEYS = {"action", "unit_id", "count", "edge", "target", "timeout_sec", "source", "delay_sec"}
_EDGES = {"first", "second", "two", "midpoint", "west"}
_SOURCES = {"auto", "army", "event", "clan_reinforcement"}
_SCENES = {"enemy_village", "battle", "settlement"}
_RELATIVE = {"relative:first_midpoint", "relative:second_midpoint", "relative:center"}
_BUILDINGS = {"spell_factory", "air_defense", "objective"}


def _keys(value: dict, allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {sorted(unknown)}")


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{label} must be a nonempty string")
    return value.strip()


def _step(raw: dict[str, Any], index: int) -> StrategyStep:
    if not isinstance(raw, dict):
        raise ConfigError(f"step {index} must be a table")
    _keys(raw, _STEP_KEYS, f"step {index}")
    action = _text(raw.get("action"), f"step {index}.action")
    if action not in _ACTIONS:
        raise ConfigError(f"step {index} has unsupported action {action!r}")
    unit_id = _text(raw.get("unit_id", ""), f"step {index}.unit_id", allow_empty=True)
    target = _text(raw.get("target", ""), f"step {index}.target", allow_empty=True)
    edge = _text(raw.get("edge", "first"), f"step {index}.edge")
    count = raw.get("count", 1)
    timeout = raw.get("timeout_sec", 15)
    source = _text(raw.get("source", "auto"), f"step {index}.source")
    delay = raw.get("delay_sec", 0.0)
    if edge not in _EDGES or type(timeout) is not int or not 1 <= timeout <= 180:
        raise ConfigError(f"step {index} has invalid edge or timeout")
    if source not in _SOURCES or type(delay) not in {int, float} or not 0 <= delay <= 60:
        raise ConfigError(f"step {index} has invalid source or delay")
    if action in {"deploy_troop", "cast_spell"}:
        if not unit_id or not (count == "all" and action == "deploy_troop" or
                               type(count) is int and 1 <= count <= 100):
            raise ConfigError(f"step {index} requires a unit and a bounded count")
        if action == "cast_spell" and not target:
            raise ConfigError(f"step {index} requires a spell target")
    elif action in {"deploy_hero", "deploy_siege", "activate_ability"}:
        if not unit_id or type(count) is not int or count != 1:
            raise ConfigError(f"step {index} requires one named hero or siege unit")
    elif action == "wait":
        if target not in (*_SCENES, "", "hero_ready") or target == "" and delay == 0:
            raise ConfigError(f"step {index} wait needs a known scene, hero_ready, or delay")
        if target == "hero_ready" and not unit_id:
            raise ConfigError(f"step {index} hero_ready needs a named hero")
    elif action == "end_battle" and target not in {"", "target_destroyed"}:
        raise ConfigError(f"step {index} end_battle condition is unsupported")
    if action not in {"deploy_troop", "cast_spell"} and (type(count) is not int or count != 1):
        raise ConfigError(f"step {index} count applies only to troops or spells")
    if action not in {"deploy_troop", "deploy_hero", "deploy_siege"} and "edge" in raw:
        raise ConfigError(f"step {index} edge applies only to deployments")
    if action != "wait" and "delay_sec" in raw:
        raise ConfigError(f"step {index} delay applies only to wait")
    if action not in {"deploy_troop", "deploy_hero", "deploy_siege", "activate_ability", "cast_spell", "wait"} and "source" in raw:
        raise ConfigError(f"step {index} source is not applicable")
    if action in {"deploy_troop", "deploy_hero", "deploy_siege", "activate_ability", "cast_spell"} or target == "hero_ready":
        if unit_id != "*":
            unit = get_unit(unit_id)
            expected_kind = {"deploy_troop": "troop", "deploy_hero": "hero", "deploy_siege": "siege",
                             "activate_ability": "hero", "cast_spell": "spell", "wait": "hero"}[action]
            if unit is None or unit.kind != expected_kind:
                raise ConfigError(f"step {index} unit_id is not a catalogued {expected_kind}")
        elif action != "deploy_troop":
            raise ConfigError(f"step {index} wildcard is only valid for troop deployment")
    if action == "cast_spell" and target.startswith("relative:") and target not in _RELATIVE:
        raise ConfigError(f"step {index} has unsupported relative target")
    if action == "cast_spell" and target not in _RELATIVE | _BUILDINGS and not re.fullmatch(
            r"(?:spell_factory|air_defense):[0-9]+:[0-9]+", target):
        raise ConfigError(f"step {index} has unsupported building target")
    return StrategyStep(action, unit_id, count, "first" if edge == "west" else edge,
                        target, timeout, source, float(delay))


def _validate_plan_steps(steps: tuple[StrategyStep, ...], mode: str,
                         requirements: tuple[ArmyRequirement, ...]) -> None:
    available = {item.unit_id: item.count for item in requirements}
    demanded: dict[str, int] = {}
    for index, step in enumerate(steps):
        if step.action == "end_battle" and index != len(steps) - 1:
            raise ConfigError("end_battle must be the final action")
        if step.unit_id == "*" and mode != "captured":
            raise ConfigError("Wildcard troop deployment requires captured-army mode")
        if mode == "recipe" and step.action in {
                "deploy_troop", "deploy_hero", "deploy_siege", "cast_spell", "activate_ability"}:
            if step.unit_id not in available:
                raise ConfigError(f"step {index} uses {step.unit_id} absent from army recipe")
            if step.action in {"deploy_troop", "deploy_hero", "deploy_siege", "cast_spell"} and type(step.count) is int:
                demanded[step.unit_id] = demanded.get(step.unit_id, 0) + step.count
        if mode == "recipe" and step.action == "wait" and step.target == "hero_ready" and step.unit_id not in available:
            raise ConfigError(f"step {index} waits for a hero absent from the army recipe")
    if any(count > available[unit_id] for unit_id, count in demanded.items()):
        raise ConfigError("Strategy action count exceeds army recipe")
    if any(step.action == "end_battle" and step.target == "target_destroyed" for step in steps) and not any(
            step.action == "cast_spell" and not step.target.startswith("relative:") for step in steps):
        raise ConfigError("Target destruction ending needs a preceding building spell action")


def load_strategy(path: str | Path) -> StrategyDefinition:
    source = Path(path).resolve()
    try:
        with source.open("rb") as file:
            raw = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot load strategy {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Strategy must be a TOML table")
    _keys(raw, {"id", "label", "army_mode", "planner", "army", "steps"}, "strategy")
    name = _text(raw.get("id"), "strategy.id")
    label = _text(raw.get("label"), "strategy.label")
    mode = raw.get("army_mode", "recipe")
    if mode not in {"recipe", "captured"}:
        raise ConfigError("army_mode must be recipe or captured")
    army = raw.get("army", {})
    if not isinstance(army, dict):
        raise ConfigError("army must be a table")
    _keys(army, {"units"}, "army")
    units = army.get("units", [])
    if not isinstance(units, list):
        raise ConfigError("army.units must be a list")
    requirements = []
    seen = set()
    for index, item in enumerate(units):
        if not isinstance(item, dict):
            raise ConfigError(f"army unit {index} must be a table")
        _keys(item, {"unit_id", "count", "optional"}, f"army unit {index}")
        unit_id = _text(item.get("unit_id"), f"army unit {index}.unit_id")
        count = item.get("count")
        optional = item.get("optional", False)
        if unit_id in seen or type(count) is not int or count <= 0 or type(optional) is not bool:
            raise ConfigError(f"army unit {index} has duplicate or invalid quantity")
        seen.add(unit_id)
        try:
            requirements.append(ArmyRequirement(unit_id, count, optional))
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    if mode == "captured" and requirements or mode == "recipe" and not requirements:
        raise ConfigError("captured army needs no recipe; recipe mode needs units")
    raw_steps = raw.get("steps", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ConfigError("strategy.steps must contain at least one action")
    steps = tuple(_step(item, index) for index, item in enumerate(raw_steps))
    _validate_plan_steps(steps, mode, tuple(requirements))
    planner_text = raw.get("planner", "")
    if not isinstance(planner_text, str):
        raise ConfigError("planner must be a local Python path")
    planner = (source.parent / planner_text).resolve() if planner_text else None
    if planner and (not planner.is_file() or planner.suffix != ".py"):
        raise ConfigError("planner must resolve to a local Python file")
    return StrategyDefinition(name, label,
                              ArmyRecipe(tuple(requirements)) if requirements else None,
                              steps, mode, planner)


def _validate_planner_capabilities(definition: StrategyDefinition,
                                   steps: tuple[StrategyStep, ...], objective_target: str) -> None:
    """Static steps declare input kinds and map capabilities needed before search."""
    declared_actions = {step.action for step in definition.steps}

    def building_type(target: str) -> str:
        return objective_target if target == "objective" and objective_target else target.split(":", 1)[0]

    declared_buildings = {building_type(step.target) for step in definition.steps
                          if step.action == "cast_spell" and not step.target.startswith("relative:")}
    declared_relative = any(step.action == "cast_spell" and step.target.startswith("relative:")
                            for step in definition.steps)
    for index, step in enumerate(steps):
        # Waiting only observes a known state. Every other action may issue
        # input, so its kind must already be visible in the TOML declaration.
        if step.action != "wait" and step.action not in declared_actions:
            raise ConfigError(f"Planner step {index} adds undeclared action {step.action}")
        if step.action != "cast_spell":
            continue
        if step.target.startswith("relative:"):
            if not declared_relative:
                raise ConfigError(f"Planner step {index} adds undeclared relative spell target")
        elif building_type(step.target) not in declared_buildings:
            raise ConfigError(f"Planner step {index} adds undeclared building target {step.target}")


def planned_steps(definition: StrategyDefinition, context: dict[str, object], *,
                  objective_target: str = "") -> tuple[StrategyStep, ...]:
    """An explicitly configured local planner can change actions, not execute them."""
    if definition.planner is None:
        return definition.steps
    spec = importlib.util.spec_from_file_location("autococ_local_strategy_planner", definition.planner)
    if spec is None or spec.loader is None:
        raise ConfigError("Local strategy planner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.plan(copy.deepcopy(context))
    if not isinstance(result, (tuple, list)) or not result or len(result) > 100:
        raise ConfigError("Local strategy planner must return 1-100 action tables")
    steps = tuple(_step(value, index) for index, value in enumerate(result))
    _validate_plan_steps(steps, definition.army_mode,
                         definition.recipe.units if definition.recipe else ())
    _validate_planner_capabilities(definition, steps, objective_target)
    return steps
