"""Shared, versioned daily-task configuration used by the CLI and desktop."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from .errors import ConfigError

RESOURCES = ("gold", "elixir", "dark_elixir")
TASK_KINDS = ("collect", "request", "donate", "resources", "event", "clan_games")
BATTLE_KINDS = frozenset(("resources", "event", "clan_games"))
_DEFAULT_RESOURCE_FILTER = object()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ResourceFilter:
    enabled: bool = True
    min_gold: int | None = None
    min_elixir: int | None = None
    min_dark_elixir: int | None = None
    min_total: int | None = None

    def validate(self):
        if type(self.enabled) is not bool:
            raise ConfigError("resource_filter.enabled must be boolean")
        for key in ("min_gold", "min_elixir", "min_dark_elixir", "min_total"):
            if getattr(self, key) is not None:
                _integer(getattr(self, key), f"resource_filter.{key}")


@dataclass(frozen=True)
class GoalConfig:
    resource_targets: dict[str, int] = field(default_factory=dict)
    full_resources: tuple[str, ...] = ()
    adapter_path: str = ""
    target: int | None = None
    building_type: str = ""

    def validate(self):
        if not isinstance(self.resource_targets, dict) or set(self.resource_targets) - set(RESOURCES):
            raise ConfigError("goal.resource_targets supports gold, elixir and dark_elixir")
        for key, value in self.resource_targets.items():
            _integer(value, f"goal.resource_targets.{key}", 1)
        if not isinstance(self.full_resources, (tuple, list)) or any(x not in RESOURCES for x in self.full_resources):
            raise ConfigError("goal.full_resources contains an unknown resource")
        if len(set(self.full_resources)) != len(self.full_resources):
            raise ConfigError("goal.full_resources contains duplicates")
        if self.target is not None:
            _integer(self.target, "goal.target", 1)
        if not isinstance(self.adapter_path, str) or not isinstance(self.building_type, str):
            raise ConfigError("goal adapter_path and building_type must be strings")
        if self.building_type not in ("", "air_defense", "spell_factory"):
            raise ConfigError("Supported clan targets: air_defense, spell_factory")


@dataclass(frozen=True)
class TaskSpec:
    id: str
    kind: str
    enabled: bool = True
    strategy: str = "two_edge"
    strategy_file: str = ""
    max_battles: int = 10
    max_duration_sec: int = 1800
    max_searches: int = 30
    resource_filter: ResourceFilter = field(default=_DEFAULT_RESOURCE_FILTER)
    goal: GoalConfig = field(default_factory=GoalConfig)

    def __post_init__(self):
        if self.resource_filter is _DEFAULT_RESOURCE_FILTER:
            default = (ResourceFilter(min_total=300000) if self.kind == "resources"
                       else ResourceFilter(enabled=False))
            object.__setattr__(self, "resource_filter", default)

    def validate(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.id):
            raise ConfigError("task.id must be a short lowercase identifier")
        if self.kind not in TASK_KINDS or type(self.enabled) is not bool:
            raise ConfigError("Invalid daily task kind/enabled")
        if not isinstance(self.strategy, str) or not isinstance(self.strategy_file, str):
            raise ConfigError("task strategy and strategy_file must be strings")
        if self.strategy not in ("two_edge", "edrag_line", "verified"):
            raise ConfigError("Unknown legacy strategy; use strategy_file for a TOML strategy")
        for key in ("max_battles", "max_duration_sec", "max_searches"):
            _integer(getattr(self, key), f"task.{key}", 1)
        if not isinstance(self.resource_filter, ResourceFilter) or not isinstance(self.goal, GoalConfig):
            raise ConfigError("task resource_filter and goal must use their shared definitions")
        self.resource_filter.validate()
        self.goal.validate()


@dataclass(frozen=True)
class RoutineConfig:
    tasks: tuple[TaskSpec, ...]
    maintenance_interval_sec: int = 0
    version: int = 1

    def validate(self):
        if type(self.version) is not int or self.version != 1:
            raise ConfigError("Unsupported routine version")
        _integer(self.maintenance_interval_sec, "maintenance_interval_sec")
        if not isinstance(self.tasks, (tuple, list)) or not self.tasks:
            raise ConfigError("A routine requires tasks")
        seen = set()
        for task in self.tasks:
            if not isinstance(task, TaskSpec):
                raise ConfigError("Invalid task specification")
            task.validate()
            if task.id in seen:
                raise ConfigError(f"Duplicate task id: {task.id}")
            seen.add(task.id)


def _construct(cls, values):
    if not isinstance(values, dict):
        raise ConfigError(f"{cls.__name__} must be a table")
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid {cls.__name__}: {exc}") from exc


def routine_from_dict(payload: dict) -> RoutineConfig:
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ConfigError("routine.tasks must be an array of task tables")
    values = dict(payload)
    tasks = []
    for item in values.pop("tasks"):
        if not isinstance(item, dict):
            raise ConfigError("Each routine task must be a table")
        data = dict(item)
        if "resource_filter" in data:
            data["resource_filter"] = _construct(ResourceFilter, data["resource_filter"])
        if not isinstance(data.get("goal", {}), dict):
            raise ConfigError("task.goal must be a table")
        goal = dict(data.get("goal", {}))
        if isinstance(goal.get("full_resources"), list):
            goal["full_resources"] = tuple(goal["full_resources"])
        data["goal"] = _construct(GoalConfig, goal)
        tasks.append(_construct(TaskSpec, data))
    result = _construct(RoutineConfig, {**values, "tasks": tuple(tasks)})
    result.validate()
    return result


def routine_to_dict(routine: RoutineConfig) -> dict:
    routine.validate()
    # JSON-compatible containers also make settings independent of dataclass internals.
    import json
    return json.loads(json.dumps(asdict(routine)))


def default_routine(config) -> RoutineConfig:
    if getattr(config, "routine", None) is not None:
        return config.routine
    from .strategies import is_line_strategy
    profile = config.profiles.get("core-loop", next(iter(config.profiles.values())))
    tasks = []
    for kind in TASK_KINDS:
        enabled = ("battle" if kind == "resources" else kind) in profile.enabled_tasks
        tasks.append(TaskSpec(kind, kind, enabled=enabled, strategy=config.battle.strategy,
            strategy_file=config.battle.strategy_file if kind == "resources" else "",
            max_battles=config.stop.max_runs, max_duration_sec=config.stop.max_duration_sec,
            max_searches=config.battle.max_searches,
            resource_filter=(config.battle.resource_filter if kind == "resources" and config.battle.resource_filter is not None
                else ResourceFilter(enabled=kind == "resources" and not is_line_strategy(config.battle.strategy),
                                    min_total=config.battle.min_expected_resources))))
    result = RoutineConfig(tuple(tasks))
    result.validate()
    return result
