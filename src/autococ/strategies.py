"""Pure deployment planners; game input and verification belong to the executor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .errors import FlowError


@dataclass(frozen=True)
class BattleContext:
    troops: tuple[dict, ...]
    heroes: tuple[dict, ...]
    terrain: tuple[dict, ...]
    spells: tuple[dict, ...] = ()
    buildings: tuple[dict, ...] = ()
    hero_states: tuple[dict, ...] = ()
    target_progress: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Placement:
    card: dict
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DeploymentPlan:
    troops: tuple[Placement, ...]
    heroes: tuple[Placement, ...] = ()


class BattleStrategy(Protocol):
    def build_plan(self, context: BattleContext) -> DeploymentPlan: ...


def spread_on_edge(terrain: tuple[dict, ...] | list[dict], edge: int, count: int) -> tuple[tuple[int, int], ...]:
    if type(count) is not int or count < 0:
        raise FlowError("Line deployment requires a non-negative troop count")
    points = sorted((item["point"] for item in terrain if item.get("edge") == edge), key=lambda p: p[1])
    if len(points) < 2 or points[0] == points[-1]:
        raise FlowError("Line deployment requires a distinct observed edge")
    start, end = points[0], points[-1]
    return tuple(tuple(round(start[axis] + (index / (count - 1) if count > 1 else .5)
                             * (end[axis] - start[axis])) for axis in (0, 1)) for index in range(count))


class TwoEdgeStrategy:
    def build_plan(self, context: BattleContext) -> DeploymentPlan:
        cards = sorted(context.troops, key=lambda card: (card.get("source") != "event", card["point"][0]))
        return DeploymentPlan(tuple(Placement(card,
            spread_on_edge(context.terrain, 0, (card["count"] + 1) // 2)
            + spread_on_edge(context.terrain, 1, card["count"] // 2)) for card in cards))


class EdragLineStrategy:
    def build_plan(self, context: BattleContext) -> DeploymentPlan:
        return DeploymentPlan(
            tuple(Placement(card, spread_on_edge(context.terrain, 0, card["count"]))
                  for card in sorted(context.troops, key=lambda card: card["point"][0])),
            tuple(Placement(card, spread_on_edge(context.terrain, 0, 1))
                  for card in sorted(context.heroes, key=lambda card: card["point"][0])))


@dataclass(frozen=True)
class StrategyRegistration:
    label: str
    planner: BattleStrategy | None


# Register new line planners here; configuration, CLI and GUI share this table.
# None retains the existing adaptive resource-mode executor.
STRATEGIES = {
    "two_edge": StrategyRegistration("活动兵两边划", TwoEdgeStrategy()),
    "edrag_line": StrategyRegistration("雷龙一字划跟英雄", EdragLineStrategy()),
    "verified": StrategyRegistration("原有资源模式", None),
}


def is_line_strategy(name: str) -> bool:
    return STRATEGIES[name].planner is not None
