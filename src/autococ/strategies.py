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


class TwoEdgeSplitStrategy:
    """Fixed, count-preserving two-edge deployment order for the sampled roster."""

    def build_plan(self, context: BattleContext) -> DeploymentPlan:
        troops = {card["unit_id"]: card for card in context.troops}
        heroes = {card["unit_id"]: card for card in context.heroes}
        base = {"electro_dragon", "dragon", "baby_dragon", "balloon"}
        expected_heroes = {"grand_warden", "archer_queen", "dragon_duke", "minion_prince"}
        if (len(troops) != len(context.troops) or len(heroes) != len(context.heroes) or
                set(troops) not in (base, base | {"archer"}) or
                set(heroes) != expected_heroes or
                any(type(card.get("count")) is not int or card["count"] <= 0
                    for card in context.troops)):
            raise FlowError("Two-edge split requires the identified current line roster")

        def place(unit_id: str, first_count: int) -> Placement:
            card = troops[unit_id]
            count = card["count"]
            if not 0 <= first_count <= count:
                raise FlowError("Two-edge split exceeds a verified troop count")
            return Placement(card, spread_on_edge(context.terrain, 0, first_count) +
                             spread_on_edge(context.terrain, 1, count - first_count))

        placements = [place("baby_dragon", troops["baby_dragon"]["count"]),
                      place("dragon", 0),
                      place("balloon", (troops["balloon"]["count"] + 1) // 2),
                      place("electro_dragon", (2 * troops["electro_dragon"]["count"] + 2) // 3)]
        if "archer" in troops:
            placements.append(place("archer", (troops["archer"]["count"] + 1) // 2))
        hero_edges = (("grand_warden", 0), ("archer_queen", 0),
                      ("dragon_duke", 1), ("minion_prince", 1))
        hero_placements = tuple(Placement(heroes[unit_id],
                                          spread_on_edge(context.terrain, edge, 1))
                                for unit_id, edge in hero_edges)
        return DeploymentPlan(tuple(placements), hero_placements)


class TwoEdgeFurnaceStrategy:
    """Current mixed roster: separate edge probes before the frozen remainder."""

    def build_plan(self, context: BattleContext) -> DeploymentPlan:
        from .mixed_roster import MIXED_CORE_COUNTS, mixed_archer_count

        archers = mixed_archer_count(context.troops)
        troops = {card["unit_id"]: card for card in context.troops}
        heroes = {card["unit_id"]: card for card in context.heroes}
        hero_ids = {"grand_warden", "archer_queen", "dragon_duke", "minion_prince"}
        if len(heroes) != len(context.heroes) or set(heroes) != hero_ids:
            raise FlowError("Mixed furnace probes require the exact identified current roster")

        def place(unit_id: str, first_count: int) -> Placement:
            count = archers if unit_id == "archer" else MIXED_CORE_COUNTS[unit_id]
            return Placement(troops[unit_id], spread_on_edge(context.terrain, 0, first_count) +
                             spread_on_edge(context.terrain, 1, count - first_count))

        placements = (place("balloon", 1), place("dragon", 0),
                      place("furnace", 1), place("electro_dragon", 6))
        if archers:
            placements += (place("archer", (archers + 1) // 2),)
        hero_edges = (("grand_warden", 0), ("archer_queen", 0),
                      ("dragon_duke", 1), ("minion_prince", 1))
        hero_placements = tuple(Placement(heroes[unit_id], spread_on_edge(context.terrain, edge, 1))
                                for unit_id, edge in hero_edges)
        return DeploymentPlan(placements, hero_placements)


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
