"""Observable resource thresholds and bounded battle target selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .config import BattleConfig
from .strategies import is_line_strategy


RESOURCE_PATTERN = re.compile(r"(?P<number>\d[\d.,]*)(?P<suffix>[kKmM]?)")


@dataclass(frozen=True)
class BattleTarget:
    gold: int | None = None
    elixir: int | None = None
    dark_elixir: int | None = None

    @property
    def gold_elixir_total(self) -> int | None:
        if any(type(value) is not int or value < 0 for value in (self.gold, self.elixir)):
            return None
        return self.gold + self.elixir


@dataclass(frozen=True)
class BattleScore:
    target: BattleTarget
    gold_elixir_total: int | None
    should_attack: bool
    can_search_next: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DeploymentStep:
    x: int
    y: int
    label: str
    delay_sec: float = 0.0


@dataclass(frozen=True)
class DeploymentPlan:
    reason: str
    steps: tuple[DeploymentStep, ...]


def score_target(target: BattleTarget, config: BattleConfig, *, searches: int = 1) -> BattleScore:
    """Evaluate the current candidate; searches counts observed candidates from one.

    The final allowed candidate may still be attacked when it meets the threshold.
    Exhausting the search budget never relaxes the threshold.
    """
    if type(searches) is not int or searches < 1:
        raise ValueError("searches must count observed candidates starting at 1")
    total = target.gold_elixir_total
    if searches > config.max_searches:
        return BattleScore(target, total, False, False, ("search limit exceeded",))
    if config.resource_filter is not None:
        rule = config.resource_filter
        rule.validate()
        if not rule.enabled:
            return BattleScore(target, total, True, False, ("resource filtering disabled",))
        checks = (("gold", target.gold, rule.min_gold), ("elixir", target.elixir, rule.min_elixir),
                  ("dark_elixir", target.dark_elixir, rule.min_dark_elixir),
                  ("gold_elixir", total, rule.min_total))
        reasons = []
        for name, amount, minimum in checks:
            if minimum is None:
                continue
            if type(amount) is not int or amount < 0:
                reasons.append(f"{name} unreadable")
            elif amount < minimum:
                reasons.append(f"{name}={amount}; below minimum={minimum}")
        accepted = not reasons
        if not accepted and searches == config.max_searches:
            reasons.append("search limit reached")
        return BattleScore(target, total, accepted, not accepted and searches < config.max_searches,
                           tuple(reasons or ("all resource floors met",)))
    if is_line_strategy(config.strategy):
        return BattleScore(target, total, True, False, ("first clear opponent for round count",))
    should_attack = total is not None and total >= config.min_expected_resources
    reason = "resource floor met" if should_attack else "resource floor not met"
    if total is None:
        reason = "gold or elixir unreadable"
    reasons = [reason, f"gold_elixir={total}; minimum={config.min_expected_resources}",
               f"dark_elixir={target.dark_elixir}; reported separately"]
    if searches == config.max_searches:
        reasons.append("search limit reached")
    return BattleScore(target, total, should_attack, not should_attack and searches < config.max_searches, tuple(reasons))


def parse_resource_number(text: str) -> int | None:
    normalized = text.strip().replace(" ", "")
    match = RESOURCE_PATTERN.fullmatch(normalized)
    if not match:
        return None
    number_text = _normalize_number_text(match.group("number"), bool(match.group("suffix")))
    try:
        number = float(number_text)
        suffix = match.group("suffix").lower()
        if suffix == "k":
            number *= 1_000
        elif suffix == "m":
            number *= 1_000_000
        return int(number)
    except (ValueError, OverflowError):
        return None


def _normalize_number_text(value: str, has_suffix: bool) -> str:
    if not has_suffix:
        return value.replace(",", "").replace(".", "")
    if "," not in value:
        return value
    parts = value.split(",")
    if len(parts) == 2 and len(parts[1]) != 3:
        return ".".join(parts)
    return "".join(parts)


def basic_deployment_plan(points: tuple[tuple[int, int], ...]) -> DeploymentPlan:
    steps = tuple(
        DeploymentStep(x=x, y=y, label=f"deploy-{index + 1}", delay_sec=0.2)
        for index, (x, y) in enumerate(points)
    )
    return DeploymentPlan(reason="basic edge deployment", steps=steps)
