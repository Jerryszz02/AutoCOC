"""Explicit action plans; task handlers own verified state transitions."""

from dataclasses import dataclass, field
from typing import Any, Literal

ActionKind = Literal["tap", "tap_xy", "swipe", "wait", "screenshot", "back", "stop"]


@dataclass(frozen=True)
class ActionStep:
    kind: ActionKind
    reason: str
    args: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionPlan:
    reason: str
    actions: tuple[ActionStep, ...]
    expected_next_scene: str | None = None
