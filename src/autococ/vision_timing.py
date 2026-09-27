"""Evidence-preserving timing for optional in-battle visual actions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal


ActionStatus = Literal["completed", "verified", "input_sent", "failed", "rejected",
                       "fallback", "timeout", "cancelled"]
ThermalState = Literal["cold", "warm"]


@dataclass(frozen=True)
class ActionTiming:
    action_type: str
    thermal_state: ThermalState
    status: ActionStatus
    t_request: float
    t_pixels: float | None = None
    t_state: float | None = None
    t_decision: float | None = None
    t_input_done: float | None = None
    t_effect_verified: float | None = None

    def __post_init__(self) -> None:
        if self.t_request is None:
            raise ValueError("t_request is required for every attempt")
        if not isinstance(self.action_type, str) or not self.action_type.strip():
            raise ValueError("action_type must be a nonempty string")
        if self.thermal_state not in {"cold", "warm"}:
            raise ValueError("thermal_state must be cold or warm")
        if self.status not in {"completed", "verified", "input_sent", "failed", "rejected",
                               "fallback", "timeout", "cancelled"}:
            raise ValueError("Unknown action timing status")
        previous = None
        missing = False
        for name in ("t_request", "t_pixels", "t_state", "t_decision", "t_input_done", "t_effect_verified"):
            value = getattr(self, name)
            if value is None:
                missing = True
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative timestamp")
            if missing or (previous is not None and value < previous):
                raise ValueError("Action timing stages must be complete and ordered")
            previous = value
        if self.status in {"completed", "input_sent", "verified"} and self.t_input_done is None:
            raise ValueError("Completed input requires t_input_done")
        if self.status == "verified" and self.t_effect_verified is None:
            raise ValueError("Verified effect requires t_effect_verified")
        if self.status == "rejected" and self.t_input_done is not None:
            raise ValueError("Rejected action cannot have completed input")

    @property
    def input_latency_sec(self) -> float | None:
        return self.t_input_done - self.t_request if self.t_input_done is not None else None

    @property
    def effect_latency_sec(self) -> float | None:
        return self.t_effect_verified - self.t_request if self.t_effect_verified is not None else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(records: list[ActionTiming], target_sec: float) -> dict[str, object]:
    inputs = [record.input_latency_sec for record in records if record.input_latency_sec is not None]
    effects = [record.effect_latency_sec for record in records if record.effect_latency_sec is not None]
    over = sum(value > target_sec for value in inputs)
    on_time = sum(value <= target_sec for value in inputs)
    attempts = len(records)
    return {
        "attempts": attempts,
        "statuses": {status: sum(record.status == status for record in records)
                     for status in ("completed", "verified", "input_sent", "failed", "rejected",
                                    "fallback", "timeout", "cancelled")},
        "input_completed_count": len(inputs),
        "unknown_input_latency_count": attempts - len(inputs),
        "over_target_count": over,
        "over_target_ratio": over / attempts if attempts else None,
        "not_on_time_count": attempts - on_time,
        "not_on_time_ratio": (attempts - on_time) / attempts if attempts else None,
        "input_latency_sec": {"p50": _percentile(inputs, .5), "p95": _percentile(inputs, .95),
                              "max": max(inputs) if inputs else None},
        "effect_latency_sec": {"p50": _percentile(effects, .5), "p95": _percentile(effects, .95),
                               "max": max(effects) if effects else None},
    }


def summarize_action_timings(records: Iterable[ActionTiming], *, target_sec: float = .5) -> dict[str, object]:
    """Include every attempt in ratios; unknown durations never become fast samples."""
    if type(target_sec) not in (int, float) or not math.isfinite(target_sec) or target_sec <= 0:
        raise ValueError("target_sec must be finite and positive")
    records = list(records)
    if any(not isinstance(record, ActionTiming) for record in records):
        raise TypeError("records must contain ActionTiming instances")
    groups = {}
    for action_type in sorted({record.action_type for record in records}):
        selected = [record for record in records if record.action_type == action_type]
        groups[action_type] = {"all": _summary(selected, target_sec),
                               "cold": _summary([record for record in selected if record.thermal_state == "cold"],
                                                target_sec),
                               "warm": _summary([record for record in selected if record.thermal_state == "warm"],
                                                target_sec)}
    return {"target_sec": target_sec, "overall": _summary(records, target_sec),
            "by_action_type": groups}
