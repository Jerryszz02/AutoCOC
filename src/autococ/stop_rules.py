"""Stop-rule tracking for flow execution."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .config import StopConfig


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str


@dataclass
class StopController:
    config: StopConfig
    started_monotonic: float = 0.0
    attempts: int = 0
    failures: int = 0

    def __post_init__(self) -> None:
        if self.started_monotonic == 0.0:
            self.started_monotonic = time.monotonic()

    def record_success(self) -> None:
        self.attempts += 1

    def record_failure(self) -> None:
        self.attempts += 1
        self.failures += 1

    def check(self) -> StopDecision:
        if self.attempts >= self.config.max_runs:
            return StopDecision(True, f"max_runs reached: {self.config.max_runs}")
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed >= self.config.max_duration_sec:
            return StopDecision(True, f"max_duration_sec reached: {self.config.max_duration_sec}")
        if self.failures >= self.config.max_failures:
            return StopDecision(True, f"max_failures reached: {self.config.max_failures}")
        return StopDecision(False, "")
