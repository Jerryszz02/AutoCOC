"""Collect visible village resources and verify their inventory effects."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import time
from typing import Literal, TypedDict

from .errors import AutoCOCError, FlowError
from .reporting import TaskResult
from .scene import SCENE_VILLAGE, SceneSnapshot
from .session import GameSession


RESOURCES = ("gold", "elixir", "dark_elixir")
MAX_TARGETS = 30
VERIFICATION_OBSERVATIONS = 3


class CollectionTarget(TypedDict):
    resource: str
    point: tuple[int, int]
    bbox: tuple[int, int, int, int]
    confidence: float
    template: str


def collect_resources(session: GameSession) -> TaskResult:
    """Each credited collection needs a vanished bubble and an inventory increase."""
    started_at = datetime.now()
    started = time.monotonic()
    evidence: list[Path] = []
    verified: list[dict[str, object]] = []
    collected = {resource: 0 for resource in RESOURCES}
    attempted_targets = 0
    metrics: dict[str, object] = {
        "resources_before": None, "resources_after": None,
        "verified_collections": verified,
    }

    def finish(status: Literal["succeeded", "skipped", "failed"], reason: str) -> TaskResult:
        metrics["attempted_targets"] = attempted_targets
        metrics.update({f"collected_{resource}": value for resource, value in collected.items()})
        return TaskResult("collect", status, reason, started_at=started_at,
                          elapsed_sec=time.monotonic() - started, evidence=evidence, metrics=metrics)

    def observe(label: str) -> tuple[SceneSnapshot, dict[str, int], list[CollectionTarget]]:
        snapshot = session.observe(label)
        evidence.append(snapshot.screenshot_path)
        metrics["resources_after"] = snapshot.observations.get("resources")
        if len(evidence) == 1:
            metrics["resources_before"] = snapshot.observations.get("resources")
        inventory = _inventory(snapshot)
        return snapshot, inventory, _targets(snapshot)

    try:
        current, inventory, targets = observe("collect-before")
        while True:
            session.check_deadline()
            if not targets:
                return finish("succeeded" if verified else "skipped",
                              "collected_resources" if verified else "no_collectibles")
            available = [target for target in targets if not _storage_full(current, target["resource"], inventory)]
            if not available:
                metrics["storage_full_resources"] = sorted({target["resource"] for target in targets})
                return finish("succeeded" if verified else "skipped",
                              "collected_resources_remaining_storage_full" if verified else "storage_full")
            if attempted_targets >= MAX_TARGETS:
                return finish("failed", "collection_target_limit")

            target = available[0]
            before, before_inventory = current, inventory
            attempted_targets += 1
            session.tap(before, target["point"], reason=f"Collect {target['resource']} bubble: {target['template']}")
            for observation in range(VERIFICATION_OBSERVATIONS):
                time.sleep(min(session.config.runtime.poll_interval_sec, 1.0))
                current, inventory, targets = observe("collect-verify")
                if current.screenshot_path == before.screenshot_path:
                    raise FlowError("Collection verification reused its before screenshot")
                if any(inventory[resource] < before_inventory[resource] for resource in RESOURCES):
                    raise FlowError("Inventory decreased during collection; resource change is not attributable")
                resource = target["resource"]
                increase = inventory[resource] - before_inventory[resource]
                disappeared = not any(_overlaps(target["bbox"], item["bbox"]) for item in targets)
                if disappeared and increase > 0:
                    collected[resource] += increase
                    event = {
                        "target": target, "resource": resource, "increase": increase,
                        "resources_before": dict(before_inventory), "resources_after": dict(inventory),
                        "before_frame": str(before.screenshot_path), "after_frame": str(current.screenshot_path),
                    }
                    verified.append(event)
                    session.event("collection_verified", **event)
                    break
                if observation == VERIFICATION_OBSERVATIONS - 1:
                    return finish("failed", "collection_postcondition_unverified")
    except AutoCOCError as exc:
        return finish("failed", str(exc))


def _inventory(snapshot: SceneSnapshot) -> dict[str, int]:
    if snapshot.scene != SCENE_VILLAGE or snapshot.confidence < 0.8:
        raise FlowError(f"Collection requires a recognized village, observed {snapshot.scene}")
    if snapshot.observations.get("resource_source") != "village_inventory":
        raise FlowError("Collection requires village inventory readings")
    values = snapshot.observations.get("resources")
    if not isinstance(values, dict) or any(
        type(values.get(resource)) is not int or values[resource] < 0 for resource in RESOURCES
    ):
        raise FlowError("Village resource inventory is incomplete")
    return {resource: values[resource] for resource in RESOURCES}


def _targets(snapshot: SceneSnapshot) -> list[CollectionTarget]:
    items = snapshot.observations.get("collectibles")
    if not isinstance(items, list):
        raise FlowError("Collection observations are unavailable")
    targets: list[CollectionTarget] = []
    for item in items:
        if not isinstance(item, dict):
            raise FlowError("Invalid collection target")
        point, box = item.get("point"), item.get("bbox")
        confidence = item.get("confidence")
        if (
            item.get("resource") not in RESOURCES
            or not isinstance(point, (tuple, list)) or len(point) != 2
            or not isinstance(box, (tuple, list)) or len(box) != 4
            or any(type(value) is not int for value in (*point, *box))
            or not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])
            or type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0.88 <= confidence <= 1
            or not isinstance(item.get("template"), str) or not item["template"]
        ):
            raise FlowError("Collection target lacks reliable template or geometry evidence")
        targets.append(CollectionTarget(resource=item["resource"], point=tuple(point), bbox=tuple(box),
                                        confidence=float(confidence), template=item["template"]))
    return targets


def _storage_full(snapshot: SceneSnapshot, resource: str, inventory: dict[str, int]) -> bool:
    # Capacity is optional evidence; an unchanged balance never establishes full storage.
    capacities = snapshot.observations.get("resource_capacities", {})
    capacity = capacities.get(resource) if isinstance(capacities, dict) else None
    return type(capacity) is int and capacity > 0 and inventory[resource] >= capacity


def _overlaps(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    area = min((first[2] - first[0]) * (first[3] - first[1]),
               (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / area > 0.3
