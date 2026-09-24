from pathlib import Path
from types import SimpleNamespace
import unittest

from autococ.errors import FlowError
from autococ.scene import SceneSnapshot
from autococ.village import collect_resources


def bubble(resource: str = "gold", x: int = 400, y: int = 300) -> dict:
    return {"resource": resource, "point": [x, y], "bbox": [x - 12, y - 12, x + 12, y + 12],
            "confidence": 0.95, "template": f"collect_{resource}.png", "scale": 1.0}


def frame(index: int, targets: list | None = None, *, gold: int | None = 100, elixir: int = 200,
          dark_elixir: int = 50, scene: str = "village", confidence: float = 0.95,
          capacities: dict | None = None) -> SceneSnapshot:
    return SceneSnapshot(scene, confidence, Path(f"frame-{index}.png"), {
        "resource_source": "village_inventory",
        "resources": {"gold": gold, "elixir": elixir, "dark_elixir": dark_elixir},
        "collectibles": targets if targets is not None else [], "resource_capacities": capacities or {},
    })


class FakeSession:
    def __init__(self, frames: list[SceneSnapshot | Exception]) -> None:
        self.frames = iter(frames)
        self.config = SimpleNamespace(runtime=SimpleNamespace(poll_interval_sec=0))
        self.taps = []
        self.events = []
        self.last_snapshot = None
        self.expired = False

    def check_deadline(self) -> None:
        if self.expired:
            raise FlowError("Session or task deadline exceeded")

    def observe(self, label: str = "observe") -> SceneSnapshot:
        self.check_deadline()
        snapshot = next(self.frames)
        if isinstance(snapshot, Exception):
            raise snapshot
        self.last_snapshot = snapshot
        return snapshot

    def tap(self, snapshot: SceneSnapshot, point: tuple[int, int], *, reason: str) -> None:
        if snapshot is not self.last_snapshot:
            raise AssertionError("Clicked stale snapshot")
        self.taps.append((snapshot.screenshot_path, point))

    def event(self, kind: str, **data: object) -> None:
        self.events.append((kind, data))


class VillageTests(unittest.TestCase):
    def test_collection_requires_disappearance_and_corresponding_resource_increase(self) -> None:
        session = FakeSession([frame(0, [bubble()]), frame(1, gold=150)])
        result = collect_resources(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["collected_gold"], 50)
        self.assertEqual(result.metrics["collected_elixir"], 0)
        self.assertEqual(result.metrics["resources_before"]["gold"], 100)
        self.assertEqual(result.metrics["resources_after"]["gold"], 150)
        self.assertEqual(result.evidence, [Path("frame-0.png"), Path("frame-1.png")])
        self.assertEqual(len(session.taps), 1)

    def test_each_next_target_comes_from_the_latest_frame(self) -> None:
        session = FakeSession([
            frame(0, [bubble(), bubble("elixir", 500)]),
            frame(1, [bubble("elixir", 520)], gold=150),
            frame(2, gold=150, elixir=240),
        ])
        result = collect_resources(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(session.taps[1], (Path("frame-1.png"), (520, 300)))
        self.assertEqual(result.metrics["collected_elixir"], 40)

    def test_unchanged_screenshots_never_report_success(self) -> None:
        session = FakeSession([frame(index, [bubble()]) for index in range(4)])
        result = collect_resources(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "collection_postcondition_unverified")
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(result.metrics["collected_gold"], 0)

    def test_disappearance_without_inventory_increase_is_not_success(self) -> None:
        session = FakeSession([frame(0, [bubble()]), frame(1), frame(2), frame(3)])
        self.assertEqual(collect_resources(session).status, "failed")

    def test_inventory_increase_with_bubble_remaining_is_not_success(self) -> None:
        session = FakeSession([frame(0, [bubble()])] + [frame(index, [bubble()], gold=150) for index in range(1, 4)])
        self.assertEqual(collect_resources(session).status, "failed")

    def test_other_resource_increase_cannot_verify_target(self) -> None:
        session = FakeSession([frame(0, [bubble()])] + [frame(index, elixir=250) for index in range(1, 4)])
        result = collect_resources(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["collected_elixir"], 0)

    def test_animation_can_settle_before_verification(self) -> None:
        session = FakeSession([frame(0, [bubble()]), frame(1, [bubble()]), frame(2, gold=150)])
        self.assertEqual(collect_resources(session).status, "succeeded")
        self.assertEqual(len(session.taps), 1)

    def test_no_collectibles_is_a_skip_not_success(self) -> None:
        session = FakeSession([frame(0)])
        result = collect_resources(session)
        self.assertEqual((result.status, result.reason), ("skipped", "no_collectibles"))
        self.assertEqual(session.taps, [])

    def test_known_capacity_allows_full_storage_skip(self) -> None:
        session = FakeSession([frame(0, [bubble()], capacities={"gold": 100})])
        result = collect_resources(session)
        self.assertEqual((result.status, result.reason), ("skipped", "storage_full"))
        self.assertEqual(session.taps, [])

    def test_full_gold_does_not_block_elixir_collection(self) -> None:
        session = FakeSession([
            frame(0, [bubble(), bubble("elixir", 500)], capacities={"gold": 100}),
            frame(1, [bubble()], elixir=220, capacities={"gold": 100}),
        ])
        result = collect_resources(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["collected_elixir"], 20)
        self.assertEqual(session.taps[0][1], (500, 300))

    def test_unknown_inventory_and_wrong_scenes_fail_without_clicking(self) -> None:
        for snapshot in (frame(0, gold=None), frame(0, scene="battle"), frame(0, confidence=0.6)):
            with self.subTest(snapshot=snapshot):
                session = FakeSession([snapshot])
                self.assertEqual(collect_resources(session).status, "failed")
                self.assertEqual(session.taps, [])

    def test_scene_change_or_missing_inventory_after_tap_is_failure(self) -> None:
        for after in (frame(1, scene="unknown"), frame(1, gold=None)):
            with self.subTest(after=after):
                result = collect_resources(FakeSession([frame(0, [bubble()]), after]))
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.metrics["collected_gold"], 0)

    def test_inventory_decrease_is_not_misattributed_to_collection(self) -> None:
        result = collect_resources(FakeSession([frame(0, [bubble()]), frame(1, gold=150, elixir=190)]))
        self.assertEqual(result.status, "failed")

    def test_target_limit_preserves_verified_partial_amounts(self) -> None:
        snapshots = [frame(index, [bubble(x=400 + index)], gold=100 + index) for index in range(31)]
        session = FakeSession(snapshots)
        # Keep each replacement bubble outside the previous target's overlap region.
        for index, snapshot in enumerate(snapshots):
            snapshot.observations["collectibles"] = [bubble(x=400 if index % 2 == 0 else 500)]
        result = collect_resources(session)
        self.assertEqual((result.status, result.reason), ("failed", "collection_target_limit"))
        self.assertEqual(len(session.taps), 30)
        self.assertEqual(result.metrics["collected_gold"], 30)

    def test_task_deadline_is_reported_and_does_not_click_again(self) -> None:
        session = FakeSession([frame(0, [bubble()]), FlowError("Session or task deadline exceeded")])
        result = collect_resources(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("deadline", result.reason)
        self.assertEqual(len(session.taps), 1)

    def test_partial_success_is_preserved_without_calling_the_task_successful(self) -> None:
        session = FakeSession([
            frame(0, [bubble(), bubble("elixir", 500)]),
            frame(1, [bubble("elixir", 500)], gold=150),
            FlowError("Screenshot capture failed"),
        ])
        result = collect_resources(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["collected_gold"], 50)
        self.assertEqual(result.metrics["collected_elixir"], 0)
        self.assertEqual(len(result.metrics["verified_collections"]), 1)
        self.assertEqual(len(session.taps), 2)

    def test_changed_resource_label_does_not_mean_the_bubble_disappeared(self) -> None:
        session = FakeSession([frame(0, [bubble()])] + [
            frame(index, [bubble("elixir")], gold=150) for index in range(1, 4)
        ])
        self.assertEqual(collect_resources(session).status, "failed")

    def test_non_inventory_resources_are_rejected(self) -> None:
        snapshot = frame(0, [bubble()])
        snapshot.observations["resource_source"] = "available_enemy_loot"
        session = FakeSession([snapshot])
        self.assertEqual(collect_resources(session).status, "failed")
        self.assertEqual(session.taps, [])

    def test_low_confidence_target_is_failure_not_empty_skip(self) -> None:
        target = bubble()
        target["confidence"] = 0.5
        session = FakeSession([frame(0, [target])])
        self.assertEqual(collect_resources(session).status, "failed")
        self.assertEqual(session.taps, [])

    def test_missing_collection_observations_is_failure(self) -> None:
        snapshot = frame(0)
        del snapshot.observations["collectibles"]
        self.assertEqual(collect_resources(FakeSession([snapshot])).status, "failed")

    def test_reused_screenshot_cannot_verify_new_result(self) -> None:
        session = FakeSession([frame(0, [bubble()]), frame(0, gold=150)])
        self.assertEqual(collect_resources(session).status, "failed")


if __name__ == "__main__":
    unittest.main()
