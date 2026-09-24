from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from autococ.deployment import _remaining, deploy_army
from autococ.errors import CaptureError, DeploymentError, FlowError
from autococ.scene import SceneSnapshot
from autococ.session import GameSession


def card(kind: str = "troop", count: int | None = 10, left: int = 180) -> dict:
    return {"kind": kind, "count": count, "bbox": [left, 590, left + 70, 700],
            "point": [left + 35, 645]}


class Clock:
    now = 100.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSession:
    _validate_snapshot = GameSession._validate_snapshot

    def __init__(self, clock: Clock, cards: list[dict] | None = None) -> None:
        self.clock = clock
        self.cards = deepcopy(cards or [card()])
        self.counts = {item["point"][0]: item["count"] for item in self.cards}
        self.config = SimpleNamespace(
            battle=SimpleNamespace(deploy_timeout_sec=180),
            runtime=SimpleNamespace(poll_interval_sec=0.25),
            game=SimpleNamespace(baseline_resolution=(1280, 720)),
        )
        self.last_snapshot = None
        self.frames, self.taps, self.events, self.swipes = [], [], [], []
        self.action_count = 0
        self.context = SimpleNamespace(swipe=self.swipe)
        self.selected = None
        self.blocked_points = set()
        self.observe_hook = None
        self.tap_hook = None
        self.tap_seconds = 0.05
        self.capture_seconds = 0.5
        self.scout = self.observe("scout")
        self.scout.observations["expected_army_manifest"] = {
            "frame": "army-before.png", "complete": True, "supported_layout": True,
            "troops": [{"count": item["count"]} for item in self.cards if item["kind"] == "troop"],
            "spells": [{"count": item["count"]} for item in self.cards if item["kind"] == "spell"],
        }

    def check_deadline(self) -> None:
        pass

    def observe(self, label: str) -> SceneSnapshot:
        captured_at = self.clock.now
        self.clock.advance(self.capture_seconds)
        slots, ocr = [], []
        for item in self.cards:
            count = self.counts[item["point"][0]]
            if count != 0:
                # A previous card's count can be stale; the OCR is authoritative.
                slots.append(deepcopy(item))
            if count is not None:
                left = item["bbox"][0]
                ocr.append({"text": f"x{count}", "confidence": 0.99,
                            "bbox": [left + 5, 590, left + 50, 618]})
        frame = SceneSnapshot("enemy_village" if not self.taps else "battle", 0.95,
                              Path(f"deployment-{len(self.frames)}.png"), {
                                  "observed_at_monotonic": captured_at,
                                  "battle": {"slots": slots}, "ocr": ocr,
                              })
        if self.observe_hook is not None:
            frame = self.observe_hook(label, frame)
        self.frames.append(frame)
        self.last_snapshot = frame
        return frame

    def tap(self, snapshot: SceneSnapshot, point: list[int], *, reason: str) -> None:
        self._validate_snapshot(snapshot)
        self.taps.append((snapshot, list(point), reason, self.clock.now))
        self.clock.advance(self.tap_seconds)
        self.action_count += 1
        if point[1] > 500:
            self.selected = point[0]
        elif tuple(point) not in self.blocked_points and self.selected is not None:
            count = self.counts[self.selected]
            if count is not None:
                self.counts[self.selected] = max(0, count - 1)
        if self.tap_hook is not None:
            self.tap_hook(snapshot, point)

    def swipe(self, *args: int) -> None:
        self.swipes.append(args)

    def event(self, kind: str, **data: object) -> None:
        self.events.append((kind, data))

    @property
    def placements(self) -> list:
        return [tap for tap in self.taps if tap[1][1] < 500]


class DeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.clock.now = 100.0
        for target, replacement in (("autococ.deployment.time.monotonic", lambda: self.clock.now),
                                    ("autococ.deployment.time.sleep", self.clock.advance)):
            patcher = patch(target, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("autococ.terrain.find_west_deployment_points", return_value=[
            {"point": [300, 330], "confidence": 0.99, "evidence": {"deployment_confirmed": False}},
            {"point": [310, 365], "confidence": 0.99, "evidence": {"deployment_confirmed": False}},
        ])
        self.terrain = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("autococ.terrain.measure_camera_motion", return_value={"stationary": None})
        self.camera_motion = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("autococ.terrain.find_clear_ground_probes", return_value=[])
        self.ground_probes = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("autococ.hero_state.recognize_hero_state", return_value={
            "state": "unknown", "deployed": None, "ability_ready": None, "ability_used": None})
        self.hero_state = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("autococ.hero_state.recognize_siege_state", return_value={"state": "unknown", "deployed": None})
        self.siege_state = patcher.start()
        self.addCleanup(patcher.stop)

    def test_quantity_uses_new_ocr_in_original_header_including_gray_zero_card(self) -> None:
        session = FakeSession(self.clock)
        original = session.cards[0]
        for count in (10, 9, 0):
            session.counts[original["point"][0]] = count
            frame = session.observe("quantity")
            self.assertEqual(_remaining(frame, original), count)
        self.assertEqual(frame.observations["battle"]["slots"], [])
        self.assertEqual(original["count"], 10)

    def test_missing_ambiguous_or_unreliable_header_is_unknown_not_zero(self) -> None:
        session = FakeSession(self.clock)
        original = session.cards[0]
        for confidence in (0.89, float("nan"), float("inf")):
            frame = session.observe("quantity")
            frame.observations["ocr"][0]["confidence"] = confidence
            with self.subTest(confidence=confidence):
                self.assertIsNone(_remaining(frame, original))
        frame = session.observe("quantity")
        frame.observations["ocr"].append(deepcopy(frame.observations["ocr"][0]))
        self.assertIsNone(_remaining(frame, original))
        frame.observations["ocr"] = []
        self.assertIsNone(_remaining(frame, original))

    def test_count_below_original_header_or_on_adjacent_card_is_ignored(self) -> None:
        session = FakeSession(self.clock)
        for box in ([185, 660, 230, 690], [290, 590, 335, 618]):
            frame = session.observe("quantity")
            frame.observations["ocr"][0]["bbox"] = box
            self.assertIsNone(_remaining(frame, session.cards[0]))

    def test_deploys_ten_with_verified_batches_and_gray_zero_evidence(self) -> None:
        session = FakeSession(self.clock)
        result = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(result["verified"])
        self.assertTrue(result["completed"])
        self.assertEqual(result["deployed_units"], 10)
        self.assertEqual(len(session.placements), 10)
        self.assertEqual([event["consumed"] for event in result["consumption_events"]], [1, 8, 1])
        self.assertEqual(result["consumption_events"][-1]["after_count"], 0)
        self.assertTrue(all(event["before_frame"] != event["after_frame"] for event in result["consumption_events"]))
        self.assertLessEqual(len(session.frames), 10, "Do not capture once per deployed troop")

    def test_transient_card_loss_is_reobserved_without_repeating_placements(self) -> None:
        for missing_label in ("deploy-selected", "deploy-consumption"):
            with self.subTest(missing_label=missing_label):
                session = FakeSession(self.clock)
                missing, reacquired = [], []

                def observed(label, snapshot):
                    if label == missing_label and not missing:
                        missing.append(snapshot.screenshot_path)
                        snapshot.observations["battle"]["slots"] = []
                    if label == "deploy-card-reacquire":
                        reacquired.append(snapshot.screenshot_path)
                    return snapshot

                session.observe_hook = observed
                receipt = deploy_army(session, session.scout).observations["deployment"]
                self.assertTrue(receipt["completed"])
                self.assertEqual(len(reacquired), 1)
                self.assertEqual(len(session.placements), 10)
                self.assertEqual(receipt["deployed_units"], 10)
                if missing_label == "deploy-selected":
                    self.assertEqual(receipt["consumption_events"][0]["before_frame"], str(reacquired[0]))

    def test_delayed_batch_waits_for_all_issued_clicks_without_repeating_them(self):
        session = FakeSession(self.clock)
        observed_placements = []

        def observed(label, snapshot):
            if label == "deploy-consumption" and len(session.placements) == 9:
                observed_placements.append(len(session.placements))
                if len(observed_placements) == 1:
                    snapshot.observations["ocr"][0]["text"] = "x8"
            return snapshot

        session.observe_hook = observed
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(receipt["completed"])
        self.assertEqual(observed_placements, [9, 9])
        self.assertEqual(len(session.placements), 10)
        self.assertEqual([event["consumed"] for event in receipt["consumption_events"]], [1, 8, 1])
        self.assertEqual(receipt["deployed_units"], 10)

    def test_unconfirmed_rest_of_batch_stops_with_only_the_verified_partial_count(self):
        session = FakeSession(self.clock)

        def observed(label, snapshot):
            if label == "deploy-consumption" and len(session.placements) == 9:
                snapshot.observations["ocr"][0]["text"] = "x8"
            return snapshot

        session.observe_hook = observed
        with self.assertRaisesRegex(DeploymentError, "Not all issued placements") as caught:
            deploy_army(session, session.scout)
        receipt = caught.exception.partial_receipt
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 2)
        self.assertEqual(len(session.placements), 9)
        self.assertEqual(receipt["consumption_events"][-1]["issued_placements"], 8)

    def test_capture_failure_during_delayed_batch_retains_observed_consumption(self):
        session = FakeSession(self.clock)
        partial_reads = []

        def observed(label, snapshot):
            if label == "deploy-consumption" and len(session.placements) == 9:
                partial_reads.append(True)
                if len(partial_reads) == 2:
                    raise CaptureError("capture failed during pending batch")
                snapshot.observations["ocr"][0]["text"] = "x8"
            return snapshot

        session.observe_hook = observed
        with self.assertRaisesRegex(DeploymentError, "capture failed") as caught:
            deploy_army(session, session.scout)
        self.assertEqual(caught.exception.partial_receipt["deployed_units"], 2)
        self.assertEqual(len(session.placements), 9)

    def test_persistent_card_loss_retains_partial_receipt_and_stops_without_more_input(self) -> None:
        session = FakeSession(self.clock)
        lost, rereads = [], []

        def observed(label, snapshot):
            if label == "deploy-consumption":
                lost.append(True)
            if lost:
                snapshot.observations["battle"]["slots"] = []
            if label == "deploy-card-reacquire":
                rereads.append(snapshot.screenshot_path)
            return snapshot

        session.observe_hook = observed
        with self.assertRaisesRegex(DeploymentError, "position is no longer verified") as caught:
            deploy_army(session, session.scout)
        self.assertEqual(len(rereads), 2)
        self.assertEqual(len(session.placements), 1)
        self.assertEqual(caught.exception.partial_receipt["deployed_units"], 1)

    def test_quantity_change_during_card_reacquisition_prevents_placement(self) -> None:
        session = FakeSession(self.clock)

        def observed(label, snapshot):
            if label == "deploy-selected":
                snapshot.observations["battle"]["slots"] = []
                session.counts[session.cards[0]["point"][0]] = 9
            return snapshot

        session.observe_hook = observed
        with self.assertRaisesRegex(DeploymentError, "quantity changed before field placement"):
            deploy_army(session, session.scout)
        self.assertEqual(session.placements, [])

    def test_full_army_with_five_second_captures_keeps_verification_inside_deadline(self) -> None:
        session = FakeSession(self.clock, [
            card(count=10, left=80), card(count=1, left=180), card(count=1, left=280),
            card("siege", None, 380), *[card("hero", None, left) for left in (480, 580, 680, 780)],
            card("spell", 6, 880), card("spell", 5, 980),
        ])
        session.capture_seconds = 5
        labels = {}

        def observed(label, snapshot):
            labels[snapshot.screenshot_path] = label
            return snapshot

        session.observe_hook = observed

        def hero_state(path, *args, **kwargs):
            used = labels[path] == "hero-ability-verify"
            return {"state": "ability_used" if used else "ability_ready", "deployed": True,
                    "ability_ready": not used, "ability_used": used}

        self.hero_state.side_effect = hero_state
        self.siege_state.return_value = {"state": "deployed", "deployed": True}
        started = self.clock.now
        result = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(result["completed"])
        self.assertTrue(result["numeric_completed"])
        self.assertTrue(result["hero_abilities_verified"])
        self.assertEqual(result["deployed_units"], 12)
        self.assertLess(self.clock.now - started, 180)
        self.assertEqual(sum(event["consumed"] for event in result["consumption_events"] if event["unit_kind"] == "spell"), 11)

    def test_visible_quantity_without_card_stops_before_any_deployment(self) -> None:
        session = FakeSession(self.clock)
        session.scout.observations["ocr"].append({
            "text": "x6", "confidence": .99, "bbox": [1044, 592, 1086, 622]})
        with self.assertRaisesRegex(DeploymentError, "no unique numeric card") as raised:
            deploy_army(session, session.scout)
        self.assertEqual(session.taps, [])
        self.assertEqual(raised.exception.partial_receipt["deployed_units"], 0)

    def test_entire_card_and_ocr_missing_cannot_shrink_independent_plan(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card(count=5, left=280)])
        session.scout.observations["battle"]["slots"].pop()
        session.scout.observations["ocr"].pop()
        with self.assertRaisesRegex(DeploymentError, "independent prebattle army manifest"):
            deploy_army(session, session.scout)
        self.assertEqual(session.taps, [])

    def test_missing_prebattle_manifest_stops_before_any_deployment(self) -> None:
        session = FakeSession(self.clock)
        session.scout.observations.pop("expected_army_manifest")
        with self.assertRaisesRegex(DeploymentError, "no independent prebattle army manifest"):
            deploy_army(session, session.scout)
        self.assertEqual(session.taps, [])

    def test_changed_quantity_does_not_match_prebattle_manifest(self) -> None:
        session = FakeSession(self.clock)
        session.scout.observations["expected_army_manifest"]["troops"][0]["count"] = 9
        with self.assertRaisesRegex(DeploymentError, "independent prebattle army manifest"):
            deploy_army(session, session.scout)
        self.assertEqual(session.taps, [])

    def test_received_clan_spell_is_additional_and_must_be_fully_consumed(self) -> None:
        reinforcement = {**card("spell", 3, 280), "source": "clan_reinforcement",
                         "evidence": {"clan_badge": {"confidence": .99}}}
        session = FakeSession(self.clock, [card(count=1), reinforcement])
        session.scout.observations["expected_army_manifest"]["spells"] = []
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(receipt["completed"])
        self.assertTrue(receipt["army_manifest_matched"])
        self.assertEqual(session.counts[315], 0)
        self.assertEqual(receipt["planned_numeric_cards"][1]["source"], "clan_reinforcement")
        self.assertEqual(sum(event["consumed"] for event in receipt["consumption_events"] if event["unit_kind"] == "spell"), 3)

    def test_unverified_clan_label_cannot_bypass_independent_recipe(self) -> None:
        reinforcement = {**card("spell", 3, 280), "source": "clan_reinforcement"}
        session = FakeSession(self.clock, [card(count=1), reinforcement])
        session.scout.observations["expected_army_manifest"]["spells"] = []
        with self.assertRaisesRegex(DeploymentError, "reinforcement badge"):
            deploy_army(session, session.scout)
        self.assertEqual(session.taps, [])

    def test_overlapping_numeric_cards_do_not_share_one_quantity(self) -> None:
        session = FakeSession(self.clock)
        session.scout.observations["battle"]["slots"].append(deepcopy(session.cards[0]))
        with self.assertRaisesRegex(DeploymentError, "no unique numeric card"):
            deploy_army(session, session.scout)
        self.assertEqual(session.placements, [])

    def test_visible_quantity_on_unreadable_card_cannot_be_excluded_from_plan(self) -> None:
        session = FakeSession(self.clock)
        session.scout.observations["battle"]["slots"][0]["count"] = None
        with self.assertRaisesRegex(DeploymentError, "no unique numeric card"):
            deploy_army(session, session.scout)
        self.assertEqual(session.placements, [])

    def test_late_visible_unplanned_quantity_cannot_be_complete(self) -> None:
        session = FakeSession(self.clock)

        def reveal_missed_card(label: str, snapshot: SceneSnapshot) -> SceneSnapshot:
            if label == "deployment-complete":
                snapshot.observations["ocr"].append({
                    "text": "x5", "confidence": .99, "bbox": [1142, 592, 1182, 622]})
            return snapshot

        session.observe_hook = reveal_missed_card
        with self.assertRaisesRegex(DeploymentError, "no unique numeric card") as raised:
            deploy_army(session, session.scout)
        self.assertEqual(raised.exception.partial_receipt["deployed_units"], 10)
        self.assertFalse(raised.exception.partial_receipt["completed"])

    def test_final_quantity_check_rejects_returned_or_still_present_units(self) -> None:
        session = FakeSession(self.clock)

        def remaining_again(label: str, snapshot: SceneSnapshot) -> SceneSnapshot:
            if label == "deployment-complete":
                snapshot.observations["ocr"][0]["text"] = "x1"
            return snapshot

        session.observe_hook = remaining_again
        with self.assertRaisesRegex(DeploymentError, "not verified empty"):
            deploy_army(session, session.scout)

    def test_local_gray_header_read_completes_last_unit_and_retains_evidence(self) -> None:
        session = FakeSession(self.clock)

        def omit_gray_count(label: str, frame: SceneSnapshot) -> SceneSnapshot:
            if session.counts[215] == 0:
                frame.observations["ocr"] = []
            return frame

        def read_header(path: Path, bbox: tuple[int, int, int, int]) -> dict:
            return {"count": 0, "confidence": .97, "frame": str(path), "slot_bbox": list(bbox),
                    "readings": [{"text": "x0", "confidence": .97, "source": "header_threshold_line"}]}

        session.observe_hook = omit_gray_count
        session.recognizer = Mock()
        session.recognizer.recognize_slot_count.side_effect = read_header
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 10)
        self.assertEqual([event["consumed"] for event in receipt["consumption_events"]], [1, 8, 1])
        # One read proves consumption; the final, fresh frame proves it remains empty.
        self.assertEqual(session.recognizer.recognize_slot_count.call_count, 2)
        self.assertEqual(len(receipt["count_reads"]), 2)
        self.assertEqual(receipt["count_reads"][0]["readings"][0]["text"], "x0")
        self.assertEqual(receipt["count_reads"][0]["frame"], receipt["consumption_events"][-1]["after_frame"])
        self.assertIn(receipt["count_reads"][0]["frame"], receipt["evidence"])
        self.assertNotEqual(receipt["count_reads"][0]["frame"], receipt["count_reads"][1]["frame"])
        self.assertEqual(sum(kind == "deployment_count_read" for kind, _ in session.events), 2)

    def test_selected_card_zero_is_reused_only_within_the_same_consumption_frame(self) -> None:
        for final_count in (0, None):
            with self.subTest(final_count=final_count):
                session = FakeSession(self.clock, [card(count=1)])
                labels = {}

                def resize_selected(label: str, frame: SceneSnapshot) -> SceneSnapshot:
                    labels[str(frame.screenshot_path)] = label
                    if label == "deploy-selected":
                        frame.observations["battle"]["slots"][0]["bbox"] = [176, 582, 254, 704]
                    if session.counts[215] == 0:
                        frame.observations["ocr"] = []
                    return frame

                def read_header(path: Path, bbox: tuple[int, int, int, int]) -> dict:
                    label = labels[str(path)]
                    count = final_count if label == "deployment-complete" else (0 if bbox[0] == 176 else None)
                    return {"count": count, "confidence": .97, "frame": str(path), "slot_bbox": list(bbox)}

                session.observe_hook = resize_selected
                session.recognizer = Mock()
                session.recognizer.recognize_slot_count.side_effect = read_header
                if final_count == 0:
                    receipt = deploy_army(session, session.scout).observations["deployment"]
                    self.assertTrue(receipt["completed"])
                    self.assertEqual(receipt["deployed_units"], 1)
                else:
                    with self.assertRaisesRegex(DeploymentError, "not verified empty"):
                        deploy_army(session, session.scout)
                calls = session.recognizer.recognize_slot_count.call_args_list
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0].args[1][0], 176)
                self.assertEqual(calls[1].args[1][0], 180)
                self.assertNotEqual(calls[0].args[0], calls[1].args[0])

    def test_unverified_local_count_preserves_nine_units_not_assumed_ten(self) -> None:
        for count, confidence in ((None, .99), (0, .89), (0, float("nan")), (0, float("inf"))):
            with self.subTest(count=count, confidence=confidence):
                session = FakeSession(self.clock)

                def omit_gray_count(label: str, frame: SceneSnapshot) -> SceneSnapshot:
                    if session.counts[215] == 0:
                        frame.observations["ocr"] = []
                    return frame

                def read_header(path: Path, bbox: tuple[int, int, int, int]) -> dict:
                    return {"count": count, "confidence": confidence, "frame": str(path), "slot_bbox": list(bbox)}

                session.observe_hook = omit_gray_count
                session.recognizer = Mock()
                session.recognizer.recognize_slot_count.side_effect = read_header
                with self.assertRaisesRegex(DeploymentError, "stopped making progress") as stopped:
                    deploy_army(session, session.scout)
                receipt = stopped.exception.partial_receipt
                self.assertFalse(receipt["completed"])
                self.assertEqual(receipt["deployed_units"], 9)
                self.assertEqual(len(session.placements), 10)
                self.assertEqual(len(receipt["count_reads"]), 3)
                self.assertTrue(all(read["frame"] in receipt["evidence"] for read in receipt["count_reads"]))

    def test_conflicting_full_header_counts_cannot_be_overridden_by_local_zero(self) -> None:
        session = FakeSession(self.clock)
        frame = session.scout
        frame.observations["ocr"].append({**frame.observations["ocr"][0], "text": "x0"})
        recognizer = Mock()
        recognizer.recognize_slot_count.return_value = {"count": 0, "confidence": .99}
        self.assertIsNone(_remaining(frame, session.cards[0], recognizer=recognizer))
        recognizer.recognize_slot_count.assert_not_called()

    def test_local_count_reader_receives_only_explicit_high_confidence_prior_bbox(self) -> None:
        session = FakeSession(self.clock)
        original = deepcopy(session.cards[0])
        original["evidence"] = {"count": {"text": "x10", "confidence": .99, "bbox": [205, 592, 245, 620]}}
        frame = session.scout
        frame.observations["ocr"] = []
        for confidence in (.99, .89):
            with self.subTest(confidence=confidence):
                original["evidence"]["count"]["confidence"] = confidence
                frame.observations.pop("deployment_count_reads", None)
                recognizer = Mock()
                recognizer.recognize_slot_count.return_value = {
                    "count": 0, "confidence": .97, "frame": str(frame.screenshot_path), "slot_bbox": original["bbox"],
                }
                self.assertEqual(_remaining(frame, original, recognizer=recognizer), 0)
                arguments = {"count_bbox": (205, 592, 245, 620)} if confidence == .99 else {}
                recognizer.recognize_slot_count.assert_called_once_with(frame.screenshot_path, tuple(original["bbox"]), **arguments)

    def test_local_read_from_other_frame_or_card_is_not_consumption_evidence(self) -> None:
        session = FakeSession(self.clock)
        frame = session.scout
        frame.observations["ocr"] = []
        for change in ({"frame": "older.png"}, {"slot_bbox": [280, 590, 350, 700]}):
            with self.subTest(change=change):
                frame.observations.pop("deployment_count_reads", None)
                recognizer = Mock()
                recognizer.recognize_slot_count.return_value = {
                    "count": 0, "confidence": .99, "frame": str(frame.screenshot_path),
                    "slot_bbox": list(session.cards[0]["bbox"]), **change,
                }
                self.assertIsNone(_remaining(frame, session.cards[0], recognizer=recognizer))

    def test_local_read_finishing_after_deadline_does_not_credit_last_unit(self) -> None:
        session = FakeSession(self.clock)

        def omit_gray_count(label: str, frame: SceneSnapshot) -> SceneSnapshot:
            if session.counts[215] == 0:
                frame.observations["ocr"] = []
            return frame

        def slow_read(path: Path, bbox: tuple[int, int, int, int]) -> dict:
            self.clock.advance(session.config.battle.deploy_timeout_sec)
            return {"count": 0, "confidence": .99, "frame": str(path), "slot_bbox": list(bbox)}

        session.observe_hook = omit_gray_count
        session.recognizer = Mock()
        session.recognizer.recognize_slot_count.side_effect = slow_read
        with self.assertRaisesRegex(DeploymentError, "deadline") as stopped:
            deploy_army(session, session.scout)
        receipt = stopped.exception.partial_receipt
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 9)
        self.assertEqual(len(session.placements), 10)
        self.assertEqual(len(receipt["count_reads"]), 1)

    def test_select_and_placement_use_different_observations(self) -> None:
        session = FakeSession(self.clock, [card(count=2)])
        deploy_army(session, session.scout)
        for index, (_, point, _, _) in enumerate(session.taps):
            if point[1] > 500:
                self.assertIsNot(session.taps[index][0], session.taps[index + 1][0])
        for frame, _, _, action_time in session.taps:
            self.assertLessEqual(action_time - frame.observations["observed_at_monotonic"], 30)

    def test_edge_without_real_consumption_never_verifies_deployment(self) -> None:
        session = FakeSession(self.clock)
        session.blocked_points = {(300, 330), (310, 365)}
        with self.assertRaisesRegex(FlowError, "consumed a troop"):
            deploy_army(session, session.scout)
        self.assertFalse(any(kind == "deployment_consumed" for kind, _ in session.events))
        self.assertNotIn("deployment", session.last_snapshot.observations)

    def test_skips_rejected_edge_then_counts_only_consumed_units(self) -> None:
        session = FakeSession(self.clock, [card(count=2)])
        session.blocked_points = {(300, 330)}
        result = deploy_army(session, session.scout).observations["deployment"]
        self.assertEqual(result["deployed_units"], 2)
        self.assertEqual(len(session.placements), 3)
        self.assertTrue(all(event["point"] == [310, 365] for event in result["consumption_events"]))

    def test_unknown_interruptions_and_nonfinite_confidence_never_start_actions(self) -> None:
        for scene, confidence in (("unknown", .99), ("disconnected", .99), ("maintenance", .99),
                                  ("battle", .79), ("battle", float("nan")), ("battle", float("inf"))):
            session = FakeSession(self.clock)
            session.scout = replace(session.scout, scene=scene, confidence=confidence)
            session.last_snapshot = session.scout
            with self.subTest(scene=scene, confidence=confidence), self.assertRaises(FlowError):
                deploy_army(session, session.scout)
            self.assertEqual(session.taps, [])
            self.assertEqual(session.swipes, [])

    def test_stale_initial_snapshot_never_clicks(self) -> None:
        session = FakeSession(self.clock)
        self.clock.advance(31)
        with self.assertRaisesRegex(FlowError, "stale"):
            deploy_army(session, session.scout)
        self.assertEqual(session.taps, [])

    def test_noncurrent_snapshot_cannot_start_a_batch(self) -> None:
        session = FakeSession(self.clock)
        session.observe("newer")
        with self.assertRaisesRegex(FlowError, "obsolete"):
            deploy_army(session, session.scout)
        self.assertEqual(session.taps, [])

    def test_moved_or_missing_card_after_selection_cannot_deploy(self) -> None:
        for moved in (False, True):
            session = FakeSession(self.clock)

            def change_card(label: str, frame: SceneSnapshot) -> SceneSnapshot:
                if moved:
                    frame.observations["battle"]["slots"][0]["point"][0] += 100
                else:
                    frame.observations["battle"]["slots"] = []
                return frame

            session.observe_hook = change_card
            with self.subTest(moved=moved), self.assertRaisesRegex(FlowError, "card position"):
                deploy_army(session, session.scout)
            self.assertEqual(session.placements, [])

    def test_ocr_drop_larger_than_issued_placements_is_not_consumption_proof(self) -> None:
        session = FakeSession(self.clock)

        def impossible_count(label: str, frame: SceneSnapshot) -> SceneSnapshot:
            if label == "deploy-consumption":
                frame.observations["ocr"][0]["text"] = "x0"
            return frame

        session.observe_hook = impossible_count
        with self.assertRaisesRegex(FlowError, "exceeds issued"):
            deploy_army(session, session.scout)
        self.assertEqual(len(session.placements), 1)
        self.assertFalse(any(kind == "deployment_consumed" for kind, _ in session.events))

    def test_count_change_during_selection_is_not_credited_to_placement(self) -> None:
        session = FakeSession(self.clock)

        def changed_header(label: str, frame: SceneSnapshot) -> SceneSnapshot:
            frame.observations["ocr"][0]["text"] = "x9"
            return frame

        session.observe_hook = changed_header
        with self.assertRaisesRegex(FlowError, "before field placement"):
            deploy_army(session, session.scout)
        self.assertEqual(session.placements, [])

    def test_explicit_pan_recomputes_candidates_from_the_new_frame(self) -> None:
        session = FakeSession(self.clock, [card(count=1)])
        self.terrain.side_effect = [[], [{"point": [410, 350], "confidence": .99}]]
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(self.terrain.call_count, 2)
        self.assertNotEqual(self.terrain.call_args_list[0].args[0], self.terrain.call_args_list[1].args[0])
        self.assertEqual(receipt["consumption_events"][0]["point"], [410, 350])

    def test_absent_boundary_after_one_pan_fails_without_taps(self) -> None:
        session = FakeSession(self.clock)
        self.terrain.return_value = []
        with self.assertRaisesRegex(FlowError, "boundary"):
            deploy_army(session, session.scout)
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(session.taps, [])
        self.assertEqual(self.terrain.call_count, 4)
        self.assertEqual(len({call.args[0] for call in self.terrain.call_args_list}), 4)

    def test_camera_pan_starts_on_current_clear_ground_with_slower_gesture(self) -> None:
        session = FakeSession(self.clock, [card(count=1)])
        self.terrain.side_effect = [[], [{"point": [410, 350], "confidence": .99}]]
        self.ground_probes.return_value = [{"point": [578, 336], "evidence": {"boundary_verified": False}}]
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertEqual(session.swipes, [(578, 336, 978, 336, 1200)])
        self.assertEqual(self.ground_probes.call_args.args[0], session.scout.screenshot_path)
        self.assertTrue(receipt["completed"])
        self.assertEqual(receipt["consumption_events"][0]["point"], [410, 350])

    def test_camera_ground_start_and_end_respect_configured_baseline(self) -> None:
        session = FakeSession(self.clock, [card(count=1)])
        session.config.game.baseline_resolution = (2560, 1440)
        self.terrain.side_effect = [[], [{"point": [410, 350], "confidence": .99}]]
        self.ground_probes.return_value = [{"point": [1500, 600]}]
        deploy_army(session, session.scout)
        self.assertEqual(session.swipes, [(1500, 600, 2100, 600, 1200)])
        self.assertEqual(self.ground_probes.call_args.kwargs["baseline_resolution"], (2560, 1440))

    def test_stationary_retry_reacquires_ground_from_latest_frame(self) -> None:
        session = FakeSession(self.clock, [card(count=1)])
        self.terrain.side_effect = [[], [], [], [], [{"point": [410, 350], "confidence": .99}]]
        self.camera_motion.return_value = {"stationary": True}
        self.ground_probes.side_effect = [[{"point": [400, 320]}], [{"point": [580, 340]}]]
        deploy_army(session, session.scout)
        self.assertEqual(session.swipes, [(400, 320, 800, 320, 1200), (580, 340, 980, 340, 1200)])
        self.assertEqual(self.ground_probes.call_args_list[1].args[0], self.terrain.call_args_list[3].args[0])

    def test_delayed_pan_checks_three_fresh_frames_without_another_pan(self) -> None:
        session = FakeSession(self.clock, [card(count=1)])
        self.terrain.side_effect = [[], [], [], [{"point": [410, 350], "confidence": .99}]]
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(sum(kind == "camera_pan" for kind, _ in session.events), 1)
        paths = [call.args[0] for call in self.terrain.call_args_list]
        self.assertEqual(len(set(paths)), 4)
        self.assertEqual(session.taps[0][0].screenshot_path, paths[-1])
        self.assertEqual(receipt["consumption_events"][0]["point"], [410, 350])
        self.assertTrue(all(str(path) in receipt["evidence"] for path in paths))

    def test_stationary_map_allows_one_slower_pan_before_verified_deployment(self) -> None:
        session = FakeSession(self.clock, [card(count=1)])
        self.terrain.side_effect = [[], [], [], [], [{"point": [410, 350], "confidence": .99}]]
        self.camera_motion.return_value = {"stationary": True, "inliers": 200}
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertEqual([swipe[-1] for swipe in session.swipes], [600, 1200])
        self.assertEqual(self.camera_motion.call_count, 1)
        self.assertEqual(self.camera_motion.call_args.args, (
            self.terrain.call_args_list[0].args[0], self.terrain.call_args_list[3].args[0]))
        self.assertEqual(session.taps[0][0].screenshot_path, self.terrain.call_args_list[4].args[0])
        self.assertEqual(receipt["deployed_units"], 1)
        self.assertTrue(receipt["completed"])
        self.assertEqual(receipt["camera_pans"][0]["motion"]["inliers"], 200)

    def test_repeated_stationary_map_stops_after_two_pans_without_taps(self) -> None:
        session = FakeSession(self.clock)
        self.terrain.return_value = []
        self.camera_motion.return_value = {"stationary": True}
        with self.assertRaisesRegex(DeploymentError, "boundary") as caught:
            deploy_army(session, session.scout)
        self.assertEqual([swipe[-1] for swipe in session.swipes], [600, 1200])
        self.assertEqual(self.terrain.call_count, 7)
        self.assertEqual(session.taps, [])
        self.assertEqual(len(caught.exception.partial_receipt["camera_pans"]), 2)

    def test_moved_or_uncertain_map_never_triggers_another_pan(self) -> None:
        for stationary in (False, None):
            with self.subTest(stationary=stationary):
                session = FakeSession(self.clock)
                self.terrain.return_value = []
                self.camera_motion.return_value = {"stationary": stationary}
                with self.assertRaisesRegex(DeploymentError, "boundary"):
                    deploy_army(session, session.scout)
                self.assertEqual(len(session.swipes), 1)
                self.assertEqual(session.taps, [])

    def test_deployment_deadline_bounds_waiting_for_pan_animation(self) -> None:
        session = FakeSession(self.clock)
        session.config.battle.deploy_timeout_sec = 1
        self.terrain.return_value = []
        with self.assertRaisesRegex(FlowError, "deadline"):
            deploy_army(session, session.scout)
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(session.taps, [])
        self.assertLess(self.terrain.call_count, 4)

    def test_clear_ground_fallback_requires_one_consumed_troop_before_bursts(self) -> None:
        session = FakeSession(self.clock)
        self.terrain.return_value = []
        self.ground_probes.return_value = [{"point": [318, 438], "evidence": {
            "method": "clear_textured_grass_probe", "boundary_verified": False, "deployment_confirmed": False}}]
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 10)
        self.assertEqual([event["consumed"] for event in receipt["consumption_events"]], [1, 8, 1])
        self.assertEqual(receipt["consumption_events"][0]["point"], [318, 438])
        self.assertFalse(receipt["terrain_evidence"][0]["evidence"]["boundary_verified"])

    def test_failed_ground_probe_is_not_used_for_the_rest_of_the_army(self) -> None:
        session = FakeSession(self.clock)
        session.blocked_points = {(318, 438)}
        self.terrain.return_value = []
        self.ground_probes.return_value = [{"point": [318, 438]}, {"point": [476, 421]}]
        receipt = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(receipt["completed"])
        self.assertEqual(len(session.placements), 11)
        self.assertEqual(session.placements[0][1], [318, 438])
        self.assertTrue(all(tap[1] == [476, 421] for tap in session.placements[1:]))
        self.assertTrue(all(event["point"] == [476, 421] for event in receipt["consumption_events"]))

    def test_three_unproductive_ground_probes_stop_without_committing_the_army(self) -> None:
        session = FakeSession(self.clock)
        points = [[318, 438], [476, 421], [248, 219]]
        session.blocked_points = set(map(tuple, points))
        self.terrain.return_value = []
        self.ground_probes.return_value = [{"point": point} for point in points]
        with self.assertRaisesRegex(DeploymentError, "No attempted edge placement consumed") as caught:
            deploy_army(session, session.scout)
        self.assertEqual([tap[1] for tap in session.placements], points)
        self.assertFalse(caught.exception.partial_receipt["verified"])
        self.assertEqual(caught.exception.partial_receipt["deployed_units"], 0)

    def test_late_terrain_result_after_pan_cannot_authorize_deployment(self) -> None:
        session = FakeSession(self.clock)
        session.config.battle.deploy_timeout_sec = 2

        def slow_terrain(path: Path, **kwargs) -> list[dict]:
            if path == session.scout.screenshot_path:
                return []
            self.clock.advance(3)
            return [{"point": [410, 350], "confidence": .99}]

        self.terrain.side_effect = slow_terrain
        with self.assertRaisesRegex(FlowError, "deadline"):
            deploy_army(session, session.scout)
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(session.taps, [])

    def test_session_deadline_still_applies_while_waiting_for_pan(self) -> None:
        session = FakeSession(self.clock)
        stop_at = self.clock.now + 0.4

        def check_session_deadline() -> None:
            if self.clock.now >= stop_at:
                raise FlowError("Session deadline exceeded")

        session.check_deadline = check_session_deadline
        self.terrain.return_value = []
        with self.assertRaisesRegex(FlowError, "Session deadline"):
            deploy_army(session, session.scout)
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(session.taps, [])
        self.assertEqual(self.terrain.call_count, 1)

    def test_interruption_after_pan_does_not_trigger_another_pan(self) -> None:
        session = FakeSession(self.clock)
        session.observe_hook = lambda label, frame: replace(frame, scene="disconnected")
        self.terrain.return_value = []
        with self.assertRaisesRegex(FlowError, "disconnected"):
            deploy_army(session, session.scout)
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(session.taps, [])
        self.assertEqual(self.terrain.call_count, 1)

    def test_interrupt_after_selection_prevents_any_field_click(self) -> None:
        for interruption in ("unknown", "disconnected", "maintenance"):
            session = FakeSession(self.clock)
            session.observe_hook = lambda label, frame: replace(frame, scene=interruption)
            with self.subTest(interruption=interruption), self.assertRaises(FlowError):
                deploy_army(session, session.scout)
            self.assertEqual(session.placements, [])

    def test_keyboard_interrupt_during_selected_observation_stops_placement(self) -> None:
        session = FakeSession(self.clock)
        session.observe_hook = lambda label, frame: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            deploy_army(session, session.scout)
        self.assertEqual(session.placements, [])

    def test_observation_finishing_after_deployment_deadline_cannot_click(self) -> None:
        session = FakeSession(self.clock)
        session.config.battle.deploy_timeout_sec = 1
        session.capture_seconds = 2
        with self.assertRaisesRegex(FlowError, "deadline"):
            deploy_army(session, session.scout)
        self.assertEqual(session.placements, [])

    def test_slow_actions_stop_batch_and_refresh_within_short_budget(self) -> None:
        session = FakeSession(self.clock, [card(count=8)])
        session.tap_seconds = 2
        deploy_army(session, session.scout)
        grouped = {}
        for frame, _, _, at in session.placements:
            grouped.setdefault(frame.screenshot_path, []).append(at)
        self.assertTrue(all(max(times) - min(times) <= 4 for times in grouped.values()))

    def test_unverified_hero_attempt_never_activates_an_ability(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card("hero", None, 280)])
        result = deploy_army(session, session.scout).observations["deployment"]
        self.assertFalse(result["support_attempts"][0]["verified"])
        self.assertEqual(result["support_attempts"][0]["status"], "deployment_unverified")
        self.assertTrue(result["numeric_completed"])
        self.assertFalse(result["completed"])
        self.assertFalse(any("ability" in reason.lower() for _, _, reason, _ in session.taps))

    def test_verified_hero_ability_clicked_once_and_waits_through_flash(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card("hero", None, 280)])
        self.hero_state.side_effect = [
            {"state": "ability_ready", "deployed": True, "ability_ready": True, "ability_used": False},
            {"state": "unknown", "deployed": None, "ability_ready": None, "ability_used": None},
            {"state": "ability_used", "deployed": True, "ability_ready": False, "ability_used": True},
        ]
        result = deploy_army(session, session.scout).observations["deployment"]
        attempt = result["support_attempts"][0]
        self.assertTrue(attempt["verified"])
        self.assertTrue(attempt["ability_verified"])
        self.assertEqual(attempt["ability_clicks"], 1)
        self.assertEqual(len(attempt["states"]), 3)
        self.assertTrue(result["hero_abilities_verified"])
        self.assertTrue(result["completed"])
        self.assertEqual(sum("Activate hero ability" in reason for _, _, reason, _ in session.taps), 1)
        self.assertNotEqual(attempt["ability_before_frame"], attempt["ability_after_frame"])

    def test_uncertain_hero_activation_never_retries_click(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card("hero", None, 280)])
        ready = {"state": "ability_ready", "deployed": True, "ability_ready": True, "ability_used": False}
        self.hero_state.return_value = ready
        result = deploy_army(session, session.scout).observations["deployment"]
        attempt = result["support_attempts"][0]
        self.assertEqual(attempt["ability_clicks"], 1)
        self.assertFalse(attempt["ability_verified"])
        self.assertFalse(result["hero_abilities_verified"])
        self.assertFalse(result["completed"])
        self.assertEqual(self.hero_state.call_count, 4)

    def test_deployed_hero_with_unknown_equipment_is_not_clicked_again(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card("hero", None, 280)])
        self.hero_state.return_value = {"state": "deployed", "deployed": True, "ability_ready": None, "ability_used": None}
        result = deploy_army(session, session.scout).observations["deployment"]
        self.assertTrue(result["support_attempts"][0]["verified"])
        self.assertFalse(result["hero_abilities_verified"])
        self.assertFalse(any("Activate hero ability" in reason for _, _, reason, _ in session.taps))

    def test_siege_deployment_requires_state_evidence_and_never_releases_it(self) -> None:
        for verified in (None, True):
            with self.subTest(deployed=verified):
                session = FakeSession(self.clock, [card(count=1), card("siege", None, 280)])
                self.siege_state.return_value = {"state": "deployed" if verified else "unknown", "deployed": verified}
                receipt = deploy_army(session, session.scout).observations["deployment"]
                self.assertEqual(receipt["support_attempts"][0]["verified"], verified is True)
                self.assertEqual(receipt["completed"], verified is True)
                self.assertEqual(sum(point == [315, 645] for _, point, _, _ in session.taps), 1)

    def test_settlement_at_final_observation_preserves_verified_empty_army(self) -> None:
        session = FakeSession(self.clock, [card(count=1)])
        def finish(label: str, snapshot: SceneSnapshot) -> SceneSnapshot:
            if label == "deployment-complete":
                return replace(snapshot, scene="settlement", observations={"settlement": {}})
            return snapshot
        session.observe_hook = finish
        result = deploy_army(session, session.scout)
        self.assertEqual(result.scene, "settlement")
        receipt = result.observations["deployment"]
        self.assertTrue(receipt["completed"])
        self.assertNotEqual(receipt["numeric_completion_frame"], str(result.screenshot_path))

    def test_interruption_after_verified_batch_preserves_consumption_event(self) -> None:
        session = FakeSession(self.clock, [card(count=10)])

        def disconnect_next_selection(label: str, frame: SceneSnapshot) -> SceneSnapshot:
            if label == "deploy-selected" and session.counts[215] == 9:
                return replace(frame, scene="disconnected")
            return frame

        session.observe_hook = disconnect_next_selection
        with self.assertRaises(DeploymentError) as stopped:
            deploy_army(session, session.scout)
        confirmed = [data for kind, data in session.events if kind == "deployment_consumed"]
        self.assertEqual([event["consumed"] for event in confirmed], [1])
        self.assertEqual(len(session.placements), 1)
        self.assertNotIn("deployment", session.last_snapshot.observations)
        receipt = stopped.exception.partial_receipt
        self.assertFalse(receipt["completed"])
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["deployed_units"], 1)
        self.assertEqual(receipt["consumption_events"], confirmed)
        self.assertEqual(receipt["evidence"], [str(frame.screenshot_path) for frame in session.frames])

    def test_capture_failure_preserves_receipt_even_when_last_snapshot_is_cleared(self) -> None:
        session = FakeSession(self.clock)
        original_observe = session.observe

        def fail_capture(label: str) -> SceneSnapshot:
            if label == "deploy-selected" and session.counts[215] == 9:
                session.last_snapshot = None
                raise CaptureError("Screenshot capture failed")
            return original_observe(label)

        session.observe = fail_capture
        with self.assertRaises(DeploymentError) as stopped:
            deploy_army(session, session.scout)
        self.assertIsNone(session.last_snapshot)
        self.assertIsInstance(stopped.exception.__cause__, CaptureError)
        receipt = stopped.exception.partial_receipt
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 1)
        self.assertEqual([event["consumed"] for event in receipt["consumption_events"]], [1])
        self.assertEqual(receipt["evidence"], [str(frame.screenshot_path) for frame in session.frames])
        self.assertEqual(len(session.placements), 1)

    def test_support_selection_failure_remains_attempted_without_extra_verified_units(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card("hero", None, 280)])
        original_observe = session.observe

        def fail_capture(label: str) -> SceneSnapshot:
            if label == "deploy-support-selected":
                session.last_snapshot = None
                raise CaptureError("Support screenshot failed")
            return original_observe(label)

        session.observe = fail_capture
        with self.assertRaises(DeploymentError) as stopped:
            deploy_army(session, session.scout)
        receipt = stopped.exception.partial_receipt
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 1)
        attempt = receipt["support_attempts"][0]
        self.assertEqual(attempt["status"], "attempted")
        self.assertFalse(attempt["verified"])
        self.assertIsNone(attempt["placement_frame"])
        self.assertIn(attempt["selection_frame"], receipt["evidence"])
        self.assertEqual(len(session.placements), 1)

    def test_fresh_frame_is_retained_if_observation_log_fails_before_returning(self) -> None:
        session = FakeSession(self.clock)
        original_observe = session.observe

        def fail_observation_log(label: str) -> SceneSnapshot:
            snapshot = original_observe(label)
            if label == "deploy-selected" and session.counts[215] == 9:
                raise OSError("Observation event could not be written")
            return snapshot

        session.observe = fail_observation_log
        with self.assertRaises(DeploymentError) as stopped:
            deploy_army(session, session.scout)
        receipt = stopped.exception.partial_receipt
        self.assertIsInstance(stopped.exception.__cause__, OSError)
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 1)
        self.assertEqual(receipt["evidence"], [str(frame.screenshot_path) for frame in session.frames])
        self.assertEqual(len(session.placements), 1)

    def test_failure_after_spell_consumption_keeps_spell_counts_separate(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card("spell", 2, 280)])
        original_observe = session.observe

        def fail_final_capture(label: str) -> SceneSnapshot:
            if label == "deployment-complete":
                session.last_snapshot = None
                raise CaptureError("Final deployment frame unavailable")
            return original_observe(label)

        session.observe = fail_final_capture
        with self.assertRaises(DeploymentError) as stopped:
            deploy_army(session, session.scout)
        receipt = stopped.exception.partial_receipt
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["deployed_units"], 1)
        spells = [event for event in receipt["consumption_events"] if event["unit_kind"] == "spell"]
        self.assertEqual(sum(event["consumed"] for event in spells), 2)
        self.assertEqual(receipt["evidence"], [str(frame.screenshot_path) for frame in session.frames])

    def test_keyboard_interrupt_after_consumption_is_not_wrapped_or_resumed(self) -> None:
        session = FakeSession(self.clock)

        def stop_after_first_unit(label: str, frame: SceneSnapshot) -> SceneSnapshot:
            if label == "deploy-selected" and session.counts[215] == 9:
                raise KeyboardInterrupt()
            return frame

        session.observe_hook = stop_after_first_unit
        with self.assertRaises(KeyboardInterrupt):
            deploy_army(session, session.scout)
        self.assertEqual(len(session.placements), 1)
        self.assertEqual(sum(kind == "deployment_consumed" for kind, _ in session.events), 1)

    def test_spells_are_bounded_and_do_not_inflate_troop_count(self) -> None:
        session = FakeSession(self.clock, [card(count=1), card("spell", 10, 280)])
        result = deploy_army(session, session.scout).observations["deployment"]
        self.assertEqual(result["deployed_units"], 1)
        self.assertEqual(session.counts[315], 0)
        spell_events = [event for event in result["consumption_events"] if event["unit_kind"] == "spell"]
        self.assertEqual(sum(event["consumed"] for event in spell_events), 10)
        self.assertTrue(all(event["consumed"] <= 8 for event in spell_events))


if __name__ == "__main__":
    unittest.main()
