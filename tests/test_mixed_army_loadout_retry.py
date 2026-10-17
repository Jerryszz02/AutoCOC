"""The after-fill loadout retry uses only new complete current-army frames."""

from pathlib import Path
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import time

from autococ.army_control import reobserve_mixed_archer_missing_loadout, _read_current
from autococ.errors import StopRequested
from autococ.reporting import TaskResult
from test_mixed_army_fill import current


class _Session:
    client_version = "18.647.2"
    action_count = 0
    _input_attempt_generation = 0

    def __init__(self, initial, following):
        self.last_snapshot = initial
        self.following = iter(following)
        self.observes = []
        self.capture = SimpleNamespace(adb=object(), serial="offline",
                                       input_display_id=2, screenshot_display_id="physical-2")
        source = initial.screenshot_path.parent / "offline-config.toml"
        source.write_text("[runtime]\n", encoding="utf-8")
        self.config = SimpleNamespace(
            source_path=source,
            runtime=SimpleNamespace(step_timeout_sec=5),
            game=SimpleNamespace(package_name="com.supercell.clashofclans", display_id=2))

    def check_deadline(self):
        pass

    def observe(self, label):
        self.observes.append(label)
        self.last_snapshot = next(self.following)
        return self.last_snapshot


class MixedArmyLoadoutRetryTests(TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.counter = 0

    def shot(self, name, *, loadout=True, duplicate_bytes=False):
        self.counter += 1
        path = self.root / name
        path.write_bytes(b"same" if duplicate_bytes else f"frame-{self.counter}".encode())
        shot = current(str(path), capacity=343)
        shot.observations["army"]["hero_evidence"] = {
            "complete": True, "unknowns": [], "frame": str(path),
            "cards": [{"unit_id": "grand_warden", "kind": "hero", "source": "army",
                       "count": 1, "evidence": {"identity": {
                           "unit_id": "grand_warden", "confidence": .99,
                           "version_verified": True,
                           "template_client_version": "18.647.2"}}}],
        }
        if not loadout:
            shot.observations["army"]["hero_loadout_complete"] = False
            shot.observations["army"]["hero_loadout"] = {}
        shot.observations.update(frame_id=f"id-{self.counter}",
                                 observed_at_monotonic=time.monotonic() - 1 + self.counter * .0001)
        return shot

    @staticmethod
    def receipt(initial):
        observed, issue = _read_current(initial, full=True)
        assert issue is None
        return TaskResult("prepare_army", "succeeded", "already full", evidence=[initial.screenshot_path],
                          metrics={"recipe_mutations": 0, "archer_target": 0,
                                   "observed": observed,
                                   "verified_loadout": {"grand_warden": {
                                       "pet_id": "owl", "equipment_ids": ["book", "gem"]}},
                                   "final_frame": "previous-preparation.png"})

    def replay(self, initial, following, prepared=None):
        session = _Session(initial, following)
        prepared = prepared or self.receipt(initial)
        evidence = []
        with patch("autococ.device.foreground_package", return_value="com.supercell.clashofclans"), \
             patch("autococ.device.resolve_game_display",
                   return_value=SimpleNamespace(logical_id=2, physical_id="physical-2")):
            shot, issue = reobserve_mixed_archer_missing_loadout(session, initial, prepared, evidence)
        return session, shot, issue, evidence

    def test_second_new_full_frame_can_authorize_after_one_transient_gap(self):
        initial = self.shot("after-fill.png", loadout=False)
        transient = self.shot("reread-1.png", loadout=False)
        recovered = self.shot("reread-2.png")
        session, shot, issue, evidence = self.replay(initial, [transient, recovered])
        self.assertIsNone(issue)
        self.assertIs(shot, recovered)
        self.assertEqual(evidence, [transient.screenshot_path, recovered.screenshot_path])
        self.assertEqual(len(session.observes), 2)

    def test_two_more_gaps_exhaust_without_borrowing_preparation_loadout(self):
        initial = self.shot("after-fill.png", loadout=False)
        session, _, issue, evidence = self.replay(initial, [self.shot("one.png", loadout=False),
                                                           self.shot("two.png", loadout=False)])
        self.assertEqual(issue, "mixed_army_hero_loadout_unverified")
        self.assertEqual(len(evidence), 2)
        self.assertEqual(len(session.observes), 2)

    def test_other_current_army_failure_is_not_retried(self):
        initial = self.shot("after-fill.png", loadout=False)
        initial.observations["army"]["troops"]["used"] = 342
        session, _, issue, evidence = self.replay(initial, [])
        self.assertNotEqual(issue, "mixed_army_hero_loadout_unverified")
        self.assertEqual(session.observes, [])
        self.assertEqual(evidence, [])

    def test_mutated_army_does_not_get_extra_frames(self):
        initial = self.shot("after-fill.png", loadout=False)
        prepared = self.receipt(initial)
        prepared.metrics["recipe_mutations"] = 1
        session, _, issue, _ = self.replay(initial, [], prepared)
        self.assertEqual(issue, "mixed_army_hero_loadout_unverified")
        self.assertEqual(session.observes, [])

    def test_missing_loadout_does_not_mask_current_spell_roster_change(self):
        initial = self.shot("drift-initial.png", loadout=False)
        prepared = self.receipt(initial)
        initial.observations["army"]["manifest"]["spells"][0]["count"] = 5
        session, _, issue, _ = self.replay(initial, [], prepared)
        self.assertEqual(issue, "mixed_army_changed_before_attack")
        self.assertEqual(session.observes, [])

        initial = self.shot("drift-after-fill.png", loadout=False)
        prepared = self.receipt(initial)
        changed = self.shot("drift-reread.png", loadout=False)
        changed.observations["army"]["manifest"]["spells"][0]["count"] = 5
        session, shot, issue, evidence = self.replay(initial, [changed,
                                                               self.shot("not-observed.png")], prepared)
        self.assertIs(shot, changed)
        self.assertEqual(issue, "mixed_army_changed_before_attack")
        self.assertEqual(evidence, [changed.screenshot_path])
        self.assertEqual(len(session.observes), 1)

    def test_fresh_frame_violations_and_other_strong_failures_stop(self):
        for defect in ("same_path", "same_hash", "old_time", "wrong_version",
                       "changed_roster", "changed_capacity", "hero_evidence", "loadout_drift"):
            with self.subTest(defect=defect):
                initial = self.shot(f"{defect}-initial.png", loadout=False)
                fresh = self.shot(f"{defect}-fresh.png")
                if defect == "same_path":
                    fresh = replace(fresh, screenshot_path=initial.screenshot_path)
                elif defect == "same_hash":
                    fresh.screenshot_path.write_bytes(initial.screenshot_path.read_bytes())
                elif defect == "old_time":
                    fresh.observations["observed_at_monotonic"] = initial.observations["observed_at_monotonic"]
                elif defect == "changed_roster":
                    fresh.observations["army"]["manifest"]["troops"][0]["count"] = 9
                elif defect == "changed_capacity":
                    fresh.observations["army"]["spells"]["capacity"] = 12
                elif defect == "hero_evidence":
                    fresh.observations["army"]["hero_evidence"]["cards"][0]["evidence"]["identity"]["version_verified"] = False
                elif defect == "loadout_drift":
                    fresh.observations["army"]["hero_loadout"]["grand_warden"]["pet_id"] = "fox"
                session = _Session(initial, [fresh])
                if defect == "wrong_version":
                    session.client_version = "18.647.1"
                evidence = []
                with patch("autococ.device.foreground_package", return_value="com.supercell.clashofclans"), \
                     patch("autococ.device.resolve_game_display",
                           return_value=SimpleNamespace(logical_id=2, physical_id="physical-2")):
                    _, issue = reobserve_mixed_archer_missing_loadout(
                        session, initial, self.receipt(initial), evidence)
                self.assertIsNotNone(issue)
                self.assertLessEqual(len(evidence), 1)

    def test_foreground_or_display_change_prevents_reread(self):
        for changed in ("foreground", "display"):
            with self.subTest(changed=changed):
                initial = self.shot(f"{changed}-initial.png", loadout=False)
                session = _Session(initial, [self.shot(f"{changed}-fresh.png")])
                with patch("autococ.device.foreground_package", return_value=(
                        "other" if changed == "foreground" else "com.supercell.clashofclans")), \
                     patch("autococ.device.resolve_game_display", return_value=SimpleNamespace(
                         logical_id=3 if changed == "display" else 2, physical_id="physical-2")):
                    _, issue = reobserve_mixed_archer_missing_loadout(session, initial,
                                                                      self.receipt(initial), [])
                self.assertEqual(issue, "mixed_army_loadout_retry_environment_changed")
                self.assertEqual(session.observes, [])

    def test_stop_propagates_and_input_attempt_change_blocks_new_authorization(self):
        initial = self.shot("stop-initial.png", loadout=False)
        session = _Session(initial, [self.shot("stop-fresh.png")])
        session.check_deadline = lambda: (_ for _ in ()).throw(StopRequested("stopped"))
        with self.assertRaises(StopRequested):
            reobserve_mixed_archer_missing_loadout(session, initial, self.receipt(initial), [])
        self.assertEqual(session.observes, [])

        initial = self.shot("input-initial.png", loadout=False)
        session = _Session(initial, [self.shot("input-fresh.png")])
        session._input_attempt_generation = 1
        original_observe = session.observe

        def observe_after_input(label):
            result = original_observe(label)
            session._input_attempt_generation += 1
            return result

        session.observe = observe_after_input
        with patch("autococ.device.foreground_package", return_value="com.supercell.clashofclans"), \
             patch("autococ.device.resolve_game_display",
                   return_value=SimpleNamespace(logical_id=2, physical_id="physical-2")):
            _, issue = reobserve_mixed_archer_missing_loadout(
                session, initial, self.receipt(initial), [])
        self.assertEqual(issue, "mixed_army_loadout_retry_frame_or_environment_changed")

    def test_source_change_after_capture_blocks_authorization(self):
        initial = self.shot("source-initial.png", loadout=False)
        session = _Session(initial, [self.shot("source-fresh.png")])
        original_observe = session.observe

        def observe_after_source_change(label):
            result = original_observe(label)
            session.config.source_path.write_text("[runtime]\nchanged=true\n", encoding="utf-8")
            return result

        session.observe = observe_after_source_change
        with patch("autococ.device.foreground_package", return_value="com.supercell.clashofclans"), \
             patch("autococ.device.resolve_game_display",
                   return_value=SimpleNamespace(logical_id=2, physical_id="physical-2")):
            _, issue = reobserve_mixed_archer_missing_loadout(
                session, initial, self.receipt(initial), [])
        self.assertEqual(issue, "mixed_army_loadout_retry_frame_or_environment_changed")
