from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from autococ.donation import execute_donation, read_donation_view
from autococ.errors import FlowError
from autococ.scene import SceneSnapshot


def frame(index, *, filled=0, own_units=0, elixir=10000, gems=100, request_id="fixture-request", units=None):
    path = Path(f"synthetic-donation-{index}.png")
    return SceneSnapshot("donation", 0.99, path, {"donation_view": {
        "frame": str(path), "request_id": request_id, "filled": filled, "capacity": 2,
        "own_units": own_units, "gems": gems, "elixir": elixir, "dark_elixir": 5000,
        "units": units if units is not None else [{"id": "synthetic-unit", "bbox": [100, 200, 200, 300],
            "space": 1, "cost": 1000, "currency": "elixir", "allowed": True, "enabled": True}]}})


class Session:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.config = SimpleNamespace(runtime=SimpleNamespace(poll_interval_sec=0))
        self.last_snapshot = None
        self.taps, self.events = [], []

    def check_deadline(self):
        pass

    def observe(self, label=""):
        try:
            self.last_snapshot = next(self.frames)
        except StopIteration:
            raise FlowError("fixture observation deadline")
        return self.last_snapshot

    def tap(self, snapshot, point, *, reason):
        if snapshot is not self.last_snapshot:
            raise AssertionError("stale fixture frame")
        self.taps.append((snapshot.screenshot_path, point))

    def event(self, kind, **data):
        self.events.append((kind, data))


class DonationTests(unittest.TestCase):
    def metrics(self):
        return {"donated_units": 0, "donation_cost_gems": 0, "donation_cost_gold": 0,
                "donation_cost_elixir": 0, "donation_cost_dark_elixir": 0}

    def test_two_units_require_two_distinct_verified_receipts(self):
        session = Session([frame(0), frame(1, filled=1, own_units=1, elixir=9000),
                           frame(2, filled=2, own_units=2, elixir=8000)])
        metrics, evidence = self.metrics(), []
        status, reason = execute_donation(session, session.observe(), evidence, metrics)
        self.assertEqual(status, "succeeded")
        self.assertEqual(metrics["donated_units"], 2)
        self.assertEqual(metrics["donation_cost_elixir"], 2000)
        self.assertEqual(metrics["donation_cost_gems"], 0)
        self.assertEqual(len(session.taps), 2)
        self.assertEqual(len(session.events), 2)
        self.assertEqual(len(evidence), 2)

    def test_delayed_result_is_reobserved_without_repeat_click(self):
        session = Session([frame(0), frame(1), frame(2, filled=2, own_units=1, elixir=9000)])
        metrics = self.metrics()
        result = execute_donation(session, session.observe(), [], metrics)
        self.assertEqual(result[0], "succeeded")
        self.assertEqual(metrics["donated_units"], 1)
        self.assertEqual(len(session.taps), 1)

    def test_other_player_filling_request_is_not_our_donation(self):
        session = Session([frame(0)] + [frame(i, filled=2) for i in range(1, 4)])
        metrics = self.metrics()
        with self.assertRaisesRegex(FlowError, "unverified"):
            execute_donation(session, session.observe(), [], metrics)
        self.assertEqual(metrics["donated_units"], 0)
        self.assertIsNone(metrics["donation_cost_elixir"])
        self.assertEqual(len(session.taps), 1)

    def test_unreadable_real_dialog_cannot_click_a_unit(self):
        session = Session([SceneSnapshot("donation", 0.99, Path("no-layout.png"))])
        with self.assertRaisesRegex(FlowError, "donation_layout_unverified"):
            execute_donation(session, session.observe(), [], self.metrics())
        self.assertEqual(session.taps, [])

    def test_gem_disabled_disallowed_expensive_and_oversized_units_are_not_clicked(self):
        original = frame(0).observations["donation_view"]["units"][0]
        for change in ({"currency": "gems"}, {"enabled": False}, {"allowed": False}, {"cost": 10001}, {"space": 3}):
            with self.subTest(change=change):
                session = Session([frame(0, units=[{**original, **change}])])
                result = execute_donation(session, session.observe(), [], self.metrics())
                self.assertEqual(result[0], "skipped")
                self.assertEqual(session.taps, [])

    def test_gem_change_or_request_replacement_stops_without_retry(self):
        for after in (frame(1, gems=99), frame(1, request_id="different-fixture")):
            with self.subTest(after=after):
                session = Session([frame(0), after])
                with self.assertRaises(FlowError):
                    execute_donation(session, session.observe(), [], self.metrics())
                self.assertEqual(len(session.taps), 1)

    def test_partial_verified_work_is_retained_after_later_failure(self):
        session = Session([frame(0), frame(1, filled=1, own_units=1, elixir=9000)] +
                          [frame(i, filled=1, own_units=1, elixir=9000) for i in range(2, 5)])
        metrics = self.metrics()
        with self.assertRaisesRegex(FlowError, "unverified"):
            execute_donation(session, session.observe(), [], metrics)
        self.assertEqual(metrics["donated_units"], 1)
        self.assertEqual(len(metrics["verified_donations"]), 1)
        self.assertEqual(len(session.taps), 2)

    def test_stale_provenance_invalid_numbers_and_ambiguous_choices_are_rejected(self):
        for change in ({"frame": "different.png"}, {"gems": None}, {"capacity": False},
                       {"filled": 3}, {"units": [{"id": "unreadable"}]}):
            with self.subTest(change=change):
                snapshot = deepcopy(frame(0))
                snapshot.observations["donation_view"].update(change)
                with self.assertRaises(FlowError):
                    read_donation_view(snapshot)
