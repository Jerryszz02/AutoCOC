"""Capacity evidence, including optional replay of the 2026-10-15 HUD probes."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from autococ.combat import _reconcile_return_inventory
from autococ.errors import FlowError
from autococ.ocr import OCRText
from autococ.resource_capacity import _tooltip_border_absent, read_resource_capacity, read_tooltip_capacity
from autococ.scene import SceneSnapshot


PROBES = Path(r"E:\projects\AutoCOC\reports\completion-20261015\probes")
INVENTORY = {"gold": 100, "elixir": 200, "dark_elixir": 30, "gems": 5}


def home(name="home.png", *, scene="village", resources=None):
    path = Path(name)
    return SceneSnapshot(scene, .95, path, {
        "village_type": "home", "village_type_evidence": {"type": "home", "frame": str(path)},
        "resource_source": "village_inventory", "resources": INVENTORY.copy() if resources is None else resources,
        "resource_evidence": {key: {"confidence": .99} for key in ("gold", "elixir", "dark_elixir")},
        "buttons": [{"name": "attack", "confidence": .95}], "baseline_resolution": [1280, 720],
    })


class Provider:
    def __init__(self, texts):
        self.texts = iter(texts)

    def recognize_line_image(self, image):
        return [OCRText(*next(self.texts))]


class ReaderSession:
    client_version = "18.647.2"

    def __init__(self, frames):
        self.frames = iter(frames)
        self.recognizer = SimpleNamespace(provider=Provider([]))
        self.taps = []

    def tap(self, snapshot, point, *, reason):
        self.taps.append((snapshot, point))

    def observe(self, label):
        return next(self.frames)


class CapacityTests(unittest.TestCase):
    def test_tooltip_requires_border_and_dual_ocr_agreement(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[61, 984:1258] = 255
        texts = [("最大储量：", .99), ("最大储量：", .99),
                 ("29000000", .99), ("29 000 000", .99)]
        with patch("autococ.resource_capacity.read_frame", return_value=image):
            self.assertEqual(read_tooltip_capacity(Path("synthetic.png"), "gold", Provider(texts))["capacity"],
                             29_000_000)
            image[61] = 0
            self.assertIsNone(read_tooltip_capacity(Path("synthetic.png"), "gold", Provider(texts)))
            image[61, 984:1258] = 255
            wrong = texts[:3] + [("28 000 000", .99)]
            self.assertIsNone(read_tooltip_capacity(Path("synthetic.png"), "gold", Provider(wrong)))

    def test_reader_closes_only_verified_home_detail_and_checks_inventory(self):
        before, detail, closed = home(), home("detail.png"), home("closed.png")
        session = ReaderSession([detail, closed])
        proof = {"capacity": 500, "resource": "gold", "frame": str(detail.screenshot_path)}
        with patch("autococ.resource_capacity.read_tooltip_capacity", side_effect=[proof, None]), patch(
                "autococ.resource_capacity._tooltip_border_absent", return_value=True), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}):
            result, final, paths = read_resource_capacity(session, before, "gold")
        self.assertEqual(result["capacity"], 500)
        self.assertIs(final, closed)
        self.assertEqual(paths, [detail.screenshot_path, closed.screenshot_path])
        self.assertEqual([point for _, point in session.taps], [(1150, 37), (1150, 37)])

        session = ReaderSession([home("unknown.png", scene="unknown"),
                                 home("unknown-settled.png", scene="unknown")])
        with patch("autococ.resource_capacity.read_tooltip_capacity", return_value=proof), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}):
            with self.assertRaises(FlowError):
                read_resource_capacity(session, before, "gold")
        self.assertEqual(len(session.taps), 1)

    def test_reader_rejects_unreadable_but_visibly_open_tooltip_after_close_tap(self):
        before, detail = home(), home("detail.png")
        session = ReaderSession([detail, home("closing.png"), home("closing-settled.png")])
        proof = {"capacity": 500, "resource": "gold", "frame": str(detail.screenshot_path)}
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[61, 984:1258] = 255
        with patch("autococ.resource_capacity.read_tooltip_capacity", side_effect=[proof, None, None]), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}), patch(
                "autococ.resource_capacity.read_frame", return_value=image):
            with self.assertRaises(FlowError):
                read_resource_capacity(session, before, "gold")
        self.assertEqual(len(session.taps), 2)

    def test_reader_observes_one_animation_frame_at_each_boundary_without_retapping(self):
        before = home()
        opening, detail = home("opening.png", scene="unknown"), home("detail.png")
        closing, closed = home("closing.png"), home("closed.png")
        session = ReaderSession([opening, detail, closing, closed])
        proof = {"capacity": 500, "resource": "gold", "frame": str(detail.screenshot_path)}
        with patch("autococ.resource_capacity.read_tooltip_capacity", side_effect=[None, proof, proof, None]), patch(
                "autococ.resource_capacity._tooltip_border_absent", return_value=True), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}):
            result, final, paths = read_resource_capacity(session, before, "gold")
        self.assertEqual(result, proof)
        self.assertIs(final, closed)
        self.assertEqual(paths, [frame.screenshot_path for frame in (opening, detail, closing, closed)])
        self.assertEqual(len(session.taps), 2)

    def test_reader_rejects_bad_inventory_or_confidence_before_input(self):
        bad = home(resources={**INVENTORY, "gems": None})
        session = ReaderSession([])
        with self.assertRaises(FlowError):
            read_resource_capacity(session, bad, "gold")
        self.assertEqual(session.taps, [])
        bad = home()
        bad.observations["resource_evidence"]["gold"]["confidence"] = float("nan")
        with self.assertRaises(FlowError):
            read_resource_capacity(session, bad, "gold")
        self.assertEqual(session.taps, [])

    def test_reconciliation_uses_verified_cap_only_for_positive_shortfall(self):
        settlement = {"loot": {"gold": 30, "elixir": 20, "dark_elixir": 0},
                      "bonus": {"gold": 10, "elixir": 0, "dark_elixir": 0}}
        before = INVENTORY.copy()
        after = {"gold": 120, "elixir": 220, "dark_elixir": 30, "gems": 5}
        session = SimpleNamespace(client_version="18.647.2")
        current = home(resources=after)
        proof = {"capacity": 120, "resource": "gold", "frame": "detail.png"}
        with patch("autococ.combat.read_resource_capacity", return_value=(proof, current, [Path("detail.png")])) as reader:
            metrics, _ = _reconcile_return_inventory(session, current, before, after, settlement, 0, [],
                                                     allow_capacity=True)
        reader.assert_called_once()
        self.assertTrue(metrics["inventory_reconciled"])
        self.assertEqual(metrics["inventory_reconciliation_method"], "capacity_capped")
        self.assertEqual((metrics["battle_credited_gold"], metrics["overflow_gold"]), (20, 20))

        exact = {**after, "gold": 140}
        with patch("autococ.combat.read_resource_capacity") as reader:
            metrics, _ = _reconcile_return_inventory(session, current, before, exact, settlement, 0, [],
                                                     allow_capacity=True)
        reader.assert_not_called()
        self.assertEqual(metrics["inventory_reconciliation_method"], "exact")
        self.assertEqual((metrics["battle_credited_gold"], metrics["overflow_gold"]), (40, 0))

        with patch("autococ.combat.read_resource_capacity") as reader:
            metrics, _ = _reconcile_return_inventory(session, current, before, after, settlement, 0, [],
                                                     allow_capacity=False)
        reader.assert_not_called()
        self.assertFalse(metrics["inventory_reconciled"])
        self.assertNotIn("battle_credited_gold", metrics)

        too_high = {**after, "gold": 141}
        with patch("autococ.combat.read_resource_capacity") as reader:
            metrics, _ = _reconcile_return_inventory(session, current, before, too_high, settlement, 0, [],
                                                     allow_capacity=True)
        reader.assert_not_called()
        self.assertFalse(metrics["inventory_reconciled"])

    def test_real_tooltip_probes_when_available(self):
        # Source frames stay outside Git because the village image contains the account name.
        if not PROBES.is_dir():
            self.skipTest("Local 2026-10-15 probe frames unavailable")
        from autococ.ocr import RapidOCRProvider

        provider = RapidOCRProvider()
        for directory, name, expected in (
            ("20261015-215828-tap", "gold", 29_000_000),
            ("20261015-220028-tap", "elixir", 29_000_000),
            ("20261015-220121-tap", "dark_elixir", 470_000),
        ):
            with self.subTest(resource=name):
                frames = PROBES / directory / "frames"
                self.assertIsNone(read_tooltip_capacity(frames / "00001-before.png", name, provider))
                self.assertTrue(_tooltip_border_absent(frames / "00001-before.png", name))
                self.assertFalse(_tooltip_border_absent(frames / "00002-after.png", name))
                self.assertEqual(read_tooltip_capacity(frames / "00002-after.png", name, provider)["capacity"],
                                 expected)
        recent = PROBES / "20261015-221320-capacity-reader" / "frames"
        if recent.is_dir():
            self.assertIsNone(read_tooltip_capacity(recent / "00002-gold-storage-detail.png", "gold", provider))
            settled = read_tooltip_capacity(recent / "00003-gold-storage-detail-settled.png", "gold", provider)
            self.assertEqual(settled["capacity"], 29_000_000)
            self.assertGreaterEqual(settled["ocr"]["number_raw"], .95)
            self.assertGreaterEqual(settled["ocr"]["number_white"], .95)


if __name__ == "__main__":
    unittest.main()
