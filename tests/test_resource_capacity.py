"""Capacity evidence, including optional replay of the 2026-10-15 HUD probes."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from autococ.combat import _reconcile_return_inventory
from autococ.config import OCRConfig
from autococ.errors import FlowError, StopRequested
from autococ.ocr import OCRText, RapidOCRProvider
from autococ.resource_capacity import (_tooltip_border_absent, read_resource_capacity,
                                       read_tooltip_capacity, verified_unknown_tooltip)
from autococ.scene import SceneSnapshot
from autococ.session import GameSession
from autococ.vision import ScreenshotRecognizer


PROBES = Path(r"E:\projects\AutoCOC\reports\completion-20261015\probes")
ANONYMOUS_GOLD = Path(__file__).parent / "fixtures" / "resource_capacity_18_647_2"
ANONYMOUS_DARK = Path(__file__).parent / "fixtures" / "resource_capacity_18_647_2_dark_20261016"
ANONYMOUS_135135 = Path(__file__).parent / "fixtures" / "resource_capacity_gold_135135"
INVENTORY = {"gold": 100, "elixir": 200, "dark_elixir": 30, "gems": 5}


def home(name="home.png", *, scene="village", resources=None):
    path = Path(name)
    return SceneSnapshot(scene, .95, path, {
        "village_type": "home", "village_type_evidence": {"type": "home", "frame": str(path)},
        "resource_source": "village_inventory", "resources": INVENTORY.copy() if resources is None else resources,
        "resource_evidence": {key: {"confidence": .99} for key in ("gold", "elixir", "dark_elixir")},
        "buttons": [{"name": "attack", "confidence": .95}], "baseline_resolution": [1280, 720],
        "frame_id": name, "observed_at_monotonic": time.monotonic(),
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
    def test_135135_anonymous_unknown_tooltip_has_strict_source_bound_proof(self):
        provenance = json.loads((ANONYMOUS_135135 / "provenance.json").read_text(encoding="utf-8"))
        provider = RapidOCRProvider()
        for name in ("home", "opening", "settled"):
            path = ANONYMOUS_135135 / f"{name}.png"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             provenance["fixtures"][name]["derived_sha256"])
            proof = read_tooltip_capacity(path, "gold", provider)
            if name != "settled":
                self.assertIsNone(proof)
                self.assertTrue(_tooltip_border_absent(path, "gold"))
                continue
            self.assertEqual(proof["capacity"], 29_000_000)
            self.assertEqual(proof["ocr"]["label_source"], "native_2x")
            self.assertTrue(all(proof["ocr"][key] >= .95 for key in (
                "label_raw", "label_white", "number_raw", "number_white")))
            detail = ScreenshotRecognizer(OCRConfig(provider="rapidocr"), (1280, 720)).recognize(path)
            self.assertEqual(detail.scene, "unknown")
            self.assertTrue(verified_unknown_tooltip(detail, "gold", proof, provider))
            detail.observations["account_in_use"] = {"message": "another device"}
            self.assertFalse(verified_unknown_tooltip(detail, "gold", proof, provider))
            detail.observations["account_in_use"] = None
            detail.observations["request_dialog"] = {"title": "request"}
            self.assertFalse(verified_unknown_tooltip(detail, "gold", proof, provider))
            detail.observations["request_dialog"] = None
            detail.observations["ocr"].append({"text": "另一个设备正在连接这个村庄。", "confidence": .99})
            self.assertFalse(verified_unknown_tooltip(detail, "gold", proof, provider))
            detail.observations["ocr"].pop()
            self.assertFalse(verified_unknown_tooltip(detail, "gold", {**proof, "capacity": 28_000_000}, provider))
            with patch("autococ.resource_capacity.recognize_village_type",
                       return_value={"type": "builder_base", "frame": str(path)}):
                self.assertFalse(verified_unknown_tooltip(detail, "gold", proof, provider))

    def test_native_label_fallback_rejects_conflict_and_number_disagreement(self):
        path = ANONYMOUS_135135 / "settled.png"
        common = [("最大储量:", .917), ("最大储量：", .999),
                  ("29000000", .999), ("29 000 000", .999), ("最大储量:", .917)]
        self.assertIsNone(read_tooltip_capacity(path, "gold", Provider(common + [
            ("最大储量：", .999), ("每小时产量：", .999)])))
        conflict = common.copy()
        conflict[3] = ("28 000 000", .999)
        self.assertIsNone(read_tooltip_capacity(path, "gold", Provider(conflict + [
            ("最大储量：", .999), ("最大储量：", .999)])))
        # A strong, conflicting baseline white label must not be overwritten.
        white_conflict = common.copy()
        white_conflict[1] = ("每小时产量：", .999)
        self.assertIsNone(read_tooltip_capacity(path, "gold", Provider(white_conflict)))
        # The native fallback is an atomic pair; old baseline white cannot
        # substitute for a missing or conflicting native white result.
        self.assertIsNone(read_tooltip_capacity(path, "gold", Provider(common + [
            ("最大储量：", .999), ("每小时产量：", .91)])))

    def test_tooltip_source_must_remain_unchanged_during_ocr(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mutable.png"
            path.write_bytes(b"before")
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            image[61, 984:1258] = 255

            class MutatingProvider(Provider):
                def recognize_line_image(self, image):
                    path.write_bytes(b"after")
                    return super().recognize_line_image(image)

            texts = [("最大储量：", .99), ("最大储量：", .99),
                     ("29000000", .99), ("29 000 000", .99)]
            with patch("autococ.resource_capacity.read_frame", return_value=image):
                self.assertIsNone(read_tooltip_capacity(path, "gold", MutatingProvider(texts)))

    def test_unknown_tooltip_close_is_one_source_bound_input(self):
        path = ANONYMOUS_135135 / "settled.png"
        provider = RapidOCRProvider()
        proof = read_tooltip_capacity(path, "gold", provider)
        detail = ScreenshotRecognizer(OCRConfig(provider="rapidocr"), (1280, 720)).recognize(path)
        detail.observations["observed_at_monotonic"] = time.monotonic()
        sent = []
        session = GameSession.__new__(GameSession)
        session.client_version, session.client_version_code = "18.647.2", 180647003
        session.config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720),
                                                              package_name="com.supercell.clashofclans"),
                                         runtime=SimpleNamespace(step_timeout_sec=5))
        session.capture = SimpleNamespace(adb=object(), serial="fake", input_display_id=1)
        session.context = SimpleNamespace(capture=session.capture, screen_resolution=(2560, 1440),
                                          tap_xy=lambda *args, **kwargs: sent.append((args, kwargs)))
        session.recognizer = SimpleNamespace(provider=provider)
        session.last_snapshot = detail
        session._active_resource_capacity = {"resource": "gold", "input_generation": 1}
        session._input_attempt_generation = 1
        session.action_count = 1
        session.check_deadline = lambda: None
        session.event = lambda *args, **kwargs: None
        with patch("autococ.session.foreground_package", return_value="com.supercell.clashofclans"):
            session.close_verified_resource_tooltip(detail, "gold", (1150, 37), proof)
            self.assertEqual(len(sent), 1)
            self.assertEqual(session.action_count, 2)
            self.assertEqual(session._input_attempt_generation, 2)
            self.assertIsNone(session.last_snapshot)
            with self.assertRaises(FlowError):
                session.close_verified_resource_tooltip(detail, "gold", (1150, 37), proof)
        self.assertEqual(len(sent), 1)
        for change in ("stale", "stop", "foreground", "display", "modal", "expired_after_ocr"):
            with self.subTest(change=change):
                session.last_snapshot = detail
                session._active_resource_capacity = {"resource": "gold", "input_generation": 2}
                detail.observations["observed_at_monotonic"] = time.monotonic()
                session.check_deadline = lambda: None
                session.capture.input_display_id = 1
                detail.observations["request_dialog"] = None
                if change == "stale":
                    detail.observations["observed_at_monotonic"] -= 31
                elif change == "stop":
                    session.check_deadline = lambda: (_ for _ in ()).throw(StopRequested("stop"))
                elif change == "display":
                    session.capture.input_display_id = None
                elif change == "modal":
                    detail.observations["request_dialog"] = {"open": True}
                with patch("autococ.session.foreground_package", return_value=(
                        "other.package" if change == "foreground" else "com.supercell.clashofclans")):
                    if change == "expired_after_ocr":
                        real_monotonic = time.monotonic
                        readings = iter((real_monotonic(), real_monotonic() + 31))
                        with patch("autococ.session.time.monotonic",
                                   side_effect=lambda: next(readings, real_monotonic() + 31)):
                            with self.assertRaises(FlowError):
                                session.close_verified_resource_tooltip(detail, "gold", (1150, 37), proof)
                    else:
                        with self.assertRaises((FlowError, StopRequested)):
                            session.close_verified_resource_tooltip(detail, "gold", (1150, 37), proof)
                self.assertEqual(len(sent), 1)
        detail.observations["request_dialog"] = None
        detail.observations["observed_at_monotonic"] = time.monotonic()
        session.check_deadline = lambda: None
        session.capture.input_display_id = 1
        session.last_snapshot = detail
        session._active_resource_capacity = {"resource": "gold", "input_generation": 2}
        session.context.tap_xy = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("uncertain transport"))
        with patch("autococ.session.foreground_package", return_value="com.supercell.clashofclans"):
            with self.assertRaises(OSError):
                session.close_verified_resource_tooltip(detail, "gold", (1150, 37), proof)
            self.assertIsNone(session.last_snapshot)
            self.assertIsNone(session._active_resource_capacity)
            self.assertEqual(session._input_attempt_generation, 3)
            with self.assertRaises(FlowError):
                session.close_verified_resource_tooltip(detail, "gold", (1150, 37), proof)
        self.assertEqual(len(sent), 1)

    def test_anonymous_real_dark_elixir_tooltip_replay(self):
        from autococ.ocr import RapidOCRProvider

        provenance = json.loads((ANONYMOUS_DARK / "provenance.json").read_text(encoding="utf-8"))
        provider = RapidOCRProvider()
        for name in ("old_home", "new_home", "new_animation"):
            with self.subTest(frame=name):
                frame = ANONYMOUS_DARK / f"{name}.png"
                self.assertEqual(hashlib.sha256(frame.read_bytes()).hexdigest(),
                                 provenance["fixtures"][name]["derived_sha256"])
                self.assertIsNone(read_tooltip_capacity(frame, "dark_elixir", provider))
                self.assertTrue(_tooltip_border_absent(frame, "dark_elixir"))
        for name in ("old_settled", "new_settled"):
            with self.subTest(frame=name):
                frame = ANONYMOUS_DARK / f"{name}.png"
                self.assertEqual(hashlib.sha256(frame.read_bytes()).hexdigest(),
                                 provenance["fixtures"][name]["derived_sha256"])
                proof = read_tooltip_capacity(frame, "dark_elixir", provider)
                self.assertIsNotNone(proof)
                self.assertEqual(proof["capacity"], 470_000)
                self.assertGreaterEqual(proof["ocr"]["number_raw"], .95)
                self.assertGreaterEqual(proof["ocr"]["number_white"], .95)
                self.assertFalse(_tooltip_border_absent(frame, "dark_elixir"))

    def test_dark_elixir_rejects_clipped_or_conflicting_number(self):
        from autococ.ocr import RapidOCRProvider
        import cv2

        frame = ANONYMOUS_DARK / "new_settled.png"
        clipped = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        clipped[182:202, 1150:1190] = 24
        with patch("autococ.resource_capacity.read_frame", return_value=clipped):
            self.assertIsNone(read_tooltip_capacity(frame, "dark_elixir", RapidOCRProvider()))
        conflict = [("最大储量：", .99), ("最大储量：", .99),
                    ("470000", .99), ("480000", .99)]
        self.assertIsNone(read_tooltip_capacity(frame, "dark_elixir", Provider(conflict)))

    def test_anonymous_real_gold_tooltip_replay(self):
        from autococ.ocr import RapidOCRProvider

        provenance = json.loads((ANONYMOUS_GOLD / "provenance.json").read_text(encoding="utf-8"))
        provider = RapidOCRProvider()
        for name in ("old_home", "old_animation", "new_animation"):
            with self.subTest(frame=name):
                frame = ANONYMOUS_GOLD / f"{name}.png"
                self.assertEqual(hashlib.sha256(frame.read_bytes()).hexdigest(),
                                 provenance["fixtures"][name]["derived_sha256"])
                self.assertIsNone(read_tooltip_capacity(frame, "gold", provider))
                self.assertTrue(_tooltip_border_absent(frame, "gold"))
        for name in ("old_settled", "new_settled"):
            with self.subTest(frame=name):
                frame = ANONYMOUS_GOLD / f"{name}.png"
                self.assertEqual(hashlib.sha256(frame.read_bytes()).hexdigest(),
                                 provenance["fixtures"][name]["derived_sha256"])
                proof = read_tooltip_capacity(frame, "gold", provider)
                self.assertIsNotNone(proof)
                self.assertEqual(proof["capacity"], 29_000_000)
                self.assertGreaterEqual(proof["ocr"]["number_raw"], .95)
                self.assertGreaterEqual(proof["ocr"]["number_white"], .95)
                self.assertFalse(_tooltip_border_absent(frame, "gold"))

    def test_anonymous_gold_tooltip_rejects_clipped_or_conflicting_number(self):
        from autococ.ocr import RapidOCRProvider
        import cv2

        frame = ANONYMOUS_GOLD / "new_settled.png"
        clipped = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        clipped[78:99, 1108:1140] = 24
        with patch("autococ.resource_capacity.read_frame", return_value=clipped):
            self.assertIsNone(read_tooltip_capacity(frame, "gold", RapidOCRProvider()))
        conflict = [("最大储量：", .99), ("最大储量：", .99),
                    ("29000000", .99), ("28 000 000", .99)]
        self.assertIsNone(read_tooltip_capacity(frame, "gold", Provider(conflict)))

    def test_tooltip_requires_border_and_dual_ocr_agreement(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[61, 984:1258] = 255
        texts = [("最大储量：", .99), ("最大储量：", .99),
                 ("29000000", .99), ("29 000 000", .99)]
        with TemporaryDirectory() as directory, patch("autococ.resource_capacity.read_frame", return_value=image):
            frame = Path(directory) / "synthetic.png"
            frame.write_bytes(b"synthetic fixture; image supplied by read_frame")
            self.assertEqual(read_tooltip_capacity(frame, "gold", Provider(texts))["capacity"], 29_000_000)
            image[61] = 0
            self.assertIsNone(read_tooltip_capacity(frame, "gold", Provider(texts)))
            image[61, 984:1258] = 255
            wrong = texts[:3] + [("28 000 000", .99)]
            self.assertIsNone(read_tooltip_capacity(frame, "gold", Provider(wrong)))

    def test_reader_closes_only_verified_home_detail_and_checks_inventory(self):
        before, detail, closing, closed = home(), home("detail.png"), home("closing.png"), home("closed.png")
        session = ReaderSession([detail, closing, closed])
        proof = {"capacity": 500, "resource": "gold", "frame": str(detail.screenshot_path)}
        with patch("autococ.resource_capacity.read_tooltip_capacity", side_effect=[proof, None, None]), patch(
                "autococ.resource_capacity._tooltip_border_absent", return_value=True), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}):
            result, final, paths = read_resource_capacity(session, before, "gold")
        self.assertEqual(result["capacity"], 500)
        self.assertIs(final, closed)
        self.assertEqual(paths, [detail.screenshot_path, closing.screenshot_path, closed.screenshot_path])
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
        session = ReaderSession([detail, *(home(f"closing-{n}.png") for n in range(4))])
        proof = {"capacity": 500, "resource": "gold", "frame": str(detail.screenshot_path)}
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[61, 984:1258] = 255
        with patch("autococ.resource_capacity.read_tooltip_capacity", side_effect=[proof, None, None, None, None]), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}), patch(
                "autococ.resource_capacity.read_frame", return_value=image):
            with self.assertRaises(FlowError):
                read_resource_capacity(session, before, "gold")
        self.assertEqual(len(session.taps), 2)

    def test_reader_stops_on_modal_during_close_observation(self):
        before, detail = home(), home("detail.png")
        modal = home("account-modal.png", scene="unknown")
        modal.observations["account_in_use"] = {"message": "other device"}
        session = ReaderSession([detail, modal])
        proof = {"capacity": 500, "resource": "gold", "frame": str(detail.screenshot_path)}
        with patch("autococ.resource_capacity.read_tooltip_capacity", return_value=proof), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}):
            with self.assertRaisesRegex(FlowError, "another scene or modal"):
                read_resource_capacity(session, before, "gold")
        self.assertEqual(len(session.taps), 2)
        partial = home("partial-account.png", scene="unknown")
        partial.observations["ocr"] = [{"text": "另一个设备正在连接这个村庄。", "confidence": .99}]
        session = ReaderSession([detail, partial])
        with patch("autococ.resource_capacity.read_tooltip_capacity", return_value=proof), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}):
            with self.assertRaisesRegex(FlowError, "another scene or modal"):
                read_resource_capacity(session, before, "gold")
        self.assertEqual(len(session.taps), 2)

    def test_reader_waits_through_one_close_animation_for_two_clean_frames(self):
        before = home()
        opening, detail = home("opening.png", scene="unknown"), home("detail.png")
        closing, closed, closed_again = home("closing.png"), home("closed.png"), home("closed-again.png")
        session = ReaderSession([opening, detail, closing, closed, closed_again])
        proof = {"capacity": 500, "resource": "gold", "frame": str(detail.screenshot_path)}
        with patch("autococ.resource_capacity.read_tooltip_capacity", side_effect=[None, proof, proof, None, None]), patch(
                "autococ.resource_capacity._tooltip_border_absent", return_value=True), patch(
                "autococ.resource_capacity.recognize_village_type", return_value={"type": "home"}):
            result, final, paths = read_resource_capacity(session, before, "gold")
        self.assertEqual(result, proof)
        self.assertIs(final, closed_again)
        self.assertEqual(paths, [frame.screenshot_path for frame in (opening, detail, closing, closed, closed_again)])
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
        latest = PROBES / "20261016-074804-075059-capacity-all-v1" / "frames"
        if latest.is_dir():
            self.assertIsNone(read_tooltip_capacity(latest / "00002-gold-storage-detail.png", "gold", provider))
            settled = read_tooltip_capacity(latest / "00003-gold-storage-detail-settled.png", "gold", provider)
            self.assertEqual(settled["capacity"], 29_000_000)
            self.assertGreaterEqual(settled["ocr"]["number_raw"], .95)
            self.assertGreaterEqual(settled["ocr"]["number_white"], .95)


if __name__ == "__main__":
    unittest.main()
