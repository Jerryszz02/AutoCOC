"""Current-client HOME HUD proof when the shop caption animates away."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import unittest

import cv2

from autococ.flow import return_to_village
from autococ.home_evidence import home_navigation_verified
from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures"
PROMO = FIXTURES / "home_promo_current_18_647_2.png"
MODAL = FIXTURES / "home_event_modal_current_18_647_2.png"
HOME_OCR = [
    OCRText("28370645", .99979, (1090, 25, 1220, 52)),
    OCRText("16 093 534", .96815, (1094, 71, 1220, 101)),
    OCRText("470 000", .95488, (1115, 123, 1217, 148)),
    OCRText("241", .99999, (1176, 174, 1220, 199)),
    OCRText("建造", .99998, (1174, 674, 1217, 700)),
]


class HomeEvidenceTests(unittest.TestCase):
    def _recognize(self, path=PROMO, *, rows=None, version="18.647.2"):
        provider = Mock()
        provider.recognize.return_value = list(HOME_OCR if rows is None else rows)
        provider.recognize_line.return_value = []
        recognizer = ScreenshotRecognizer(provider=provider)
        recognizer.client_version = version
        return recognizer.recognize(path)

    def test_animated_shop_caption_has_source_bound_read_only_home(self):
        shot = self._recognize()
        self.assertEqual(shot.scene, "village")
        self.assertEqual([button["name"] for button in shot.observations["buttons"]], ["attack"])
        self.assertEqual(shot.observations["resources"], {
            "gold": 28370645, "elixir": 16093534, "dark_elixir": 470000, "gems": 241})
        self.assertTrue(home_navigation_verified(shot))
        self.assertEqual(shot.observations["home_hud_proof"]["client_version"], "18.647.2")
        session = Mock()
        session.observe.return_value = shot
        returned = return_to_village(session)
        self.assertIs(returned, shot)
        session.back.assert_not_called()
        session.click.assert_not_called()

    def test_existing_shop_caption_still_uses_old_two_control_contract(self):
        shot = self._recognize(rows=HOME_OCR + [OCRText("商店", .99, (1190, 376, 1235, 407))])
        self.assertEqual(shot.scene, "village")
        self.assertIsNone(shot.observations["home_hud_proof"])
        self.assertEqual({button["name"] for button in shot.observations["buttons"]}, {"attack", "shop"})
        self.assertTrue(home_navigation_verified(shot))

    def test_missing_conflicting_or_relocated_independent_anchors_reject(self):
        variants = [
            HOME_OCR[:-1],
            HOME_OCR[:3] + HOME_OCR[4:],
            HOME_OCR + [HOME_OCR[-1]],
            HOME_OCR[:3] + [OCRText("241", .99999, (800, 174, 850, 199))] + HOME_OCR[4:],
            HOME_OCR + [OCRText("28370646", .99, (1090, 25, 1220, 52))],
            HOME_OCR[:1] + [OCRText("16 093 534", .94, (1094, 71, 1220, 101))] + HOME_OCR[2:],
        ]
        for rows in variants:
            with self.subTest(rows=rows):
                shot = self._recognize(rows=rows)
                self.assertEqual(shot.scene, "unknown")
                self.assertFalse(home_navigation_verified(shot))

    def test_wrong_version_or_missing_attack_icon_reject(self):
        self.assertEqual(self._recognize(version="unknown").scene, "unknown")
        self.assertEqual(self._recognize(FIXTURES / "builder_base_anchor_20260926.png").scene, "unknown")
        with TemporaryDirectory() as directory:
            source = cv2.imread(str(PROMO))
            source[570:700, 25:145] = 0
            path = Path(directory) / "no-icon.png"
            path.write_bytes(cv2.imencode(".png", source)[1].tobytes())
            self.assertEqual(self._recognize(path).scene, "unknown")

    def test_modal_disconnected_account_and_battle_are_not_home(self):
        modal_rows = HOME_OCR + [
            OCRText("活动", .99991, (609, 36, 672, 76)),
            OCRText("X", .90786, (1034, 32, 1076, 74)),
            OCRText("今日活动", .99982, (214, 100, 281, 124)),
        ]
        modal = self._recognize(MODAL, rows=modal_rows)
        self.assertEqual(modal.scene, "popup")
        self.assertFalse(home_navigation_verified(modal))
        for extra, expected in (
            (OCRText("重新载入游戏", .99, (350, 400, 500, 435)), "disconnected"),
            (OCRText("结束战斗", .99, (50, 520, 120, 560)), "battle"),
            (OCRText("另一个设备正在连接这个村庄。", .99, (320, 330, 650, 370)), "unknown"),
        ):
            with self.subTest(extra=extra):
                shot = self._recognize(rows=HOME_OCR + [extra])
                self.assertEqual(shot.scene, expected)
                self.assertFalse(home_navigation_verified(shot))

    def test_proof_rejects_changed_frame_ocr_resource_roi_version_and_template(self):
        shot = self._recognize()
        self.assertTrue(home_navigation_verified(shot))
        changes = [
            ("ocr", lambda o: o["ocr"].append({"text": "另一个设备正在连接这个村庄。"})),
            ("resource", lambda o: o["resources"].__setitem__("elixir", 1)),
            ("dimension", lambda o: o.__setitem__("source_resolution", [2560, 1440])),
            ("dimension_claim", lambda o: (o.__setitem__("source_resolution", [2560, 1440]),
                                            o["home_hud_proof"].__setitem__("source_resolution", [2560, 1440]))),
            ("attack_point", lambda o: o["buttons"][0].__setitem__("point", [50, 600])),
            ("missing_attack", lambda o: o.__setitem__("buttons", [])),
            ("duplicate_attack", lambda o: o["buttons"].append(deepcopy(o["buttons"][0]))),
            ("version", lambda o: o["home_hud_proof"].__setitem__("client_version", "18.600.7")),
            ("roi", lambda o: o["home_hud_proof"]["builder"].__setitem__("bbox", [0, 0, 20, 20])),
            ("template", lambda o: o["home_hud_proof"].__setitem__("attack_template_sha256", "0" * 64)),
            ("frame_hash", lambda o: o["home_hud_proof"].__setitem__("frame_sha256", "0" * 64)),
        ]
        for label, mutate in changes:
            with self.subTest(label=label):
                observations = deepcopy(shot.observations)
                mutate(observations)
                self.assertFalse(home_navigation_verified(replace(shot, observations=observations)))
        with TemporaryDirectory() as directory:
            other = Path(directory) / "other.png"
            other.write_bytes(PROMO.read_bytes())
            self.assertFalse(home_navigation_verified(replace(shot, screenshot_path=other)))


if __name__ == "__main__":
    unittest.main()
