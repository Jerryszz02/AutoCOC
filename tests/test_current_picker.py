"""The current-army picker must not be confused with a saved-plan editor."""

from pathlib import Path
import hashlib
import json
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf
from unittest.mock import patch
from importlib.util import find_spec

import cv2
import numpy as np

from autococ.army_editor import _picker_count, recognize_army_editor
from autococ.ocr import OCRText
from autococ.ocr import RapidOCRProvider
from autococ.vision import ScreenshotRecognizer
from autococ.unit_catalog import recognize_card_identity, template_manifest


class CurrentPickerTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / "picker.png"
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.image[460:665, 50:1200] = 230

    def tearDown(self):
        self.directory.cleanup()

    def recognize(self, texts, identities=None):
        cv2.imwrite(str(self.path), self.image)
        identities = identities or {}

        def identity(_image, box, kind, **_kwargs):
            unit_id = identities.get((kind, box[0]))
            return {"unit_id": unit_id, "confidence": .98 if unit_id else None,
                    "reason": "portrait_template" if unit_id else "no_supported_template"}

        with patch("autococ.army_editor.recognize_card_identity", side_effect=identity):
            return recognize_army_editor(self.path, None, texts, client_version="18.600.7")

    def minus(self, x, y):
        self.image[y-13:y+13, x-13:x+13] = (15, 20, 210)
        self.image[y-2:y+2, x-8:x+8] = (245, 245, 245)

    def test_troop_picker_counts_only_independently_named_top_cards(self):
        self.minus(636, 210)
        self.minus(740, 210)
        texts = [OCRText("我的军队", .99, (210, 60, 310, 90)),
                 OCRText("已保存的配置", .99, (570, 60, 710, 90)),
                 OCRText("325/335", .99, (580, 150, 650, 180)),
                 OCRText("x10", .99, (562, 198, 592, 219)),
                 OCRText("x1", .99, (666, 198, 690, 219))]
        editor = self.recognize(texts, {("troop", 558): "electro_dragon",
                                        ("troop", 662): "dragon_rider"})
        self.assertEqual(editor["surface"], "current_picker")
        self.assertEqual(editor["editing_kind"], "troop")
        self.assertTrue(editor["complete_kinds"]["troop"])
        self.assertEqual([(card["unit_id"], card["count"]) for card in editor["cards"]],
                         [("electro_dragon", 10), ("dragon_rider", 1)])
        self.assertEqual([control["action"] for control in editor["controls"]],
                         ["close_picker", "decrement", "decrement"])

    def test_missing_identity_blocks_complete_count_and_decrement(self):
        self.minus(636, 210)
        editor = self.recognize([
            OCRText("我的军队", .99, (210, 60, 310, 90)),
            OCRText("已保存的配置", .99, (570, 60, 710, 90)),
            OCRText("335/335", .99, (580, 150, 650, 180)),
            OCRText("x10", .99, (562, 198, 592, 219))])
        self.assertFalse(editor["complete_kinds"]["troop"])
        self.assertEqual([control["action"] for control in editor["controls"]],
                         ["close_picker"])

    def test_shifted_spell_picker_uses_separate_page_anchors(self):
        self.minus(636, 303)
        self.minus(740, 303)
        texts = [OCRText("4/4", .99, (50, 66, 95, 90)),
                 OCRText("强化军队", .99, (890, 45, 960, 75)),
                 OCRText("强化英雄", .99, (1080, 45, 1160, 75)),
                 OCRText("335/335", .99, (580, 66, 660, 94)),
                 OCRText("11/11", .99, (580, 245, 635, 275)),
                 OCRText("x6", .99, (562, 290, 590, 311)),
                 OCRText("x5", .99, (666, 290, 690, 311))]
        editor = self.recognize(texts, {("spell", 558): "lightning_spell",
                                        ("spell", 662): "totem_spell"})
        self.assertEqual((editor["surface"], editor["editing_kind"]),
                         ("current_picker", "spell"))
        self.assertEqual(editor["capacities"]["spell"], {"used": 11, "total": 11})
        self.assertTrue(editor["complete_kinds"]["spell"])
        self.assertEqual([control["action"] for control in editor["controls"]],
                         ["close_picker", "decrement", "decrement"])
        self.assertEqual(editor["controls"][0]["point"], [1050, 240])

    def test_current_gray_lightning_info_requires_its_own_portrait_icon_and_border(self):
        root = Path(__file__).resolve().parent / "fixtures"
        provider = RapidOCRProvider()
        for name in ("spell_picker_pages", "spell_picker_before_info"):
            source = root / f"lightning_info_{name}_18_647_2.png"
            texts = provider.recognize(source)
            editor = recognize_army_editor(source, provider, texts,
                                           client_version="18.647.2")
            info = [item for item in editor["controls"] if item["action"] == "open_info"]
            self.assertEqual((editor["surface"], editor["editing_kind"]),
                             ("current_picker", "spell"))
            by_id = {item["unit_id"]: item for item in info}
            self.assertEqual(set(by_id), {"lightning_spell", "totem_spell"})
            self.assertEqual(by_id["lightning_spell"]["point"], [138, 450])
            self.assertEqual(by_id["totem_spell"]["point"], [659, 587])
            self.assertTrue(all(item["identity_verified"] for item in info))
            self.assertTrue(all(item["confidence"] >= .94 for item in info))
            self.assertFalse(any(item["action"] in {"upgrade", "buy"}
                                 for item in editor["controls"]))
            wrong = recognize_army_editor(source, provider, texts,
                                          client_version="18.647.3")
            self.assertFalse(any(item["action"] == "open_info"
                                 for item in wrong["controls"]))

            image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            for label, roi in (("portrait", (40, 454, 80, 494)),
                               ("blue_info", (123, 436, 151, 464)),
                               ("border", (33, 459, 40, 525))):
                with self.subTest(frame=name, missing=label):
                    altered = image.copy()
                    l, t, r, b = roi
                    altered[t:b, l:r] = 0
                    path = Path(self.directory.name) / f"{name}-{label}.png"
                    cv2.imencode(".png", altered)[1].tofile(path)
                    observed = recognize_army_editor(path, provider, texts,
                                                     client_version="18.647.2")
                    self.assertFalse(any(item["action"] == "open_info"
                                         and item["unit_id"] == "lightning_spell"
                                         for item in observed["controls"]))

    def test_real_current_totem_gray_info_is_read_only_and_version_bound(self):
        source = Path(__file__).parent / "fixtures/totem_spell_picker_gray_18_647_2.png"
        provider = RapidOCRProvider()
        texts = provider.recognize(source)
        editor = recognize_army_editor(source, provider, texts, client_version="18.647.2")
        self.assertEqual((editor["surface"], editor["editing_kind"]),
                         ("current_picker", "spell"))
        self.assertEqual(editor["capacities"]["spell"], {"used": 11, "total": 11})
        self.assertEqual([(card["unit_id"], card["count"]) for card in editor["cards"]],
                         [("lightning_spell", 6), ("totem_spell", 5)])
        controls = [item for item in editor["controls"]
                    if item.get("unit_id") == "totem_spell"]
        self.assertEqual([(item["action"], item["enabled"]) for item in controls],
                         [("decrement", True), ("increment", False), ("open_info", True)])
        info = controls[-1]
        self.assertEqual(info["point"], [659, 587])
        self.assertTrue(info["identity_verified"])
        self.assertGreaterEqual(info["confidence"], .94)
        self.assertEqual(info["evidence"]["identity"]["template_client_version"],
                         "18.647.2")
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        identity = recognize_card_identity(image, (554, 568, 678, 692), "spell",
                                           surface="current_picker", client_version="18.647.2")
        self.assertEqual(identity["unit_id"], "totem_spell")
        self.assertIsNone(recognize_card_identity(
            image, (424, 568, 548, 692), "spell", surface="current_picker",
            client_version="18.647.2")["unit_id"])
        self.assertIsNone(recognize_card_identity(
            image, (554, 568, 678, 692), "spell", surface="current_picker",
            client_version="18.647.3")["unit_id"])
        wrong = recognize_army_editor(source, provider, texts, client_version="18.647.3")
        self.assertFalse(any(item["action"] == "open_info" for item in wrong["controls"]))

    def test_current_totem_info_rejects_missing_portrait_icon_border_and_capacity(self):
        source = Path(__file__).parent / "fixtures/totem_spell_picker_gray_18_647_2.png"
        provider = RapidOCRProvider()
        texts = provider.recognize(source)
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        for label, roi in (("wrong_card", (570, 591, 610, 631)),
                           ("missing_blue_i", (645, 571, 673, 599)),
                           ("missing_top_edge", (586, 569, 637, 575)),
                           ("missing_left_edge", (556, 596, 560, 662)),
                           ("missing_right_edge", (674, 602, 678, 662))):
            with self.subTest(case=label):
                altered = image.copy()
                left, top, right, bottom = roi
                altered[top:bottom, left:right] = 0
                path = Path(self.directory.name) / f"{label}.png"
                cv2.imencode(".png", altered)[1].tofile(path)
                observed = recognize_army_editor(path, provider, texts,
                                                 client_version="18.647.2")
                self.assertFalse(any(item["action"] == "open_info"
                                     and item.get("unit_id") == "totem_spell"
                                     for item in observed["controls"]))
                self.assertFalse(any(item["action"] == "increment"
                                     and item.get("unit_id") == "totem_spell"
                                     and item["enabled"] for item in observed["controls"]))
        colored = image.copy()
        colored[591:631, 570:610] = image[309:349, 667:707]
        colored_path = Path(self.directory.name) / "unsampled-active-card.png"
        cv2.imencode(".png", colored)[1].tofile(colored_path)
        unsampled = recognize_army_editor(colored_path, provider, texts,
                                          client_version="18.647.2")
        self.assertFalse(any(item.get("unit_id") == "totem_spell"
                             and item["action"] in {"increment", "open_info"}
                             for item in unsampled["controls"]))
        partial_texts = [OCRText("10/11", .99, item.bbox)
                         if item.text.strip() == "11/11" else item for item in texts]
        partial = recognize_army_editor(source, provider, partial_texts,
                                        client_version="18.647.2")
        self.assertEqual(partial["capacities"]["spell"], {"used": 10, "total": 11})
        self.assertFalse(any(item["action"] == "open_info"
                             and item.get("unit_id") == "totem_spell"
                             for item in partial["controls"]))

    def test_totem_detail_and_templates_have_exact_source_provenance(self):
        root = Path(__file__).resolve().parents[1]
        metadata = json.loads((root / "tests/fixtures/totem_spell_identity_18_647_2.json")
                              .read_text(encoding="utf-8"))
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        for entry in metadata["templates"] + metadata["fixtures"]:
            with self.subTest(path=entry["path"]):
                self.assertEqual(sha(root / entry["path"]), entry["sha256"])
        detail = root / "tests/fixtures/totem_spell_detail_identity_18_647_2.png"
        words = {item.text for item in RapidOCRProvider().recognize(detail)
                 if item.confidence >= .9}
        self.assertIn("2级图腾法术", words)
        self.assertIn("击晕/瘫痪时间（秒）：", words)
        self.assertIn("所需空间：", words)
        current = template_manifest(client_version="18.647.2")["templates"]
        self.assertFalse(any(item["unit_id"] == "earthquake_spell" for item in current))
        for surface in ("army", "current_picker", "battle"):
            entry, = [item for item in current if item["unit_id"] == "totem_spell"
                      and item["surface"] == surface]
            self.assertEqual(entry["identity_evidence_title"], "2级图腾法术")
            self.assertEqual(entry["identity_evidence_sha256"],
                             metadata["detail_source"]["sha256"])
            self.assertEqual(sha(root / "assets/catalogs" / entry["path"]),
                             entry["template_sha256"])
        # External raw captures are available on the development host; CI only
        # requires distributable fixture/template hashes and detail OCR above.
        for item in (metadata["picker_source"], metadata["detail_source"]):
            source = Path(item["path"])
            if source.is_file():
                self.assertEqual(sha(source), item["sha256"])
        picker_source = Path(metadata["picker_source"]["path"])
        if picker_source.is_file():
            original = cv2.resize(cv2.imdecode(np.fromfile(picker_source, dtype=np.uint8),
                                               cv2.IMREAD_COLOR), (1280, 720),
                                  interpolation=cv2.INTER_AREA)
            picker_template = root / "assets/catalogs/units/totem_spell_picker_capacity_gray_18_647_2.png"
            self.assertTrue(np.array_equal(
                cv2.imdecode(np.fromfile(picker_template, dtype=np.uint8), cv2.IMREAD_COLOR),
                original[591:631, 570:610]))
            masked = original.copy()
            for left, top, right, bottom in ((16, 91, 440, 326),
                                             (445, 101, 1258, 188),
                                             (800, 195, 1260, 320)):
                masked[top:bottom, left:right] = (45, 45, 45)
            self.assertTrue(np.array_equal(
                cv2.imdecode(np.fromfile(root / "tests/fixtures/totem_spell_picker_gray_18_647_2.png",
                                         dtype=np.uint8), cv2.IMREAD_COLOR), masked))
        detail_source = Path(metadata["detail_source"]["path"])
        if detail_source.is_file():
            original = cv2.resize(cv2.imdecode(np.fromfile(detail_source, dtype=np.uint8),
                                               cv2.IMREAD_COLOR), (1280, 720),
                                  interpolation=cv2.INTER_AREA)
            masked = np.full_like(original, 45)
            for left, top, right, bottom in ((480, 25, 770, 85),
                                             (595, 120, 1125, 270),
                                             (135, 115, 205, 195)):
                masked[top:bottom, left:right] = original[top:bottom, left:right]
            self.assertTrue(np.array_equal(
                cv2.imdecode(np.fromfile(detail, dtype=np.uint8), cv2.IMREAD_COLOR), masked))

    def test_live_spell_capacity_local_roi_recovers_info_but_never_overrides_conflict(self):
        source = (Path(__file__).parent / "fixtures" /
                  "lightning_info_spell_picker_failed_capacity_18_647_2.png")
        provider = RapidOCRProvider()
        texts = provider.recognize(source)
        self.assertFalse(any(item.text.strip() == "11/11" and item.confidence >= .88
                             for item in texts))  # Real full OCR saw LL/LL.
        recognizer = ScreenshotRecognizer(provider=provider)
        recognizer.client_version = "18.647.2"
        observed = recognizer.recognize(source)
        editor = observed.observations["army_editor"]
        self.assertEqual((observed.scene, editor["surface"], editor["editing_kind"]),
                         ("training", "current_picker", "spell"))
        self.assertEqual(editor["capacities"]["spell"], {"used": 11, "total": 11})
        self.assertEqual([item["unit_id"] for item in editor["controls"]
                          if item["action"] == "open_info"],
                         ["lightning_spell", "totem_spell"])
        self.assertFalse(any(item["action"] == "increment" and item["enabled"]
                             for item in editor["controls"]))

        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        image[488:548, 1176:1296] = 0
        blank = Path(self.directory.name) / "capacity-blank.png"
        cv2.imencode(".png", image)[1].tofile(blank)
        unreadable = recognizer.recognize(blank).observations["army_editor"]
        self.assertFalse(any(item["action"] == "open_info"
                             for item in unreadable["controls"]))

        read_line = provider.recognize_line
        def wrong_capacity(path, roi):
            if tuple(roi) == (1120, 470, 1360, 550):
                return [OCRText("10/11", .99, tuple(roi))]
            return read_line(path, roi)
        with patch.object(provider, "recognize_line", side_effect=wrong_capacity):
            partial = recognizer.recognize(source).observations["army_editor"]
        self.assertEqual(partial["capacities"]["spell"], {"used": 10, "total": 11})
        self.assertFalse(any(item["action"] == "open_info" for item in partial["controls"]))

        conflict = texts + [OCRText("11/11", .99, (1170, 490, 1250, 540)),
                            OCRText("10/11", .99, (1170, 490, 1250, 540))]
        with patch.object(provider, "recognize", return_value=conflict):
            contradictory = recognizer.recognize(source).observations["army_editor"]
        self.assertIn("picker_spell_capacity_conflict", contradictory["unknowns"])
        self.assertNotIn("spell", contradictory["capacities"])
        self.assertFalse(any(item["action"] == "open_info"
                             for item in contradictory["controls"]))

    def test_conflicting_ocr_counts_cannot_be_overridden_by_x1_sample(self):
        sample = cv2.imread(str(Path(__file__).resolve().parents[1] /
                               "assets/catalogs/current_picker_x1_full_18_600_7.png"),
                            cv2.IMREAD_GRAYSCALE)
        self.image[194:224, 663:708] = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        count, evidence = _picker_count(
            None, self.path,
            [OCRText("x1", .99, (665, 198, 688, 220)),
             OCRText("x2", .99, (665, 198, 688, 220))],
            (662, 194, 760, 292), (1280, 720), self.image,
            "18.600.7", "dragon_rider")
        self.assertIsNone(count)
        self.assertEqual(evidence["source"], "full_ocr")

    def test_extra_digit_after_x1_blocks_sample_fallback(self):
        source = Path(__file__).resolve().parents[1] / "tests/fixtures/army_current_new_20260926.png"
        image = cv2.resize(cv2.imread(str(source)), (1280, 720))
        # Use the independently sampled troop-picker header and add a second
        # numeral to its right. The x1 glyph remains, but the whole header
        # must no longer match an exact one-unit quantity.
        picker = Path(__file__).resolve().parents[1] / "assets/catalogs/current_picker_x1_full_18_600_7.png"
        full = cv2.imread(str(picker), cv2.IMREAD_GRAYSCALE)
        image[194:224, 663:708] = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
        image[197:219, 695:706] = image[197:219, 675:686]
        count, evidence = _picker_count(None, self.path, [],
                                         (662, 194, 760, 292), (1280, 720),
                                         image, "18.600.7", "dragon_rider")
        self.assertIsNone(count)
        self.assertGreaterEqual(evidence["x1_sample_score"], .9)
        self.assertLess(evidence["x1_full_score"], .96)

    def test_selected_count_must_balance_observed_capacity(self):
        self.minus(636, 303)
        editor = self.recognize([
            OCRText("4/4", .99, (50, 66, 95, 90)),
            OCRText("强化军队", .99, (890, 45, 960, 75)),
            OCRText("强化英雄", .99, (1080, 45, 1160, 75)),
            OCRText("11/11", .99, (580, 245, 635, 275)),
            OCRText("x6", .99, (562, 290, 590, 311))],
            {("spell", 558): "lightning_spell"})
        self.assertFalse(editor["complete_kinds"]["spell"])
        self.assertIn("picker_capacity_or_housing_mismatch", editor["unknowns"])

    def test_local_contrast_agreement_reads_quantity_after_full_ocr_misses(self):
        class Provider:
            def __init__(self, responses):
                self.responses = iter(responses)

            def recognize_line(self, _path, _roi):
                return []

            def recognize_line_image(self, _image):
                return [next(self.responses)]

        provider = Provider([OCRText("x7", .803), OCRText("x7", .787),
                             OCRText("x7", .813)])
        count, evidence = _picker_count(provider, self.path, [],
                                         (558, 287, 656, 385), (1280, 720),
                                         self.image, "18.600.7", "lightning_spell")
        self.assertEqual((count, evidence["source"]), (7, "contrast_consensus"))
        provider = Provider([OCRText("x7", .803), OCRText("x8", .787),
                             OCRText("x7", .813)])
        count, evidence = _picker_count(provider, self.path, [],
                                         (558, 287, 656, 385), (1280, 720),
                                         self.image, "18.600.7", "lightning_spell")
        self.assertIsNone(count)

    def test_conflicting_contrast_votes_block_version_sample_fallback(self):
        class Provider:
            def __init__(self):
                self.reads = iter((OCRText("x7", .803), OCRText("x8", .787),
                                   OCRText("x7", .813)))

            def recognize_line(self, _path, _roi):
                return []

            def recognize_line_image(self, _image):
                return [next(self.reads)]

        sample = cv2.imread(str(Path(__file__).resolve().parents[1] /
                               "assets/catalogs/current_picker_x6_18_600_7.png"),
                            cv2.IMREAD_GRAYSCALE)
        self.image[287:317, 559:604] = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        count, _ = _picker_count(Provider(), self.path, [],
                                 (558, 287, 656, 385), (1280, 720),
                                 self.image, "18.600.7", "lightning_spell")
        self.assertIsNone(count)


if __name__ == "__main__":
    import unittest
    unittest.main()


@skipIf(find_spec("rapidocr") is None, "bundled RapidOCR is unavailable")
class SampledCurrentPickerTests(TestCase):
    def test_real_x7_and_capacity_gray_to_active_samples(self):
        recognizer = ScreenshotRecognizer(provider=RapidOCRProvider())
        recognizer.client_version = "18.600.7"
        root = Path(__file__).resolve().parent / "fixtures"
        cases = (
            ("current_picker_spell_x7_20260926.png", 7, 4, 11, False),
            ("current_picker_spell_x6_x4_20260926.png", 6, 4, 10, True),
        )
        for filename, lightning, totem, used, enabled in cases:
            with self.subTest(frame=filename):
                snapshot = recognizer.recognize(root / filename, battle_details=False)
                editor = snapshot.observations["army_editor"]
                self.assertEqual((snapshot.scene, editor["surface"], editor["editing_kind"]),
                                 ("training", "current_picker", "spell"))
                self.assertTrue(editor["complete_kinds"]["spell"])
                self.assertEqual(editor["capacities"]["spell"],
                                 {"used": used, "total": 11})
                cards = {card["unit_id"]: card for card in editor["cards"]}
                self.assertEqual((cards["lightning_spell"]["count"],
                                  cards["totem_spell"]["count"]),
                                 (lightning, totem))
                increments = {item["unit_id"]: item for item in editor["controls"]
                              if item["action"] == "increment"}
                self.assertEqual(set(increments), {"lightning_spell", "totem_spell"})
                self.assertTrue(all(item["enabled"] is enabled and
                                    item["capacity_blocked"] is not enabled
                                    for item in increments.values()))
                if lightning == 7:
                    self.assertEqual(cards["lightning_spell"]["evidence"]["count"]["source"],
                                     "contrast_consensus")
