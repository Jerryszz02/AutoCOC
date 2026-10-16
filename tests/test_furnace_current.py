"""Current-client Furnace identity and detail controls from sampled pixels."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.army_editor import _bottom_picker_controls, _closed_furnace_picker_frame
from autococ.ocr import OCRText
from autococ.unit_catalog import get_unit, recognize_card_identity, template_manifest
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def _read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _detail_texts() -> list[OCRText]:
    # Cached OCR from the exact detail source named in furnace_identity_18_647_2.json.
    return [OCRText(label, confidence, box) for label, confidence, box in (
        ("4级烈焰熔炉", .99758, (564, 32, 713, 70)),
        ("生命值：", .99961, (682, 126, 758, 152)),
        ("1800", .99998, (680, 146, 760, 178)),
        ("持续时间（秒）：", .99937, (949, 124, 1100, 152)),
        ("60", .99892, (950, 149, 990, 177)),
        ("小火怪数量：", .99969, (683, 204, 794, 228)),
        ("22", .99978, (680, 224, 720, 253)),
        ("小火怪等级：", .99913, (950, 203, 1061, 227)),
        ("4", .99999, (952, 226, 974, 251)),
        ("所需空间：", .99988, (624, 286, 718, 312)),
        ("18", .99996, (622, 313, 652, 338)),
    )]


class CurrentFurnaceTests(TestCase):
    def test_templates_fixtures_and_housing_are_bound_to_recorded_pixels(self):
        metadata = json.loads((FIXTURES / "furnace_identity_18_647_2.json").read_text(encoding="utf-8"))
        self.assertEqual(get_unit("furnace").housing_space, 18)
        manifest = [entry for entry in template_manifest(client_version="18.647.2")["templates"]
                    if entry["unit_id"] == "furnace"]
        self.assertEqual({(entry["surface"], entry["state"]) for entry in manifest},
                         {("army", "active"), ("current_picker", "capacity_gray")})
        self.assertFalse(any(entry["surface"] == "battle" for entry in manifest))
        for entry in metadata["templates"] + metadata["fixtures"]:
            with self.subTest(path=entry["path"]):
                path = ROOT / entry["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
                source = Path(metadata["sources"][entry["source_key"]]["path"])
                if source.is_file():
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                                     metadata["sources"][entry["source_key"]]["sha256"])
                    original = cv2.resize(_read(source), (1280, 720), interpolation=cv2.INTER_AREA)
                    sample = _read(path)
                    if "source_bbox_1280" in entry:
                        left, top, right, bottom = entry["source_bbox_1280"]
                        self.assertTrue(np.array_equal(sample, original[top:bottom, left:right]))
                    else:
                        expected = np.full((720, 1280, 3), 45, dtype=np.uint8)
                        for left, top, right, bottom in entry["retained_baseline_rectangles_xyxy"]:
                            expected[top:bottom, left:right] = original[top:bottom, left:right]
                        self.assertTrue(np.array_equal(sample, expected))
        for entry in manifest:
            matching, = [item for item in metadata["templates"]
                         if item["path"] == "assets/catalogs/" + entry["path"]]
            self.assertEqual(entry["template_sha256"], matching["sha256"])
            self.assertEqual(entry["identity_evidence_title"], "4级烈焰熔炉")
            self.assertEqual(entry["identity_evidence_sha256"], metadata["sources"]["detail"]["sha256"])

    def test_independent_army_and_gray_picker_pixels_match_only_their_surfaces(self):
        for name, box, surface in (
            ("furnace_army_holdout_18_647_2.png", (766, 194, 864, 292), "army"),
            ("furnace_picker_holdout_18_647_2.png", (999, 568, 1123, 692), "current_picker"),
        ):
            with self.subTest(surface=surface):
                image = _read(FIXTURES / name)
                found = recognize_card_identity(image, box, "troop", surface=surface,
                                                client_version="18.647.2")
                self.assertEqual(found["unit_id"], "furnace")
                self.assertGreaterEqual(found["confidence"], .94)
                self.assertIsNone(recognize_card_identity(
                    image, box, "troop", surface="battle", client_version="18.647.2")["unit_id"])
                self.assertIsNone(recognize_card_identity(
                    image, box, "troop", surface=surface, client_version="18.647.3")["unit_id"])

    def test_scrolled_furnace_picker_requires_closed_frame_and_full_capacity(self):
        image = _read(FIXTURES / "furnace_picker_holdout_18_647_2.png")
        def controls(frame, capacity=None, version="18.647.2"):
            return _bottom_picker_controls(frame, "troop", version, (1280, 720),
                                           {"furnace"}, capacity or {"used": 343, "total": 343})

        found, availability = controls(image)
        furnace, = [item for item in found if item["unit_id"] == "furnace"]
        self.assertEqual((furnace["point"], furnace["enabled"], furnace["capacity_blocked"]),
                         ([1060, 625], False, True))
        self.assertEqual(furnace["evidence"]["card_bbox"], [999, 568, 1123, 692])
        self.assertEqual(availability["furnace"]["reason"], "present_in_current_army")
        source = Path(json.loads((FIXTURES / "furnace_identity_18_647_2.json")
                                 .read_text(encoding="utf-8"))["sources"]["picker_template"]["path"])
        if source.is_file():
            raw = cv2.resize(_read(source), (1280, 720), interpolation=cv2.INTER_AREA)
            raw_furnace = [item for item in controls(raw)[0] if item["unit_id"] == "furnace"]
            self.assertEqual([item["evidence"]["card_bbox"] for item in raw_furnace],
                             [[999, 568, 1123, 692]])
        self.assertFalse(any(item["unit_id"] == "furnace" for item in controls(image, {"used": 342, "total": 343})[0]))
        self.assertFalse(any(item["unit_id"] == "furnace" for item in controls(image, version="18.647.3")[0]))
        # A full card can sit exactly in a legacy fixed slot. Its portrait and
        # four edges alone must not bypass the missing information glyph.
        fixed = np.full_like(image, 45)
        fixed[568:692, 944:1068] = image[568:692, 999:1123]
        fixed[575:590, 1040:1050] = 0
        fixed_box = (944, 568, 1068, 692)
        self.assertEqual(recognize_card_identity(
            fixed, fixed_box, "troop", surface="current_picker",
            client_version="18.647.2")["unit_id"], "furnace")
        self.assertTrue(_closed_furnace_picker_frame(fixed, fixed_box))
        self.assertFalse(any(item["unit_id"] == "furnace" for item in controls(fixed)[0]))
        for name, rectangle in (
            ("portrait", (1015, 591, 1055, 631)),
            ("information_icon", (1089, 571, 1117, 599)),
            ("top_edge", (1009, 570, 1113, 575)),
            ("left_edge", (1001, 578, 1006, 682)),
            ("right_edge", (1116, 578, 1121, 682)),
            ("bottom_edge", (1009, 685, 1113, 690)),
        ):
            with self.subTest(missing=name):
                altered = image.copy()
                left, top, right, bottom = rectangle
                altered[top:bottom, left:right] = 0
                self.assertFalse(any(item["unit_id"] == "furnace" for item in controls(altered)[0]))

    def test_detail_exposes_only_close_after_exact_same_frame_evidence(self):
        path = FIXTURES / "furnace_detail_close_18_647_2.png"
        texts = _detail_texts()
        recognizer = ScreenshotRecognizer(provider=None)
        recognizer.client_version = "18.647.2"
        detail = recognizer._unit_detail_close(path, texts)
        self.assertEqual(detail["title"]["text"], "4级烈焰熔炉")
        self.assertEqual((detail["button"]["name"], detail["button"]["point"]),
                         ("close_unit_details", [1130, 53]))

        mocked = Mock()
        mocked.recognize.return_value = texts + [OCRText("升级", .999, (674, 590, 712, 615))]
        observed = ScreenshotRecognizer(provider=mocked)
        observed.client_version = "18.647.2"
        snapshot = observed.recognize(path)
        self.assertEqual(snapshot.scene, "popup")
        self.assertEqual([button["name"] for button in snapshot.observations["buttons"]],
                         ["close_unit_details"])

        wrong_version = ScreenshotRecognizer(provider=None)
        wrong_version.client_version = "18.647.3"
        self.assertIsNone(wrong_version._unit_detail_close(path, texts))
        for item in texts:
            with self.subTest(missing=item.text):
                self.assertIsNone(recognizer._unit_detail_close(path, [other for other in texts if other is not item]))
        for old, new in (("4级烈焰熔炉", "5级烈焰熔炉"),
                         ("4级烈焰熔炉", "4.级烈焰熔炉"),
                         ("1800", "1801"), ("1800", "180.0"),
                         ("1800", "1,800"), ("60", "61"),
                         ("60", "60.0"), ("22", "23"),
                         ("22", "22.0"), ("4", "5"),
                         ("18", "19"), ("18", "1.8")):
            with self.subTest(replaced=old):
                altered = [OCRText(new, item.confidence, item.bbox) if item.text == old else item
                           for item in texts]
                self.assertIsNone(recognizer._unit_detail_close(path, altered))
        for numeric in ("1800", "60", "22", "4", "18"):
            original = next(item for item in texts if item.text == numeric)
            with self.subTest(conflict=numeric):
                conflicting = texts + [OCRText(str(int(numeric) + 1), .99, original.bbox)]
                self.assertIsNone(recognizer._unit_detail_close(path, conflicting))
                duplicated = texts + [OCRText(numeric, .99, original.bbox)]
                self.assertIsNone(recognizer._unit_detail_close(path, duplicated))
        moved = [OCRText(item.text, item.confidence, (30, 30, 190, 75))
                 if item.text == "4级烈焰熔炉" else item for item in texts]
        self.assertIsNone(recognizer._unit_detail_close(path, moved))
        self.assertIsNone(recognizer._unit_detail_close(
            FIXTURES / "furnace_picker_holdout_18_647_2.png", texts))
        with TemporaryDirectory() as directory:
            altered = _read(path)
            altered[24:82, 1104:1156] = 0
            hidden = Path(directory) / "furnace-no-close.png"
            cv2.imencode(".png", altered)[1].tofile(hidden)
            self.assertIsNone(recognizer._unit_detail_close(hidden, texts))
