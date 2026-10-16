from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.ocr import OCRText
from autococ.combat import _settlement
from autococ.scene import SceneSnapshot
from autococ.settlement import BONUS_TEMPLATE_CROPS, locate_number_line, recognize_earned_stars
from autococ.vision import ScreenshotRecognizer


class EarnedStarShapeTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def star_region(self, name: str) -> np.ndarray:
        crop = cv2.imdecode(np.fromfile(self.root / "tests/fixtures" / name, dtype=np.uint8),
                            cv2.IMREAD_COLOR)
        image = np.zeros((720, 1280, 3), np.uint8)
        image[65:220, 430:845] = crop
        return image

    def current_two_star_region(self) -> np.ndarray:
        crop = cv2.imdecode(np.fromfile(
            self.root / "tests/fixtures/settlement-two-stars-current-20261015.png",
            dtype=np.uint8), cv2.IMREAD_COLOR)
        image = np.zeros((720, 1280, 3), np.uint8)
        image[20:220, 430:845] = crop
        return image

    def current_three_star_region(self, number: int) -> np.ndarray:
        crop = cv2.imdecode(np.fromfile(
            self.root / f"tests/fixtures/settlement-three-stars-current-054932-{number:05d}.png",
            dtype=np.uint8), cv2.IMREAD_COLOR)
        image = np.zeros((720, 1280, 3), np.uint8)
        image[20:220, 430:845] = crop
        return image

    def split_concavity_two_star_region(self, number: int) -> np.ndarray:
        crop = cv2.imdecode(np.fromfile(
            self.root / f"tests/fixtures/settlement-two-stars-233306-{number:05d}.png",
            dtype=np.uint8), cv2.IMREAD_COLOR)
        image = np.zeros((720, 1280, 3), np.uint8)
        image[20:220, 430:845] = crop
        return image

    def test_current_two_star_split_concavity_is_one_complete_notch(self):
        metadata = json.loads((self.root / "tests/fixtures/settlement-two-stars-233306.provenance.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(metadata["client_version"], "18.647.2")
        self.assertFalse(metadata["human_ground_truth"])
        self.assertFalse(metadata["formal_success"])
        self.assertEqual(len(metadata["fixtures"]), 2)
        for number, sample in zip((60, 61), metadata["fixtures"]):
            with self.subTest(frame=number):
                fixture = self.root / "tests/fixtures" / sample["fixture"]
                self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                                 sample["fixture_sha256"])
                image = self.split_concavity_two_star_region(number)
                result = recognize_earned_stars(image)
                self.assertEqual(result["count"], 2)
                self.assertTrue(all(shape["verified"] for shape in result["evidence"]["shapes"]))
                left = min(result["evidence"]["shapes"], key=lambda shape: shape["bbox_at_1280x720"][0])
                self.assertEqual(len(left["concavity_depths"]), 6)
                merge = left["merged_concavity_pair"]
                self.assertIsNotNone(merge)
                self.assertEqual(len(merge["raw_valleys_at_1280x720"]), 6)
                self.assertEqual(len(merge["grouped_valleys_at_1280x720"]), 5)
                self.assertEqual(len(merge["merged_valleys_at_1280x720"]), 2)
                self.assertLess(merge["valley_gap_pixels"], 21.5)
                cached = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                          for item in sample["cached_result_ocr"]]
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "masked-stars.png"
                    cv2.imencode(".png", image)[1].tofile(path)
                    observation = ScreenshotRecognizer(provider=Mock())._settlement_observation(
                        path, cached, (1280, 720))
                self.assertEqual(observation["stars"], 2)

    def test_current_two_star_split_source_provenance_when_available(self):
        metadata = json.loads((self.root / "tests/fixtures/settlement-two-stars-233306.provenance.json")
                              .read_text(encoding="utf-8"))
        sources = [(Path(sample["source_frame"]), Path(sample["source_events"]))
                   for sample in metadata["fixtures"]]
        if any(not original.is_file() or not event_path.is_file()
               for original, event_path in sources):
            self.skipTest("Original local live evidence is unavailable")
        for sample, (original, event_path) in zip(metadata["fixtures"], sources):
            with self.subTest(source=sample["source_frame"]):
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                                 sample["source_sha256"])
                self.assertEqual(hashlib.sha256(event_path.read_bytes()).hexdigest(),
                                 sample["source_events_sha256"])
                native = cv2.imdecode(np.fromfile(original, dtype=np.uint8), cv2.IMREAD_COLOR)
                baseline = cv2.resize(native, (1280, 720), interpolation=cv2.INTER_AREA)
                fixture = self.root / "tests/fixtures" / sample["fixture"]
                crop = cv2.imdecode(np.fromfile(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
                expected = np.zeros((200, 415, 3), np.uint8)
                for x0, y0, x1, y1 in metadata["retained_rectangles_at_1280x720"]:
                    expected[y0 - 20:y1 - 20, x0 - 430:x1 - 430] = baseline[y0:y1, x0:x1]
                self.assertTrue(np.array_equal(crop, expected))
                observations = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
                                if line.strip()]
                matched = [row for row in observations if row.get("kind") == "observation"
                           and row.get("frame") == str(original)]
                self.assertEqual(len(matched), 1)
                self.assertEqual([item for item in matched[0]["observations"]["ocr"]
                                  if item.get("text") in {"胜利", "74%"}],
                                 sample["cached_result_ocr"])

    def test_split_concavity_does_not_accept_missing_tip_corner_or_six_point_shape(self):
        source = self.split_concavity_two_star_region(60)
        for name, box in (("missing_top_tip", (492, 78, 574, 140)),
                          ("missing_left_tip", (455, 127, 485, 156)),
                          ("missing_inner_corner", (470, 145, 530, 190)),
                          ("partial_occlusion", (498, 124, 546, 163))):
            with self.subTest(name=name):
                image = source.copy()
                x0, y0, x1, y1 = box
                image[y0:y1, x0:x1] = 0
                self.assertIsNone(recognize_earned_stars(image)["count"])
        image = np.zeros_like(source)
        vertices = []
        for index in range(12):
            angle = -math.pi / 2 + index * math.pi / 6
            radius = 65 if index % 2 == 0 else 27
            vertices.append([round(640 + radius * math.cos(angle)),
                             round(125 + radius * math.sin(angle))])
        cv2.fillPoly(image, [np.array(vertices, np.int32)], (240, 240, 240))
        mask = cv2.inRange(cv2.cvtColor(image[20:220, 430:845], cv2.COLOR_BGR2HSV),
                           np.array([0, 0, 140]), np.array([180, 75, 255]))
        contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                      key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        defects = cv2.convexityDefects(polygon, cv2.convexHull(polygon, returnPoints=False))
        self.assertEqual(len(defects), 6)
        self.assertTrue(all(float(row[3]) / 256 >= 10 for row in defects[:, 0]))
        self.assertIsNone(recognize_earned_stars(image)["count"])

        # Two separately forked tips create seven deep notches. They cannot be
        # explained by the one-short-tip animation seen in the source frames.
        vertices = []
        for index in range(5):
            angle = -math.pi / 2 + index * 2 * math.pi / 5
            tips = ((angle - math.radians(12), 60), (angle, 27),
                    (angle + math.radians(12), 60)) if index in (0, 2) else ((angle, 60),)
            for tip_angle, radius in (*tips, (angle + math.pi / 5, 27)):
                vertices.append([round(640 + radius * math.cos(tip_angle)),
                                 round(125 + radius * math.sin(tip_angle))])
        image = np.zeros_like(source)
        cv2.fillPoly(image, [np.array(vertices, np.int32)], (240, 240, 240))
        mask = cv2.inRange(cv2.cvtColor(image[20:220, 430:845], cv2.COLOR_BGR2HSV),
                           np.array([0, 0, 140]), np.array([180, 75, 255]))
        contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                      key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        defects = cv2.convexityDefects(polygon, cv2.convexHull(polygon, returnPoints=False))
        self.assertEqual(len(defects), 7)
        self.assertTrue(all(float(row[3]) / 256 >= 10 for row in defects[:, 0]))
        self.assertIsNone(recognize_earned_stars(image)["count"])

    def test_current_real_three_star_animation_keeps_all_complete_shapes(self):
        for number in range(43, 49):
            with self.subTest(frame=number):
                result = recognize_earned_stars(self.current_three_star_region(number))
                self.assertEqual(result["count"], 3)
                self.assertEqual(len(result["evidence"]["shapes"]), 3)
                self.assertTrue(all(shape["verified"] for shape in result["evidence"]["shapes"]))
                self.assertTrue(all(len(shape["concavity_depths"]) == 5
                                    for shape in result["evidence"]["shapes"]))
                center = next(shape for shape in result["evidence"]["shapes"]
                              if shape["bbox_at_1280x720"][1] < 50)
                self.assertEqual(center["polygon_approximation"],
                                 .025 if number in (43, 45, 47, 48) else .02)

    def test_current_three_star_partial_center_or_extra_shape_stays_unknown(self):
        source = self.current_three_star_region(48)
        for mode in ("missing_top_tip", "missing_lower_tip", "missing_inner_corner",
                     "partial_occlusion", "roi_edge_fragment", "extra_bright_shape"):
            with self.subTest(mode=mode):
                image = source.copy()
                if mode == "missing_top_tip":
                    image[20:62, 600:690] = 0
                elif mode == "missing_lower_tip":
                    image[160:196, 570:607] = 0
                elif mode == "missing_inner_corner":
                    image[90:145, 580:630] = 0
                elif mode == "partial_occlusion":
                    image[92:137, 550:635] = 0
                elif mode == "roi_edge_fragment":
                    image[95:150, 430:482] = 240
                else:
                    cv2.circle(image, (837, 110), 17, (240, 240, 240), -1)
                self.assertIsNone(recognize_earned_stars(image)["count"])

    def test_current_three_star_fixture_provenance(self):
        metadata = json.loads((self.root / "tests/fixtures/settlement-three-stars-current-054932.provenance.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(metadata["client_version"], "18.647.2")
        self.assertFalse(metadata["human_ground_truth"])
        self.assertFalse(metadata["formal_success"])
        for name, record in metadata["fixtures"].items():
            with self.subTest(fixture=name):
                fixture = self.root / "tests/fixtures" / name
                self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                                 record["fixture_sha256"])
                original = Path("E:/projects/AutoCOC") / record["source_frame"]
                if original.is_file():
                    self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                                     record["source_sha256"])
                    raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                    baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
                    crop = cv2.imdecode(np.fromfile(fixture, np.uint8), cv2.IMREAD_COLOR)
                    self.assertTrue(np.array_equal(crop, baseline[20:220, 430:845]))

    def test_current_real_two_star_screen_needs_both_complete_five_concavity_shapes(self):
        image = self.current_two_star_region()
        result = recognize_earned_stars(image)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["evidence"]["roi_at_1280x720"], [430, 20, 845, 220])
        self.assertEqual(len(result["evidence"]["shapes"]), 2)
        for shape in result["evidence"]["shapes"]:
            self.assertTrue(shape["verified"])
            self.assertEqual(len(shape["concavity_depths"]), 5)

        texts = [OCRText("胜利", .99, (608, 196, 671, 236)),
                 OCRText("74%", .99, (610, 102, 672, 132))]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "two-stars.png"
            cv2.imencode(".png", image)[1].tofile(path)
            vision = ScreenshotRecognizer(provider=Mock())
            self.assertEqual(vision._settlement_observation(path, texts, (1280, 720))["stars"], 2)
            # Victory text and percentage must not infer a count if star pixels are absent.
            blank = image.copy()
            blank[20:175, 430:845] = 0
            cv2.imencode(".png", blank)[1].tofile(path)
            self.assertIsNone(vision._settlement_observation(path, texts, (1280, 720))["stars"])

    def test_current_real_two_star_crop_or_flash_remains_unknown(self):
        original = self.current_two_star_region()
        for mode in ("missing_center_tip", "flash_shape", "roi_edge_fragment"):
            with self.subTest(mode=mode):
                image = original.copy()
                if mode == "missing_center_tip":
                    image[20:62, 600:690] = 0
                elif mode == "flash_shape":
                    cv2.circle(image, (790, 75), 24, (240, 240, 240), -1)
                else:
                    cv2.rectangle(image, (430, 95), (482, 154), (240, 240, 240), -1)
                self.assertIsNone(recognize_earned_stars(image)["count"])

    def test_two_real_complete_one_star_outlines(self):
        # Crops from hero-follow-1 frame 00060 and the earlier 20260924
        # victory reread frame 00061; neither needs victory text for counting.
        for name in ("settlement-star-hero-follow-20260926.png",
                     "settlement-star-old-victory-20260924.png"):
            with self.subTest(source=name):
                result = recognize_earned_stars(self.star_region(name))
                self.assertEqual(result["count"], 1)
                self.assertEqual(len(result["evidence"]["shapes"]), 1)
                self.assertEqual(len(result["evidence"]["shapes"][0]["concavity_depths"]), 5)

    def test_complete_two_and_three_star_outlines(self):
        for count in (2, 3):
            with self.subTest(count=count):
                image = np.zeros((720, 1280, 3), np.uint8)
                for index in range(count):
                    points = []
                    for vertex in range(10):
                        angle = -math.pi / 2 + vertex * math.pi / 5 + (index - 1) * .12
                        radius = 60 if vertex % 2 == 0 else 27
                        points.append([round(510 + 135 * index + math.cos(angle) * radius),
                                       round(143 + math.sin(angle) * radius)])
                    cv2.fillPoly(image, [np.array(points, np.int32)], (240, 240, 240))
                self.assertEqual(recognize_earned_stars(image)["count"], count)

    def test_occlusion_and_decorations_do_not_become_earned_stars(self):
        for mode in ("covered_tip", "extra_rectangle", "extra_circle", "blank"):
            with self.subTest(mode=mode):
                image = self.star_region("settlement-star-hero-follow-20260926.png")
                if mode == "covered_tip":
                    image[80:116, 495:546] = 0
                elif mode == "extra_rectangle":
                    image[90:205, 610:705] = 240
                elif mode == "extra_circle":
                    cv2.circle(image, (660, 148), 50, (240, 240, 240), -1)
                else:
                    image[65:220, 430:845] = 0
                self.assertIsNone(recognize_earned_stars(image)["count"])

    def test_old_three_star_screen_with_clipped_middle_star_stays_unknown(self):
        # The 20260926-004627 frame shows three stars, but the middle contour
        # touches this detector's top ROI and overlaps the percent label.
        result = recognize_earned_stars(self.star_region("settlement-star-clipped-three-20260926.png"))
        self.assertEqual(len(result["evidence"]["shapes"]), 3)
        self.assertEqual(sum(shape["verified"] for shape in result["evidence"]["shapes"]), 2)
        self.assertIsNone(result["count"])


class DefeatSettlementLayoutTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_received_short_label_allows_verified_no_bonus_layout(self):
        # Redacted resource-row crop from hero-follow-2 settlement frame 00061.
        crop = cv2.imdecode(np.fromfile(
            self.root / "tests/fixtures/settlement-defeat-rows-hero-follow-2-20260926.png",
            dtype=np.uint8), cv2.IMREAD_COLOR)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[290:457, 340:940] = crop
        readings = [OCRText(text, confidence, bbox) for text, confidence, bbox in (
            ("20%", .99481, (610, 102, 672, 132)),
            ("失败", .9986, (608, 196, 671, 236)),
            ("您得到", .99985, (600, 289, 652, 310)),
            ("159286", .99997, (549, 313, 671, 348)),
            ("221064", .99967, (540, 358, 670, 397)),
            ("1085", .9999, (579, 408, 670, 444)),
            ("损耗的部队", .99984, (592, 462, 678, 485)),
            ("回营", .99636, (623, 608, 660, 631)),
        )]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "defeat.png"
            cv2.imwrite(str(path), image)
            recognizer = ScreenshotRecognizer(provider=Mock())
            for label in ("您得到", "您得到了"):
                with self.subTest(label=label):
                    texts = [replace(item, text=label) if item.text == "您得到" else item for item in readings]
                    result = recognizer._settlement_observation(path, texts, (1280, 720))
                    self.assertEqual(result["loot"], {"gold": 159286, "elixir": 221064, "dark_elixir": 1085})
                    self.assertEqual(result["bonus"], {"gold": 0, "elixir": 0, "dark_elixir": 0})
                    self.assertEqual(result["stars"], 0)
                    self.assertEqual(result["evidence"]["layout"], "regular_defeat_three_rows_v1")
                    self.assertEqual(result["evidence"]["bonus"]["source"], "verified_no_bonus_layout")
                    self.assertEqual(_settlement(SceneSnapshot("settlement", .99, path,
                                                                 {"settlement": result})), result)
            for name, texts in (
                ("missing_received", [item for item in readings if item.text != "您得到"]),
                ("wrong_received_roi", [replace(item, bbox=(300, 289, 352, 310))
                                        if item.text == "您得到" else item for item in readings]),
                ("bonus_marker", readings + [OCRText("奖励", .99, (900, 320, 940, 345))]),
            ):
                with self.subTest(negative=name):
                    result = recognizer._settlement_observation(path, texts, (1280, 720))
                    self.assertIsNone(result["evidence"]["layout"])
                    self.assertTrue(all(value is None for value in result["bonus"].values()))


class SparseSettlementNumberTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    fixture_directory = root / "reports/cleanup-partial-battle-20260924-021726"

    def setUp(self):
        self.path = self.fixture_directory / "frames/00003-natural-settlement.png"
        if not self.path.is_file():
            self.skipTest("Optional isolated-zero settlement fixture is absent")
        events = [json.loads(line) for line in (self.fixture_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observed = next(item for item in events if item["kind"] == "observation" and Path(item["frame"]).name == self.path.name)
        self.texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"])) for item in observed["observations"]["ocr"]]
        self.image = cv2.resize(cv2.imread(str(self.path)), (1280, 720), interpolation=cv2.INTER_AREA)

    def result(self, *, texts=None, image=None, tight_text="0", tight_confidence=.9956, wrong_row=False):
        factor = 2 if image is None else 1
        provider = Mock()
        provider.recognize.return_value = [replace(item, bbox=tuple(v * factor for v in item.bbox))
                                           for item in (self.texts if texts is None else texts)]

        def read_line(path, roi):
            if roi[2] - roi[0] > 40 * factor:
                return [OCRText("O", .60306, roi)]
            box = tuple(v * factor for v in (570, 388, 670, 424)) if wrong_row else roi
            return [OCRText(tight_text, tight_confidence, box)]

        provider.recognize_line.side_effect = read_line
        recognizer = ScreenshotRecognizer(provider=provider)
        if image is None:
            frame = recognizer.recognize(self.path)
        else:
            with TemporaryDirectory() as directory:
                path = Path(directory) / "changed.png"
                cv2.imwrite(str(path), image)
                frame = recognizer.recognize(path)
        return frame.observations["settlement"], provider

    def test_isolated_zero_is_read_from_its_glyphs_not_inferred_from_inventory(self):
        result, provider = self.result()
        self.assertEqual(result["loot"], {"gold": 0, "elixir": 67192, "dark_elixir": 0})
        self.assertEqual(result["bonus"], {"gold": 0, "elixir": 0, "dark_elixir": 0})
        self.assertEqual(result["evidence"]["loot"]["gold"]["source"], "glyph_line_roi")
        self.assertEqual(result["evidence"]["loot"]["gold"]["text"], "0")
        self.assertEqual(provider.recognize_line.call_count, 2)
        wide, tight = [call.args[1] for call in provider.recognize_line.call_args_list]
        self.assertGreater(wide[2] - wide[0], 7 * (tight[2] - tight[0]))
        self.assertEqual(tight, (1284, 682, 1332, 730))

    def test_glyph_localization_alone_never_assigns_zero(self):
        for text, confidence in (("O", .999), ("D", .999), ("0", .89), ("0", float("nan")), ("0", float("inf"))):
            with self.subTest(text=text, confidence=confidence):
                result, _ = self.result(tight_text=text, tight_confidence=confidence)
                self.assertIsNone(result["loot"]["gold"])
                self.assertIsNone(result["evidence"]["layout"])
                self.assertTrue(all(value is None for value in result["bonus"].values()))

    def test_missing_character_does_not_trigger_tight_read_or_zero(self):
        image = self.image.copy()
        image[328:374, 479:672] = 0
        self.assertIsNone(locate_number_line(image, (479, 328, 672, 379)))
        result, provider = self.result(image=image)
        self.assertIsNone(result["loot"]["gold"])
        self.assertEqual(provider.recognize_line.call_count, 1)

    def test_conflicting_loot_readings_remain_unknown_without_preferred_reread(self):
        values = [OCRText("0", .999, (640, 339, 668, 367)), OCRText("10", .99, (640, 339, 668, 367))]
        result, provider = self.result(texts=self.texts + values)
        self.assertIsNone(result["loot"]["gold"])
        provider.recognize_line.assert_not_called()

    def test_tight_ocr_result_from_another_resource_row_is_ignored(self):
        result, _ = self.result(wrong_row=True)
        self.assertIsNone(result["loot"]["gold"])
        self.assertEqual(result["loot"]["elixir"], 67192)

    def test_clipped_and_multiline_glyph_regions_are_not_tightened(self):
        for box in ((650, 328, 672, 379), (479, 350, 672, 379)):
            with self.subTest(roi=box):
                self.assertIsNone(locate_number_line(self.image, box))
        changed = self.image.copy()
        cv2.rectangle(changed, (600, 331), (612, 345), (255, 255, 255), -1)
        self.assertIsNone(locate_number_line(changed, (479, 328, 672, 379)))


class CurrentReceivedSettlementTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    provenance = root / "tests/fixtures/settlement-received-ascii-tail-18_647_2.provenance.json"

    def setUp(self):
        self.metadata = json.loads(self.provenance.read_text(encoding="utf-8"))
        self.assertEqual(self.metadata["client_version"], "18.647.2")

    def replay(self, sample, *, texts=None, image=None):
        left, top, right, bottom = self.metadata["crop_xyxy_at_1280x720"]
        fixture = self.provenance.parent / sample["fixture"]
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), sample["fixture_sha256"])
        crop = cv2.imdecode(np.fromfile(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(crop.shape[:2], (bottom - top, right - left))
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            image[top:bottom, left:right] = crop
        readings = [OCRText(row["text"], row["confidence"], tuple(row["bbox"]))
                    for row in sample["ocr"]] if texts is None else texts
        provider = Mock()
        provider.recognize_line = None
        with TemporaryDirectory() as directory:
            path = Path(directory) / "anonymous-settlement.png"
            cv2.imencode(".png", image)[1].tofile(path)
            return ScreenshotRecognizer(provider=provider)._settlement_observation(
                path, readings, (1280, 720))

    def test_two_real_frames_replay_original_ocr_and_verified_bonus(self):
        for sample, received in zip(self.metadata["samples"], ("您得到了d:", "您得到了Nd:")):
            with self.subTest(frame=sample["source_frame"]):
                self.assertIn(received, [row["text"] for row in sample["ocr"]])
                result = self.replay(sample)
                self.assertEqual(result["percentage"], 69)
                self.assertEqual(result["stars"], 1)
                self.assertEqual(result["loot"], {"gold": 733740, "elixir": 743908, "dark_elixir": 6243})
                self.assertEqual(result["bonus"], {"gold": 371250, "elixir": 371250, "dark_elixir": 2747})
                self.assertEqual(result["evidence"]["layout"], "victory_with_bonus_three_rows_v1")
                self.assertEqual(result["evidence"]["bonus"]["marker"]["text"], "99% 奖励：")
                self.assertEqual(result["evidence"]["bonus"]["source"], "explicit_bonus_panel")

    def test_wrong_received_roi_or_title_cannot_unlock_bonus(self):
        sample = self.metadata["samples"][1]
        readings = [OCRText(row["text"], row["confidence"], tuple(row["bbox"]))
                    for row in sample["ocr"]]
        received = next(item for item in readings if "得到了" in item.text)
        variants = [replace(received, bbox=(300, 287, 380, 312))]
        variants += [replace(received, text=text) for text in
                     ("您没有得到Nd:", "您得到了额外Nd:", "您得到奖励Nd:",
                      "您得到了x:", "您得到了nothing:")]
        for variant in variants:
            with self.subTest(received=variant):
                result = self.replay(sample, texts=[variant if item is received else item for item in readings])
                self.assertIsNone(result["evidence"]["layout"])
                self.assertTrue(all(value is None for value in result["bonus"].values()))
                self.assertEqual(result["loot"]["gold"], 733740)


    def test_marker_signed_amount_conflict_and_other_anchors_still_gate_bonus(self):
        sample = self.metadata["samples"][1]
        readings = [OCRText(row["text"], row["confidence"], tuple(row["bbox"]))
                    for row in sample["ocr"]]
        gold = next(item for item in readings if item.text == "+371250" and item.bbox[1] == 352)
        variants = {
            "missing_marker": [item for item in readings if "奖励" not in item.text],
            "unsigned_gold": [replace(item, text="371250") if item is gold else item for item in readings],
            "weak_gold": [replace(item, confidence=.89) if item is gold else item for item in readings],
            "conflicting_gold": readings + [replace(gold, text="+371251")],
            "title_alone": [item for item in readings if "奖励" not in item.text and not item.text.startswith("+")],
            "missing_return": [item for item in readings if item.text != "回营"],
            "missing_losses": [item for item in readings if item.text != "损耗的部队"],
        }
        for name, texts in variants.items():
            with self.subTest(negative=name):
                result = self.replay(sample, texts=texts)
                self.assertIsNone(result["evidence"]["layout"])
                if name in {"unsigned_gold", "weak_gold", "conflicting_gold"}:
                    self.assertIsNone(result["bonus"]["gold"])
                else:
                    self.assertTrue(all(value is None for value in result["bonus"].values()))
        result = self.replay(sample, texts=variants["conflicting_gold"])
        self.assertEqual(result["bonus"]["elixir"], 371250)
        self.assertEqual(result["bonus"]["dark_elixir"], 2747)
        left, top, right, bottom = self.metadata["crop_xyxy_at_1280x720"]
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[top:bottom, left:right] = cv2.imdecode(
            np.fromfile(self.provenance.parent / sample["fixture"], dtype=np.uint8), cv2.IMREAD_COLOR)
        image[20:175, 430:845] = 0
        without_star = self.replay(sample, image=image)
        self.assertIsNone(without_star["stars"])
        self.assertIsNone(without_star["evidence"]["layout"])


class VictorySettlementTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    fixture_directory = root / "reports/20260924-004347-675110-2e2b80bd"

    def setUp(self):
        self.path = self.fixture_directory / "frames/00061-battle-settlement-reread.png"
        if not self.path.is_file():
            self.skipTest("Optional real victory fixture is absent")
        events = [json.loads(line) for line in (self.fixture_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observed = next(item for item in events if item["kind"] == "observation" and Path(item["frame"]).name == self.path.name)
        self.texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"])) for item in observed["observations"]["ocr"]]
        self.image = cv2.resize(cv2.imread(str(self.path)), (1280, 720), interpolation=cv2.INTER_AREA)

    def result(self, *, texts=None, image=None, line=()):
        texts = self.texts if texts is None else texts
        provider = Mock()
        factor = 2 if image is None else 1
        provider.recognize.return_value = [replace(item, bbox=tuple(v * factor for v in item.bbox)) for item in texts]
        provider.recognize_line.return_value = [replace(item, bbox=tuple(v * factor for v in item.bbox)) for item in line]
        recognizer = ScreenshotRecognizer(provider=provider)
        if image is None:
            snapshot = recognizer.recognize(self.path)
        else:
            with TemporaryDirectory() as directory:
                path = Path(directory) / "changed.png"
                cv2.imwrite(str(path), image)
                snapshot = recognizer.recognize(path)
        return snapshot.observations["settlement"], provider

    def test_real_victory_keeps_loot_bonus_and_earned_star_separate_and_reconciles(self):
        result, _ = self.result()
        self.assertEqual(result["percentage"], 42)
        self.assertEqual(result["stars"], 1)
        self.assertEqual(result["loot"], {"gold": 90196, "elixir": 226443, "dark_elixir": 495})
        self.assertEqual(result["bonus"], {"gold": 237850, "elixir": 237850, "dark_elixir": 1758})
        self.assertEqual(result["evidence"]["layout"], "victory_with_bonus_three_rows_v1")
        self.assertEqual(result["evidence"]["bonus"]["source"], "explicit_bonus_panel")
        report = json.loads((self.root / "reports/run-20260924-004347-675110-2e2b80bd.json").read_text(encoding="utf-8"))
        metrics = report["task_results"][-1]["metrics"]
        for resource in result["loot"]:
            cost = metrics["search_cost_gold"] if resource == "gold" else 0
            self.assertEqual(metrics["resources_before"][resource] + result["loot"][resource] + result["bonus"][resource] - cost,
                             metrics["resources_after"][resource])
        self.assertEqual(metrics["resources_before"]["gems"], metrics["resources_after"]["gems"])
        self.assertEqual(report["task_results"][-1]["status"], "failed", "An offline reread never rewrites the old failure")

    def test_missing_conflicting_unsigned_or_unreliable_bonus_stays_unknown(self):
        gold = next(item for item in self.texts if item.text == "+237850" and item.bbox[1] == 351)
        without = [item for item in self.texts if item is not gold]
        variants = [[], [replace(gold, text="237850")], [replace(gold, text="-237850")],
                    [gold, replace(gold, text="+238850")]]
        variants += [[replace(gold, confidence=value)] for value in (.89, float("nan"), float("inf"))]
        for variant in variants:
            with self.subTest(variant=variant):
                result, _ = self.result(texts=without + variant)
                self.assertIsNone(result["bonus"]["gold"])
                self.assertEqual(result["bonus"]["elixir"], 237850)
                self.assertEqual(result["loot"]["gold"], 90196)
                self.assertIsNone(result["evidence"]["layout"])

    def test_bonus_line_fallback_remains_bound_to_the_matched_icon(self):
        gold = next(item for item in self.texts if item.text == "+237850" and item.bbox[1] == 351)
        result, provider = self.result(texts=[item for item in self.texts if item is not gold], line=[gold])
        self.assertEqual(result["bonus"]["gold"], 237850)
        self.assertEqual(result["evidence"]["bonus"]["resources"]["gold"]["source"], "line_roi")
        provider.recognize_line.assert_called_once()
        wrong_row = replace(gold, bbox=(915, 387, 1004, 413))
        result, _ = self.result(texts=[item for item in self.texts if item is not gold], line=[wrong_row])
        self.assertIsNone(result["bonus"]["gold"])

    def test_missing_bonus_marker_does_not_assign_zero_or_parse_unlabelled_panel(self):
        result, _ = self.result(texts=[item for item in self.texts if "奖励" not in item.text])
        self.assertTrue(all(value is None for value in result["bonus"].values()))
        self.assertEqual(result["stars"], 1)

    def test_missing_bonus_icon_does_not_relabel_another_resource(self):
        for name, (left, top, right, bottom) in BONUS_TEMPLATE_CROPS.items():
            with self.subTest(resource=name):
                changed = self.image.copy()
                changed[top:bottom, left:right] = 0
                result, _ = self.result(image=changed)
                self.assertIsNone(result["bonus"][name])
                self.assertTrue(all(value is not None for key, value in result["bonus"].items() if key != name))

    def test_two_separated_copies_of_a_bonus_icon_are_ambiguous(self):
        image = self.image.copy()
        image[421:453, 1007:1036] = image[349:381, 1007:1036].copy()
        result, _ = self.result(image=image)
        self.assertIsNone(result["bonus"]["gold"])
        self.assertEqual(result["bonus"]["elixir"], 237850)

    def test_star_shape_counter_accepts_rotated_complete_stars(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                image = np.zeros((720, 1280, 3), np.uint8)
                for index in range(count):
                    points = []
                    for vertex in range(10):
                        angle = -math.pi / 2 + vertex * math.pi / 5 + (index - 1) * .12
                        radius = 60 if vertex % 2 == 0 else 27
                        points.append([round(510 + 135 * index + math.cos(angle) * radius), round(143 + math.sin(angle) * radius)])
                    cv2.fillPoly(image, [np.array(points, np.int32)], (240, 240, 240))
                self.assertEqual(recognize_earned_stars(image)["count"], count)

    def test_incomplete_or_unexplained_star_region_cannot_be_ignored(self):
        for mode in ("missing_tip", "extra_rectangle", "extra_circle", "blank"):
            with self.subTest(mode=mode):
                image = self.image.copy()
                if mode == "missing_tip":
                    image[80:116, 495:546] = 0
                elif mode == "extra_rectangle":
                    image[90:205, 610:705] = 240
                elif mode == "extra_circle":
                    cv2.circle(image, (660, 148), 50, (240, 240, 240), -1)
                else:
                    image[65:220, 430:845] = 0
                self.assertIsNone(recognize_earned_stars(image)["count"])

    def test_conflicting_outcomes_are_not_zero_stars_or_zero_bonus(self):
        result, _ = self.result(texts=self.texts + [OCRText("失败", .99, (608, 194, 671, 237))])
        self.assertIsNone(result["stars"])
        self.assertTrue(all(value is None for value in result["bonus"].values()))


class BackgroundMergedSettlementTitleTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.path = self.root / "tests/fixtures/settlement-title-background-203115.png"
        self.metadata = json.loads(self.path.with_suffix(".provenance.json").read_text(encoding="utf8"))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.metadata["fixture_sha256"])
        self.assertFalse(self.metadata["formal_training_data"])
        self.texts = [OCRText(row["text"], row["confidence"], tuple(row["bbox"])) for row in self.metadata["ocr"]]
        self.title = [OCRText(row["text"], row["confidence"], tuple(row["bbox"]))
                      for row in self.metadata["title_line_native"]]
        self.title_roi = tuple(v * 2 for v in self.metadata["title_line_roi_at_1280x720"])

    def replay(self, *, texts=None, title=None, path=None):
        gold = self.metadata["original_gold_line"]
        gold_roi = tuple(v * 2 for v in gold["bbox"])
        provider = Mock()

        def read_line(frame, roi):
            self.assertEqual(frame, self.path if path is None else path)
            if roi == self.title_roi:
                return self.title if title is None else title
            if roi == gold_roi:
                return [OCRText(gold["text"], gold["confidence"], gold_roi)]
            return []

        provider.recognize_line.side_effect = read_line
        result = ScreenshotRecognizer(provider=provider)._settlement_observation(
            self.path if path is None else path, self.texts if texts is None else texts, (2560, 1440))
        return result, provider

    def test_current_title_line_recovers_complete_two_row_defeat(self):
        result, provider = self.replay()
        self.assertEqual(result["loot"], {"gold": 0, "elixir": 3470, "dark_elixir": 0})
        self.assertEqual(result["bonus"], {"gold": 0, "elixir": 0, "dark_elixir": 0})
        self.assertEqual(result["evidence"]["layout"], "regular_defeat_gold_elixir_only_v1")
        self.assertEqual(result["stars"], 0)
        self.assertEqual(result["percentage"], 2)
        self.assertTrue(result["evidence"]["loot"]["dark_elixir"]["requires_inventory_reconciliation"])
        anchor = result["evidence"]["anchors"]["received"]
        self.assertEqual(anchor["source"], "current_title_line_roi")
        self.assertEqual(anchor["frame"], str(self.path))
        self.assertEqual(anchor["text"], "您得到了")
        self.assertEqual(sum(call.args[1] == self.title_roi for call in provider.recognize_line.call_args_list), 1)

    def test_merged_prefix_alone_or_weak_conflicting_line_never_supplies_zero(self):
        good = self.title[0]
        variants = [[], [replace(good, text="您得到了KISTAN")], [replace(good, text="您没有得到")],
                    [replace(good, confidence=.89)], [replace(good, confidence=math.nan)],
                    [replace(good, confidence=math.inf)], [good, replace(good, text="奖励")],
                    [replace(good, bbox=(0, 0, 140, 52))]]
        for line in variants:
            with self.subTest(line=line):
                result, _ = self.replay(title=line)
                self.assertIsNone(result["loot"]["dark_elixir"])
                self.assertIsNone(result["evidence"]["layout"])
                self.assertTrue(all(value is None for value in result["bonus"].values()))

    def test_wrong_or_ambiguous_title_geometry_does_not_trigger_recheck(self):
        merged = next(item for item in self.texts if item.text.startswith("您得到"))
        variants = [[replace(item, bbox=(300, 302, 409, 327)) if item is merged else item for item in self.texts],
                    [replace(item, text="您没有得到KISTAN") if item is merged else item for item in self.texts],
                    [replace(item, confidence=.89) if item is merged else item for item in self.texts],
                    self.texts + [replace(merged, text="您得到了OTHER")]]
        for readings in variants:
            with self.subTest(readings=readings):
                result, provider = self.replay(texts=readings)
                self.assertIsNone(result["evidence"]["layout"])
                self.assertNotIn(self.title_roi, [call.args[1] for call in provider.recognize_line.call_args_list])

    def test_other_anchors_bonus_and_complete_rows_are_still_required(self):
        for readings in ([item for item in self.texts if item.text != "回营"],
                         [item for item in self.texts if item.text != "损耗的部队"],
                         self.texts + [OCRText("奖励", .99, (900, 320, 940, 345))]):
            with self.subTest(readings=readings):
                result, _ = self.replay(texts=readings)
                self.assertIsNone(result["evidence"]["layout"])
                self.assertIsNone(result["loot"]["dark_elixir"])
                self.assertTrue(all(value is None for value in result["bonus"].values()))
        image = cv2.imdecode(np.fromfile(self.path, np.uint8), cv2.IMREAD_COLOR)
        image[750:762, 680:1880] = 0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing-gold-divider.png"
            cv2.imencode(".png", image)[1].tofile(path)
            result, _ = self.replay(path=path)
        self.assertIn("received", result["evidence"]["anchors"])
        self.assertIsNone(result["evidence"]["layout"])
