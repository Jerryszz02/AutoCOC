from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import hashlib
import json

import cv2
import numpy as np

from autococ.hero_state import (_current_warden_gray_card_topology, _match,
                                locate_current_deployed_hero_card, recognize_hero_state)


class HeroStateTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    slot = [599, 595, 688, 711]
    yellow_fixture_dir = root / "tests/fixtures/hero_yellow_hp_18_647_2"

    low_health_slot = [499, 588, 594, 713]

    def test_gray_used_card_topology_requires_both_long_continuous_side_edges(self) -> None:
        gray = np.full((720, 1280), 30, dtype=np.uint8)
        gray[625:695, 648:735] = 100
        proof = _current_warden_gray_card_topology(gray, 647, 736)
        self.assertIsNotNone(proof)
        self.assertEqual(proof["paired_rows"], 70)
        self.assertEqual(proof["longest_paired_run"], 70)
        for name, x1, x2 in (("left_gap", 642, 653), ("right_gap", 731, 740)):
            with self.subTest(name=name):
                broken = gray.copy()
                broken[650:664, x1:x2] = 30
                self.assertIsNone(_current_warden_gray_card_topology(broken, 647, 736))
        one_side = gray.copy()
        one_side[625:695, 700:740] = 30
        self.assertIsNone(_current_warden_gray_card_topology(one_side, 647, 736))

    def test_real_animated_gray_warden_card_needs_closed_sides_hp_and_used_equipment(self) -> None:
        fixture = self.root / "tests/fixtures/hero_warden_gray_used_083520_18_647_2.png"
        source = cv2.imdecode(np.fromfile(fixture, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(source)
        locator = locate_current_deployed_hero_card(fixture, "grand_warden",
                                                    client_version="18.647.2")
        self.assertIsNotNone(locator)
        self.assertEqual(locator["bbox"], [647, 595, 736, 711])
        self.assertIsNotNone(locator["gray_used_topology"])
        baseline = recognize_hero_state(fixture, locator["bbox"], unit_id="grand_warden",
                                        client_version="18.647.2")
        self.assertIs(baseline["ability_used"], True)
        self.assertIsNotNone(baseline["evidence"]["health_bar"])
        equipment = baseline["evidence"]["current_equipment_matches"]
        self.assertIsNotNone(equipment["used_book"])
        self.assertIsNotNone(equipment["used_card"])
        self.assertIsNone(equipment["ready_book"])
        self.assertIsNone(equipment["ready_card"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "modified.png"
            for name, box in {"left_border": (646, 625, 651, 695),
                              "right_border": (732, 625, 740, 695),
                              "bottom_border": (655, 709, 728, 714),
                              "hp": (654, 573, 731, 586),
                              "used_book": (650, 530, 690, 567),
                              "used_card": (680, 626, 729, 680)}.items():
                with self.subTest(case=name):
                    image = source.copy()
                    x0, y0, x1, y1 = box
                    if name == "right_border":
                        image[y0:y1, x0:x1] = source[y0:y1, 740:741]
                    else:
                        image[y0:y1, x0:x1] = (30, 30, 30)
                    cv2.imencode(".png", image)[1].tofile(path)
                    broken_locator = locate_current_deployed_hero_card(
                        path, "grand_warden", client_version="18.647.2")
                    if name.endswith("border"):
                        self.assertIsNone(broken_locator)
                    state = recognize_hero_state(path, locator["bbox"], unit_id="grand_warden",
                                                 client_version="18.647.2")
                    self.assertIsNot(state["ability_used"], True)

    def test_real_animated_gray_warden_card_fixture_provenance(self) -> None:
        fixture = self.root / "tests/fixtures/hero_warden_gray_used_083520_18_647_2.png"
        provenance = self.root / "tests/fixtures/hero_warden_gray_used_083520_18_647_2.provenance.json"
        record = json.loads(provenance.read_text(encoding="utf-8"))
        self.assertFalse(record["human_ground_truth"])
        self.assertFalse(record["formal_battle_acceptance"])
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), record["fixture_sha256"])
        original = Path(record["source_frame"])
        if original.is_file():
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), record["source_sha256"])
            raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
            baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
            expected = np.zeros_like(baseline)
            expected[515:720, :] = baseline[515:720, :]
            saved = cv2.imdecode(np.fromfile(fixture, np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(saved, expected))

    def test_narrow_dark_warden_outline_keeps_current_hp_and_used_pair_required(self) -> None:
        fixture = self.root / "tests/fixtures/hero_warden_gray_used_092829_18_647_2.png"
        source = cv2.imdecode(np.fromfile(fixture, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(source)
        locator = locate_current_deployed_hero_card(fixture, "grand_warden",
                                                    client_version="18.647.2")
        self.assertIsNotNone(locator)
        self.assertEqual(locator["bbox"], [647, 595, 736, 711])
        self.assertEqual(locator["gray_used_topology"]["contrast_proof"], "paired_dark_troughs")
        self.assertGreaterEqual(locator["gray_used_topology"]["dark_line"]["paired_dark_trough_rows"], 45)
        state = recognize_hero_state(fixture, locator["bbox"], unit_id="grand_warden",
                                     client_version="18.647.2")
        self.assertIs(state["ability_used"], True)
        self.assertIsNotNone(state["evidence"]["health_bar"])
        equipment = state["evidence"]["current_equipment_matches"]
        self.assertIsNotNone(equipment["used_book"])
        self.assertIsNotNone(equipment["used_card"])
        self.assertIsNone(equipment["ready_book"])
        self.assertIsNone(equipment["ready_card"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "modified.png"
            for name in ("left_erased", "right_erased", "left_gap", "right_gap",
                         "left_shifted", "right_shifted", "neighbor_edge", "bright_neighbors",
                         "missing_hp", "missing_book", "missing_card"):
                with self.subTest(case=name):
                    image = source.copy()
                    if name == "left_erased":
                        image[625:695, 646:651] = (30, 30, 30)
                    elif name == "right_erased":
                        image[625:695, 732:740] = source[625:695, 740:741]
                    elif name == "left_gap":
                        image[650:664, 642:653] = (30, 30, 30)
                    elif name == "right_gap":
                        image[650:664, 731:740] = (30, 30, 30)
                    elif name == "left_shifted":
                        image[625:695, 644:654] = np.roll(source[625:695, 644:654], 3, axis=1)
                    elif name == "right_shifted":
                        image[625:695, 730:740] = np.roll(source[625:695, 730:740], -3, axis=1)
                    elif name == "neighbor_edge":
                        image[625:695, 646:651] = (30, 30, 30)
                        image[625:695, 642:644] = (20, 20, 20)
                    elif name == "bright_neighbors":
                        image[625:695, 638:645] = (190, 190, 190)
                        image[625:695, 739:746] = (190, 190, 190)
                    elif name == "missing_hp":
                        image[573:586, 654:731] = (30, 30, 30)
                    elif name == "missing_book":
                        image[530:567, 650:690] = (30, 30, 30)
                    else:
                        image[626:680, 680:729] = (30, 30, 30)
                    cv2.imencode(".png", image)[1].tofile(path)
                    changed = locate_current_deployed_hero_card(path, "grand_warden",
                                                               client_version="18.647.2")
                    if name == "bright_neighbors":
                        self.assertIsNotNone(changed)
                    elif name not in ("missing_hp", "missing_book", "missing_card"):
                        self.assertIsNone(changed)
                    current = recognize_hero_state(path, locator["bbox"], unit_id="grand_warden",
                                                   client_version="18.647.2")
                    if name == "bright_neighbors":
                        self.assertIs(current["ability_used"], True)
                    if name.startswith("missing_"):
                        self.assertIsNot(current["ability_used"], True)

    def test_narrow_dark_warden_fixture_provenance(self) -> None:
        fixture = self.root / "tests/fixtures/hero_warden_gray_used_092829_18_647_2.png"
        provenance = fixture.with_suffix(".provenance.json")
        record = json.loads(provenance.read_text(encoding="utf-8"))
        self.assertFalse(record["human_ground_truth"])
        self.assertFalse(record["formal_battle_acceptance"])
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), record["fixture_sha256"])
        original = Path(record["source_frame"])
        if original.is_file():
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), record["source_sha256"])
            raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
            baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
            expected = np.zeros_like(baseline)
            expected[515:720, :] = baseline[515:720, :]
            saved = cv2.imdecode(np.fromfile(fixture, np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(saved, expected))

    def test_current_warden_yellow_hp_proves_live_deployment(self) -> None:
        for name, fill in (("green_24", "green"), ("yellow_25", "yellow"),
                           ("yellow_26", "yellow")):
            with self.subTest(frame=name):
                path = self.yellow_fixture_dir / f"{name}.png"
                locator = locate_current_deployed_hero_card(path, "grand_warden",
                                                            client_version="18.647.2")
                self.assertIsNotNone(locator)
                self.assertEqual(locator["bbox"], [647, 595, 736, 711])
                state = recognize_hero_state(path, locator["bbox"], unit_id="grand_warden",
                                             client_version="18.647.2")
                self.assertTrue(state["deployed"])
                self.assertFalse(state["defeated"])
                self.assertEqual(state["evidence"]["portrait_identity"]["unit_id"], "grand_warden")
                if fill == "green":
                    self.assertEqual(state["evidence"]["health_bar"]["method"],
                                     "green_fill_enclosed_by_dark_hp_frame")
                    self.assertIsNone(state["evidence"]["yellow_health_bar"])
                    self.assertTrue(state["ability_ready"])
                else:
                    self.assertIsNone(state["evidence"]["health_bar"])
                    self.assertEqual(state["evidence"]["yellow_health_bar"]["fill_bbox"],
                                     [655, 573, 686, 582])
                    self.assertGreaterEqual(min(state["evidence"]["yellow_health_bar"]["frame_support"]), .85)
                    # This historical HP fixture has no reviewed ready label.
                    # The 113248 TRAIN regression owns yellow-ready evidence.
                    self.assertIsNot(state["ability_used"], True)

    def test_current_warden_yellow_hp_requires_correct_frame_card_and_identity(self) -> None:
        path = self.yellow_fixture_dir / "yellow_25.png"
        source = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        cases = {}
        empty = np.zeros_like(source)
        cases["empty"] = empty
        for name, box in {"missing_fill": (655, 573, 686, 582),
                          "missing_top_frame": (654, 570, 730, 573),
                          "missing_bottom_frame": (654, 582, 730, 585),
                          "missing_card_side": (646, 625, 650, 695)}.items():
            image = source.copy()
            x1, y1, x2, y2 = box
            image[y1:y2, x1:x2] = 100
            cases[name] = image
        clutter = source.copy()
        clutter[573:582, 655:686] = 0
        clutter[573:582, 690:721] = source[573:582, 655:686]
        cases["yellow_clutter_elsewhere"] = clutter
        with TemporaryDirectory() as directory:
            modified = Path(directory) / "modified.png"
            for name, image in cases.items():
                with self.subTest(case=name):
                    cv2.imencode(".png", image)[1].tofile(modified)
                    state = recognize_hero_state(modified, [647, 595, 736, 711],
                                                 unit_id="grand_warden", client_version="18.647.2")
                    self.assertIsNot(state["deployed"], True)
                    self.assertIsNone(state["evidence"].get("yellow_health_bar"))
                    self.assertIsNot(state["ability_ready"], True)
            for unit_id, version, box in (("grand_warden", "18.600.7", [647, 595, 736, 711]),
                                          ("archer_queen", "18.647.2", [647, 595, 736, 711]),
                                          ("grand_warden", "18.647.2", [744, 595, 833, 711])):
                with self.subTest(unit_id=unit_id, version=version, box=box):
                    state = recognize_hero_state(path, box, unit_id=unit_id, client_version=version)
                    self.assertIsNone(state["evidence"].get("yellow_health_bar"))
                    if version == "18.647.2":
                        self.assertIsNot(state["deployed"], True)

    def test_current_warden_yellow_card_fixture_provenance(self) -> None:
        metadata = json.loads((self.yellow_fixture_dir / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["human_ground_truth"])
        self.assertFalse(metadata["dynamic_event_validation"])
        for name, record in metadata["fixtures"].items():
            with self.subTest(frame=name):
                fixture = self.yellow_fixture_dir / f"{name}.png"
                self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), record["fixture_sha256"])
                image = cv2.imdecode(np.fromfile(fixture, np.uint8), cv2.IMREAD_COLOR)
                x1, y1, x2, y2 = record["card_bbox_1280"]
                mask = np.ones(image.shape[:2], dtype=bool)
                mask[y1:y2, x1:x2] = False
                self.assertFalse(np.any(image[mask]))
                source = Path("E:/projects/AutoCOC/reports/completion-20261015") / record["source_relative_to_reports"]
                if source.is_file():
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), record["source_sha256"])
                    raw = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)
                    baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
                    crop = baseline[y1:y2, x1:x2]
                    self.assertEqual(hashlib.sha256(crop.tobytes()).hexdigest(), record["card_pixels_sha256"])
                    self.assertTrue(np.array_equal(image[y1:y2, x1:x2], crop))

    def current_skill_fixture(self, number: int) -> Path:
        return self.root / f"tests/fixtures/hero_skill_035742_{number:05d}_18_647_2.png"

    def current_skill_phase_fixture(self, battle: str, number: int) -> Path:
        return self.root / f"tests/fixtures/hero_skill_phase_{battle}_{number:05d}_18_647_2.png"

    def test_current_warden_ready_card_animation_phase_has_independent_battle_proof(self) -> None:
        for battle, number in (("053200", 25), ("053200", 28),
                               ("035742", 21), ("035742", 22)):
            with self.subTest(battle=battle, frame=number):
                path = self.current_skill_phase_fixture(battle, number)
                located = locate_current_deployed_hero_card(path, "grand_warden",
                                                            client_version="18.647.2")
                self.assertIsNotNone(located)
                state = recognize_hero_state(path, located["bbox"], unit_id="grand_warden",
                                             client_version="18.647.2")
                self.assertEqual(state["state"], "ability_ready")
                self.assertTrue(state["ability_ready"])
                self.assertFalse(state["ability_used"])
                self.assertIsNotNone(state["evidence"]["health_bar"])
                self.assertIsNotNone(state["evidence"]["closed_card"])
                self.assertEqual(state["evidence"]["ability_ready"]["hero"], "grand_warden")
                if (battle, number) in (("053200", 28), ("035742", 21)):
                    proof = state["evidence"]["ability_ready"]
                    self.assertEqual(Path(proof["ready_card"]["template"]).name,
                                     "hero_warden_18_647_2_ready_card_phase.png")
                    self.assertTrue(proof["template_source"].endswith(
                        "053200-457276-365d1d63/frames/00028-post-deployment-skill-before.png"))
        self.assertTrue(recognize_hero_state(self.current_skill_fixture(23),
                                             [647, 595, 736, 711], unit_id="grand_warden",
                                             client_version="18.647.2")["ability_ready"])
        for number in (26, 27):
            with self.subTest(used_or_transition=number):
                state = recognize_hero_state(self.current_skill_fixture(number),
                                             [647, 595, 736, 711], unit_id="grand_warden",
                                             client_version="18.647.2")
                self.assertIsNot(state["ability_ready"], True)
        phase_template = self.root / "assets/templates/hero_warden_18_647_2_ready_card_phase.png"
        for path in (self.current_skill_phase_fixture("035742", 21),
                     self.current_skill_phase_fixture("035742", 22), self.current_skill_fixture(23)):
            with self.subTest(phase_template_ready=path.name):
                image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(_match(image, phase_template, (675, 621, 734, 685)))
        for number in (26, 27):
            with self.subTest(phase_template_used=number):
                image = cv2.imdecode(np.fromfile(self.current_skill_fixture(number), np.uint8),
                                     cv2.IMREAD_COLOR)
                self.assertIsNone(_match(image, phase_template, (675, 621, 734, 685)))

    def test_current_warden_ready_card_phase_requires_book_hp_closed_card_and_identity(self) -> None:
        path = self.current_skill_phase_fixture("053200", 28)
        source = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        with TemporaryDirectory() as directory:
            modified = Path(directory) / "modified.png"
            for name, box in {"book": (650, 530, 690, 567),
                              "card_art": (680, 626, 734, 675),
                              "hp": (654, 573, 731, 586),
                              "bottom_border": (655, 709, 728, 711)}.items():
                with self.subTest(missing=name):
                    frame = source.copy()
                    left, top, right, bottom = box
                    if name == "bottom_border":
                        # The bottom edge is outside the ready-art template.
                        frame[top:bottom, left:right] = source[712:713, left:right]
                    else:
                        frame[top:bottom, left:right] = 0
                    cv2.imencode(".png", frame)[1].tofile(modified)
                    state = recognize_hero_state(modified, [647, 595, 736, 711],
                                                 unit_id="grand_warden", client_version="18.647.2")
                    self.assertIsNot(state["ability_ready"], True)
            for hero, version, box in (("archer_queen", "18.647.2", [647, 595, 736, 711]),
                                       ("grand_warden", "18.600.7", [647, 595, 736, 711]),
                                       ("grand_warden", "18.647.2", [744, 595, 833, 711])):
                with self.subTest(hero=hero, version=version, bbox=box):
                    state = recognize_hero_state(path, box, unit_id=hero, client_version=version)
                    self.assertIsNot(state["ability_ready"], True)

    def test_current_warden_ready_card_phase_source_and_anonymized_pixels_are_bound(self) -> None:
        metadata = json.loads((self.root / "tests/fixtures/hero_skill_ready_card_phase_18_647_2.provenance.json")
                              .read_text(encoding="utf-8"))
        self.assertFalse(metadata["human_ground_truth"])
        self.assertFalse(metadata["dynamic_event_validation"])
        for name, record in metadata["assets"].items():
            target = self.root / "assets/templates" / name
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), record["asset_sha256"])
            original = Path("E:/projects/AutoCOC") / record["source_frame"]
            if original.is_file():
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                                 record["source_sha256"])
                raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
                left, top, right, bottom = record["source_bbox_1280"]
                saved = cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_COLOR)
                self.assertTrue(np.array_equal(saved, baseline[top:bottom, left:right]))
        for name, record in metadata["fixtures"].items():
            target = self.root / "tests/fixtures" / name
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), record["fixture_sha256"])
            original = Path("E:/projects/AutoCOC") / record["source_frame"]
            if original.is_file():
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                                 record["source_sha256"])
                raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
                reconstructed = np.zeros_like(baseline)
                reconstructed[515:720, :] = baseline[515:720, :]
                saved = cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_COLOR)
                self.assertTrue(np.array_equal(saved, reconstructed))

    def test_current_warden_ready_and_used_have_separate_same_frame_proofs(self) -> None:
        for number, phase in ((23, "ability_ready"), (24, "ability_ready"),
                              (25, "ability_ready"), (27, "ability_used"),
                              (28, "ability_used"), (29, "ability_used")):
            with self.subTest(frame=number):
                path = self.current_skill_fixture(number)
                locator = locate_current_deployed_hero_card(path, "grand_warden",
                                                             client_version="18.647.2")
                self.assertIsNotNone(locator)
                state = recognize_hero_state(path, locator["bbox"], unit_id="grand_warden",
                                             client_version="18.647.2")
                self.assertEqual(state["state"], phase)
                self.assertTrue(state["deployed"])
                self.assertIsNotNone(state["evidence"]["health_bar"])
                self.assertEqual(state["evidence"]["portrait_identity"]["unit_id"], "grand_warden")
                self.assertEqual(state["ability_ready"], phase == "ability_ready")
                self.assertEqual(state["ability_used"], phase == "ability_used")
        self.assertIsNone(locate_current_deployed_hero_card(self.current_skill_fixture(26),
                                                            "grand_warden", client_version="18.647.2"))

    def test_current_warden_state_rejects_version_identity_missing_anchors_and_conflict(self) -> None:
        ready = cv2.imdecode(np.fromfile(self.current_skill_fixture(25), np.uint8), cv2.IMREAD_COLOR)
        used = cv2.imdecode(np.fromfile(self.current_skill_fixture(27), np.uint8), cv2.IMREAD_COLOR)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "modified.png"
            cases = {}
            for name, box in {"book": (650, 530, 690, 567), "card": (680, 626, 729, 680),
                              "hp": (654, 573, 731, 586), "left_border": (646, 625, 651, 695),
                              "right_border": (732, 625, 740, 695),
                              "bottom_border": (655, 709, 728, 714)}.items():
                frame = used.copy()
                x0, y0, x1, y1 = box
                frame[y0:y1, x0:x1] = (30, 30, 30)
                if name == "right_border":
                    # Replace the border with its neighbouring background,
                    # rather than drawing a new dark line at the same place.
                    frame[y0:y1, x0:x1] = used[y0:y1, 740:741]
                cases[name] = frame
            mixed = ready.copy()
            mixed[626:680, 680:729] = used[626:680, 680:729]
            cases["mixed_phase"] = mixed
            for name, frame in cases.items():
                with self.subTest(case=name):
                    cv2.imencode(".png", frame)[1].tofile(path)
                    locator = locate_current_deployed_hero_card(path, "grand_warden",
                                                                 client_version="18.647.2")
                    box = locator["bbox"] if locator else [647, 595, 736, 711]
                    state = recognize_hero_state(path, box, unit_id="grand_warden",
                                                 client_version="18.647.2")
                    self.assertIsNot(state["ability_used"], True)
                    self.assertIsNot(state["ability_ready"], True)
            cv2.imencode(".png", ready)[1].tofile(path)
            for unit_id, version, box in (("archer_queen", "18.647.2", [647, 595, 736, 711]),
                                          ("grand_warden", "18.600.7", [647, 595, 736, 711]),
                                          ("grand_warden", "18.647.2", [744, 595, 833, 711])):
                with self.subTest(unit_id=unit_id, version=version, box=box):
                    state = recognize_hero_state(path, box, unit_id=unit_id, client_version=version)
                    self.assertIsNot(state["ability_ready"], True)
                    self.assertIsNot(state["ability_used"], True)

    def test_current_skill_fixture_provenance_hashes(self) -> None:
        metadata = json.loads((self.root / "tests/fixtures/hero_skill_035742_18_647_2.provenance.json")
                              .read_text(encoding="utf-8"))
        self.assertFalse(metadata["human_ground_truth"])
        self.assertFalse(metadata["dynamic_event_validation"])
        for name, record in metadata["assets"].items():
            folder = "assets/catalogs/units" if name.startswith("grand_") else "assets/templates"
            target = self.root / folder / name
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),
                             record["asset_sha256"])
            original = Path("E:/projects/AutoCOC") / record["source_frame"]
            if original.is_file():
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                                 record["source_sha256"])
                raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
                left, top, right, bottom = record["source_bbox_1280"]
                saved = cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_COLOR)
                self.assertTrue(np.array_equal(saved, baseline[top:bottom, left:right]))
        for name, record in metadata["fixtures"].items():
            target = self.root / "tests/fixtures" / name
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),
                             record["fixture_sha256"])
            original = Path("E:/projects/AutoCOC") / record["source_frame"]
            if original.is_file():
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                                 record["source_sha256"])
                raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                baseline = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_AREA)
                reconstructed = np.zeros_like(baseline)
                reconstructed[515:720, :] = baseline[515:720, :]
                saved = cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_COLOR)
                self.assertTrue(np.array_equal(saved, reconstructed))

    def test_current_client_never_uses_legacy_ability_templates(self) -> None:
        bar = cv2.imread(str(self.root / "tests/fixtures/battle_bar_heldout_18_647_2_20261015.png"))
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[590:720, 85:1190] = bar
        with TemporaryDirectory() as directory:
            path = Path(directory) / "current-bar.png"
            cv2.imwrite(str(path), image)
            pet = {"template": str(self.root / "assets/templates/battle_hero_pet_0.png"),
                   "confidence": .99, "bbox": [604, 600, 638, 636]}
            with patch("autococ.hero_state._match", return_value=pet) as match, \
                    patch("autococ.hero_state._health_bar", return_value={"method": "verified_hp"}):
                state = recognize_hero_state(path, [599, 595, 687, 711],
                                             unit_id="grand_warden", client_version="18.647.2")
        self.assertTrue(state["deployed"])
        self.assertIsNone(state["ability_ready"])
        self.assertIsNone(state["ability_used"])
        self.assertEqual(state["evidence"]["ability_phase_reason"],
                         "current_warden_card_not_independently_closed")
        self.assertTrue(all(call.args[1].name.startswith("battle_hero_pet_")
                            for call in match.call_args_list))

    def low_health_fixture(self, phase: str) -> Path:
        return self.root / f"tests/fixtures/hero-warden-low-hp-{phase}-20260926.png"

    def test_current_low_health_warden_requires_three_ready_anchors(self) -> None:
        state = recognize_hero_state(self.low_health_fixture("ready"), self.low_health_slot,
                                     unit_id="grand_warden", client_version="18.600.7")
        self.assertEqual(state["state"], "ability_ready")
        self.assertTrue(state["deployed"])
        self.assertTrue(state["ability_ready"])
        self.assertFalse(state["ability_used"])
        self.assertIsNone(state["defeated"])
        self.assertIsNone(state["evidence"]["health_bar"])
        ready = state["evidence"]["ability_ready"]
        self.assertEqual(ready["phase_portrait"]["phase"], "ready")
        self.assertGreaterEqual(ready["phase_portrait"]["confidence"], .9)
        self.assertGreaterEqual(ready["active_equipment"]["confidence"], .9)
        self.assertGreaterEqual(ready["card_book_glow"]["confidence"], .9)
        self.assertIsNone(ready["gray_equipment"])
        self.assertIsNone(ready["gray_card_book"])
        for unit_id, version in (("archer_queen", "18.600.7"), ("grand_warden", "unknown"),
                                 ("grand_warden", None), (None, "18.600.7")):
            with self.subTest(unit_id=unit_id, version=version):
                wrong = recognize_hero_state(self.low_health_fixture("ready"), self.low_health_slot,
                                             unit_id=unit_id, client_version=version)
                self.assertIsNone(wrong["deployed"])
                self.assertIsNone(wrong["ability_ready"])

    def test_gray_low_health_card_and_undeployed_card_are_not_ready_or_used(self) -> None:
        for path, slot in ((self.low_health_fixture("gray"), self.low_health_slot),
                           (self.root / "tests/fixtures/hero-border-before.png", [502, 592, 591, 711])):
            with self.subTest(path=path.name):
                state = recognize_hero_state(path, slot, unit_id="grand_warden", client_version="18.600.7")
                self.assertEqual(state["state"], "unknown")
                self.assertIsNone(state["deployed"])
                self.assertIsNone(state["ability_ready"])
                self.assertIsNone(state["ability_used"])
                self.assertIsNone(state["defeated"])
        gray = recognize_hero_state(self.low_health_fixture("gray"), self.low_health_slot,
                                    unit_id="grand_warden", client_version="18.600.7")
        self.assertEqual(gray["evidence"]["phase_portrait"]["phase"], "used")
        self.assertIsNone(gray["evidence"]["health_bar"])

    def test_low_health_ready_fails_when_any_independent_anchor_is_destroyed(self) -> None:
        original = cv2.imdecode(np.fromfile(self.low_health_fixture("ready"), dtype=np.uint8), cv2.IMREAD_COLOR)
        regions = {"phase_portrait": (545, 598, 581, 627),
                   "active_equipment": (507, 532, 541, 563),
                   "card_book_glow": (520, 651, 580, 677)}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "one-anchor-missing.png"
            for name, (left, top, right, bottom) in regions.items():
                with self.subTest(anchor=name):
                    image = original.copy()
                    image[top:bottom, left:right] = (30, 30, 30)
                    cv2.imencode(".png", image)[1].tofile(path)
                    state = recognize_hero_state(path, self.low_health_slot,
                                                 unit_id="grand_warden", client_version="18.600.7")
                    self.assertIsNone(state["deployed"])
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])
                    self.assertIsNone(state["defeated"])

    def duke_card_image(self) -> np.ndarray:
        # Redacted 100x150 card crop from daily-combined-4 scout frame 00039.
        crop = cv2.imdecode(np.fromfile(self.root / "tests/fixtures/hero-duke-undeployed-20260926.png",
                                          dtype=np.uint8), cv2.IMREAD_COLOR)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[565:715, 595:695] = crop
        return image

    def test_current_frame_portrait_allows_petless_duke_but_not_stale_identity(self) -> None:
        slot = [599, 595, 690, 711]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duke.png"
            cv2.imwrite(str(path), self.duke_card_image())
            state = recognize_hero_state(path, slot, unit_id="dragon_duke", client_version="18.600.7")
            self.assertEqual(state["evidence"]["portrait_identity"]["unit_id"], "dragon_duke")
            self.assertNotIn("pet_anchor", state["evidence"])
            self.assertFalse(state["selected"])
            self.assertIsNone(state["deployed"])
            self.assertIsNone(state["ability_ready"])
            for unit_id, version in (("archer_queen", "18.600.7"), ("dragon_duke", "unknown")):
                with self.subTest(unit_id=unit_id, version=version):
                    mismatch = recognize_hero_state(path, slot, unit_id=unit_id, client_version=version)
                    self.assertEqual(mismatch["state"], "unknown")
                    self.assertIsNone(mismatch["selected"])
                    self.assertIsNone(mismatch["deployed"])
                    self.assertEqual(mismatch["evidence"]["reason"], "current_frame_hero_identity_unverified")

    def test_petless_duke_requires_enclosed_hp_and_never_infers_ability(self) -> None:
        image = self.duke_card_image()
        slot = [599, 595, 690, 711]
        # Controlled counterfactual: only the HP-frame pixels are changed. This
        # exercises the geometry without claiming a live Duke deployment sample.
        image[575:588, 605:684] = 25
        image[579:585, 608:650] = (0, 240, 0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hp.png"
            cv2.imwrite(str(path), image)
            state = recognize_hero_state(path, slot, unit_id="dragon_duke", client_version="18.600.7")
            self.assertTrue(state["deployed"])
            self.assertFalse(state["defeated"])
            self.assertIsNone(state["ability_ready"])
            self.assertIsNone(state["ability_used"])
            image[575:579, 605:684] = 220
            cv2.imwrite(str(path), image)
            no_frame = recognize_hero_state(path, slot, unit_id="dragon_duke", client_version="18.600.7")
            self.assertIsNone(no_frame["deployed"])

    def test_real_duke_deployed_phase_requires_fresh_portrait_and_enclosed_hp(self) -> None:
        # Source and held-out frames are consecutive real battle observations;
        # the fixture retains only the Duke card and its HP frame.
        path = self.root / "tests/fixtures/hero-duke-deployed-20260926.png"
        slot = [599, 595, 690, 711]
        state = recognize_hero_state(path, slot, unit_id="dragon_duke", client_version="18.600.7")
        self.assertEqual(state["state"], "deployed")
        self.assertTrue(state["deployed"])
        self.assertEqual(state["evidence"]["portrait_identity"]["unit_id"], "dragon_duke")
        self.assertIn("dragon_duke_battle_deployed", state["evidence"]["portrait_identity"]["evidence"][0]["template"])
        self.assertGreaterEqual(min(state["evidence"]["health_bar"]["frame_support"]), .85)
        self.assertIsNone(state["ability_ready"])
        self.assertIsNone(state["ability_used"])
        for unit_id, version in (("archer_queen", "18.600.7"), ("dragon_duke", "unknown")):
            with self.subTest(unit_id=unit_id, version=version):
                wrong = recognize_hero_state(path, slot, unit_id=unit_id, client_version=version)
                self.assertEqual(wrong["state"], "unknown")
                self.assertIsNone(wrong["deployed"])
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        with TemporaryDirectory() as directory:
            modified = Path(directory) / "modified.png"
            for region in ((638, 636, 683, 678), (605, 567, 683, 581)):
                with self.subTest(region=region):
                    changed = image.copy()
                    left, top, right, bottom = region
                    changed[top:bottom, left:right] = (20, 20, 20)
                    cv2.imencode(".png", changed)[1].tofile(modified)
                    result = recognize_hero_state(modified, slot, unit_id="dragon_duke",
                                                  client_version="18.600.7")
                    self.assertIsNone(result["deployed"])

    def test_selected_border_still_needs_fresh_hero_portrait(self) -> None:
        path = self.root / "tests/fixtures/hero-border-selected.png"
        slot = [502, 592, 591, 711]
        selected = recognize_hero_state(path, slot, unit_id="grand_warden", client_version="18.600.7")
        self.assertTrue(selected["selected"])
        self.assertIsNone(selected["deployed"])
        wrong = recognize_hero_state(path, slot, unit_id="dragon_duke", client_version="18.600.7")
        self.assertIsNone(wrong["selected"])

    def test_selected_border_with_shaded_segment_requires_a_closed_outline(self) -> None:
        folder = self.root / "tests/fixtures"
        slot = [502, 592, 591, 711]
        before = recognize_hero_state(folder / "hero-border-before.png", slot,
                                      unit_id="grand_warden", client_version="18.600.7")
        selected = recognize_hero_state(folder / "hero-border-selected.png", slot,
                                        unit_id="grand_warden", client_version="18.600.7")
        self.assertFalse(before["selected"])
        self.assertTrue(selected["selected"])
        self.assertEqual(selected["state"], "selected")
        self.assertIsNone(selected["deployed"])
        self.assertIsNone(selected["ability_ready"])

    def fixture(self, number: int) -> Path:
        folder = self.root / "reports/live-20260922/hero-state-023106/frames"
        matches = list(folder.glob(f"{number:05d}-*.png"))
        if not matches:
            self.skipTest("Optional controlled hero-state fixture is absent")
        return matches[0]

    def changed(self, image: np.ndarray) -> dict:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "changed.png"
            cv2.imencode(".png", image)[1].tofile(path)
            return recognize_hero_state(path, self.slot)

    def baseline_image(self) -> np.ndarray:
        return cv2.resize(cv2.imdecode(np.fromfile(self.fixture(4), dtype=np.uint8), cv2.IMREAD_COLOR),
                          (1280, 720), interpolation=cv2.INTER_AREA)

    def test_white_selection_is_separate_from_deployment_and_ability(self) -> None:
        before = recognize_hero_state(self.fixture(1), self.slot)
        selected = recognize_hero_state(self.fixture(2), self.slot)
        self.assertFalse(before["selected"])
        self.assertTrue(selected["selected"])
        self.assertEqual(selected["state"], "selected")
        for result in (before, selected):
            self.assertIsNone(result["deployed"])
            self.assertIsNone(result["ability_ready"])
            self.assertIsNone(result["ability_used"])
            self.assertIsNone(result["defeated"])

    def test_both_stable_deployed_frames_have_health_and_warden_ready_evidence(self) -> None:
        for number in (3, 4):
            with self.subTest(frame=number):
                result = recognize_hero_state(self.fixture(number), self.slot)
                self.assertTrue(result["deployed"])
                self.assertTrue(result["ability_ready"])
                self.assertFalse(result["selected"])
                self.assertEqual(result["state"], "ability_ready")
                self.assertGreaterEqual(min(result["evidence"]["health_bar"]["frame_support"]), .85)
                self.assertEqual(result["evidence"]["ability_ready"]["hero"], "grand_warden")
                self.assertEqual(result["evidence"]["ability_ready"]["equipment_layout"], "observed_gold_book")
                self.assertFalse(result["ability_used"])
                self.assertFalse(result["defeated"])

    def test_old_selected_hero_with_deployed_siege_does_not_inherit_its_health_bar(self) -> None:
        path = self.root / "reports/20260923-021847-763016-3632dc81/frames/00015-deploy-support-selected.png"
        if not path.is_file():
            self.skipTest("Optional older selected-hero negative is absent")
        result = recognize_hero_state(path, self.slot)
        self.assertTrue(result["selected"])
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_intermediate_selection_animation_is_identified_without_deployment(self) -> None:
        path = self.root / "reports/20260924-003409-344711-0e4a3e7a/frames/00024-deploy-support-selected.png"
        if not path.is_file():
            self.skipTest("Optional intermediate selected-hero fixture is absent")
        result = recognize_hero_state(path, [502, 595, 590, 711])
        self.assertTrue(result["selected"])
        self.assertEqual(result["evidence"]["pet_anchor"]["scale"], 1.05)
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_deployed_pet_icons_over_map_background_still_require_health(self) -> None:
        folder = self.root / "reports/20260924-004347-675110-2e2b80bd/frames"
        for number, slot in ((31, [599, 595, 687, 711]), (35, [696, 595, 784, 711])):
            with self.subTest(frame=number):
                paths = list(folder.glob(f"{number:05d}-*.png"))
                if not paths:
                    self.skipTest("Optional varied-background hero fixture is absent")
                result = recognize_hero_state(paths[0], slot)
                self.assertTrue(result["deployed"])
                self.assertEqual(result["evidence"]["pet_anchor"]["comparison"], "opaque_pet_foreground")
                self.assertIsNot(result["ability_used"], True)
                image = cv2.resize(cv2.imread(str(paths[0])), (1280, 720), interpolation=cv2.INTER_AREA)
                image[567:585, slot[0]:slot[2]] = (25, 25, 25)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "missing-health.png"
                    cv2.imwrite(str(path), image)
                    self.assertIsNone(recognize_hero_state(path, slot)["deployed"])

    def test_green_patch_without_dark_hp_frame_is_not_deployment(self) -> None:
        image = self.baseline_image()
        image[571:574, 604:684] = (220, 220, 220)
        image[582:585, 604:684] = (220, 220, 220)
        result = self.changed(image)
        self.assertIn("pet_anchor", result["evidence"])
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_gray_ability_button_alone_does_not_prove_used(self) -> None:
        image = self.baseline_image()
        image[526:568, 600:642] = cv2.cvtColor(cv2.cvtColor(image[526:568, 600:642], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        result = self.changed(image)
        self.assertTrue(result["deployed"])
        self.assertIsNone(result["ability_ready"])
        self.assertIsNone(result["ability_used"])

    def test_controlled_ability_use_gray_card_is_alive_not_defeated(self) -> None:
        folder = self.root / "reports/live-20260922/warden-activate-023930/frames"
        if not folder.is_dir():
            self.skipTest("Optional controlled ability-use fixture is absent")
        before = recognize_hero_state(folder / "00001-before.png", self.slot)
        self.assertTrue(before["ability_ready"])
        self.assertFalse(before["ability_used"])
        for number in (3, 4):
            with self.subTest(frame=number):
                result = recognize_hero_state(folder / f"{number:05d}-after-ability.png", self.slot)
                self.assertEqual(result["state"], "ability_used")
                self.assertTrue(result["deployed"])
                self.assertTrue(result["ability_used"])
                self.assertFalse(result["ability_ready"])
                self.assertFalse(result["defeated"])
                self.assertIn("health_bar", result["evidence"])
                self.assertEqual(result["evidence"]["ability_used"]["hero"], "grand_warden")

    def test_first_ability_flash_frame_is_not_stable_ready_or_used(self) -> None:
        path = self.root / "reports/live-20260922/warden-activate-023930/frames/00002-after-ability.png"
        if not path.is_file():
            self.skipTest("Optional ability-flash fixture is absent")
        result = recognize_hero_state(path, self.slot)
        self.assertIsNone(result["ability_ready"])
        self.assertIsNone(result["ability_used"])
        self.assertIsNone(result["defeated"])

    def test_gray_portrait_without_hp_cannot_prove_used_or_defeated(self) -> None:
        path = self.root / "reports/live-20260922/warden-activate-023930/frames/00004-after-ability.png"
        if not path.is_file():
            self.skipTest("Optional used portrait fixture is absent")
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR),
                           (1280, 720), interpolation=cv2.INTER_AREA)
        image[571:585, 604:684] = (35, 35, 35)
        result = self.changed(image)
        self.assertIn("pet_anchor", result["evidence"])
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_used"])
        self.assertIsNone(result["defeated"])

    def test_active_book_alone_without_warden_card_glow_is_not_ready(self) -> None:
        image = self.baseline_image()
        image[631:680, 603:685] = (70, 60, 80)
        result = self.changed(image)
        self.assertTrue(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_other_heroes_do_not_inherit_warden_ability_state(self) -> None:
        for slot in ([696, 595, 784, 711], [792, 595, 881, 711], [889, 595, 978, 711]):
            with self.subTest(slot=slot):
                result = recognize_hero_state(self.fixture(4), slot)
                self.assertIn("pet_anchor", result["evidence"])
                self.assertIsNone(result["ability_ready"])
                self.assertIsNone(result["ability_used"])

    def test_warden_glow_animation_phases_keep_both_positive_ready_anchors(self) -> None:
        folder = self.root / "reports/20260924-004347-675110-2e2b80bd/frames"
        for number in (27, 28, 29, 30, 31):
            with self.subTest(frame=number):
                paths = list(folder.glob(f"{number:05d}-*.png"))
                if not paths:
                    self.skipTest("Optional Warden glow-animation fixture is absent")
                result = recognize_hero_state(paths[0], [502, 595, 591, 711])
                self.assertTrue(result["deployed"])
                self.assertTrue(result["ability_ready"])
                self.assertFalse(result["ability_used"])
                self.assertGreaterEqual(result["evidence"]["ability_ready"]["active_equipment"]["confidence"], .9)
                self.assertGreaterEqual(result["evidence"]["ability_ready"]["card_book_glow"]["confidence"], .9)

    def test_missing_identity_and_invalid_card_geometry_stay_unknown(self) -> None:
        result = self.changed(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["deployed"])
        result = recognize_hero_state(self.fixture(4), [500, 300, 700, 500])
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["deployed"])

    def test_source_and_coordinate_scaling_preserve_state(self) -> None:
        result = recognize_hero_state(self.fixture(4), [1198, 1190, 1376, 1422], baseline_resolution=(2560, 1440))
        self.assertTrue(result["ability_ready"])
        resized = self.changed(self.baseline_image())
        self.assertTrue(resized["ability_ready"])
        self.assertEqual(result["evidence"]["health_bar"]["fill_bbox"], resized["evidence"]["health_bar"]["fill_bbox"])


class SampledHeroAbilityTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    folder = root / "reports/hero-ability-samples-20260924-015325-792650/frames"
    cases = (("barbarian_king", 599, 14, 16), ("minion_prince", 696, 20, 22), ("archer_queen", 792, 26, 28))

    def fixture(self, number):
        paths = list(self.folder.glob(f"{number:05d}-*.png"))
        if not paths:
            self.skipTest("Optional controlled three-hero ability fixture is absent")
        return paths[0]

    def read(self, number, left, image=None):
        slot = [left, 595, left + 89, 711]
        if image is None:
            return recognize_hero_state(self.fixture(number), slot)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "changed.png"
            cv2.imwrite(str(path), image)
            return recognize_hero_state(path, slot)

    def image(self, number):
        return cv2.resize(cv2.imread(str(self.fixture(number))), (1280, 720), interpolation=cv2.INTER_AREA)

    def test_controlled_clicks_produce_ready_then_stable_used_states(self):
        for hero, left, before, after in self.cases:
            with self.subTest(hero=hero):
                ready = self.read(before, left)
                self.assertTrue(ready["deployed"])
                self.assertTrue(ready["ability_ready"])
                self.assertFalse(ready["ability_used"])
                self.assertEqual(ready["evidence"]["ability_ready"]["hero"], hero)
                for number in (after, after + 1):
                    used = self.read(number, left)
                    self.assertTrue(used["deployed"])
                    self.assertTrue(used["ability_used"])
                    self.assertFalse(used["ability_ready"])
                    self.assertFalse(used["defeated"])
                    self.assertEqual(used["evidence"]["ability_used"]["hero"], hero)
                # Enlarged first animation frames do not justify another click.
                transient = self.read(before + 1, left)
                self.assertIsNone(transient["ability_ready"])
                self.assertIsNone(transient["ability_used"])

    def test_undeployed_and_selected_heroes_are_not_ready_or_used(self):
        for hero, left, before, _ in self.cases:
            with self.subTest(hero=hero):
                for number in (before - 2, before - 1):
                    state = self.read(number, left)
                    self.assertIsNone(state["deployed"])
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])

    def test_removing_health_preserves_identity_but_cancels_ability_evidence(self):
        for hero, left, before, after in self.cases:
            for number in (before, after):
                with self.subTest(hero=hero, frame=number):
                    image = self.image(number)
                    x0, y0, x1, y1 = self.read(number, left)["evidence"]["health_bar"]["fill_bbox"]
                    image[y0:y1, x0:x1] = 25
                    state = self.read(number, left, image)
                    self.assertIn("pet_anchor", state["evidence"])
                    self.assertIsNone(state["deployed"])
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])
                    self.assertIsNone(state["defeated"])

    def test_equipment_and_portrait_or_weapon_are_both_required(self):
        regions = (("ready_equipment", 0, 528, 42, 569), ("ready_portrait", 40, 594, 82, 646),
                   ("used_portrait", 40, 594, 82, 646), ("used_weapon", 35, 647, 82, 679))
        for hero, left, before, after in self.cases:
            for name, x0, y0, x1, y1 in regions:
                with self.subTest(hero=hero, removed=name):
                    number = before if name.startswith("ready") else after
                    image = self.image(number)
                    image[y0:y1, left + x0:left + x1] = 25
                    state = self.read(number, left, image)
                    self.assertTrue(state["deployed"])
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])

    def test_pet_reassignment_does_not_transfer_another_hero_ability(self):
        image, donor = self.image(14), self.image(20)
        a = self.read(14, 599)["evidence"]["pet_anchor"]["bbox"]
        b = self.read(20, 696)["evidence"]["pet_anchor"]["bbox"]
        image[a[1]:a[3], a[0]:a[2]] = cv2.resize(donor[b[1]:b[3], b[0]:b[2]], (a[2] - a[0], a[3] - a[1]))
        state = self.read(14, 599, image)
        self.assertTrue(state["deployed"])
        self.assertTrue(state["evidence"]["pet_anchor"]["template"].endswith("battle_hero_pet_2.png"))
        self.assertIsNone(state["ability_ready"])
        self.assertIsNone(state["ability_used"])

    def test_empty_and_unrecognized_red_health_do_not_prove_ability_use(self):
        for left in (599, 696):
            with self.subTest(left=left):
                state = self.read(29, left)
                self.assertIsNone(state["deployed"])
                self.assertIsNone(state["ability_used"])
                self.assertIsNone(state["defeated"])


if __name__ == "__main__":
    unittest.main()
