"""Independent-frame checks for sampled 18.647.2 deployed hero poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.hero_state import locate_current_deployed_hero_card, recognize_hero_state
from autococ.unit_catalog import CATALOG_ROOT, template_manifest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PROVENANCE = FIXTURES / "current_post_hero_005304.provenance.json"
VERSION = "18.647.2"
HEROES = ("grand_warden", "dragon_duke", "minion_prince", "archer_queen")


def frame(number: int) -> Path:
    return FIXTURES / f"current_post_hero_005304_{number:05d}.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CurrentPostHeroPoseTests(unittest.TestCase):
    def test_provenance_and_source_frames_are_separate_from_holdouts(self) -> None:
        metadata = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        self.assertEqual(metadata["client_version"], VERSION)
        manifest = template_manifest(client_version=VERSION)["templates"]
        holdouts = {metadata["fixtures"][frame(n).name]["source_frame"] for n in (14, 15, 16)}
        for name, item in metadata["fixtures"].items():
            with self.subTest(fixture=name):
                self.assertEqual(sha256(FIXTURES / name), item["fixture_sha256"])
                self.assertEqual(item["retained_roi_1280"], [0, 545, 1280, 720])
        for asset in metadata["assets"]:
            with self.subTest(asset=asset["path"]):
                self.assertIn(asset, manifest)
                self.assertEqual(asset["client_version"], VERSION)
                self.assertEqual(asset["state"], "deployed")
                self.assertNotIn(asset["source_frame"], holdouts)
                self.assertEqual(sha256(CATALOG_ROOT / asset["path"]), asset["template_sha256"])
                source = ROOT / asset["source_frame"]
                if source.is_file():
                    self.assertEqual(sha256(source), asset["source_sha256"])

    def test_two_independent_post_frames_verify_all_four_heroes_in_one_frame(self) -> None:
        for number in (15, 16):
            with self.subTest(frame=number):
                boxes = []
                for hero in HEROES:
                    locator = locate_current_deployed_hero_card(frame(number), hero, client_version=VERSION)
                    self.assertIsNotNone(locator, hero)
                    self.assertEqual(locator["frame"], str(frame(number)))
                    self.assertGreaterEqual(locator["portrait_score"], .94)
                    self.assertGreaterEqual(min(locator["card_sides"]), .5)
                    self.assertGreaterEqual(locator["bottom_contrast"], 40)
                    boxes.append(tuple(locator["bbox"]))
                    state = recognize_hero_state(frame(number), locator["bbox"],
                                                 unit_id=hero, client_version=VERSION)
                    self.assertIs(state["deployed"], True, hero)
                    self.assertIsNotNone(state["evidence"]["health_bar"])
                    identity = state["evidence"]["portrait_identity"]
                    self.assertEqual(identity["unit_id"], hero)
                    self.assertIs(identity["version_verified"], True)
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])
                self.assertEqual(len(set(boxes)), len(HEROES))

    def test_template_source_frames_have_independent_hp_evidence(self) -> None:
        # These are source frames, so their portrait matches are not counted
        # as holdout validation. Their enclosed HP bars establish that the
        # sampled faces came from deployed, living hero cards.
        for number, heroes in ((18, ("grand_warden", "minion_prince")),
                               (20, ("archer_queen",))):
            for hero in heroes:
                with self.subTest(source_frame=number, hero=hero):
                    locator = locate_current_deployed_hero_card(frame(number), hero,
                                                                client_version=VERSION)
                    self.assertIsNotNone(locator)
                    state = recognize_hero_state(frame(number), locator["bbox"],
                                                 unit_id=hero, client_version=VERSION)
                    self.assertIs(state["deployed"], True)
                    self.assertIsNotNone(state["evidence"]["health_bar"])

    def test_queen_transient_open_card_stays_unlocated(self) -> None:
        # 00014 precedes both holdouts; its Queen portrait and bottom border
        # are in transition. A new pose must not turn that frame positive.
        self.assertIsNone(locate_current_deployed_hero_card(frame(14), "archer_queen",
                                                         client_version=VERSION))

    def test_wrong_hero_and_wrong_client_cannot_borrow_a_pose(self) -> None:
        queen = locate_current_deployed_hero_card(frame(15), "archer_queen", client_version=VERSION)
        self.assertIsNotNone(queen)
        wrong_hero = recognize_hero_state(frame(15), queen["bbox"],
                                          unit_id="grand_warden", client_version=VERSION)
        self.assertIsNone(wrong_hero["deployed"])
        self.assertEqual(wrong_hero["evidence"]["reason"], "current_frame_hero_identity_unverified")
        self.assertIsNone(locate_current_deployed_hero_card(frame(15), "archer_queen",
                                                         client_version="18.600.7"))
        wrong_version = recognize_hero_state(frame(15), queen["bbox"],
                                             unit_id="archer_queen", client_version="unknown")
        self.assertIsNone(wrong_version["deployed"])

    def test_missing_portrait_health_or_card_edges_remains_unknown(self) -> None:
        image = cv2.imdecode(np.fromfile(frame(15), dtype=np.uint8), cv2.IMREAD_COLOR)
        queen = locate_current_deployed_hero_card(frame(15), "archer_queen", client_version=VERSION)
        self.assertIsNotNone(queen)
        left, _, right, _ = queen["bbox"]
        state = recognize_hero_state(frame(15), queen["bbox"],
                                     unit_id="archer_queen", client_version=VERSION)
        fill = state["evidence"]["health_bar"]["fill_bbox"]
        variants = {}
        variants["portrait"] = image.copy()
        variants["portrait"][632:675, left + 20:right - 5] = 20
        variants["health"] = image.copy()
        variants["health"][fill[1]:fill[3], fill[0]:fill[2]] = 20
        variants["left_edge"] = image.copy()
        variants["left_edge"][625:695, left - 4:left + 6] = 20
        variants["bottom_edge"] = image.copy()
        variants["bottom_edge"][709:713, left + 8:right - 8] = 20
        with TemporaryDirectory() as directory:
            for name, changed in variants.items():
                with self.subTest(missing=name):
                    path = Path(directory) / f"{name}.png"
                    cv2.imencode(".png", changed)[1].tofile(path)
                    locator = locate_current_deployed_hero_card(path, "archer_queen",
                                                                 client_version=VERSION)
                    if name == "health":
                        self.assertIsNotNone(locator)
                        missing = recognize_hero_state(path, locator["bbox"],
                                                       unit_id="archer_queen", client_version=VERSION)
                        self.assertIsNone(missing["deployed"])
                    else:
                        self.assertIsNone(locator)

    def test_later_absent_hp_is_not_deployed_from_identity_alone(self) -> None:
        locator = locate_current_deployed_hero_card(frame(19), "minion_prince", client_version=VERSION)
        self.assertIsNotNone(locator)
        state = recognize_hero_state(frame(19), locator["bbox"],
                                     unit_id="minion_prince", client_version=VERSION)
        self.assertEqual(state["evidence"]["portrait_identity"]["unit_id"], "minion_prince")
        self.assertIsNone(state["deployed"])


if __name__ == "__main__":
    unittest.main()
