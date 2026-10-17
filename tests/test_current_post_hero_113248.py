"""Current-client hero poses from one TRAIN battle; no model truth labels."""

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
PROVENANCE = FIXTURES / "current_post_hero_113248.provenance.json"
VERSION = "18.647.2"
HEROES = ("grand_warden", "dragon_duke", "minion_prince", "archer_queen")


def frame(number: int) -> Path:
    return FIXTURES / f"current_post_hero_113248_{number:05d}.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CurrentPostHero113248Tests(unittest.TestCase):
    def test_source_binding_and_agent_visual_scope(self) -> None:
        proof = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        self.assertEqual(proof["client_version"], VERSION)
        self.assertEqual(proof["source_split"], "TRAIN")
        self.assertIn("not human ground truth", proof["review"])
        manifest = template_manifest(client_version=VERSION)["templates"]
        self.assertEqual({entry["unit_id"] for entry in proof["assets"]},
                         {"archer_queen", "minion_prince"})
        for entry in proof["assets"]:
            with self.subTest(asset=entry["path"]):
                self.assertIn(entry, manifest)
                self.assertEqual(entry["state"], "deployed")
                self.assertEqual(sha256(CATALOG_ROOT / entry["path"]), entry["template_sha256"])
        for name, item in proof["fixtures"].items():
            with self.subTest(fixture=name):
                self.assertEqual(sha256(FIXTURES / name), item["fixture_sha256"])
                self.assertEqual(item["retained_roi_1280"], [0, 545, 1280, 720])
                self.assertEqual(item["split"], "TRAIN")
                source = Path("E:/projects/AutoCOC") / item["source_frame"]
                if source.is_file():
                    self.assertEqual(sha256(source), item["source_sha256"])

    def test_new_poses_complete_four_independent_same_frame_hero_proofs(self) -> None:
        for number in (156, 157):
            with self.subTest(frame=number):
                boxes = []
                for hero in HEROES:
                    locator = locate_current_deployed_hero_card(frame(number), hero,
                                                                client_version=VERSION)
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
                    self.assertEqual(state["evidence"]["portrait_identity"]["unit_id"], hero)
                    self.assertIs(state["evidence"]["portrait_identity"]["version_verified"], True)
                    if hero != "grand_warden":
                        outline = state["evidence"]["current_deployed_card_outline"]
                        self.assertIs(outline["verified"], True)
                        self.assertGreaterEqual(outline["directed_side_edges"][0], 150)
                        self.assertLessEqual(outline["directed_side_edges"][1], -150)
                self.assertEqual(len(set(boxes)), len(HEROES))

    def test_transient_minion_prince_stays_unknown(self) -> None:
        # 155 has a transparent card transition, no complete bottom or HP.
        self.assertIsNone(locate_current_deployed_hero_card(frame(155), "minion_prince",
                                                         client_version=VERSION))
        self.assertIsNotNone(locate_current_deployed_hero_card(frame(155), "archer_queen",
                                                            client_version=VERSION))

    def test_distinct_prior_train_battle_poses_still_work(self) -> None:
        # Previously sampled 005304 frames are separate from the 113248 run.
        for number in (15, 16):
            prior = FIXTURES / f"current_post_hero_005304_{number:05d}.png"
            for hero in ("minion_prince", "archer_queen"):
                with self.subTest(frame=number, hero=hero):
                    locator = locate_current_deployed_hero_card(prior, hero,
                                                                client_version=VERSION)
                    self.assertIsNotNone(locator)
                    state = recognize_hero_state(prior, locator["bbox"],
                                                 unit_id=hero, client_version=VERSION)
                    self.assertIs(state["deployed"], True)
                    self.assertEqual(state["evidence"]["portrait_identity"]["unit_id"], hero)

    def test_wrong_identity_version_or_geometry_never_borrows_new_pose(self) -> None:
        for number, hero, wrong in ((156, "archer_queen", "minion_prince"),
                                    (157, "minion_prince", "archer_queen")):
            with self.subTest(frame=number, hero=hero):
                locator = locate_current_deployed_hero_card(frame(number), hero,
                                                            client_version=VERSION)
                self.assertIsNotNone(locator)
                self.assertIsNone(locate_current_deployed_hero_card(frame(number), hero,
                                                                 client_version="18.600.7"))
                mistaken = recognize_hero_state(frame(number), locator["bbox"],
                                                unit_id=wrong, client_version=VERSION)
                self.assertIsNone(mistaken["deployed"])
                shifted_box = [v + 12 if i % 2 == 0 else v for i, v in enumerate(locator["bbox"])]
                shifted = recognize_hero_state(frame(number), shifted_box,
                                               unit_id=hero, client_version=VERSION)
                self.assertIsNone(shifted["deployed"])
                self.assertIs(shifted["evidence"]["current_deployed_card_outline"]["verified"], False)

    def test_missing_portrait_hp_or_closed_card_cannot_verify(self) -> None:
        proof = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        by_hero = {entry["unit_id"]: entry for entry in proof["assets"]}
        for number, hero in ((156, "archer_queen"), (157, "minion_prince")):
            with self.subTest(frame=number, hero=hero):
                image = cv2.imdecode(np.fromfile(frame(number), np.uint8), cv2.IMREAD_COLOR)
                locator = locate_current_deployed_hero_card(frame(number), hero,
                                                            client_version=VERSION)
                self.assertIsNotNone(locator)
                left, _, right, _ = locator["bbox"]
                state = recognize_hero_state(frame(number), locator["bbox"],
                                             unit_id=hero, client_version=VERSION)
                fill = state["evidence"]["health_bar"]["fill_bbox"]
                x0, y0, x1, y1 = by_hero[hero]["source_bbox_1280"]
                variants = {}
                variants["portrait"] = image.copy()
                variants["portrait"][y0:y1, x0:x1] = 20
                variants["shifted_portrait"] = image.copy()
                variants["shifted_portrait"][y0:y1, x0:x1] = 20
                variants["shifted_portrait"][y0:y1, x0 + 12:x1 + 12] = image[y0:y1, x0:x1]
                variants["health"] = image.copy()
                variants["health"][fill[1]:fill[3], fill[0]:fill[2]] = 20
                variants["left_edge"] = image.copy()
                variants["left_edge"][625:695, left - 4:left + 6] = 20
                variants["right_edge"] = image.copy()
                variants["right_edge"][625:695, right - 6:right + 4] = 20
                variants["bottom_edge"] = image.copy()
                variants["bottom_edge"][709:713, left + 8:right - 8] = 20
                with TemporaryDirectory() as directory:
                    for name, changed in variants.items():
                        with self.subTest(frame=number, missing=name):
                            path = Path(directory) / f"{name}.png"
                            cv2.imencode(".png", changed)[1].tofile(path)
                            revised = locate_current_deployed_hero_card(path, hero,
                                                                        client_version=VERSION)
                            if name == "health":
                                self.assertIsNotNone(revised)
                                unknown = recognize_hero_state(path, revised["bbox"],
                                                               unit_id=hero, client_version=VERSION)
                                self.assertIsNone(unknown["deployed"])
                            else:
                                self.assertIsNone(revised)


if __name__ == "__main__":
    unittest.main()
