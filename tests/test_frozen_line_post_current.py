"""Anonymized current-client toolbar replay; no game/device interaction."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from autococ.frozen_line import _post_troop_reading
from autococ.hero_state import locate_current_deployed_hero_card, recognize_hero_state
from autococ.unit_catalog import CATALOG_ROOT, template_manifest
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures"


def toolbar(number: int) -> Path:
    return FIXTURES / f"frozen-line-{number:05d}-toolbar-current.png"


def test_real_toolbar_provenance_and_current_deployed_templates_are_hash_bound():
    evidence = json.loads((FIXTURES / "frozen-line-post-current.provenance.json").read_text(encoding="utf-8"))
    for name, row in evidence.items():
        assert sha256((FIXTURES / name).read_bytes()).hexdigest() == row["fixture_sha256"]
        assert row["retained_roi_1280"] == [0, 545, 1280, 720]
    entries = [row for row in template_manifest(client_version="18.647.2")["templates"]
               if row.get("state") == "deployed" and row.get("surface") == "battle"]
    assert {row["unit_id"] for row in entries} == {
        "grand_warden", "dragon_duke", "minion_prince", "archer_queen"}
    for row in entries:
        assert row["source_sha256"] == evidence[toolbar(16).name]["source_sha256"]
        assert sha256((CATALOG_ROOT / row["path"]).read_bytes()).hexdigest() == row["template_sha256"]


@pytest.mark.parametrize("hero_id", ["grand_warden", "dragon_duke", "minion_prince"])
def test_post_burst_heroes_relocated_without_frozen_slot(hero_id):
    # Samples come from frame 16; frame 15 is a distinct immediate post-burst
    # frame. Each identity still needs current card edges and enclosed HP.
    locator = locate_current_deployed_hero_card(toolbar(15), hero_id, client_version="18.647.2")
    assert locator is not None and locator["portrait_score"] >= .94
    state = recognize_hero_state(toolbar(15), locator["bbox"],
                                 unit_id=hero_id, client_version="18.647.2")
    assert state["deployed"] is True
    assert state["evidence"]["portrait_identity"]["unit_id"] == hero_id
    assert state["ability_ready"] is None


def test_queen_animation_requires_new_frame_and_has_independent_holdout():
    assert locate_current_deployed_hero_card(toolbar(12), "archer_queen", client_version="18.647.2") is None
    assert locate_current_deployed_hero_card(toolbar(15), "archer_queen", client_version="18.647.2") is None
    assert locate_current_deployed_hero_card(toolbar(16), "archer_queen", client_version="18.647.2") is not None
    # Frame 20 was neither the template source nor the immediate retry.
    locator = locate_current_deployed_hero_card(toolbar(20), "archer_queen", client_version="18.647.2")
    assert locator is not None
    assert recognize_hero_state(toolbar(20), locator["bbox"], unit_id="archer_queen",
                                client_version="18.647.2")["deployed"] is True
    assert locate_current_deployed_hero_card(toolbar(16), "archer_queen", client_version=None) is None


def test_deployed_hero_locator_requires_current_closed_card(tmp_path):
    image = cv2.imread(str(toolbar(16)))
    image[709:713, 889:978] = 0  # remove the queen's current bottom border
    altered = tmp_path / "broken-card.png"
    cv2.imwrite(str(altered), image)
    assert locate_current_deployed_hero_card(altered, "archer_queen", client_version="18.647.2") is None


def test_current_hero_face_without_hp_does_not_prove_deployment(tmp_path):
    image = cv2.imread(str(toolbar(16)))
    image[574:582, 607:678] = 0  # remove Warden's positive current HP fill
    altered = tmp_path / "missing-hp.png"
    cv2.imwrite(str(altered), image)
    locator = locate_current_deployed_hero_card(altered, "grand_warden", client_version="18.647.2")
    assert locator is not None
    state = recognize_hero_state(altered, locator["bbox"], unit_id="grand_warden",
                                 client_version="18.647.2")
    assert state["evidence"]["portrait_identity"]["unit_id"] == "grand_warden"
    assert state["deployed"] is None and state["ability_ready"] is None


def test_weak_full_ocr_locates_gray_dragon_but_fresh_narrow_ocr_proves_zero():
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    post = toolbar(15)
    weak = {"text": "x0", "confidence": .8671, "bbox": [234, 592, 276, 622]}
    snapshot = SimpleNamespace(screenshot_path=post, observations={"ocr": [weak]})
    session = SimpleNamespace(recognizer=recognizer, last_snapshot=snapshot)
    args = (session, post, str(toolbar(12)), (234, 650), (190, 595, 278, 711))
    reading = _post_troop_reading(*args)
    assert reading["count"] == 0 and reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92
    assert reading["locator"]["confidence"] < .9
    # A previous frame contributes only the ROI geometry: frame 16 is read anew.
    session._frozen_count_roi_hints = {(190, 595, 278, 711): reading["roi"]}
    session.last_snapshot = SimpleNamespace(screenshot_path=toolbar(16), observations={"ocr": []})
    later = _post_troop_reading(session, toolbar(16), str(toolbar(12)), (234, 650),
                                (190, 595, 278, 711))
    assert later["count"] == 0 and later["confidence"] >= .9
    assert later["frame"] == str(toolbar(16))
    assert later["locator"]["source"] == "prior_post_frame_roi_geometry"


def test_four_spent_troops_and_four_heroes_bind_on_same_fresh_toolbar_frame():
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    source = str(toolbar(12))
    cards = [
        ("electro_dragon", (94, 595, 181, 711), (138, 650)),
        ("dragon", (190, 595, 278, 711), (234, 650)),
        ("baby_dragon", (287, 595, 375, 711), (331, 650)),
        ("balloon", (385, 595, 474, 711), (429, 650)),
    ]
    hints = {}
    for number in (15, 16):
        path = toolbar(number)
        ocr = [{"text": "x0", "confidence": .8671, "bbox": [234, 592, 276, 622]}] if number == 15 else []
        session = SimpleNamespace(recognizer=recognizer,
            last_snapshot=SimpleNamespace(screenshot_path=path, observations={"ocr": ocr}),
            _frozen_count_roi_hints=hints)
        for _name, bbox, point in cards:
            reading = _post_troop_reading(session, path, source, point, bbox)
            assert reading["count"] == 0 and reading["confidence"] >= .9
            assert reading["frame"] == str(path) and reading["portrait_score"] >= .92
            hints[bbox] = reading["roi"]
        deployed = []
        for hero_id in ("grand_warden", "dragon_duke", "minion_prince", "archer_queen"):
            locator = locate_current_deployed_hero_card(path, hero_id, client_version="18.647.2")
            state = (recognize_hero_state(path, locator["bbox"], unit_id=hero_id,
                                          client_version="18.647.2") if locator else None)
            deployed.append(state is not None and state["deployed"] is True)
        assert deployed == ([True, True, True, False] if number == 15 else [True] * 4)


def test_weak_count_conflict_or_wrong_portrait_stays_unknown(tmp_path):
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    post = toolbar(15)
    snapshot = SimpleNamespace(screenshot_path=post, observations={"ocr": [
        {"text": "x1", "confidence": .85, "bbox": [234, 592, 276, 622]}]})
    session = SimpleNamespace(recognizer=recognizer, last_snapshot=snapshot)
    args = (session, post, str(toolbar(12)), (234, 650), (190, 595, 278, 711))
    assert _post_troop_reading(*args) == {}
    image = cv2.imread(str(post))
    image[633:676, 210:260] = 0  # erase this card's current portrait only
    altered = tmp_path / "wrong-portrait.png"
    cv2.imwrite(str(altered), image)
    session.last_snapshot = SimpleNamespace(screenshot_path=altered, observations={"ocr": [
        {"text": "x0", "confidence": .85, "bbox": [234, 592, 276, 622]}]})
    assert _post_troop_reading(session, altered, str(toolbar(12)), (234, 650),
                               (190, 595, 278, 711)) == {}
