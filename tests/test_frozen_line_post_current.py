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
from autococ.ocr import OCRText
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
    current = json.loads((FIXTURES / "frozen-line-001038.provenance.json").read_text(encoding="utf-8"))
    for name, row in current.items():
        if name.startswith("_"):
            continue
        assert sha256((FIXTURES / name).read_bytes()).hexdigest() == row["fixture_sha256"]
        assert row["retained_roi_native"] == [0, 1090, 2560, 1440]
    new_queen = current["_queen_template"]
    later = json.loads((FIXTURES / "current_post_hero_005304.provenance.json").read_text(encoding="utf-8"))
    later_sources = {row["path"]: row["source_sha256"] for row in later["assets"]}
    for row in entries:
        expected_source = (later_sources[row["path"]] if row["path"] in later_sources
                           else new_queen["source_sha256"] if row["path"].endswith("001038_18_647_2.png")
                           else evidence[toolbar(16).name]["source_sha256"])
        assert row["source_sha256"] == expected_source
        assert sha256((CATALOG_ROOT / row["path"]).read_bytes()).hexdigest() == row["template_sha256"]
    assert new_queen["template_sha256"] == next(row["template_sha256"] for row in entries
        if row["path"].endswith("001038_18_647_2.png"))


def current_010553_toolbar(number: int) -> Path:
    return FIXTURES / f"frozen-line-010553-{number:05d}-toolbar.png"


def _geometry_session(path: Path, *, ocr=None, client_version="18.647.2"):
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = client_version
    return SimpleNamespace(recognizer=recognizer, client_version=client_version,
        last_snapshot=SimpleNamespace(scene="battle", screenshot_path=path,
                                      observations={"ocr": [] if ocr is None else ocr}),
        _frozen_count_roi_hints={})


def _geometry_dragon(session, post: Path, *, source=None, unit_id="dragon"):
    return _post_troop_reading(session, post, str(source or current_010553_toolbar(11)),
                               (185, 653), (141, 595, 230, 711), unit_id=unit_id)


def test_010553_toolbar_provenance_and_current_frame_geometry_reading():
    evidence = json.loads((FIXTURES / "frozen-line-010553.provenance.json").read_text(encoding="utf-8"))
    assert set(evidence) == {current_010553_toolbar(n).name for n in (11, 14, 15, 16)}
    for name, row in evidence.items():
        assert sha256((FIXTURES / name).read_bytes()).hexdigest() == row["fixture_sha256"]
        assert row["retained_roi_native"] == [0, 1090, 2560, 1440]
    for number in (14, 15, 16):
        post = current_010553_toolbar(number)
        result = _geometry_dragon(_geometry_session(post), post)
        assert result["count"] == 0 and result["confidence"] >= .9
        assert result["portrait_score"] >= .92
        assert result["source_portrait_identity"]["unit_id"] == "dragon"
        assert result["source_portrait_identity"]["version_verified"] is True
        assert result["locator"]["source"] == "current_frame_closed_card_header_geometry"
        assert result["locator"]["bbox"] == [185, 592, 228, 623]
        assert result["current_card_border"]["left_edge"] >= 250
        assert result["current_card_border"]["right_edge"] <= -150
        assert result["current_card_border"]["bottom_contrast"] >= 40
        assert result["frame"] == str(post) and result["slot_bbox"] == [141, 595, 230, 711]


def test_010553_five_spent_troops_and_four_deployed_heroes_share_one_frame():
    post = current_010553_toolbar(15)
    source = str(current_010553_toolbar(11))
    session = _geometry_session(post)
    cards = (
        ("electro_dragon", (44, 591, 135, 708), (89, 649)),
        ("dragon", (141, 595, 230, 711), (185, 653)),
        ("baby_dragon", (238, 595, 327, 711), (282, 653)),
        ("balloon", (335, 595, 423, 711), (379, 653)),
        ("archer", (432, 594, 520, 711), (476, 652)),
    )
    for unit_id, bbox, point in cards:
        reading = _post_troop_reading(session, post, source, point, bbox, unit_id=unit_id)
        assert reading["count"] == 0 and reading["confidence"] >= .9, unit_id
        assert reading["portrait_score"] >= .92 and reading["frame"] == str(post)
        assert reading["slot_bbox"] == list(bbox)
    for hero_id in ("grand_warden", "dragon_duke", "minion_prince", "archer_queen"):
        locator = locate_current_deployed_hero_card(post, hero_id, client_version="18.647.2")
        assert locator is not None and locator["frame"] == str(post), hero_id
        state = recognize_hero_state(post, locator["bbox"], unit_id=hero_id,
                                     client_version="18.647.2")
        assert state["deployed"] is True, hero_id
        assert state["evidence"]["portrait_identity"]["unit_id"] == hero_id
        assert state["evidence"]["health_bar"] is not None


def test_010553_unspent_card_stays_positive_not_inferred_zero(monkeypatch):
    from autococ.battle_vision import BattleObserver

    post = current_010553_toolbar(11)
    monkeypatch.setattr(BattleObserver, "count", lambda *_args: None)
    result = _geometry_dragon(_geometry_session(post), post)
    assert result["count"] == 1 and result["confidence"] >= .9
    assert result["locator"]["source"] == "current_frame_closed_card_header_geometry"


def test_010553_geometry_rejects_raw_binary_or_full_ocr_strong_conflicts(monkeypatch):
    post = current_010553_toolbar(15)
    session = _geometry_session(post)
    from autococ.battle_vision import BattleObserver

    monkeypatch.setattr(BattleObserver, "count", lambda *_args: None)
    def contradiction(image):
        if image.shape[1] > 100:  # broad header, never the fixed right-aligned ROI
            return []
        return [OCRText("x0" if image.ndim == 2 else "x1", .98)]
    monkeypatch.setattr(session.recognizer.provider, "recognize_line_image", contradiction)
    result = _geometry_dragon(session, post)
    assert result["conflict"] is True and result["counts"] == [0, 1]
    session.last_snapshot.observations["ocr"] = [
        {"text": "x1", "confidence": .99, "bbox": [185, 592, 228, 623]}]
    monkeypatch.setattr(session.recognizer.provider, "recognize_line_image",
                        lambda _image: [OCRText("x0", .98)])
    result = _geometry_dragon(session, post)
    assert result["conflict"] is True and result["counts"] == [0, 1]


def test_010553_geometry_equal_confidence_raw_binary_does_not_compare_ocr_objects(monkeypatch):
    post = current_010553_toolbar(15)
    session = _geometry_session(post)
    from autococ.battle_vision import BattleObserver

    monkeypatch.setattr(BattleObserver, "count", lambda *_args: None)
    monkeypatch.setattr(session.recognizer.provider, "recognize_line_image",
                        lambda _image: [OCRText("x0", .98)])
    result = _geometry_dragon(session, post)
    assert result["count"] == 0 and result["confidence"] == .98


def test_010553_geometry_requires_same_frame_snapshot_and_current_client():
    post = current_010553_toolbar(15)
    session = _geometry_session(post)
    session.last_snapshot = None
    assert _geometry_dragon(session, post) == {}
    session = _geometry_session(post, client_version="unknown")
    assert _geometry_dragon(session, post) == {}
    session = _geometry_session(post)
    del session.client_version
    assert _geometry_dragon(session, post) == {}
    session = _geometry_session(post)
    session.last_snapshot.screenshot_path = current_010553_toolbar(14)
    assert _geometry_dragon(session, post) == {}
    assert _geometry_dragon(_geometry_session(post), post, unit_id="balloon") == {}
    assert _geometry_dragon(_geometry_session(post), post, unit_id=None) == {}


def test_010553_geometry_requires_current_portrait_and_closed_card(tmp_path):
    post = current_010553_toolbar(15)
    image = cv2.imread(str(post))
    variants = {}
    variants["portrait"] = image.copy()
    variants["portrait"][1266:1350, 320:432] = 20
    variants["left_edge"] = image.copy()
    variants["left_edge"][1250:1390, 274:290] = 20
    variants["right_edge"] = image.copy()
    variants["right_edge"][1250:1390, 452:468] = 20
    variants["bottom_edge"] = image.copy()
    variants["bottom_edge"][1412:1428, 298:444] = 20
    for name, pixels in variants.items():
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), pixels)
        assert _geometry_dragon(_geometry_session(path), path) == {}, name


def current_toolbar(number: int) -> Path:
    return FIXTURES / f"frozen-line-001038-{number:05d}-toolbar.png"


def test_five_card_real_post_dragon_requires_padded_independent_x0_proof():
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    first = current_toolbar(20)
    weak = {"text": "x0", "confidence": .89086, "bbox": [187, 594, 226, 621]}
    session = SimpleNamespace(recognizer=recognizer,
        last_snapshot=SimpleNamespace(screenshot_path=first, observations={"ocr": [weak]}),
        _frozen_count_roi_hints={})
    before = str(current_toolbar(18))
    args = (session, first, before, (185, 653), (141, 595, 230, 711))
    reading = _post_troop_reading(*args)
    assert reading["count"] == 0 and reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92 and reading["roi"] == [185, 592, 228, 623]
    assert reading["locator"]["confidence"] < .9
    for number in (21, 22):
        post = current_toolbar(number)
        session.last_snapshot = SimpleNamespace(screenshot_path=post, observations={"ocr": []})
        session._frozen_count_roi_hints = {(141, 595, 230, 711): reading["roi"]}
        reading = _post_troop_reading(session, post, before, (185, 653), (141, 595, 230, 711))
        assert reading["count"] == 0 and reading["confidence"] >= .9
        assert reading["frame"] == str(post)
        assert reading["locator"]["source"] == "prior_post_frame_roi_geometry"
    # An independently read x0 cannot override a contradictory same-frame x1.
    session.last_snapshot = SimpleNamespace(screenshot_path=first, observations={"ocr": [
        {"text": "x1", "confidence": .99, "bbox": [187, 594, 226, 621]}]})
    assert _post_troop_reading(*args).get("conflict") is True


def test_real_count_to_post_propagates_strong_crop_conflict_without_fallback(monkeypatch):
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    post = current_toolbar(20)
    session = SimpleNamespace(recognizer=recognizer,
        last_snapshot=SimpleNamespace(screenshot_path=post, observations={"ocr": [
            {"text": "x0", "confidence": .89, "bbox": [187, 594, 226, 621]}]}),
        _frozen_count_roi_hints={})
    args = (session, post, str(current_toolbar(18)), (185, 653), (141, 595, 230, 711))
    for outputs in (
        (("x0", "x1"),),  # raw crop disagrees internally
        (("x1", "xO"), ("x0",)),  # threshold rendition disagrees with raw crop
    ):
        calls = []
        responses = iter(outputs)
        def read(_image):
            calls.append(True)
            return [OCRText(value, .98) for value in next(responses)]
        monkeypatch.setattr(recognizer.provider, "recognize_line_image", read)
        result = _post_troop_reading(*args)  # real BattleObserver.count, current portrait and card
        assert result["reason"] == "strong_count_conflict"
        assert result["conflict"] is True and result["counts"] == [0, 1]
        assert result["frame"] == str(post)
        assert len(calls) == len(outputs)  # no tight/padded override after conflict


def test_current_header_strong_count_conflicts_with_independent_line_read(monkeypatch):
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    post = current_toolbar(20)
    header = {"text": "x1", "confidence": .99, "bbox": [187, 594, 226, 621]}
    session = SimpleNamespace(recognizer=recognizer,
        last_snapshot=SimpleNamespace(screenshot_path=post, observations={"ocr": [header]}),
        _frozen_count_roi_hints={})
    args = (session, post, str(current_toolbar(18)), (185, 653), (141, 595, 230, 711))
    monkeypatch.setattr(recognizer.provider, "recognize_line_image",
                        lambda _image: [OCRText("x0", .98)])
    result = _post_troop_reading(*args)
    assert result["conflict"] is True and result["counts"] == [0, 1]
    # A weaker locator is not independent numerical proof of a conflict.
    header["confidence"] = .85
    responses = iter(([OCRText("xO", .6)], [OCRText("xO", .6)],
                      [OCRText("x0", .98)], [OCRText("x0", .98)]))
    monkeypatch.setattr(recognizer.provider, "recognize_line_image", lambda _image: next(responses))
    assert _post_troop_reading(*args) == {}


def test_padded_dragon_read_rejects_conflicting_or_missing_glyphs(monkeypatch, tmp_path):
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    post = current_toolbar(20)
    weak = {"text": "x0", "confidence": .89086, "bbox": [187, 594, 226, 621]}
    session = SimpleNamespace(recognizer=recognizer,
        last_snapshot=SimpleNamespace(screenshot_path=post, observations={"ocr": [weak]}),
        _frozen_count_roi_hints={})
    before = str(current_toolbar(18))
    args = (session, post, before, (185, 653), (141, 595, 230, 711))
    with monkeypatch.context() as patcher:
        patcher.setattr("autococ.battle_vision.BattleObserver.count", lambda *_args: None)
        for outputs in (
            (("x0",), ("x1",)),
            (("x0",), ("x0", "x1")),
        ):
            responses = iter(outputs)
            patcher.setattr(recognizer.provider, "recognize_line_image",
                            lambda _image: [SimpleNamespace(text=value, confidence=.98)
                                            for value in next(responses)])
            assert _post_troop_reading(*args).get("conflict") is True  # same-frame reads disagree

    image = cv2.imread(str(post))
    image[1184:1246, 370:456] = 25  # native glyph region, not the saved portrait
    missing = tmp_path / "missing-numeral.png"
    cv2.imwrite(str(missing), image)
    session.last_snapshot = SimpleNamespace(screenshot_path=missing, observations={"ocr": [weak]})
    assert _post_troop_reading(session, missing, before, (185, 653), (141, 595, 230, 711)) == {}


def test_real_prior_unspent_dragon_cannot_be_counted_as_zero():
    root = Path(r"E:\projects\AutoCOC\reports\completion-20261015\live") / \
           "20261015-235131-370965-f8aa237e" / "frames"
    before, post = root / "00014-line-boundary.png", root / "00015-frozen-line-post.png"
    if not before.is_file() or not post.is_file():
        pytest.skip("local 235131 battle evidence unavailable")
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(recognizer=recognizer, last_snapshot=SimpleNamespace(
        screenshot_path=post, observations={"ocr": [{"text": "x1", "confidence": .99583,
                                                  "bbox": [195, 594, 228, 621]}]}),
        _frozen_count_roi_hints={})
    reading = _post_troop_reading(session, post, str(before), (185, 653), (141, 595, 230, 711))
    assert reading["count"] == 1 and reading["confidence"] >= .9


def test_current_queen_new_deployed_template_uses_independent_stable_frame():
    assert locate_current_deployed_hero_card(current_toolbar(20), "archer_queen",
                                             client_version="18.647.2") is None
    for number in (22, 23, 25):
        path = current_toolbar(number)
        locator = locate_current_deployed_hero_card(path, "archer_queen", client_version="18.647.2")
        assert locator is not None and locator["portrait_score"] >= .94
        state = recognize_hero_state(path, locator["bbox"], unit_id="archer_queen",
                                     client_version="18.647.2")
        if number in (22, 23):
            assert state["deployed"] is True
        assert state["ability_ready"] is None and state["ability_used"] is None
        assert locate_current_deployed_hero_card(path, "archer_queen", client_version=None) is None


def test_current_queen_face_without_hp_or_closed_card_cannot_prove_deployment(tmp_path):
    image = cv2.imread(str(current_toolbar(23)))
    image[1140:1180, 1870:2060] = 0  # native pixels: current card's enclosed HP
    no_hp = tmp_path / "no-hp.png"
    cv2.imwrite(str(no_hp), image)
    locator = locate_current_deployed_hero_card(no_hp, "archer_queen", client_version="18.647.2")
    assert locator is not None
    assert recognize_hero_state(no_hp, locator["bbox"], unit_id="archer_queen",
                                client_version="18.647.2")["deployed"] is None
    image = cv2.imread(str(current_toolbar(23)))
    image[1418:1426, 1876:2054] = 0  # remove closed bottom edge
    no_bottom = tmp_path / "no-bottom.png"
    cv2.imwrite(str(no_bottom), image)
    assert locate_current_deployed_hero_card(no_bottom, "archer_queen", client_version="18.647.2") is None
    image = cv2.imread(str(current_toolbar(23)))
    image[1264:1348, 1930:2004] = 0  # remove the matched Queen face only
    no_face = tmp_path / "no-queen-face.png"
    cv2.imwrite(str(no_face), image)
    assert locate_current_deployed_hero_card(no_face, "archer_queen", client_version="18.647.2") is None


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
