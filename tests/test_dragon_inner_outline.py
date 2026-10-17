"""TRAIN-development checks for the source-bound current dragon inner rim.

The separate 063750 original-frame replay is an external diagnostic, not a
training fixture or an independent generalization result.
"""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.errors import FlowError
from autococ.frozen_line import (_closed_dragon_inner_outline_18_647_2,
                                 _file_sha256, _post_troop_reading)
from autococ.images import read_frame
from autococ.mixed_probe import _probe_count
from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures"
ASSET = Path(__file__).resolve().parents[1] / "assets" / "catalogs" / "units" / \
    "dragon_spent_inner_outline_18_647_2.png"
CASES = (
    ("dragon_post_014520_18_647_2", 38, 45, 46, (141, 595, 230, 711)),
    ("dragon_shifted_gray_041151_18_647_2", 28, 35, 36, (142, 595, 230, 711)),
    ("dragon_exact_gray_053700_18_647_2", 91, 97, 99, (141, 595, 230, 711)),
)


def _image(folder, number):
    return cv2.resize(read_frame(FIXTURES / folder / f"{number:05d}-toolbar.png"),
                      (1280, 720), interpolation=cv2.INTER_AREA)


def _session(path):
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    return SimpleNamespace(recognizer=recognizer, client_version="18.647.2",
                           last_snapshot=SimpleNamespace(
                               screenshot_path=path, scene="battle",
                               observations={"ocr": [], "battle": {"slots": []}}))


def test_versioned_asset_is_exact_crop_of_existing_train_development_fixture():
    metadata = json.loads(ASSET.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    source = Path(__file__).resolve().parents[1] / metadata["source_fixture"]
    assert metadata["source_role"] == "existing_TRAIN_development_fixture"
    assert sha256(source.read_bytes()).hexdigest() == metadata["source_fixture_sha256"]
    assert sha256(ASSET.read_bytes()).hexdigest() == metadata["asset_sha256"]
    x0, y0, x1, y1 = metadata["crop_at_1280x720"]
    expected = cv2.cvtColor(cv2.resize(read_frame(source), (1280, 720),
                                        interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    actual = cv2.imdecode(np.frombuffer(ASSET.read_bytes(), dtype=np.uint8),
                          cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("folder,source,selected,spent,bbox", CASES)
def test_train_development_complete_rim_is_unique_and_incomplete_cards_fail(
        folder, source, selected, spent, bbox):
    physical_origin = (141, 595)
    for number in (source, selected):
        assert _closed_dragon_inner_outline_18_647_2(
            _image(folder, number), bbox, physical_origin) is None
    proof = _closed_dragon_inner_outline_18_647_2(
        _image(folder, spent), bbox, physical_origin)
    assert proof is not None
    assert proof["physical_card_bbox"] == [141, 595, 230, 711]
    assert proof["source_card_bbox"] == list(bbox)
    assert proof["unique_match"] is True
    assert proof["masked_max_error"] == 0
    assert set(proof["segment_max_errors"]) == {"upper", "lower", "left", "right", "upper_join"}


@pytest.mark.parametrize("box", [
    (144, 646, 148, 676), (223, 646, 227, 676),  # straight sides
    (144, 599, 148, 604), (223, 599, 227, 604),  # top corners
    (160, 597, 210, 598), (169, 704, 202, 708),  # upper join, lower center
    (144, 704, 153, 708), (218, 704, 227, 708),  # bottom corners
])
def test_disconnected_or_missing_one_segment_rejects(box):
    image = _image(CASES[2][0], 99)
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = 0
    assert _closed_dragon_inner_outline_18_647_2(
        image, CASES[2][4], (141, 595)) is None


def test_wrong_source_geometry_portrait_and_asset_reject(monkeypatch):
    image = _image(CASES[2][0], 99)
    bbox = CASES[2][4]
    for wrong in ((130, 595, 219, 711), (151, 595, 240, 711),
                  (230, 595, 319, 711)):
        assert _closed_dragon_inner_outline_18_647_2(image, wrong, (141, 595)) is None
    assert _closed_dragon_inner_outline_18_647_2(image, bbox, (149, 595)) is None
    monkeypatch.setattr("autococ.unit_catalog.CATALOG_ROOT", Path(__file__).parent)
    assert _closed_dragon_inner_outline_18_647_2(image, bbox, (141, 595)) is None


def test_mixed_fallback_uses_only_current_reading_with_exact_binding(monkeypatch):
    folder, source_no, _, spent_no, bbox = CASES[2]
    source = FIXTURES / folder / f"{source_no:05d}-toolbar.png"
    spent = FIXTURES / folder / f"{spent_no:05d}-toolbar.png"
    session = _session(spent)
    point = (185, 653)
    actual_reading = _post_troop_reading(session, spent, str(source), point, bbox,
                                        unit_id="dragon")
    assert actual_reading["count"] == 0
    proof = _closed_dragon_inner_outline_18_647_2(_image(folder, spent_no), bbox,
                                                   (141, 595))
    assert proof is not None
    proof.update({"frame": str(spent), "frame_sha256": _file_sha256(spent),
                  "source_frame": str(source), "source_sha256": _file_sha256(source),
                  "client_version": "18.647.2", "unique_portrait": True})
    base = SimpleNamespace(toolbar_before_frame_id=str(source),
                           toolbar_before_sha256=_file_sha256(source),
                           client_version="18.647.2",
                           frame=SimpleNamespace(screenshot_path=source, frame_id=str(source),
                                                 resolution=(1280, 720)),
                           frame_sha256=_file_sha256(source))
    plan = SimpleNamespace(base=base, dragon_point=point, dragon_bbox=bbox)
    monkeypatch.setattr("autococ.mixed_probe._closed_card_border", lambda *_: None)
    monkeypatch.setattr("autococ.mixed_probe._closed_dragon_card_border_18_647_2", lambda *_: None)
    current = {**actual_reading, "current_card_border": proof}
    monkeypatch.setattr("autococ.mixed_probe._post_troop_reading", lambda *_a, **_k: current)
    assert _probe_count(session, plan, session.last_snapshot, "dragon", 0,
                        selected=True, pending=1)["current_card_border"] == proof
    for key, value in (("frame", "older-frame"), ("frame_sha256", "0" * 64),
                       ("source_frame", "other-source"), ("source_sha256", "0" * 64),
                       ("source_card_bbox", [230, 595, 319, 711]),
                       ("client_version", "unknown"), ("unique_match", False),
                       ("unique_portrait", False)):
        current["current_card_border"] = {**proof, key: value}
        with pytest.raises(FlowError, match="closed card border"):
            _probe_count(session, plan, session.last_snapshot, "dragon", 0,
                         selected=True, pending=1)
    current["current_card_border"] = proof
    session.last_snapshot.observations["current_card_border"] = proof
    current.pop("current_card_border")
    with pytest.raises(FlowError, match="closed card border"):
        _probe_count(session, plan, session.last_snapshot, "dragon", 0,
                     selected=True, pending=1)
    current["count"] = 1
    with pytest.raises(FlowError, match="count differs"):
        _probe_count(session, plan, session.last_snapshot, "dragon", 0,
                     selected=True, pending=None)


def test_anonymous_063750_rule_development_replay_is_source_bound():
    folder = FIXTURES / "dragon_inner_outline_063750_18_647_2"
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    assert "never_TRAIN" in provenance["allocation"]
    x0, y0, x1, y1 = provenance["retained_roi_native"]
    for filename, row in provenance["assets"].items():
        path = folder / filename
        image = read_frame(path).copy()
        assert sha256(path.read_bytes()).hexdigest() == row["fixture_sha256"]
        assert sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == row["retained_pixel_sha256"]
        image[y0:y1, x0:x1] = 0
        assert np.max(image) == 0
    source = folder / "00020-toolbar.png"
    point, bbox = (185, 653), (141, 595, 230, 711)
    for number, expected in ((27, 1), (28, 0)):
        current = folder / f"{number:05d}-toolbar.png"
        session = _session(current)
        reading = _post_troop_reading(session, current, str(source), point, bbox,
                                      unit_id="dragon")
        assert reading["count"] == expected
        assert reading["portrait_score"] >= .92
        if expected == 0:
            border = reading["current_card_border"]
            assert border["method"] == "18.647.2_dragon_source_bound_inner_outline"
            assert border["frame_sha256"] == _file_sha256(current)
            assert border["source_sha256"] == _file_sha256(source)
            base = SimpleNamespace(toolbar_before_frame_id=str(source),
                                   toolbar_before_sha256=_file_sha256(source),
                                   client_version="18.647.2",
                                   frame=SimpleNamespace(screenshot_path=source,
                                                         frame_id=str(source),
                                                         resolution=(1280, 720)),
                                   frame_sha256=_file_sha256(source))
            plan = SimpleNamespace(base=base, dragon_point=point, dragon_bbox=bbox)
            proof = _probe_count(session, plan, session.last_snapshot, "dragon", 0,
                                 selected=True, pending=1)
            assert proof["count"] == 0
            assert proof["current_card_border"]["unique_match"] is True


@pytest.mark.parametrize("contradiction", [1, 2])
def test_anonymous_063750_strong_count_conflict_still_rejects(monkeypatch, contradiction):
    folder = FIXTURES / "dragon_inner_outline_063750_18_647_2"
    source, current = folder / "00020-toolbar.png", folder / "00028-toolbar.png"
    session = _session(current)
    reader = session.recognizer.provider.recognize_line_image

    def contradictory_raw_tight(image):
        if image.ndim == 3 and image.shape[:2] == (62, 86):
            return [OCRText(f"x{contradiction}", .97)]
        return reader(image)

    monkeypatch.setattr(session.recognizer.provider, "recognize_line_image",
                        contradictory_raw_tight)
    reading = _post_troop_reading(session, current, str(source), (185, 653),
                                  (141, 595, 230, 711), unit_id="dragon")
    assert reading["reason"] == "strong_count_conflict"
    assert reading["counts"] == [0, contradiction]
