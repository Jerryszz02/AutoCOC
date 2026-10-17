"""Source-bound 18.647.2 exact-slot dragon gray-card regression."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.errors import FlowError
from autococ.frozen_line import (_closed_dragon_card_border_18_647_2,
                                 _post_troop_reading)
from autococ.images import read_frame
from autococ.mixed_probe import _probe_count
from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures" / "dragon_exact_gray_053700_18_647_2"
BBOX = (141, 595, 230, 711)
POINT = (185, 653)


def frame(number: int) -> Path:
    return FIXTURES / f"{number:05d}-toolbar.png"


def session_for(path: Path, *, version: str = "18.647.2") -> SimpleNamespace:
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = version
    return SimpleNamespace(recognizer=recognizer, client_version=version,
                           last_snapshot=SimpleNamespace(
                               screenshot_path=path, scene="battle",
                               observations={"ocr": [], "battle": {"slots": []}}))


def plan_for(source: Path) -> SimpleNamespace:
    base = SimpleNamespace(toolbar_before_frame_id=str(source), client_version="18.647.2",
                           frame=SimpleNamespace(screenshot_path=source, frame_id=str(source),
                                                 resolution=(1280, 720)),
                           frame_sha256=sha256(source.read_bytes()).hexdigest())
    return SimpleNamespace(base=base, dragon_point=POINT, dragon_bbox=BBOX)


def test_anonymous_pixels_match_bound_original_toolbar_roi():
    meta = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf8"))
    assert meta["source_run"] == "20261017-053700-913562-8b4d661b"
    assert meta["client_version"] == "18.647.2"
    assert meta["retained_roi_native"] == [260, 1160, 500, 1440]
    x0, y0, x1, y1 = meta["retained_roi_native"]
    events = Path(meta["events_source"])
    if events.exists():
        assert sha256(events.read_bytes()).hexdigest() == meta["events_sha256"]
    for number in (91, 97, 98, 99):
        path = frame(number)
        row = meta["assets"][path.name]
        pixels = read_frame(path)
        assert sha256(path.read_bytes()).hexdigest() == row["fixture_sha256"]
        assert sha256(pixels[y0:y1, x0:x1].tobytes()).hexdigest() == \
            row["retained_pixel_sha256"]
        masked = pixels.copy()
        masked[y0:y1, x0:x1] = 0
        assert masked.max() == 0
        source = Path(row["source_frame"])
        if source.exists():
            assert sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
            assert np.array_equal(pixels[y0:y1, x0:x1], read_frame(source)[y0:y1, x0:x1])


def test_original_current_zero_needs_exact_gray_outline_and_all_probe_gates():
    source, current = frame(91), frame(99)
    session = session_for(current)
    reading = _post_troop_reading(session, current, str(source), POINT, BBOX,
                                  unit_id="dragon")
    assert reading["count"] == 0
    assert reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92
    assert reading["frame"] == str(current)
    assert reading["slot_bbox"] == list(BBOX)
    gated = _probe_count(session, plan_for(source), session.last_snapshot, "dragon", 0,
                         selected=True, pending=1)
    assert gated["count"] == 0
    assert gated["source_identity"]["unit_id"] == "dragon"
    assert gated["source_identity"]["version_verified"] is True
    assert gated["roi"] == [185, 592, 228, 623]
    assert gated["readings"][0]["confidence"] >= .9
    border = gated["current_card_border"]
    assert border["method"] == "18.647.2_dragon_exact_gray_closed_card"
    assert border["source_card_bbox"] == list(BBOX)
    assert border["physical_card_bbox"] == list(BBOX)
    assert border["physical_offset_from_source"] == [0, 0, 0, 0]
    assert border["upper_outline"]["connectivity"] == 8
    assert border["lower_outline"]["connectivity"] == 4


def test_source_selected_transition_and_neighbor_do_not_prove_new_gray_shape():
    for number in (91, 97, 98):
        image = cv2.resize(read_frame(frame(number)), (1280, 720),
                           interpolation=cv2.INTER_AREA)
        assert _closed_dragon_card_border_18_647_2(image, BBOX) is None
    spent = cv2.resize(read_frame(frame(99)), (1280, 720),
                       interpolation=cv2.INTER_AREA)
    assert _closed_dragon_card_border_18_647_2(spent, BBOX) is not None
    for wrong_box in ((140, 595, 229, 711), (142, 595, 231, 711),
                      (230, 595, 319, 711)):
        assert _closed_dragon_card_border_18_647_2(spent, wrong_box) is None


@pytest.mark.parametrize("region", [
    (138, 645, 146, 677), (226, 645, 234, 677),  # opposing straight sides
    (138, 596, 147, 606), (225, 596, 233, 606),  # rounded upper corners
    (140, 594, 149, 599),                         # left top-to-corner join
    (160, 594, 210, 598),                         # continuous top ridge
    (170, 706, 201, 711),                         # connected inner bottom
])
def test_damaged_complete_outline_cannot_prove_zero(region):
    image = cv2.resize(read_frame(frame(99)), (1280, 720),
                       interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = region
    image[y0:y1, x0:x1] = 0
    assert _closed_dragon_card_border_18_647_2(image, BBOX) is None


@pytest.mark.parametrize("contradiction", [1, 2])
def test_wrong_version_and_strong_count_conflicts_remain_hard(contradiction,
                                                               monkeypatch):
    source, current = frame(91), frame(99)
    assert _post_troop_reading(session_for(current, version="unknown"), current,
                               str(source), POINT, BBOX, unit_id="dragon") == {}
    session = session_for(current)
    reader = session.recognizer.provider.recognize_line_image

    def contradictory_raw_tight(image):
        if image.ndim == 3 and image.shape[:2] == (62, 86):
            return [OCRText(f"x{contradiction}", .97)]
        return reader(image)

    monkeypatch.setattr(session.recognizer.provider, "recognize_line_image",
                        contradictory_raw_tight)
    reading = _post_troop_reading(session, current, str(source), POINT, BBOX,
                                  unit_id="dragon")
    assert reading["reason"] == "strong_count_conflict"
    assert reading["counts"] == [0, contradiction]
    with pytest.raises(FlowError, match="strong_count_conflict"):
        _probe_count(session, plan_for(source), session.last_snapshot, "dragon", 0,
                     selected=True, pending=1)
