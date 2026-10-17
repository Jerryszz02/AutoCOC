"""Source-bound 18.647.2 dragon post-use gray-card regression."""

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


FIXTURES = Path(__file__).parent / "fixtures" / "dragon_shifted_gray_041151_18_647_2"
BBOX = (142, 595, 230, 711)
POINT = (186, 653)


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


def test_anonymous_toolbar_pixels_are_bound_to_original_frames():
    meta = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf8"))
    assert meta["source_run"] == "20261017-041151-957581-a04a2fd0"
    assert meta["client_version"] == "18.647.2"
    x0, y0, x1, y1 = meta["retained_roi_native"]
    assert [x0, y0, x1, y1] == [250, 1120, 870, 1440]
    events = Path(meta["events_source"])
    if events.exists():
        assert sha256(events.read_bytes()).hexdigest() == meta["events_sha256"]
    for number in (28, 35, 36):
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


def test_original_pending_one_then_current_zero_keeps_source_bbox_and_unique_rim():
    source = frame(28)
    plan = plan_for(source)
    for number, count in ((35, 1), (36, 0)):
        current = frame(number)
        session = session_for(current)
        reading = _post_troop_reading(session, current, str(source), POINT, BBOX,
                                      unit_id="dragon")
        assert reading["count"] == count
        assert reading["confidence"] >= .9
        assert reading["portrait_score"] >= .92
        assert reading["frame"] == str(current)
        assert reading["slot_bbox"] == list(BBOX)
        gated = _probe_count(session, plan, session.last_snapshot, "dragon", 0,
                             selected=True, pending=1)
        assert gated["count"] == count
        assert gated["source_identity"]["unit_id"] == "dragon"
        assert gated["source_identity"]["version_verified"] is True
        if count == 1:
            assert "selected_card_border" in gated
        else:
            assert gated["roi"] == [185, 592, 228, 623]
            assert gated["readings"][0]["confidence"] >= .9
            border = gated["current_card_border"]
            assert border["method"] == "18.647.2_dragon_shifted_gray_closed_card"
            assert border["source_card_bbox"] == list(BBOX)
            assert border["physical_card_bbox"] == [141, 595, 230, 711]
            assert border["physical_offset_from_source"] == [-1, 0, 0, 0]
            assert border["upper_outline"]["central_gradient_min"] >= 250
            assert border["lower_outline"]["bottom_inner_gradient_min"] >= 100


def test_shifted_gray_shape_rejects_active_neighbor_and_more_than_one_pixel_shift():
    spent = cv2.resize(read_frame(frame(36)), (1280, 720), interpolation=cv2.INTER_AREA)
    active = cv2.resize(read_frame(frame(35)), (1280, 720), interpolation=cv2.INTER_AREA)
    assert _closed_dragon_card_border_18_647_2(spent, BBOX) is not None
    assert _closed_dragon_card_border_18_647_2(active, BBOX) is None
    assert _closed_dragon_card_border_18_647_2(spent, (143, 595, 231, 711)) is None
    assert _closed_dragon_card_border_18_647_2(spent, (230, 595, 318, 711)) is None


def test_upper_ridge_must_connect_to_both_corner_traces():
    image = cv2.resize(read_frame(frame(36)), (1280, 720), interpolation=cv2.INTER_AREA)
    assert _closed_dragon_card_border_18_647_2(image, BBOX) is not None
    # This two-pixel cut preserves the earlier individual ridge, join and
    # corner scalar gates; it separates the top ridge from the left corner.
    image[595:597, 146:147] = 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    top_ridge = vertical[595:597, 145:226]
    assert float(np.min(vertical[595, 149:222])) >= 250
    assert float(np.min(np.max(top_ridge, axis=0))) >= 140
    assert float(vertical[597, 144]) >= 140
    assert float(vertical[597, 226]) >= 140
    assert _closed_dragon_card_border_18_647_2(image, BBOX) is None


@pytest.mark.parametrize("region", [
    (139, 645, 146, 677), (226, 645, 233, 677),  # opposite straight sides
    (139, 596, 147, 606), (225, 596, 232, 606),  # rounded upper corners
    (160, 594, 210, 598),                         # continuous top rim
    (170, 706, 201, 711),                         # connected inner bottom
])
def test_each_independent_outline_part_is_required(region):
    image = cv2.resize(read_frame(frame(36)), (1280, 720), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = region
    image[y0:y1, x0:x1] = 0
    assert _closed_dragon_card_border_18_647_2(image, BBOX) is None


@pytest.mark.parametrize("contradiction", [1, 2])
def test_wrong_version_and_strong_conflicting_count_cannot_prove_zero(
        monkeypatch, contradiction):
    source, current = frame(28), frame(36)
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
