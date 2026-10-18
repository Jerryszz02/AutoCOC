"""Source-bound 18.647.2 dragon post-use gray-card regression."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.frozen_line import (_closed_dragon_card_border_18_647_2,
                                 _post_troop_reading)
from autococ.images import read_frame
from autococ.mixed_probe import _probe_count
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures" / "dragon_post_014520_18_647_2"
BBOX = (141, 595, 230, 711)
POINT = (185, 653)


def frame(number):
    return FIXTURES / f"{number:05d}-toolbar.png"


def session_for(path, *, version="18.647.2"):
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = version
    return SimpleNamespace(recognizer=recognizer, client_version=version,
                           last_snapshot=SimpleNamespace(
                               screenshot_path=path, scene="battle",
                               observations={"ocr": [], "battle": {"slots": []}}))


def plan_for(source):
    return SimpleNamespace(base=SimpleNamespace(
        toolbar_before_frame_id=str(source), client_version="18.647.2",
        frame=SimpleNamespace(screenshot_path=source, frame_id=str(source),
                              resolution=(1280, 720)),
        frame_sha256=sha256(source.read_bytes()).hexdigest()),
        dragon_point=POINT, dragon_bbox=BBOX)


def test_anonymous_toolbar_pixels_are_source_bound():
    meta = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf8"))
    assert meta["client_version"] == "18.647.2"
    assert meta["retained_roi_native"] == [250, 1120, 870, 1440]
    x0, y0, x1, y1 = meta["retained_roi_native"]
    events = Path(meta["events_source"])
    if events.exists():
        assert sha256(events.read_bytes()).hexdigest() == meta["events_sha256"]
    for number in (38, 45, 46):
        path = frame(number)
        row = meta["assets"][path.name]
        pixels = read_frame(path)
        assert sha256(path.read_bytes()).hexdigest() == row["fixture_sha256"]
        assert sha256(pixels[y0:y1, x0:x1].tobytes()).hexdigest() == row["retained_pixel_sha256"]
        masked = pixels.copy()
        masked[y0:y1, x0:x1] = 0
        assert masked.max() == 0
        source = Path(row["source_frame"])
        if source.exists():
            assert sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
            assert np.array_equal(pixels[y0:y1, x0:x1], read_frame(source)[y0:y1, x0:x1])


def test_actual_post_frames_pending_then_prove_zero_with_all_probe_gates():
    source = frame(38)
    plan = plan_for(source)
    for number, expected in ((45, 1), (46, 0)):
        current = frame(number)
        session = session_for(current)
        local = _post_troop_reading(session, current, str(source), POINT, BBOX,
                                    unit_id="dragon")
        assert local["count"] == expected and local["confidence"] >= .9
        assert local["portrait_score"] >= .92
        assert local["frame"] == str(current)
        assert local["readings"][0]["confidence"] >= .9
        gated = _probe_count(session, plan, session.last_snapshot, "dragon", 0,
                             selected=True, pending=1)
        assert gated["count"] == expected
        assert gated["source_identity"]["unit_id"] == "dragon"
        assert gated["source_identity"]["version_verified"] is True
        if expected == 0:
            assert gated["locator"]["source"] == "current_frame_closed_card_header_geometry"
            assert gated["roi"] == [185, 592, 228, 623]
            assert gated["current_card_border"]["method"] == \
                "18.647.2_dragon_low_contrast_closed_card"
        else:
            assert "selected_card_border" in gated


def test_low_contrast_card_requires_complete_gray_outline_and_rejects_active_or_neighbor():
    spent = cv2.resize(read_frame(frame(46)), (1280, 720), interpolation=cv2.INTER_AREA)
    active = cv2.resize(read_frame(frame(45)), (1280, 720), interpolation=cv2.INTER_AREA)
    border = _closed_dragon_card_border_18_647_2(spent, BBOX)
    assert border is not None
    assert border["lower_outline"]["connectivity"] == 4
    assert border["bottom_inner_gradient_min"] >= 100
    assert len(border["upper_outline"]["left_inset"]) == 5
    assert len(border["upper_outline"]["right_inset"]) == 5
    assert border["upper_outline"]["horizontal"]["central_gradient_min"] >= 250
    assert border["upper_outline"]["horizontal"]["connected_columns"] == 81
    assert _closed_dragon_card_border_18_647_2(active, BBOX) is None
    assert _closed_dragon_card_border_18_647_2(spent, (238, 595, 327, 711)) is None
    for region in ((139, 645, 149, 677), (223, 645, 234, 677),
                   (139, 596, 149, 606), (223, 596, 234, 606),
                   (180, 706, 201, 711),
                   (161, 594, 181, 597)):
        damaged = spent.copy()
        x0, y0, x1, y1 = region
        damaged[y0:y1, x0:x1] = 0
        assert _closed_dragon_card_border_18_647_2(damaged, BBOX) is None, region


@pytest.mark.parametrize("damage", [(446, 1250, 468, 1390),
                                    (322, 1188, 362, 1194)])
def test_wrong_unit_version_and_damaged_current_frame_cannot_prove_zero(tmp_path, damage):
    source, current = frame(38), frame(46)
    assert _post_troop_reading(session_for(current), current, str(source), POINT, BBOX,
                               unit_id="balloon") == {}
    assert _post_troop_reading(session_for(current, version="unknown"), current,
                               str(source), POINT, BBOX, unit_id="dragon") == {}
    pixels = read_frame(current)
    x0, y0, x1, y1 = damage
    pixels[y0:y1, x0:x1] = 0
    damaged = tmp_path / "broken-current-card.png"
    success, encoded = cv2.imencode(".png", pixels)
    assert success
    damaged.write_bytes(encoded.tobytes())
    assert _post_troop_reading(session_for(damaged), damaged, str(source), POINT, BBOX,
                               unit_id="dragon") == {}
