"""Development-only 18.647.2 dragon upper-corner evidence regressions."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autococ.frozen_line import _closed_dragon_card_border_18_647_2
from autococ.images import read_frame
from autococ.unit_catalog import recognize_card_identity


FIXTURES = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2"
CURRENT_BOX = (190, 595, 278, 711)
OLDER_BOX = (141, 595, 230, 711)


def frame(name: str) -> np.ndarray:
    return cv2.resize(read_frame(FIXTURES / f"{name}.png"), (1280, 720),
                      interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("name,box", [
    ("mixed_212333_post22", CURRENT_BOX),
    ("mixed_203115_post40", CURRENT_BOX),
    ("mixed_192541_post22", CURRENT_BOX),
    ("prior_a_post1", OLDER_BOX), ("prior_b_post1", OLDER_BOX),
    ("new_post1", OLDER_BOX), ("new_post2", OLDER_BOX), ("new_post3", OLDER_BOX),
])
def test_independent_spent_cards_have_two_connected_inward_upper_corner_traces(name, box):
    proof = _closed_dragon_card_border_18_647_2(frame(name), box)
    assert proof is not None
    upper = proof["upper_outline"]
    assert upper["gradient_min"] == 140
    assert upper["side_join_min"] == 100
    for side in ("left_inset", "right_inset"):
        trace = upper[side]
        assert len(trace) == 5
        assert trace[0] >= 1 and trace[1] >= 1 and trace[-1] == 0
        assert all(0 <= first - second <= 1 for first, second in zip(trace, trace[1:]))


def test_new_real_card_failed_the_old_single_upper_corner_row():
    image = frame("mixed_212333_post22")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    side = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    left, top, right, _ = CURRENT_BOX
    left_edge = np.median(side[625:695, left])
    right_edge = np.median(side[625:695, right - 1])
    assert side[top + 5, left] > .7 * left_edge
    assert side[top + 5, right - 1] < .75 * right_edge
    assert _closed_dragon_card_border_18_647_2(image, CURRENT_BOX) is not None


@pytest.mark.parametrize("side,region", [
    ("left upper corner", (188, 597, 197, 603)),
    ("right upper corner", (271, 597, 280, 603)),
    ("left upper-to-side junction", (188, 600, 197, 605)),
    ("right upper-to-side junction", (271, 600, 280, 605)),
])
def test_damaged_upper_corner_or_connection_is_rejected(side, region):
    image = frame("mixed_212333_post22").copy()
    x0, y0, x1, y1 = region
    image[y0:y1, x0:x1] = 0
    assert _closed_dragon_card_border_18_647_2(image, CURRENT_BOX) is None, side


def test_rectangular_upper_edge_and_isolated_dark_pixel_are_rejected():
    image = frame("mixed_212333_post22")
    rectangular = image.copy()
    # A square cap has straight left/right sides but no inset-to-side arc.
    rectangular[595:604, 190:278] = 120
    assert _closed_dragon_card_border_18_647_2(rectangular, CURRENT_BOX) is None
    isolated = image.copy()
    isolated[595:604, 188:197] = 0
    isolated[598, 192] = 255
    assert _closed_dragon_card_border_18_647_2(isolated, CURRENT_BOX) is None


def test_per_row_strong_pixels_with_a_broken_upper_corner_step_are_rejected(monkeypatch):
    image = frame("mixed_212333_post22")
    original_sobel = cv2.Sobel

    def broken_left_corner(*args, **kwargs):
        side = original_sobel(*args, **kwargs)
        if args[2:4] == (1, 0):
            side[598:603, 190:194] = 0
            # Each row still has a strong signed edge, but the first two rows
            # jump two columns and cannot form even an eight-neighbor path.
            for y, inset in zip(range(598, 603), (3, 1, 1, 0, 0)):
                side[y, 190 + inset] = 200
        return side

    monkeypatch.setattr(cv2, "Sobel", broken_left_corner)
    assert _closed_dragon_card_border_18_647_2(image, CURRENT_BOX) is None


@pytest.mark.parametrize("side,region", [
    ("left", (188, 610, 194, 612)),
    ("right", (274, 610, 280, 612)),
])
def test_upper_corner_cannot_disconnect_before_the_straight_side_bands(side, region):
    image = frame("mixed_212333_post22").copy()
    x0, y0, x1, y1 = region
    image[y0:y1, x0:x1] = 0
    assert _closed_dragon_card_border_18_647_2(image, CURRENT_BOX) is None, side


def test_active_card_neighbor_shift_and_flat_image_are_rejected():
    assert _closed_dragon_card_border_18_647_2(frame("mixed_212333_source14"),
                                                 CURRENT_BOX) is None
    image = frame("mixed_212333_post22")
    for box in ((279, 595, 367, 711), (198, 595, 286, 711),
                (190, 595, 277, 711)):
        assert _closed_dragon_card_border_18_647_2(image, box) is None
    assert _closed_dragon_card_border_18_647_2(np.zeros_like(image), CURRENT_BOX) is None


def test_source_card_still_has_versioned_dragon_identity():
    identity = recognize_card_identity(frame("mixed_212333_source14"), CURRENT_BOX,
                                       "troop", surface="battle", client_version="18.647.2")
    assert identity["unit_id"] == "dragon"
    assert identity["version_verified"] is True
    assert identity["template_client_version"] == "18.647.2"


@pytest.mark.parametrize("name", ["mixed_212333_source14", "mixed_212333_post22"])
def test_new_anonymous_toolbar_matches_its_original_capture(name):
    path = FIXTURES / f"{name}.png"
    metadata = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    image = read_frame(path)
    x0, y0, x1, y1 = metadata["native_preserved_xyxy"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["output_sha256"]
    assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == \
        metadata["native_preserved_pixel_sha256"]
    masked = image.copy()
    masked[y0:y1, x0:x1] = 0
    assert masked.max() == 0
    source = Path(metadata["source_frame"])
    events = Path(metadata["events_source"])
    if not source.exists() or not events.exists():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["source_sha256"]
    assert hashlib.sha256(events.read_bytes()).hexdigest() == metadata["events_sha256"]
    assert np.array_equal(image[y0:y1, x0:x1], read_frame(source)[y0:y1, x0:x1])
