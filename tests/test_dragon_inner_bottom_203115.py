"""Development-only current-client dragon lower-card evidence regressions."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autococ.frozen_line import _closed_dragon_card_border_18_647_2
from autococ.images import read_frame


FIXTURES = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2"
CURRENT_BOX = (190, 595, 278, 711)
OLDER_BOX = (141, 595, 230, 711)


def frame(name):
    return cv2.resize(read_frame(FIXTURES / f"{name}.png"), (1280, 720),
                      interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("name,box", [
    ("mixed_203115_post40", CURRENT_BOX),
    ("mixed_192541_post22", CURRENT_BOX),
    ("prior_a_post1", OLDER_BOX), ("prior_b_post1", OLDER_BOX),
    ("new_post1", OLDER_BOX), ("new_post2", OLDER_BOX), ("new_post3", OLDER_BOX),
])
def test_source_bound_gray_cards_have_five_connected_inner_bottom_bands(name, box):
    proof = _closed_dragon_card_border_18_647_2(frame(name), box)
    assert proof is not None
    assert proof["lower_outline"]["connectivity"] == 4
    assert proof["bottom_inner_gradient_min"] >= 40
    assert len(proof["bottom_inner_gradient_bands"]) == 5
    assert min(proof["bottom_inner_gradient_bands"]) >= 40


def test_current_active_source_does_not_look_like_a_spent_gray_card():
    assert _closed_dragon_card_border_18_647_2(frame("mixed_203115_source30"),
                                                 CURRENT_BOX) is None


def test_bottom_evidence_is_invariant_to_pixels_below_card():
    image = frame("mixed_203115_post40")
    assert _closed_dragon_card_border_18_647_2(image, CURRENT_BOX) is not None
    for level in (0, 255):
        changed = image.copy()
        changed[711:720, 202:266] = level
        assert _closed_dragon_card_border_18_647_2(changed, CURRENT_BOX) is not None
    textured = image.copy()
    textured[711:720, 202:266:2] = 0
    textured[711:720, 203:266:2] = 255
    assert _closed_dragon_card_border_18_647_2(textured, CURRENT_BOX) is not None


@pytest.mark.parametrize("segment", range(5))
def test_each_inner_bottom_segment_is_required(segment):
    image = frame("mixed_203115_post40")
    left, _, right, _ = CURRENT_BOX
    x_positions = np.array_split(np.arange(left + 8, right - 8), 5)[segment]
    changed = image.copy()
    changed[706:711, int(x_positions[0]):int(x_positions[-1]) + 1] = 0
    assert _closed_dragon_card_border_18_647_2(changed, CURRENT_BOX) is None


@pytest.mark.parametrize("region", [
    (188, 645, 201, 677), (267, 645, 280, 677),  # left/right straight sides
    (188, 700, 200, 711), (268, 700, 280, 711),  # rounded corners
    (190, 707, 202, 709), (266, 707, 278, 709),  # side-to-bottom junctions
])
def test_broken_side_corner_or_bottom_connection_is_rejected(region):
    image = frame("mixed_203115_post40").copy()
    x0, y0, x1, y1 = region
    image[y0:y1, x0:x1] = 0
    assert _closed_dragon_card_border_18_647_2(image, CURRENT_BOX) is None


def test_neighbor_shift_and_flat_gray_are_rejected():
    image = frame("mixed_203115_post40")
    assert _closed_dragon_card_border_18_647_2(image, (279, 595, 367, 711)) is None
    assert _closed_dragon_card_border_18_647_2(image, (198, 595, 286, 711)) is None
    assert _closed_dragon_card_border_18_647_2(np.zeros_like(image), CURRENT_BOX) is None


@pytest.mark.parametrize("name", ["mixed_203115_source30", "mixed_203115_post40"])
def test_new_masked_toolbar_is_exactly_bound_to_original_frame_and_events(name):
    path = FIXTURES / f"{name}.png"
    provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    image = read_frame(path)
    x0, y0, x1, y1 = provenance["native_preserved_xyxy"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["output_sha256"]
    assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == \
        provenance["native_preserved_pixel_sha256"]
    masked = image.copy()
    masked[y0:y1, x0:x1] = 0
    assert masked.max() == 0
    source = Path(provenance["source_frame"])
    events = Path(provenance["events_source"])
    if not source.exists() or not events.exists():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["source_sha256"]
    assert hashlib.sha256(events.read_bytes()).hexdigest() == provenance["events_sha256"]
    assert np.array_equal(image[y0:y1, x0:x1], read_frame(source)[y0:y1, x0:x1])
