"""Development regression for the 18.647.2 dragon's connected gray-card rim."""

from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import pytest

from autococ.frozen_line import _closed_dragon_card_border_18_647_2
from autococ.images import read_frame


FIXTURES = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2"
CURRENT_BOX = (190, 595, 278, 711)
OLDER_BOX = (141, 595, 230, 711)


def frame(name: str) -> np.ndarray:
    return cv2.resize(read_frame(FIXTURES / f"{name}.png"), (1280, 720),
                      interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("name,box", [
    ("mixed_192541_post22", CURRENT_BOX),
    ("prior_a_post1", OLDER_BOX),
    ("prior_b_post1", OLDER_BOX),
    ("new_post1", OLDER_BOX),
])
def test_real_gray_dragon_has_connected_lower_rim(name, box):
    evidence = _closed_dragon_card_border_18_647_2(frame(name), box)
    assert evidence is not None
    assert evidence["lower_outline"]["connectivity"] == 4
    assert evidence["lower_outline"]["gradient_min"] == 40


def test_current_source_is_not_a_spent_gray_card():
    assert _closed_dragon_card_border_18_647_2(frame("mixed_192541_source"), CURRENT_BOX) is None


def test_neighbor_and_invalid_geometry_do_not_match():
    current = frame("mixed_192541_post22")
    for box in ((279, 595, 367, 711), (198, 595, 286, 711),
                (190, 595, 277, 711), (190, 595, 278, 720),
                (190, 580, 278, 711)):
        assert _closed_dragon_card_border_18_647_2(current, box) is None


@pytest.mark.parametrize("part,region", [
    ("left side", (188, 645, 201, 677)),
    ("right side", (267, 645, 280, 677)),
    ("left corner", (188, 700, 200, 711)),
    ("right corner", (268, 700, 280, 711)),
    ("bottom", (222, 706, 247, 715)),
    ("left junction", (190, 706, 199, 708)),
    ("right junction", (269, 706, 278, 708)),
])
def test_damaged_continuous_rim_is_rejected(part, region):
    current = frame("mixed_192541_post22")
    x0, y0, x1, y1 = region
    current[y0:y1, x0:x1] = 0
    assert _closed_dragon_card_border_18_647_2(current, CURRENT_BOX) is None, part


def test_flat_card_is_rejected():
    assert _closed_dragon_card_border_18_647_2(np.zeros_like(frame("mixed_192541_post22")),
                                                 CURRENT_BOX) is None


@pytest.mark.parametrize("name", ["mixed_192541_source", "mixed_192541_post22"])
def test_new_anonymous_toolbar_preserves_only_source_pixels(name):
    from autococ.images import read_frame

    path = FIXTURES / f"{name}.png"
    metadata = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    image = read_frame(path)
    x0, y0, x1, y1 = metadata["native_preserved_xyxy"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["output_sha256"]
    assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == \
        metadata["native_preserved_pixel_sha256"]
    without_toolbar = image.copy()
    without_toolbar[y0:y1, x0:x1] = 0
    assert without_toolbar.max() == 0
    source = Path(metadata["source_frame"])
    events = Path(metadata["events_source"])
    if not source.exists() or not events.exists():
        pytest.skip("Original local live evidence is unavailable")
    original = read_frame(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["source_sha256"]
    assert hashlib.sha256(events.read_bytes()).hexdigest() == metadata["events_sha256"]
    assert np.array_equal(image[y0:y1, x0:x1], original[y0:y1, x0:x1])
