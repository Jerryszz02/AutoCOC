"""Source-bound selected dragon outline from the failed 043910 mixed run."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.battle_vision import selected_card_bbox
from autococ.errors import FlowError
from autococ.frozen_line import _selected_dragon_thin_card_18_647_2
from autococ.images import read_frame
from autococ.mixed_probe import ProbeEvidenceConflict, _probe_count
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures" / "mixed_dragon_selection_043910_18_647_2"
BBOX = (134, 595, 230, 711)
POINT = (182, 653)


def frame(number: int) -> Path:
    return FIXTURES / f"{number:05d}-toolbar.png"


def session_for(current: Path, *, version: str = "18.647.2",
                fast_count: int = 1) -> SimpleNamespace:
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = version
    reading = {"count": fast_count, "confidence": .95142,
               "frame": str(current), "slot_bbox": list(BBOX),
               "readings": [{"text": f"x{fast_count}", "confidence": .95142,
                             "bbox": [132, 585, 233, 620], "source": "fast_header_line"}],
               "portrait_score": .97207, "portrait_source": "current_frame_saved_battle_portrait"}
    snapshot = SceneSnapshot("battle", .9989, current, {
        "recognition_scope": "mixed_selection_wait", "fast_fallback": False,
        "matched_reasons": ["two_current_hud_anchors"],
        "deployment_count_reads": [reading], "ocr": [], "battle": {"slots": []}})
    return SimpleNamespace(recognizer=recognizer, client_version=version,
                           last_snapshot=snapshot)


def plan_for(source: Path, *, version: str = "18.647.2") -> SimpleNamespace:
    base = SimpleNamespace(toolbar_before_frame_id=str(source), client_version=version,
                           frame=SimpleNamespace(screenshot_path=source,
                                                 frame_id=str(source), resolution=(1280, 720)),
                           frame_sha256=sha256(source.read_bytes()).hexdigest())
    return SimpleNamespace(base=base, dragon_point=POINT, dragon_bbox=BBOX)


def test_anonymous_toolbar_pixels_are_bound_to_failed_source():
    meta = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf8"))
    assert meta["source_run"] == "20261017-043910-242928-a79cf8b6"
    assert meta["client_version"] == "18.647.2"
    x0, y0, x1, y1 = meta["retained_roi_native"]
    assert [x0, y0, x1, y1] == [240, 1120, 870, 1440]
    events = Path(meta["events_source"])
    if events.exists():
        assert sha256(events.read_bytes()).hexdigest() == meta["events_sha256"]
    for number in (18, 23, 24, 25):
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


def test_first_selected_observation_has_no_complete_border():
    assert selected_card_bbox(frame(23), BBOX, (1280, 720)) is None


@pytest.mark.parametrize("number", [24, 25])
def test_later_selected_frames_prove_fresh_count_identity_and_closed_border(number):
    source, current = frame(18), frame(number)
    # The existing generic outline detector misses the thin selected rim.
    assert selected_card_bbox(current, BBOX, (1280, 720)) is None
    session = session_for(current)
    reading = _probe_count(session, plan_for(source), session.last_snapshot,
                           "dragon", 1, selected=True)
    assert reading["count"] == 1 and reading["confidence"] >= .9
    assert reading["roi"] == [132, 585, 233, 620]
    assert reading["portrait_score"] >= .92
    assert reading["source_identity"]["unit_id"] == "dragon"
    assert reading["source_identity"]["version_verified"] is True
    assert reading["selected_card_border"] == [139, 589, 233, 710]
    assert reading["selected_card_border_evidence"]["method"] == \
        "18.647.2_dragon_thin_selected_closed_card"
    assert reading["selected_card_border_evidence"]["source_card_bbox"] == list(BBOX)
    assert reading["selected_card_border_evidence"]["physical_offset_from_source"] == \
        [5, -6, 3, -1]


def test_thin_outline_is_unique_to_selected_current_card_and_bounded_geometry():
    source = cv2.resize(read_frame(frame(18)), (1280, 720), interpolation=cv2.INTER_AREA)
    pending = cv2.resize(read_frame(frame(23)), (1280, 720), interpolation=cv2.INTER_AREA)
    selected = cv2.resize(read_frame(frame(25)), (1280, 720), interpolation=cv2.INTER_AREA)
    assert _selected_dragon_thin_card_18_647_2(source, BBOX) is None
    assert _selected_dragon_thin_card_18_647_2(pending, BBOX) is None
    assert _selected_dragon_thin_card_18_647_2(selected, (335, 595, 431, 711)) is None
    assert _selected_dragon_thin_card_18_647_2(selected, (125, 595, 221, 711)) is None
    assert _selected_dragon_thin_card_18_647_2(selected, (141, 595, 230, 711)) is None
    old_selected = (Path(__file__).parent / "fixtures" / "dragon_post_014520_18_647_2" /
                    "00045-toolbar.png")
    assert selected_card_bbox(old_selected, (141, 595, 230, 711), (1280, 720)) is not None


@pytest.mark.parametrize("region", [
    (137, 645, 142, 677), (229, 645, 235, 677),  # both vertical sides
    (160, 587, 212, 595), (160, 704, 212, 711),  # top and bottom
    (136, 587, 147, 601), (226, 587, 236, 601),  # upper corners
    (136, 698, 148, 712), (226, 698, 236, 712),  # lower corners
])
def test_thin_selected_outline_rejects_a_broken_side_or_corner(region):
    image = cv2.resize(read_frame(frame(25)), (1280, 720), interpolation=cv2.INTER_AREA)
    assert _selected_dragon_thin_card_18_647_2(image, BBOX) is not None
    x0, y0, x1, y1 = region
    image[y0:y1, x0:x1] = 0
    assert _selected_dragon_thin_card_18_647_2(image, BBOX) is None


def test_strong_fast_count_conflict_wrong_version_and_changed_source_stay_hard(tmp_path):
    source, current = frame(18), frame(25)
    with pytest.raises(ProbeEvidenceConflict, match="fast count conflicts"):
        session = session_for(current, fast_count=2)
        _probe_count(session, plan_for(source), session.last_snapshot,
                     "dragon", 1, selected=True)
    with pytest.raises(FlowError):
        session = session_for(current, version="unknown")
        _probe_count(session, plan_for(source), session.last_snapshot,
                     "dragon", 1, selected=True)
    copied_source = tmp_path / "source.png"
    copied_source.write_bytes(source.read_bytes())
    plan = plan_for(copied_source)
    changed = read_frame(copied_source)
    changed[0, 0] = [255, 255, 255]
    ok, encoded = cv2.imencode(".png", changed)
    assert ok
    copied_source.write_bytes(encoded.tobytes())
    with pytest.raises(FlowError, match="source card pixels changed"):
        session = session_for(current)
        _probe_count(session, plan, session.last_snapshot, "dragon", 1, selected=True)
