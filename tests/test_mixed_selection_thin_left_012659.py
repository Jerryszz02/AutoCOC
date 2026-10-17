"""The 012659 dragon rim is one continuous pixel, not a filled three-pixel band."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.battle_vision import _selected_count_header, selected_card_bbox
from autococ.images import read_frame
from autococ.mixed_probe import ProbeSelectionPending, _probe_count
from autococ.ocr import RapidOCRProvider
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\projects\AutoCOC")
METADATA = ROOT / "tests/fixtures/dragon_border_012659_18_647_2.json"
BOX = (141, 595, 230, 711)
SELECTED = (140, 590, 232, 709)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows():
    return {Path(row["path"]).stem[-5:]: row for row in
            json.loads(METADATA.read_text(encoding="utf8"))["frames"]}


def _fixture(row):
    return ROOT / row["path"]


def test_anonymous_toolbar_pixels_and_cached_event_evidence_are_source_bound():
    metadata = json.loads(METADATA.read_text(encoding="utf8"))
    assert metadata["client_version"] == "18.647.2"
    assert metadata["human_reviewed"] is False
    assert metadata["formal_success"] is False
    assert metadata["counts_toward_acceptance"] is False
    assert [Path(row["path"]).stem[-5:] for row in metadata["frames"]] == [
        "00029", "00034", "00035", "00036"]
    events_path = SOURCE_ROOT / metadata["source_events"]
    events = None
    if events_path.is_file():
        assert _sha(events_path) == metadata["source_events_sha256"]
        events = [json.loads(line) for line in events_path.read_text(encoding="utf8").splitlines()]
    left, top, right, bottom = metadata["retained_native_rectangle_xyxy"]
    for row in metadata["frames"]:
        path = _fixture(row)
        image = read_frame(path)
        assert image.shape == (1440, 2560, 3)
        assert _sha(path) == row["sha256"]
        outside = np.ones(image.shape[:2], dtype=bool)
        outside[top:bottom, left:right] = False
        assert np.all(image[outside] == metadata["outside_pixel_value"])
        source = SOURCE_ROOT / row["source_frame"]
        if source.is_file():
            assert _sha(source) == row["source_sha256"]
            assert np.array_equal(image[top:bottom, left:right],
                                  read_frame(source)[top:bottom, left:right])
        if events is not None:
            (event,) = [item for item in events if item.get("kind") == "observation"
                        and item.get("frame") == str(source)]
            assert event["scene"] == row["cached_scene"]
            observed = event["observations"]
            assert row["cached_ocr"] == [item for item in observed.get("ocr", [])
                                         if item.get("text") == "x1" and
                                         item.get("bbox", [0])[0] < 240]
            assert row["cached_fast_count"] == [
                {key: value for key, value in item.items() if key != "frame"}
                for item in observed.get("deployment_count_reads", [])
                if item.get("slot_bbox") == list(BOX)]


def test_current_dragon_border_is_proven_only_on_second_and_third_fresh_frames():
    rows = _rows()
    assert selected_card_bbox(_fixture(rows["00034"]), BOX) is None
    for number in ("00035", "00036"):
        path = _fixture(rows[number])
        assert selected_card_bbox(path, BOX) == SELECTED
        assert selected_card_bbox(path, (45, 595, 133, 711)) is None
        image = cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)
        header = _selected_count_header(path, image, BOX, client_version="18.647.2",
                                        baseline_resolution=(1280, 720))
        assert header["selected_card_bbox"] == list(SELECTED)
        # The header's full-width white ink still forbids the narrow OCR ROI.
        assert header["header_ink_coverage"]["complete"] is False


@pytest.mark.parametrize("mutation", ["missing_left", "sparse_left", "left_gap",
                                      "missing_right", "missing_top", "dim"])
def test_neighbor_highlights_or_broken_outlines_cannot_prove_selection(tmp_path, mutation):
    image = cv2.resize(read_frame(_fixture(_rows()["00035"])), (1280, 720),
                       interpolation=cv2.INTER_AREA)
    if mutation == "missing_left":
        image[615:701, 136:143] = 45
    elif mutation == "sparse_left":
        image[615:701, 136:143] = 45
        image[615:701:9, 140] = 255
    elif mutation == "left_gap":
        image[635:660, 136:143] = 45
    elif mutation == "missing_right":
        image[615:701, 228:236] = 45
    elif mutation == "missing_top":
        image[585:595, 152:220] = 45
    else:
        image = np.uint8(np.round(image.astype(np.float32) * .55))
    path = tmp_path / f"{mutation}.png"
    assert cv2.imwrite(str(path), image)
    assert selected_card_bbox(path, BOX) is None


@pytest.mark.parametrize("number", ["00034", "00035", "00036"])
def test_full_probe_count_gate_replays_same_frame_count_portrait_source_and_border(number):
    rows = _rows()
    source = _fixture(rows["00029"])
    row = rows[number]
    path = _fixture(row)
    (fast,) = row["cached_fast_count"]
    assert fast["count"] == 1 and fast["confidence"] >= .9
    assert fast["portrait_score"] >= .92
    assert fast["portrait_source"] == "current_frame_saved_battle_portrait"
    recognizer = ScreenshotRecognizer(provider=RapidOCRProvider())
    recognizer.client_version = "18.647.2"
    snapshot = SceneSnapshot(row["cached_scene"], .98, path, {
        "recognition_scope": "mixed_selection_wait",
        "ocr": row["cached_ocr"], "battle": {"slots": []},
        "deployment_count_reads": [{**fast, "frame": str(path)}],
    })
    base = SimpleNamespace(client_version="18.647.2",
                           frame=SimpleNamespace(screenshot_path=source, frame_id=str(source),
                                                 resolution=(1280, 720)),
                           frame_sha256=_sha(source), toolbar_before_frame_id=str(source))
    plan = SimpleNamespace(base=base, dragon_point=(185, 653), dragon_bbox=BOX)
    session = SimpleNamespace(recognizer=recognizer, last_snapshot=snapshot,
                              client_version="18.647.2")
    if number == "00034":
        with pytest.raises(ProbeSelectionPending, match="closed card border"):
            _probe_count(session, plan, snapshot, "dragon", 1, selected=True)
    else:
        proof = _probe_count(session, plan, snapshot, "dragon", 1, selected=True)
        assert proof["count"] == 1
        assert proof["portrait_score"] >= .92
        assert proof["portrait_source"] == "current_frame_saved_battle_portrait"
        assert proof["source_identity"]["unit_id"] == "dragon"
        assert proof["selected_card_border"] == list(SELECTED)
