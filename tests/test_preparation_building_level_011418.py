"""Source-bound countdown geometry for the 01:14 live preview on client 18.647.2."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autococ.battle_vision import attach_preparation_countdown
from autococ.scene import SceneSnapshot


METADATA = Path(__file__).parent / "fixtures/preparation_building_level_18_647_2_20261017.json"


def _frames():
    return json.loads(METADATA.read_text(encoding="utf-8"))["frames"]


def _snapshot(frame, *, ocr=None, scene="enemy_village", baseline=(1280, 720)):
    items = deepcopy(frame["cached_ocr"] if ocr is None else ocr)
    if baseline == (2560, 1440):
        for item in items:
            item["bbox"] = [coordinate * 2 for coordinate in item["bbox"]]
    return SceneSnapshot(scene, .92, Path(frame["source_frame"]),
                         {"battle": {"phase": "scout"}, "ocr": items,
                          "observed_at_monotonic": frame["observed_at_monotonic"],
                          "frame_id": frame["frame_id"]})


def test_actual_current_frame_timer_and_unrelated_building_level():
    frames = _frames()
    for frame, expected in zip(frames, (29, 25, 22)):
        snapshot = _snapshot(frame)
        attach_preparation_countdown(snapshot)
        battle = snapshot.observations["battle"]
        assert battle["countdown_seconds"] == expected
        assert battle["countdown_evidence"]["frame"] == str(Path(frame["source_frame"]))
        assert battle["countdown_evidence"]["timer"]["text"] == f"{expected}秒"
        assert battle["countdown_evidence"]["source"] == "current_frame_ocr"
    assert [item["text"] for item in frames[1]["cached_ocr"]] == ["开战倒计时", "25秒", "42"]


def test_current_frame_timer_geometry_scales_without_accepting_building_level():
    snapshot = _snapshot(_frames()[1], baseline=(2560, 1440))
    attach_preparation_countdown(snapshot, (2560, 1440))
    assert snapshot.observations["battle"]["countdown_seconds"] == 25


@pytest.mark.parametrize("change", ["nearby_42", "second_nearby_timer", "timer_missing",
                                    "weak_timer", "duplicate_heading", "wrong_scene"])
def test_other_strong_or_missing_current_frame_evidence_never_grants_time(change):
    frame = _frames()[1]
    ocr = deepcopy(frame["cached_ocr"])
    scene = "enemy_village"
    if change == "nearby_42":
        ocr[2]["bbox"] = [610, 70, 620, 80]  # Within the measured 40px band.
    elif change == "second_nearby_timer":
        ocr.append({"text": "24秒", "confidence": .99, "bbox": [600, 60, 681, 90]})
    elif change == "timer_missing":
        ocr.pop(1)
    elif change == "weak_timer":
        ocr[1]["confidence"] = .89
    elif change == "duplicate_heading":
        ocr.append(deepcopy(ocr[0]))
    else:
        scene = "battle"
    snapshot = _snapshot(frame, ocr=ocr, scene=scene)
    attach_preparation_countdown(snapshot)
    assert "countdown_seconds" not in snapshot.observations["battle"]


def test_anonymous_header_pixels_are_exact_and_bound_to_original_when_available():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    root = Path(__file__).parents[1]
    source_root = Path(r"E:\projects\AutoCOC")
    left, top, right, bottom = metadata["native_preserved_xyxy"]
    for frame in metadata["frames"]:
        fixture_path = root / "tests/fixtures" / frame["fixture"]
        assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == frame["fixture_sha256"]
        fixture = cv2.imdecode(np.fromfile(fixture_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert fixture.shape == (1440, 2560, 3)
        assert not np.any(fixture[bottom:])
        assert not np.any(fixture[:bottom, :left])
        assert not np.any(fixture[:bottom, right:])
    events = source_root / metadata["source_events"]
    sources = [source_root / frame["source_frame"] for frame in metadata["frames"]]
    if not events.is_file() or any(not path.is_file() for path in sources):
        pytest.skip("Original local live frames or cached events are unavailable")
    assert hashlib.sha256(events.read_bytes()).hexdigest() == metadata["source_events_sha256"]
    observations = {Path(event["frame"]).name: event["observations"] for line in
                    events.read_text(encoding="utf-8").splitlines()
                    if (event := json.loads(line)).get("kind") == "observation"}
    for frame, source in zip(metadata["frames"], sources):
        assert hashlib.sha256(source.read_bytes()).hexdigest() == frame["source_sha256"]
        original = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        fixture = cv2.imdecode(np.fromfile(root / "tests/fixtures" / frame["fixture"],
                                          dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(fixture[top:bottom, left:right], original[top:bottom, left:right])
        observed = observations[source.name]
        retained = [item for item in observed["ocr"] if isinstance(item.get("bbox"), list)
                    and len(item["bbox"]) == 4 and 575 <= item["bbox"][0] < item["bbox"][2] <= 710
                    and 0 <= item["bbox"][1] < item["bbox"][3] <= 120]
        assert retained == frame["cached_ocr"]
        assert observed["observed_at_monotonic"] == frame["observed_at_monotonic"]
        assert observed["frame_id"] == frame["frame_id"]
