"""Source-bound 184015 dragon selection replay with cached local line OCR only."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autococ.battle_vision import selected_card_bbox
from autococ.frozen_line import _file_sha256
from autococ.images import read_frame
from autococ.mixed_probe import ProbeEvidenceConflict, _probe_count
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\projects\AutoCOC")
METADATA = ROOT / "tests/fixtures/mixed_dragon_wait_184015_18_647_2.json"
DRAGON = {"point": [234, 653], "bbox": [190, 595, 278, 711]}


def _metadata():
    return json.loads(METADATA.read_text(encoding="utf-8"))


def _rows():
    return {Path(row["source_frame"]).name[:5]: row for row in _metadata()["frames"]}


def _retained(items, rectangles):
    return [item for item in items if item.get("bbox") and
            any(left <= item["bbox"][0] < item["bbox"][2] <= right and
                top <= item["bbox"][1] < item["bbox"][3] <= bottom
                for left, top, right, bottom in rectangles)]


def _reader(calls):
    calls = iter(calls)

    def read(crop):
        call = next(calls)
        assert list(crop.shape) == call["shape"]
        assert hashlib.sha256(crop.tobytes()).hexdigest() == call["pixel_sha256"]
        return [OCRText(item["text"], item["confidence"],
                        tuple(item["bbox"]) if item["bbox"] else None)
                for item in call["results"]]

    return read, calls


def test_mixed_selection_wait_fixtures_are_anonymous_and_source_bound():
    metadata = _metadata()
    assert metadata["client_version"] == "18.647.2"
    assert metadata["human_reviewed"] is False
    assert metadata["formal_success"] is False
    assert metadata["counts_toward_acceptance"] is False
    assert [Path(row["source_frame"]).name[:5] for row in metadata["frames"]] == [
        "00012", "00018", "00019", "00020"]
    report = SOURCE_ROOT / metadata["local_cpu_ocr_replay_report"]
    if report.is_file():
        assert _file_sha256(report) == metadata["local_cpu_ocr_replay_report_sha256"]
    events = SOURCE_ROOT / metadata["source_events"]
    if events.is_file():
        assert _file_sha256(events) == metadata["source_events_sha256"]
    for row in metadata["frames"]:
        path = ROOT / row["path"]
        image = read_frame(path)
        assert image.shape == (1440, 2560, 3)
        assert _file_sha256(path) == row["sha256"]
        retained = np.zeros(image.shape[:2], dtype=bool)
        for left, top, right, bottom in row["retained_native_rectangles_xyxy"]:
            retained[top:bottom, left:right] = True
        assert np.all(image[~retained] == 45)
        assert row["cached_ocr"] == _retained(
            row["cached_ocr"], row["retained_baseline_rectangles_xyxy"])
        source = SOURCE_ROOT / row["source_frame"]
        if source.is_file():
            assert _file_sha256(source) == row["source_sha256"]
            assert np.array_equal(image[retained], read_frame(source)[retained])
        if events.is_file():
            (original,) = [event for line in events.read_text(encoding="utf-8").splitlines()
                           if (event := json.loads(line)).get("kind") == "observation"
                           and Path(event["frame"]) == source]
            observations = original["observations"]
            assert row["cached_ocr"] == _retained(
                observations.get("ocr") or [],
                row["retained_baseline_rectangles_xyxy"])
            assert row["cached_end_battle_buttons"] == _retained(
                [item for item in observations.get("buttons", [])
                 if item.get("name") == "end_battle"],
                row["retained_baseline_rectangles_xyxy"])
            assert row["cached_dragon_slots"] == [
                item for item in (observations.get("battle") or {}).get("slots", [])
                if item.get("kind") == "troop" and item.get("unit_id") == "dragon"]


def test_real_dragon_transition_waits_without_count_then_proves_on_fresh_frame():
    rows = _rows()
    remembered = rows["00018"]
    previous = SceneSnapshot(remembered["scene"], .92, ROOT / remembered["path"], {
        "recognition_scope": "full", "ocr": remembered["cached_ocr"],
        "buttons": remembered["cached_end_battle_buttons"],
        "battle": {"slots": remembered["cached_dragon_slots"]}})
    provider = SimpleNamespace(recognize_line_image=None)
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    recognizer.recognize = lambda _path: pytest.fail("mixed selection must not run full OCR")
    observed = []
    for number in ("00019", "00020"):
        row = rows[number]
        path = ROOT / row["path"]
        provider.recognize_line_image, calls = _reader(row["cached_remembered_count_line_ocr"])
        frame = recognizer.recognize_battle(path, purpose="mixed_selection_wait",
                                            slot=DRAGON, previous=previous)
        assert next(calls, None) is None
        assert frame.scene == "battle"
        assert frame.observations["matched_reasons"] == ["two_current_hud_anchors"]
        assert frame.observations["fast_fallback"] is False
        observed.append(frame)
        previous = frame
    first, second = observed
    assert selected_card_bbox(first.screenshot_path, DRAGON["bbox"]) is None
    assert first.observations["deployment_count_pending"] == {
        "frame": str(first.screenshot_path), "slot_bbox": DRAGON["bbox"],
        "purpose": "mixed_selection_wait", "reason": "current_frame_count_unavailable"}
    assert first.observations["ocr"] == []
    assert "deployment_count_reads" not in first.observations
    assert selected_card_bbox(second.screenshot_path, DRAGON["bbox"]) == (188, 590, 281, 709)
    assert second.observations["deployment_count_reads"][0]["count"] == 1
    assert second.observations["deployment_count_reads"][0]["confidence"] == .98997
    assert "deployment_count_pending" not in second.observations

    source = ROOT / rows["00012"]["path"]
    base = SimpleNamespace(client_version="18.647.2",
                           frame=SimpleNamespace(screenshot_path=source, frame_id=str(source),
                                                 resolution=(1280, 720)),
                           frame_sha256=_file_sha256(source), toolbar_before_frame_id=str(source))
    plan = SimpleNamespace(base=base, dragon_point=(234, 653),
                           dragon_bbox=tuple(DRAGON["bbox"]))
    session = SimpleNamespace(last_snapshot=second, recognizer=recognizer,
                              client_version="18.647.2")
    provider.recognize_line_image, calls = _reader(rows["00020"]["cached_remembered_count_line_ocr"])
    proof = _probe_count(session, plan, second, "dragon", 1, selected=True)
    assert next(calls, None) is None
    assert proof["count"] == 1
    assert proof["portrait_source"] == "current_frame_saved_battle_portrait"
    assert proof["source_identity"]["unit_id"] == "dragon"
    assert proof["selected_card_border"] == [188, 590, 281, 709]

    conflicting = SceneSnapshot(second.scene, second.confidence, second.screenshot_path, {
        **second.observations,
        "deployment_count_reads": [{**second.observations["deployment_count_reads"][0], "count": 2}]})
    session.last_snapshot = conflicting
    provider.recognize_line_image, calls = _reader(rows["00020"]["cached_remembered_count_line_ocr"])
    with pytest.raises(ProbeEvidenceConflict, match="fast count conflicts"):
        _probe_count(session, plan, conflicting, "dragon", 1, selected=True)
    assert next(calls, None) is None
