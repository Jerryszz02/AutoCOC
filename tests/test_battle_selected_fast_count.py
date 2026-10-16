"""Source-bound, local-OCR regression for the selected 18.647.2 troop header."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.battle_vision import BattleObserver, _selected_count_header, selected_card_bbox
from autococ.frozen_line import _file_sha256
from autococ.images import read_frame
from autococ.locator import scale_box
from autococ.ocr import OCRText, RapidOCRProvider
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\projects\AutoCOC")
METADATA = ROOT / "tests/fixtures/mixed_balloon_fast_roi_18_647_2_20261016.json"
BOX = (383, 595, 472, 711)


def _rows():
    return json.loads(METADATA.read_text(encoding="utf-8"))["frames"]


def _retained(items, rectangles):
    return [item for item in items if item.get("bbox") and
            any(left <= item["bbox"][0] < item["bbox"][2] <= right and
                top <= item["bbox"][1] < item["bbox"][3] <= bottom
                for left, top, right, bottom in rectangles)]


def _observer(source_path, provider, *, version="18.647.2", baseline=(1280, 720)):
    recognizer = ScreenshotRecognizer(provider=provider, baseline_resolution=baseline)
    recognizer.client_version = version
    observer = BattleObserver(recognizer)
    image = cv2.resize(read_frame(source_path), (1280, 720), interpolation=cv2.INTER_AREA)
    left, top, right, bottom = BOX
    patch = cv2.cvtColor(image[top + 38:top + 78, left + 20:right - 15],
                         cv2.COLOR_BGR2GRAY)
    point = scale_box((427, 653, 428, 654), from_resolution=(1280, 720),
                      to_resolution=baseline)[:2]
    slot = {"point": list(point),
            "bbox": list(scale_box(BOX, from_resolution=(1280, 720),
                                   to_resolution=baseline))}
    observer.cards[point[0]] = (BOX, patch)
    return observer, slot


def _count(observer, slot, path):
    image = cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)
    return observer.count(path, image, slot)


def test_native_fixtures_keep_only_timer_toolbar_and_source_ocr():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["client_version"] == "18.647.2"
    assert metadata["human_reviewed"] is False
    assert metadata["counts_toward_acceptance"] is False
    assert [Path(row["source_frame"]).name[:5] for row in metadata["frames"]] == [
        "00023", "00024", "00025"]
    for row in metadata["frames"]:
        path = ROOT / row["path"]
        image = read_frame(path)
        assert image.shape == (1440, 2560, 3)
        assert _file_sha256(path) == row["sha256"]
        mask = np.zeros(image.shape[:2], dtype=bool)
        for left, top, right, bottom in row["retained_native_rectangles_xyxy"]:
            mask[top:bottom, left:right] = True
        assert np.all(image[~mask] == 45)
        assert row["cached_ocr"] == _retained(
            row["cached_ocr"], row["retained_baseline_rectangles_xyxy"])
        source = SOURCE_ROOT / row["source_frame"]
        if source.is_file():
            assert _file_sha256(source) == row["source_sha256"]
            assert np.array_equal(image[mask], read_frame(source)[mask])
        events = SOURCE_ROOT / row["source_events"]
        if events.is_file():
            assert _file_sha256(events) == row["source_events_sha256"]
            observations = [event for line in events.read_text(encoding="utf-8").splitlines()
                            if (event := json.loads(line)).get("kind") == "observation"
                            and Path(str(event.get("frame"))) == source]
            assert len(observations) == 1
            original = observations[0]["observations"]
            assert row["cached_ocr"] == _retained(
                original["ocr"], row["retained_baseline_rectangles_xyxy"])
            if "cached_fast_count" in row:
                assert row["cached_fast_count"] == [
                    {key: value for key, value in item.items() if key != "frame"}
                    for item in original["deployment_count_reads"]]


def test_real_local_ocr_reads_selected_count_without_changing_threshold():
    first, second, third = _rows()
    source = ROOT / first["path"]
    provider = RapidOCRProvider()
    for row in (first, second, third):
        path = ROOT / row["path"]
        observer, slot = _observer(source, provider)
        reading = _count(observer, slot, path)
        assert reading["count"] == 1 and reading["confidence"] >= .9
        assert reading["portrait_score"] >= .92
        if row is first:
            assert selected_card_bbox(path, BOX) is None
            assert reading["readings"][0]["source"] == "fast_header_line"
        else:
            assert reading["roi"] == [437, 587, 472, 620]
            assert reading["readings"][0]["source"] == "fast_selected_header_line"
            audit = reading["selected_header_evidence"]
            assert audit["selected_card_bbox"] == [380, 589, 475, 709]
            assert audit["header_ink_coverage"]["complete"] is True
            assert audit["selected_header_roi"] == reading["roi"]
            assert audit["strong_counts"] == [1]
            assert len(audit["count_reads"]) == 4
    old = ROOT / "tests/fixtures/mixed_balloon_selection_00016_18_647_2.png"
    old_source = ROOT / "tests/fixtures/mixed_balloon_selection_00015_18_647_2.png"
    observer, slot = _observer(old_source, provider)
    old_reading = _count(observer, slot, old)
    assert old_reading["count"] == 1 and old_reading["confidence"] >= .9
    assert old_reading["roi"] == [437, 587, 472, 620]


def test_selected_count_roi_obeys_nondefault_baseline_coordinates():
    first, second, _third = _rows()
    provider = RapidOCRProvider()
    observer, slot = _observer(ROOT / first["path"], provider, baseline=(2560, 1440))
    reading = _count(observer, slot, ROOT / second["path"])
    assert reading["count"] == 1
    assert reading["slot_bbox"] == [766, 1190, 944, 1422]
    assert reading["roi"] == [874, 1174, 944, 1240]
    assert reading["selected_header_evidence"]["selected_card_bbox"] == [760, 1178, 950, 1418]


@pytest.mark.parametrize("mutation", ["extra_x10_prefix", "shifted_ink", "missing_side"])
def test_selected_fallback_requires_entire_glyph_and_closed_border(tmp_path, mutation):
    first, second, _third = _rows()
    image = cv2.resize(read_frame(ROOT / second["path"]), (1280, 720),
                       interpolation=cv2.INTER_AREA)
    if mutation == "missing_side":
        image[620:700, 473:478] = 45
    else:
        source = image.copy()
        if mutation == "extra_x10_prefix":
            source_patch = source[592:620, 130:179]
            target = image[592:620, 422:471]
        else:
            source_patch = source[598:615, 137:149]
            target = image[598:615, 426:438]
        hsv = cv2.cvtColor(source_patch, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 95, 255])) > 0
        target[white] = source_patch[white]
    path = tmp_path / f"{mutation}.png"
    cv2.imencode(".png", image)[1].tofile(path)
    evidence = _selected_count_header(path, image, BOX, client_version="18.647.2",
                                      baseline_resolution=(1280, 720))
    if mutation == "missing_side":
        assert selected_card_bbox(path, BOX) is None
        assert evidence is None
    else:
        assert selected_card_bbox(path, BOX) == (380, 589, 475, 709)
        assert evidence["header_ink_coverage"]["complete"] is False
        assert evidence["header_ink_coverage"]["ink_bbox_at_1280x720"][0] < 437
    reads = []

    def reader(crop):
        reads.append(crop.shape[1])
        return [OCRText("x1", .99, (0, 0, 1, 1))] if crop.shape[1] < 50 else []

    observer, slot = _observer(ROOT / first["path"], SimpleNamespace(recognize_line_image=reader))
    assert _count(observer, slot, path) is None
    assert all(width >= 50 for width in reads)  # The risky narrow crop never ran.


def test_other_client_version_does_not_use_selected_crop():
    first, second, _third = _rows()
    reads = []

    def reader(crop):
        reads.append(crop.shape[1])
        return [OCRText("x1", .99, (0, 0, 1, 1))] if crop.shape[1] < 100 else []

    observer, slot = _observer(ROOT / first["path"],
                               SimpleNamespace(recognize_line_image=reader),
                               version="18.647.1")
    assert _count(observer, slot, ROOT / second["path"]) is None
    assert all(width >= 100 for width in reads)
    assert observer.last_count_diagnostic is None


@pytest.mark.parametrize("duplicate_in_wide", [False, True])
def test_selected_wide_and_narrow_strong_counts_conflict(duplicate_in_wide):
    first, second, _third = _rows()

    def reader(crop):
        if crop.shape[1] > 100:
            results = [OCRText("x2", .96, (0, 0, 1, 1))]
            if duplicate_in_wide:
                results.append(OCRText("x1", .95, (0, 0, 1, 1)))
            return results
        return [OCRText("x1", .99, (0, 0, 1, 1))]

    observer, slot = _observer(ROOT / first["path"],
                               SimpleNamespace(recognize_line_image=reader))
    reading = _count(observer, slot, ROOT / second["path"])
    assert reading["reason"] == "strong_count_conflict"
    assert reading["counts"] == [1, 2]
    audit = reading.get("selected_header_evidence") or reading.get("count_diagnostic")
    assert audit["strong_counts"] == [1, 2]
    assert audit["count_reads"]


def test_incomplete_selected_header_keeps_wide_conflict_and_geometry_audit(tmp_path):
    first, second, _third = _rows()
    image = cv2.resize(read_frame(ROOT / second["path"]), (1280, 720),
                       interpolation=cv2.INTER_AREA)
    source_patch = image[592:620, 130:179].copy()  # A real x10 header in this frame.
    target = image[592:620, 422:471]
    hsv = cv2.cvtColor(source_patch, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 95, 255])) > 0
    target[white] = source_patch[white]
    path = tmp_path / "incomplete-strong-conflict.png"
    cv2.imencode(".png", image)[1].tofile(path)
    calls = 0

    def reader(_crop):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [OCRText("x2", .98, (0, 0, 1, 1)),
                    OCRText("noise", .3, (0, 0, 1, 1))]
        return [OCRText("x1", .98, (0, 0, 1, 1))]

    observer, slot = _observer(ROOT / first["path"],
                               SimpleNamespace(recognize_line_image=reader))
    reading = _count(observer, slot, path)
    assert calls == 2  # Coverage rejects the selected narrow crop.
    assert reading["reason"] == "strong_count_conflict"
    assert reading["counts"] == [1, 2]
    audit = reading["count_diagnostic"]
    assert audit["selected_card_bbox"] == [380, 589, 475, 709]
    assert audit["header_ink_coverage"]["complete"] is False
    assert audit["selected_header_roi"] == [437, 587, 472, 620]
    assert [item["method"] for item in audit["count_reads"]] == [
        "fast_header_line", "fast_header_threshold_line"]
