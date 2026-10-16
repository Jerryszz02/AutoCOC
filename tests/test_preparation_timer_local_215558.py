"""Development-only current-frame timer line recovery for client 18.647.2."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autococ.battle_vision import (attach_preparation_countdown,
                                   preparation_timer_line_recovery_candidate)
from autococ.config import OCRConfig
from autococ.images import read_frame
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


FIXTURE = Path(__file__).parent / "fixtures" / "preparation_countdown_215558"
IMAGE = FIXTURE / "frame09_timer_hud.png"
METADATA = FIXTURE / "frame09_timer_hud.provenance.json"
NATIVE_ROI = (1198, 60, 1380, 178)


class LocalLineProvider:
    def __init__(self, reads=None):
        self.reads = ([OCRText("24秒", .99983, NATIVE_ROI)] if reads is None else reads)
        self.calls = []

    def recognize_line(self, path, roi):
        self.calls.append((Path(path), roi))
        return self.reads


def metadata():
    return json.loads(METADATA.read_text(encoding="utf-8"))


def snapshot(*, change=None, scene="enemy_village", phase="scout", baseline=(1280, 720)):
    source = metadata()
    ocr = deepcopy(source["cached_ocr"])
    battle = {"phase": phase}
    if change == "duplicate_heading":
        ocr.append(dict(ocr[0]))
    elif change == "duplicate_damaged_timer":
        ocr.append(dict(ocr[1]))
    elif change == "conflicting_strong_timer":
        ocr.append({"text": "23秒", "confidence": .99, "bbox": [710, 45, 745, 75]})
    elif change == "weak_heading":
        ocr[0]["confidence"] = .89
    elif change == "weak_damaged_timer":
        ocr[1]["confidence"] = .89
    elif change == "shifted_heading":
        ocr[0]["bbox"] = [550, 12, 680, 40]
    elif change == "shifted_timer":
        ocr[1]["bbox"] = [700, 32, 791, 87]
    elif change == "punctuated_heading":
        ocr[0]["text"] = "开战倒计时。"
    elif change == "invalid_number":
        ocr[1]["text"] = ".124秒"
    elif change == "stale_evidence":
        battle.update(countdown_seconds=24, countdown_evidence={"frame": "old.png"})
    if baseline == (2560, 1440):
        for item in ocr:
            item["bbox"] = [2 * value for value in item["bbox"]]
    return SceneSnapshot(scene, .92, IMAGE, {
        "battle": battle, "ocr": ocr, **source["capture_metadata"]})


def recover(frame, provider=None, *, version="18.647.2", baseline=(1280, 720)):
    provider = provider or LocalLineProvider()
    recognizer = ScreenshotRecognizer(OCRConfig(provider="none"), baseline, provider=provider)
    recognizer.client_version = version
    recognizer._recover_preparation_timer(frame, (2560, 1440))
    attach_preparation_countdown(frame, baseline)
    return provider


def test_actual_frame9_damaged_full_ocr_requires_new_exact_same_frame_line():
    frame = snapshot()
    original_ocr = deepcopy(frame.observations["ocr"])
    original_time = frame.observations["observed_at_monotonic"]
    attach_preparation_countdown(frame)
    assert "countdown_seconds" not in frame.observations["battle"]
    candidate = preparation_timer_line_recovery_candidate(frame)
    assert candidate is not None and candidate["expected_full_text"] == "24秒"
    assert candidate["baseline_roi"] == (599, 30, 690, 89)
    provider = recover(frame)
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert frame.observations["ocr"][:2] == original_ocr
    assert frame.observations["ocr"][2]["source"] == "current_frame_timer_line_roi"
    assert frame.observations["ocr"][2]["frame"] == str(IMAGE)
    assert frame.observations["preparation_timer_recovery"]["status"] == "exact_current_frame_timer"
    assert frame.observations["observed_at_monotonic"] == original_time
    assert frame.observations["battle"]["countdown_seconds"] == 24
    proof = frame.observations["battle"]["countdown_evidence"]
    assert proof["source"] == "current_frame_ocr" and proof["frame"] == str(IMAGE)
    assert proof["timer"]["text"] == "24秒" and proof["timer"]["confidence"] == .99983


def test_unproved_null_countdown_fields_from_battle_observation_allow_recovery():
    frame = snapshot()
    frame.observations["battle"].update(countdown_seconds=None, countdown_evidence=None)
    candidate = preparation_timer_line_recovery_candidate(frame)
    assert candidate is not None and candidate["expected_full_text"] == "24秒"


@pytest.mark.parametrize("old_seconds,old_evidence", [
    (0, None), (24, None), (None, {}), (None, ""),
    (None, {"frame": "old.png"}),
])
def test_nonnull_prior_countdown_fields_cannot_be_replaced(old_seconds, old_evidence):
    frame = snapshot()
    frame.observations["battle"].update(countdown_seconds=old_seconds,
                                         countdown_evidence=old_evidence)
    provider = LocalLineProvider()
    assert preparation_timer_line_recovery_candidate(frame) is None
    recognizer = ScreenshotRecognizer(OCRConfig(provider="none"), (1280, 720), provider=provider)
    recognizer.client_version = "18.647.2"
    recognizer._recover_preparation_timer(frame, (2560, 1440))
    assert provider.calls == []
    assert frame.observations["battle"]["countdown_seconds"] is old_seconds
    assert frame.observations["battle"]["countdown_evidence"] is old_evidence


def test_full_recognize_uses_actual_cached_frame9_ocr_before_countdown_attachment():
    source = metadata()
    original, events = Path(source["source_frame"]), Path(source["events_source"])
    if not original.exists() or not events.exists():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source["source_sha256"]
    assert hashlib.sha256(events.read_bytes()).hexdigest() == source["events_sha256"]
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    (event,) = [row for row in rows if row.get("kind") == "observation"
                and row.get("frame") == str(original)]
    raw_ocr = event["observations"]["ocr"]
    assert [item for item in raw_ocr if item in source["cached_ocr"]] == source["cached_ocr"]

    class CachedFrameProvider(LocalLineProvider):
        def recognize(self, path, roi=None):
            assert Path(path) == original and roi is None
            return [OCRText(item["text"], item["confidence"],
                            tuple(2 * value for value in item["bbox"]))
                    for item in raw_ocr]

        def recognize_line(self, path, roi):
            self.calls.append((Path(path), roi))
            return self.reads if roi == NATIVE_ROI else []

    provider = CachedFrameProvider()
    recognizer = ScreenshotRecognizer(OCRConfig(provider="none"), (1280, 720), provider=provider)
    recognizer.client_version = "18.647.2"
    frame = recognizer.recognize(original)
    assert frame.scene == "enemy_village"
    assert provider.calls.count((original, NATIVE_ROI)) == 1
    assert frame.observations["preparation_timer_recovery"]["status"] == "exact_current_frame_timer"
    assert frame.observations["battle"]["countdown_seconds"] == 24
    assert frame.observations["battle"]["countdown_evidence"]["frame"] == str(original)
    assert frame.observations["ocr"][-1]["source"] == "current_frame_timer_line_roi"


@pytest.mark.parametrize("reads", [
    [OCRText("4秒", .99997, NATIVE_ROI)],  # Reject a possibly clipped first digit.
    [OCRText("23秒", .99997, NATIVE_ROI)],
    [OCRText(".24秒", .99997, NATIVE_ROI)],
    [OCRText("24秒", .899, NATIVE_ROI)],
    [OCRText("24秒", .99, (1204, 66, 1368, 170))],
    [OCRText("24秒", .99, NATIVE_ROI), OCRText("24秒", .99, NATIVE_ROI)],
    [],
])
def test_weak_partial_conflicting_or_ambiguous_local_line_never_grants_time(reads):
    frame = snapshot()
    provider = recover(frame, LocalLineProvider(reads))
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert len(frame.observations["ocr"]) == 2
    assert frame.observations["preparation_timer_recovery"]["status"] == "unproved"
    assert "countdown_seconds" not in frame.observations["battle"]


@pytest.mark.parametrize("change,scene,phase,version", [
    ("duplicate_heading", "enemy_village", "scout", "18.647.2"),
    ("duplicate_damaged_timer", "enemy_village", "scout", "18.647.2"),
    ("conflicting_strong_timer", "enemy_village", "scout", "18.647.2"),
    ("weak_heading", "enemy_village", "scout", "18.647.2"),
    ("weak_damaged_timer", "enemy_village", "scout", "18.647.2"),
    ("shifted_heading", "enemy_village", "scout", "18.647.2"),
    ("shifted_timer", "enemy_village", "scout", "18.647.2"),
    ("punctuated_heading", "enemy_village", "scout", "18.647.2"),
    ("invalid_number", "enemy_village", "scout", "18.647.2"),
    ("stale_evidence", "enemy_village", "scout", "18.647.2"),
    (None, "account_in_use", "scout", "18.647.2"),
    (None, "enemy_village", "active", "18.647.2"),
    (None, "enemy_village", "scout", "18.647.1"),
])
def test_incomplete_ambiguous_stale_or_wrong_version_input_never_triggers_local_ocr(
        change, scene, phase, version):
    frame = snapshot(change=change, scene=scene, phase=phase)
    provider = recover(frame, version=version)
    assert provider.calls == []
    assert "preparation_timer_recovery" not in frame.observations
    assert "countdown_seconds" not in frame.observations["battle"]


def test_nondefault_baseline_scales_the_fixed_current_frame_roi():
    frame = snapshot(baseline=(2560, 1440))
    candidate = preparation_timer_line_recovery_candidate(frame, (2560, 1440))
    assert candidate is not None and candidate["baseline_roi"] == NATIVE_ROI
    provider = recover(frame, baseline=(2560, 1440))
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert frame.observations["battle"]["countdown_seconds"] == 24


def test_hud_pixels_and_cached_ocr_are_exactly_bound_to_the_original_frame():
    source = metadata()
    image = read_frame(IMAGE)
    x0, y0, x1, y1 = source["native_preserved_xyxy"]
    assert hashlib.sha256(IMAGE.read_bytes()).hexdigest() == source["output_sha256"]
    assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == \
        source["native_preserved_pixel_sha256"]
    masked = image.copy()
    masked[y0:y1, x0:x1] = 0
    assert masked.max() == 0
    # In the actual frame's numeral band, the fixed ROI begins before the
    # first white digit foreground at x606; it did not clip the first numeral.
    baseline = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    white = baseline[42:82, 590:620].min(axis=2) > 190
    assert not white[:, :9].any() and white[:, 16:].any()
    original, events = Path(source["source_frame"]), Path(source["events_source"])
    if not original.exists() or not events.exists():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source["source_sha256"]
    assert hashlib.sha256(events.read_bytes()).hexdigest() == source["events_sha256"]
    assert np.array_equal(image[y0:y1, x0:x1], read_frame(original)[y0:y1, x0:x1])
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    (event,) = [row for row in rows if row.get("kind") == "observation"
                and row.get("frame") == str(original)]
    retained = [item for item in event["observations"]["ocr"]
                if isinstance(item, dict) and isinstance(item.get("bbox"), list)
                and len(item["bbox"]) == 4 and
                560 <= item["bbox"][0] < item["bbox"][2] <= 710 and
                0 <= item["bbox"][1] < item["bbox"][3] <= 95]
    assert retained == source["cached_ocr"]
    assert {key: event["observations"][key] for key in source["capture_metadata"]} == \
        source["capture_metadata"]
