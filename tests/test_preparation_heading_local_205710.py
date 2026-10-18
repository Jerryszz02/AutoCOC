"""Development-only same-frame countdown heading recovery regressions."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from autococ.battle_vision import (attach_preparation_countdown,
                                   preparation_label_line_recovery_candidate)
from autococ.config import OCRConfig
from autococ.images import read_frame
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


FIXTURE = Path(__file__).parent / "fixtures" / "countdown_heading_18_647_2"
IMAGE = FIXTURE / "scout_frame9_205710.png"
METADATA = FIXTURE / "scout_frame9_205710.provenance.json"
NATIVE_ROI = (1156, 16, 1348, 78)


class LocalLineProvider:
    def __init__(self, reads=None):
        self.reads = ([OCRText("开战倒计时", .99856, NATIVE_ROI)]
                      if reads is None else reads)
        self.calls = []

    def recognize_line(self, path, roi):
        self.calls.append((Path(path), roi))
        return self.reads


def source_metadata():
    return json.loads(METADATA.read_text(encoding="utf-8"))


def snapshot(*, change=None, scene="enemy_village", phase="scout",
             baseline=(1280, 720)):
    ocr = deepcopy(source_metadata()["cached_ocr_heading_and_timer"])
    if change == "duplicate_label":
        ocr.append(dict(ocr[0]))
    elif change == "duplicate_timer":
        ocr.append({"text": "24秒", "confidence": .99, "bbox": [605, 40, 690, 78]})
    elif change == "distant_strong_timer":
        ocr.append({"text": "24秒", "confidence": .99, "bbox": [710, 45, 745, 75]})
    elif change == "shifted_heading":
        ocr[0]["bbox"] = [550, 10, 680, 42]
    elif change == "weak_timer":
        ocr[1]["confidence"] = .89
    elif change == "strong_original_label":
        ocr[0]["text"] = "开战倒计时。"
        ocr[0]["confidence"] = .93513
    elif change == "unrelated_heading":
        ocr[0]["text"] = "战斗结束倒计时↑"
    if baseline == (2560, 1440):
        for item in ocr:
            item["bbox"] = [2 * value for value in item["bbox"]]
    return SceneSnapshot(scene, .92, IMAGE,
                         {"battle": {"phase": phase}, "ocr": ocr})


def recover(frame, provider=None, *, version="18.647.2", baseline=(1280, 720)):
    provider = provider or LocalLineProvider()
    recognizer = ScreenshotRecognizer(OCRConfig(provider="none"), baseline, provider=provider)
    recognizer.client_version = version
    recognizer._recover_preparation_heading(frame, (2560, 1440))
    attach_preparation_countdown(frame, baseline)
    return provider


def test_source_bound_frame9_weak_merged_label_recovers_only_from_exact_same_frame_line():
    frame = snapshot()
    raw = deepcopy(frame.observations["ocr"])
    assert preparation_label_line_recovery_candidate(frame) is not None
    provider = recover(frame)
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert frame.observations["ocr"][:2] == raw
    assert frame.observations["ocr"][2]["source"] == "current_frame_heading_line_roi"
    assert frame.observations["ocr"][2]["frame"] == str(IMAGE)
    assert frame.observations["preparation_label_recovery"]["status"] == "exact_current_frame_heading"
    proof = frame.observations["battle"]["countdown_evidence"]
    assert frame.observations["battle"]["countdown_seconds"] == 25
    assert proof["frame"] == str(IMAGE) and proof["source"] == "current_frame_ocr"
    assert proof["label"]["confidence"] == .99856
    assert proof["timer"]["text"] == "25秒" and proof["timer"]["confidence"] == .99985


@pytest.mark.parametrize("reads", [
    [OCRText("开战倒计时34", .93964, NATIVE_ROI)],
    [OCRText("开战倒计时↑", .99, NATIVE_ROI)],
    [OCRText("开战倒计时", .899, NATIVE_ROI)],
    [OCRText("开战倒计时", .99, (1158, 16, 1360, 78))],
    [OCRText("开战倒计时", .99, NATIVE_ROI), OCRText("开战倒计时", .99, NATIVE_ROI)],
    [],
])
def test_wrong_weak_misaligned_or_ambiguous_local_line_never_grants_time(reads):
    frame = snapshot()
    provider = recover(frame, LocalLineProvider(reads))
    assert len(provider.calls) == 1
    assert len(frame.observations["ocr"]) == 2
    assert "countdown_seconds" not in frame.observations["battle"]
    assert frame.observations["preparation_label_recovery"]["status"] == "unproved"


@pytest.mark.parametrize("change,scene,phase,version", [
    ("duplicate_label", "enemy_village", "scout", "18.647.2"),
    ("duplicate_timer", "enemy_village", "scout", "18.647.2"),
    ("distant_strong_timer", "enemy_village", "scout", "18.647.2"),
    ("shifted_heading", "enemy_village", "scout", "18.647.2"),
    ("weak_timer", "enemy_village", "scout", "18.647.2"),
    ("unrelated_heading", "enemy_village", "scout", "18.647.2"),
    (None, "account_in_use", "scout", "18.647.2"),
    (None, "battle", "active", "18.647.2"),
    (None, "enemy_village", "active", "18.647.2"),
    (None, "enemy_village", "scout", "18.647.1"),
])
def test_ambiguous_scene_geometry_timer_or_version_never_triggers_local_ocr(
        change, scene, phase, version):
    frame = snapshot(change=change, scene=scene, phase=phase)
    provider = recover(frame, version=version)
    assert provider.calls == []
    assert "preparation_label_recovery" not in frame.observations
    assert "countdown_seconds" not in frame.observations["battle"]


def test_existing_strong_original_heading_uses_unchanged_full_ocr_gate():
    frame = snapshot(change="strong_original_label")
    provider = recover(frame)
    assert provider.calls == []
    assert frame.observations["battle"]["countdown_seconds"] == 25
    assert frame.observations["battle"]["countdown_evidence"]["label"]["confidence"] == .93513


def test_nondefault_baseline_maps_the_same_fixed_native_crop():
    frame = snapshot(baseline=(2560, 1440))
    candidate = preparation_label_line_recovery_candidate(frame, (2560, 1440))
    assert candidate is not None and candidate["baseline_roi"] == NATIVE_ROI
    provider = recover(frame, baseline=(2560, 1440))
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert frame.observations["battle"]["countdown_seconds"] == 25


def test_masked_hud_and_cached_ocr_are_bound_to_original_source_when_available():
    metadata = source_metadata()
    image = read_frame(IMAGE)
    x0, y0, x1, y1 = metadata["native_preserved_xyxy"]
    assert hashlib.sha256(IMAGE.read_bytes()).hexdigest() == metadata["output_sha256"]
    assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == metadata["native_preserved_pixel_sha256"]
    masked = image.copy()
    masked[y0:y1, x0:x1] = 0
    assert np.max(masked) == 0
    source, events = Path(metadata["source_frame"]), Path(metadata["events_source"])
    if not source.exists() or not events.exists():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["source_sha256"]
    assert hashlib.sha256(events.read_bytes()).hexdigest() == metadata["events_sha256"]
    assert np.array_equal(image[y0:y1, x0:x1], read_frame(source)[y0:y1, x0:x1])
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    (event,) = [row for row in rows if row.get("kind") == "observation"
                and Path(row.get("frame", "")) == source]
    retained = [{key: item.get(key) for key in ("text", "confidence", "bbox")}
                for item in event["observations"]["ocr"]
                if isinstance(item, dict) and isinstance(item.get("bbox"), list)
                and 575 <= item["bbox"][0] < item["bbox"][2] <= 720
                and item["bbox"][1] < 90]
    assert retained == metadata["cached_ocr_heading_and_timer"]
