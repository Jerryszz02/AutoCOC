"""A level glyph merged into the current countdown heading never supplies seconds."""

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
from autococ.prepared_battle import _remaining_seconds
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


FIXTURE = Path(__file__).parent / "fixtures" / "countdown_heading_level_18_647_2"
IMAGE = FIXTURE / "frame16_hud.png"
METADATA = FIXTURE / "provenance.json"
NATIVE_ROI = (1176, 20, 1380, 82)


class LocalLineProvider:
    def __init__(self, reads=None):
        self.reads = ([OCRText("开战倒计时：", .99977, NATIVE_ROI)]
                      if reads is None else reads)
        self.calls = []

    def recognize_line(self, path, roi):
        self.calls.append((Path(path), roi))
        return self.reads


def snapshot(*, changed=None, scene="enemy_village", phase="scout",
             baseline=(1280, 720)):
    ocr = deepcopy(json.loads(METADATA.read_text(encoding="utf-8"))[
        "recorded_full_ocr_heading_and_timer"])
    if changed == "duplicate_heading":
        ocr.append(dict(ocr[0]))
    elif changed == "extra_strong_number":
        ocr.append({"text": "11秒", "confidence": .99, "bbox": [640, 45, 675, 75]})
    elif changed == "weak_heading":
        ocr[0]["confidence"] = .89
    elif changed == "weak_timer":
        ocr[1]["confidence"] = .89
    elif changed == "wrong_prefix":
        ocr[0]["text"] = "200开战倒计时"
    elif changed == "wrong_heading":
        ocr[0]["text"] = "20开战倒计秒"
    elif changed == "shifted_heading":
        ocr[0]["bbox"] = [550, 12, 680, 39]
    elif changed == "shifted_timer":
        ocr[1]["bbox"] = [602, 55, 678, 95]
    if baseline == (2560, 1440):
        for item in ocr:
            item["bbox"] = [2 * value for value in item["bbox"]]
    return SceneSnapshot(scene, .92, IMAGE, {"battle": {"phase": phase}, "ocr": ocr})


def recover(frame, provider=None, *, version="18.647.2", baseline=(1280, 720)):
    provider = provider or LocalLineProvider()
    recognizer = ScreenshotRecognizer(OCRConfig(provider="none"), baseline, provider=provider)
    recognizer.client_version = version
    recognizer._recover_preparation_heading(frame, (2560, 1440))
    attach_preparation_countdown(frame, baseline)
    return provider


def test_source_bound_current_heading_and_original_timer_prove_12_seconds():
    frame = snapshot()
    original = deepcopy(frame.observations["ocr"])
    candidate = preparation_label_line_recovery_candidate(frame)
    assert candidate is not None
    assert candidate["reason"] == "one_level_merged_heading_and_one_strong_timer"
    provider = recover(frame)
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert frame.observations["ocr"][:2] == original
    corrected = frame.observations["ocr"][2]
    assert corrected["text"] == "开战倒计时"
    assert corrected["source"] == "current_frame_heading_line_roi"
    assert corrected["frame"] == str(IMAGE)
    assert frame.observations["preparation_label_recovery"]["readings"][0]["text"] == "开战倒计时："
    proof = frame.observations["battle"]["countdown_evidence"]
    assert _remaining_seconds(frame) == 12
    assert proof["timer"] == original[1]
    assert proof["label"] == {key: corrected[key] for key in ("text", "bbox", "confidence")}


@pytest.mark.parametrize("text", ["开战倒计时", "开战倒计时:"])
def test_exact_title_with_no_colon_or_one_ascii_colon_is_equally_source_bound(text):
    frame = snapshot()
    provider = recover(frame, LocalLineProvider([OCRText(text, .999, NATIVE_ROI)]))
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert _remaining_seconds(frame) == 12
    assert frame.observations["ocr"][2]["text"] == "开战倒计时"


@pytest.mark.parametrize("changed", [
    "duplicate_heading", "extra_strong_number", "weak_heading", "weak_timer",
    "wrong_prefix", "wrong_heading", "shifted_heading", "shifted_timer",
])
def test_conflicting_or_unanchored_full_ocr_never_starts_local_read(changed):
    frame = snapshot(changed=changed)
    provider = recover(frame)
    assert provider.calls == []
    assert "countdown_seconds" not in frame.observations["battle"]


@pytest.mark.parametrize("reads", [
    [OCRText("开战倒计时。", .999, NATIVE_ROI)],
    [OCRText("开战倒计时：：", .999, NATIVE_ROI)],
    [OCRText("开战倒计时：", .899, NATIVE_ROI)],
    [OCRText("开战倒计时：", .999, (1180, 20, 1380, 82))],
    [OCRText("开战倒计时：", .999, NATIVE_ROI),
     OCRText("开战倒计时：", .999, NATIVE_ROI)],
    [],
])
def test_wrong_weak_misaligned_or_duplicate_local_result_cannot_prove_time(reads):
    frame = snapshot()
    provider = recover(frame, LocalLineProvider(reads))
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert len(frame.observations["ocr"]) == 2
    assert frame.observations["preparation_label_recovery"]["status"] == "unproved"
    assert "countdown_seconds" not in frame.observations["battle"]


@pytest.mark.parametrize("scene,phase,version", [
    ("account_in_use", "scout", "18.647.2"),
    ("battle", "active", "18.647.2"),
    ("enemy_village", "active", "18.647.2"),
    ("enemy_village", "scout", "18.647.1"),
])
def test_scene_phase_or_version_change_rejects(scene, phase, version):
    frame = snapshot(scene=scene, phase=phase)
    provider = recover(frame, version=version)
    assert provider.calls == []
    assert "countdown_seconds" not in frame.observations["battle"]


def test_nondefault_baseline_maps_to_same_native_crop():
    frame = snapshot(baseline=(2560, 1440))
    candidate = preparation_label_line_recovery_candidate(frame, (2560, 1440))
    assert candidate is not None and candidate["baseline_roi"] == NATIVE_ROI
    provider = recover(frame, baseline=(2560, 1440))
    assert provider.calls == [(IMAGE, NATIVE_ROI)]
    assert _remaining_seconds(frame) == 12


def test_anonymized_hud_preserves_only_source_bound_pixels():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    image = read_frame(IMAGE)
    x0, y0, x1, y1 = metadata["preserved_xyxy"]
    assert hashlib.sha256(IMAGE.read_bytes()).hexdigest() == metadata["output_sha256"]
    assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == metadata["preserved_pixel_sha256"]
    outside = image.copy()
    outside[y0:y1, x0:x1] = 0
    assert np.max(outside) == 0
