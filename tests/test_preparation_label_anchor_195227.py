"""Source-bound preparation HUD label guard regression from the 195227 run."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.battle_vision import BattleObserver
from autococ.images import read_frame
from autococ.scene import SceneSnapshot


FIXTURES = Path(__file__).parent / "fixtures" / "preparation_label_anchor_18_647_2"


def fixture(number):
    path = FIXTURES / f"frame{number}.png"
    meta = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    return path, meta


def snapshot(*, baseline=(1280, 720), changes=None, scene="enemy_village"):
    path, meta = fixture(15)
    factor = baseline[0] / 1280
    assert factor == baseline[1] / 720

    def scaled(box):
        return [int(value * factor) for value in box]

    ocr = copy.deepcopy(meta["cached_ocr"])
    buttons = copy.deepcopy(meta["cached_buttons"])
    for item in ocr + buttons:
        item["bbox"] = scaled(item["bbox"])
    evidence = copy.deepcopy(meta["cached_countdown_evidence"])
    evidence["frame"] = str(path)  # Fixture maps the same pixels to a masked path.
    for key in ("label", "timer"):
        evidence[key]["bbox"] = scaled(evidence[key]["bbox"])
    battle = {"countdown_seconds": 13, "countdown_evidence": evidence}
    if changes is not None:
        changes(ocr, buttons, battle)
    return SceneSnapshot(scene, .92, path, {"ocr": ocr, "buttons": buttons, "battle": battle})


def observer(baseline=(1280, 720)):
    return BattleObserver(SimpleNamespace(baseline_resolution=baseline))


def pixels(number):
    path, _ = fixture(number)
    return cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)


def test_actual_frame16_old_full_label_fails_and_static_label_passes():
    current = pixels(16)
    prior = snapshot(changes=lambda _ocr, _buttons, battle:
                     battle["countdown_evidence"].update(frame="stale-frame"))
    old = observer()
    old.remember(prior)
    assert old.anchors["enemy_village"][0][0] == (584, 12, 692, 40)
    assert old.guard(current) is None

    fixed = observer()
    fixed.remember(snapshot())
    assert fixed.anchors["enemy_village"][0][0] == (584, 12, 692, 34)
    assert fixed.guard(current)[0] == "enemy_village"
    assert fixed.guard(pixels(17))[0] == "enemy_village"


@pytest.mark.parametrize("variant", ["button", "heading", "dimmed"])
def test_current_frame_still_requires_both_unchanged_hud_anchors(variant):
    current = pixels(16).copy()
    if variant == "button":
        current[526:552, 58:128] = 0
    elif variant == "heading":
        current[12:34, 584:692] = 0
    else:
        current = np.uint8(current * .6)
    saved = observer()
    saved.remember(snapshot())
    assert saved.guard(current) is None


@pytest.mark.parametrize("variant", [
    "wrong_frame", "wrong_evidence_bbox", "wrong_evidence_text", "missing_evidence_with_count",
    "ambiguous_timer", "far_strong_timer", "weak_timer", "weak_label", "merged_label_timer", "offset_timer",
    "small_label", "invalid_evidence", "explicit_null_evidence",
])
def test_unsupported_or_ambiguous_source_keeps_original_label_box(variant):
    def change(ocr, _buttons, battle):
        if variant == "wrong_frame":
            battle["countdown_evidence"]["frame"] = "prior-frame"
        elif variant == "wrong_evidence_bbox":
            battle["countdown_evidence"]["timer"]["bbox"][0] += 5
        elif variant == "wrong_evidence_text":
            battle["countdown_evidence"]["label"]["text"] = "开战倒计时X"
        elif variant == "missing_evidence_with_count":
            battle.pop("countdown_evidence")
        elif variant == "ambiguous_timer":
            ocr.append(copy.deepcopy(ocr[1]))
        elif variant == "far_strong_timer":
            extra = copy.deepcopy(ocr[1])
            extra["bbox"] = [800, 34, 870, 76]
            ocr.append(extra)
        elif variant == "weak_timer":
            ocr[1]["confidence"] = .89
        elif variant == "weak_label":
            ocr[0]["confidence"] = .89
        elif variant == "merged_label_timer":
            ocr[0]["text"] = "开战倒计时13秒"
            ocr.pop()
        elif variant == "offset_timer":
            ocr[1]["bbox"] = [604, 40, 676, 76]
        elif variant == "small_label":
            ocr[0]["bbox"] = [584, 12, 620, 27]
            ocr[1]["bbox"] = [600, 22, 625, 40]
        elif variant == "invalid_evidence":
            battle["countdown_evidence"] = {"frame": "bad"}
        else:
            battle["countdown_evidence"] = None
            battle.pop("countdown_seconds")

    saved = observer()
    saved.remember(snapshot(changes=change))
    if variant == "weak_label":
        assert "enemy_village" not in saved.anchors
        return
    assert saved.anchors["enemy_village"][0][0] == tuple(snapshot(changes=change).observations["ocr"][0]["bbox"])


def test_non_default_baseline_scales_static_box_and_battle_label_is_unchanged():
    saved = observer((2560, 1440))
    saved.remember(snapshot(baseline=(2560, 1440)))
    assert saved.anchors["enemy_village"][0][0] == (584, 12, 692, 34)
    assert saved.guard(pixels(16))[0] == "enemy_village"
    battle = observer()
    battle.remember(snapshot(scene="battle"))
    assert battle.anchors["battle"][0][0] == (584, 12, 692, 40)


@pytest.mark.parametrize("number", [15, 16, 17])
def test_masked_hud_and_cached_ocr_are_bound_to_original_events(number):
    path, meta = fixture(number)
    image = read_frame(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["output_sha256"]
    masked = image.copy()
    for box, expected_hash in zip(meta["native_preserved_xyxy"],
                                  meta["native_preserved_pixel_sha256"]):
        x0, y0, x1, y1 = box
        assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == expected_hash
        masked[y0:y1, x0:x1] = 0
    assert masked.max() == 0
    source, events = Path(meta["source_frame"]), Path(meta["events_source"])
    if not source.exists() or not events.exists():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["source_sha256"]
    assert hashlib.sha256(events.read_bytes()).hexdigest() == meta["events_sha256"]
    original = read_frame(source)
    for x0, y0, x1, y1 in meta["native_preserved_xyxy"]:
        assert np.array_equal(image[y0:y1, x0:x1], original[y0:y1, x0:x1])
    event = next(json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()
                 if line.startswith("{") and json.loads(line).get("frame") == str(source))
    assert [item for item in event["observations"].get("ocr", [])
            if isinstance(item, dict) and isinstance(item.get("bbox"), list)
            and item["bbox"][0] >= 540 and item["bbox"][2] <= 710
            and item["bbox"][1] < 100
            and ("倒计时" in item.get("text", "") or "秒" in item.get("text", ""))] == meta["cached_ocr"]
    assert [item for item in event["observations"].get("buttons", [])
            if item.get("name") == "end_battle"] == meta["cached_buttons"]
    assert (event["observations"].get("battle") or {}).get("countdown_evidence") == \
        meta["cached_countdown_evidence"]
    assert (event["scene"], event["confidence"]) == \
        (meta["source_scene"], meta["source_confidence"])
