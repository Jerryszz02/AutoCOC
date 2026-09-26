from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from autococ.errors import FlowError
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot, surrender_dialog_evidence
from autococ.strategy_execution import _end_battle_confirmation
from autococ.vision import ScreenshotRecognizer


FIXTURE = Path(__file__).parent / "fixtures/surrender-dialog-20260926.png"


def _ocr():
    return json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))["ocr"]


def _recognize(ocr):
    provider = Mock()
    provider.recognize.return_value = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                                       for item in ocr]
    provider.recognize_line.return_value = []
    return ScreenshotRecognizer(provider=provider).recognize(FIXTURE)


def test_actual_surrender_dialog_overrides_background_hud_and_pale_cloud_check():
    frame = _recognize(_ocr())
    assert frame.scene == "popup"
    assert frame.observations["matched_reasons"] == ["surrender_dialog_geometry_verified"]
    assert {item["name"] for item in frame.observations["buttons"]} == {"cancel", "confirm"}
    assert frame.observations["resource_source"] is None
    assert frame.observations["battle"] is None
    assert _end_battle_confirmation(frame) == (300, 200, 980, 650)


@pytest.mark.parametrize("change", ["body", "title", "cancel", "confidence", "position", "duplicate"])
def test_incomplete_or_unrelated_confirmation_is_not_surrender(change):
    ocr = _ocr()
    if change == "body":
        for item in ocr:
            if "撤退" in item["text"]:
                item["text"] = "确定花费宝石购买资源吗？"
    elif change == "title":
        next(item for item in ocr if item["text"] == "放弃？")["text"] = "确认购买？"
    elif change == "cancel":
        ocr = [item for item in ocr if item["text"] != "取消"]
    else:
        confirm = next(item for item in ocr if item["text"] == "确定")
        if change == "confidence":
            confirm["confidence"] = .8
        elif change == "position":
            confirm["bbox"] = [50, 446, 114, 486]
        else:
            ocr.append(deepcopy(confirm))
    assert surrender_dialog_evidence(ocr) is None
    frame = _recognize(ocr)
    assert frame.scene == "unknown"
    assert frame.observations["buttons"] == []
    with pytest.raises(FlowError):
        _end_battle_confirmation(SceneSnapshot("popup", .95, FIXTURE, {
            "ocr": ocr, "buttons": [{"name": "confirm", "point": [778, 466]}]}))


def test_confirmation_must_be_bound_to_observed_dialog_button():
    frame = _recognize(_ocr())
    observation = deepcopy(frame.observations)
    next(item for item in observation["buttons"] if item["name"] == "confirm")["point"] = [400, 250]
    with pytest.raises(FlowError, match="not bound"):
        _end_battle_confirmation(replace(frame, observations=observation))


def test_disconnect_keeps_priority_over_any_lingering_surrender_text():
    frame = _recognize(_ocr() + [{"text": "连接丢失", "confidence": .99, "bbox": [530, 280, 720, 320]}])
    assert frame.scene == "disconnected"
    assert frame.observations["buttons"] == []
    assert frame.observations["surrender_dialog"] is None
