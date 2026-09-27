from pathlib import Path

import pytest

from autococ.battle_vision import attach_preparation_countdown
from autococ.scene import SceneSnapshot


def item(text, box=(540, 10, 740, 35), confidence=.99):
    return {"text": text, "bbox": list(box), "confidence": confidence}


def read(items, scene="enemy_village"):
    frame = SceneSnapshot(scene, .95, Path("current.png"), {"battle": {}, "ocr": items})
    attach_preparation_countdown(frame)
    return frame.observations["battle"]


@pytest.mark.parametrize("text", ["开战倒计时30秒", "战斗开始倒计时：30", "Battle starts in 30s"])
def test_combined_timer(text):
    result = read([item(text)])
    assert result["countdown_seconds"] == 30
    assert result["countdown_evidence"]["frame"] == "current.png"


def test_split_timer_and_nonstandard_resolution():
    frame = SceneSnapshot("enemy_village", .95, Path("frame.png"), {"battle": {}, "ocr": [
        item("Battle starts in", (1080, 20, 1480, 70)), item("0:24", (1220, 76, 1340, 130))]})
    attach_preparation_countdown(frame, (2560, 1440))
    assert frame.observations["battle"]["countdown_seconds"] == 24


@pytest.mark.parametrize("items,scene", [
    ([item("战斗结束倒计时30秒")], "enemy_village"),
    ([item("开战倒计时30秒")], "battle"),
    ([item("开战倒计时30秒", confidence=.7)], "enemy_village"),
    ([item("开战倒计时30秒", confidence=float("nan"))], "enemy_village"),
    ([item("开战倒计时30秒", (10, 200, 210, 235))], "enemy_village"),
    ([item("开战倒计时"), item("30秒", (1100, 45, 1160, 75))], "enemy_village"),
    ([item("开战倒计时"), item("30秒", (610, 40, 670, 65)), item("29秒", (610, 70, 670, 95))], "enemy_village"),
    ([item("开战倒计时0秒")], "enemy_village"),
    ([item("开战倒计时120秒")], "enemy_village"),
])
def test_unknown_or_ambiguous_timer_has_no_budget(items, scene):
    assert "countdown_seconds" not in read(items, scene)


def test_does_not_reuse_prior_timer():
    frame = SceneSnapshot("enemy_village", .95, Path("new.png"), {
        "battle": {"countdown_seconds": 30, "countdown_evidence": {"frame": "old.png"}}, "ocr": []})
    attach_preparation_countdown(frame)
    assert frame.observations["battle"] == {}
