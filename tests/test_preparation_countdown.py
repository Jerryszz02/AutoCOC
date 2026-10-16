import hashlib
import json
from pathlib import Path

import pytest

from autococ.battle_vision import attach_preparation_countdown, preparation_countdown_retry_candidate
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


def test_live_002_slightly_overlapping_ocr_boxes_keep_current_timer():
    # Original 00012-scout-camera-ready: OCR boxes overlap by seven pixels.
    result = read([item("开战倒计时", (581, 12, 681, 40), .99976),
                   item("23秒", (599, 33, 682, 80), .99997),
                   item("133", (790, 12, 812, 28), .99978)])
    assert result["countdown_seconds"] == 23
    assert result["countdown_evidence"]["frame"] == "current.png"
    assert result["countdown_evidence"]["source"] == "current_frame_ocr"


def _live_countdown_fixture():
    path = Path(__file__).parent / "fixtures" / "preparation_countdown_18_647_2_20261016.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "0e664eb0913c454568846762ded1733e81f0c81c0fa5aaa0b4bdf057137587bc")
    assert data["source_events_sha256"] == (
        "aeaafc055d9580e322d6a3790c90cd7a56b37987f227f24eef4bf145a5c2fec1")
    assert [frame["source_sha256"] for frame in data["frames"]] == [
        "88a5742dfd5af664122f7b0443d9c3fb583c38e4be2f0cbce50b2f866496ae2c",
        "f8ecf873ed790852201ef8b1991bb57f914bbf560e7af4c0e47db2d5814c0805",
    ]
    assert all([item["text"] for item in frame["cached_ocr"]] ==
               ["开战倒计时", "25秒"] for frame in data["frames"])
    return data["frames"]


def test_live_cached_countdown_fixture_matches_source_when_available():
    metadata = json.loads((Path(__file__).parent / "fixtures" /
                           "preparation_countdown_18_647_2_20261016.json").read_text(encoding="utf-8"))
    source_root = Path(r"E:\projects\AutoCOC")
    events = source_root / metadata["source_events"]
    if not events.is_file():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(events.read_bytes()).hexdigest() == metadata["source_events_sha256"]
    observations = [event for line in events.read_text(encoding="utf-8").splitlines()
                    if (event := json.loads(line)).get("kind") == "observation"]
    for frame in metadata["frames"]:
        source = source_root / frame["source_frame"]
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == frame["source_sha256"]
        (observation,) = [event for event in observations if Path(event["frame"]) == source]
        retained = [item for item in observation["observations"]["ocr"]
                    if item.get("text") in ("开战倒计时", "25秒")
                    and len(item.get("bbox", [])) == 4
                    and 500 <= item["bbox"][0] < item["bbox"][2] <= 750
                    and 0 <= item["bbox"][1] < item["bbox"][3] <= 100]
        assert retained == frame["cached_ocr"]


@pytest.mark.parametrize("frame_index,overlap", [(0, 7), (1, 10)])
def test_live_cached_countdown_geometry_accepts_current_frame(frame_index, overlap):
    frame = _live_countdown_fixture()[frame_index]
    label, timer = frame["cached_ocr"]
    assert label["bbox"][3] - timer["bbox"][1] == overlap
    snapshot = SceneSnapshot(frame["scene"], .92, Path(frame["source_frame"]),
                             {"battle": {}, "ocr": frame["cached_ocr"]})
    attach_preparation_countdown(snapshot)
    battle = snapshot.observations["battle"]
    assert battle["countdown_seconds"] == frame["expected_seconds"]
    assert battle["countdown_evidence"]["label"] == label
    assert battle["countdown_evidence"]["timer"] == timer
    assert battle["countdown_evidence"]["frame"] == str(Path(frame["source_frame"]))


@pytest.mark.parametrize("change", ["excessive_overlap", "misaligned", "duplicate_timer",
                                            "duplicate_anchor", "low_confidence", "wrong_scene"])
def test_live_cached_countdown_rejects_ambiguous_or_unrelated_timer(change):
    frame = _live_countdown_fixture()[1]
    ocr = [dict(item) for item in frame["cached_ocr"]]
    scene = frame["scene"]
    if change == "excessive_overlap":
        ocr[1]["bbox"] = [594, 28, 686, 88]  # 12px > 28 * .40 = 11.2px.
    elif change == "misaligned":
        ocr[1]["bbox"] = [700, 30, 792, 88]
    elif change == "duplicate_timer":
        ocr.append({"text": "24秒", "confidence": .99, "bbox": [594, 40, 686, 88]})
    elif change == "duplicate_anchor":
        ocr.append(dict(ocr[0]))
    elif change == "low_confidence":
        ocr[1]["confidence"] = .89
    else:
        scene = "battle"
    snapshot = SceneSnapshot(scene, .92, Path(frame["source_frame"]),
                             {"battle": {}, "ocr": ocr})
    attach_preparation_countdown(snapshot)
    assert "countdown_seconds" not in snapshot.observations["battle"]


def test_large_overlap_does_not_claim_a_timer():
    result = read([item("开战倒计时", (581, 12, 681, 40)),
                   item("23秒", (599, 20, 682, 80))])
    assert "countdown_seconds" not in result


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


def _line_recheck_fixture():
    path = Path(__file__).parent / "fixtures" / "line_countdown_recheck_18_647_2_20261016.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_live_190802_cached_timer_provenance_when_available():
    metadata = _line_recheck_fixture()
    root = Path(r"E:\projects\AutoCOC")
    events = root / metadata["source_events"]
    if not events.is_file():
        pytest.skip("Original local live evidence is unavailable")
    assert hashlib.sha256(events.read_bytes()).hexdigest() == metadata["source_events_sha256"]
    observations = [event for line in events.read_text(encoding="utf-8").splitlines()
                    if (event := json.loads(line)).get("kind") == "observation"]
    for frame in metadata["frames"]:
        source = root / frame["source_frame"]
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == frame["source_sha256"]
        (observation,) = [event for event in observations if Path(event["frame"]) == source]
        retained = [item for item in observation["observations"]["ocr"]
                    if item.get("text") in ("开战倒计时", "21秒", "18秒")
                    and isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4
                    and 500 <= item["bbox"][0] < item["bbox"][2] <= 750
                    and 0 <= item["bbox"][1] < item["bbox"][3] <= 100]
        assert retained == frame["cached_ocr"]
        assert observation["observations"]["observed_at_monotonic"] == frame["observed_at_monotonic"]


def test_live_190802_weak_timer_only_qualifies_for_new_observation():
    frames = _line_recheck_fixture()["frames"]
    weak = SceneSnapshot("enemy_village", .92, Path(frames[0]["source_frame"]),
                         {"battle": {}, "ocr": frames[0]["cached_ocr"]})
    attach_preparation_countdown(weak)
    assert "countdown_seconds" not in weak.observations["battle"]
    candidate = preparation_countdown_retry_candidate(weak)
    assert candidate is not None and candidate["reason"] == "unique_weak_timer"
    assert candidate["timer"]["confidence"] == .87578
    assert weak.observations["battle"] == {}  # A weak value never grants time.
    later = SceneSnapshot("enemy_village", .92, Path(frames[1]["source_frame"]),
                          {"battle": {}, "ocr": frames[1]["cached_ocr"]})
    attach_preparation_countdown(later)
    assert later.observations["battle"]["countdown_seconds"] == 18
    assert preparation_countdown_retry_candidate(later) is None


@pytest.mark.parametrize("change", ["duplicate_weak", "duplicate_strong", "duplicate_anchor",
                                            "weak_label", "invalid_confidence", "wrong_scene",
                                            "misaligned", "invalid_timer", "stale_evidence"])
def test_weak_timer_diagnostic_rejects_ambiguous_or_untrusted_evidence(change):
    frame = _line_recheck_fixture()["frames"][0]
    ocr = [dict(item) for item in frame["cached_ocr"]]
    scene, battle = "enemy_village", {}
    if change == "duplicate_weak":
        ocr.append(dict(ocr[1]))
    elif change == "duplicate_strong":
        ocr.append(item("20秒", (710, 40, 744, 76), .99))
    elif change == "duplicate_anchor":
        ocr.append(dict(ocr[0]))
    elif change == "weak_label":
        ocr[0]["confidence"] = .89
    elif change == "invalid_confidence":
        ocr[1]["confidence"] = float("nan")
    elif change == "wrong_scene":
        scene = "account_in_use"
    elif change == "misaligned":
        ocr[1]["bbox"] = [700, 35, 775, 78]
    elif change == "invalid_timer":
        ocr[1]["text"] = "??秒"
    else:
        battle = {"countdown_seconds": 21, "countdown_evidence": {
            "frame": "old.png", "source": "current_frame_ocr", "confidence": .99}}
    snapshot = SceneSnapshot(scene, .92, Path("weak.png"), {"battle": battle, "ocr": ocr})
    assert preparation_countdown_retry_candidate(snapshot) is None
