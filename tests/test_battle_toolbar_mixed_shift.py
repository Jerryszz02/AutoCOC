"""Source-bound cached-OCR replay of the 18.647.2 connected battle toolbar."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autococ.battle_vision import selected_card_bbox
from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\projects\AutoCOC")
METADATA = ROOT / "tests/fixtures/battle_toolbar_mixed_shift_18_647_2_20261016.json"
EXPECTED = {"electro_dragon": (10, [93, 595, 182, 711], [137, 653]),
            "dragon": (1, [190, 595, 278, 711], [234, 653]),
            "furnace": (1, [287, 595, 375, 711], [331, 653]),
            "balloon": (1, [383, 595, 472, 711], [427, 653])}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retained_ocr(items, rectangles):
    return [item for item in items if item.get("bbox") and
            any(left <= item["bbox"][0] < item["bbox"][2] <= right and
                top <= item["bbox"][1] < item["bbox"][3] <= bottom
                for left, top, right, bottom in rectangles)]


def _read(path: Path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _texts(row):
    return [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
            for item in row["cached_ocr"] if item.get("bbox")]


def _observe(path: Path, texts: list[OCRText], scene: str, *, version="18.647.2"):
    recognizer = ScreenshotRecognizer(provider=None)
    recognizer.client_version = version
    return recognizer._battle_observation(path, texts, scene)["slots"]


def _troops(slots):
    return {item["unit_id"]: item for item in slots
            if item["kind"] == "troop" and item["unit_id"] is not None}


def test_anonymous_frames_are_hash_bound_and_keep_only_toolbar_and_timer():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["client_version"] == "18.647.2"
    assert metadata["human_reviewed"] is False
    assert metadata["counts_toward_acceptance"] is False
    assert metadata["cached_ocr_filter"] == (
        "Retain only OCR items whose entire xyxy bbox lies inside one of the "
        "frame's retained_baseline_rectangles_xyxy; preserve original order and values.")
    assert [Path(item["source_frame"]).name[:5] for item in metadata["frames"]] == [
        "00012", "00013", "00014", "00015"]
    for row in metadata["frames"]:
        path = ROOT / row["path"]
        image = _read(path)
        assert image.shape == (720, 1280, 3)
        assert _sha(path) == row["sha256"]
        mask = np.zeros(image.shape[:2], dtype=bool)
        for left, top, right, bottom in row["retained_baseline_rectangles_xyxy"]:
            mask[top:bottom, left:right] = True
        assert np.all(image[~mask] == 45)
        assert row["cached_ocr"] == _retained_ocr(
            row["cached_ocr"], row["retained_baseline_rectangles_xyxy"])
        source = SOURCE_ROOT / row["source_frame"]
        if source.is_file():
            assert _sha(source) == row["source_sha256"]
            original = cv2.resize(_read(source), (1280, 720), interpolation=cv2.INTER_AREA)
            assert np.array_equal(image[mask], original[mask])
        events = SOURCE_ROOT / row["source_events"]
        if events.is_file():
            assert _sha(events) == row["source_events_sha256"]
            observations = [event for line in events.read_text(encoding="utf-8").splitlines()
                            if (event := json.loads(line)).get("kind") == "observation"
                            and Path(str(event.get("frame"))) == source]
            assert len(observations) == 1
            assert row["cached_ocr"] == _retained_ocr(
                observations[0]["observations"]["ocr"],
                row["retained_baseline_rectangles_xyxy"])


def test_cached_ocr_and_real_pixels_locate_all_four_before_selection():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    for row in metadata["frames"][:2]:
        slots = _observe(ROOT / row["path"], _texts(row), row["scene"])
        troops = _troops(slots)
        assert set(troops) == set(EXPECTED)
        for unit_id, (count, bbox, point) in EXPECTED.items():
            item = troops[unit_id]
            assert (item["count"], item["bbox"], item["point"]) == (count, bbox, point)
            assert item["evidence"]["identity"]["version_verified"] is True
            assert item["evidence"]["identity"]["template_client_version"] == "18.647.2"
        assert troops["dragon"]["bbox"][0] > troops["electro_dragon"]["bbox"][2]


def test_corrected_source_box_proves_closed_selected_outline_and_stable_center():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    before = _troops(_observe(ROOT / metadata["frames"][0]["path"],
                              _texts(metadata["frames"][0]), "enemy_village"))
    dragon = before["dragon"]
    for row in metadata["frames"][2:]:
        path = ROOT / row["path"]
        assert selected_card_bbox(path, (181, 595, 278, 711)) is None
        selected = selected_card_bbox(path, tuple(dragon["bbox"]))
        assert selected == (188, 590, 281, 709)
        center = [(selected[0] + selected[2]) // 2,
                  (selected[1] + selected[3]) // 2]
        assert all(abs(a - b) <= 4 for a, b in zip(center, dragon["point"]))
        slots = _troops(_observe(path, _texts(row), row["scene"]))
        assert set(slots) == set(EXPECTED)
        for unit_id in ("electro_dragon", "furnace", "balloon"):
            assert slots[unit_id]["bbox"] == EXPECTED[unit_id][1]
        assert slots["dragon"]["count"] == 1


@pytest.mark.parametrize("mutation", ["missing_side", "missing_header", "wrong_portrait",
                                      "shifted_header", "wrong_version"])
def test_connected_toolbar_rejects_incomplete_or_misbound_card_evidence(tmp_path, mutation):
    row = json.loads(METADATA.read_text(encoding="utf-8"))["frames"][0]
    image = _read(ROOT / row["path"])
    texts = _texts(row)
    version = "18.647.2"
    if mutation == "missing_side":
        image[625:695, 185:195] = 45
    elif mutation == "wrong_portrait":
        image[630:690, 215:268] = 45
    elif mutation == "missing_header":
        texts = [item for item in texts if not (item.text == "x1" and
                                                item.bbox is not None and 220 < item.bbox[0] < 280)]
    elif mutation == "shifted_header":
        texts = [replace(item, bbox=tuple(value + (65 if index % 2 == 0 else 0)
                                          for index, value in enumerate(item.bbox)))
                 if item.text == "x1" and item.bbox is not None and 220 < item.bbox[0] < 280
                 else item for item in texts]
    else:
        version = "18.647.1"
    path = tmp_path / f"{mutation}.png"
    cv2.imencode(".png", image)[1].tofile(path)
    troops = _troops(_observe(path, texts, row["scene"], version=version))
    assert {unit_id: troop["count"] for unit_id, troop in troops.items()} != \
           {unit_id: value[0] for unit_id, value in EXPECTED.items()}
    assert "dragon" not in troops or troops["dragon"]["bbox"] == EXPECTED["dragon"][1]
