"""Current-client HUD guard replay from source-bound masked development frames."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from autococ.battle_vision import selected_card_bbox
from autococ.images import read_frame
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures" / "line_boundary_wait_18_647_2"
META = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_masked_frames_retain_only_declared_hud_and_one_card():
    rectangles = META["retained_rectangles_at_1280x720"]
    for row in META["frames"].values():
        path = FIXTURES / row["path"]
        assert _sha(path) == row["derived_sha256"]
        image = read_frame(path)
        assert image.shape == (720, 1280, 3)
        outside = np.ones((720, 1280), dtype=bool)
        for x0, y0, x1, y1 in rectangles:
            outside[y0:y1, x0:x1] = False
        assert not np.any(image[outside])


@pytest.mark.parametrize("prefix", ["negative", "positive"])
def test_real_masked_hud_wait_uses_no_ocr_or_full_fallback(prefix):
    source = META["frames"][f"{prefix}_source"]
    target = META["frames"][f"{prefix}_wait"]
    source_path, target_path = FIXTURES / source["path"], FIXTURES / target["path"]
    # The original countdown evidence belongs to the original native PNG.
    # The derived image only replays the independent OCR label and timer boxes
    # needed to choose a static HUD patch; it does not mint a countdown.
    prior = SceneSnapshot("enemy_village", source["source_confidence"], source_path, {
        "ocr": source["cached_ocr_retained"],
        "buttons": source["cached_buttons_retained"],
        "battle": {"slots": source["cached_battle_retained"]["slots"]},
    })
    provider = Mock()
    reader = ScreenshotRecognizer(provider=provider)
    reader.client_version = META["client_version"]
    reader.recognize = Mock(side_effect=AssertionError("full OCR is forbidden"))
    wait = reader.recognize_battle(target_path, purpose="line_boundary_wait", previous=prior)
    assert wait.scene == "enemy_village"
    assert wait.observations["recognition_scope"] == "line_boundary_wait"
    assert wait.observations["fast_fallback"] is False
    assert wait.observations["matched_reasons"] == ["two_current_hud_anchors"]
    assert wait.observations["ocr"] == []
    assert wait.observations["battle"] == {"slots": []}
    first = source["cached_battle_retained"]["slots"][0]
    assert selected_card_bbox(target_path, first["bbox"]) is not None
    reader.recognize.assert_not_called()
    provider.recognize.assert_not_called()
    provider.recognize_line_image.assert_not_called()


def test_local_source_provenance_when_originals_are_available():
    originals = [Path(row["source_frame"]) for row in META["frames"].values()]
    if not all(path.is_file() for path in originals):
        pytest.skip("Original local development frames are unavailable")
    for row in META["frames"].values():
        original, event_path = Path(row["source_frame"]), Path(row["source_events"])
        assert _sha(original) == row["source_frame_sha256"]
        assert _sha(event_path) == row["source_events_sha256"]
        original_rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
        observed = [item for item in original_rows if item.get("kind") == "observation"
                    and item.get("frame") == str(original)]
        assert len(observed) == 1
        source_ocr = observed[0]["observations"]["ocr"]
        filtered_ocr = [item for item in source_ocr
                        if ("开战倒计时" in item.get("text", "")
                            or item.get("text", "").endswith("秒"))
                        and item.get("bbox") and
                        555 <= item["bbox"][0] < item["bbox"][2] <= 715 and
                        0 <= item["bbox"][1] < item["bbox"][3] <= 95]
        assert filtered_ocr == row["cached_ocr_retained"]
        native = read_frame(original)
        assert native.shape == (1440, 2560, 3)
        small = cv2.resize(native, (1280, 720), interpolation=cv2.INTER_AREA)
        derived = read_frame(FIXTURES / row["path"])
        for x0, y0, x1, y1 in META["retained_rectangles_at_1280x720"]:
            assert np.array_equal(derived[y0:y1, x0:x1], small[y0:y1, x0:x1])
