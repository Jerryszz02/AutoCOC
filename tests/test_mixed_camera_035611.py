"""Offline original/anonymous camera replay; no result retroactively authorizes input."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.frozen_line import _file_sha256
from autococ.images import read_frame
from autococ.mixed_probe import _probe_count
from autococ.probed_anchor import ExactCameraError, verify_exact_anchor_camera
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/mixed_camera_035611_18_647_2"
LIVE = (Path(r"E:\projects\AutoCOC") / "reports/completion-20261015/live" /
        "20261017-035611-364323-dbbca001")
ANCHORS = ((423, 140), (296, 467))
FRAMES = {row["role"]: row for row in json.loads(
    (FIXTURES / "provenance.json").read_text(encoding="utf-8"))["rows"]}


def _path(role: str, source: str) -> Path:
    row = FRAMES[role]
    return (FIXTURES / row["fixture"] if source == "anonymous" else
            LIVE / "frames" / row["source_frame"])


def _sha_bound(role: str, source: str) -> Path:
    path = _path(role, source)
    assert _file_sha256(path) == FRAMES[role][
        "fixture_sha256" if source == "anonymous" else "source_sha256"]
    return path


def test_fixture_masks_everything_outside_camera_and_dragon_card():
    metadata = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "autococ.anonymous_mixed_camera_035611.v1"
    assert metadata["client_version"] == "18.647.2"
    assert metadata["truth_status"] == "offline_camera_and_card_evidence_only"
    assert metadata["human_reviewed"] is False
    assert metadata["formal_acceptance"] is False
    assert metadata["camera_roi_at_1280x720"] == [240, 150, 1050, 490]
    if LIVE.is_dir():
        assert _file_sha256(LIVE / "events.jsonl") == metadata["source_events_sha256"]
        assert _file_sha256(LIVE.parent / f"run-{LIVE.name}.json") == metadata[
            "source_report_sha256"]
    for role in FRAMES:
        anon = read_frame(_sha_bound(role, "anonymous"))
        assert anon.shape == (1440, 2560, 3)
        mask = np.zeros(anon.shape[:2], dtype=bool)
        for left, top, right, bottom in metadata["retained_native_rectangles_xyxy"]:
            mask[top:bottom, left:right] = True
        assert np.all(anon[~mask] == 45)
        original = _path(role, "original")
        if original.is_file():
            _sha_bound(role, "original")
            assert np.array_equal(anon[mask], read_frame(original)[mask])
            # The camera function reads exactly this baseline ROI. Keeping
            # its native 2x pixels yields the same ORB input after resize.
            a = cv2.resize(anon, (1280, 720), interpolation=cv2.INTER_AREA)
            b = cv2.resize(read_frame(original), (1280, 720),
                           interpolation=cv2.INTER_AREA)
            assert np.array_equal(a[150:490, 240:1050], b[150:490, 240:1050])


@pytest.mark.parametrize("source", ["anonymous", "original"])
def test_real_camera_gate_rejects_24_and_accepts_22_23_25(source):
    if source == "original" and not LIVE.is_dir():
        pytest.skip("Optional original live run is unavailable")
    frame = _sha_bound("source", source)
    for role in ("balloon_post", "dragon_selected_0", "later_battle"):
        proof = verify_exact_anchor_camera(frame, _sha_bound(role, source), ANCHORS)
        assert proof["inliers"] >= 80
        assert proof["inliers"] / proof["matches"] >= .9
        assert proof["residual_p95"] <= .5
        assert max(proof["anchor_offsets"]) <= .25
    with pytest.raises(ExactCameraError, match="insufficient stationary map inliers"):
        verify_exact_anchor_camera(frame, _sha_bound("dragon_selected_1", source), ANCHORS)


@pytest.mark.parametrize("source", ["anonymous", "original"])
def test_later_25_has_independent_count_and_border_but_was_after_failure(source):
    if source == "original" and not LIVE.is_dir():
        pytest.skip("Optional original live run is unavailable")
    before = _sha_bound("before", source)
    frozen = _sha_bound("source", source)
    later = _sha_bound("later_battle", source)
    snapshot = SceneSnapshot("battle", .99, later, {
        "recognition_scope": "full", "ocr": [], "battle": {"slots": []}})
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    base = SimpleNamespace(toolbar_before_frame_id=str(before),
                           frame=SimpleNamespace(screenshot_path=frozen,
                                                 frame_id=str(frozen), resolution=(1280, 720)),
                           frame_sha256=_file_sha256(frozen), client_version="18.647.2")
    plan = SimpleNamespace(base=base, dragon_point=(185, 653),
                           dragon_bbox=(141, 595, 230, 711))
    reading = _probe_count(session, plan, snapshot, "dragon", 1, selected=True)
    assert reading["count"] == 1 and reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92
    assert reading["portrait_source"] == "current_frame_saved_battle_portrait"
    assert reading["source_identity"]["unit_id"] == "dragon"
    assert reading["selected_card_border"] == [140, 590, 232, 709]
