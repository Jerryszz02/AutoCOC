"""Read-only helpers on anonymized 032110 frames; never replay an input."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import cv2
import numpy as np

from autococ.post_deployment_skill import (
    _bounded_stabilization_candidate,
    _current_warden,
    _minimum_zoom_motion,
)
from autococ.terrain import measure_camera_motion


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skill_camera_032110_18_647_2"
PROVENANCE = json.loads((FIXTURE / "provenance.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape == (1440, 2560, 3)
    return image


def test_anonymized_frames_are_hash_bound_and_masked() -> None:
    assert PROVENANCE["schema"] == "skill_camera_anonymized_fixture_v1"
    assert PROVENANCE["human_ground_truth"] is False
    assert PROVENANCE["dynamic_event_validation"] is False
    assert PROVENANCE["device_input_proof"] is False
    assert PROVENANCE["input_replay_allowed"] is False
    assert PROVENANCE["original_skill_attempted"] is False
    assert len(PROVENANCE["frames"]) == 4
    allowed = np.zeros((1440, 2560), dtype=bool)
    for left, top, right, bottom in PROVENANCE["mask_rectangles_native_xyxy"]:
        allowed[top:bottom, left:right] = True
    source_root = os.environ.get("AUTOCOC_SOURCE_ROOT")
    for row in PROVENANCE["frames"]:
        fixture = FIXTURE / row["name"]
        assert fixture.is_file() and _sha(fixture) == row["fixture_sha256"]
        image = _image(fixture)
        assert not np.any(image[~allowed])
        source_relative = PurePosixPath(row["source_relative"])
        assert not source_relative.is_absolute() and ".." not in source_relative.parts
        if source_root:
            source = Path(source_root, *source_relative.parts)
            assert source.is_file() and _sha(source) == row["source_sha256"]
            assert np.array_equal(image[allowed], _image(source)[allowed])


def test_actual_readonly_motion_and_current_warden_survive_masking() -> None:
    assert PROVENANCE["source_diagnostic_sha256"] == (
        "a397bf17f8295af219658c72882fb04be95d51721da6db0cbdbcb3a9b98b6b42")
    assert len(PROVENANCE["readonly_diagnostic_motion_pairs"]) == 3
    frame_dirs = [FIXTURE]
    source_root = os.environ.get("AUTOCOC_SOURCE_ROOT")
    if source_root:
        source_run = PurePosixPath(PROVENANCE["source_run_relative"])
        frame_dirs.append(Path(source_root, *source_run.parts, "frames"))
    for directory in frame_dirs:
        for row in PROVENANCE["readonly_diagnostic_motion_pairs"]:
            motion = measure_camera_motion(directory / row["from"], directory / row["to"])
            translation = math.hypot(motion["affine"][0][2], motion["affine"][1][2])
            # Masked pixels can affect ORB selection; compare the measured gate
            # and a narrow observed interval, not exact affine coefficients.
            assert abs(translation - row["translation_px"]) < .1
            assert motion["matched_points"] >= 40 and motion["inliers"] >= 40
            assert _minimum_zoom_motion(motion) is row["strict_minimum_zoom_motion"]
            assert _bounded_stabilization_candidate(motion) is row["bounded_stabilization_candidate"]
        for row in PROVENANCE["readonly_diagnostic_warden"]:
            frame = SimpleNamespace(screenshot_path=directory / row["frame"])
            target = _current_warden(SimpleNamespace(client_version="18.647.2"), frame)
            state = target["state"] or {}
            assert target["verified"] is row["verified"]
            assert target["locator"]["bbox"] == row["bbox"]
            assert state.get("ability_ready") is row["ability_ready"]
            assert state.get("ability_used") is row["ability_used"]
            health = state["evidence"].get("health_bar") or state["evidence"].get("red_health_bar")
            assert health is not None and health["method"] == row["health_method"]
