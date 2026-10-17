import json
from unittest.mock import patch

import pytest

from scripts.vision_camera_evidence import camera_evidence


def test_camera_calibration_requires_current_frame_and_does_not_survive_motion(tmp_path):
    frames = []
    for number in range(5):
        path = tmp_path / f"{number}.png"
        path.write_bytes(f"frame {number}".encode())
        frames.append(path)
    rows = [{"kind": "session_connected", "client_version": "18.647.2"}]
    for index, path in enumerate(frames):
        rows.append({"kind": "observation", "scene": "battle", "frame": str(path),
                     "observations": {"frame_id": str(index), "captured_at_utc": "untrusted log time"}})
        if index == 1:
            rows.append({"kind": "line_minimum_zoom", "frame": str(path), "verified": True,
                         "motion": {"stationary": True}})
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    # 0->1 stable, 1->2 stable + calibration stable, 2->3 moved, 3->4 stable.
    with patch("scripts.vision_camera_evidence.measure_camera_motion", side_effect=[
        {"stationary": True}, {"stationary": True}, {"stationary": True},
        {"stationary": False}, {"stationary": True},
    ]):
        result = camera_evidence(tmp_path)["frames"]
    assert [row["candidate_zoom_verified"] for row in result] == [False, False, True, False, False]
    assert result[2]["candidate_camera_epoch"] != result[3]["candidate_camera_epoch"]
    assert all(row["captured_at_utc"] is None and row["review_status"] == "pending" for row in result)
    assert all(len(row["source_sha256"]) == 64 for row in result)


def test_camera_evidence_rejects_external_frame(tmp_path):
    rows = [{"kind": "session_connected", "client_version": "18.647.2"},
            {"kind": "observation", "scene": "battle", "frame": str(tmp_path.parent / "external.png")}]
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the session"):
        camera_evidence(tmp_path)
