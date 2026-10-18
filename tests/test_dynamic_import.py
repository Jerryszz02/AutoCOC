"""Contract tests for the reviewed R queue-to-snapshot bridge."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.vision.data import DatasetError, sha256
from training.vision.dynamic_import import import_reviewed_r_windows
from training.vision.dynamic_snapshot import verify_dynamic_snapshot


POLICY = "candidate_queue_only; no model training truth or formal human review is claimed"


def _fixture(root: Path, *, kind: str = "r_card", sessions: int = 1):
    queue_sessions, windows = [], []
    for index in range(sessions):
        session_id, frame_id = f"session-{index}", f"frame-{index}"
        path = root / session_id / "frames" / "one.png"
        path.parent.mkdir(parents=True)
        image = np.full((16, 16, 3), index + 30, np.uint8)
        path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
        digest = sha256(path)
        queue_sessions.append({"source_session": session_id, "client_version": "18.647.2",
            "battle_frames": [{"frame": str(path), "sha256": digest, "frame_id": frame_id,
                "capture_timestamp_basis": "capture_request_start",
                "capture_started_monotonic": 100.0 + index,
                "capture_started_utc": "2026-10-15T20:00:00+00:00",
                "hero_card_candidates": [{"target_id": "hero-card:archer_queen",
                    "candidate_roi_bbox_source": [0, 0, 16, 16],
                    "identity_confidence": .99}]}]})
        reviewed_frame = {"queue_session": session_id, "queue_frame_id": frame_id,
            "battle_id": f"battle-{index}", "layout_id": f"layout-{index}",
            "camera_epoch": "stable-1", "camera_stable": True, "zoom_verified": True,
            "roi_bbox": [0, 0, 16, 16],
            "state_label": "unknown" if kind == "r_card" else "intact",
            "state_review_status": "reviewed", "state_review_method": "agent_visual",
            "state_reviewer": "fixture reviewer", "state_review_evidence": "original pixels inspected",
            "state_target_id": "hero_skill:archer_queen" if kind == "r_card" else "building:air_defense:one",
            "state_source_frame_sha256": digest, "state_roi_bbox": [0, 0, 16, 16]}
        if kind == "r_card":
            reviewed_frame["hero_deployment_truth"] = {
                "hero_id": "archer_queen", "card_identity_verified": True,
                "deployed": True, "review_status": "reviewed",
                "review_method": "agent_visual", "reviewer": "fixture reviewer",
                "review_evidence": "friendly portrait and enclosed HP inspected separately",
                "source_frame_sha256": digest, "roi_bbox": [0, 0, 16, 16],
                "frame_id": frame_id, "session_id": session_id,
                "camera_epoch": "stable-1", "captured_at_monotonic": 100.0 + index,
                "client_version": "18.647.2"}
        else:
            reviewed_frame["target_id"] = "building:air_defense:one"
        windows.append({"schema_version": 1, "kind": kind,
            "window_id": f"window-{index}", "event_group_id": f"event-{index}",
            "split": "train", "review_status": "reviewed", "reviewer": "fixture reviewer",
            "review_method": "agent_visual", "review_evidence": "event inspected in original pixels",
            "prototype_only": True, "action_kind": "skill_available" if kind == "r_card" else "building_destroyed",
            "truth_event": False, "event_at_monotonic": None, "frames": [reviewed_frame]})
    queue_path = root / "queue.json"
    reviews_path = root / "reviews.jsonl"
    queue = {"schema_version": 1, "client_version": "18.647.2",
             "policy": POLICY, "sessions": queue_sessions, "collections": []}
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    reviews_path.write_text("\n".join(json.dumps(row) for row in windows) + "\n", encoding="utf-8")
    return queue_path, reviews_path, queue, windows


def _run(root: Path, queue: Path, reviews: Path, *, kind: str = "r_card",
         prototype_only: bool = True):
    return import_reviewed_r_windows(queue, reviews, root, root / "frozen",
                                     kind=kind, prototype_only=prototype_only)


def _save(queue_path: Path, reviews_path: Path, queue: dict, windows: list[dict]):
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    reviews_path.write_text("\n".join(json.dumps(row) for row in windows) + "\n", encoding="utf-8")


@pytest.mark.parametrize("kind", ["r_card", "r_building"])
def test_import_freezes_multiple_independent_reviewed_groups(tmp_path, kind):
    queue, reviews, _, _ = _fixture(tmp_path, kind=kind, sessions=2)
    result = _run(tmp_path, queue, reviews, kind=kind)
    assert result["inventory"]["groups"] == 2
    frozen = verify_dynamic_snapshot(tmp_path / "frozen", kind=kind, prototype_only=True)
    assert frozen["metadata"]["windows"] == 2
    assert all(frame["target_id"] == "hero_skill:archer_queen"
               for row in frozen["windows"] for frame in row["frames"]) if kind == "r_card" else True


@pytest.mark.parametrize("mutation", [
    "source_tampered", "state_hash", "state_roi", "missing_state_review",
    "identity_mismatch", "identity_review_unbound", "pending", "unknown_field",
    "candidate_mismatch", "capture_basis", "wrong_session_path",
])
def test_import_rejects_unreviewed_or_unbound_source_without_output(tmp_path, mutation):
    queue_path, reviews_path, queue, windows = _fixture(tmp_path)
    frame = windows[0]["frames"][0]
    source = Path(queue["sessions"][0]["battle_frames"][0]["frame"])
    if mutation == "source_tampered":
        source.write_bytes(b"changed")
    elif mutation == "state_hash":
        frame["state_source_frame_sha256"] = "0" * 64
    elif mutation == "state_roi":
        frame["state_roi_bbox"] = [1, 0, 16, 16]
    elif mutation == "missing_state_review":
        del frame["state_review_evidence"]
    elif mutation == "identity_mismatch":
        frame["hero_deployment_truth"]["hero_id"] = "grand_warden"
    elif mutation == "identity_review_unbound":
        frame["hero_deployment_truth"]["source_frame_sha256"] = "0" * 64
    elif mutation == "pending":
        frame["state_review_status"] = "pending"
    elif mutation == "unknown_field":
        frame["inferred_state"] = "available"
    elif mutation == "candidate_mismatch":
        queue["sessions"][0]["battle_frames"][0]["hero_card_candidates"][0]["candidate_roi_bbox_source"] = [1, 0, 16, 16]
    elif mutation == "capture_basis":
        queue["sessions"][0]["battle_frames"][0]["capture_timestamp_basis"] = None
    elif mutation == "wrong_session_path":
        queue["sessions"][0]["source_session"] = "other-session"
        frame["queue_session"] = "other-session"
        frame["hero_deployment_truth"]["session_id"] = "other-session"
    _save(queue_path, reviews_path, queue, windows)
    with pytest.raises(DatasetError):
        _run(tmp_path, queue_path, reviews_path)
    assert not (tmp_path / "frozen").exists()


def test_import_never_replaces_existing_output(tmp_path):
    queue, reviews, _, _ = _fixture(tmp_path)
    output = tmp_path / "frozen"
    output.mkdir()
    marker = output / "unrelated.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(DatasetError, match="already exists"):
        _run(tmp_path, queue, reviews)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_import_rejects_duplicate_json_field_before_freeze(tmp_path):
    queue, reviews, _, _ = _fixture(tmp_path)
    text = reviews.read_text(encoding="utf-8")
    reviews.write_text(text.replace('"window_id": "window-0",',
                                    '"window_id": "window-0", "window_id": "forged",'),
                       encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate JSON field"):
        _run(tmp_path, queue, reviews)
    assert not (tmp_path / "frozen").exists()


def test_formal_requires_human_review_at_every_level(tmp_path):
    queue, reviews, queue_data, windows = _fixture(tmp_path)
    windows[0]["prototype_only"] = False
    _save(queue, reviews, queue_data, windows)
    with pytest.raises(DatasetError, match="review method"):
        _run(tmp_path, queue, reviews, prototype_only=False)
    assert not (tmp_path / "frozen").exists()
    windows[0]["review_method"] = "human"
    windows[0]["frames"][0]["state_review_method"] = "human"
    windows[0]["frames"][0]["hero_deployment_truth"]["review_method"] = "human"
    _save(queue, reviews, queue_data, windows)
    assert _run(tmp_path, queue, reviews, prototype_only=False)["metadata"]["review_policy"] == "human_only_formal"


def test_connected_windows_cannot_cross_split_or_conflict(tmp_path):
    queue, reviews, queue_data, windows = _fixture(tmp_path)
    second = json.loads(json.dumps(windows[0]))
    second["window_id"], second["event_group_id"], second["split"] = "other", "other", "val"
    windows.append(second)
    _save(queue, reviews, queue_data, windows)
    with pytest.raises(DatasetError, match="crosses train/val/test"):
        _run(tmp_path, queue, reviews)
    assert not (tmp_path / "frozen").exists()
    second["split"] = "train"
    second["frames"][0]["state_label"] = "selected"
    _save(queue, reviews, queue_data, windows)
    with pytest.raises(DatasetError, match="Conflicting R state labels"):
        _run(tmp_path, queue, reviews)
    assert not (tmp_path / "frozen").exists()


def test_building_window_cannot_switch_reviewed_target_between_frames(tmp_path):
    queue_path, reviews_path, queue, windows = _fixture(tmp_path, kind="r_building")
    first_candidate = queue["sessions"][0]["battle_frames"][0]
    second_image = Path(first_candidate["frame"]).with_name("two.png")
    second_image.write_bytes(Path(first_candidate["frame"]).read_bytes())
    second_candidate = json.loads(json.dumps(first_candidate))
    second_candidate.update(frame=str(second_image), frame_id="frame-next",
                            capture_started_monotonic=100.5)
    queue["sessions"][0]["battle_frames"].append(second_candidate)
    second_review = json.loads(json.dumps(windows[0]["frames"][0]))
    second_review.update(queue_frame_id="frame-next", target_id="building:air_defense:two",
                         state_target_id="building:air_defense:two")
    windows[0]["frames"].append(second_review)
    _save(queue_path, reviews_path, queue, windows)
    with pytest.raises(DatasetError, match="building window changes target identity"):
        _run(tmp_path, queue_path, reviews_path, kind="r_building")
    assert not (tmp_path / "frozen").exists()

    second_review["target_id"] = second_review["state_target_id"] = "building:air_defense:one"
    _save(queue_path, reviews_path, queue, windows)
    assert _run(tmp_path, queue_path, reviews_path, kind="r_building")["inventory"]["windows"] == 1
