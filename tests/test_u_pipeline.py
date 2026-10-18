"""Synthetic-only contracts for the repository U candidate pipeline.

These tests use generated images only and never access real datasets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from training.vision.data import DatasetError
from training.vision.u_candidates import CandidateError, build_candidate_queue, load_candidate_queue
from training.vision.u_review import UReviewJournal, _digest, authorize_local_request
from training.vision.u_import import import_reviewed_u


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path.read_bytes()


def _source(tmp_path, *, count=1, split="train", blockers=False,
            session_id="session-1", event_group_id="battle-group-1"):
    now = datetime.now(timezone.utc)
    source_root = tmp_path / "run"
    (source_root / "frames").mkdir(parents=True)
    frame_records = []
    planned = []
    observations = []
    image_bytes = []
    for n in range(count):
        frame_id = f"frame-{n + 1}"
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[20:60, 35:70] = (10 + n, 30, 90)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        raw = encoded.tobytes()
        image_bytes.append(raw)
        frame_path = source_root / "frames" / f"{frame_id}.png"
        frame_path.write_bytes(raw)
        roi = [0, 0, 128, 96]
        planned.append({"frame_id": frame_id, "roi_bbox": roi})
        captured = now - timedelta(seconds=15 - n)
        mono = 100.0 + n
        frame_records.append({
            "frame_id": frame_id,
            "image": f"frames/{frame_id}.png",
            "sha256": _sha(raw),
            "width": 128,
            "height": 96,
            "captured_at_utc": captured.isoformat(),
            "captured_at_monotonic": mono,
            "camera_epoch": "epoch-1",
            "client_version": "18.647.2",
            "camera_stable": not blockers,
            "zoom_verified": True,
            "battle_id": "battle-1",
            "layout_id": "layout-1",
            "roi_bbox": roi,
        })
        observations.append({
            "time": 1.0 + n,
            "kind": "observation",
            "frame": str(frame_path),
            "observations": {
                "frame_id": frame_id,
                "observed_at_monotonic": mono,
                "captured_at_utc": captured.isoformat(),
                "capture_timestamp_basis": "capture_request_start",
            },
        })
    events_raw = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in observations)
    events_path = source_root / "events.jsonl"
    events_path.write_bytes(events_raw)
    plan = {
        "schema": "u_capture_plan_v1",
        "schema_version": 1,
        "kind": "u_position",
        "plan_id": f"plan-{session_id}",
        "created_at_utc": (now - timedelta(minutes=1)).isoformat(),
        "session_id": session_id,
        "event_group_id": event_group_id,
        "split": split,
        "battle_id": "battle-1",
        "layout_id": "layout-1",
        "client_version": "18.647.2",
        "planned_frames": planned,
    }
    plan_path = tmp_path / "plan.json"
    plan_raw = _json(plan_path, plan)
    capture = {
        "schema": "u_capture_manifest_v1",
        "schema_version": 1,
        "plan_sha256": _sha(plan_raw),
        "session_id": session_id,
        "source_events_path": "events.jsonl",
        "source_events_sha256": _sha(events_raw),
        "frames": frame_records,
    }
    capture_path = tmp_path / "capture.json"
    _json(capture_path, capture)
    batch_path = tmp_path / "batch.json"
    _json(batch_path, {
        "schema": "u_candidate_batch_v1",
        "schema_version": 1,
        "sources": [{
            "source_root": str(source_root),
            "plan_path": str(plan_path),
            "capture_manifest_path": str(capture_path),
        }],
    })
    return source_root, batch_path, image_bytes


def _queue(tmp_path, **kwargs):
    _, batch, _ = _source(tmp_path, **kwargs)
    output = tmp_path / "candidate-pool"
    build_candidate_queue(batch, output)
    return output / "queue.json"


def test_predeclared_source_plan_and_exact_original_bytes_are_bound(tmp_path):
    queue_path = _queue(tmp_path, count=2)
    queue, pool, digest = load_candidate_queue(queue_path)
    assert len(queue["frames"]) == 2
    assert all((pool / row["image"]).read_bytes() ==
               (Path(row["source_root"]) / row["source_image_path"]).read_bytes()
               for row in queue["frames"])
    assert len(digest) == 64
    assert {row["split"] for row in queue["frames"]} == {"train"}


def test_queue_cannot_omit_a_whole_source_from_its_batch(tmp_path):
    _, batch_a, _ = _source(tmp_path / "a", session_id="session-a",
                            event_group_id="group-a")
    _, batch_b, _ = _source(tmp_path / "b", session_id="session-b",
                            event_group_id="group-b")
    combined = json.loads(batch_a.read_text(encoding="utf-8"))
    combined["sources"].extend(json.loads(batch_b.read_text(encoding="utf-8"))["sources"])
    batch = tmp_path / "combined-batch.json"
    _json(batch, combined)
    output = tmp_path / "combined-pool"
    build_candidate_queue(batch, output)
    queue_path = output / "queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    omitted_source = queue["sources"].pop()
    queue["frames"] = [row for row in queue["frames"]
                        if row["plan_id"] != omitted_source["plan_id"]]
    queue["counts"] = {
        "candidate_frames": len(queue["frames"]), "groups": 1,
        "groups_by_split": {"train": 1, "val": 0, "test": 0},
        "metadata_blocked": 0,
    }
    queue_path.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")
    with pytest.raises(CandidateError, match="omitted or reordered a predeclared batch source"):
        load_candidate_queue(queue_path)


def test_existing_output_and_test_plan_are_not_reassigned(tmp_path):
    _, batch, _ = _source(tmp_path)
    output = tmp_path / "candidate-pool"
    build_candidate_queue(batch, output)
    with pytest.raises(CandidateError, match="overwrite"):
        build_candidate_queue(batch, output)
    _, test_batch, _ = _source(tmp_path / "second", split="test")
    test_output = tmp_path / "second" / "candidate-pool"
    build_candidate_queue(test_batch, test_output)
    test_queue, _, _ = load_candidate_queue(test_output / "queue.json")
    assert {row["split"] for row in test_queue["frames"]} == {"test"}


def test_metadata_blocker_cannot_be_reviewed_complete(tmp_path):
    queue_path = _queue(tmp_path, blockers=True)
    queue, _, _ = load_candidate_queue(queue_path)
    frame = queue["frames"][0]
    assert "camera_not_verified_stable" in frame["import_blockers"]
    journal = UReviewJournal(queue_path, tmp_path / "review.jsonl", prototype_only=True)
    with pytest.raises(DatasetError, match="metadata"):
        journal.append_human(frame["queue_frame_id"], decision="complete",
                             boxes=[], exhaustive=True, reviewer="Reviewer")
    event = journal.append_human(frame["queue_frame_id"], decision="quarantine",
                                 reviewer="Reviewer", quarantine_reason="Camera evidence missing")
    assert event["decision"] == "quarantine"


def test_complete_empty_requires_exhaustive_roi_confirmation(tmp_path):
    queue_path = _queue(tmp_path)
    queue, _, _ = load_candidate_queue(queue_path)
    journal = UReviewJournal(queue_path, tmp_path / "review.jsonl", prototype_only=True)
    frame_id = queue["frames"][0]["queue_frame_id"]
    with pytest.raises(DatasetError, match="exhaustive"):
        journal.append_human(frame_id, decision="complete", boxes=[],
                             exhaustive=False, reviewer="Reviewer")
    event = journal.append_human(frame_id, decision="complete", boxes=[],
                                 exhaustive=True, reviewer="Reviewer")
    assert event["boxes"] == []
    assert event["review_method"] == "human"


def test_unknown_identity_and_out_of_roi_box_are_rejected(tmp_path):
    queue_path = _queue(tmp_path)
    queue, _, _ = load_candidate_queue(queue_path)
    journal = UReviewJournal(queue_path, tmp_path / "review.jsonl", prototype_only=True)
    frame_id = queue["frames"][0]["queue_frame_id"]
    with pytest.raises(DatasetError, match="identity"):
        journal.append_human(frame_id, decision="complete",
            boxes=[{"label": "hero:made_up", "bbox": [1, 1, 10, 10]}],
            exhaustive=True, reviewer="Reviewer")
    with pytest.raises(DatasetError, match="bounds"):
        journal.append_human(frame_id, decision="complete",
            boxes=[{"label": "troop_group", "bbox": [-1, 1, 10, 10]}],
            exhaustive=True, reviewer="Reviewer")


def test_agent_visual_stays_explicit_and_is_prototype_only(tmp_path):
    queue_path = _queue(tmp_path)
    queue, _, _ = load_candidate_queue(queue_path)
    frame_id = queue["frames"][0]["queue_frame_id"]
    formal = UReviewJournal(queue_path, tmp_path / "formal.jsonl", prototype_only=False)
    with pytest.raises(DatasetError, match="prototype-only"):
        formal.append_agent_visual(frame_id, decision="complete",
            boxes=[{"label": "hero:grand_warden", "bbox": [10, 10, 30, 35]}],
            exhaustive=True, reviewer="agent", evidence="Direct visual inspection")
    prototype = UReviewJournal(queue_path, tmp_path / "agent.jsonl", prototype_only=True)
    event = prototype.append_agent_visual(frame_id, decision="complete",
        boxes=[{"label": "hero:grand_warden", "bbox": [10, 10, 30, 35]}],
        exhaustive=True, reviewer="agent", evidence="Direct visual inspection")
    assert event["review_method"] == "agent_visual"
    assert event["reviewer"] == "agent"


def test_review_state_restores_last_boxes_instead_of_replacing_them_with_empty(tmp_path):
    queue_path = _queue(tmp_path)
    queue, _, _ = load_candidate_queue(queue_path)
    frame_id = queue["frames"][0]["queue_frame_id"]
    journal = UReviewJournal(queue_path, tmp_path / "review.jsonl", prototype_only=True)
    boxes = [{"label": "hero:grand_warden", "bbox": [10, 10, 30, 35]}]
    journal.append_human(frame_id, decision="complete", boxes=boxes,
                         exhaustive=True, reviewer="Reviewer")
    latest = journal.state()["frames"][0]["latest"]
    assert latest["decision"] == "complete"
    assert latest["boxes"] == boxes
    assert latest["exhaustive"] is True


def test_journal_rejects_wrong_source_frame_id_even_with_recomputed_row_hash(tmp_path):
    queue_path = _queue(tmp_path)
    queue, _, _ = load_candidate_queue(queue_path)
    frame_id = queue["frames"][0]["queue_frame_id"]
    journal_path = tmp_path / "review.jsonl"
    journal = UReviewJournal(queue_path, journal_path, prototype_only=True)
    journal.append_human(frame_id, decision="complete", boxes=[], exhaustive=True,
                         reviewer="Reviewer")
    event = json.loads(journal_path.read_text(encoding="utf-8"))
    event["source_frame_id"] = "another-frame"
    event["event_sha256"] = _digest({k: v for k, v in event.items() if k != "event_sha256"})
    journal_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="source frame/ROI"):
        UReviewJournal(queue_path, journal_path, prototype_only=True)


def test_quarantine_is_not_imported_as_empty_negative_and_missing_review_fails(tmp_path):
    queue_path = _queue(tmp_path, count=2)
    queue, source_pool, _ = load_candidate_queue(queue_path)
    journal_path = tmp_path / "review.jsonl"
    journal = UReviewJournal(queue_path, journal_path, prototype_only=True)
    first, second = [row["queue_frame_id"] for row in queue["frames"]]
    journal.append_human(first, decision="quarantine", reviewer="Reviewer",
                         quarantine_reason="Hero identity is obscured")
    with pytest.raises(DatasetError, match="Every candidate"):
        journal.review_rows()
    journal.append_human(second, decision="complete", boxes=[],
                         exhaustive=True, reviewer="Reviewer")
    reviews_path = tmp_path / "reviews.jsonl"
    journal.export_reviews(reviews_path)
    with pytest.raises(DatasetError, match="separate from the immutable candidate source pool"):
        import_reviewed_u(queue_path, journal_path, reviews_path,
                          source_pool / "nested-snapshot", prototype_only=True)
    snapshot = tmp_path / "snapshot"
    result = import_reviewed_u(queue_path, journal_path, reviews_path, snapshot,
                               prototype_only=True)
    assert result["window_count"] == 1
    assert result["quarantined_frames"][0]["queue_frame_id"] == first
    assert result["formal_acceptance"] is False


def test_loopback_server_guard_rejects_remote_origin_host_and_token():
    args = {"expected_token": "secret", "port": 8765, "require_origin": True}
    assert authorize_local_request(host="127.0.0.1:8765",
        origin="http://127.0.0.1:8765", token="secret", **args)
    assert not authorize_local_request(host="localhost:8765",
        origin="http://127.0.0.1:8765", token="secret", **args)
    assert not authorize_local_request(host="127.0.0.1:8765",
        origin="http://evil.invalid", token="secret", **args)
    assert not authorize_local_request(host="127.0.0.1:8765",
        origin="http://127.0.0.1:8765", token="wrong", **args)
