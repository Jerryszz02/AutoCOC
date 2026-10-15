import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.vision_review_queue import _hash, review_queue


def test_review_queue_hashes_originals_and_never_fills_state_truth():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        live = root / "live-1"
        (live / "frames").mkdir(parents=True)
        frame = live / "frames" / "001.png"
        frame.write_bytes(b"original pixels")
        collection = root / "collection"
        (collection / "images").mkdir(parents=True)
        sample = collection / "images" / "one.png"
        sample.write_bytes(b"collection pixels")
        rows = [
            {"kind": "session_connected", "client_version": "18.647.2"},
            {"kind": "observation", "time": 1792088849.0, "scene": "battle", "frame": str(frame),
             "observations": {"source_resolution": [2560, 1440], "baseline_resolution": [1280, 720],
                              "frame_id": "frame-1", "requested_at_monotonic": 100.0,
                              "battle": {"phase": "active", "slots": [
                                  {"kind": "hero", "unit_id": "archer_queen", "bbox": [10, 600, 100, 700],
                                   "confidence": .99}]}}},
        ]
        (live / "events.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
        manifest = collection / "unreviewed-manifest.jsonl"
        manifest.write_text(json.dumps({"sample_id": "C1", "client_version": "18.647.2",
            "image_path": "images/one.png", "file_sha256": _hash(sample),
            "source_session": "collection-1", "battle_id": "battle-1",
            "leakage_group_id": "collection:collection-1"}) + "\n", encoding="utf-8")
        annotation = root / "reviewed.jsonl"
        annotation.write_text(json.dumps({"sample_id": "C1", "client_version": "18.647.2",
            "prototype_only": True, "source_sha256": _hash(sample), "buildings": {
                "complete": True, "review_status": "reviewed", "review_method": "agent_visual",
                "reviewer": "fixture", "review_evidence": "original inspected"}}) + "\n", encoding="utf-8")
        report = review_queue([live], [manifest], client_version="18.647.2",
                              annotation_files=[annotation])
        source = report["sessions"][0]["battle_frames"][0]
        assert source["sha256"] == _hash(frame)
        assert source["capture_started_utc"] is None
        assert source["capture_timestamp_basis"] is None
        assert source["camera_epoch"] is None
        assert source["event_truth"] is None
        assert source["hero_card_candidates"] == [{"target_id": "hero-card:archer_queen",
            "candidate_roi_bbox_source": [20, 1200, 200, 1400], "identity_confidence": .99,
            "state_label": None, "state_review_status": "pending"}]
        assert report["collections"][0]["samples"][0]["scene_label"] is None
        assert report["collections"][0]["samples"][0]["review_status"] == "agent_visual_reviewed_prototype"
        assert report["collections"][0]["samples"][0]["human_review_status"] == "pending"
        assert report["collections"][0]["samples"][0]["building_review"]["complete"] is True
        assert report["collections"][0]["samples"][0]["building_review"]["building_scope"] is None
        sample.write_bytes(b"changed")
        with pytest.raises(ValueError, match="hash mismatch"):
            review_queue([live], [manifest], client_version="18.647.2", annotation_files=[annotation])
        with pytest.raises(ValueError, match="version"):
            review_queue([live], [], client_version="18.600.7")


def test_review_queue_only_accepts_bound_capture_request_start_utc():
    with TemporaryDirectory() as temporary:
        live = Path(temporary) / "live"
        live.mkdir()
        frame = live / "one.png"
        frame.write_bytes(b"source")
        connected = {"kind": "session_connected", "client_version": "18.647.2"}
        observed = {"kind": "observation", "time": 1792088849.0, "scene": "battle",
                    "frame": str(frame), "observations": {"captured_at_utc": "2026-10-15T18:00:00+00:00",
                    "capture_timestamp_basis": "capture_request_start"}}
        events = live / "events.jsonl"

        def read():
            events.write_text("\n".join(map(json.dumps, [connected, observed])) + "\n", encoding="utf-8")
            return review_queue([live], [], client_version="18.647.2")["sessions"][0]["battle_frames"][0]

        assert read()["capture_started_utc"] == "2026-10-15T18:00:00+00:00"
        observed["observations"]["capture_timestamp_basis"] = "event_log_time"
        assert read()["capture_started_utc"] is None
        observed["observations"]["capture_timestamp_basis"] = "capture_request_start"
        observed["observations"]["captured_at_utc"] = "not-a-time"
        assert read()["capture_started_utc"] is None


def test_review_queue_preserves_explicit_partial_building_scope():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        image = root / "one.png"
        image.write_bytes(b"reviewed source")
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps({"sample_id": "C1", "client_version": "18.647.2",
            "image_path": "one.png", "file_sha256": _hash(image)}) + "\n", encoding="utf-8")
        annotation = root / "reviewed.jsonl"
        row = {"sample_id": "C1", "client_version": "18.647.2", "prototype_only": True,
               "source_sha256": _hash(image), "buildings": {
                   "complete": False, "complete_classes": ["town_hall", "air_defense"],
                   "review_status": "reviewed", "review_method": "agent_visual",
                   "reviewer": "fixture", "review_evidence": "original inspected"}}

        def read():
            annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
            return review_queue([], [manifest], client_version="18.647.2",
                                annotation_files=[annotation])["collections"][0]["samples"][0]

        building = read()["building_review"]
        assert building["complete"] is False
        assert building["complete_classes"] == ["town_hall", "air_defense"]
        assert building["building_scope"] == {"classes": ["town_hall", "air_defense"],
                                              "stable_ids": {"town_hall": 1, "air_defense": 3},
                                              "output_indices": {"town_hall": 0, "air_defense": 1}}
        assert read()["human_review_status"] == "pending"
        for invalid in ([], ["town_hall", "town_hall"], ["invented"], "town_hall"):
            row["buildings"]["complete_classes"] = invalid
            with pytest.raises(ValueError, match="scope"):
                read()
        row["buildings"].update(complete=False, complete_classes=None)
        with pytest.raises(ValueError, match="scope"):
            read()
        row["buildings"].update(complete=False, complete_classes=["town_hall", "air_defense"])
        row["source_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="source hash"):
            read()
