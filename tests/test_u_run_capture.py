"""Synthetic metadata and byte-binding checks for a normal-run U queue.

No live run, game image, model, OCR engine, or device is accessed here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
from training.vision.u_candidates import CandidateError, load_candidate_queue
from training.vision.u_run_capture import produce_run_candidate_queue
from training.vision.u_adjudication import adjudicate_capture_metadata


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    return path


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _png(path: Path, *, width: int = 1, height: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixel_row = b"\x00" + b"\xff\x00\x00" * width
    pixels = pixel_row * height
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
                     + _chunk(b"IDAT", zlib.compress(pixels))
                     + _chunk(b"IEND", b""))
    return path


def _fixture(tmp_path: Path, *, naive_capture: bool = False,
             missing_request: bool = False) -> tuple[dict, Path]:
    run_id = "synthetic-normal-run"
    source = tmp_path / "live" / run_id
    first = _png(source / "frames" / "00001-battle.png")
    second = _png(source / "frames" / "00002-battle.png")
    home = _png(source / "frames" / "00003-home.png")
    now = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)
    events = []
    for number, (path, scene) in enumerate(((first, "battle"), (second, "battle"),
                                             (home, "village")), 1):
        utc = (now + timedelta(seconds=number)).isoformat()
        if naive_capture and number == 1:
            utc = utc.removesuffix("+00:00")
        events.append({
            "kind": "observation", "scene": scene, "frame": str(path),
            "observations": {
                "frame_id": f"source-frame-{number}",
                "source_resolution": [1, 1],
                "observed_at_monotonic": float(number),
                "requested_at_monotonic": (None if missing_request and number == 1
                                           else float(number) + .001),
                "capture_timestamp_basis": "capture_request_start",
                "captured_at_utc": utc,
            },
        })
    event_path = source / "events.jsonl"
    event_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
                          encoding="utf-8")
    source_hash = "a" * 64
    source_files = {"src/autococ/example.py": source_hash}
    report_path = _json(tmp_path / "live" / f"run-{run_id}.json", {
        "run_id": run_id, "events_path": str(event_path),
        "started_at": "2026-10-17T11:00:00",
        "provenance": {"captured_at": "2026-10-17T10:59:59",
                       "client_version": "18.647.2",
                       "source_fingerprint_sha256": "b" * 64,
                       "effective_config_sha256": "c" * 64,
                       "files_sha256": source_files},
    })
    binding_path = _json(tmp_path / "binding.json", {
        "initial_modules_unchanged": True, "all_loaded_sources_match_snapshot": True,
        "modules_after": {"autococ.example": {
            "source_relative": "src/autococ/example.py", "sha256": source_hash}},
    })
    launch_path = _json(tmp_path / "launch.json", {
        "commit": "d" * 40, "origin_commit": "d" * 40, "head_after": "d" * 40,
        "clean": True, "source_files_unchanged": True,
        "external_files_unchanged": True, "evidence_unchanged": True,
        "runtime_binding_verified": True, "normal_cli": True,
        "raw_sampler_installed": False,
        "source_files": source_files,
        "runtime_binding_path": str(binding_path),
        "runtime_binding_sha256": _sha(binding_path),
        "external_files": {},
        "report_candidates": [str(report_path)],
        "started_at_utc": (now - timedelta(seconds=30)).isoformat(),
    })
    prospective_path = _json(tmp_path / "prospective.json", {
        "schema": "prospective_continuous_daily_train_candidate_v1",
        "created_at_utc": (now - timedelta(minutes=1)).isoformat(),
        "source_head": "d" * 40, "split_role": "train_candidate",
        "declared_before_any_new_session_pixels": True,
        "human_review": False, "admitted_to_training": False,
        "normal_cli": True, "raw_sampler_installed": False,
        "scope": "synthetic whole-run TRAIN candidates",
        "candidate_uses": ["U existing post-batch frames"],
    })
    return {
        "report": report_path, "launch": launch_path,
        "binding": binding_path, "prospective": prospective_path,
        "events": event_path, "source": source,
        "first": first, "second": second, "home": home,
    }, tmp_path / "pending-run"


def _produce(args: dict, output: Path, *, manifest_only: bool = False) -> dict:
    return produce_run_candidate_queue(args["report"], args["launch"], args["binding"],
                                       args["prospective"], output,
                                       manifest_only=manifest_only)


def _bind_prospective_to_launch_and_child(inputs: dict) -> None:
    launch = json.loads(inputs["launch"].read_text(encoding="utf-8"))
    prospective = json.loads(inputs["prospective"].read_text(encoding="utf-8"))
    scope = {
        "path": str(inputs["prospective"]), "sha256": _sha(inputs["prospective"]),
        "schema": prospective["schema"], "source_head": prospective["source_head"],
        "created_at_utc": prospective["created_at_utc"],
        "split_role": prospective["split_role"], "scope": prospective["scope"],
        "bound_before_capture": True,
    }
    binding = json.loads(inputs["binding"].read_text(encoding="utf-8"))
    binding["prospective_plan_verified_before_cli"] = True
    binding["prospective_plan"] = scope
    _json(inputs["binding"], binding)
    launch["runtime_binding_sha256"] = _sha(inputs["binding"])
    launch["external_files"] = {str(inputs["prospective"]): _sha(inputs["prospective"])}
    launch["prospective_plan"] = scope
    launch["prelaunch_scope_binding_verified"] = True
    _json(inputs["launch"], launch)


def _development_val_scope(inputs: dict) -> None:
    prospective = json.loads(inputs["prospective"].read_text(encoding="utf-8"))
    prospective.update(schema="prospective_u_normal_development_val_v1",
                       split_role="development_validation",
                       validation_purpose="development_validation",
                       independent_final_heldout=False,
                       scope="synthetic whole-run U development validation")
    _json(inputs["prospective"], prospective)
    launch = json.loads(inputs["launch"].read_text(encoding="utf-8"))
    launch.update(kind="normal_u_development_val_candidate",
                  split_role="development_validation",
                  independent_final_heldout=False,
                  max_battles=1, max_searches_per_battle=3)
    _json(inputs["launch"], launch)


def test_real_schema_producer_keeps_source_ids_bytes_raw_times_and_pending_only(tmp_path):
    inputs, output = _fixture(tmp_path)
    result = _produce(inputs, output)
    queue, pool, digest = load_candidate_queue(output / "queue" / "queue.json")
    assert result["pending_only"] is True and result["training_ready"] is False
    assert result["human_review"] is False and result["formal_acceptance"] is False
    assert len(digest) == 64 and len(queue["frames"]) == 2
    assert queue["sources"][0]["planned_frame_ids"] == ["source-frame-1", "source-frame-2"]
    assert {item["split"] for item in queue["frames"]} == {"train"}
    assert {item["width"] for item in queue["frames"]} == {1}
    assert {item["height"] for item in queue["frames"]} == {1}
    assert [item["captured_at_monotonic"] for item in queue["frames"]] == [1.001, 2.001]
    assert [item["observed_at_monotonic"] for item in queue["frames"]] == [1.0, 2.0]
    assert [item["requested_at_monotonic"] for item in queue["frames"]] == [1.001, 2.001]
    assert queue["frames"][0]["captured_at_utc"] == "2026-10-17T10:00:01+00:00"
    for row in queue["frames"]:
        assert row["review_status"] == "pending" and row["boxes"] is None
        assert row["identity_labels"] is None and row["roi_bbox"] is None
        assert "source_roi_not_independently_reviewed" in row["import_blockers"]
        assert "prelaunch_scope_binding_unverified" in row["import_blockers"]
        assert "camera_epoch_unverified" in row["import_blockers"]
        assert "battle_or_layout_identity_unverified" in row["import_blockers"]
        assert (pool / row["image"]).read_bytes() == (
            inputs["source"] / row["source_image_path"]).read_bytes()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["report_started_at_raw"] == "2026-10-17T11:00:00"
    assert plan["report_provenance_captured_at_raw"] == "2026-10-17T10:59:59"
    assert plan["prelaunch_scope_binding_verified"] is False
    assert plan["frame_selection"]["frame_ids_discovered_after_run"] is True
    assert all(item["roi_bbox"] is None for item in plan["planned_frames"])
    assert inputs["home"].name not in {row["source_image_path"] for row in queue["frames"]}


def test_naive_event_utc_is_retained_verbatim_but_blocked(tmp_path):
    inputs, output = _fixture(tmp_path, naive_capture=True)
    _produce(inputs, output)
    queue, _, _ = load_candidate_queue(output / "queue" / "queue.json")
    assert queue["frames"][0]["captured_at_utc"] == "2026-10-17T10:00:01"
    assert "source_capture_utc_timezone_unverified" in queue["frames"][0]["import_blockers"]


def test_missing_request_time_is_not_replaced_by_observed_time(tmp_path):
    inputs, output = _fixture(tmp_path, missing_request=True)
    _produce(inputs, output)
    queue, _, _ = load_candidate_queue(output / "queue" / "queue.json")
    first = queue["frames"][0]
    assert first["observed_at_monotonic"] == 1.0
    assert first["requested_at_monotonic"] is None
    assert first["captured_at_monotonic"] is None
    assert "capture_request_monotonic_missing" in first["import_blockers"]


def test_future_explicit_launch_binding_of_prospective_is_revalidated(tmp_path):
    inputs, output = _fixture(tmp_path)
    _bind_prospective_to_launch_and_child(inputs)
    launch = json.loads(inputs["launch"].read_text(encoding="utf-8"))
    _produce(inputs, output)
    queue, _, _ = load_candidate_queue(output / "queue" / "queue.json")
    assert all("prelaunch_scope_binding_unverified" not in row["import_blockers"]
               for row in queue["frames"])
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["prelaunch_scope_binding_verified"] is True
    launch["external_files"][str(inputs["prospective"])] = "0" * 64
    _json(inputs["launch"], launch)
    with pytest.raises(CandidateError, match="source changed"):
        load_candidate_queue(output / "queue" / "queue.json")


def test_normal_run_development_val_is_full_run_bound_and_pending(tmp_path):
    inputs, output = _fixture(tmp_path)
    _development_val_scope(inputs)
    _bind_prospective_to_launch_and_child(inputs)
    result = _produce(inputs, output)
    queue, _, _ = load_candidate_queue(output / "queue" / "queue.json")
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert result["split"] == plan["split"] == "val"
    assert plan["prelaunch_scope_binding_verified"] is True
    assert plan["event_group_id"] == "u-run:synthetic-normal-run"
    assert queue["sources"][0]["planned_frame_ids"] == ["source-frame-1", "source-frame-2"]
    assert {row["split"] for row in queue["frames"]} == {"val"}
    assert all(row["review_status"] == "pending" and row["boxes"] is None
               and row["identity_labels"] is None and row["import_blockers"]
               for row in queue["frames"])
    assert inputs["home"].name not in {row["source_image_path"] for row in queue["frames"]}


def test_development_val_rejects_unbound_or_misdeclared_scope(tmp_path):
    inputs, output = _fixture(tmp_path)
    _development_val_scope(inputs)
    with pytest.raises(CandidateError, match="prelaunch parent/child"):
        _produce(inputs, output)
    assert not output.exists()
    _bind_prospective_to_launch_and_child(inputs)
    prospective = json.loads(inputs["prospective"].read_text(encoding="utf-8"))
    prospective["independent_final_heldout"] = True
    _json(inputs["prospective"], prospective)
    with pytest.raises(CandidateError, match="final heldout"):
        _produce(inputs, output)
    prospective["independent_final_heldout"] = False
    prospective["split_role"] = "train_candidate"
    _json(inputs["prospective"], prospective)
    with pytest.raises(CandidateError, match="schema/split role"):
        _produce(inputs, output)
    assert not output.exists()


def test_source_drift_is_rejected_after_queue_generation(tmp_path):
    inputs, output = _fixture(tmp_path)
    _produce(inputs, output)
    inputs["first"].write_bytes(inputs["first"].read_bytes() + b"changed")
    with pytest.raises(CandidateError, match="source changed|image SHA mismatch"):
        load_candidate_queue(output / "queue" / "queue.json")


def test_prospective_or_binding_drift_refuses_production(tmp_path):
    inputs, output = _fixture(tmp_path)
    prospective = json.loads(inputs["prospective"].read_text(encoding="utf-8"))
    prospective["declared_before_any_new_session_pixels"] = False
    _json(inputs["prospective"], prospective)
    with pytest.raises(CandidateError, match="Prospective U"):
        _produce(inputs, output)
    assert not output.exists()
    prospective["declared_before_any_new_session_pixels"] = True
    _json(inputs["prospective"], prospective)
    binding = json.loads(inputs["binding"].read_text(encoding="utf-8"))
    binding["modules_after"]["autococ.example"]["sha256"] = "0" * 64
    _json(inputs["binding"], binding)
    with pytest.raises(CandidateError, match="exact report/runtime|Loaded module"):
        _produce(inputs, output)
    assert not output.exists()


def test_duplicate_battle_id_or_native_header_mismatch_refuses_all(tmp_path):
    inputs, output = _fixture(tmp_path)
    rows = [json.loads(line) for line in inputs["events"].read_text(encoding="utf-8").splitlines()]
    rows[1]["observations"]["frame_id"] = rows[0]["observations"]["frame_id"]
    inputs["events"].write_text("".join(json.dumps(row) + "\n" for row in rows),
                               encoding="utf-8")
    with pytest.raises(CandidateError, match="duplicated"):
        _produce(inputs, output)
    assert not output.exists()
    rows[1]["observations"]["frame_id"] = "source-frame-2"
    rows[0]["observations"]["source_resolution"] = [2, 1]
    inputs["events"].write_text("".join(json.dumps(row) + "\n" for row in rows),
                               encoding="utf-8")
    with pytest.raises(CandidateError, match="native PNG dimensions"):
        _produce(inputs, output)
    assert not output.exists()


def test_existing_output_refused_and_manifest_only_has_no_queue(tmp_path):
    inputs, output = _fixture(tmp_path)
    _produce(inputs, output, manifest_only=True)
    assert (output / "plan.json").is_file() and (output / "capture.json").is_file()
    assert not (output / "queue").exists()
    with pytest.raises(CandidateError, match="new directory"):
        _produce(inputs, output)


def _adjudication(tmp_path: Path, queue_path: Path) -> tuple[Path, dict]:
    parent, _, parent_sha = load_candidate_queue(queue_path)
    evidence_root = tmp_path / "evidence"
    first = parent["frames"][0]
    values = {"roi_bbox": [0, 0, 1, 1], "camera_epoch": "camera-1",
              "camera_stable": True, "zoom_verified": True,
              "battle_layout": {"battle_id": "battle-1", "layout_id": "layout-1"}}
    references = {}
    for fact, value in values.items():
        path = _json(evidence_root / f"{fact}.json", {
            "schema": "u_capture_fact_evidence_v1", "schema_version": 1,
            "parent_queue_sha256": parent_sha,
            "queue_frame_id": first["queue_frame_id"],
            "source_frame_id": first["frame_id"],
            "source_image_sha256": first["source_image_sha256"],
            "fact": fact, "value": value, "review_method": "agent_visual",
            "reviewer": "synthetic-reviewer", "basis": f"synthetic {fact} evidence",
            "created_at_utc": "2026-10-17T11:00:00+00:00",
        })
        references[fact] = {"path": str(path), "sha256": _sha(path)}
    decisions = {
        "schema": "u_capture_adjudication_v1", "schema_version": 1,
        "parent_queue_sha256": parent_sha, "review_method": "agent_visual",
        "reviewer": "synthetic-reviewer",
        "created_at_utc": "2026-10-17T11:00:01+00:00",
        "frames": [
            {"queue_frame_id": first["queue_frame_id"],
             "source_image_sha256": first["source_image_sha256"],
             "decision": "admit", "reason": None, "evidence": references},
            {"queue_frame_id": parent["frames"][1]["queue_frame_id"],
             "source_image_sha256": parent["frames"][1]["source_image_sha256"],
             "decision": "quarantine", "reason": "synthetic unavailable evidence",
             "evidence": None},
        ],
    }
    path = _json(tmp_path / "decisions.json", decisions)
    return path, decisions


def test_adjudication_preserves_every_parent_and_only_resolves_bound_facts(tmp_path):
    inputs, output = _fixture(tmp_path)
    _bind_prospective_to_launch_and_child(inputs)
    _produce(inputs, output)
    parent_path = output / "queue" / "queue.json"
    parent_bytes = parent_path.read_bytes()
    decisions_path, _ = _adjudication(tmp_path, parent_path)
    derived_path = tmp_path / "derived"
    result = adjudicate_capture_metadata(parent_path, decisions_path, derived_path)
    derived, pool, _ = load_candidate_queue(derived_path / "queue.json")
    assert result["counts"]["admitted"] == 1
    assert result["counts"]["quarantined"] == 1
    assert len(derived["frames"]) == 2 and parent_path.read_bytes() == parent_bytes
    assert derived["human_review"] is False and derived["training_ready"] is False
    assert derived["formal_acceptance"] is False and derived["prototype_only"] is True
    first, second = derived["frames"]
    assert first["import_blockers"] == []
    assert first["battle_id"] == "battle-1" and first["layout_id"] == "layout-1"
    assert first["roi_bbox"] == [0, 0, 1, 1]
    assert first["review_status"] == "pending" and first["identity_labels"] is None
    assert second["import_blockers"][-1] == "capture_metadata_quarantined"
    assert (pool / first["image"]).read_bytes() == inputs["first"].read_bytes()
    from training.vision.data import DatasetError
    from training.vision.u_import import import_reviewed_u
    with pytest.raises(DatasetError, match="prototype snapshots only"):
        import_reviewed_u(derived_path / "queue.json", tmp_path / "missing-journal",
                          tmp_path / "missing-reviews", tmp_path / "formal",
                          prototype_only=False)


def test_adjudication_cannot_clear_missing_prelaunch_scope(tmp_path):
    inputs, output = _fixture(tmp_path)
    _produce(inputs, output)
    parent_path = output / "queue" / "queue.json"
    decisions_path, _ = _adjudication(tmp_path, parent_path)
    derived_path = tmp_path / "derived"
    adjudicate_capture_metadata(parent_path, decisions_path, derived_path)
    derived, _, _ = load_candidate_queue(derived_path / "queue.json")
    assert "prelaunch_scope_binding_unverified" in derived["frames"][0]["import_blockers"]
    assert "capture_metadata_quarantined" in derived["frames"][1]["import_blockers"]


def test_adjudication_requires_all_dispositions_and_rejects_human_claim(tmp_path):
    inputs, output = _fixture(tmp_path)
    _produce(inputs, output)
    parent_path = output / "queue" / "queue.json"
    decisions_path, decisions = _adjudication(tmp_path, parent_path)
    decisions["frames"].pop()
    _json(decisions_path, decisions)
    with pytest.raises(CandidateError, match="Every parent frame"):
        adjudicate_capture_metadata(parent_path, decisions_path, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()
    decisions_path, decisions = _adjudication(tmp_path, parent_path)
    decisions["review_method"] = "human"
    _json(decisions_path, decisions)
    with pytest.raises(CandidateError, match="agent-visual"):
        adjudicate_capture_metadata(parent_path, decisions_path, tmp_path / "derived")


def test_adjudication_fact_binding_and_new_directory_are_strict(tmp_path):
    inputs, output = _fixture(tmp_path)
    _produce(inputs, output)
    parent_path = output / "queue" / "queue.json"
    decisions_path, decisions = _adjudication(tmp_path, parent_path)
    refs = decisions["frames"][0]["evidence"]
    refs["roi_bbox"] = refs["camera_epoch"]
    _json(decisions_path, decisions)
    with pytest.raises(CandidateError, match="source binding|two facts"):
        adjudicate_capture_metadata(parent_path, decisions_path, tmp_path / "derived")
    decisions_path, decisions = _adjudication(tmp_path, parent_path)
    _json(decisions_path, decisions)
    existing = tmp_path / "derived"
    existing.mkdir()
    with pytest.raises(CandidateError, match="separate new directory"):
        adjudicate_capture_metadata(parent_path, decisions_path, existing)


def test_adjudication_cli_loads_draft_package_in_a_fresh_process(tmp_path):
    inputs, output = _fixture(tmp_path)
    _produce(inputs, output)
    parent_path = output / "queue" / "queue.json"
    decisions_path, _ = _adjudication(tmp_path, parent_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    result = subprocess.run([
        sys.executable, "-B", str(REPO_ROOT / "scripts" / "adjudicate_u_capture.py"),
        "--parent-queue", str(parent_path), "--decisions", str(decisions_path),
        "--output", str(tmp_path / "cli-derived"), "--policy", "prototype",
    ], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["prototype_only"] is True
    load_candidate_queue(tmp_path / "cli-derived" / "queue.json")


def test_adjudication_rechecks_fact_bytes_and_cannot_cure_missing_request(tmp_path):
    inputs, output = _fixture(tmp_path, missing_request=True)
    _bind_prospective_to_launch_and_child(inputs)
    _produce(inputs, output)
    parent_path = output / "queue" / "queue.json"
    decisions_path, decisions = _adjudication(tmp_path, parent_path)
    derived_path = tmp_path / "derived"
    adjudicate_capture_metadata(parent_path, decisions_path, derived_path)
    derived, _, _ = load_candidate_queue(derived_path / "queue.json")
    assert derived["frames"][0]["import_blockers"] == ["capture_request_monotonic_missing"]
    fact_path = Path(decisions["frames"][0]["evidence"]["roi_bbox"]["path"])
    fact = json.loads(fact_path.read_text(encoding="utf-8"))
    fact["value"] = [0, 0, 2, 1]
    _json(fact_path, fact)
    with pytest.raises(CandidateError, match="evidence bytes changed"):
        load_candidate_queue(derived_path / "queue.json")


def test_future_bound_run_adjudication_review_and_prototype_import(tmp_path):
    inputs, output = _fixture(tmp_path)
    _bind_prospective_to_launch_and_child(inputs)
    _produce(inputs, output)
    parent_path = output / "queue" / "queue.json"
    decisions_path, _ = _adjudication(tmp_path, parent_path)
    derived_path = tmp_path / "derived"
    adjudicate_capture_metadata(parent_path, decisions_path, derived_path)
    queue_path = derived_path / "queue.json"
    queue, _, _ = load_candidate_queue(queue_path)
    from training.vision.u_review import UReviewJournal
    from training.vision.u_import import import_reviewed_u
    journal_path = tmp_path / "review.jsonl"
    journal = UReviewJournal(queue_path, journal_path, prototype_only=True)
    journal.append_agent_visual(queue["frames"][0]["queue_frame_id"],
                                decision="complete", boxes=[], exhaustive=True,
                                reviewer="synthetic-agent", evidence="Synthetic empty ROI check")
    journal.append_agent_visual(queue["frames"][1]["queue_frame_id"],
                                decision="quarantine", reviewer="synthetic-agent",
                                quarantine_reason="Synthetic capture metadata unavailable",
                                evidence="Synthetic source-bound decision")
    reviews = tmp_path / "reviews.jsonl"
    journal.export_reviews(reviews)
    result = import_reviewed_u(queue_path, journal_path, reviews,
                               tmp_path / "prototype-snapshot", prototype_only=True)
    assert result["window_count"] == 1
    assert result["prototype_only"] is True and result["formal_acceptance"] is False
    assert len(result["quarantined_frames"]) == 1
