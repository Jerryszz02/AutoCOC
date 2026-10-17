"""Source-bound, prototype-only adjudication of normal-run U capture metadata.

The overlay records independent visual decisions. It never changes the parent
queue, invents labels, or promotes an agent review to human/formal evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .u_candidates import (
    ADJUDICATED_QUEUE_SCHEMA, QUEUE_SCHEMA, RUN_SCOPE_PLAN_SCHEMA,
    _aware_utc,
    _load_json, _need, _safe_absolute, _write_bytes_new, load_candidate_queue,
    sha256_file,
)


DECISIONS_SCHEMA = "u_capture_adjudication_v1"
FACT_SCHEMA = "u_capture_fact_evidence_v1"
FACTS = ("roi_bbox", "camera_epoch", "camera_stable", "zoom_verified",
         "battle_layout")
RESOLVED = {
    "roi_bbox": "source_roi_not_independently_reviewed",
    "camera_epoch": "camera_epoch_unverified",
    "camera_stable": "camera_not_verified_stable",
    "zoom_verified": "zoom_not_verified",
    "battle_layout": "battle_or_layout_identity_unverified",
}
POLICY = "agent_visual_capture_metadata_only; prototype_only; no_human_or_formal_claim"


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _reviewer(value: object, label: str) -> None:
    _need(isinstance(value, str) and bool(value.strip()) and len(value) <= 120,
          f"{label}: reviewer is missing")


def _fact_value(name: str, value: object, frame: dict) -> None:
    if name == "roi_bbox":
        _need(isinstance(value, list) and len(value) == 4
              and all(type(x) is int for x in value)
              and 0 <= value[0] < value[2] <= frame["width"]
              and 0 <= value[1] < value[3] <= frame["height"],
              "Adjudicated ROI is outside the original image")
    elif name == "camera_epoch":
        _need(isinstance(value, str) and bool(value.strip())
              and value not in {"unknown", "unassigned"},
              "Adjudicated camera epoch is unknown")
    elif name in {"camera_stable", "zoom_verified"}:
        _need(value is True, f"{name} must be independently proven true for admission")
    else:
        _need(isinstance(value, dict) and set(value) == {"battle_id", "layout_id"}
              and all(isinstance(value[key], str) and bool(value[key].strip())
                      and value[key] not in {"unknown", "unassigned"}
                      for key in ("battle_id", "layout_id")),
              "Battle and layout identity must both be explicit")


def _verified_fact(ref: dict, name: str, frame: dict, parent_sha: str,
                   reviewer: str, forbidden: set[Path]) -> tuple[object, dict]:
    _need(isinstance(ref, dict) and set(ref) == {"path", "sha256"},
          f"{name}: evidence path/hash is missing")
    path = _safe_absolute(ref["path"], f"{name}.evidence_path")
    _need(path not in forbidden, f"{name}: source/queue cannot be self-evidence")
    digest = sha256_file(path)
    _need(ref["sha256"] == digest, f"{name}: evidence bytes changed")
    proof, _ = _load_json(path)
    _need(set(proof) == {"schema", "schema_version", "parent_queue_sha256",
                           "queue_frame_id", "source_frame_id", "source_image_sha256",
                           "fact", "value", "review_method", "reviewer", "basis",
                           "created_at_utc"}
          and proof.get("schema") == FACT_SCHEMA and proof.get("schema_version") == 1
          and proof.get("parent_queue_sha256") == parent_sha
          and proof.get("queue_frame_id") == frame["queue_frame_id"]
          and proof.get("source_frame_id") == frame["frame_id"]
          and proof.get("source_image_sha256") == frame["source_image_sha256"]
          and proof.get("fact") == name
          and proof.get("review_method") == "agent_visual"
          and proof.get("reviewer") == reviewer
          and isinstance(proof.get("basis"), str) and bool(proof["basis"].strip()),
          f"{name}: fact evidence has missing or mismatched source binding")
    _aware_utc(proof.get("created_at_utc"), f"{name}.created_at_utc")
    _fact_value(name, proof["value"], frame)
    return proof["value"], {"path": str(path), "sha256": digest}


def _derive(parent: dict, parent_sha: str, decisions: dict,
            *, parent_path: Path, decisions_path: Path, output_path: Path) -> dict:
    _need(parent.get("schema") == QUEUE_SCHEMA,
          "Adjudication parent must be an original source-bound queue")
    for source in parent["sources"]:
        plan_path = _safe_absolute(source["plan_path"], "source.plan_path")
        _need(_load_json(plan_path)[0].get("schema") == RUN_SCOPE_PLAN_SCHEMA,
              "Adjudication only supports normal-run capture metadata")
    source_roots = [Path(source["source_root"]).resolve(strict=True)
                    for source in parent["sources"]]
    _need(all(not decisions_path.is_relative_to(root) for root in source_roots),
          "Decisions cannot be written inside an original source run")
    _need(set(decisions) == {"schema", "schema_version", "parent_queue_sha256",
                             "review_method", "reviewer", "created_at_utc", "frames"}
          and decisions.get("schema") == DECISIONS_SCHEMA
          and decisions.get("schema_version") == 1
          and decisions.get("parent_queue_sha256") == parent_sha
          and decisions.get("review_method") == "agent_visual",
          "Capture adjudication is not an exact agent-visual parent binding")
    reviewer = decisions.get("reviewer")
    _reviewer(reviewer, "adjudication")
    _aware_utc(decisions.get("created_at_utc"), "adjudication.created_at_utc")
    rows = decisions.get("frames")
    _need(isinstance(rows, list) and len(rows) == len(parent["frames"]),
          "Every parent frame needs one explicit disposition")
    parent_by_id = {row["queue_frame_id"]: row for row in parent["frames"]}
    _need(len(parent_by_id) == len(parent["frames"]), "Duplicate parent frame identity")
    by_id = {}
    for decision in rows:
        _need(isinstance(decision, dict)
              and set(decision) == {"queue_frame_id", "source_image_sha256",
                                   "decision", "reason", "evidence"},
              "Adjudication frame fields are incomplete")
        key = decision["queue_frame_id"]
        _need(key in parent_by_id and key not in by_id
              and decision["source_image_sha256"] == parent_by_id[key]["source_image_sha256"],
              "Adjudication omitted, duplicated, or rebound a source frame")
        by_id[key] = decision
    forbidden = {parent_path, decisions_path, output_path}
    output_frames, admitted, quarantined = [], 0, 0
    for original in parent["frames"]:
        row = deepcopy(original)
        decision = by_id[row["queue_frame_id"]]
        status = decision["decision"]
        _need(status in {"admit", "quarantine"}, "Unknown capture disposition")
        if status == "quarantine":
            _need(isinstance(decision["reason"], str) and bool(decision["reason"].strip())
                  and decision["evidence"] is None,
                  "Quarantine needs a reason and cannot claim verified facts")
            row["import_blockers"].append("capture_metadata_quarantined")
            quarantined += 1
        else:
            _need(decision["reason"] is None and isinstance(decision["evidence"], dict)
                  and set(decision["evidence"]) == set(FACTS),
                  "Admission requires five independent fact references")
            refs = {}
            paths = set()
            for name in FACTS:
                value, ref = _verified_fact(decision["evidence"][name], name, row,
                                            parent_sha, reviewer, forbidden)
                path = Path(ref["path"])
                _need(path not in paths, "One evidence file cannot prove two facts")
                _need(all(not path.is_relative_to(root) for root in source_roots),
                      "Fact evidence cannot be written inside an original source run")
                paths.add(path)
                refs[name] = ref
                if name == "battle_layout":
                    row["battle_id"] = value["battle_id"]
                    row["layout_id"] = value["layout_id"]
                else:
                    row[name] = value
                if RESOLVED[name] in row["import_blockers"]:
                    row["import_blockers"].remove(RESOLVED[name])
            admitted += 1
        row["capture_adjudication"] = {
            "decision": status, "reason": decision["reason"],
            "review_method": "agent_visual", "reviewer": reviewer,
            "evidence": refs if status == "admit" else None,
        }
        output_frames.append(row)
    counts = dict(parent["counts"])
    counts["metadata_blocked"] = sum(bool(row["import_blockers"]) for row in output_frames)
    counts["admitted"] = admitted
    counts["quarantined"] = quarantined
    return {
        "schema": ADJUDICATED_QUEUE_SCHEMA, "schema_version": 1,
        "policy": POLICY, "parent_queue_path": str(parent_path),
        "parent_queue_sha256": parent_sha,
        "decisions_path": str(decisions_path),
        "decisions_sha256": sha256_file(decisions_path),
        "source_pool": str((parent_path.parent / parent["source_pool"]).resolve(strict=True)),
        "sources": parent["sources"], "frames": output_frames, "counts": counts,
        "training_ready": False, "formal_acceptance": False,
        "human_review": False, "prototype_only": True,
    }


def adjudicate_capture_metadata(parent_queue_path: Path, decisions_path: Path,
                                output_dir: Path) -> dict:
    parent_path = _safe_absolute(str(parent_queue_path), "parent_queue_path")
    decisions_path = _safe_absolute(str(decisions_path), "decisions_path")
    _need(parent_path != decisions_path, "Parent and decision files must differ")
    parent, pool, parent_sha = load_candidate_queue(parent_path)
    decisions, _ = _load_json(decisions_path)
    output_dir = Path(output_dir).resolve()
    _need(not output_dir.exists() and not output_dir.is_relative_to(pool)
          and not pool.is_relative_to(output_dir)
          and not output_dir.is_relative_to(parent_path.parent)
          and not parent_path.parent.is_relative_to(output_dir)
          and not decisions_path.parent.is_relative_to(output_dir),
          "Adjudicated output must be a separate new directory")
    derived = _derive(parent, parent_sha, decisions, parent_path=parent_path,
                      decisions_path=decisions_path, output_path=output_dir / "queue.json")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_bytes_new(output_dir / "queue.json", _bytes(derived))
        load_adjudicated_queue(output_dir / "queue.json", derived)
    except Exception as exc:
        _write_bytes_new(output_dir / "INCOMPLETE",
                         f"Adjudication failed: {type(exc).__name__}\n".encode("utf-8"))
        raise
    return {"queue": str(output_dir / "queue.json"),
            "queue_sha256": sha256_file(output_dir / "queue.json"),
            "counts": derived["counts"], "prototype_only": True,
            "human_review": False, "formal_acceptance": False,
            "training_ready": False}


def load_adjudicated_queue(queue_path: Path, queue: dict | None = None) -> tuple[dict, Path, str]:
    queue_path = _safe_absolute(str(queue_path), "adjudicated_queue")
    queue = _load_json(queue_path)[0] if queue is None else queue
    parent_path = _safe_absolute(queue.get("parent_queue_path"), "parent_queue_path")
    decisions_path = _safe_absolute(queue.get("decisions_path"), "decisions_path")
    _need(parent_path != queue_path and decisions_path != queue_path
          and parent_path != decisions_path,
          "Adjudicated queue cannot reference itself")
    parent, pool, parent_sha = load_candidate_queue(parent_path)
    _need(parent.get("schema") == QUEUE_SCHEMA
          and queue.get("parent_queue_sha256") == parent_sha
          and queue.get("decisions_sha256") == sha256_file(decisions_path),
          "Parent queue or adjudication decisions changed")
    decisions, _ = _load_json(decisions_path)
    expected = _derive(parent, parent_sha, decisions, parent_path=parent_path,
                       decisions_path=decisions_path, output_path=queue_path)
    _need(queue == expected, "Adjudicated queue changed from source-bound decisions")
    _need(Path(queue["source_pool"]) == pool,
          "Adjudicated queue points to a different source pool")
    return queue, pool, sha256_file(queue_path)
