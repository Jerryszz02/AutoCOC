"""Convert explicitly reviewed U candidate frames into a frozen dynamic snapshot."""
from __future__ import annotations

from pathlib import Path

from .data import DatasetError, sha256
from .dynamic_data import verify_dynamic_windows
from .dynamic_snapshot import freeze_dynamic_snapshot
from .u_candidates import ADJUDICATED_QUEUE_SCHEMA, CandidateError, load_candidate_queue
from .u_review import UReviewJournal


def import_reviewed_u(queue_path: Path, journal_path: Path, reviews_path: Path,
                      output_root: Path, *, prototype_only: bool) -> dict:
    """Require an explicit complete/quarantine decision for every queued frame.

    Quarantined and incomplete rows are retained in the external queue/journal but
    never become boxes or negatives. Split/group assignments are copied unchanged.
    """
    if type(prototype_only) is not bool:
        raise DatasetError("Explicit prototype_only policy is required")
    queue_path = Path(queue_path).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise DatasetError(f"Refusing to overwrite U snapshot: {output_root}")
    try:
        queue, source_pool, queue_sha = load_candidate_queue(queue_path)
    except CandidateError as exc:
        raise DatasetError(str(exc)) from exc
    if queue["schema"] == ADJUDICATED_QUEUE_SCHEMA and prototype_only is not True:
        raise DatasetError("Agent-visual capture adjudication can enter prototype snapshots only")
    if output_root.is_relative_to(source_pool) or source_pool.is_relative_to(output_root):
        raise DatasetError("U snapshot output must be separate from the immutable candidate source pool")
    journal = UReviewJournal(queue_path, journal_path, prototype_only=prototype_only)
    expected_reviews = journal.review_rows()
    reviews_path = Path(reviews_path).resolve(strict=True)
    supplied = []
    import json
    from .u_candidates import _json_unique
    for number, line in enumerate(reviews_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_json_unique)
        except (json.JSONDecodeError, CandidateError) as exc:
            raise DatasetError(f"Review export line {number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"Review export line {number} must be an object")
        supplied.append(row)
    if supplied != expected_reviews:
        raise DatasetError("Review export does not exactly match latest source-bound journal decisions")
    if not supplied:
        raise DatasetError("No explicit U candidate review decisions")
    frames = {row["queue_frame_id"]: row for row in queue["frames"]}
    source_by_plan = {row["plan_id"]: row for row in queue["sources"]}
    windows, quarantined = [], []
    for review in supplied:
        candidate = frames.get(review.get("queue_frame_id"))
        if candidate is None:
            raise DatasetError("Review references an unknown U queue frame")
        if review.get("queue_sha256") != queue_sha:
            raise DatasetError("Review export is bound to another queue")
        if (review.get("source_frame_sha256") != candidate["source_image_sha256"]
                or review.get("source_roi_bbox") != candidate.get("roi_bbox")):
            raise DatasetError("Review frame or ROI changed from source queue")
        if review["decision"] == "quarantine":
            quarantined.append({"queue_frame_id": candidate["queue_frame_id"],
                                "reason": review["quarantine_reason"],
                                "journal_event_sha256": review["journal_event_sha256"]})
            continue
        if review["decision"] != "complete" or review.get("boxes_complete") is not True or review.get("exhaustive") is not True:
            raise DatasetError("Only explicit exhaustive complete reviews can enter a U snapshot")
        if candidate.get("import_blockers"):
            raise DatasetError("Reviewed candidate still has capture metadata blockers")
        plan_source = source_by_plan.get(candidate["plan_id"])
        if plan_source is None:
            raise DatasetError("Candidate plan provenance is missing")
        method = review["review_method"]
        evidence = (
            f"u_candidate_queue_sha256={queue_sha};"
            f"review_journal_sha256={review['review_journal_sha256']};"
            f"event_sha256={review['journal_event_sha256']}"
        )
        frame_id = candidate["frame_id"]
        boxes = [{"frame_id": frame_id, "label": item["label"], "bbox": item["bbox"]}
                 for item in review["boxes"]]
        windows.append({
            "schema_version": 1,
            "kind": "u_position",
            "window_id": candidate["queue_frame_id"],
            "event_group_id": candidate["event_group_id"],
            "split": candidate["split"],
            "review_status": "reviewed",
            "reviewer": review["reviewer"],
            "review_method": method,
            "review_evidence": evidence,
            "prototype_only": prototype_only,
            "truth_boxes": boxes,
            "frames": [{
                "session_id": plan_source["session_id"],
                "camera_epoch": candidate["camera_epoch"],
                "frame_id": frame_id,
                "battle_id": candidate["battle_id"],
                "layout_id": candidate["layout_id"],
                "parent_image_sha256": candidate["source_image_sha256"],
                "image": candidate["image"],
                "sha256": candidate["sha256"],
                "captured_at_utc": candidate["captured_at_utc"],
                "captured_at_monotonic": candidate["captured_at_monotonic"],
                "client_version": candidate["client_version"],
                "camera_stable": candidate["camera_stable"],
                "zoom_verified": candidate["zoom_verified"],
                "roi_bbox": candidate["roi_bbox"],
                "boxes_complete": True,
                "boxes_review_status": "reviewed",
                "boxes_review_method": method,
                "boxes_reviewer": review["reviewer"],
                "boxes_review_evidence": evidence,
                "boxes_source_frame_sha256": candidate["source_image_sha256"],
                "boxes_roi_bbox": candidate["roi_bbox"],
            }],
        })
    if not windows:
        raise DatasetError("No complete U frames remain after explicit quarantine decisions")
    inventory = verify_dynamic_windows(source_pool, windows, kind="u_position",
                                       prototype_only=prototype_only)
    frozen = freeze_dynamic_snapshot(source_pool, windows, output_root,
                                     kind="u_position", prototype_only=prototype_only)
    return {
        "snapshot": frozen["root"],
        "snapshot_sha256": frozen["snapshot_sha256"],
        "split_sha256": frozen["split_sha256"],
        "queue_sha256": queue_sha,
        "journal_sha256": sha256(Path(journal_path)),
        "reviews_sha256": sha256(reviews_path),
        "window_count": inventory["windows"],
        "groups_by_split": inventory["groups_by_split"],
        "quarantined_frames": quarantined,
        "human_review": all(row["review_method"] == "human" for row in supplied
                            if row["decision"] == "complete"),
        "prototype_only": prototype_only,
        "formal_acceptance": False,
        "training_started": False,
    }
