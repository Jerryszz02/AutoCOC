"""Bounded, unlabelled enemy-preview collection without troop deployment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np

from .combat import (_candidate_changed, _leave_scout, _next_candidate,
                     _prepare_building_view, _read_army, _require_scene, _search_price)
from .errors import AutoCOCError, FlowError, StopRequested
from .flow import return_to_village
from .locator import scale_box
from .prepared_battle import _remaining_seconds
from .vision import parse_resource_number


PRICE_ROI_1280 = {"search": (145, 530, 245, 573),
                  "enemy_village": (1098, 515, 1250, 560)}
PRICE_MIN_CONFIDENCE = .95


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _price_candidate(item: dict, roi: tuple[int, int, int, int]) -> int | None:
    if not isinstance(item, dict):
        return None
    box = item.get("bbox")
    text = item.get("text")
    if (not isinstance(text, str) or not isinstance(box, (tuple, list)) or len(box) != 4 or
            any(type(value) is not int for value in box) or
            not (box[0] < box[2] and box[1] < box[3])):
        return None
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    overlap = (max(0, min(box[2], roi[2]) - max(box[0], roi[0])) *
               max(0, min(box[3], roi[3]) - max(box[1], roi[1])))
    if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3] and
            overlap >= .8 * (box[2] - box[0]) * (box[3] - box[1])):
        return None
    return parse_resource_number(text)


def _budgeted_price(session, frame, committed: int, limit: int) -> tuple[int, dict]:
    fee = _search_price(frame)
    observations = frame.observations
    expected = PRICE_ROI_1280.get(frame.scene)
    evidence = observations.get("search_cost_evidence")
    resolution = observations.get("baseline_resolution")
    if (expected is None or not isinstance(evidence, dict) or
            not isinstance(resolution, (tuple, list)) or len(resolution) != 2 or
            any(type(value) is not int or value <= 0 for value in resolution)):
        raise FlowError("Search fee lacks current-frame expected-ROI evidence")
    roi = scale_box(expected, from_resolution=(1280, 720), to_resolution=tuple(resolution))
    confidence = evidence.get("confidence")
    if (type(confidence) not in (int, float) or not math.isfinite(confidence) or
            confidence < PRICE_MIN_CONFIDENCE or confidence > 1 or
            evidence.get("expected_roi_at_1280x720") != list(expected) or
            evidence.get("candidate_count") != 1 or
            _price_candidate(evidence, roi) != fee or
            evidence.get("frame") != str(frame.screenshot_path) or
            not frame.screenshot_path.is_file() or
            evidence.get("frame_sha256") != _sha256(frame.screenshot_path)):
        raise FlowError("Search fee OCR evidence is low-confidence, ambiguous or not bound to this frame")
    ocr = observations.get("ocr")
    candidates = [item for item in ocr if _price_candidate(item, roi) is not None] if isinstance(ocr, list) else []
    if (len(candidates) != 1 or
            any(candidates[0].get(key) != evidence.get(key) for key in ("text", "confidence", "bbox"))):
        raise FlowError("Search fee has ambiguous or inconsistent full-frame OCR")
    recognizer = getattr(session, "recognizer", None)
    if recognizer is None or not callable(getattr(recognizer, "recognize_region", None)):
        raise FlowError("Independent same-frame search fee OCR is unavailable")
    independent = recognizer.recognize_region(frame.screenshot_path, roi, single_line=True)
    reads = [{"text": item.text, "confidence": item.confidence, "bbox": item.bbox}
             for item in independent]
    numeric = [item for item in reads if _price_candidate(item, roi) is not None]
    if (len(numeric) != 1 or
            type(numeric[0]["confidence"]) not in (int, float) or
            not math.isfinite(numeric[0]["confidence"]) or
            numeric[0]["confidence"] < PRICE_MIN_CONFIDENCE or
            numeric[0]["confidence"] > 1 or
            _price_candidate(numeric[0], roi) != fee):
        raise FlowError("Independent same-frame search fee OCR is ambiguous or disagrees with quote")
    if committed + fee > limit:
        raise FlowError(f"Next search fee {fee} exceeds remaining gold budget {limit - committed}")
    return fee, {"frame": str(frame.screenshot_path), "frame_sha256": evidence["frame_sha256"],
                 "roi_baseline": list(roi), "full_ocr": {"text": evidence["text"],
                 "confidence": confidence}, "independent_ocr": numeric[0]}


def _time_left(frame) -> float | None:
    try:
        seconds = _remaining_seconds(frame)
    except FlowError:
        return None
    observed = frame.observations.get("observed_at_monotonic")
    if type(observed) not in (int, float) or not math.isfinite(observed):
        return None
    return seconds - max(0., time.monotonic() - observed)


def _safe_exit(session) -> None:
    # A stale preview cannot authorize a paid or destructive input.  Observe
    # afresh; if it is already battle/unknown, leave the state for diagnosis.
    current = session.observe("vision-collection-exit-fresh")
    _require_scene(current, {"enemy_village"})
    _leave_scout(session, current)


def _save_sample(frame, output: Path, session_id: str, number: int,
                 *, zoom_verified: bool, source_frame: str | None) -> dict:
    _require_scene(frame, {"enemy_village"})
    source = frame.screenshot_path
    if not source.is_file():
        raise FlowError(f"Candidate screenshot is missing: {source}")
    name = f"{session_id}-{number:02d}.png"
    destination = output / "images" / name
    shutil.copyfile(source, destination)
    file_hash = _sha256(source)
    if _sha256(destination) != file_hash:
        raise FlowError("Copied candidate screenshot hash differs from capture")
    image = cv2.imdecode(np.fromfile(destination, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FlowError("Candidate screenshot cannot be decoded")
    width, height = image.shape[1], image.shape[0]
    pixel_hash = hashlib.sha256(str((width, height)).encode("ascii")
        + cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
    observed = frame.observations.get("observed_at_monotonic")
    finished = frame.observations.get("pixels_ready_at_monotonic")
    row = {"schema_version": 1, "sample_id": f"C-{session_id}-{number:02d}",
        "image_path": f"images/{name}", "file_sha256": file_hash, "pixel_sha256": pixel_hash,
        "width": width, "height": height, "parent_sample_id": None,
        "source_domain": "live_collection", "source_ref": str(source),
        "source_session": session_id, "battle_id": session_id,
        "layout_group_id": f"provisional:{session_id}",
        "leakage_group_id": f"collection:{session_id}",
        "client_version": getattr(frame, "client_version", None),
        "capture_started_monotonic": observed if type(observed) in (int, float) and math.isfinite(observed) else None,
        "capture_finished_monotonic": finished if type(finished) in (int, float) and math.isfinite(finished) else None,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_quality": "utc_recorded_after_capture",
        "zoom": "minimum_stable" if zoom_verified else "as_observed",
        "zoom_evidence": {"verified": zoom_verified, "method": "_prepare_building_view" if zoom_verified else "none",
                          "source_frame": source_frame, "capture_frame": str(source)},
        "scene_candidate": frame.scene, "scene_confidence": frame.confidence,
        "annotation_version": None, "review_status": "pending", "reviewer": None,
        "review_evidence": None, "review_method": None, "split": "unassigned",
        "intended_uses": ["scene", "building"],
    }
    return row


def collect_samples(session, output: Path, *, max_samples: int,
                    max_search_cost_gold: int) -> dict:
    """Collect at most ten distinct preview candidates and preserve a fee ledger."""
    if type(max_samples) is not int or not 1 <= max_samples <= 10:
        raise ValueError("max_samples must be an integer from 1 to 10")
    if type(max_search_cost_gold) is not int or max_search_cost_gold < 0:
        raise ValueError("max_search_cost_gold must be a nonnegative integer")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite collection directory: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    session_id = uuid4().hex
    report = {"schema_version": 1, "status": "running", "session_id": session_id,
              "max_samples": max_samples, "max_search_cost_gold": max_search_cost_gold,
              "client_version": getattr(session, "client_version", "unknown"),
              "samples": 0, "attempts": [], "quoted_confirmed_gold": 0,
              "quoted_attempted_gold": 0, "actual_search_cost_gold": None,
              "actual_search_cost_basis": "observed_home_gold_delta_when_available",
              "fee_reconciled": None,
              "home_gold_before": None, "home_gold_after": None,
              "returned_home": False, "reason": None,
              "manifest": "unreviewed-manifest.jsonl"}
    manifest = output / report["manifest"]
    manifest.touch()
    _write_json(output / "collection.json", report)
    current = None
    committed = 0
    try:
        home = return_to_village(session)
        gold = (home.observations.get("resources") or {}).get("gold")
        if home.observations.get("resource_source") != "village_inventory" or type(gold) is not int or gold < 0:
            raise FlowError("Home gold before collection is unverified")
        report["home_gold_before"] = gold
        _write_json(output / "collection.json", report)
        session.click(home, "attack")
        menu = session.wait_for({"search"}, timeout_sec=30, label="collection-menu")
        fee, fee_evidence = _budgeted_price(session, menu, committed, max_search_cost_gold)
        # Some client versions may pay and enter the preview directly from this
        # button. Reserve its whole fee before the first potentially paid input.
        first_attempt = {"candidate": 1, "quoted_fee_gold": fee,
                         "fee_evidence": fee_evidence,
                         "action": "find_match_regular", "action_history": ["find_match_regular"],
                         "source_frame": str(menu.screenshot_path),
                         "state": "input_uncertain", "candidate_frame": None}
        report["attempts"].append(first_attempt)
        committed += fee
        report["quoted_attempted_gold"] = committed
        _write_json(output / "collection.json", report)
        session.click(menu, "find_match_regular")
        transition = session.wait_for({"training", "enemy_village"}, timeout_sec=45,
                                      label="collection-first-transition")
        if transition.scene == "training":
            _read_army(transition)
            first_attempt["state"] = "confirmation_opened"
            first_attempt["action_history"].append("attack")
            first_attempt["confirmation_frame"] = str(transition.screenshot_path)
            _write_json(output / "collection.json", report)
            first_attempt["state"] = "input_uncertain"
            _write_json(output / "collection.json", report)
            session.click(transition, "attack")
        else:
            _require_scene(transition, {"enemy_village"})
        previous = None
        for number in range(1, max_samples + 1):
            if number > 1:
                if current is None:
                    raise FlowError("Previous candidate frame is missing")
                remaining = _time_left(current)
                if remaining is None:
                    report["reason"] = "Current preview countdown unavailable; no next search"
                    break
                if remaining < 10:
                    report["reason"] = "Current preview countdown too short for another search"
                    break
                try:
                    fee, fee_evidence = _budgeted_price(session, current, committed,
                                                        max_search_cost_gold)
                    if (_time_left(current) or 0) < 10:
                        raise FlowError("Current preview countdown too short after fee verification")
                except FlowError as exc:
                    report["reason"] = str(exc)
                    break
                action_frame, action = current, "next"
                attempt = {"candidate": number, "quoted_fee_gold": fee, "action": action,
                           "fee_evidence": fee_evidence,
                           "source_frame": str(action_frame.screenshot_path),
                           "state": "input_uncertain", "candidate_frame": None}
                report["attempts"].append(attempt)
                committed += fee  # reserve even if input/transition becomes uncertain
                report["quoted_attempted_gold"] = committed
                _write_json(output / "collection.json", report)
                session.click(action_frame, action)  # never replay an uncertain click
            else:
                attempt = first_attempt
            scout = _next_candidate(session, previous=previous)
            _require_scene(scout, {"enemy_village"})
            if previous is not None and not _candidate_changed(previous, scout):
                raise FlowError("Paid next candidate did not have a distinct verified identity")
            attempt["state"] = "candidate_confirmed"
            attempt["candidate_frame"] = str(scout.screenshot_path)
            report["quoted_confirmed_gold"] += fee
            current = scout
            _write_json(output / "collection.json", report)
            zoom_verified = False
            original_frame = str(scout.screenshot_path)
            remaining = _time_left(scout)
            if remaining is not None and remaining >= 25 and getattr(session, "native", None) is not None:
                try:
                    scout = _prepare_building_view(session, scout)
                    zoom_verified = True
                    current = scout
                except AutoCOCError as exc:
                    report["reason"] = f"Zoom evidence unavailable: {exc}"
                    # After an uncertain zoom, reacquire a fresh frame before any exit.
                    current = session.observe("collection-zoom-failure-fresh")
                    _require_scene(current, {"enemy_village"})
                    scout = current
            row = _save_sample(scout, output, session_id, number,
                               zoom_verified=zoom_verified, source_frame=original_frame)
            row["client_version"] = report["client_version"]
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            report["samples"] += 1
            _write_json(output / "collection.json", report)
            previous = scout
            if report["reason"]:
                break
        if current is not None:
            _safe_exit(session)
        home = return_to_village(session)
        report["returned_home"] = True
        after = (home.observations.get("resources") or {}).get("gold")
        if home.observations.get("resource_source") == "village_inventory" and type(after) is int and after >= 0:
            report["home_gold_after"] = after
            delta = report["home_gold_before"] - after
            if delta >= 0:
                report["actual_search_cost_gold"] = delta
        if report["actual_search_cost_gold"] is None:
            report["status"] = "needs_review"
            report["reason"] = report["reason"] or "Home gold delta could not be verified"
        elif report["actual_search_cost_gold"] != report["quoted_confirmed_gold"]:
            report["status"] = "needs_review"
            report["fee_reconciled"] = False
            report["reason"] = report["reason"] or "Home gold delta differs from confirmed quoted fees"
        else:
            report["status"] = "completed"
            report["fee_reconciled"] = True
            report["reason"] = report["reason"] or "Sample limit or safe stopping condition reached"
    except (AutoCOCError, OSError, ValueError) as exc:
        report["status"] = "failed"
        report["reason"] = str(exc)
        if not report["attempts"]:
            last = getattr(session, "last_snapshot", None)
            if last is not None and last.scene in {"village", "search", "training"}:
                try:
                    home = return_to_village(session, initial_snapshot=last)
                    report["returned_home"] = True
                    after = (home.observations.get("resources") or {}).get("gold")
                    if (home.observations.get("resource_source") == "village_inventory"
                            and type(after) is int and after >= 0):
                        report["home_gold_after"] = after
                        if report["home_gold_before"] is not None and report["home_gold_before"] >= after:
                            report["actual_search_cost_gold"] = report["home_gold_before"] - after
                except AutoCOCError as recovery_error:
                    report["recovery_error"] = str(recovery_error)
    except BaseException:
        report["status"] = "interrupted"
        report["reason"] = "Collection interrupted after recorded actions"
        raise
    finally:
        _write_json(output / "collection.json", report)
    return report
