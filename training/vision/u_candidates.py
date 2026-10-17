"""Hash-bound candidate queue for reviewed U-position windows.

Reads capture metadata and hashes original PNG bytes. It never decodes pixels,
creates labels, or changes predeclared split assignments.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath


QUEUE_SCHEMA = "u_candidate_queue_v1"
BATCH_SCHEMA = "u_candidate_batch_v1"
PLAN_SCHEMA = "u_capture_plan_v1"
CAPTURE_SCHEMA = "u_capture_manifest_v1"
RUN_SCOPE_PLAN_SCHEMA = "u_run_scope_plan_v1"
RUN_CAPTURE_SCHEMA = "u_run_capture_manifest_v1"
RUN_PROSPECTIVE_ROLES = {
    "prospective_continuous_daily_train_candidate_v1": ("train_candidate", "train"),
    "prospective_u_normal_development_val_v1": ("development_validation", "val"),
}
ADJUDICATED_QUEUE_SCHEMA = "u_adjudicated_candidate_queue_v1"
FRAME_FIELDS = {
    "frame_id", "image", "sha256", "width", "height", "captured_at_utc",
    "captured_at_monotonic", "camera_epoch", "client_version", "camera_stable",
    "zoom_verified", "battle_id", "layout_id", "roi_bbox",
}
RUN_FRAME_FIELDS = FRAME_FIELDS | {"requested_at_monotonic", "observed_at_monotonic"}


class CandidateError(ValueError):
    """Candidate source evidence is incomplete, changed, or unsafe."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise CandidateError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _json_unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise CandidateError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> tuple[dict, bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_json_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"Cannot read JSON input {path}: {exc}") from exc
    _need(isinstance(value, dict), f"JSON input must be an object: {path}")
    return value, raw


def _load_jsonl(raw: bytes, label: str) -> list[dict]:
    rows = []
    try:
        for index, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line.decode("utf-8"), object_pairs_hook=_json_unique)
            _need(isinstance(item, dict), f"{label}:{index}: expected object")
            rows.append(item)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"Invalid JSONL {label}: {exc}") from exc
    return rows


def _aware_utc(value: object, label: str) -> datetime:
    _need(isinstance(value, str) and bool(value), f"{label}: timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateError(f"{label}: invalid ISO timestamp") from exc
    _need(parsed.tzinfo is not None and parsed.utcoffset() is not None,
          f"{label}: timestamp must include a timezone")
    return parsed


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _safe_rel(root: Path, raw: object, label: str) -> Path:
    _need(isinstance(raw, str) and bool(raw), f"{label}: path missing")
    normalized = raw.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    _need(parts and not PurePosixPath(normalized).is_absolute()
          and all(part not in {"", ".", ".."} for part in parts)
          and ":" not in parts[0], f"{label}: path must be contained and relative")
    path = root.joinpath(*parts)
    resolved = path.resolve(strict=True)
    _need(resolved.is_relative_to(root), f"{label}: path escapes source root")
    cursor = root
    is_junction = getattr(os.path, "isjunction", lambda _: False)
    for part in parts:
        cursor = cursor / part
        _need(not cursor.is_symlink() and not is_junction(cursor),
              f"{label}: symlink or junction path is not allowed")
    _need(resolved.is_file(), f"{label}: source is not a file")
    return resolved


def _safe_absolute(raw: object, label: str) -> Path:
    _need(isinstance(raw, str) and bool(raw), f"{label}: absolute path missing")
    original = Path(raw)
    _need(original.is_absolute(), f"{label}: path must be absolute")
    cursor = original
    is_junction = getattr(os.path, "isjunction", lambda _: False)
    while cursor != cursor.parent:
        _need(not cursor.is_symlink() and not is_junction(cursor),
              f"{label}: symlink or junction path is not allowed")
        cursor = cursor.parent
    path = original.resolve(strict=True)
    _need(path.is_file(), f"{label}: expected a regular file")
    return path


def _prelaunch_scope_bound(prospective: dict, launch: dict, binding: dict,
                           path: Path, digest: str) -> bool:
    """Require both launcher and loaded child to name the same pre-capture scope."""
    external = launch.get("external_files")
    scope = launch.get("prospective_plan")
    expected = {
        "path": str(path), "sha256": digest,
        "schema": prospective.get("schema"),
        "source_head": prospective.get("source_head"),
        "created_at_utc": prospective.get("created_at_utc"),
        "split_role": prospective.get("split_role"),
        "scope": prospective.get("scope"),
        "bound_before_capture": True,
    }
    return bool(isinstance(external, dict) and external.get(str(path)) == digest
                and isinstance(scope, dict) and set(scope) == set(expected)
                and isinstance(expected["scope"], str) and bool(expected["scope"].strip())
                and scope == expected
                and launch.get("prelaunch_scope_binding_verified") is True
                and binding.get("prospective_plan_verified_before_cli") is True
                and binding.get("prospective_plan") == scope)


def _prospective_run_split(prospective: dict) -> str:
    """Map an explicit normal-run declaration to its immutable dataset role."""
    schema = prospective.get("schema")
    role = RUN_PROSPECTIVE_ROLES.get(schema) if isinstance(schema, str) else None
    _need(role is not None and prospective.get("split_role") == role[0],
          "Unsupported prospective normal-run schema/split role")
    if role[1] == "val":
        _need(prospective.get("validation_purpose") == "development_validation"
              and prospective.get("independent_final_heldout") is False,
              "Development VAL declaration cannot claim final heldout status")
    return role[1]


def _verify_run_scope_evidence(plan: dict) -> None:
    """Recheck the exact launch/report/scope files behind a derived run queue."""
    evidence = plan.get("source_evidence")
    expected = {"report", "events", "launch", "binding", "prospective"}
    _need(isinstance(evidence, dict) and set(evidence) == expected,
          "Run-scope plan source evidence is incomplete")
    loaded = {}
    for name, item in evidence.items():
        _need(isinstance(item, dict) and set(item) == {"path", "sha256"},
              f"Run-scope {name} evidence has invalid fields")
        path = _safe_absolute(item["path"], f"source_evidence.{name}.path")
        digest = sha256_file(path)
        _need(item["sha256"] == digest, f"Run-scope {name} source changed")
        loaded[name] = _load_json(path)[0] if name != "events" else None

    report, launch = loaded["report"], loaded["launch"]
    binding, prospective = loaded["binding"], loaded["prospective"]
    report_path = Path(evidence["report"]["path"]).resolve(strict=True)
    events_path = Path(evidence["events"]["path"]).resolve(strict=True)
    _need(report.get("run_id") == plan.get("session_id"),
          "Run report session ID differs from derived plan")
    report_provenance = report.get("provenance")
    _need(isinstance(report_provenance, dict)
          and plan.get("report_started_at_raw") == report.get("started_at")
          and plan.get("report_provenance_captured_at_raw") ==
          report_provenance.get("captured_at"),
          "Derived run plan changed the report's original timestamp strings")
    _need(Path(report.get("events_path", "")).resolve(strict=True) == events_path,
          "Run report does not name the bound event log")
    provenance = report.get("provenance")
    _need(isinstance(provenance, dict)
          and provenance.get("client_version") == plan.get("client_version")
          and provenance.get("source_fingerprint_sha256") == plan.get("source_fingerprint_sha256")
          and provenance.get("effective_config_sha256") == plan.get("effective_config_sha256"),
          "Run report version/source/config differs from derived plan")
    _need(launch.get("commit") == plan.get("source_head")
          and launch.get("head_after") == plan.get("source_head")
          and launch.get("origin_commit") == plan.get("source_head")
          and launch.get("clean") is True
          and launch.get("source_files_unchanged") is True
          and launch.get("evidence_unchanged") is True
          and launch.get("runtime_binding_verified") is True
          and launch.get("normal_cli") is True
          and launch.get("raw_sampler_installed") is False,
          "Launch proof is not a clean, unchanged normal CLI run")
    if plan.get("split") == "val":
        _need(launch.get("kind") == "normal_u_development_val_candidate"
              and launch.get("split_role") == "development_validation"
              and launch.get("independent_final_heldout") is False
              and launch.get("max_battles") == 1
              and launch.get("max_searches_per_battle") == 3,
              "Development VAL launch bounds/role are not proven")
    candidates = launch.get("report_candidates")
    _need(isinstance(candidates, list)
          and str(report_path) in {str(Path(item).resolve()) for item in candidates
                                   if isinstance(item, str)},
          "Launch proof does not bind this report")
    binding_modules = binding.get("modules_after")
    _need(binding.get("initial_modules_unchanged") is True
          and binding.get("all_loaded_sources_match_snapshot") is True
          and isinstance(binding_modules, dict) and bool(binding_modules),
          "Loaded-module binding is incomplete")
    source_hashes = provenance.get("files_sha256")
    _need(isinstance(source_hashes, dict) and bool(source_hashes),
          "Run report has no source-file hashes")
    for name, module in binding_modules.items():
        relative = module.get("source_relative") if isinstance(module, dict) else None
        module_hash = module.get("sha256") if isinstance(module, dict) else None
        _need(isinstance(relative, str) and relative in source_hashes
              and module_hash == source_hashes[relative],
              f"Loaded module {name} does not match the run source snapshot")
    prospective_created = prospective.get("created_at_utc")
    split = _prospective_run_split(prospective)
    _need(prospective.get("source_head") == plan.get("source_head")
          and prospective.get("declared_before_any_new_session_pixels") is True
          and prospective.get("admitted_to_training") is False
          and prospective.get("human_review") is False
          and prospective.get("raw_sampler_installed") is False
          and prospective.get("normal_cli") is True
          and "U existing post-batch frames" in prospective.get("candidate_uses", []),
          "Prospective declaration does not preassign this normal run to U candidates")
    _need(plan.get("created_at_utc") == prospective_created,
          "Derived run plan does not preserve prospective scope creation time")
    _need(_aware_utc(prospective_created, "prospective.created_at_utc") <
          _aware_utc(launch.get("started_at_utc"), "launch.started_at_utc"),
          "Prospective U scope was not established before launch")
    _need(plan.get("split") == split and plan.get("event_group_id") ==
          f'u-run:{plan["session_id"]}' and plan.get("battle_id") is None
          and plan.get("layout_id") is None,
          "Run-scope assignment invents a battle/layout or changes the declared group")
    _need(launch.get("runtime_binding_path") == str(Path(evidence["binding"]["path"]).resolve())
          and launch.get("runtime_binding_sha256") == evidence["binding"]["sha256"]
          and launch.get("external_files_unchanged") is True,
          "Launch does not bind the unchanged loaded-module evidence")
    prospective_bound = _prelaunch_scope_bound(
        prospective, launch, binding,
        Path(evidence["prospective"]["path"]).resolve(),
        evidence["prospective"]["sha256"])
    _need(plan.get("prelaunch_scope_binding_verified") is prospective_bound,
          "Run plan misstates the launch binding of its prospective declaration")
    if split == "val":
        _need(prospective_bound,
              "Development VAL requires complete prelaunch parent/child scope binding")
    launch_files = launch.get("source_files")
    _need(isinstance(launch_files, dict)
          and all(launch_files.get(name) == digest for name, digest in source_hashes.items()),
          "Run source files differ from the clean launch snapshot")
    event_rows = _load_jsonl(events_path.read_bytes(), str(events_path))
    selected = []
    for row in event_rows:
        if row.get("kind") != "observation" or row.get("scene") != "battle":
            continue
        details = row.get("observations")
        _need(isinstance(details, dict) and isinstance(details.get("frame_id"), str)
              and bool(details["frame_id"]) and isinstance(row.get("frame"), str),
              "Battle observation lacks a source frame identity")
        frame_path = Path(row["frame"])
        _need(frame_path.is_absolute(), "Run battle observation path is not absolute")
        frame_path = _safe_rel(events_path.parent, frame_path.relative_to(events_path.parent).as_posix(),
                               "battle_observation.frame") if frame_path.is_relative_to(events_path.parent) else None
        _need(frame_path is not None and frame_path.suffix.lower() == ".png",
              "Battle observation frame escapes the source run")
        selected.append(details["frame_id"])
    _need(len(selected) == len(set(selected)) and bool(selected),
          "Run battle observation IDs are missing or duplicated")
    _need(plan.get("planned_frames") == [
        {"frame_id": frame_id, "roi_bbox": None,
         "frame_id_discovered_after_run": True} for frame_id in selected],
          "Run-scope plan omits, reorders, or invents battle observations")


def _validate_plan(plan: dict) -> dict[str, dict]:
    schema = plan.get("schema")
    if schema == PLAN_SCHEMA:
        required = {"schema", "schema_version", "kind", "plan_id", "created_at_utc",
                    "session_id", "event_group_id", "split", "battle_id", "layout_id",
                    "client_version", "planned_frames"}
        _need(required <= plan.keys() and not plan.keys() - required,
              "Capture plan has missing or unrecognized fields")
        _need(plan.get("schema_version") == 1 and plan.get("kind") == "u_position",
              "Unsupported U capture plan")
    elif schema == RUN_SCOPE_PLAN_SCHEMA:
        required = {"schema", "schema_version", "kind", "plan_id", "created_at_utc",
                    "generated_at_utc", "session_id", "event_group_id", "split",
                    "battle_id", "layout_id", "client_version", "source_head",
                    "source_fingerprint_sha256", "effective_config_sha256",
                    "source_evidence", "frame_selection", "roi_policy", "planned_frames",
                    "report_started_at_raw", "report_provenance_captured_at_raw",
                    "prelaunch_scope_binding_verified"}
        _need(required <= plan.keys() and not plan.keys() - required
              and plan.get("schema_version") == 1 and plan.get("kind") == "u_position",
              "Run-scope plan has missing or unrecognized fields")
        _aware_utc(plan.get("generated_at_utc"), "plan.generated_at_utc")
        _need(plan.get("roi_policy") == "independent_source_bbox_review_required"
              and plan.get("frame_selection") == {
                  "kind": "observation", "scene": "battle",
                  "policy": "all_matching_unique_observations",
                  "frame_ids_discovered_after_run": True,
              }, "Run-scope plan selection/ROI policy is unsupported")
        for key in ("source_head", "source_fingerprint_sha256", "effective_config_sha256"):
            _need(isinstance(plan.get(key), str) and bool(plan[key]),
                  f"Run-scope plan {key} missing")
        for key in ("report_started_at_raw", "report_provenance_captured_at_raw"):
            _need(isinstance(plan.get(key), str) and bool(plan[key]),
                  f"Run-scope plan {key} missing")
        _need(type(plan.get("prelaunch_scope_binding_verified")) is bool,
              "Run-scope prospective launch binding must be explicit")
        _verify_run_scope_evidence(plan)
    else:
        raise CandidateError("Unsupported U capture plan")
    for key in ("plan_id", "session_id", "event_group_id", "client_version"):
        _need(isinstance(plan.get(key), str) and bool(plan[key]), f"Plan {key} missing")
    if schema == PLAN_SCHEMA:
        for key in ("battle_id", "layout_id"):
            _need(isinstance(plan.get(key), str) and bool(plan[key]), f"Plan {key} missing")
    else:
        _need(plan.get("battle_id") is None and plan.get("layout_id") is None,
              "Run-scope battle/layout IDs must remain unverified")
    _need(plan["split"] in {"train", "val", "test"}, "Plan split must be frozen")
    _aware_utc(plan.get("created_at_utc"), "plan.created_at_utc")
    planned = plan.get("planned_frames")
    _need(isinstance(planned, list) and bool(planned), "Plan requires predeclared frame IDs")
    result = {}
    for item in planned:
        allowed_item_fields = {"frame_id", "roi_bbox"}
        if schema == RUN_SCOPE_PLAN_SCHEMA:
            allowed_item_fields.add("frame_id_discovered_after_run")
        _need(isinstance(item, dict) and set(item) == allowed_item_fields,
              "Planned frame has missing or unrecognized fields")
        frame_id = item.get("frame_id")
        _need(isinstance(frame_id, str) and bool(frame_id) and frame_id not in result,
              "Duplicate or missing planned frame ID")
        roi = item.get("roi_bbox")
        if roi is not None:
            _need(isinstance(roi, list) and len(roi) == 4
                  and all(type(x) is int for x in roi)
                  and 0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3],
                  f"{frame_id}: invalid predeclared ROI")
        if schema == RUN_SCOPE_PLAN_SCHEMA:
            _need(item.get("frame_id_discovered_after_run") is True,
                  f"{frame_id}: run-scope frame discovery marker missing")
        result[frame_id] = {"frame_id": frame_id, "roi_bbox": roi}
    return result


def _event_digest(row: dict) -> str:
    raw = json.dumps(row, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(raw)


def _observation_index(events: list[dict], source_root: Path) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for row in events:
        if row.get("kind") != "observation" or not isinstance(row.get("frame"), str):
            continue
        observations = row.get("observations")
        if not isinstance(observations, dict):
            continue
        frame_id = observations.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        raw_path = Path(row["frame"])
        if not raw_path.is_absolute():
            raw_path = source_root / raw_path
        try:
            resolved = raw_path.resolve(strict=True)
        except OSError:
            continue
        index.setdefault(frame_id, []).append({
            "path": resolved, "observations": observations, "scene": row.get("scene"),
            "event_sha256": _event_digest(row),
        })
    return index


def _validate_capture_frame(frame: dict, *, plan: dict, planned: dict,
                            source_root: Path,
                            observation_index: dict[str, list[dict]]) -> tuple[dict, bytes]:
    run_scope = plan.get("schema") == RUN_SCOPE_PLAN_SCHEMA
    _need(isinstance(frame, dict) and set(frame) ==
          (RUN_FRAME_FIELDS if run_scope else FRAME_FIELDS),
          "Capture frame has missing or unknown fields")
    frame_id = frame["frame_id"]
    _need(frame_id in planned, f"Unplanned capture frame: {frame_id}")
    for key in ("battle_id", "layout_id", "client_version"):
        _need(frame[key] == plan[key], f"{frame_id}: captured {key} differs from predeclared plan")
    if run_scope:
        _need(frame["camera_epoch"] is None or
              isinstance(frame["camera_epoch"], str),
              f"{frame_id}: invalid camera epoch evidence")
        _need(all(value is None or type(value) is bool for value in
                  (frame["camera_stable"], frame["zoom_verified"])),
              f"{frame_id}: invalid camera/zoom evidence")
    else:
        _need(isinstance(frame["camera_epoch"], str) and bool(frame["camera_epoch"]),
              f"{frame_id}: camera_epoch missing")
        _need(type(frame["camera_stable"]) is bool and type(frame["zoom_verified"]) is bool,
              f"{frame_id}: camera/zoom facts must be explicit booleans")
    width, height = frame["width"], frame["height"]
    _need(type(width) is int and width > 0 and type(height) is int and height > 0,
          f"{frame_id}: source dimensions missing")
    roi = frame["roi_bbox"]
    _need((run_scope and roi is None) or
          (isinstance(roi, list) and len(roi) == 4 and all(type(x) is int for x in roi)
           and 0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height),
          f"{frame_id}: invalid source ROI")
    _need(planned[frame_id]["roi_bbox"] == roi,
          f"{frame_id}: source ROI differs from pre-capture plan")
    utc = None
    if run_scope:
        _need(isinstance(frame["captured_at_utc"], str) and bool(frame["captured_at_utc"]),
              f"{frame_id}: original capture UTC value missing")
        try:
            utc = datetime.fromisoformat(frame["captured_at_utc"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise CandidateError(f"{frame_id}: invalid original capture UTC value") from exc
    else:
        utc = _aware_utc(frame["captured_at_utc"], f"{frame_id}.captured_at_utc")
    if run_scope:
        _need(_finite(frame["observed_at_monotonic"]) and
              frame["observed_at_monotonic"] >= 0,
              f"{frame_id}: observed monotonic value is absent")
        requested = frame["requested_at_monotonic"]
        _need(requested is None or (_finite(requested) and requested >= 0),
              f"{frame_id}: invalid requested monotonic value")
        _need(frame["captured_at_monotonic"] == requested,
              f"{frame_id}: capture start was not copied from the original request time")
    else:
        _need(_finite(frame["captured_at_monotonic"]) and frame["captured_at_monotonic"] >= 0,
              f"{frame_id}: invalid capture monotonic time")
    if utc.tzinfo is not None and utc.utcoffset() is not None:
        _need(utc >= _aware_utc(plan["created_at_utc"], "plan.created_at_utc"),
              f"{frame_id}: capture predates the split scope")
    source_path = _safe_rel(source_root, frame["image"], f"{frame_id}.image")
    _need(source_path.suffix.lower() == ".png", f"{frame_id}: U source must be an original PNG")
    source_bytes = source_path.read_bytes()
    digest = sha256_bytes(source_bytes)
    _need(frame["sha256"] == digest if run_scope else frame["sha256"] in (None, digest),
          f"{frame_id}: image SHA mismatch")
    matches = observation_index.get(frame_id, [])
    exact = [item for item in matches if item["path"] == source_path]
    _need(len(exact) == 1, f"{frame_id}: source must match one session observation")
    if run_scope:
        _need(exact[0]["scene"] == "battle",
              f"{frame_id}: run-scope frame is not a battle observation")
    observed = exact[0]["observations"]
    if run_scope:
        _need(observed.get("observed_at_monotonic") == frame["observed_at_monotonic"]
              and observed.get("requested_at_monotonic") == frame["requested_at_monotonic"],
              f"{frame_id}: request/observation monotonic values changed")
    else:
        _need(observed.get("observed_at_monotonic") == frame["captured_at_monotonic"],
              f"{frame_id}: capture monotonic differs from source observation")
    blockers = []
    if run_scope and not plan["prelaunch_scope_binding_verified"]:
        blockers.append("prelaunch_scope_binding_unverified")
    if run_scope and frame["requested_at_monotonic"] is None:
        blockers.append("capture_request_monotonic_missing")
    if run_scope and roi is None:
        blockers.append("source_roi_not_independently_reviewed")
    if utc.tzinfo is None or utc.utcoffset() is None:
        blockers.append("source_capture_utc_timezone_unverified")
    if observed.get("capture_timestamp_basis") != "capture_request_start":
        blockers.append("source_event_timestamp_basis_not_capture_request_start")
    if observed.get("captured_at_utc") != frame["captured_at_utc"]:
        blockers.append("source_event_utc_missing_or_mismatched")
    if frame["layout_id"] in {None, "unknown", "unassigned"} or frame["battle_id"] in {None, "unknown", "unassigned"}:
        blockers.append("battle_or_layout_identity_unverified")
    if frame["camera_epoch"] in {None, "unknown", "unassigned"}:
        blockers.append("camera_epoch_unverified")
    if frame["client_version"] in {"unknown", "unassigned"}:
        blockers.append("client_version_unverified")
    if not frame["camera_stable"]:
        blockers.append("camera_not_verified_stable")
    if not frame["zoom_verified"]:
        blockers.append("zoom_not_verified")
    return ({
        "frame_id": frame_id, "image": f"images/{digest}.png",
        "sha256": digest, "width": width, "height": height,
        "captured_at_utc": frame["captured_at_utc"],
        "captured_at_monotonic": frame["captured_at_monotonic"],
        **({"requested_at_monotonic": frame["requested_at_monotonic"],
            "observed_at_monotonic": frame["observed_at_monotonic"]} if run_scope else {}),
        "camera_epoch": frame["camera_epoch"],
        "client_version": frame["client_version"],
        "camera_stable": frame["camera_stable"],
        "zoom_verified": frame["zoom_verified"],
        "battle_id": frame["battle_id"], "layout_id": frame["layout_id"],
        "roi_bbox": roi, "observation_event_sha256": exact[0]["event_sha256"],
        "import_blockers": blockers,
    }, source_bytes)


def _write_bytes_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_candidate_queue(batch_path: Path, output_dir: Path) -> dict:
    """Build a new byte-preserving source pool and pending candidate queue."""
    batch_path = Path(batch_path).resolve(strict=True)
    batch, batch_raw = _load_json(batch_path)
    _need(set(batch) == {"schema", "schema_version", "sources"}
          and batch.get("schema") == BATCH_SCHEMA and batch.get("schema_version") == 1
          and isinstance(batch.get("sources"), list) and bool(batch["sources"]),
          "Unsupported or empty U candidate batch")
    output_dir = Path(output_dir).resolve()
    _need(not output_dir.exists(), f"Refusing to overwrite candidate output: {output_dir}")
    _need(not batch_path.is_relative_to(output_dir),
          "Candidate output cannot contain its batch control file")
    for item in batch["sources"]:
        _need(isinstance(item, dict) and isinstance(item.get("source_root"), str),
              "Each batch source requires a source root")
        source_root_check = Path(item["source_root"]).resolve(strict=True)
        _need(not output_dir.is_relative_to(source_root_check)
              and not source_root_check.is_relative_to(output_dir),
              "Candidate output must be separate from every source run")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "source_pool").mkdir()
    frames_out, sources_out = [], []
    group_assignments: dict[str, tuple[str, str, str]] = {}
    try:
        for spec in batch["sources"]:
            _need(isinstance(spec, dict) and set(spec) == {
                "source_root", "plan_path", "capture_manifest_path"
            }, "Each source requires root, plan, and capture manifest")
            source_root = Path(spec["source_root"]).resolve(strict=True)
            _need(source_root.is_dir(), "Source root must be a capture directory")
            plan_path = _safe_absolute(spec["plan_path"], "plan_path")
            capture_path = _safe_absolute(spec["capture_manifest_path"], "capture_manifest_path")
            plan, plan_raw = _load_json(plan_path)
            planned = _validate_plan(plan)
            plan_digest = sha256_bytes(plan_raw)
            capture, capture_raw = _load_json(capture_path)
            expected_capture_schema = (RUN_CAPTURE_SCHEMA if plan["schema"] == RUN_SCOPE_PLAN_SCHEMA
                                       else CAPTURE_SCHEMA)
            _need(set(capture) == {"schema", "schema_version", "plan_sha256", "session_id",
                                  "source_events_path", "source_events_sha256", "frames"}
                  and capture.get("schema") == expected_capture_schema
                  and capture.get("schema_version") == 1,
                  "Unsupported U capture manifest")
            _need(capture.get("plan_sha256") == plan_digest
                  and capture.get("session_id") == plan["session_id"],
                  "Capture manifest is not bound to its pre-capture plan/session")
            events_path = _safe_rel(source_root, capture.get("source_events_path"), "source_events_path")
            events_raw = events_path.read_bytes()
            events_digest = sha256_bytes(events_raw)
            _need(capture.get("source_events_sha256") == events_digest,
                  "Session event log changed or capture manifest binding is wrong")
            event_rows = _load_jsonl(events_raw, str(events_path))
            observations = _observation_index(event_rows, source_root)
            capture_frames = capture.get("frames")
            _need(isinstance(capture_frames, list) and bool(capture_frames),
                  "Capture manifest has no frames")
            actual_ids = set()
            prior_times: dict[str, float] = {}
            for raw_frame in capture_frames:
                frame, source_bytes = _validate_capture_frame(
                    raw_frame, plan=plan, planned=planned, source_root=source_root,
                    observation_index=observations)
                _need(frame["frame_id"] not in actual_ids, "Duplicate capture frame ID")
                actual_ids.add(frame["frame_id"])
                previous = prior_times.get(frame["camera_epoch"])
                if (previous is not None and frame["captured_at_monotonic"] is not None
                        and frame["captured_at_monotonic"] <= previous):
                    frame["import_blockers"].append("capture_time_not_increasing")
                if frame["captured_at_monotonic"] is not None:
                    prior_times[frame["camera_epoch"]] = frame["captured_at_monotonic"]
                target = output_dir / "source_pool" / frame["image"]
                if not target.exists():
                    _write_bytes_new(target, source_bytes)
                _need(sha256_file(target) == frame["sha256"], "Copied source bytes changed")
                assignment = (plan["split"], plan["battle_id"], plan["layout_id"])
                prior = group_assignments.get(plan["event_group_id"])
                _need(prior in (None, assignment), "One event group has conflicting split/source assignment")
                group_assignments[plan["event_group_id"]] = assignment
                queue_id = "u-" + hashlib.sha256(
                    f'{plan["session_id"]}\0{frame["frame_id"]}'.encode("utf-8")
                ).hexdigest()[:24]
                queue_frame = {
                    "queue_frame_id": queue_id,
                    "event_group_id": plan["event_group_id"],
                    "split": plan["split"],
                    "session_id": plan["session_id"],
                    "plan_id": plan["plan_id"],
                    "plan_sha256": plan_digest,
                    "capture_manifest_sha256": sha256_bytes(capture_raw),
                    "source_events_sha256": events_digest,
                    "source_events_path": capture["source_events_path"],
                    "source_image_path": raw_frame["image"],
                    "source_image_sha256": frame["sha256"],
                    "source_root": str(source_root),
                    **{key: value for key, value in frame.items()
                       if key != "observation_event_sha256"},
                    "source_observation_event_sha256": frame["observation_event_sha256"],
                    "review_status": "pending", "boxes": None,
                    "identity_labels": None,
                }
                frames_out.append(queue_frame)
            _need(actual_ids == set(planned) if plan["schema"] == RUN_SCOPE_PLAN_SCHEMA
                  else actual_ids <= set(planned),
                  "Capture omitted or invented a run-scope battle frame")
            sources_out.append({
                "source_root": str(source_root), "plan_path": str(plan_path),
                "plan_id": plan["plan_id"],
                "plan_sha256": plan_digest, "capture_manifest_path": str(capture_path),
                "capture_manifest_sha256": sha256_bytes(capture_raw),
                "events_path": capture["source_events_path"], "events_sha256": events_digest,
                "session_id": plan["session_id"], "event_group_id": plan["event_group_id"],
                "split": plan["split"], "planned_frame_ids": list(planned),
                "captured_frame_ids": sorted(actual_ids),
                "not_captured_frame_ids": sorted(set(planned) - actual_ids),
            })
        _need(bool(frames_out), "No source frames were collected")
        queue = {
            "schema": QUEUE_SCHEMA, "schema_version": 1,
            "policy": "source-bound-candidates-only; labels-review-splits-and-formal-status-are-not-inferred",
            "batch_path": str(batch_path), "batch_sha256": sha256_bytes(batch_raw),
            "source_pool": "source_pool", "sources": sources_out, "frames": frames_out,
            "counts": {
                "candidate_frames": len(frames_out), "groups": len(group_assignments),
                "groups_by_split": {name: sum(item[0] == name for item in group_assignments.values())
                                    for name in ("train", "val", "test")},
                "metadata_blocked": sum(bool(item["import_blockers"]) for item in frames_out),
            },
            "training_ready": False, "formal_acceptance": False, "human_review": False,
        }
        queue_path = output_dir / "queue.json"
        _write_bytes_new(queue_path, (json.dumps(queue, ensure_ascii=False, sort_keys=True,
            indent=2, allow_nan=False) + "\n").encode("utf-8"))
        pool_rows = [{"path": path.relative_to(output_dir / "source_pool").as_posix(),
                      "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                     for path in sorted((output_dir / "source_pool").rglob("*.png"))]
        _write_bytes_new(output_dir / "source_pool" / "SHA256SUMS.json",
                         (json.dumps(pool_rows, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        _write_bytes_new(output_dir / "batch-audit.json", (json.dumps({
            "schema_version": 1, "batch_sha256": sha256_bytes(batch_raw),
            "queue_sha256": sha256_file(queue_path), "source_files": sources_out,
            "candidate_frames": len(frames_out), "labels_created": False,
            "images_decoded": False, "split_changed": False,
            "prototype_only": "not applicable until separately reviewed/imported",
        }, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        return {"output": str(output_dir), "queue": str(queue_path),
                "queue_sha256": sha256_file(queue_path), "frames": len(frames_out),
                "metadata_blocked": queue["counts"]["metadata_blocked"]}
    except Exception as exc:
        _write_bytes_new(output_dir / "INCOMPLETE",
                         f"Candidate collection failed: {type(exc).__name__}\n".encode("utf-8"))
        raise


def load_candidate_queue(queue_path: Path) -> tuple[dict, Path, str]:
    """Verify queue against pre-capture plans, source logs and exact original bytes."""
    queue_path = Path(queue_path).resolve(strict=True)
    queue, _ = _load_json(queue_path)
    if queue.get("schema") == ADJUDICATED_QUEUE_SCHEMA:
        from .u_adjudication import load_adjudicated_queue
        return load_adjudicated_queue(queue_path, queue)
    queue_digest = sha256_file(queue_path)
    _need(queue.get("schema") == QUEUE_SCHEMA and queue.get("schema_version") == 1
          and queue.get("policy") ==
          "source-bound-candidates-only; labels-review-splits-and-formal-status-are-not-inferred",
          "Unsupported U candidate queue")
    expected_queue_keys = {
        "schema", "schema_version", "policy", "batch_path", "batch_sha256",
        "source_pool", "sources", "frames", "counts", "training_ready",
        "formal_acceptance", "human_review",
    }
    _need(set(queue) == expected_queue_keys and queue.get("training_ready") is False
          and queue.get("formal_acceptance") is False and queue.get("human_review") is False,
          "Candidate queue has unknown fields or overstated review/training status")
    batch_path = _safe_absolute(queue.get("batch_path"), "batch_path")
    batch, batch_raw = _load_json(batch_path)
    _need(sha256_bytes(batch_raw) == queue.get("batch_sha256")
          and set(batch) == {"schema", "schema_version", "sources"}
          and batch.get("schema") == BATCH_SCHEMA and batch.get("schema_version") == 1
          and isinstance(batch.get("sources"), list) and bool(batch["sources"]),
          "Candidate batch changed or has unsupported schema")
    batch_sources = []
    batch_keys = set()
    for spec in batch["sources"]:
        _need(isinstance(spec, dict) and set(spec) == {
            "source_root", "plan_path", "capture_manifest_path"
        }, "Candidate batch source fields changed")
        source_root = Path(spec["source_root"]).resolve(strict=True)
        plan_path = _safe_absolute(spec["plan_path"], "plan_path")
        capture_path = _safe_absolute(spec["capture_manifest_path"], "capture_manifest_path")
        key = (str(source_root), str(plan_path), str(capture_path))
        _need(key not in batch_keys, "Duplicate source in candidate batch")
        batch_keys.add(key)
        batch_sources.append(key)
    pool = (queue_path.parent / queue.get("source_pool", "")).resolve(strict=True)
    _need(queue.get("source_pool") == "source_pool"
          and pool.is_relative_to(queue_path.parent) and pool.is_dir(),
          "Source pool escapes queue directory or has an unexpected location")
    sources = queue.get("sources")
    frames = queue.get("frames")
    _need(isinstance(sources, list) and bool(sources) and isinstance(frames, list) and bool(frames),
          "Candidate queue has no source records or frames")
    source_by_plan, expected_frames, assignments = {}, {}, {}
    actual_batch_sources = []
    for source in sources:
        _need(isinstance(source, dict) and set(source) == {
            "source_root", "plan_path", "plan_id", "plan_sha256",
            "capture_manifest_path", "capture_manifest_sha256", "events_path",
            "events_sha256", "session_id", "event_group_id", "split",
            "planned_frame_ids", "captured_frame_ids", "not_captured_frame_ids",
        }, "Queue source record has missing or unknown fields")
        plan_path = _safe_absolute(source.get("plan_path"), "plan_path")
        capture_path = _safe_absolute(source.get("capture_manifest_path"), "capture_manifest_path")
        plan, plan_raw = _load_json(plan_path)
        planned = _validate_plan(plan)
        plan_sha = sha256_bytes(plan_raw)
        _need(plan_sha == source.get("plan_sha256"), "Pre-capture plan changed")
        capture, capture_raw = _load_json(capture_path)
        capture_sha = sha256_bytes(capture_raw)
        expected_capture_schema = (RUN_CAPTURE_SCHEMA if plan["schema"] == RUN_SCOPE_PLAN_SCHEMA
                                   else CAPTURE_SCHEMA)
        _need(capture_sha == source.get("capture_manifest_sha256")
              and capture.get("schema") == expected_capture_schema
              and capture.get("schema_version") == 1
              and capture.get("plan_sha256") == plan_sha
              and capture.get("session_id") == plan["session_id"],
              "Capture manifest is stale or not bound to its plan")
        root = Path(source.get("source_root", "")).resolve(strict=True)
        _need(root.is_dir() and str(root) == source.get("source_root"),
              "Source root path changed")
        source_key = (str(root), str(plan_path), str(capture_path))
        _need(source_key in batch_keys and source_key not in actual_batch_sources,
              "Queue includes an unplanned or duplicate batch source")
        actual_batch_sources.append(source_key)
        events_path = _safe_rel(root, capture.get("source_events_path"), "source_events_path")
        events_raw = events_path.read_bytes()
        events_sha = sha256_bytes(events_raw)
        _need(events_sha == capture.get("source_events_sha256")
              and events_sha == source.get("events_sha256")
              and capture.get("source_events_path") == source.get("events_path"),
              "Source event log changed")
        observations = _observation_index(_load_jsonl(events_raw, str(events_path)), root)
        raw_frames = capture.get("frames")
        _need(isinstance(raw_frames, list) and bool(raw_frames), "Capture manifest frame list missing")
        normalized, original_by_id, prior_times = {}, {}, {}
        for raw_frame in raw_frames:
            frame, _ = _validate_capture_frame(raw_frame, plan=plan, planned=planned,
                                                source_root=root, observation_index=observations)
            frame_id = frame["frame_id"]
            _need(frame_id not in normalized, "Duplicate capture frame identity")
            previous_time = prior_times.get(frame["camera_epoch"])
            if (previous_time is not None and frame["captured_at_monotonic"] is not None
                    and frame["captured_at_monotonic"] <= previous_time):
                frame["import_blockers"].append("capture_time_not_increasing")
            if frame["captured_at_monotonic"] is not None:
                prior_times[frame["camera_epoch"]] = frame["captured_at_monotonic"]
            normalized[frame_id] = frame
            original_by_id[frame_id] = raw_frame
        actual_ids = set(normalized)
        _need(actual_ids == set(planned) if plan["schema"] == RUN_SCOPE_PLAN_SCHEMA
              else actual_ids <= set(planned),
              "Capture omitted or invented a run-scope battle frame")
        assignment = (plan["split"], plan["battle_id"], plan["layout_id"])
        prior = assignments.get(plan["event_group_id"])
        _need(prior in (None, assignment), "One predeclared group has conflicting split/battle/layout")
        assignments[plan["event_group_id"]] = assignment
        _need(source.get("plan_id") == plan["plan_id"]
              and source.get("session_id") == plan["session_id"]
              and source.get("event_group_id") == plan["event_group_id"]
              and source.get("split") == plan["split"]
              and source.get("planned_frame_ids") == list(planned)
              and source.get("captured_frame_ids") == sorted(actual_ids)
              and source.get("not_captured_frame_ids") == sorted(set(planned) - actual_ids),
              "Queue source assignment differs from its pre-capture plan")
        _need(plan["plan_id"] not in source_by_plan, "Duplicate plan ID in queue")
        source_by_plan[plan["plan_id"]] = {
            "source": source, "plan": plan, "plan_sha": plan_sha,
            "capture_sha": capture_sha, "events_sha": events_sha, "root": root,
            "normalized": normalized, "original": original_by_id,
        }
        expected_frames[plan["plan_id"]] = set(actual_ids)
    seen, queued_by_plan = set(), {plan_id: set() for plan_id in source_by_plan}
    frame_keys = {
        "queue_frame_id", "event_group_id", "split", "session_id", "plan_id",
        "plan_sha256", "capture_manifest_sha256", "source_events_sha256",
        "source_events_path", "source_image_path", "source_image_sha256", "source_root",
        "frame_id", "image", "sha256", "width", "height", "captured_at_utc",
        "captured_at_monotonic", "camera_epoch", "client_version", "camera_stable",
        "zoom_verified", "battle_id", "layout_id", "roi_bbox",
        "source_observation_event_sha256", "import_blockers",
        "review_status", "boxes", "identity_labels",
    }
    for row in frames:
        _need(isinstance(row, dict), "Queue frame must be an object")
        _need(row.get("review_status") == "pending" and row.get("boxes") is None
              and row.get("identity_labels") is None,
              "Queue cannot contain labels or completed review")
        queue_id, plan_id, frame_id = (row.get("queue_frame_id"), row.get("plan_id"),
                                       row.get("frame_id"))
        _need(isinstance(queue_id, str) and queue_id and queue_id not in seen,
              "Duplicate or missing queue frame ID")
        seen.add(queue_id)
        binding = source_by_plan.get(plan_id)
        _need(binding is not None and frame_id in binding["normalized"],
              "Queue frame is absent from source plan/capture")
        expected_keys = (frame_keys | {"requested_at_monotonic", "observed_at_monotonic"}
                         if binding["plan"]["schema"] == RUN_SCOPE_PLAN_SCHEMA else frame_keys)
        _need(set(row) == expected_keys,
              "Queue frame has missing or unknown fields")
        candidate = binding["normalized"][frame_id]
        raw_frame = binding["original"][frame_id]
        plan, source = binding["plan"], binding["source"]
        expected_id = "u-" + hashlib.sha256(
            f'{plan["session_id"]}\0{frame_id}'.encode("utf-8")
        ).hexdigest()[:24]
        _need(queue_id == expected_id
              and row.get("session_id") == plan["session_id"]
              and row.get("event_group_id") == plan["event_group_id"]
              and row.get("split") == plan["split"]
              and row.get("plan_sha256") == binding["plan_sha"]
              and row.get("capture_manifest_sha256") == binding["capture_sha"]
              and row.get("source_events_sha256") == binding["events_sha"]
              and row.get("source_events_path") == source["events_path"]
              and row.get("source_root") == source["source_root"]
              and row.get("source_image_path") == raw_frame["image"]
              and row.get("source_image_sha256") == candidate["sha256"]
              and row.get("source_observation_event_sha256") == candidate["observation_event_sha256"],
              f"{queue_id}: source/group/split binding mismatch")
        for key, value in candidate.items():
            queue_key = "source_observation_event_sha256" if key == "observation_event_sha256" else key
            _need(row.get(queue_key) == value, f"{queue_id}: captured field changed: {queue_key}")
        pooled_rel = f'images/{candidate["sha256"]}.png'
        _need(row.get("image") == pooled_rel, f"{queue_id}: unexpected source-pool path")
        pooled = _safe_rel(pool, pooled_rel, f"{queue_id}.pool_image")
        _need(sha256_file(pooled) == candidate["sha256"], f"{queue_id}: source-pool bytes changed")
        original_path = _safe_rel(binding["root"], raw_frame["image"], f"{queue_id}.original")
        _need(sha256_file(original_path) == candidate["sha256"], f"{queue_id}: original source changed")
        queued_by_plan[plan_id].add(frame_id)
    _need(actual_batch_sources == batch_sources,
          "Queue silently omitted or reordered a predeclared batch source")
    _need(all(queued_by_plan[key] == expected_frames[key] for key in expected_frames),
          "Queue silently omitted or added a captured frame")
    _need(seen, "Candidate queue has no frames")
    group_split = {group: assignment[0] for group, assignment in assignments.items()}
    expected_counts = {
        "candidate_frames": len(frames),
        "groups": len(group_split),
        "groups_by_split": {name: sum(split == name for split in group_split.values())
                             for name in ("train", "val", "test")},
        "metadata_blocked": sum(bool(row["import_blockers"]) for row in frames),
    }
    _need(queue.get("counts") == expected_counts,
          "Candidate queue summary counts do not match its source rows")
    return queue, pool, queue_digest
