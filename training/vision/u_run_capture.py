"""Derive a pending U queue from one prospectively scoped normal CLI run.

This is a metadata producer. It reads PNG headers and hashes original bytes;
it never decodes images, proposes object boxes, or infers a camera/layout.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import zlib

from .u_candidates import (
    BATCH_SCHEMA, RUN_CAPTURE_SCHEMA, RUN_SCOPE_PLAN_SCHEMA, CandidateError,
    _load_json, _load_jsonl, _need, _prelaunch_scope_bound, _safe_absolute,
    _safe_rel, _validate_plan, _prospective_run_split,
    _write_bytes_new, build_candidate_queue, sha256_bytes, sha256_file,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Check the native IHDR header without claiming full PNG decode."""
    with path.open("rb") as stream:
        header = stream.read(33)
    _need(len(header) == 33 and header[:8] == PNG_SIGNATURE
          and header[8:12] == b"\x00\x00\x00\x0d" and header[12:16] == b"IHDR"
          and zlib.crc32(header[12:29]) == struct.unpack(">I", header[29:33])[0],
          f"Invalid PNG IHDR: {path}")
    width, height = struct.unpack(">II", header[16:24])
    _need(0 < width <= 16384 and 0 < height <= 16384,
          f"Invalid native PNG dimensions: {path}")
    return width, height


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _require_prospective(prospective: dict, launch: dict, report: dict) -> str:
    split = _prospective_run_split(prospective)
    if split == "val":
        _need(launch.get("kind") == "normal_u_development_val_candidate"
              and launch.get("split_role") == "development_validation"
              and launch.get("independent_final_heldout") is False
              and launch.get("max_battles") == 1
              and launch.get("max_searches_per_battle") == 3,
              "Development VAL launch bounds/role are not proven")
    _need(prospective.get("declared_before_any_new_session_pixels") is True
          and prospective.get("human_review") is False
          and prospective.get("admitted_to_training") is False
          and prospective.get("normal_cli") is True
          and prospective.get("raw_sampler_installed") is False
          and "U existing post-batch frames" in prospective.get("candidate_uses", [])
          and prospective.get("source_head") == launch.get("commit"),
          "Prospective U scope is absent or changed")
    created = prospective.get("created_at_utc")
    started = launch.get("started_at_utc")
    try:
        first = datetime.fromisoformat(created.replace("Z", "+00:00"))
        second = datetime.fromisoformat(started.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise CandidateError("Prospective/launch timestamp is invalid") from exc
    _need(first.tzinfo is not None and second.tzinfo is not None and first < second,
          "Prospective scope was not established before launch")
    _need(isinstance(launch.get("report_candidates"), list),
          "Launch has no explicit report candidates")
    return split


def _require_runtime(report: dict, report_path: Path, launch: dict,
                     binding: dict, binding_path: Path, binding_sha: str) -> None:
    provenance = report.get("provenance")
    _need(isinstance(provenance, dict)
          and isinstance(provenance.get("files_sha256"), dict)
          and bool(provenance["files_sha256"])
          and provenance.get("client_version") == "18.647.2",
          "Run provenance/version is unavailable")
    _need(launch.get("commit") == launch.get("origin_commit") == launch.get("head_after")
          and launch.get("clean") is True
          and launch.get("source_files_unchanged") is True
          and launch.get("external_files_unchanged") is True
          and launch.get("evidence_unchanged") is True
          and launch.get("runtime_binding_verified") is True
          and launch.get("normal_cli") is True
          and launch.get("raw_sampler_installed") is False,
          "Launch is not a clean unchanged normal CLI run")
    _need(launch.get("runtime_binding_path") == str(binding_path)
          and launch.get("runtime_binding_sha256") == binding_sha
          and str(report_path) in launch.get("report_candidates", []),
          "Launch does not bind the exact report/runtime evidence")
    source_files = launch.get("source_files")
    _need(isinstance(source_files, dict)
          and all(source_files.get(name) == digest for name, digest in
                  provenance["files_sha256"].items()),
          "Report source hashes differ from the launch snapshot")
    modules = binding.get("modules_after")
    _need(binding.get("initial_modules_unchanged") is True
          and binding.get("all_loaded_sources_match_snapshot") is True
          and isinstance(modules, dict) and bool(modules),
          "Loaded runtime modules are not proven unchanged")
    for name, item in modules.items():
        _need(isinstance(item, dict)
              and provenance["files_sha256"].get(item.get("source_relative")) == item.get("sha256"),
              f"Loaded module {name} differs from the run source snapshot")


def _selected_observations(events: list[dict], source_root: Path) -> list[dict]:
    selected, ids = [], set()
    for row in events:
        if row.get("kind") != "observation" or row.get("scene") != "battle":
            continue
        observations = row.get("observations")
        _need(isinstance(observations, dict), "Battle observation metadata is absent")
        frame_id = observations.get("frame_id")
        _need(isinstance(frame_id, str) and bool(frame_id) and frame_id not in ids,
              "Battle source frame identity is missing or duplicated")
        ids.add(frame_id)
        raw_path = row.get("frame")
        _need(isinstance(raw_path, str) and Path(raw_path).is_absolute(),
              f"{frame_id}: battle frame path must be absolute")
        original = Path(raw_path)
        _need(original.is_relative_to(source_root),
              f"{frame_id}: battle frame path escapes the run")
        relative = original.relative_to(source_root).as_posix()
        image = _safe_rel(source_root, relative, f"{frame_id}.image")
        _need(image.suffix.lower() == ".png", f"{frame_id}: original is not PNG")
        resolution = observations.get("source_resolution")
        _need(isinstance(resolution, list) and len(resolution) == 2
              and all(type(value) is int for value in resolution)
              and tuple(resolution) == _png_dimensions(image),
              f"{frame_id}: recorded and native PNG dimensions differ")
        captured_at = observations.get("captured_at_utc")
        _need(isinstance(captured_at, str) and bool(captured_at),
              f"{frame_id}: original capture timestamp is absent")
        observed = observations.get("observed_at_monotonic")
        requested = observations.get("requested_at_monotonic")
        _need(type(observed) in (int, float) and 0 <= observed < float("inf"),
              f"{frame_id}: original observed monotonic time is absent")
        _need(requested is None or
              (type(requested) in (int, float) and 0 <= requested < float("inf")),
              f"{frame_id}: original request monotonic time is invalid")
        selected.append({
            "frame_id": frame_id, "image": relative, "sha256": sha256_file(image),
            "width": resolution[0], "height": resolution[1],
            # Keep the JSON string verbatim. A naive value is a queue blocker,
            # never silently interpreted as local time or UTC.
            "captured_at_utc": captured_at,
            "captured_at_monotonic": requested,
            "requested_at_monotonic": requested,
            "observed_at_monotonic": observed,
            "camera_epoch": None, "client_version": None,
            "camera_stable": None, "zoom_verified": None,
            "battle_id": None, "layout_id": None, "roi_bbox": None,
        })
    _need(bool(selected), "Run has no battle observations for a U queue")
    return selected


def produce_run_candidate_queue(report_path: Path, launch_path: Path,
                                binding_path: Path, prospective_path: Path,
                                output_dir: Path, *, manifest_only: bool = False) -> dict:
    """Produce a new source-bound pending queue; never alter source evidence."""
    paths = {
        "report": _safe_absolute(str(report_path), "report"),
        "launch": _safe_absolute(str(launch_path), "launch"),
        "binding": _safe_absolute(str(binding_path), "binding"),
        "prospective": _safe_absolute(str(prospective_path), "prospective"),
    }
    report = _load_json(paths["report"])[0]
    launch = _load_json(paths["launch"])[0]
    binding = _load_json(paths["binding"])[0]
    prospective = _load_json(paths["prospective"])[0]
    events_path = _safe_absolute(report.get("events_path"), "report.events_path")
    source_root = events_path.parent
    _need(source_root.name == report.get("run_id") and events_path.name == "events.jsonl",
          "Report does not bind this normal run event log")
    _need(paths["report"].parent == source_root.parent,
          "Report is not adjacent to its source run")
    paths["events"] = events_path
    split = _require_prospective(prospective, launch, report)
    _require_runtime(report, paths["report"], launch, binding, paths["binding"],
                     sha256_file(paths["binding"]))
    event_rows = _load_jsonl(events_path.read_bytes(), str(events_path))
    frames = _selected_observations(event_rows, source_root)
    version = report["provenance"]["client_version"]
    for frame in frames:
        frame["client_version"] = version
    evidence = {name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()}
    prospective_bound = _prelaunch_scope_bound(
        prospective, launch, binding, paths["prospective"],
        evidence["prospective"]["sha256"])
    if split == "val":
        _need(prospective_bound,
              "Development VAL requires complete prelaunch parent/child scope binding")
    run_id = report["run_id"]
    plan = {
        "schema": RUN_SCOPE_PLAN_SCHEMA, "schema_version": 1,
        "kind": "u_position",
        "plan_id": f'u-run-scope:{run_id}:{evidence["prospective"]["sha256"][:12]}',
        "created_at_utc": prospective["created_at_utc"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": run_id, "event_group_id": f"u-run:{run_id}",
        "split": split, "battle_id": None, "layout_id": None,
        "client_version": version, "source_head": launch["commit"],
        "source_fingerprint_sha256": report["provenance"]["source_fingerprint_sha256"],
        "effective_config_sha256": report["provenance"]["effective_config_sha256"],
        "report_started_at_raw": report["started_at"],
        "report_provenance_captured_at_raw": report["provenance"]["captured_at"],
        "prelaunch_scope_binding_verified": prospective_bound,
        "source_evidence": evidence,
        "frame_selection": {"kind": "observation", "scene": "battle",
                            "policy": "all_matching_unique_observations",
                            "frame_ids_discovered_after_run": True},
        "roi_policy": "independent_source_bbox_review_required",
        "planned_frames": [{"frame_id": item["frame_id"], "roi_bbox": None,
                            "frame_id_discovered_after_run": True} for item in frames],
    }
    # Recheck the full independent launch/report/plan contract before writing.
    _validate_plan(plan)
    capture = {
        "schema": RUN_CAPTURE_SCHEMA, "schema_version": 1,
        "plan_sha256": sha256_bytes(_json_bytes(plan)),
        "session_id": run_id,
        "source_events_path": "events.jsonl",
        "source_events_sha256": evidence["events"]["sha256"],
        "frames": frames,
    }
    output = Path(output_dir).resolve()
    _need(not output.exists() and not output.is_relative_to(source_root)
          and not source_root.is_relative_to(output),
          "Run candidate output must be a new directory separate from source")
    output.mkdir(parents=True, exist_ok=False)
    try:
        _write_bytes_new(output / "plan.json", _json_bytes(plan))
        _write_bytes_new(output / "capture.json", _json_bytes(capture))
        batch = {
            "schema": BATCH_SCHEMA, "schema_version": 1,
            "sources": [{"source_root": str(source_root),
                         "plan_path": str(output / "plan.json"),
                         "capture_manifest_path": str(output / "capture.json")}],
        }
        _write_bytes_new(output / "batch.json", _json_bytes(batch))
        result = {
            "schema": "u_run_candidate_producer_result_v1",
            "source_run_id": run_id, "frames": len(frames), "split": split,
            "plan": str(output / "plan.json"),
            "plan_sha256": sha256_file(output / "plan.json"),
            "capture": str(output / "capture.json"),
            "capture_sha256": sha256_file(output / "capture.json"),
            "batch": str(output / "batch.json"),
            "batch_sha256": sha256_file(output / "batch.json"),
            "queue": None, "pending_only": True,
            "human_review": False, "formal_acceptance": False,
            "training_ready": False, "images_decoded": False,
        }
        if not manifest_only:
            queue = build_candidate_queue(output / "batch.json", output / "queue")
            result["queue"] = queue
        _write_bytes_new(output / "producer-result.json", _json_bytes(result))
        return result
    except Exception as exc:
        _write_bytes_new(output / "INCOMPLETE",
                         f"Run candidate production failed: {type(exc).__name__}\n".encode("utf-8"))
        raise
