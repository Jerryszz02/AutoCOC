"""Import a verified read-only picker run into an unreviewed image queue."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from training.vision.data import DatasetError, decode, sha256


SOURCE_KIND = "agent_development_readonly_picker_sampling_v2"
QUEUE_SCHEMA = "picker_candidate_queue_v1"
CLIENT_VERSION = "18.647.2"
CLIENT_VERSION_CODE = 180647003
COUNT_KINDS = ("heroes", "troops", "spells", "siege")
SOURCE_FILES = frozenset({
    "src/autococ/army_navigation.py", "src/autococ/army_editor.py",
    "src/autococ/session.py", "src/autococ/vision.py",
})
PROJECT_COLLECTOR_VERSION = "project_picker_collection_v1"
PROJECT_CALIBRATION_SHA256 = "89fbc6d767ccb7ff86924a3ca6e51599851280bd6e435fdb76363af2c6dc81a6"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(f"Picker import: {message}")


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must be an object")
    return value


def _inside(source: Path, raw: object, *, folder: str) -> Path:
    _require(isinstance(raw, str) and bool(raw), "missing source path")
    path = Path(raw)
    _require(path.is_absolute(), "source path must be absolute")
    directory = (source / folder).resolve()
    _require(directory.is_relative_to(source), "source directory escapes run")
    resolved = path.resolve()
    _require(resolved.is_relative_to(directory), "source path escapes run")
    _require(resolved.parent == directory, "source path must be a direct run file")
    _require(resolved.is_file(), "missing source file")
    return resolved


def _png(path: Path) -> None:
    """Use the existing data pipeline's image decoder to reject invalid pixels."""
    with path.open("rb") as stream:
        _require(stream.read(8) == b"\x89PNG\r\n\x1a\n", f"invalid PNG: {path.name}")
    decode(path)


def _project_calibration(summary: dict, calibration: dict) -> list[dict]:
    _require(summary.get("collector_version") == PROJECT_COLLECTOR_VERSION,
             "unsupported collector version")
    _require(_digest(summary.get("collector_module_sha256_before"))
             and summary["collector_module_sha256_before"] == summary.get("collector_module_sha256_after")
             and summary["collector_module_sha256_before"] == summary.get("runner_sha256_before"),
             "collector module hash drift")
    _require(summary.get("calibration_path") == "assets/picker/18.647.2/calibration.json"
             and summary.get("calibration_sha256") == PROJECT_CALIBRATION_SHA256,
             "unsupported project calibration")
    repo = Path(__file__).resolve().parents[2]
    collector_path = (repo / "src/autococ/picker_collection.py").resolve()
    _require(collector_path.is_relative_to(repo.resolve()) and collector_path.is_file()
             and summary["collector_module_sha256_before"] == sha256(collector_path),
             "project collector module hash mismatch")
    calibration_path = (repo / summary["calibration_path"]).resolve()
    _require(calibration_path.is_relative_to(repo.resolve()) and calibration_path.is_file(),
             "calibration path escapes project")
    _require(sha256(calibration_path) == PROJECT_CALIBRATION_SHA256,
             "project calibration SHA-256 mismatch")
    expected = _json(calibration_path)
    _require(calibration == expected and calibration.get("schema") == "readonly_picker_calibration_v1"
             and calibration.get("client_version") == summary["client_version"]
             and calibration.get("client_version_code") == summary["client_version_code"],
             "project calibration evidence mismatch")
    template = calibration.get("next_arrow_template")
    _require(isinstance(template, dict) and template.get("path") == "next-arrow.png"
             and _digest(template.get("sha256")), "calibration template invalid")
    template_path = (calibration_path.parent / template["path"]).resolve()
    _require(template_path.parent == calibration_path.parent and template_path.is_relative_to(repo.resolve())
             and template_path.is_file() and sha256(template_path) == template["sha256"]
             and summary.get("calibration_template_sha256_before") == template["sha256"]
             and summary.get("calibration_template_sha256_after") == template["sha256"],
             "calibration template path/hash mismatch")
    def provenance_name(raw: object) -> str:
        _require(isinstance(raw, str) and bool(raw) and "\\" not in raw,
                 "calibration provenance name invalid")
        name = PurePosixPath(raw)
        _require(not name.is_absolute() and all(part not in (".", "..") for part in name.parts)
                 and ":" not in raw and str(name) == raw,
                 "calibration provenance path escapes")
        return raw
    refs = []
    for key, role in (("next_arrow_template", "template_crop"),
                      ("source_spell_first_page", "source_spell_frame")):
        item = calibration[key]
        name = item["path"] if key == "next_arrow_template" else item["frame_name"]
        digest = item["sha256"]
        _require(_digest(digest), "calibration provenance hash invalid")
        refs.append({"role": role, "name": provenance_name(name), "sha256": digest,
                     "provenance_only": True})
    first = calibration["next_arrow_template"]
    _require(_digest(first.get("source_frame_sha256")), "troop source frame hash invalid")
    refs.append({"role": "source_troop_frame", "name": provenance_name(first.get("source_frame_name")),
                 "sha256": first["source_frame_sha256"], "provenance_only": True})
    swipes = calibration.get("source_troop_swipe_events")
    _require(isinstance(swipes, list) and len(swipes) == 2, "calibration swipe provenance incomplete")
    for item in swipes:
        _require(isinstance(item, dict) and _digest(item.get("sha256")), "calibration swipe hash invalid")
        refs.append({"role": "source_swipe_event", "name": provenance_name(item.get("name")),
                     "sha256": item["sha256"], "provenance_only": True})
    return refs


def _army_signature(observation: dict, frame: str) -> tuple[list, list]:
    army = observation.get("observations", {}).get("army")
    _require(isinstance(army, dict), "army observation missing")
    counts = []
    for kind in COUNT_KINDS:
        value = army.get(kind)
        _require(isinstance(value, dict) and type(value.get("used")) is int
                 and type(value.get("capacity")) is int and 0 <= value["used"] <= value["capacity"],
                 f"incomplete {kind} capacity")
        counts.append([kind, value["used"], value["capacity"]])
    manifest = army.get("manifest")
    _require(isinstance(manifest, dict) and manifest.get("frame") == frame
             and manifest.get("layout") == "my_army_rows_v1"
             and manifest.get("supported_layout") is True and manifest.get("complete") is True
             and manifest.get("identity_complete") is True and manifest.get("unknowns") == []
             and manifest.get("complete_kinds") == {"troop": True, "spell": True},
             "full army roster proof missing")
    coverage = army.get("identity_coverage")
    _require(isinstance(coverage, dict) and coverage.get("hero") is True
             and coverage.get("siege") is True, "hero/siege coverage missing")
    for key in ("hero_evidence", "siege_evidence"):
        proof = army.get(key)
        _require(isinstance(proof, dict) and proof.get("frame") == frame
                 and proof.get("complete") is True and proof.get("unknowns") == [],
                 f"{key} missing")
    roster = []
    for name, kind in (("troops", "troop"), ("spells", "spell")):
        cards = manifest.get(name)
        _require(isinstance(cards, list), f"{name} missing")
        rows = []
        for card in cards:
            _require(isinstance(card, dict) and card.get("kind") == kind
                     and card.get("source") == "army" and isinstance(card.get("unit_id"), str)
                     and bool(card["unit_id"]) and type(card.get("count")) is int and card["count"] > 0
                     and isinstance(card.get("bbox"), list) and len(card["bbox"]) == 4
                     and all(type(x) is int for x in card["bbox"]), f"{name} card incomplete")
            rows.append([card["bbox"], card["unit_id"], card["count"]])
        _require(len({row[1] for row in rows}) == len(rows), f"{name} duplicate identity")
        roster.append([name, rows])
    cards = army.get("identity_cards")
    _require(isinstance(cards, list) and bool(cards), "hero/siege cards missing")
    identities = []
    for card in cards:
        _require(isinstance(card, dict) and card.get("kind") in ("hero", "siege")
                 and isinstance(card.get("unit_id"), str) and bool(card["unit_id"])
                 and type(card.get("count")) is int and card["count"] > 0
                 and isinstance(card.get("bbox"), list) and len(card["bbox"]) == 4
                 and all(type(x) is int for x in card["bbox"]), "hero/siege card incomplete")
        identities.append([card["kind"], card["unit_id"], card["count"], card["bbox"]])
    _require(len({row[1] for row in identities}) == len(identities), "duplicate hero/siege identity")
    for kind, count_kind in (("hero", "heroes"), ("siege", "siege")):
        _require(sum(row[2] for row in identities if row[0] == kind) == army[count_kind]["used"],
                 f"{kind} roster count mismatch")
    roster.append(["hero_and_siege", identities])
    return counts, roster


def _visible(observation: dict, frame: str, kind: str, counts: list, roster: list) -> dict:
    editor = observation.get("observations", {}).get("army_editor")
    _require(observation.get("scene") == "training" and isinstance(editor, dict)
             and editor.get("frame") == frame and editor.get("surface") == "current_picker"
             and editor.get("ready") is True and editor.get("editing_kind") == kind,
             "picker observation does not match page")
    _require(editor.get("complete_kinds", {}).get(kind) is True and editor.get("unknowns") == [],
             "visible picker row incomplete")
    name = kind + "s"
    expected = next([used, total] for group, used, total in counts if group == name)
    _require(editor.get("capacities", {}).get(kind) == {"used": expected[0], "total": expected[1]},
             "visible picker capacity changed")
    cards = editor.get("cards")
    _require(isinstance(cards, list), "visible picker cards missing")
    selected = []
    for card in cards:
        _require(isinstance(card, dict) and card.get("kind") == kind and card.get("source") == "army"
                 and isinstance(card.get("unit_id"), str) and bool(card["unit_id"])
                 and type(card.get("count")) is int and card["count"] > 0,
                 "visible picker card incomplete")
        selected.append([card["unit_id"], card["count"]])
    baseline = next(rows for group, rows in roster if group == name)
    _require(selected == [[card[1], card[2]] for card in baseline]
             and len({card[0] for card in selected}) == len(selected), "visible picker roster changed")
    army = observation.get("observations", {}).get("army")
    if army is not None:
        _require(isinstance(army, dict) and army.get(name, {}).get("used") == expected[0]
                 and army[name].get("capacity") == expected[1], "picker army header conflict")
    return {"kind": kind, "capacity": {"used": expected[0], "total": expected[1]},
            "selected": selected, "army_header_present": army is not None}


def import_picker_candidates(source_run: Path, output: Path) -> dict:
    """Publish four untouched picker PNGs only after full run/evidence validation."""
    source = source_run.resolve()
    _require(source.is_dir(), "source run missing")
    _require(not output.resolve().is_relative_to(source), "output must be outside source run")
    _require(not output.exists(), "refusing to overwrite output directory")
    summary_path, events_path = source / "summary.json", source / "events.jsonl"
    summary = _json(summary_path)
    _require(summary.get("kind") == SOURCE_KIND and summary.get("status") == "completed"
             and summary.get("reason") == "read_only_capture_sequence_completed", "unsupported run schema/status")
    _require(summary.get("counts_toward_acceptance") is False
             and summary.get("formal_acceptance") is False and summary.get("human_review") is False
             and summary.get("unit_identity_labels") is None, "run acceptance/labels inconsistent")
    _require(summary.get("client_version") == CLIENT_VERSION
             and summary.get("client_version_code") == CLIENT_VERSION_CODE,
             "unsupported client version")
    _require(summary.get("head_before") == summary.get("head_after")
             and isinstance(summary.get("head_before"), str) and len(summary["head_before"]) == 40
             and summary.get("source_status_after") == "", "source revision changed")
    before, after = summary.get("source_sha256_before"), summary.get("source_sha256_after")
    _require(isinstance(before, dict) and set(before) == SOURCE_FILES and before == after
             and all(_digest(v) for v in before.values()),
             "source hash drift")
    for name in ("config", "runner"):
        _require(_digest(summary.get(f"{name}_sha256_before"))
                 and summary[f"{name}_sha256_before"] == summary.get(f"{name}_sha256_after"),
                 f"{name} hash drift")
    _require(type(summary.get("elapsed_sec")) in (int, float) and summary["elapsed_sec"] > 0,
             "run elapsed time missing")
    _require(_inside(source, summary.get("events_path"), folder="") == events_path.resolve(),
             "events path mismatch")
    _require(_digest(summary.get("events_sha256")) and sha256(events_path) == summary["events_sha256"],
             "events SHA-256 mismatch")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(events and all(isinstance(row, dict) for row in events), "events invalid")
    frames = summary.get("frames")
    _require(isinstance(frames, list) and len(frames) >= 4, "frame inventory missing")
    inventory = {}
    for record in frames:
        _require(isinstance(record, dict) and _digest(record.get("sha256")), "frame inventory malformed")
        path = _inside(source, record.get("path"), folder="frames")
        _require(path.suffix.lower() == ".png" and str(path) not in inventory,
                 "duplicate or non-PNG frame")
        _require(sha256(path) == record["sha256"], f"frame SHA-256 mismatch: {path.name}")
        inventory[str(path)] = record["sha256"]
    _require({str(path.resolve()) for path in (source / "frames").iterdir()} == set(inventory),
             "frame inventory missing or extra files")
    flow = summary.get("flow")
    _require(isinstance(flow, dict) and flow.get("zero_roster_change_verified") is True,
             "zero roster change unverified")
    observations = [row for row in events if row.get("kind") == "observation"]
    def frame_ref(raw: object, expected: object) -> Path:
        path = _inside(source, raw, folder="frames")
        _require(inventory.get(str(path)) == expected, "referenced frame SHA-256 mismatch")
        return path
    def observation(frame: str) -> dict:
        matches = [row for row in observations if row.get("frame") == frame]
        _require(len(matches) == 1, "missing or duplicate page observation")
        return matches[0]
    armies = []
    for name in ("army_before", "army_after"):
        army = flow.get(name)
        _require(isinstance(army, dict), f"{name} missing")
        path = frame_ref(army.get("frame"), army.get("sha256"))
        counts, roster = _army_signature(observation(str(path)), str(path))
        _require(army.get("counts") == counts and army.get("roster") == roster,
                 f"{name} summary/observation mismatch")
        armies.append((counts, roster))
    counts, roster = armies[0]
    _require(armies[0] == armies[1], "army capacity or roster changed")
    pages = flow.get("troop_pages")
    _require(isinstance(pages, list) and 1 <= len(pages) <= 5, "troop pages missing")
    _require(type(summary.get("action_count")) is int and summary["action_count"] == 5 + len(pages),
             "unexpected action count")
    _require(flow.get("troop_termination") in
             ("configured_page_limit_not_catalog_end", "no_supported_next_arrow_not_catalog_end"),
             "troop termination missing")
    candidates, page_paths, page_hashes = [], set(), set()
    page_events = [row for row in events if row.get("kind") == "development_readonly_picker_page"]
    _require(len(page_events) == len(pages), "troop page event count mismatch")
    for index, page in enumerate(pages, 1):
        _require(isinstance(page, dict) and page.get("index") == index
                 and page.get("identity_labels") is None, "troop page index/labels invalid")
        path = frame_ref(page.get("frame"), page.get("sha256"))
        _require(str(path) not in page_paths and page["sha256"] not in page_hashes,
                 "duplicate picker page")
        page_paths.add(str(path))
        page_hashes.add(page["sha256"])
        _png(path)
        _require(all(page.get(key) == page_events[index - 1].get(key) for key in
                     ("index", "frame", "sha256", "visible_roster_proof", "identity_labels")),
                 "troop page event mismatch")
        _require(page.get("visible_roster_proof") == _visible(observation(str(path)), str(path), "troop", counts, roster),
                 "troop visible roster proof mismatch")
        if index < len(pages):
            following = pages[index]
            swipe = page.get("swipe_result")
            _require(isinstance(swipe, dict) and swipe.get("verified") is True
                     and swipe.get("after_frame") == following.get("frame")
                     and swipe.get("after_sha256") == following.get("sha256"), "page advance unverified")
            change = [row for row in events if row.get("kind") == "development_readonly_picker_page_change"
                      and row.get("index") == index]
            _require(len(change) == 1 and change[0].get("before_frame") == str(path)
                     and change[0].get("before_sha256") == page["sha256"]
                     and all(change[0].get(k) == v for k, v in swipe.items()), "page change event mismatch")
        candidates.append(("troop", index, path, page["sha256"], page["visible_roster_proof"]))
    spell = flow.get("spell_first_page")
    _require(isinstance(spell, dict) and spell.get("identity_labels") is None
             and spell.get("swipe_attempted") is False, "spell first page invalid")
    path = frame_ref(spell.get("frame"), spell.get("sha256"))
    _require(str(path) not in page_paths and spell["sha256"] not in page_hashes,
             "duplicate picker page")
    _png(path)
    _require(spell.get("visible_roster_proof") == _visible(observation(str(path)), str(path), "spell", counts, roster),
             "spell visible roster proof mismatch")
    candidates.append(("spell", 1, path, spell["sha256"], spell["visible_roster_proof"]))
    term = [row for row in events if row.get("kind") == "development_readonly_picker_termination"]
    _require(len(term) == 1 and term[0].get("reason") == flow["troop_termination"]
             and term[0].get("pages_seen") == len(pages)
             and term[0].get("frame") == pages[-1]["frame"]
             and term[0].get("frame_sha256") == pages[-1]["sha256"], "termination event mismatch")
    summary_hash = sha256(summary_path)
    frame_identity = sorted((Path(path).name, digest) for path, digest in inventory.items())
    group_material = {"collector": summary.get("collector_version", SOURCE_KIND),
                      "client_version": summary["client_version"],
                      "client_version_code": summary["client_version_code"], "frames": frame_identity}
    group_id = hashlib.sha256(json.dumps(group_material, sort_keys=True, separators=(",", ":"))
                              .encode("utf-8")).hexdigest()
    calibration = flow.get("calibration")
    _require(isinstance(calibration, dict), "calibration reference missing")
    if summary.get("collector_version") == PROJECT_COLLECTOR_VERSION:
        calibration_refs = _project_calibration(summary, calibration)
    else:
        _require("collector_version" not in summary, "unknown collector version")
        calibration_refs = []
        for item in (calibration.get("first_troop_frame"), calibration.get("first_spell_frame"),
                     *(calibration.get("troop_swipe_events") or [])):
            _require(isinstance(item, dict) and isinstance(item.get("path"), str)
                     and Path(item["path"]).is_absolute() and _digest(item.get("sha256")),
                     "calibration reference invalid")
            calibration_refs.append({"name": Path(item["path"]).name, "sha256": item["sha256"]})
        _require(len(calibration_refs) >= 2, "calibration reference incomplete")
    metadata = {"schema": QUEUE_SCHEMA, "source_kind": SOURCE_KIND, "source_run_id": source.name,
                "source_group_id": group_id, "source_summary_sha256": summary_hash,
                "source_events_sha256": summary["events_sha256"],
                "client_version": summary["client_version"],
                "client_version_code": summary["client_version_code"],
                "source_revision": summary["head_before"], "source_sha256": before,
                "config_sha256": summary["config_sha256_before"],
                "runner_sha256": summary["runner_sha256_before"],
                "calibration_refs": calibration_refs,
                "review_status": "pending", "human_review": False, "formal_acceptance": False,
                "training_ready": False, "split": None, "scene_labels": None, "building_labels": None,
                "label": None, "identity_labels": None}
    if summary.get("collector_version") == PROJECT_COLLECTOR_VERSION:
        metadata.update({"collector_version": PROJECT_COLLECTOR_VERSION,
                         "calibration_source_provenance_only": True})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "images").mkdir()
        rows = []
        for kind, index, path, expected, proof in candidates:
            relative = f"images/{kind}-{index:02d}.png"
            shutil.copyfile(path, staging / relative)
            _require(sha256(staging / relative) == expected, "copied image hash changed")
            rows.append({**metadata, "candidate_id": f"{group_id[:16]}-{kind}-{index:02d}",
                         "page_kind": kind, "page_index": index, "image_path": relative,
                         "source_frame_name": path.name, "frame_sha256": expected,
                         "runtime_selected_row_evidence": proof,
                         "runtime_selected_row_is_identity_truth": False})
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        (staging / "queue.json").write_text(json.dumps({**metadata, "candidate_count": len(rows),
             "manifest_path": "manifest.jsonl"}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _require(not output.exists(), "refusing to overwrite output directory")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"schema": QUEUE_SCHEMA, "candidate_count": len(candidates), "source_group_id": group_id,
            "output": str(output), "review_status": "pending", "training_ready": False}
