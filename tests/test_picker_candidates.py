"""Synthetic read-only picker runs; no captured account images are checked in."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

from training.vision.data import DatasetError
from training.vision.picker_candidates import import_picker_candidates


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def _png(value: int, scanline: bytes | None = None) -> bytes:
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(scanline if scanline is not None else bytes((0, value, 13, 29))))
            + _chunk(b"IEND", b""))


def _write_run(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    source = tmp_path / "readonly-picker-run"
    frames_dir = source / "frames"
    frames_dir.mkdir(parents=True)
    names = ["00001-home.png", "00002-menu.png", "00003-menu.png", "00004-army.png",
             "00005-army-before.png", "00006-troop-1.png", "00007-troop-2.png",
             "00008-troop-3.png", "00009-closed.png", "00010-spell-1.png",
             "00011-army-after.png"]
    frames = []
    for index, name in enumerate(names, 1):
        path = frames_dir / name
        content = _png(index)
        path.write_bytes(content)
        frames.append({"path": str(path), "sha256": _sha(content)})
    counts = [["heroes", 1, 1], ["troops", 2, 10], ["spells", 1, 2], ["siege", 1, 1]]
    roster = [["troops", [[[1, 2, 3, 4], "dragon", 2]]],
              ["spells", [[[2, 3, 4, 5], "lightning", 1]]],
              ["hero_and_siege", [["hero", "queen", 1, [1, 1, 4, 4]],
                                  ["siege", "wagon", 1, [4, 4, 8, 8]]]]]
    def army_obs(frame: str) -> dict:
        return {"kind": "observation", "frame": frame, "scene": "training", "observations": {
            "army": {"heroes": {"used": 1, "capacity": 1},
                     "troops": {"used": 2, "capacity": 10},
                     "spells": {"used": 1, "capacity": 2},
                     "siege": {"used": 1, "capacity": 1},
                     "manifest": {"frame": frame, "layout": "my_army_rows_v1",
                                  "supported_layout": True, "complete": True,
                                  "identity_complete": True, "unknowns": [],
                                  "complete_kinds": {"troop": True, "spell": True},
                                  "troops": [{"kind": "troop", "source": "army", "unit_id": "dragon",
                                              "count": 2, "bbox": [1, 2, 3, 4]}],
                                  "spells": [{"kind": "spell", "source": "army", "unit_id": "lightning",
                                              "count": 1, "bbox": [2, 3, 4, 5]}]},
                     "identity_coverage": {"hero": True, "siege": True},
                     "hero_evidence": {"frame": frame, "complete": True, "unknowns": []},
                     "siege_evidence": {"frame": frame, "complete": True, "unknowns": []},
                     "identity_cards": [{"kind": "hero", "unit_id": "queen", "count": 1,
                                         "bbox": [1, 1, 4, 4]},
                                        {"kind": "siege", "unit_id": "wagon", "count": 1,
                                         "bbox": [4, 4, 8, 8]}]}}}
    def picker_obs(frame: str, kind: str) -> dict:
        cards = ([{"kind": "troop", "source": "army", "unit_id": "dragon", "count": 2}]
                 if kind == "troop" else
                 [{"kind": "spell", "source": "army", "unit_id": "lightning", "count": 1}])
        return {"kind": "observation", "frame": frame, "scene": "training", "observations": {
            "army": None if kind == "spell" else {"troops": {"used": 2, "capacity": 10}},
            "army_editor": {"frame": frame, "surface": "current_picker", "ready": True,
                            "editing_kind": kind, "complete_kinds": {kind: True}, "unknowns": [],
                            "capacities": {kind: {"used": 2 if kind == "troop" else 1,
                                                  "total": 10 if kind == "troop" else 2}},
                            "cards": cards}}}
    troop_proof = {"kind": "troop", "capacity": {"used": 2, "total": 10},
                   "selected": [["dragon", 2]], "army_header_present": True}
    spell_proof = {"kind": "spell", "capacity": {"used": 1, "total": 2},
                   "selected": [["lightning", 1]], "army_header_present": False}
    pages = []
    events = [army_obs(frames[4]["path"])]
    for index in range(1, 4):
        frame = frames[index + 4]
        page = {"index": index, "frame": frame["path"], "sha256": frame["sha256"],
                "identity_labels": None, "visible_roster_proof": troop_proof}
        events.append(picker_obs(frame["path"], "troop"))
        events.append({"kind": "development_readonly_picker_page", **page})
        if index < 3:
            next_frame = frames[index + 5]
            page["swipe_result"] = {"verified": True, "after_frame": next_frame["path"],
                                    "after_sha256": next_frame["sha256"]}
            events.append({"kind": "development_readonly_picker_page_change", "index": index,
                           "before_frame": frame["path"], "before_sha256": frame["sha256"],
                           **page["swipe_result"]})
        pages.append(page)
    events.append({"kind": "development_readonly_picker_termination", "reason": "configured_page_limit_not_catalog_end",
                   "frame": frames[7]["path"], "frame_sha256": frames[7]["sha256"], "pages_seen": 3})
    events.append(picker_obs(frames[9]["path"], "spell"))
    events.append(army_obs(frames[10]["path"]))
    events_path = source / "events.jsonl"
    events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    summary = {"kind": "agent_development_readonly_picker_sampling_v2", "status": "completed",
               "reason": "read_only_capture_sequence_completed", "counts_toward_acceptance": False,
               "formal_acceptance": False, "human_review": False, "unit_identity_labels": None,
               "client_version": "18.647.2", "client_version_code": 180647003,
               "action_count": 8, "elapsed_sec": 39.227,
               "head_before": "a" * 40, "head_after": "a" * 40, "source_status_after": "",
               "source_sha256_before": {name: "b" * 64 for name in (
                   "src/autococ/army_navigation.py", "src/autococ/army_editor.py",
                   "src/autococ/session.py", "src/autococ/vision.py")},
               "source_sha256_after": {name: "b" * 64 for name in (
                   "src/autococ/army_navigation.py", "src/autococ/army_editor.py",
                   "src/autococ/session.py", "src/autococ/vision.py")},
               "config_sha256_before": "c" * 64, "config_sha256_after": "c" * 64,
               "runner_sha256_before": "d" * 64, "runner_sha256_after": "d" * 64,
               "events_path": str(events_path), "events_sha256": _sha(events_path.read_bytes()),
               "frames": frames,
               "flow": {"army_before": {**frames[4], "frame": frames[4]["path"],
                                        "counts": counts, "roster": roster},
                        "army_after": {**frames[10], "frame": frames[10]["path"],
                                       "counts": counts, "roster": roster},
                        "troop_pages": pages,
                        "troop_termination": "configured_page_limit_not_catalog_end",
                        "spell_first_page": {"frame": frames[9]["path"], "sha256": frames[9]["sha256"],
                                             "identity_labels": None, "swipe_attempted": False,
                                             "visible_roster_proof": spell_proof},
                        "zero_roster_change_verified": True,
                        "calibration": {"first_troop_frame": {"path": str(tmp_path / "calibration-troop.png"),
                                                              "sha256": "e" * 64},
                                        "first_spell_frame": {"path": str(tmp_path / "calibration-spell.png"),
                                                              "sha256": "f" * 64},
                                        "troop_swipe_events": []}}}
    _save_summary(source, summary)
    return source, summary, events


def _save_summary(source: Path, summary: dict) -> None:
    (source / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _save_events(source: Path, summary: dict, events: list[dict]) -> None:
    data = "".join(json.dumps(row) + "\n" for row in events).encode("utf-8")
    (source / "events.jsonl").write_bytes(data)
    summary["events_sha256"] = _sha(data)
    _save_summary(source, summary)


def test_import_copies_only_picker_pages_and_keeps_pending_evidence(tmp_path: Path) -> None:
    source, summary, _events = _write_run(tmp_path)
    output = tmp_path / "queue"
    report = import_picker_candidates(source, output)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    queue = json.loads((output / "queue.json").read_text())
    assert report["candidate_count"] == queue["candidate_count"] == len(rows) == 4
    assert [(r["page_kind"], r["page_index"]) for r in rows] == [
        ("troop", 1), ("troop", 2), ("troop", 3), ("spell", 1)]
    assert {p.name for p in (output / "images").iterdir()} == {
        "troop-01.png", "troop-02.png", "troop-03.png", "spell-01.png"}
    for row, source_page in zip(rows, summary["flow"]["troop_pages"] + [summary["flow"]["spell_first_page"]]):
        assert (output / row["image_path"]).read_bytes() == Path(source_page["frame"]).read_bytes()
        assert row["frame_sha256"] == _sha((output / row["image_path"]).read_bytes())
        assert row["source_frame_name"] == Path(source_page["frame"]).name
        assert row["source_group_id"] == queue["source_group_id"]
        assert row["source_summary_sha256"] == _sha((source / "summary.json").read_bytes())
        assert row["source_events_sha256"] == summary["events_sha256"]
        assert row["review_status"] == "pending" and row["human_review"] is False
        assert row["formal_acceptance"] is False and row["training_ready"] is False
        assert row["label"] is None and row["identity_labels"] is None and row["scene_labels"] is None
        assert row["building_labels"] is None and row["split"] is None
        assert row["runtime_selected_row_is_identity_truth"] is False
        assert str(tmp_path) not in json.dumps(row)


@pytest.mark.parametrize("mutation", [
    "status", "version", "source", "config", "capacity", "roster", "zero_flag",
    "hash", "bad_png", "bad_pixels", "escape", "duplicate", "missing_page", "events_hash",
    "absolute_source_key",
])
def test_rejects_unverified_run_without_publishing(tmp_path: Path, mutation: str) -> None:
    source, summary, events = _write_run(tmp_path)
    flow = summary["flow"]
    if mutation == "status":
        summary["status"] = "failed"
    elif mutation == "version":
        summary["client_version_code"] = None
    elif mutation == "source":
        summary["source_sha256_after"]["src/autococ/session.py"] = "0" * 64
    elif mutation == "config":
        summary["config_sha256_after"] = "0" * 64
    elif mutation == "capacity":
        flow["army_after"]["counts"][1][1] = 3
    elif mutation == "roster":
        flow["army_after"]["roster"][0][1][0][2] = 3
    elif mutation == "zero_flag":
        flow["zero_roster_change_verified"] = False
    elif mutation == "hash":
        flow["troop_pages"][0]["sha256"] = "0" * 64
    elif mutation in ("bad_png", "bad_pixels"):
        path = Path(flow["troop_pages"][0]["frame"])
        path.write_bytes(b"not a png" if mutation == "bad_png" else _png(1, b"\x00"))
        digest = _sha(path.read_bytes())
        flow["troop_pages"][0]["sha256"] = digest
        summary["frames"][5]["sha256"] = digest
        for row in events:
            if row.get("kind") == "development_readonly_picker_page" and row.get("index") == 1:
                row["sha256"] = digest
            if row.get("kind") == "development_readonly_picker_page_change" and row.get("index") == 1:
                row["before_sha256"] = digest
        _save_events(source, summary, events)
    elif mutation == "escape":
        flow["troop_pages"][0]["frame"] = str(tmp_path / "outside.png")
    elif mutation == "duplicate":
        flow["troop_pages"][1]["frame"] = flow["troop_pages"][0]["frame"]
        flow["troop_pages"][1]["sha256"] = flow["troop_pages"][0]["sha256"]
    elif mutation == "missing_page":
        flow["troop_pages"].pop()
    elif mutation == "events_hash":
        summary["events_sha256"] = "0" * 64
    elif mutation == "absolute_source_key":
        summary["source_sha256_before"][str(tmp_path / "outside.py")] = "b" * 64
        summary["source_sha256_after"][str(tmp_path / "outside.py")] = "b" * 64
    _save_summary(source, summary)
    output = tmp_path / "queue"
    with pytest.raises(DatasetError):
        import_picker_candidates(source, output)
    assert not output.exists()


def test_existing_output_is_left_untouched(tmp_path: Path) -> None:
    source, _, _ = _write_run(tmp_path)
    output = tmp_path / "queue"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unrelated", encoding="utf-8")
    with pytest.raises(DatasetError, match="overwrite"):
        import_picker_candidates(source, output)
    assert marker.read_text(encoding="utf-8") == "unrelated"


def test_output_cannot_modify_source_run(tmp_path: Path) -> None:
    source, _, _ = _write_run(tmp_path)
    with pytest.raises(DatasetError, match="outside source"):
        import_picker_candidates(source, source / "derived")
    assert not (source / "derived").exists()


def test_symlinked_frames_directory_cannot_escape_run(tmp_path: Path) -> None:
    source, _, _ = _write_run(tmp_path)
    outside = tmp_path / "outside-frames"
    (source / "frames").rename(outside)
    try:
        os.symlink(outside, source / "frames", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(DatasetError, match="directory escapes"):
        import_picker_candidates(source, tmp_path / "queue")
    assert not (tmp_path / "queue").exists()


def test_summary_formatting_does_not_change_source_group(tmp_path: Path) -> None:
    source, summary, _ = _write_run(tmp_path)
    first = import_picker_candidates(source, tmp_path / "first")
    (source / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=4) + "\n", encoding="utf-8")
    second = import_picker_candidates(source, tmp_path / "second")
    assert first["source_group_id"] == second["source_group_id"]
    first_row = json.loads((tmp_path / "first" / "manifest.jsonl").read_text().splitlines()[0])
    second_row = json.loads((tmp_path / "second" / "manifest.jsonl").read_text().splitlines()[0])
    assert first_row["candidate_id"] == second_row["candidate_id"]
    assert first_row["source_summary_sha256"] != second_row["source_summary_sha256"]


def test_cli_import_picker_candidates(tmp_path: Path) -> None:
    source, _, _ = _write_run(tmp_path)
    output = tmp_path / "queue"
    script = Path(__file__).resolve().parents[1] / "scripts" / "vision_data.py"
    completed = subprocess.run([sys.executable, str(script), "import-picker-candidates",
                                "--source-run", str(source), "--output", str(output)],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["candidate_count"] == 4
    assert (output / "queue.json").is_file()
