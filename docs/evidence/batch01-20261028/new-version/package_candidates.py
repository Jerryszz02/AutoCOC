"""Package completed 18.647.5 collection windows as unreviewed building candidates.

The originals, failed windows, and frozen historical datasets are read only.
Every packaged image remains pending; this script does not assign a split or
claim that preview frames are independent layouts or reviewed labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


SOURCE_ROOT = Path(r"E:\projects\AutoCOC-training\candidates")
SOURCE_PREFIX = "building-186475-20261028-window"
WINDOWS = range(2, 7)
CLIENT_VERSION = "18.647.5"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def image_at(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid relative image path: {relative!r}")
    result = (root / relative.replace("\\", "/")).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes root: {relative!r}")
    return result


def pixels(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        size = image.size
        return size, hashlib.sha256(str(size).encode("ascii") + image.tobytes()).hexdigest()


def check_pending_label(label: dict, source: dict, collector_copy: Path) -> None:
    sample_id = source["sample_id"]
    required = {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_sha256": source["file_sha256"],
        "client_version": CLIENT_VERSION,
        "split": "unassigned",
        "export_eligible": False,
        "prototype_only": True,
        "declared_complete_classes": [],
    }
    for key, expected in required.items():
        if label.get(key) != expected:
            raise ValueError(f"{sample_id}: label {key} must be {expected!r}")
    image = label.get("source_image")
    if not isinstance(image, str) or Path(image).resolve() not in {
            Path(source["source_ref"]).resolve(), collector_copy.resolve()}:
        raise ValueError(f"{sample_id}: trial image is not the bound original or collector copy")
    if digest(Path(image)) != source["file_sha256"]:
        raise ValueError(f"{sample_id}: trial image bytes differ from original")
    if label.get("review_status") not in (None, "pending"):
        raise ValueError(f"{sample_id}: trial review must remain pending")
    buildings = label.get("buildings")
    if not isinstance(buildings, dict) or buildings.get("complete") is not False or \
            buildings.get("review_status") != "pending" or \
            not isinstance(buildings.get("objects"), list):
        raise ValueError(f"{sample_id}: partial pending building labels required")
    if buildings.get("complete_classes") is not None:
        raise ValueError(f"{sample_id}: no exhaustive class may be declared")
    if label.get("review_method") not in ("agent_visual", None) or \
            buildings.get("review_method") not in ("agent_visual", None):
        raise ValueError(f"{sample_id}: unsupported trial review method")
    if label.get("reviewer") or label.get("review_evidence"):
        raise ValueError(f"{sample_id}: trial labels cannot claim a reviewer")
    if label.get("source_group") not in (None, source["leakage_group_id"]):
        raise ValueError(f"{sample_id}: source group differs from original")
    seen = set()
    for item in buildings["objects"]:
        if not isinstance(item, dict) or item.get("review_status") != "pending" or \
                item.get("candidate_status") != "provisional":
            raise ValueError(f"{sample_id}: object is not a pending proposal")
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or not object_id or object_id in seen:
            raise ValueError(f"{sample_id}: invalid or repeated object ID")
        seen.add(object_id)
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box) or \
                not (0 <= box[0] < box[2] <= source["width"] and
                     0 <= box[1] < box[3] <= source["height"]):
            raise ValueError(f"{sample_id}: invalid candidate box")


def pending_annotation(source: dict) -> dict:
    return {
        "schema_version": 1,
        "sample_id": source["sample_id"],
        "source_image": source["source_ref"],
        "source_sha256": source["file_sha256"],
        "source_group": source["leakage_group_id"],
        "client_version": CLIENT_VERSION,
        "split": "unassigned",
        "review_status": "pending",
        "review_method": None,
        "reviewer": None,
        "review_evidence": None,
        "prototype_only": True,
        "export_eligible": False,
        "declared_complete_classes": [],
        "buildings": {
            "complete": False,
            "review_status": "pending",
            "objects": [],
            "exclusion_reason": "No complete class census or independent review; absence of a box is unknown.",
        },
        "unknown_regions": [],
    }


def collect_source_rows() -> tuple[list[tuple[int, Path, Path, dict]], list[dict]]:
    accepted = []
    windows = []
    for number in WINDOWS:
        root = SOURCE_ROOT / f"{SOURCE_PREFIX}{number:02d}"
        report_path = root / "collection.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        status = report.get("status")
        if status == "running":
            raise ValueError(f"Window {number} is still running; wait for final collection report")
        item = {"window": number, "status": status, "report": str(report_path),
                "report_sha256": digest(report_path), "samples": report.get("samples")}
        if status != "completed":
            item["packaged"] = False
            windows.append(item)
            continue
        n = report.get("samples")
        if not (type(n) is int and 1 <= n <= 10 and report.get("client_version") == CLIENT_VERSION and
                report.get("returned_home") is True and report.get("fee_reconciled") is True and
                report.get("actual_search_cost_gold") == n * 1700 and
                report.get("quoted_confirmed_gold") == n * 1700):
            raise ValueError(f"Completed window {number} lacks version/fee/return-home proof")
        manifest_path = root / "unreviewed-manifest.jsonl"
        session = SOURCE_ROOT / f"{SOURCE_PREFIX}{number:02d}-session"
        event_path = session / "events.jsonl"
        if not manifest_path.is_file() or not event_path.is_file():
            raise ValueError(f"Window {number} lacks raw manifest or session event log")
        rows = jsonl(manifest_path)
        if len(rows) != n or len(report.get("attempts", [])) != n or any(
                attempt.get("state") != "candidate_confirmed" for attempt in report["attempts"]):
            raise ValueError(f"Window {number} sample count differs from confirmed attempts")
        item.update({"packaged": True, "fee_gold": n * 1700, "source_session": report.get("session_id"),
                     "source_manifest": str(manifest_path), "source_manifest_sha256": digest(manifest_path),
                     "session_events": str(event_path), "session_events_sha256": digest(event_path)})
        windows.append(item)
        for row in rows:
            if row.get("client_version") != CLIENT_VERSION or row.get("split") != "unassigned" or \
                    row.get("review_status") != "pending" or \
                    row.get("source_session") != report.get("session_id") or \
                    row.get("battle_id") != report.get("session_id") or \
                    row.get("layout_group_id") != f"provisional:{report['session_id']}" or \
                    row.get("leakage_group_id") != f"collection:{report['session_id']}" or \
                    row.get("scene_candidate") != "enemy_village":
                raise ValueError(f"Window {number} has an unexpected manifest source: {row.get('sample_id')}")
            source = image_at(root, row.get("image_path"))
            original = Path(row.get("source_ref", ""))
            if not source.is_file() or not original.is_file() or \
                    not original.resolve().is_relative_to(session.resolve()):
                raise ValueError(f"Window {number} has a missing or foreign source frame")
            if digest(source) != row.get("file_sha256") or digest(original) != row["file_sha256"]:
                raise ValueError(f"Window {number} source bytes differ: {row['sample_id']}")
            size, pixel_hash = pixels(source)
            if list(size) != [row.get("width"), row.get("height")] or pixel_hash != row.get("pixel_sha256"):
                raise ValueError(f"Window {number} source pixels differ: {row['sample_id']}")
            accepted.append((number, root, original, row))
    if not accepted:
        raise ValueError("No completed, reconciled collection window has source images")
    return accepted, windows


def contact_sheet(paths: list[tuple[str, Path]], output: Path) -> None:
    columns, tile_width, tile_height, image_height, band = 5, 410, 264, 230, 34
    rows = (len(paths) + columns - 1) // columns
    board = Image.new("RGB", (columns * tile_width, rows * tile_height), "#111827")
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    for index, (sample_id, image_path) in enumerate(paths):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        with Image.open(image_path) as original:
            image = original.convert("RGB")
            image.thumbnail((tile_width, image_height), Image.Resampling.LANCZOS)
            # The thumbnail is for layout browsing. Mask opponent and account
            # name strips; raw copies retain their original pixels and hashes.
            mask = ImageDraw.Draw(image)
            mask.rectangle((0, 0, 102, 32), fill="#111827")
            mask.rectangle((image.width - 102, 0, image.width, 32), fill="#111827")
            board.paste(image, (x, y))
        draw.rectangle((x, y + image_height, x + tile_width, y + tile_height), fill="#111827")
        draw.text((x + 5, y + image_height + 8), sample_id, font=font, fill="white")
    board.save(output, format="JPEG", quality=82, optimize=True)


def package(labels_path: Path, output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    candidates, windows = collect_source_rows()
    ids = [row[3]["sample_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample ID across windows")
    indexed = {row[3]["sample_id"]: (row[1], row[3]) for row in candidates}
    label_rows = jsonl(labels_path)
    labels = {}
    for row in label_rows:
        sample_id = row.get("sample_id")
        if sample_id not in indexed or sample_id in labels:
            raise ValueError(f"Unknown or repeated trial annotation: {sample_id!r}")
        source_root, source_row = indexed[sample_id]
        check_pending_label(row, source_row, image_at(source_root, source_row["image_path"]))
        labels[sample_id] = row

    output.mkdir(parents=True)
    marker = output / "INCOMPLETE"
    marker.write_text("Packaging not yet verified; do not use this directory.\n", encoding="utf-8")
    try:
        copies = []
        manifest = []
        annotations = []
        thumbnails = []
        by_file = defaultdict(list)
        by_pixel = defaultdict(list)
        for number, source_root, original, row in candidates:
            sample_id = row["sample_id"]
            relative = f"raw/window{number:02d}/{sample_id}.png"
            destination = image_at(output, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_at(source_root, row["image_path"]), destination)
            if digest(destination) != row["file_sha256"] or digest(original) != row["file_sha256"]:
                raise ValueError(f"Copy or original SHA changed during packaging: {sample_id}")
            size, pixel_hash = pixels(destination)
            if list(size) != [row["width"], row["height"]] or pixel_hash != row["pixel_sha256"]:
                raise ValueError(f"Copied pixels changed during packaging: {sample_id}")
            staged = {**row, "image_path": relative, "intended_uses": ["building_trial_candidate"],
                      "prototype_only": True, "export_eligible": False,
                      "near_duplicate_review_status": "pending"}
            manifest.append(staged)
            if sample_id in labels:
                trial = dict(labels[sample_id])
                trial["trial_review_source_image"] = trial["source_image"]
                trial["source_image"] = row["source_ref"]
                trial["review_status"] = "pending"
                trial["review_method"] = "agent_visual" if trial["buildings"]["objects"] else None
                annotations.append(trial)
            else:
                annotations.append(pending_annotation(row))
            copies.append({"sample_id": sample_id, "window": number, "source": str(original),
                           "original_sha256": digest(original), "collector_copy": str(source_root / row["image_path"]),
                           "collector_copy_sha256": row["file_sha256"], "packaged": relative,
                           "packaged_sha256": digest(destination), "pixel_sha256": pixel_hash})
            thumbnails.append((sample_id, destination))
            by_file[row["file_sha256"]].append(sample_id)
            by_pixel[row["pixel_sha256"]].append(sample_id)
        for name, rows in (("manifest.jsonl", manifest), ("annotations.jsonl", annotations)):
            with (output / name).open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        contact_sheet(thumbnails, output / "contactsheet.jpg")
        report = {
            "kind": "batch1_new_client_pending_building_candidates_v1",
            "status": "pending_review",
            "client_version": CLIENT_VERSION,
            "candidate_root": str(output),
            "sample_count": len(manifest),
            "trial_box_image_count": len(labels),
            "unexamined_image_count": len(manifest) - len(labels),
            "source_windows": windows,
            "source_window_count": sum(item["packaged"] for item in windows),
            "source_labels": str(labels_path.resolve()),
            "source_labels_sha256": digest(labels_path),
            "raw_copies": copies,
            "exact_file_repeats": [v for v in by_file.values() if len(v) > 1],
            "exact_pixel_repeats": [v for v in by_pixel.values() if len(v) > 1],
            "client_and_image_sha_verified": True,
            "layout_groups_provisional": True,
            "new_independent_layouts_claimed": 0,
            "split": "unassigned",
            "complete_classes": [],
            "human_ground_truth": False,
            "prototype_eligible": False,
            "formal_data_freeze": False,
            "training_started": False,
            "manifest_sha256": digest(output / "manifest.jsonl"),
            "annotations_sha256": digest(output / "annotations.jsonl"),
            "contactsheet_sha256": digest(output / "contactsheet.jpg"),
        }
        (output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        marker.unlink()
        return report
    except Exception:
        # The marker keeps partial copies visibly unusable; never delete raw evidence.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="Pending agent_visual trial labels for window02")
    parser.add_argument("--output", type=Path, required=True,
                        help="A new directory outside the repository")
    args = parser.parse_args()
    report = package(args.labels, args.output)
    print(json.dumps({key: report[key] for key in ("status", "sample_count", "trial_box_image_count",
                                                "source_window_count", "candidate_root")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
