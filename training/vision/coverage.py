"""Read-only class, source-session and version coverage for frozen S/B snapshots."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from .data import DatasetError, read_jsonl, sha256
from .snapshot import verify_snapshot


SPLITS = ("train", "val", "test")


def _version(source: dict) -> str:
    return source.get("client_version") or "unknown"


def _counts(items: list[tuple[dict, dict]]) -> dict:
    sources = [source for _, source in items]
    return {"images": len(items),
            "instances": sum(len(row.get("boxes", [])) for row, _ in items),
            "leakage_groups": len({row["group_id"] for row, _ in items}),
            "source_sessions": len({source["source_session"] for source in sources
                                    if source.get("source_session")}),
            "reviewed_layouts": len({source["layout_group_id"] for source in sources
                                     if source.get("layout_group_id") and not
                                     source["layout_group_id"].startswith("provisional:")}),
            "versions": dict(sorted(Counter(map(_version, sources)).items())),
            "review_methods": dict(sorted(Counter(source.get("review_method") or "missing"
                                                  for source in sources).items()))}


def _deficit(value: int, threshold: int) -> int:
    return max(0, threshold - value)


def coverage_report(scene_root: Path, building_root: Path, *,
                    acceptance_path: Path, current_client_version: str) -> dict:
    """Report observed coverage and exact frozen-v1 test-set deficits.

    Formal-eligible counts require human review and the requested current client;
    no prototype is promoted or resplit by this read-only report.
    """
    if not current_client_version or current_client_version == "unknown":
        raise DatasetError("Coverage needs an explicit known client version")
    acceptance_path = Path(acceptance_path).resolve()
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    thresholds = acceptance["thresholds"]
    summaries = {}
    for task, root in (("scene", scene_root), ("building", building_root)):
        root = Path(root).resolve()
        verified = verify_snapshot(root, task)
        labels_name = "reviewed-scenes.jsonl" if task == "scene" else "reviewed-buildings.jsonl"
        labels = read_jsonl(root / labels_name)
        sources = {row["sample_id"]: row for row in read_jsonl(root / "reviewed-manifest.jsonl")}
        assignments = {path: split for split in SPLITS for path in verified["paths"][split]}
        by_split = defaultdict(list)
        by_class = defaultdict(lambda: defaultdict(list))
        for row in labels:
            source = sources[row["sample_id"]]
            split = assignments[row["image"]]
            pair = (row, source)
            by_split[split].append(pair)
            classes = [row["label"]] if task == "scene" else {box["class"] for box in row["boxes"]}
            for label in classes:
                by_class[label][split].append(pair)
        if task == "scene":
            taxonomy = json.loads((root / "taxonomy.json").read_text(encoding="utf-8"))
            all_classes = list(taxonomy["scene_classes"]) + [taxonomy["unknown_scene"]]
            enabled = sorted(by_class)
        else:
            mapping = json.loads((root / "class_mapping.json").read_text(encoding="utf-8"))
            all_classes = list(mapping["output_indices"])
            enabled = sorted(by_class)
        per_class = {}
        for label in all_classes:
            splits = {split: _counts(by_class[label][split]) for split in SPLITS}
            formal = {split: _counts([(row, source) for row, source in by_class[label][split]
                                      if source.get("review_method") == "human"
                                      and _version(source) == current_client_version])
                      for split in SPLITS}
            raw_test, eligible_test = splits["test"], formal["test"]
            if task == "scene":
                n_images = thresholds["unknown_test_samples"] if label == taxonomy["unknown_scene"] else thresholds["scene_test_samples_per_class"]
                n_groups = thresholds["unknown_test_groups"] if label == taxonomy["unknown_scene"] else thresholds["scene_test_groups_per_class"]
                deficits = {"test_images": _deficit(eligible_test["images"], n_images),
                            "test_independent_groups": _deficit(eligible_test["leakage_groups"], n_groups)}
                observed_deficits = {"test_images": _deficit(raw_test["images"], n_images),
                                     "test_independent_groups": _deficit(raw_test["leakage_groups"], n_groups)}
            else:
                n_instances = thresholds["town_hall_instances"] if label == "town_hall" else thresholds["building_test_instances_per_class"]
                n_groups = thresholds["town_hall_groups"] if label == "town_hall" else thresholds["building_test_groups_per_class"]
                deficits = {"test_instances": _deficit(eligible_test["instances"], n_instances),
                            "test_reviewed_layouts": _deficit(eligible_test["reviewed_layouts"], n_groups)}
                observed_deficits = {"test_instances": _deficit(raw_test["instances"], n_instances),
                                     "test_reviewed_layouts": _deficit(raw_test["reviewed_layouts"], n_groups)}
            per_class[label] = {"enabled_in_snapshot": label in enabled,
                                "splits": splits, "human_current_version": formal,
                                "formal_test_deficit": deficits,
                                "raw_prototype_test_deficit": observed_deficits}
        summaries[task] = {"snapshot": str(root), "snapshot_sha256": verified["snapshot_sha256"],
                           "split_sha256": verified["split_sha256"],
                           "prototype_only": verified["metadata"]["prototype_only"],
                           "review_policy": verified["metadata"]["review_policy"],
                           "enabled_classes": enabled,
                           "splits": {split: _counts(by_split[split]) for split in SPLITS},
                           "classes": per_class}
    return {"schema_version": 1, "current_client_version": current_client_version,
            "acceptance_id": acceptance["acceptance_id"],
            "acceptance_sha256": sha256(acceptance_path),
            "scope_frozen": acceptance["scope"].get("frozen") is True,
            "interpretation": "formal deficits count only human-reviewed current-version test evidence; "
                              "raw prototype deficits are diagnostic only; class rows outside enabled scope "
                              "show catalog coverage and require an explicit scope decision",
            "scene": summaries["scene"], "building": summaries["building"]}
