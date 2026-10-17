"""Read-only class/group coverage diagnostics for unassigned vision candidates."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from .data import (DatasetError, REVIEWED, SPLITS, _eligible, _ids,
                   _verify_near_duplicate_reviews, leakage_groups, load_taxonomy,
                   split as split_candidates, validate)


_FROZEN_ROOT_MARKERS = (
    "snapshot.json", "split.json", "train.txt", "val.txt", "test.txt",
    "reviewed-manifest.jsonl", "reviewed-scenes.jsonl", "reviewed-buildings.jsonl",
)
_FROZEN_ROW_FIELDS = ("snapshot_split", "snapshot_image_path", "computed_leakage_group_id")


def _require_unassigned(root: Path, manifest: list[dict]) -> None:
    root = Path(root).resolve()
    if not root.is_dir():
        raise DatasetError(f"Candidate image root is not a directory: {root}")
    markers = [name for name in _FROZEN_ROOT_MARKERS if (root / name).exists()]
    if markers:
        raise DatasetError(f"Refusing frozen/assigned dataset root; found {markers}")
    if not manifest:
        raise DatasetError("Candidate manifest is empty")
    for row in manifest:
        if not isinstance(row, dict):
            raise DatasetError("Candidate manifest rows must be objects")
        sample_id = row.get("sample_id", "<missing>")
        if row.get("split") != "unassigned":
            raise DatasetError(f"{sample_id}: candidate split must be explicitly 'unassigned'")
        found = [key for key in _FROZEN_ROW_FIELDS if key in row]
        if found:
            raise DatasetError(f"{sample_id}: refusing frozen manifest fields {found}")


def _scope(task: Literal["scene", "building"], classes: list[str] | None,
           taxonomy: dict) -> list[str]:
    if task == "scene":
        if classes is not None:
            raise DatasetError("Scene class order comes from taxonomy")
        return [*taxonomy["scene_classes"], taxonomy["unknown_scene"]]
    registered = [item["name"] for item in taxonomy.get("building_classes", [])]
    if classes is None:
        return registered
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(name, str) or name not in registered for name in classes)
            or len(set(classes)) != len(classes)):
        raise DatasetError("Building classes must be unique registered names")
    return list(classes)


def _annotation_complete_for(item: dict | None, task: str, class_name: str,
                             requested_classes: list[str] | None) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("complete") is True:
        return True
    return bool(task == "building" and requested_classes is not None
                and item.get("complete") is False
                and isinstance(item.get("complete_classes"), list)
                and class_name in item["complete_classes"])


def _reviewed_item(item: dict | None) -> bool:
    return bool(isinstance(item, dict) and item.get("review_status") == REVIEWED
                and item.get("reviewer") and item.get("review_evidence")
                and item.get("review_method") in {"human", "agent_visual"})


def _positive_instances(annotation: dict | None, task: str, class_name: str) -> int:
    if not isinstance(annotation, dict):
        return 0
    if task == "scene":
        item = annotation.get("scene")
        return int(isinstance(item, dict) and item.get("label") == class_name)
    item = annotation.get("buildings")
    if not isinstance(item, dict) or not isinstance(item.get("objects"), list):
        return 0
    return sum(1 for obj in item["objects"]
               if isinstance(obj, dict) and obj.get("class_name") == class_name
               and obj.get("ignore") is not True)


def _class_count(sample_ids: set[str], labels: dict[str, dict], groups: dict[str, str],
                 *, task: str, class_name: str) -> dict:
    positive_ids = {sample_id for sample_id in sample_ids
                    if _positive_instances(labels.get(sample_id), task, class_name) > 0}
    return {
        "positive_images": len(positive_ids),
        "positive_instances": sum(_positive_instances(labels.get(sample_id), task, class_name)
                                  for sample_id in positive_ids),
        "independent_groups": len({groups[sample_id] for sample_id in positive_ids}),
    }


def _reviewed_class_complete_ids(sample_ids: set[str], labels: dict[str, dict], *,
                                 task: str, class_name: str,
                                 requested_classes: list[str] | None) -> set[str]:
    task_key = "scene" if task == "scene" else "buildings"
    return {sample_id for sample_id in sample_ids
            if _reviewed_item(labels.get(sample_id, {}).get(task_key))
            and _annotation_complete_for(labels.get(sample_id, {}).get(task_key), task,
                                         class_name, requested_classes)}


def _summarize_forecast(split_doc: dict, labels: dict[str, dict], *, task: str,
                        class_names: list[str]) -> dict:
    assignments = split_doc["sample_assignments"]
    groups = split_doc["sample_groups"]
    by_split = {}
    for split_name in SPLITS:
        sample_ids = {sample_id for sample_id, assigned in assignments.items()
                      if assigned == split_name}
        by_class = {class_name: _class_count(sample_ids, labels, groups, task=task,
                                             class_name=class_name)
                    for class_name in class_names}
        by_split[split_name] = {
            "images": len(sample_ids),
            "independent_groups": len({groups[sample_id] for sample_id in sample_ids}),
            "classes": by_class,
        }
    missing = {class_name: [split_name for split_name in SPLITS
                            if by_split[split_name]["classes"][class_name]["independent_groups"] == 0]
               for class_name in class_names}
    missing = {class_name: splits for class_name, splits in missing.items() if splits}
    return {
        "status": "available_with_class_gaps" if missing else "available",
        "algorithm": split_doc["algorithm"],
        "target_sample_ratios": split_doc["target_sample_ratios"],
        "images": split_doc["counts"],
        "independent_groups": {name: by_split[name]["independent_groups"] for name in SPLITS},
        "by_split": by_split,
        "classes_missing_positive_groups": missing,
    }


def _with_group_deficit(counts: dict) -> dict:
    """Annotate the three-split minimum; this is not a release quota."""
    result = dict(counts)
    result["split_minimum_independent_groups"] = len(SPLITS)
    result["split_minimum_scope"] = (
        "overall independent groups required by the existing split algorithm; "
        "not a per-class-per-split or release quota"
    )
    result["missing_independent_groups_to_split_minimum"] = max(
        0, len(SPLITS) - result["independent_groups"])
    return result


def _forecast(root: Path, manifest: list[dict], labels: dict[str, dict], groups: dict[str, str],
              near: list[dict], *, task: Literal["scene", "building"], seed: int,
              prototype_only: bool, classes: list[str] | None,
              near_duplicate_reviews: list[dict] | None, class_names: list[str],
              taxonomy: dict) -> dict:
    near_by_pair = {tuple(sorted(item["sample_ids"])): item for item in near}
    if near_duplicate_reviews is None:
        supplied_reviews = []
    elif not isinstance(near_duplicate_reviews, list):
        raise DatasetError("Near-duplicate reviews must be a JSON list")
    else:
        supplied_reviews = near_duplicate_reviews
    review_by_pair = {}
    for review in supplied_reviews:
        if not isinstance(review, dict) or not isinstance(review.get("sample_ids"), list):
            raise DatasetError("Near-duplicate review requires a sample_ids pair")
        pair_values = review["sample_ids"]
        if any(not isinstance(item, str) for item in pair_values):
            raise DatasetError("Near-duplicate review sample_ids must be strings")
        pair = tuple(sorted(pair_values))
        if len(pair) != 2 or pair[0] == pair[1] or pair in review_by_pair:
            raise DatasetError(f"Invalid or duplicate near-duplicate review pair: {pair}")
        candidate = near_by_pair.get(pair)
        if candidate is None:
            raise DatasetError(f"Stale near-duplicate review pair: {pair}")
        _verify_near_duplicate_reviews([candidate], manifest, [review], prototype_only=True)
        review_by_pair[pair] = review

    eligible = {row["sample_id"] for row in manifest
                if _eligible(row, labels.get(row["sample_id"]), task,
                             prototype_only=prototype_only, building_classes=classes)}
    eligible_groups = {groups[sample_id] for sample_id in eligible}
    result = {
        "eligible_samples": len(eligible),
        "eligible_independent_groups": len(eligible_groups),
        "review_methods": dict(sorted(Counter(
            row.get("review_method") or "missing" for row in manifest
            if row["sample_id"] in eligible).items())),
        "client_versions": dict(sorted(Counter(
            row.get("client_version") or "unknown" for row in manifest
            if row["sample_id"] in eligible).items())),
        "forecast": None,
    }
    if not eligible:
        result["status"] = "no_eligible_samples"
        return result
    if len(eligible_groups) < len(SPLITS):
        result["status"] = "insufficient_independent_groups"
        return result

    relevant = [item for item in near
                if any(sample_id in eligible for sample_id in item["sample_ids"])]

    relevant_pairs = {tuple(sorted(item["sample_ids"])) for item in relevant}
    missing_pairs = sorted(relevant_pairs - set(review_by_pair))
    if missing_pairs:
        result.update(status="blocked_unresolved_near_duplicates",
                      unresolved_near_duplicate_pairs=[list(pair) for pair in missing_pairs])
        return result
    relevant_reviews = [review_by_pair[pair] for pair in sorted(relevant_pairs)]
    try:
        split_doc = split_candidates(
            root, manifest, list(labels.values()), task=task, seed=seed,
            prototype_only=prototype_only, classes=classes,
            near_duplicate_reviews=relevant_reviews,
            taxonomy=taxonomy,
        )
    except DatasetError as exc:
        # The same review may satisfy a prototype forecast but fail the stricter
        # formal split contract (for example, agent_visual instead of human).
        result.update(status="blocked_by_existing_split_contract", reason=str(exc))
        return result
    result["forecast"] = _summarize_forecast(split_doc, labels, task=task,
                                             class_names=class_names)
    result["status"] = result["forecast"]["status"]
    return result


def candidate_coverage_report(root: Path, manifest: list[dict], annotations: list[dict], *,
                              task: Literal["scene", "building"],
                              classes: list[str] | None = None,
                              taxonomy: dict | None = None, seed: int = 42,
                              near_duplicate_reviews: list[dict] | None = None) -> dict:
    """Diagnose an unassigned candidate pool without freezing or writing a split.

    Candidate annotations are reported as supplied. Agent-visual evidence can
    qualify for a prototype forecast only; it is never converted to human
    review or formal acceptance. Existing approximate-duplicate review rules
    remain in force.
    """
    if task not in {"scene", "building"}:
        raise DatasetError(f"Unsupported task: {task}")
    if type(seed) is not int:
        raise DatasetError("seed must be an integer")
    if not isinstance(manifest, list) or not isinstance(annotations, list):
        raise DatasetError("Candidate manifest and annotations must be lists")
    root = Path(root).resolve()
    _require_unassigned(root, manifest)
    taxonomy = taxonomy or load_taxonomy()
    class_names = _scope(task, classes, taxonomy)
    checks = validate(root, manifest, annotations, taxonomy=taxonomy, verify_pixels=True)
    if not checks["valid"]:
        raise DatasetError("Candidate input validation failed: " + "; ".join(checks["errors"][:10]))

    manifest_by_id = _ids(manifest, "sample_id")
    labels = _ids(annotations, "sample_id")
    groups, near = leakage_groups(root, manifest, near_duplicates=True)
    near = sorted(near, key=lambda item: (tuple(sorted(item["sample_ids"])), item["dhash_distance"]))
    all_ids = set(manifest_by_id)
    prototype_ids = {row["sample_id"] for row in manifest
                     if _eligible(row, labels.get(row["sample_id"]), task,
                                  prototype_only=True, building_classes=classes)}
    formal_ids = {row["sample_id"] for row in manifest
                  if _eligible(row, labels.get(row["sample_id"]), task,
                               prototype_only=False, building_classes=classes)}

    class_coverage = {}
    task_key = "scene" if task == "scene" else "buildings"
    for class_name in class_names:
        candidate_count = _class_count(all_ids, labels, groups, task=task, class_name=class_name)
        reviewed_complete = _reviewed_class_complete_ids(
            all_ids, labels, task=task, class_name=class_name,
            requested_classes=classes)
        prototype_complete = _reviewed_class_complete_ids(
            prototype_ids, labels, task=task, class_name=class_name,
            requested_classes=classes)
        formal_complete = _reviewed_class_complete_ids(
            formal_ids, labels, task=task, class_name=class_name,
            requested_classes=classes)
        class_coverage[class_name] = {
            "candidate_positives": candidate_count,
            "reviewed_complete": {
                "images": len(reviewed_complete),
                "independent_groups": len({groups[sample_id] for sample_id in reviewed_complete}),
                "meaning": "annotation-side reviewed completeness declaration only; not formal eligibility",
            },
            "prototype_eligible_positives": _with_group_deficit(_class_count(
                prototype_ids, labels, groups, task=task, class_name=class_name)),
            "prototype_eligible_complete": {
                "images": len(prototype_complete),
                "independent_groups": len({groups[sample_id] for sample_id in prototype_complete}),
                "meaning": "annotation-side reviewed completeness declaration only; not formal eligibility",
            },
            "formal_precondition_positives": _with_group_deficit(_class_count(
                formal_ids, labels, groups, task=task, class_name=class_name)),
            "formal_precondition_complete": {
                "images": len(formal_complete),
                "independent_groups": len({groups[sample_id] for sample_id in formal_complete}),
                "meaning": "annotation-side reviewed completeness declaration only; not formal eligibility",
            },
        }

    prototype = _forecast(root, manifest, labels, groups, near, task=task, seed=seed,
                          prototype_only=True, classes=classes,
                          near_duplicate_reviews=near_duplicate_reviews,
                          class_names=class_names, taxonomy=taxonomy)
    formal = _forecast(root, manifest, labels, groups, near, task=task, seed=seed,
                       prototype_only=False, classes=classes,
                       near_duplicate_reviews=near_duplicate_reviews,
                       class_names=class_names, taxonomy=taxonomy)
    # Forecasting reads the images again for dHash/grouping. Reject a source
    # changed since the initial validation rather than publishing mixed-image
    # evidence under a verified-source claim.
    final_checks = validate(root, manifest, annotations, taxonomy=taxonomy,
                            verify_pixels=True)
    if not final_checks["valid"]:
        raise DatasetError("Candidate images changed during report generation: "
                           + "; ".join(final_checks["errors"][:10]))
    return {
        "schema_version": 1,
        "report_kind": "unassigned_candidate_coverage_preflight",
        "diagnostic_only": True,
        "candidate_pool_only": True,
        "task": task,
        "building_scope": list(classes) if task == "building" and classes is not None else None,
        "seed": seed,
        "source_validation": {"valid": True, "image_hashes_and_pixels_verified": True},
        "candidate_inventory": {
            "samples": len(manifest),
            "annotations": len(annotations),
            "definite_independent_groups": len(set(groups.values())),
            "manifest_review_status": dict(sorted(Counter(
                row.get("review_status") or "missing" for row in manifest).items())),
            "manifest_review_methods": dict(sorted(Counter(
                row.get("review_method") or "missing" for row in manifest).items())),
            "annotation_review_methods": dict(sorted(Counter(
                (labels.get(sample_id, {}).get(task_key) or {}).get("review_method") or "missing"
                for sample_id in all_ids).items())),
            "client_versions": dict(sorted(Counter(
                row.get("client_version") or "unknown" for row in manifest).items())),
        },
        "classes": class_coverage,
        "near_duplicate_candidates": near,
        "qualification": {
            "prototype": {**prototype, "formal_eligibility_claim": False},
            "formal_preconditions": {**formal, "formal_eligibility_claim": False},
        },
        "formal_success": False,
        "snapshot_created": False,
        "split_file_written": False,
        "training_started": False,
        "heldout_test_accessed": False,
    }
