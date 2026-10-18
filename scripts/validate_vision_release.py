"""Gate a reviewed vision release against frozen scope and versioned evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.vision.snapshot import TrainingError, verify_snapshot


DEFAULT_ACCEPTANCE = Path(__file__).resolve().parents[1] / "training/vision/acceptance-v1.json"
REQUIRED_THRESHOLDS = set(json.loads(DEFAULT_ACCEPTANCE.read_text(encoding="utf-8"))["thresholds"])


def _load(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


_INPUT_FILES = ("acceptance_config", "scene_report", "building_report", "ap_stability_report",
                "parity_report", "windows_report", "core_entry_report", "dynamic_report")


def _input_manifest(args, report: dict) -> dict:
    """Keep the small source reports beside the result for portable revalidation."""
    output = args.output.resolve()
    audit = output.parent / f"{output.stem}.inputs"
    if audit.exists():
        raise ValueError(f"Refusing to overwrite {audit}")
    audit.mkdir()
    files = {}
    for key in _INPUT_FILES:
        source = getattr(args, key)
        if source is None:
            continue
        target = audit / f"{key}.json"
        shutil.copyfile(source, target)
        files[key] = {"path": target.relative_to(output.parent).as_posix(),
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    datasets = {}
    for kind in ("scene", "building"):
        path = getattr(args, f"{kind}_dataset")
        if path is None:
            continue
        datasets[kind] = {"path": os.path.relpath(path.resolve(), output.parent).replace("\\", "/"),
                          "snapshot_sha256": report["evidence"].get(f"{kind}_dataset_sha256"),
                          "split_sha256": report["evidence"].get(f"{kind}_split_sha256")}
    return {"schema_version": 1, "files": files, "datasets": datasets}


def _snapshot(path: Path | None, task: str, versions: list[str]) -> tuple[dict | None, str]:
    if path is None:
        return None, f"missing {task} dataset snapshot"
    try:
        verified = verify_snapshot(path, task)
        meta = verified["metadata"]
        if (meta.get("schema_version") != 1 or meta.get("task") != task
                or meta.get("prototype_only") is not False
                or meta.get("review_policy") != "human_reviewed"
                or meta.get("review_methods") != ["human"]):
            return None, f"{task} snapshot is prototype-only or review policy incompatible"
        if sorted(meta.get("client_versions", [])) != sorted(versions):
            return None, f"{task} snapshot client versions differ from frozen scope"
    except (OSError, ValueError, KeyError, TrainingError) as exc:
        return None, f"{task} snapshot invalid: {exc}"
    return verified, ""


def validate_release(*, acceptance: dict, scene_report: dict | None = None,
                     building_report: dict | None = None, parity: dict | None = None,
                     windows: dict | None = None, core_entry: dict | None = None,
                     dynamic: dict | None = None, ap_stability: dict | None = None,
                     scene_dataset: Path | None = None,
                     building_dataset: Path | None = None) -> dict:
    missing: list[str] = []
    failures: list[str] = []
    evidence: dict = {}

    def need(condition: bool, message: str) -> None:
        if not condition:
            missing.append(message)

    def need_count(value, threshold, message: str) -> None:
        if not _finite(value) or value < threshold:
            missing.append(f"{message}: {value!r} < {threshold}")

    def match_evidence(actual, expected, message: str) -> None:
        if actual is None:
            missing.append(message + " missing")
        elif actual != expected:
            failures.append(message + " mismatch")

    def pass_threshold(value, threshold, message: str, *, minimum: bool = True) -> None:
        if not _finite(value):
            missing.append(message + " missing")
        elif (value < threshold if minimum else value > threshold):
            failures.append(f"{message}: {value} vs {'min' if minimum else 'max'} {threshold}")

    scope = acceptance.get("scope") if isinstance(acceptance.get("scope"), dict) else {}
    t = acceptance.get("thresholds", {})
    need(acceptance.get("schema_version") == 1 and bool(acceptance.get("acceptance_id")), "acceptance config not versioned")
    need(isinstance(t, dict) and REQUIRED_THRESHOLDS <= set(t), "acceptance thresholds incomplete")
    if not isinstance(t, dict) or not REQUIRED_THRESHOLDS <= set(t):
        return {"status": "INSUFFICIENT_DATA", "missing": missing, "failures": failures,
                "evidence": evidence}
    for key in ("building_ap50_min", "building_map50_95_min"):
        if not _finite(t.get(key)) or not 0 <= t[key] <= 1:
            missing.append(f"{key} AP quality threshold not frozen")
    versions = scope.get("client_versions", []) if isinstance(scope, dict) else []
    scene_classes = scope.get("scene_classes", []) if isinstance(scope, dict) else []
    building_classes = scope.get("building_classes", []) if isinstance(scope, dict) else []
    need(scope.get("frozen") is True, "support scope not frozen")
    need(isinstance(versions, list) and bool(versions) and all(isinstance(v, str) and v and v != "unknown" for v in versions),
         "client versions not frozen")
    need(isinstance(scene_classes, list) and bool(scene_classes), "scene classes not frozen")
    need(isinstance(building_classes, list) and bool(building_classes), "building classes not frozen")
    need(isinstance(scope.get("frozen_at_utc"), str) and bool(scope["frozen_at_utc"]),
         "acceptance freeze time missing")
    need(isinstance(scope.get("evaluation_protocol_sha256"), str) and len(scope["evaluation_protocol_sha256"]) == 64,
         "evaluation protocol hash not frozen")
    need(all(label in scene_classes for label in t["scene_critical_classes"]), "critical scene classes missing from frozen scope")
    need("town_hall" in building_classes, "town_hall missing from frozen building scope")
    if not all((isinstance(versions, list), isinstance(scene_classes, list), isinstance(building_classes, list))):
        return {"status": "INSUFFICIENT_DATA", "missing": missing, "failures": failures, "evidence": evidence}
    verified_snapshots = {}
    for task, dataset in (("scene", scene_dataset), ("building", building_dataset)):
        verified, reason = _snapshot(dataset, task, versions)
        verified_snapshots[task] = verified
        if verified is None:
            (missing if dataset is None or not dataset.exists() else failures).append(reason)
        evidence[f"{task}_snapshot_verified"] = verified is not None
        if verified is not None:
            evidence[f"{task}_dataset_sha256"] = verified["snapshot_sha256"]
            evidence[f"{task}_split_sha256"] = verified["split_sha256"]

    def check_report_scope(report: dict, dataset: Path | None, task: str, expected_split: str = "test") -> None:
        scope_report = report.get("evaluation_scope", {})
        match_evidence(scope_report.get("status"), "frozen_split",
                       f"{task} evaluation scope status")
        match_evidence(scope_report.get("split"), expected_split,
                       f"{task} evaluation split")
        verified = verified_snapshots["building" if task.startswith("building") else "scene"]
        if verified is not None:
            meta = verified["metadata"]
            count_message = f"{task} evaluation image count"
            expected_count = len(verified["paths"][expected_split])
            has_count = "image_count" in scope_report
            has_legacy = "test_image_count" in scope_report
            if not has_count and not has_legacy:
                missing.append(count_message + " missing")
            else:
                count = scope_report["image_count" if has_count else "test_image_count"]
                if type(count) is not int:
                    failures.append(count_message + " invalid type")
                elif count != expected_count:
                    failures.append(count_message + " mismatch")
                if has_count and has_legacy:
                    legacy = scope_report["test_image_count"]
                    if type(legacy) is not int:
                        failures.append(count_message + " legacy invalid type")
                    elif type(count) is int and legacy != count:
                        failures.append(count_message + " new/legacy conflict")
            match_evidence(report.get("evaluation", {}).get("samples"), len(verified["paths"][expected_split]),
                           f"{task} measured sample count")
            match_evidence(scope_report.get("dataset_sha256"), verified["snapshot_sha256"],
                           f"{task} evaluation snapshot hash")
            match_evidence(scope_report.get("split_sha256"), verified["split_sha256"],
                           f"{task} evaluation split hash")
            match_evidence(scope_report.get("data_version"), meta.get("data_version"),
                           f"{task} evaluation data version")
            match_evidence(report.get("metadata", {}).get("data_version"), meta.get("data_version"),
                           f"{task} model data version")
            validation = report.get("metadata", {}).get("validation")
            if not isinstance(validation, dict):
                validation = {}
            match_evidence(validation.get("prototype_only"), False,
                           f"{task} model prototype_only")
            match_evidence(validation.get("snapshot_sha256"), verified["snapshot_sha256"],
                           f"{task} training snapshot hash")
            match_evidence(validation.get("split_sha256"), verified["split_sha256"],
                           f"{task} training split hash")

    if scene_report is None:
        missing.append("scene evaluation report missing")
    else:
        evidence["scene_model_sha256"] = scene_report.get("sha256")
        need(_sha256(scene_report.get("sha256")), "scene model SHA-256 missing")
        evidence["scene_model_contract_sha256"] = scene_report.get("model_contract_sha256")
        need(_sha256(scene_report.get("model_contract_sha256")), "scene model contract SHA-256 missing")
        match_evidence(scene_report.get("metadata", {}).get("model_contract_sha256"),
                       scene_report.get("model_contract_sha256"), "scene model contract metadata")
        scene = scene_report.get("evaluation", {})
        check_report_scope(scene_report, scene_dataset, "scene")
        need(scene_report.get("kind") == "scene" and scene.get("kind") == "scene", "invalid scene report")
        need(scene_report.get("metadata", {}).get("data_version"), "scene model data version missing")
        if _finite(scene.get("failed")) and scene["failed"] > 0:
            failures.append("scene inference/label errors present")
        else:
            need(scene.get("failed") == 0, "scene inference/label status missing")
        need(set(scene_report.get("classes", [])) == set(scene_classes), "scene classes differ from frozen scope")
        per = scene.get("per_class", {})
        f1s = []
        for label in scene_classes:
            item = per.get(label, {})
            count = item.get("tp", 0) + item.get("fn", 0)
            need_count(count, t["scene_test_samples_per_class"], f"scene {label} test samples")
            need_count(item.get("truth_groups"), t["scene_test_groups_per_class"], f"scene {label} test groups")
            if _finite(item.get("f1")):
                f1s.append(item["f1"])
            else:
                missing.append(f"scene {label} F1 missing")
            if label in t["scene_critical_classes"]:
                pass_threshold(item.get("recall"), t["scene_critical_recall"], f"scene {label} recall")
        if len(f1s) == len(scene_classes):
            pass_threshold(sum(f1s) / len(f1s), t["scene_macro_f1"], "scene macro F1")
        pass_threshold(scene.get("coverage_known"), t["scene_known_coverage"], "scene known coverage")
        need_count(scene.get("unknown_truth_samples"), t["unknown_test_samples"], "unknown test samples")
        need_count(scene.get("unknown_truth_groups"), t["unknown_test_groups"], "unknown test groups")
        pass_threshold(scene.get("unknown_actionable_false_accepts"), t["unknown_actionable_false_accepts"],
                       "unknown actionable false accepts", minimum=False)
        matrix = scene.get("confusion_matrix", {})
        for blocker in ("popup", "disconnected", "maintenance"):
            need(blocker in scene_classes, f"blocking scene {blocker} not in frozen scope")
            if blocker in scene_classes:
                actionable = sum(matrix.get(blocker, {}).get(predicted, 0) for predicted in t["actionable_scenes"])
                pass_threshold(actionable, 0, f"{blocker} actionable misclassifications", minimum=False)
        evidence["scene_samples"] = scene.get("samples")

    if building_report is None:
        missing.append("building evaluation report missing")
    else:
        evidence["building_model_sha256"] = building_report.get("sha256")
        need(_sha256(building_report.get("sha256")), "building model SHA-256 missing")
        evidence["building_model_contract_sha256"] = building_report.get("model_contract_sha256")
        need(_sha256(building_report.get("model_contract_sha256")), "building model contract SHA-256 missing")
        match_evidence(building_report.get("metadata", {}).get("model_contract_sha256"),
                       building_report.get("model_contract_sha256"), "building model contract metadata")
        runtime = building_report.get("evaluation", {})
        independent = building_report.get("independent_ap", {})
        check_report_scope(building_report, building_dataset, "building")
        need(building_report.get("kind") == "building" and runtime.get("protocol") == "runtime_point", "invalid building runtime report")
        need(building_report.get("metadata", {}).get("data_version"), "building model data version missing")
        need(set(building_report.get("classes", [])) == set(building_classes), "building classes differ from frozen scope")
        if _finite(runtime.get("failed")) and runtime["failed"] > 0:
            failures.append("building runtime errors present")
        else:
            need(runtime.get("failed") == 0, "building runtime error status missing")
        need(independent.get("full_ap_eligible") is True, "independent AP raw provenance/truncation/error unresolved")
        need(independent.get("protocol", {}).get("sha256") and independent.get("protocol", {}).get("protocol_id"),
             "independent AP protocol not frozen")
        need(independent.get("protocol", {}).get("sha256") == scope.get("evaluation_protocol_sha256"),
             "independent AP protocol differs from frozen acceptance config")
        if _finite(independent.get("failed")) and independent["failed"] > 0:
            failures.append("independent AP evaluation errors present")
        else:
            need(independent.get("failed") == 0, "independent AP error status missing")
        for label in building_classes:
            item = runtime.get("per_class", {}).get(label, {})
            ap = independent.get("per_class", {}).get(label, {})
            instances = item.get("tp", 0) + item.get("fn", 0)
            instance_min = t["town_hall_instances"] if label == "town_hall" else t["building_test_instances_per_class"]
            group_min = t["town_hall_groups"] if label == "town_hall" else t["building_test_groups_per_class"]
            precision_min = t["town_hall_precision"] if label == "town_hall" else t["building_precision"]
            recall_min = t["town_hall_recall"] if label == "town_hall" else t["building_recall"]
            need_count(instances, instance_min, f"building {label} test instances")
            need_count(item.get("truth_groups"), group_min, f"building {label} test groups")
            pass_threshold(item.get("precision"), precision_min, f"building {label} precision")
            pass_threshold(item.get("recall"), recall_min, f"building {label} recall")
            pass_threshold(ap.get("center_error_p95_normalized"), t["center_error_p95_normalized"],
                           f"building {label} center error P95", minimum=False)
            need(_finite(ap.get("ap50")) and _finite(ap.get("ap50_95")), f"building {label} AP missing")
            if _finite(t.get("building_ap50_min")):
                pass_threshold(ap.get("ap50"), t["building_ap50_min"], f"building {label} AP50")
        evidence["building_samples"] = runtime.get("samples")
        evidence["mAP50"] = independent.get("mAP50")
        evidence["mAP50_95"] = independent.get("mAP50_95")
        if _finite(t.get("building_map50_95_min")):
            pass_threshold(independent.get("mAP50_95"), t["building_map50_95_min"], "building mAP50-95")

    if ap_stability is None:
        missing.append("validation-set AP stability report missing")
    else:
        check_report_scope(ap_stability, building_dataset, "building stability", expected_split="val")
        stability = ap_stability.get("evaluation_stability", {})
        need(ap_stability.get("kind") == "building" and ap_stability.get("independent_ap", {}).get("full_ap_eligible") is True,
             "AP stability source report lacks complete raw evaluation")
        if building_report is not None:
            need(ap_stability.get("sha256") == building_report.get("sha256"),
                 "AP stability uses a different model hash")
            match_evidence(ap_stability.get("model_contract_sha256"),
                           building_report.get("model_contract_sha256"),
                           "AP stability model contract")
        need(stability.get("passed") is True, "AP stability not established")
        pass_threshold(stability.get("max_ap_delta"), t["ap_stability_max_delta"],
                       "AP stability max delta", minimum=False)
        if building_report is not None:
            need(stability.get("base_protocol_sha256") == building_report.get("independent_ap", {}).get("protocol", {}).get("sha256"),
                 "AP stability protocol differs from test protocol")

    if parity is None:
        missing.append("PyTorch/ONNX parity report missing")
    else:
        if scene_report is not None and building_report is not None:
            need(parity.get("model_sha256") == {"scene": scene_report.get("sha256"),
                                                "building": building_report.get("sha256")},
                 "parity model hashes do not match evaluated packages")
        need_count(parity.get("reviewed_images"), t["parity_reviewed_images"], "parity reviewed images")
        pass_threshold(parity.get("scene_decision_agreement"), t["parity_scene_decision_agreement"], "scene parity")
        pass_threshold(parity.get("max_building_precision_drop"), t["parity_building_metric_drop"],
                       "building precision drop", minimum=False)
        pass_threshold(parity.get("max_building_recall_drop"), t["parity_building_metric_drop"],
                       "building recall drop", minimum=False)
        pass_threshold(parity.get("box_error_p95_normalized"), t["parity_box_error_p95_normalized"],
                       "parity box error P95", minimum=False)

    if windows is None:
        missing.append("Windows complete-chain timing report missing")
    else:
        if scene_report is not None and building_report is not None:
            need(windows.get("model_sha256") == {"scene": scene_report.get("sha256"),
                                                 "building": building_report.get("sha256")},
                 "Windows model hashes do not match evaluated packages")
        need(windows.get("platform") == "Windows" and windows.get("complete_chain") is True,
             "Windows complete-chain provenance missing")
        need_count(windows.get("hot_model_images"), t["windows_hot_model_images"], "Windows hot model images")
        actions = windows.get("action_types", {})
        need(isinstance(actions, dict) and bool(actions), "Windows action types missing")
        if isinstance(actions, dict):
            for name, item in actions.items():
                if not isinstance(item, dict):
                    missing.append(f"Windows {name} action evidence missing")
                    continue
                pass_threshold(item.get("p95_seconds"), t["windows_full_chain_p95_seconds"],
                               f"Windows {name} complete-chain P95", minimum=False)
                need(all(_finite(item.get(key)) for key in ("max_seconds", "timeouts", "failures", "rejections", "stale_discards")),
                     f"Windows {name} failure/maximum counters missing")

    if core_entry is None:
        missing.append("core/entry strategy report missing")
    else:
        if building_report is not None:
            need(core_entry.get("building_model_sha256") == building_report.get("sha256"),
                 "core/entry building model hash mismatch")
        need_count(core_entry.get("test_layout_groups"), t["core_entry_test_groups"], "core/entry test layouts")
        pass_threshold(core_entry.get("target_recall"), t["core_target_recall"], "core target recall")
        pass_threshold(core_entry.get("entry_set_hit"), t["entry_set_hit"], "acceptable entry hit")
        pass_threshold(core_entry.get("illegal_deployments"), 0, "illegal deployment points", minimum=False)

    if scope.get("dynamic_required") is True:
        if dynamic is None:
            missing.append("dynamic event/action report missing")
        else:
            events = dynamic.get("action_types", {})
            need(isinstance(events, dict) and bool(events), "dynamic action types missing")
            if isinstance(events, dict):
                for name, item in events.items():
                    need_count(item.get("positive_events"), t["dynamic_positive_events"], f"{name} positive events")
                    need_count(item.get("negative_windows"), t["dynamic_negative_windows"], f"{name} negative windows")
                    pass_threshold(item.get("event_recall"), t["dynamic_event_recall"], f"{name} event recall")
                    pass_threshold(item.get("false_actions"), t["dynamic_false_actions"], f"{name} false actions", minimum=False)
    else:
        evidence["dynamic_scope"] = "not_requested_for_static_release"

    status = "FAIL" if failures else "INSUFFICIENT_DATA" if missing else "PASS"
    evidence["evaluation_protocol_sha256"] = scope.get("evaluation_protocol_sha256")
    return {"status": status, "acceptance_id": acceptance.get("acceptance_id"),
            "missing": missing, "failures": failures, "evidence": evidence,
            "scope": scope}


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--acceptance-config", type=Path, default=DEFAULT_ACCEPTANCE)
    for key in ("scene-report", "building-report", "ap-stability-report", "parity-report", "windows-report", "core-entry-report", "dynamic-report"):
        cli.add_argument("--" + key, type=Path)
    cli.add_argument("--scene-dataset", type=Path)
    cli.add_argument("--building-dataset", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args(argv)
    try:
        raw = args.acceptance_config.read_bytes()
        config = json.loads(raw)
        report = validate_release(acceptance=config, scene_report=_load(args.scene_report),
            building_report=_load(args.building_report), ap_stability=_load(args.ap_stability_report),
            parity=_load(args.parity_report),
            windows=_load(args.windows_report), core_entry=_load(args.core_entry_report),
            dynamic=_load(args.dynamic_report), scene_dataset=args.scene_dataset,
            building_dataset=args.building_dataset)
        report["acceptance_config_sha256"] = hashlib.sha256(raw).hexdigest()
        if args.output:
            if args.output.exists():
                raise ValueError(f"Refusing to overwrite {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report["input_manifest"] = _input_manifest(args, report)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return {"PASS": 0, "FAIL": 1, "INSUFFICIENT_DATA": 2}[report["status"]]
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "INSUFFICIENT_DATA", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
