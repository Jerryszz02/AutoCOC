"""Create a new scene/building bundle from a strict PASS release report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autococ.model_release import check_release_binding
from autococ.model_vision import ModelUnavailable, _manifest, model_contract_sha256
from scripts.validate_vision_release import validate_release


_AUDIT_KEYS = ("acceptance_config", "scene_report", "building_report", "ap_stability_report",
               "parity_report", "windows_report", "core_entry_report", "dynamic_report")


def verify_and_revalidate(report_path: Path, *, scene_dataset: Path | None = None,
                          building_dataset: Path | None = None) -> dict:
    """Reject hand-written PASS summaries; replay the frozen gate from hashed inputs."""
    report_path = report_path.resolve()
    report = json.loads(report_path.read_bytes())
    if not isinstance(report, dict) or report.get("status") != "PASS":
        raise ModelUnavailable("Release report is not PASS")
    audit = report.get("input_manifest")
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise ModelUnavailable("Release report has no input audit manifest")
    files = audit.get("files")
    datasets = audit.get("datasets")
    if not isinstance(files, dict) or not isinstance(datasets, dict):
        raise ModelUnavailable("Release input audit manifest is incomplete")
    loaded = {}
    for key in _AUDIT_KEYS:
        record = files.get(key)
        if record is None:
            loaded[key] = None
            continue
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ModelUnavailable(f"Invalid release input path: {key}")
        path = (report_path.parent / relative).resolve()
        audit_root = (report_path.parent / f"{report_path.stem}.inputs").resolve()
        if not path.is_relative_to(audit_root) or not path.is_file():
            raise ModelUnavailable(f"Release input missing or outside audit bundle: {key}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
            raise ModelUnavailable(f"Release input SHA-256 mismatch: {key}")
        loaded[key] = json.loads(raw)
        if not isinstance(loaded[key], dict):
            raise ModelUnavailable(f"Release input must be JSON object: {key}")
    if loaded["acceptance_config"] is None:
        raise ModelUnavailable("Acceptance config audit is missing")
    resolved_datasets = {}
    for kind, override in (("scene", scene_dataset), ("building", building_dataset)):
        record = datasets.get(kind)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ModelUnavailable(f"{kind} dataset audit is missing")
        relative = Path(record["path"])
        if relative.is_absolute():
            raise ModelUnavailable(f"{kind} dataset audit path is not portable")
        resolved_datasets[kind] = override.resolve() if override else (report_path.parent / relative).resolve()
    replay = validate_release(acceptance=loaded["acceptance_config"],
        scene_report=loaded["scene_report"], building_report=loaded["building_report"],
        ap_stability=loaded["ap_stability_report"], parity=loaded["parity_report"],
        windows=loaded["windows_report"], core_entry=loaded["core_entry_report"],
        dynamic=loaded["dynamic_report"], scene_dataset=resolved_datasets["scene"],
        building_dataset=resolved_datasets["building"])
    replay["acceptance_config_sha256"] = files["acceptance_config"]["sha256"]
    for kind in ("scene", "building"):
        if (replay["evidence"].get(f"{kind}_dataset_sha256") != datasets[kind].get("snapshot_sha256") or
                replay["evidence"].get(f"{kind}_split_sha256") != datasets[kind].get("split_sha256")):
            raise ModelUnavailable(f"{kind} dataset differs from audited snapshot")
    if replay != {key: value for key, value in report.items() if key != "input_manifest"}:
        raise ModelUnavailable("Release report differs from revalidated input evidence")
    if replay["status"] != "PASS":
        raise ModelUnavailable("Revalidated release did not pass")
    return report


def release_models(scene_dir: Path, building_dir: Path, report_path: Path, output: Path,
                   *, scene_dataset: Path | None = None,
                   building_dataset: Path | None = None) -> dict:
    scene_dir, building_dir, output = scene_dir.resolve(), building_dir.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite release output: {output}")
    if output.is_relative_to(scene_dir) or output.is_relative_to(building_dir):
        raise ValueError("Release output cannot be inside an input package")
    report_path = report_path.resolve()
    report = verify_and_revalidate(report_path, scene_dataset=scene_dataset,
                                   building_dataset=building_dataset)
    raw = report_path.read_bytes()
    if json.loads(raw) != report:
        raise ModelUnavailable("Release report changed during revalidation")
    portable = copy.deepcopy(report)
    for key, record in portable["input_manifest"]["files"].items():
        record["path"] = f"release_report.inputs/{key}.json"
    source_datasets = {}
    for kind, override in (("scene", scene_dataset), ("building", building_dataset)):
        original = report["input_manifest"]["datasets"][kind]["path"]
        source_datasets[kind] = override.resolve() if override else (report_path.parent / original).resolve()
        try:
            relative = os.path.relpath(source_datasets[kind], output).replace("\\", "/")
        except ValueError:
            relative = f"datasets/{kind}"  # caller supplies --*-dataset after moving drives
        portable["input_manifest"]["datasets"][kind]["path"] = relative
    raw = (json.dumps(portable, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    report_sha256 = hashlib.sha256(raw).hexdigest()
    manifests = {}
    for kind, source in (("scene", scene_dir), ("building", building_dir)):
        manifest, _, digest = _manifest(source, kind, None)
        validation = manifest["validation"]
        if validation.get("release_sha256") is not None:
            raise ModelUnavailable(f"{kind} source package is already release-bound")
        candidate_validation = {**validation, "status": "evaluated",
                                "report_id": report.get("acceptance_id"),
                                "release_sha256": report_sha256}
        check_release_binding(report=portable, kind=kind, model_sha256=digest,
                              model_contract_sha256_value=model_contract_sha256(manifest),
                              validation=candidate_validation,
                              client_versions=manifest["client_versions"])
        manifests[kind] = {**manifest, "sha256": digest, "validation": candidate_validation}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="vision-release-", dir=output.parent) as staging_dir:
        staging = Path(staging_dir) / "bundle"
        staging.mkdir()
        shutil.copytree(report_path.parent / f"{report_path.stem}.inputs",
                        staging / "release_report.inputs")
        for kind, source in (("scene", scene_dir), ("building", building_dir)):
            target = staging / kind
            shutil.copytree(source, target)
            (target / "manifest.json").write_text(
                json.dumps(manifests[kind], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _manifest(target, kind, None)  # verify the copied weight before publication
        (staging / "release_report.json").write_bytes(raw)
        verify_and_revalidate(staging / "release_report.json",
                              scene_dataset=source_datasets["scene"],
                              building_dataset=source_datasets["building"])
        staging.rename(output)
    return {"status": "released", "output": str(output), "release_report_sha256": report_sha256,
            "scene_model_sha256": manifests["scene"]["sha256"],
            "building_model_sha256": manifests["building"]["sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--building-package", required=True, type=Path)
    parser.add_argument("--release-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-dataset", type=Path,
                        help="Override the audited dataset location after moving the audit bundle")
    parser.add_argument("--building-dataset", type=Path,
                        help="Override the audited dataset location after moving the audit bundle")
    args = parser.parse_args(argv)
    try:
        result = release_models(args.scene_package, args.building_package,
                                args.release_report, args.output,
                                scene_dataset=args.scene_dataset,
                                building_dataset=args.building_dataset)
    except (OSError, ValueError, ModelUnavailable) as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
