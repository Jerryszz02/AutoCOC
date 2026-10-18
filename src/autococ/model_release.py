"""Verify a frozen release bundle before a model can drive live actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .model_vision import ModelUnavailable, model_contract_sha256


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_release_binding(*, report: dict, kind: str, model_sha256: str,
                          model_contract_sha256_value: str, validation: dict,
                          client_versions: list[str]) -> None:
    """Check the report against one package; usable before and after packaging."""
    if kind not in {"scene", "building"} or not _digest(model_sha256):
        raise ModelUnavailable("Release model identity is invalid")
    if (validation.get("prototype_only") is not False or
            validation.get("status") != "evaluated" or
            not isinstance(validation.get("report_id"), str) or not validation["report_id"]):
        raise ModelUnavailable("Model has no formal evaluated release")
    scope = report.get("scope")
    if (report.get("status") != "PASS" or report.get("missing") != [] or
            report.get("failures") != [] or not isinstance(scope, dict) or
            scope.get("frozen") is not True):
        raise ModelUnavailable("Release report did not pass frozen acceptance")
    audit = report.get("input_manifest")
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise ModelUnavailable("Release report input audit is missing")
    if report.get("acceptance_id") != validation["report_id"]:
        raise ModelUnavailable("Model validation report ID differs from release")
    evidence = report.get("evidence", {})
    if not isinstance(evidence, dict) or evidence.get(f"{kind}_snapshot_verified") is not True:
        raise ModelUnavailable("Release dataset verification is missing")
    expected = {"model_sha256": evidence.get(f"{kind}_model_sha256"),
                "model_contract_sha256": evidence.get(f"{kind}_model_contract_sha256"),
                "snapshot_sha256": evidence.get(f"{kind}_dataset_sha256"),
                "split_sha256": evidence.get(f"{kind}_split_sha256")}
    for field, claimed in expected.items():
        actual = (model_sha256 if field == "model_sha256" else
                  model_contract_sha256_value if field == "model_contract_sha256" else
                  validation.get(field))
        if not _digest(claimed) or actual != claimed:
            raise ModelUnavailable(f"Release {kind} {field} does not match model package")
    if not _digest(evidence.get("evaluation_protocol_sha256")) or not _digest(report.get("acceptance_config_sha256")):
        raise ModelUnavailable("Release protocol or acceptance config hash is missing")
    scope_versions = scope.get("client_versions")
    if (scope.get("evaluation_protocol_sha256") != evidence["evaluation_protocol_sha256"] or
            not isinstance(scope_versions, list) or not isinstance(client_versions, list) or
            not all(isinstance(version, str) for version in scope_versions + client_versions) or
            sorted(scope_versions) != sorted(client_versions)):
        raise ModelUnavailable("Release protocol or client versions differ from package")


def require_released_model(model) -> dict:
    """Require an intact PASS report bound to this model and its training snapshot."""
    metadata = getattr(model, "metadata", None)
    package_root = getattr(model, "package_root", None)
    model_sha256 = getattr(model, "model_sha256", None)
    if not isinstance(metadata, dict) or package_root is None:
        raise ModelUnavailable("Model release package provenance is missing")
    validation = metadata.get("validation")
    if not isinstance(validation, dict) or not _digest(validation.get("release_sha256")):
        raise ModelUnavailable("Model release report binding is missing")
    root = Path(package_root).resolve()
    try:
        raw = (root.parent / "release_report.json").read_bytes()
        report = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ModelUnavailable("Release report cannot be read") from exc
    if hashlib.sha256(raw).hexdigest() != validation["release_sha256"]:
        raise ModelUnavailable("Release report SHA-256 mismatch")
    if not isinstance(report, dict):
        raise ModelUnavailable("Release report must be an object")
    kind = metadata.get("kind")
    if metadata.get("model_sha256") != model_sha256:
        raise ModelUnavailable("Model metadata SHA-256 mismatch")
    manifest = getattr(model, "manifest", None)
    if not isinstance(manifest, dict):
        raise ModelUnavailable("Model manifest provenance is missing")
    actual_contract = model_contract_sha256(manifest)
    if metadata.get("model_contract_sha256") != actual_contract:
        raise ModelUnavailable("Model behavior contract changed after evaluation")
    check_release_binding(report=report, kind=kind, model_sha256=model_sha256,
                          model_contract_sha256_value=actual_contract,
                          validation=validation, client_versions=metadata.get("client_versions", []))
    return report
