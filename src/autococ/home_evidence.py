"""Verify a source-bound HOME HUD when the shop caption is animated away.

The alternate proof is read-only.  It never creates a shop control or makes a
different scene actionable; the attack control remains independently located.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct


VERSION = "18.647.2"
BASELINE = [1280, 720]
RESOURCE_NAMES = ("gold", "elixir", "dark_elixir", "gems")
RESOURCE_ROIS = {
    "gold": (1008, 19, 1240, 60),
    "elixir": (1008, 69, 1240, 111),
    "dark_elixir": (1064, 115, 1240, 158),
    "gems": (1130, 164, 1240, 208),
}
CURRENT_ICON_FILES = {
    "village_home_battle_18_647_2.png",
    "village_home_battle_checklist_18_647_2.png",
}


def _sha256(path: Path) -> str | None:
    try:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return None


def _inside(box: object, roi: tuple[int, int, int, int]) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value) for value in box)
            and roi[0] <= box[0] < box[2] <= roi[2]
            and roi[1] <= box[1] < box[3] <= roi[3])


def _same_frame(path: Path, claimed: object, digest: object) -> bool:
    return (isinstance(claimed, str) and Path(claimed) == path
            and isinstance(digest, str) and len(digest) == 64
            and _sha256(path) == digest)


def _png_resolution(path: Path) -> list[int] | None:
    try:
        with path.open("rb") as source:
            header = source.read(24)
    except OSError:
        return None
    if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"):
        return None
    return list(struct.unpack(">II", header[16:24]))


def ocr_sha256(rows: list[dict]) -> str:
    """Bind the entire OCR observation, including conflict and account text."""
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def home_navigation_verified(snapshot) -> bool:
    """Retain the old two-control contract or accept one strict current HOME proof.

    Callers keep their own foreground, freshness, account, resource and deadline
    checks.  This predicate never authorizes a missing attack control.
    """
    if snapshot.scene != "village" or type(snapshot.confidence) not in (int, float) or not (
            math.isfinite(snapshot.confidence) and snapshot.confidence >= .8):
        return False
    observations = snapshot.observations
    buttons = observations.get("buttons")
    if not isinstance(buttons, list) or any(not isinstance(item, dict) for item in buttons):
        return False
    count = lambda name: sum(item.get("name") == name for item in buttons)
    if count("attack") != 1:
        return False
    if count("shop") == 1:
        return True
    if count("shop") != 0:
        return False
    proof = observations.get("home_hud_proof")
    if not isinstance(proof, dict) or proof.get("method") != "current_icon_builder_four_full_ocr":
        return False
    if (proof.get("client_version") != VERSION or proof.get("baseline_resolution") != BASELINE
            or proof.get("source_resolution") != observations.get("source_resolution")
            or observations.get("baseline_resolution") != BASELINE
            or observations.get("village_type") != "home"
            or observations.get("resource_source") != "village_inventory"
            or observations.get("account_in_use") is not None):
        return False
    source_resolution = proof.get("source_resolution")
    if source_resolution not in ([1280, 720], [2560, 1440]):
        return False
    path = Path(snapshot.screenshot_path)
    if (not _same_frame(path, proof.get("frame"), proof.get("frame_sha256"))
            or _png_resolution(path) != source_resolution):
        return False
    icon = proof.get("attack_icon")
    observed_icon = observations.get("village_type_evidence")
    if not isinstance(icon, dict) or icon != observed_icon or icon.get("type") != "home":
        return False
    template = icon.get("template")
    if not isinstance(template, str) or Path(template).name not in CURRENT_ICON_FILES:
        return False
    catalog = Path(__file__).resolve().parents[2] / "assets" / "catalogs"
    if Path(template).resolve() != (catalog / Path(template).name).resolve():
        return False
    if (_sha256(Path(template)) != proof.get("attack_template_sha256")
            or icon.get("frame") != str(path)
            or not _inside(icon.get("bbox"), (25, 570, 145, 700))):
        return False
    attack = next(item for item in buttons if item.get("name") == "attack")
    if attack.get("point") != icon.get("point") or attack.get("bbox") != icon.get("bbox"):
        return False
    scores = icon.get("scores")
    if not isinstance(scores, dict) or type(scores.get("home")) not in (int, float):
        return False
    home_score = scores["home"]
    builder_score = scores.get("builder_base", -1)
    if (type(builder_score) not in (int, float) or not math.isfinite(home_score)
            or not math.isfinite(builder_score) or not .92 <= home_score <= 1
            or home_score - builder_score < .1):
        return False
    ocr = observations.get("ocr")
    builder = proof.get("builder")
    if not isinstance(ocr, list) or any(not isinstance(item, dict) for item in ocr):
        return False
    try:
        digest = ocr_sha256(ocr)
    except (TypeError, ValueError):
        return False
    if (not isinstance(builder, dict) or proof.get("ocr_sha256") != digest
            or sum(item == builder for item in ocr if isinstance(item, dict)) != 1
            or builder.get("text") != "建造" or not _inside(builder.get("bbox"), (1150, 645, 1250, 715))
            or type(builder.get("confidence")) not in (int, float)
            or not .95 <= builder["confidence"] <= 1):
        return False
    anchors = proof.get("resources")
    balances = observations.get("resources")
    evidence = observations.get("resource_evidence")
    if not isinstance(anchors, dict) or set(anchors) != set(RESOURCE_NAMES) or not (
            isinstance(balances, dict) and isinstance(evidence, dict)):
        return False
    for name in RESOURCE_NAMES:
        anchor = anchors[name]
        if not isinstance(anchor, dict):
            return False
        item, value = anchor.get("ocr"), anchor.get("value")
        if (not isinstance(item, dict) or type(value) is not int or value < 0
                or balances.get(name) != value or name not in evidence
                or sum(row == item for row in ocr if isinstance(row, dict)) != 1
                or not _inside(item.get("bbox"), RESOURCE_ROIS[name])
                or type(item.get("confidence")) not in (int, float)
                or not .95 <= item["confidence"] <= 1):
            return False
        # Reparse the full OCR text rather than trusting a copied numeric field.
        from .vision import parse_resource_number
        if (parse_resource_number(str(item.get("text", ""))) != value
                or not isinstance(evidence[name], dict)
                or parse_resource_number(str(evidence[name].get("text", ""))) != value):
            return False
    return True
