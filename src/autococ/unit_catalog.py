"""Stable home-village unit names and evidence-based card identity.

The catalogue describes possible units; it does not imply image coverage or an
unlocked unit in the current village. Templates are deliberately opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


UnitKind = Literal["troop", "hero", "spell", "siege"]
CATALOG_ROOT = Path(__file__).resolve().parents[2] / "assets" / "catalogs"


@dataclass(frozen=True)
class Unit:
    unit_id: str
    kind: UnitKind
    name: str
    source: str = "army"
    housing_space: int | None = None


def _units(kind: UnitKind, values: str, *, source: str = "army") -> tuple[Unit, ...]:
    return tuple(Unit(unit_id, kind, unit_id.replace("_", " ").title(), source)
                 for unit_id in values.split())


# IDs are intentionally English snake_case and independent of the displayed
# language, card order, level, hero equipment and pet. The list is a registry,
# not a promise that every unit has been sampled on this client version.
UNITS = (
    *_units("troop", "barbarian archer giant goblin wall_breaker balloon wizard healer dragon pekka baby_dragon miner electro_dragon yeti dragon_rider electro_titan root_rider thrower"),
    *_units("troop", "minion hog_rider valkyrie golem witch lava_hound bowler ice_golem headhunter apprentice_warden druid meteor_golem furnace"),
    *_units("troop", "super_barbarian super_archer super_giant sneaky_goblin super_wall_breaker rocket_balloon super_wizard super_dragon inferno_dragon super_miner super_minion super_valkyrie super_witch ice_hound", source="super_troop"),
    *_units("troop", "event_super_pekka", source="event"),
    *_units("hero", "barbarian_king archer_queen grand_warden royal_champion minion_prince dragon_duke"),
    *_units("spell", "lightning_spell healing_spell rage_spell jump_spell freeze_spell clone_spell invisibility_spell recall_spell revival_spell poison_spell earthquake_spell haste_spell skeleton_spell bat_spell overgrowth_spell totem_spell"),
    *_units("siege", "wall_wrecker battle_blimp stone_slammer siege_barracks log_launcher flame_flinger battle_drill troop_launcher sky_wagon"),
)
UNIT_BY_ID = {unit.unit_id: unit for unit in UNITS}


@dataclass(frozen=True)
class ArmyRequirement:
    unit_id: str
    count: int
    optional: bool = False

    def __post_init__(self) -> None:
        if get_unit(self.unit_id) is None:
            raise ValueError(f"Unknown home-village unit ID: {self.unit_id}")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("Army requirement count must be a positive integer")
        if type(self.optional) is not bool:
            raise ValueError("Army requirement optional must be boolean")


@dataclass(frozen=True)
class ArmyRecipe:
    units: tuple[ArmyRequirement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.units, tuple) or not self.units:
            raise ValueError("ArmyRecipe needs a nonempty tuple of requirements")
        if any(not isinstance(item, ArmyRequirement) for item in self.units):
            raise ValueError("ArmyRecipe contains an invalid requirement")
        ids = [item.unit_id for item in self.units]
        if len(set(ids)) != len(ids):
            raise ValueError("ArmyRecipe has duplicate unit IDs")
        if all(item.optional for item in self.units):
            raise ValueError("ArmyRecipe needs at least one required unit")


def get_unit(unit_id: str) -> Unit | None:
    return UNIT_BY_ID.get(unit_id)


def template_manifest(root: Path = CATALOG_ROOT) -> dict:
    path = root / "unit_templates.json"
    if not path.is_file():
        return {"client": "unknown", "templates": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("templates"), list):
        raise ValueError("Invalid unit template manifest")
    return data


def coverage(root: Path = CATALOG_ROOT) -> dict[str, dict[str, object]]:
    """Report samples present on disk, separately from catalogue membership."""
    manifest = template_manifest(root)
    result = {unit.unit_id: {"kind": unit.kind, "catalogued": True,
                             "recognition": "unavailable", "templates": 0}
              for unit in UNITS}
    for entry in manifest["templates"]:
        unit_id = entry.get("unit_id")
        relative = entry.get("path")
        if unit_id not in result or not isinstance(relative, str):
            continue
        path = (root / relative).resolve()
        if path.is_file() and path.is_relative_to(root.resolve()):
            result[unit_id]["templates"] += 1
            result[unit_id]["recognition"] = "sampled"
    return result


def recognize_card_identity(
    image: np.ndarray, box: tuple[int, int, int, int], kind: str, *,
    root: Path = CATALOG_ROOT, surface: str = "battle", threshold: float = .94,
    client_version: str | None = None,
) -> dict[str, object]:
    """Match the card portrait against versioned samples, never its slot index.

    The caller supplies a located card and its kind. Count OCR is independent of
    this result. A missing, ambiguous, or unsampled portrait stays unknown.
    """
    left, top, right, bottom = box
    manifest = template_manifest(root)
    manifest_version = str(manifest.get("client", "unknown"))
    version_verified = client_version is not None and client_version == manifest_version
    result: dict[str, object] = {"unit_id": None, "confidence": None,
                                  "reason": "no_supported_template", "evidence": [],
                                  "template_client_version": manifest_version,
                                  "version_verified": version_verified}
    if client_version is not None and not version_verified:
        result["reason"] = "client_version_unverified"
        return result
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        result["reason"] = "invalid_card_box"
        return result
    entries = manifest["templates"]
    candidates: list[tuple[float, str, str]] = []
    for entry in entries:
        unit_id = entry.get("unit_id")
        unit = UNIT_BY_ID.get(unit_id)
        if unit is None or unit.kind != kind or entry.get("surface") != surface:
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            continue
        template_path = (root / relative).resolve()
        if not template_path.is_relative_to(root.resolve()) or not template_path.is_file():
            continue
        template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if template is None or min(template.shape[:2]) < 12:
            continue
        # Sampled card portraits exclude the quantity header and any pet icon.
        crop_top = top + (3 if kind == "hero" else 22)
        crop = image[max(crop_top, 0):min(bottom - 6, image.shape[0]),
                     max(left + 5, 0):min(right - 5, image.shape[1])]
        if crop.size == 0:
            continue
        best = -1.0
        for scale in (.9, 1.0, 1.1):
            sample = cv2.resize(template, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_LINEAR)
            if sample.shape[0] > crop.shape[0] or sample.shape[1] > crop.shape[1]:
                continue
            score = float(cv2.matchTemplate(crop, sample, cv2.TM_CCOEFF_NORMED).max())
            if math.isfinite(score):
                best = max(best, score)
        if best >= 0:
            candidates.append((best, unit_id, str(template_path)))
    candidates.sort(reverse=True)
    result["evidence"] = [{"score": round(score, 5), "unit_id": unit_id, "template": path}
                          for score, unit_id, path in candidates[:3]]
    if not candidates:
        return result
    best, unit_id, _ = candidates[0]
    result["confidence"] = round(best, 5)
    if best < threshold:
        result["reason"] = "below_identity_threshold"
    elif any(best - score < .04 for score, other_id, _ in candidates[1:] if other_id != unit_id):
        result["reason"] = "ambiguous_identity"
    else:
        result.update(unit_id=unit_id, reason="portrait_template")
    return result


def recognize_large_hero_identity(
    image: np.ndarray, box: tuple[int, int, int, int], *,
    root: Path = CATALOG_ROOT, client_version: str | None = None,
) -> dict[str, object]:
    """Identify one independently bordered My Army hero card.

    Large portraits animate. A small face pattern and the upper-card colour
    distribution must *both* agree with samples for the same hero. Neither
    the card's position, pet nor equipment contributes to its ID.
    """
    manifest = template_manifest(root)
    version = str(manifest.get("client", "unknown"))
    result: dict[str, object] = {"unit_id": None, "confidence": None,
                                  "reason": "no_supported_template", "evidence": [],
                                  "template_client_version": version,
                                  "version_verified": client_version == version}
    if client_version != version:
        result["reason"] = "client_version_unverified"
        return result
    left, top, right, bottom = box
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]
            and 112 <= right - left <= 138 and 465 <= bottom - top <= 510):
        result["reason"] = "invalid_hero_card_box"
        return result
    portrait = image[top + 35:top + 355, left + 5:right - 6]
    full = image[top + 35:top + 355, left + 5:left + 119]
    if portrait.shape[0] != 320 or full.shape[1] != 114:
        result["reason"] = "invalid_hero_card_crop"
        return result
    current_hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
    current_hist = cv2.calcHist([current_hsv], [0, 1], None, [24, 12], [0, 180, 0, 256])
    cv2.normalize(current_hist, current_hist)
    profiles: dict[str, dict[str, float]] = {}
    for entry in manifest["templates"]:
        unit_id = entry.get("unit_id")
        unit = UNIT_BY_ID.get(unit_id)
        if unit is None or unit.kind != "hero" or entry.get("surface") != "army_hero":
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            continue
        template = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if template is None:
            continue
        profile = profiles.setdefault(unit_id, {"face": -1.0, "color_profile": -1.0})
        if entry.get("role") == "face" and (template.shape[0] <= portrait.shape[0]
                                            and template.shape[1] <= portrait.shape[1]):
            score = float(cv2.matchTemplate(portrait, template, cv2.TM_CCOEFF_NORMED).max())
            if math.isfinite(score):
                profile["face"] = max(profile["face"], score)
        elif entry.get("role") == "color_profile" and template.shape == full.shape:
            hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [24, 12], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            score = float(cv2.compareHist(current_hist, hist, cv2.HISTCMP_CORREL))
            if math.isfinite(score):
                profile["color_profile"] = max(profile["color_profile"], score)
    ranked = sorted(profiles.items(), key=lambda item: min(item[1].values()), reverse=True)
    result["evidence"] = [{"unit_id": unit_id, "face_score": round(scores["face"], 5),
                          "color_score": round(scores["color_profile"], 5)}
                          for unit_id, scores in ranked]
    if not ranked:
        return result
    best_id, best = ranked[0]
    if best["face"] < .85 or best["color_profile"] < .8:
        result["reason"] = "below_identity_threshold"
        return result
    if any((best["face"] - other["face"] < .15 or
            best["color_profile"] - other["color_profile"] < .14)
           for other_id, other in ranked[1:] if other_id != best_id):
        result["reason"] = "ambiguous_identity"
        return result
    result.update(unit_id=best_id,
                  confidence=round(min(.99, max(best["face"], best["color_profile"])), 5),
                  reason="face_and_upper_card_visual_profile")
    return result
