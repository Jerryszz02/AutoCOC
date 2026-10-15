"""Read the current Home Village storage limit from a verified HUD tooltip."""

from __future__ import annotations

from pathlib import Path
import math
import re

from .errors import FlowError
from .images import read_frame
from .scene import SceneSnapshot
from .vision import recognize_village_type


# 18.647.2, 1280 x 720. The click is inside the balance bar, never its plus
# button. Each text ROI is inside the first tooltip row, away from the HUD.
_LAYOUT = {
    "gold": ((1150, 37), 61, (977, 1265), (1005, 77, 1100, 102), (1104, 75, 1242, 101)),
    "elixir": ((1150, 87), 114, (977, 1265), (1005, 130, 1100, 155), (1104, 127, 1240, 155)),
    "dark_elixir": ((1150, 137), 163, (1014, 1265), (1045, 179, 1144, 205), (1148, 178, 1248, 204)),
}


def _valid_confidence(value: object, minimum: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= 1


def _home_evidence(snapshot: SceneSnapshot, resource: str) -> bool:
    observations = snapshot.observations
    village = observations.get("village_type_evidence") or {}
    resources = observations.get("resources") or {}
    resource_evidence = (observations.get("resource_evidence") or {}).get(resource) or {}
    return (
        snapshot.scene == "village" and _valid_confidence(snapshot.confidence, .8)
        and observations.get("resource_source") == "village_inventory"
        and observations.get("village_type") == "home" and village.get("type") == "home"
        and village.get("frame") == str(snapshot.screenshot_path)
        and any(button.get("name") == "attack" and _valid_confidence(button.get("confidence"), .9)
                for button in observations.get("buttons", []))
        and all(type(resources.get(name)) is int and resources[name] >= 0
                for name in ("gold", "elixir", "dark_elixir", "gems"))
        and _valid_confidence(resource_evidence.get("confidence"), .9)
        and tuple(observations.get("baseline_resolution", (1280, 720))) == (1280, 720)
    )


def _single_line(provider, image) -> tuple[str, float] | None:
    result = provider.recognize_line_image(image)
    if len(result) != 1 or not math.isfinite(result[0].confidence) or result[0].confidence < .95:
        return None
    return result[0].text, result[0].confidence


def _read_pair(provider, image, box: tuple[int, int, int, int]) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
    import numpy as np

    left, top, right, bottom = box
    crop = image[top:bottom, left:right]
    white = np.repeat(np.where(np.min(crop, axis=2, keepdims=True) > 180, 255, 0).astype("uint8"), 3, axis=2)
    return _single_line(provider, crop), _single_line(provider, white)


def read_tooltip_capacity(path: Path, resource: str, provider) -> dict[str, object] | None:
    """Return independently read capacity only when tooltip geometry and OCR agree."""
    import cv2
    import numpy as np

    if resource not in _LAYOUT or not callable(getattr(provider, "recognize_line_image", None)):
        return None
    source = read_frame(path, cv2.IMREAD_COLOR)
    if source is None or source.shape[1] / source.shape[0] != 16 / 9:
        return None
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    _, border_y, (left, right), label_box, number_box = _LAYOUT[resource]
    # The long pale horizontal border is absent from the closed HUD. The two
    # segments straddle the pointer apex and must both be present.
    first = image[border_y, left + 7:1115]
    second = image[border_y, 1150:right - 7]
    if (len(first) < 40 or len(second) < 40
            or float(np.mean(np.min(first, axis=1) > 150)) < .85
            or float(np.mean(np.min(second, axis=1) > 150)) < .7):
        return None
    raw_label, white_label = _read_pair(provider, image, label_box)
    raw_number, white_number = _read_pair(provider, image, number_box)
    if not all((raw_label, white_label, raw_number, white_number)):
        return None
    if any(re.sub(r"\s", "", item[0]) != "最大储量：" for item in (raw_label, white_label)):
        return None
    numbers = [re.sub(r"\s", "", item[0]) for item in (raw_number, white_number)]
    if numbers[0] != numbers[1] or not re.fullmatch(r"[1-9]\d*", numbers[0]):
        return None
    return {"capacity": int(numbers[0]), "frame": str(path), "resource": resource,
            "ocr": {"label_raw": raw_label[1], "label_white": white_label[1],
                    "number_raw": raw_number[1], "number_white": white_number[1]},
            "geometry": {"border_y": border_y, "label_roi": label_box, "number_roi": number_box}}


def _tooltip_border_absent(path: Path, resource: str) -> bool:
    """Independently verify that both segments of this tooltip's border are gone."""
    import cv2
    import numpy as np

    source = read_frame(path, cv2.IMREAD_COLOR)
    if source is None or source.ndim != 3 or source.shape[1] / source.shape[0] != 16 / 9:
        return False
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    _, y, (left, right), _, _ = _LAYOUT[resource]
    first = image[y, left + 7:1115]
    second = image[y, 1150:right - 7]
    return (len(first) >= 40 and len(second) >= 40
            and float(np.mean(np.min(first, axis=1) > 150)) <= .15
            and float(np.mean(np.min(second, axis=1) > 150)) <= .15)


def read_resource_capacity(session, home: SceneSnapshot, resource: str) -> tuple[dict[str, object], SceneSnapshot, list[Path]]:
    """Open one bounded HUD detail, verify it, close at the same safe point."""
    if getattr(session, "client_version", None) != "18.647.2" or resource not in _LAYOUT:
        raise FlowError("Storage tooltip layout is unsupported for this client")
    if not _home_evidence(home, resource):
        raise FlowError("Home HUD and inventory evidence are insufficient for storage detail")
    provider = getattr(getattr(session, "recognizer", None), "provider", None)
    if not callable(getattr(provider, "recognize_line_image", None)):
        raise FlowError("Storage tooltip OCR is unavailable")
    point = _LAYOUT[resource][0]
    session.tap(home, point, reason=f"Open {resource} balance detail")
    paths: list[Path] = []
    proof = None
    for attempt in range(2):
        detail = session.observe(f"{resource}-storage-detail" if attempt == 0 else f"{resource}-storage-detail-settled")
        paths.append(detail.screenshot_path)
        candidate = read_tooltip_capacity(detail.screenshot_path, resource, provider)
        if candidate is None or candidate["capacity"] < home.observations["resources"][resource]:
            continue
        # Require independent navigation and scene evidence before a second tap.
        home_icon = recognize_village_type(detail.screenshot_path, client_version=session.client_version)
        if detail.scene == "village" and _valid_confidence(detail.confidence, .8) and home_icon.get("type") == "home":
            proof = candidate
            break
    if proof is None:
        raise FlowError(f"{resource} storage tooltip/Home navigation could not be verified; detail left untouched")
    session.tap(detail, point, reason=f"Close verified {resource} balance detail")
    before = home.observations["resources"]
    for attempt in range(2):
        closed = session.observe(f"{resource}-storage-closed" if attempt == 0 else f"{resource}-storage-closed-settled")
        paths.append(closed.screenshot_path)
        if (_home_evidence(closed, resource)
                and read_tooltip_capacity(closed.screenshot_path, resource, provider) is None
                and _tooltip_border_absent(closed.screenshot_path, resource)
                and all(closed.observations["resources"][name] == before[name]
                        for name in ("gold", "elixir", "dark_elixir", "gems"))):
            return proof, closed, paths
    raise FlowError(f"{resource} storage detail did not close with unchanged inventory")
