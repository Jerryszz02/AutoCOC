"""Read the current Home Village storage limit from a verified HUD tooltip."""

from __future__ import annotations

from pathlib import Path
import hashlib
import math
import re

from .errors import FlowError
from .images import read_frame
from .scene import SceneSnapshot, account_in_use_dialog_evidence
from .vision import recognize_village_type


# 18.647.2, 1280 x 720. The click is inside the balance bar, never its plus
# button. Each text ROI is inside the first tooltip row, away from the HUD.
_LAYOUT = {
    "gold": ((1150, 37), 61, (977, 1265), (1005, 77, 1100, 102), (1108, 78, 1242, 99)),
    "elixir": ((1150, 87), 114, (977, 1265), (1005, 130, 1100, 155), (1104, 127, 1240, 155)),
    "dark_elixir": ((1150, 137), 163, (1014, 1265), (1045, 179, 1144, 205), (1150, 182, 1248, 202)),
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


def _label_text(value: str) -> str:
    return re.sub(r"\s", "", value).replace(":", "：")


def _native_label_pair(provider, source, box: tuple[int, int, int, int]):
    """Second strict reading at the actual 2x pixel scale, never upsampled."""
    import numpy as np

    left, top, right, bottom = (value * 2 for value in box)
    crop = source[top:bottom, left:right]
    white = np.repeat(np.where(np.min(crop, axis=2, keepdims=True) > 180, 255, 0).astype("uint8"), 3, axis=2)
    return _single_line(provider, crop), _single_line(provider, white)


def read_tooltip_capacity(path: Path, resource: str, provider) -> dict[str, object] | None:
    """Return independently read capacity only when tooltip geometry and OCR agree."""
    import cv2
    import numpy as np

    if resource not in _LAYOUT or not callable(getattr(provider, "recognize_line_image", None)):
        return None
    try:
        with path.open("rb") as stream:
            source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
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
    label_source = "baseline"
    if raw_label is None and source.shape[:2] == (1440, 2560):
        # A low-confidence baseline label is only a fallback candidate when
        # its one OCR line already has the right text. Native pixels must then
        # independently pass both unchanged .95 gates.
        if white_label is None or re.sub(r"\s", "", white_label[0]) != "最大储量：":
            return None
        left, top, right, bottom = label_box
        weak = provider.recognize_line_image(image[top:bottom, left:right])
        if (len(weak) == 1 and _label_text(weak[0].text) == "最大储量："
                and math.isfinite(weak[0].confidence) and 0 <= weak[0].confidence < .95):
            native_raw, native_white = _native_label_pair(provider, source, label_box)
            if (native_raw is None or native_white is None
                    or any(re.sub(r"\s", "", item[0]) != "最大储量："
                           for item in (native_raw, native_white))):
                return None
            raw_label, white_label = native_raw, native_white
            label_source = "native_2x"
    if not all((raw_label, white_label, raw_number, white_number)):
        return None
    if any(re.sub(r"\s", "", item[0]) != "最大储量：" for item in (raw_label, white_label)):
        return None
    numbers = [re.sub(r"\s", "", item[0]) for item in (raw_number, white_number)]
    if numbers[0] != numbers[1] or not re.fullmatch(r"[1-9]\d*", numbers[0]):
        return None
    try:
        with path.open("rb") as stream:
            frame_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None
    if frame_sha256 != source_sha256:
        return None
    return {"capacity": int(numbers[0]), "frame": str(path), "resource": resource,
            "frame_sha256": frame_sha256, "source_resolution": [source.shape[1], source.shape[0]],
            "ocr": {"label_raw": raw_label[1], "label_white": white_label[1],
                    "number_raw": raw_number[1], "number_white": white_number[1],
                    "label_source": label_source},
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


def verified_unknown_tooltip(snapshot: SceneSnapshot, resource: str, proof: dict,
                             provider) -> bool:
    """An actively opened tooltip over HOME, not a general HOME scene proof."""
    if snapshot.scene != "unknown" or resource not in _LAYOUT or not isinstance(proof, dict):
        return False
    obs = snapshot.observations
    if (obs.get("source_resolution") not in ([1280, 720], [2560, 1440])
            or obs.get("baseline_resolution") != [1280, 720]
            or any(obs.get(key) is not None for key in (
                "account_in_use", "surrender_dialog", "request_dialog", "startup_logo",
                "unit_detail", "spell_detail", "hero_detail", "battle_reward",
                "event_reward_track", "event_panel", "cloud_cover"))):
        return False
    rows = obs.get("ocr")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return False
    texts = "".join(str(row.get("text", "")) for row in rows).replace(" ", "")
    if (account_in_use_dialog_evidence(rows) is not None
            or any(marker in texts for marker in (
                "另一个设备正在连接这个村庄", "连接断开", "连接中断", "连接丢失",
                "重新载入游戏", "重新连接", "服务器维护", "维护中"))):
        return False
    path = snapshot.screenshot_path
    if (proof.get("frame") != str(path) or proof.get("resource") != resource
            or proof.get("source_resolution") != obs["source_resolution"]
            or type(proof.get("capacity")) is not int or proof["capacity"] <= 0):
        return False
    try:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != proof.get("frame_sha256"):
                return False
    except OSError:
        return False
    icon = recognize_village_type(path, client_version="18.647.2")
    scores = icon.get("scores") if isinstance(icon, dict) else None
    if (not isinstance(icon, dict) or icon.get("type") != "home" or icon.get("frame") != str(path)
            or not isinstance(scores, dict)
            or not _valid_confidence(scores.get("home"), .92)
            or type(scores.get("builder_base")) not in (int, float)
            or not math.isfinite(scores["builder_base"])
            or scores["home"] - scores["builder_base"] < .1):
        return False
    # Re-read the same immutable pixels: a caller cannot authorize an unknown
    # scene by supplying a copied capacity or by changing its OCR observations.
    return read_tooltip_capacity(path, resource, provider) == proof


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
    if callable(getattr(session, "open_resource_capacity_tooltip", None)):
        session.open_resource_capacity_tooltip(home, resource, point)
    else:
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
        if ((detail.scene == "village" and _valid_confidence(detail.confidence, .8)
             and home_icon.get("type") == "home")
                or (detail.scene == "unknown" and verified_unknown_tooltip(detail, resource, candidate, provider)
                    and callable(getattr(session, "close_verified_resource_tooltip", None)))):
            proof = candidate
            break
    if proof is None:
        raise FlowError(f"{resource} storage tooltip/Home navigation could not be verified; detail left untouched")
    if detail.scene == "unknown":
        session.close_verified_resource_tooltip(detail, resource, point, proof)
    else:
        session.tap(detail, point, reason=f"Close verified {resource} balance detail")
    before = home.observations["resources"]
    consecutive: list[SceneSnapshot] = []
    for attempt in range(4):
        label = f"{resource}-storage-closed" if attempt == 0 else f"{resource}-storage-closed-{attempt + 1}"
        closed = session.observe(label)
        paths.append(closed.screenshot_path)
        ocr_rows = closed.observations.get("ocr")
        visible_text = "".join(str(row.get("text", "")) for row in ocr_rows if isinstance(row, dict)) if isinstance(ocr_rows, list) else ""
        if (closed.scene not in {"village", "unknown"}
                or any(closed.observations.get(key) is not None for key in (
                    "account_in_use", "surrender_dialog", "request_dialog", "startup_logo",
                    "unit_detail", "spell_detail", "hero_detail", "battle_reward",
                    "event_reward_track", "event_panel", "cloud_cover"))
                or any(marker in visible_text for marker in (
                    "另一个设备正在连接这个村庄", "连接断开", "连接中断", "连接丢失",
                    "重新载入游戏", "重新连接", "服务器维护", "维护中"))):
            raise FlowError(f"{resource} storage close encountered another scene or modal")
        if (_home_evidence(closed, resource)
                and read_tooltip_capacity(closed.screenshot_path, resource, provider) is None
                and _tooltip_border_absent(closed.screenshot_path, resource)
                and all(closed.observations["resources"][name] == before[name]
                        for name in ("gold", "elixir", "dark_elixir", "gems"))):
            consecutive.append(closed)
            if len(consecutive) == 2:
                first, second = consecutive
                first_id, second_id = (frame.observations.get("frame_id") for frame in (first, second))
                first_time, second_time = (frame.observations.get("observed_at_monotonic")
                                           for frame in (first, second))
                if (first.screenshot_path != second.screenshot_path
                        and all(isinstance(value, str) and value for value in (first_id, second_id))
                        and first_id != second_id
                        and all(type(value) in (int, float) and math.isfinite(value)
                                for value in (first_time, second_time))
                        and second_time > first_time):
                    return proof, second, paths
                consecutive.clear()
        else:
            consecutive.clear()
    raise FlowError(f"{resource} storage detail did not close with unchanged inventory")
