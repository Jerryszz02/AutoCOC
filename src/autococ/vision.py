"""Screenshot observations in one baseline coordinate system."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
import time

from .config import OCRConfig
from .images import read_frame, read_template
from .errors import LocatorError, SceneError
from .locator import find_template, scale_box
from .ocr import OCRProvider, OCRText, create_ocr_provider, filter_ocr_results
from .scene import SCENE_BATTLE, SCENE_CLAN_CHAT, SCENE_DISCONNECTED, SCENE_ENEMY_VILLAGE, SCENE_MAINTENANCE, SCENE_POPUP, SCENE_REQUEST, SCENE_SEARCH, SCENE_SETTLEMENT, SCENE_STARTING, SCENE_TRAINING, SCENE_UNKNOWN, SCENE_VILLAGE, SceneSnapshot, classify_scene_text, surrender_dialog_evidence


BUTTON_LABELS: dict[str, tuple[str, ...]] = {
    "attack": ("进攻", "attack"),
    "find_match": ("搜索对手", "寻找对手", "find a match"),
    "next": ("下一个", "下个对手", "next"),
    "return_home": ("返回村庄", "回到村庄", "回营", "return home"),
    "end_battle": ("结束战斗", "投降", "放弃", "end battle", "surrender"),
    "confirm": ("确定", "确认", "是", "okay", "confirm", "yes"),
    "cancel": ("取消", "否", "cancel", "no"),
    "donate": ("捐赠", "增援", "donate"),
    "request": ("请求增援", "请求援军", "请求部队", "请求", "request"),
    "send": ("发送", "send"),
    "collect": ("收集", "领取", "collect"),
    "shop": ("商店", "shop"),
    "army": ("军队", "army"),
    "edit_army": ("编辑军队", "编辑", "edit army"),
    "retry": ("重试", "retry", "重新载入游戏"),
}


def normalize_label(text: str) -> str:
    return re.sub(r"[\s!！.。:：,，?？]", "", text.lower())


def parse_resource_number(text: str) -> int | None:
    """Accept digit groups only; timers, fractions and mixed text are not amounts."""
    normalized = text.strip().replace("，", ",")
    if not re.fullmatch(r"[0-9]+(?:[ ,.-][0-9]+)*", normalized):
        return None
    return int(re.sub(r"\D", "", normalized))


def parse_capacity(text: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if match is None:
        return None
    used, capacity = map(int, match.groups())
    return (used, capacity) if used <= capacity else None


def parse_countdown(text: str) -> int | None:
    value = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"(?:(\d+)分(?:钟)?)?(\d+)秒", value)
    if match:
        minutes, seconds = match.groups()
        return int(minutes or 0) * 60 + int(seconds) if int(seconds) < 60 else None
    match = re.fullmatch(r"(\d+):(\d{2})", value)
    if match and int(match.group(2)) < 60:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None


def recognize_village_type(screenshot_path: str | Path) -> dict[str, object]:
    """Require a positive Home/Builder Base attack-icon match on this frame."""
    import cv2
    import numpy as np

    path = Path(screenshot_path)
    source = read_frame(path, cv2.IMREAD_COLOR)
    result: dict[str, object] = {"type": "unknown", "frame": str(path), "scores": {}}
    if source is None:
        return result
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    region = image[570:700, 25:145]
    root = Path(__file__).resolve().parents[2] / "assets" / "catalogs"
    scores = {}
    matched_templates = {}
    for village_type, filename in (("home", "village_home_attack.png"),
                                   ("home", "village_home_attack_stars.png"),
                                   ("builder_base", "village_builder_attack.png")):
        template = read_template(root / filename)
        if template is None or template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
            continue
        score = float(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED).max())
        if math.isfinite(score) and score > scores.get(village_type, -1):
            scores[village_type] = round(score, 5)
            matched_templates[village_type] = filename
    result["scores"] = scores
    if scores:
        selected = max(scores, key=scores.get)
        runner_up = max((score for name, score in scores.items() if name != selected), default=-1)
        if scores[selected] >= .92 and scores[selected] - runner_up >= .1:
            result["type"] = selected
            result["template"] = str(root / matched_templates[selected])
    return result


def detect_collectibles(
    screenshot_path: str | Path,
    *,
    baseline_resolution: tuple[int, int] = (1280, 720),
    template_dir: str | Path | None = None,
    threshold: float = 0.88,
) -> list[dict[str, object]]:
    """Match resource bubbles, excluding HUD and building action controls.

    Templates were extracted from the observed village at a 1280x720 baseline.
    Resolution scaling is independent of the small scale search for bubble size.
    Call this only for a confirmed village scene.
    """
    import cv2
    import numpy as np

    path = Path(screenshot_path)
    source = read_frame(path, cv2.IMREAD_COLOR)
    if source is None:
        raise SceneError(f"Unable to decode screenshot: {path}")
    image = cv2.resize(source, baseline_resolution, interpolation=cv2.INTER_AREA)
    width, height = baseline_resolution
    template_root = Path(template_dir) if template_dir is not None else Path(__file__).resolve().parents[2] / "assets" / "templates"
    # Screen-edge HUD buttons and the bottom building menu cannot be resource bubbles.
    x0, y0, x1, y1 = round(width * 0.09), round(height * 0.1), round(width * 0.94), round(height * 0.69)
    region = image[y0:y1, x0:x1]
    candidates: list[dict[str, object]] = []
    for resource in ("gold", "elixir", "dark_elixir"):
        template_path = template_root / f"collect_{resource}.png"
        if not template_path.is_file():
            raise SceneError(f"Missing collection template: {template_path}")
        original = read_template(template_path)
        if original is None:
            raise SceneError(f"Unable to decode collection template: {template_path}")
        for size in (0.8, 0.9, 1.0, 1.1, 1.2):
            target = (
                max(2, round(original.shape[1] * width / 1280 * size)),
                max(2, round(original.shape[0] * height / 720 * size)),
            )
            template = cv2.resize(original, target, interpolation=cv2.INTER_LINEAR)
            if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
                continue
            result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            maxima = cv2.dilate(result, np.ones((5, 5), dtype=np.uint8))
            rows, columns = np.where((result >= threshold) & (result == maxima))
            for top, left in zip(rows.tolist(), columns.tolist(), strict=True):
                box = [left + x0, top + y0, left + x0 + target[0], top + y0 + target[1]]
                # The resource balance HUD occupies this corner; matches touching it are unsafe.
                if box[2] >= width * 0.78 and box[1] <= height * 0.38:
                    continue
                candidates.append({
                    "resource": resource, "bbox": box,
                    "point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                    "confidence": round(float(result[top, left]), 5),
                    "template": str(template_path), "scale": size,
                })
    kept: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: float(item["confidence"]), reverse=True):
        a = candidate["bbox"]
        overlaps = False
        for previous in kept:
            b = previous["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if intersection / area > 0.3:
                overlaps = True
                break
        if not overlaps:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item["point"][1], item["point"][0]))


class ScreenshotRecognizer:
    def __init__(
        self,
        ocr_config: OCRConfig | None = None,
        baseline_resolution: tuple[int, int] = (1280, 720),
        *,
        provider: OCRProvider | None = None,
    ) -> None:
        self.ocr_config = ocr_config or OCRConfig()
        self.baseline_resolution = baseline_resolution
        self.provider = provider or create_ocr_provider(self.ocr_config)
        self.client_version = "unknown"
        from .battle_vision import BattleObserver

        self.battle_observer = BattleObserver(self)
        self._tracked_buildings: list[dict] = []

    def recognize_battle(self, screenshot_path: str | Path, *, purpose: str,
                         slot: dict | None = None, previous: SceneSnapshot | None = None) -> SceneSnapshot:
        if purpose not in {"troop_count", "hero", "settlement"}:
            raise ValueError(f"Unknown battle observation purpose: {purpose}")
        if purpose == "troop_count" and slot is None:
            raise ValueError("Troop observation requires a known card")
        return self.battle_observer.observe(Path(screenshot_path), purpose=purpose, slot=slot, previous=previous)

    def recognize(self, screenshot_path: str | Path, *, battle_details: bool = True) -> SceneSnapshot:
        started = time.monotonic()
        path = Path(screenshot_path)
        resolution = self._resolution(path)
        decoded = time.monotonic()
        results = filter_ocr_results(self.provider.recognize(path), self.ocr_config.confidence_threshold)
        ocr_finished = time.monotonic()
        texts = [self._baseline_result(result, resolution) for result in results]
        scene, confidence, reasons = classify_scene_text("\n".join(result.text for result in texts))
        surrender_dialog = None
        if scene in {SCENE_BATTLE, SCENE_ENEMY_VILLAGE, SCENE_POPUP, SCENE_UNKNOWN}:
            surrender_dialog = surrender_dialog_evidence([asdict(item) for item in texts])
            if surrender_dialog is not None:
                # The pale dialog is not cloud cover and its background HUD is
                # not an active battle surface while confirmation is pending.
                scene, confidence, reasons = SCENE_POPUP, .95, ["surrender_dialog_geometry_verified"]
        startup_logo = self._startup_logo(path) if scene == SCENE_UNKNOWN else None
        if startup_logo is not None:
            scene, confidence, reasons = SCENE_STARTING, 0.65, ["supercell_startup_logo"]
        cloud_cover = None
        if scene in {SCENE_ENEMY_VILLAGE, SCENE_BATTLE}:
            from .terrain import measure_cloud_cover

            cloud_cover = measure_cloud_cover(path)
            if cloud_cover["obscured"]:
                # HUD text can be legible before the cloud transition has cleared.
                scene, confidence, reasons = SCENE_UNKNOWN, 0.0, ["cloud_transition_obscures_interaction"]
        request_dialog = None
        if scene == SCENE_CLAN_CHAT:
            # The chat text appears before the sliding panel reaches its final
            # position; its fixed navigation regions are not usable yet.
            panel_open = any(normalize_label(item.text) in {"友谊战", "friendlychallenge", "friendlybattle"}
                             and self._in_observed_roi(item, (302, 660, 322, 712)) for item in texts)
            if not panel_open:
                scene, confidence, reasons = SCENE_UNKNOWN, 0.0, ["clan_chat_panel_position_unverified"]
        if scene == SCENE_REQUEST:
            title_seen = any(normalize_label(item.text) in {"请求增援", "请求援军", "请求部队", "requestreinforcements"}
                             and self._in_observed_roi(item, (500, 45, 790, 120)) for item in texts)
            controls = {name for item in texts if item.bbox is not None for name in ("cancel", "send")
                        if normalize_label(item.text) in {normalize_label(label) for label in BUTTON_LABELS[name]}
                        and self._request_dialog_button(name, item.bbox)}
            if not title_seen or controls != {"cancel", "send"}:
                scene, confidence, reasons = SCENE_UNKNOWN, 0.0, ["request_dialog_geometry_unverified"]
            else:
                request_dialog = self._request_dialog_appearance(path)
                if not request_dialog["ready"]:
                    scene, confidence, reasons = SCENE_UNKNOWN, 0.0, ["request_dialog_opacity_unverified"]
        regular_search = scene == SCENE_SEARCH and any(
            normalize_label(item.text) == "常规战" and self._in_observed_roi(item, (100, 420, 350, 490))
            for item in texts
        )
        buttons = []
        for result in texts:
            if result.bbox is None:
                continue
            normalized = normalize_label(result.text)
            for name, labels in BUTTON_LABELS.items():
                if surrender_dialog is not None and (
                    name not in {"confirm", "cancel"} or
                    list(result.bbox) != list(surrender_dialog[name]["bbox"])
                ):
                    continue
                if scene in {SCENE_UNKNOWN, SCENE_MAINTENANCE}:
                    continue
                if scene == SCENE_DISCONNECTED and not (
                    name == "retry" and self._in_observed_roi(result, (305, 380, 975, 460))
                ):
                    continue
                if scene == SCENE_SETTLEMENT and not (
                    name == "return_home" and self._in_observed_roi(result, (520, 570, 760, 675))
                ):
                    continue
                if scene == SCENE_VILLAGE and name not in {"attack", "shop"}:
                    continue
                if scene == SCENE_REQUEST and not self._request_dialog_button(name, result.bbox):
                    continue
                if scene == SCENE_CLAN_CHAT and (
                    name != "donate" or result.bbox[2] > self.baseline_resolution[0] * 0.38
                ):
                    continue
                if scene == SCENE_SEARCH and (
                    name != "find_match" or not regular_search or not self._in_observed_roi(result, (100, 490, 350, 570))
                ):
                    continue
                if scene in {SCENE_ENEMY_VILLAGE, SCENE_BATTLE} and not (
                    (name == "next" and scene == SCENE_ENEMY_VILLAGE and self._in_observed_roi(result, (1040, 450, 1275, 570)))
                    or (name == "end_battle" and self._in_observed_roi(result, (0, 480, 180, 590)))
                ):
                    continue
                if any(normalized == normalize_label(label) for label in labels):
                    left, top, right, bottom = result.bbox
                    buttons.append({
                        "name": "find_match_regular" if scene == SCENE_SEARCH else name,
                        "text": result.text, "confidence": result.confidence,
                        "bbox": list(result.bbox), "point": [(left + right) // 2, (top + bottom) // 2],
                    })
        if scene == SCENE_CLAN_CHAT:
            for name in ("request_open", "chat_latest"):
                navigation = self._chat_navigation(name, path, resolution)
                if navigation is not None:
                    buttons.append(navigation)
        observations: dict[str, object] = {
            "ocr": [asdict(result) for result in texts],
            "buttons": buttons,
            "source_resolution": list(resolution),
            "baseline_resolution": list(self.baseline_resolution),
            "matched_reasons": reasons,
            "recognition_seconds": round(time.monotonic() - started, 3),
            "resource_source": None,
            "resources": {"gold": None, "elixir": None, "dark_elixir": None, "gems": None},
            "resource_capacities": {"gold": None, "elixir": None, "dark_elixir": None},
            "village_type": "unknown",
            "village_type_evidence": None,
            "resource_evidence": {},
            "collectibles": [],
            "army": None,
            "army_editor": None,
            "mode": "regular" if regular_search else None,
            "search_cost_gold": None,
            "search_cost_evidence": None,
            "battle": None,
            "buildings": [],
            "settlement": None,
            "request_dialog": request_dialog,
            "surrender_dialog": surrender_dialog,
            "cloud_cover": cloud_cover,
            "startup_logo": startup_logo,
        }
        if scene == SCENE_VILLAGE:
            village_type = recognize_village_type(path)
            observations["village_type"] = village_type["type"]
            observations["village_type_evidence"] = village_type
            resources, evidence = self._village_resources(path, texts, resolution)
            observations["resources"] = resources
            observations["resource_evidence"] = evidence
            observations["resource_source"] = "village_inventory"
            observations["resource_capacities"] = self._village_resource_capacities(texts)
            observations["collectibles"] = detect_collectibles(path, baseline_resolution=self.baseline_resolution)
        elif scene == SCENE_TRAINING:
            observations["army"] = self._army_capacities(path, texts, resolution)
            from .army_manifest import recognize_army_manifest

            observations["army"]["manifest"] = recognize_army_manifest(
                path, self.provider, texts, baseline_resolution=self.baseline_resolution,
                client_version=self.client_version, capacities=observations["army"])
        elif scene == SCENE_SETTLEMENT:
            settlement = self._settlement_observation(path, texts, resolution)
            observations["settlement"] = settlement
            observations["resources"] = {**settlement["loot"], "gems": None}
            observations["resource_evidence"] = settlement["evidence"]["loot"]
            observations["resource_source"] = "settlement_gained"
        elif scene in {SCENE_ENEMY_VILLAGE, SCENE_BATTLE} and battle_details:
            resources, evidence = self._enemy_resources(path, texts, resolution)
            observations["resources"] = resources
            observations["resource_evidence"] = evidence
            observations["resource_source"] = "enemy_available" if scene == SCENE_ENEMY_VILLAGE else "enemy_remaining"
            observations["battle"] = self._battle_observation(path, texts, scene)
            from .building_vision import detect_buildings

            buildings = detect_buildings(path, previous=self._tracked_buildings,
                                         baseline_resolution=self.baseline_resolution,
                                         client_version=self.client_version)
            observations["buildings"] = buildings
            observations["battle"]["buildings"] = buildings
            self._tracked_buildings = buildings
            if scene == SCENE_ENEMY_VILLAGE:
                prices = [item for item in texts if self._in_observed_roi(item, (1098, 515, 1250, 560))
                          and parse_resource_number(item.text) is not None]
                if prices:
                    price = max(prices, key=lambda item: item.confidence)
                    observations["search_cost_gold"] = parse_resource_number(price.text)
                    observations["search_cost_evidence"] = asdict(price)
        elif regular_search:
            prices = [item for item in texts if self._in_observed_roi(item, (145, 530, 245, 573))
                      and parse_resource_number(item.text) is not None]
            if prices:
                price = max(prices, key=lambda item: item.confidence)
                observations["search_cost_gold"] = parse_resource_number(price.text)
                observations["search_cost_evidence"] = asdict(price)
        if scene not in {SCENE_ENEMY_VILLAGE, SCENE_BATTLE, SCENE_UNKNOWN}:
            self._tracked_buildings = []
        if scene in {SCENE_TRAINING, SCENE_UNKNOWN}:
            from .army_editor import recognize_army_editor

            observations["army_editor"] = recognize_army_editor(
                path, self.provider, texts, baseline_resolution=self.baseline_resolution,
                client_version=self.client_version)
            editor = observations["army_editor"]
            if (scene == SCENE_TRAINING and editor.get("surface") == "current"
                    and isinstance(observations.get("army"), dict)
                    and observations["army"].get("manifest", {}).get("supported_layout") is True):
                from .army_manifest import recognize_army_heroes, recognize_army_siege
                from .army_editor import recognize_hero_loadout

                army = observations["army"]
                heroes = recognize_army_heroes(
                    path, army.get("heroes"), baseline_resolution=self.baseline_resolution,
                    client_version=self.client_version)
                siege = recognize_army_siege(
                    path, self.provider, army.get("siege"),
                    baseline_resolution=self.baseline_resolution,
                    client_version=self.client_version)
                army["identity_cards"] = heroes["cards"] + siege["cards"]
                army["identity_coverage"] = {"hero": heroes["complete"],
                                             "siege": siege["complete"]}
                army["heroes_complete"] = heroes["complete"]
                army["hero_evidence"] = heroes
                army["siege_evidence"] = siege
                army["hero_loadout_complete"] = False
                army["hero_loadout"] = {}
                if heroes["complete"]:
                    loadout = recognize_hero_loadout(
                        path, texts, heroes["cards"],
                        baseline_resolution=self.baseline_resolution)
                    army["hero_loadout_evidence"] = loadout
                    if loadout["complete"]:
                        army["hero_loadout_complete"] = True
                        army["hero_loadout"] = loadout["hero_loadout"]
            if (scene == SCENE_UNKNOWN and editor.get("surface") in {"picker", "current_picker"}
                    and editor.get("ready") is True
                    and isinstance(editor.get("capacities"), dict)
                    and "troop" in editor["capacities"]):
                # The expanded picker can obscure its title. Its independently
                # read troop capacity plus the two-row visual picker anchor
                # establish this training surface without inferring card IDs.
                scene, confidence = SCENE_TRAINING, .88
                observations["matched_reasons"] = ["army_picker_visual_and_capacity_anchors"]
        observations["recognition_seconds"] = round(time.monotonic() - started, 3)
        observations["recognition_timings_sec"] = {"decode": decoded - started,
            "ocr": ocr_finished - decoded, "details": time.monotonic() - ocr_finished}
        return SceneSnapshot(scene, confidence, path, observations)

    @staticmethod
    def _startup_logo(path: Path) -> dict | None:
        import cv2
        import numpy as np

        source = read_frame(path, cv2.IMREAD_COLOR)
        if source is None:
            raise SceneError(f"Unable to decode screenshot: {path}")
        image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
        outside = np.ones((720, 1280), dtype=bool)
        outside[160:560, 420:860] = False
        dark_fraction = float((image.max(axis=2)[outside] < 20).mean())
        if dark_fraction < .99:
            return None
        template_path = Path(__file__).resolve().parents[2] / "assets/templates/startup_supercell.png"
        if not template_path.is_file():
            return None
        template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise SceneError(f"Unable to decode startup template: {template_path}")
        region = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[160:560, 420:860]
        matches = []
        for scale in (.95, 1.0, 1.05):
            resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if resized.shape[0] > region.shape[0] or resized.shape[1] > region.shape[1]:
                continue
            score = float(cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED).max())
            if math.isfinite(score) and score >= .92:
                matches.append((score, scale))
        if not matches:
            return None
        score, scale = max(matches)
        return {"template": str(template_path), "match_score": round(score, 6), "scale": scale,
                "outside_dark_fraction": round(dark_fraction, 6), "roi_at_1280x720": [420, 160, 860, 560]}

    def recognize_region(
        self, screenshot_path: str | Path, roi: tuple[int, int, int, int],
        *, single_line: bool = False,
    ) -> list[OCRText]:
        """Read a baseline ROI; returned boxes remain relative to the full baseline frame."""
        path = Path(screenshot_path)
        resolution = self._resolution(path)
        source_roi = scale_box(roi, from_resolution=self.baseline_resolution, to_resolution=resolution)
        read = self.provider.recognize_line if single_line else self.provider.recognize
        results = filter_ocr_results(read(path, source_roi), self.ocr_config.confidence_threshold)
        return [self._baseline_result(result, resolution) for result in results]

    def _baseline_result(self, result: OCRText, resolution: tuple[int, int]) -> OCRText:
        bbox = None if result.bbox is None else scale_box(
            result.bbox, from_resolution=resolution, to_resolution=self.baseline_resolution,
        )
        return OCRText(result.text, result.confidence, bbox)

    def recognize_slot_count(self, screenshot_path: str | Path, bbox: tuple[int, int, int, int],
                             *, count_bbox: tuple[int, int, int, int] | None = None) -> dict[str, object]:
        """Read an existing card's header, including explicit counts on gray cards."""
        import cv2
        import numpy as np
        from tempfile import TemporaryDirectory

        path = Path(screenshot_path)
        resolution = self._resolution(path)
        left, top, right, bottom = bbox
        width, height = self.baseline_resolution
        pad_top, pad_bottom = round(16 * height / 720), round(40 * height / 720)
        roi = (max(0, left), max(0, top - pad_top), min(width, right), min(height, top + pad_bottom))
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise SceneError(f"Army card bbox is outside the baseline frame: {bbox}")
        line_roi = roi
        if count_bbox is not None:
            cl, ct, cr, cb = count_bbox
            if not (roi[0] <= cl < cr <= roi[2] and roi[1] <= ct < cb <= roi[3]):
                raise SceneError(f"Army count bbox is outside its card header: {count_bbox}")
            # The last deployment shifts the gray x0 glyph toward the card's
            # right edge. Keep enough left context for line OCR even when the
            # previous x10 detection was narrow; the crop stays inside this card.
            px, py = max(1, round(10 * width / 1280)), max(1, round(4 * height / 720))
            # A prior x1 box is narrower than x0, and deselection can move the
            # glyphs down. Anchor to it without clipping the new two-character count.
            line_right = min(roi[2], cr + px)
            line_left = max(roi[0], min(cl - px, line_right - 2 * (cb - ct)))
            line_roi = (line_left, max(roi[1], ct - py), line_right, min(roi[3], cb + 2 * py))
        native_roi = scale_box(roi, from_resolution=self.baseline_resolution, to_resolution=resolution)
        if self.battle_observer.cards:
            image = cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)
            fast = self.battle_observer.count(path, image, {"point": [(left + right) // 2, (top + bottom) // 2],
                                                         "bbox": list(bbox)})
            if fast is not None:
                return fast
        reads = []
        for item in self.provider.recognize(path, native_roi):
            item = self._baseline_result(item, resolution)
            if item.bbox is not None:
                cx, cy = (item.bbox[0] + item.bbox[2]) / 2, (item.bbox[1] + item.bbox[3]) / 2
                if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                    reads.append({**asdict(item), "source": "header_region_ocr"})

        if count_bbox is not None:
            current_boxes = [item["bbox"] for item in reads if math.isfinite(item["confidence"])
                             and 0 < item["confidence"] <= 1
                             and re.fullmatch(r"[xX×]\s*[0-9]+", item["text"].strip())]
            if len(current_boxes) == 1:
                # A weak detection may locate glyphs without proving their value.
                # Include that current extent, then require a high-confidence read.
                cl, ct, cr, cb = current_boxes[0]
                line_roi = (max(roi[0], min(line_roi[0], cl - px)), max(roi[1], min(line_roi[1], ct - py)),
                            min(roi[2], max(line_roi[2], cr + px)), min(roi[3], max(line_roi[3], cb + py)))

        read_line = getattr(self.provider, "recognize_line", None)
        if read_line is not None:
            image = read_frame(path, cv2.IMREAD_GRAYSCALE)
            x0, y0, x1, y1 = scale_box(line_roi, from_resolution=self.baseline_resolution, to_resolution=resolution)
            # Exhausted cards dim both the portrait and count. Preserve explicit
            # glyphs while separating their gray fill from the dark header.
            binary = cv2.threshold(image[y0:y1, x0:x1], 120, 255, cv2.THRESH_BINARY)[1]
            with TemporaryDirectory(prefix="autococ-card-count-") as directory:
                crop_path = Path(directory) / "header.png"
                ok, encoded = cv2.imencode(".png", binary)
                if not ok:
                    raise SceneError("Unable to encode army count header")
                encoded.tofile(crop_path)
                for item in read_line(crop_path, (0, 0, x1 - x0, y1 - y0)):
                    reads.append({"text": item.text, "confidence": item.confidence,
                                  "bbox": list(line_roi), "source": "count_bbox_threshold_line" if count_bbox else "header_threshold_line",
                                  "threshold": 120})

        confident = [item for item in reads if math.isfinite(item["confidence"])
                     and max(.9, self.ocr_config.confidence_threshold) <= item["confidence"] <= 1]
        matches = [re.fullmatch(r"[xX×]\s*([0-9]+)", item["text"].strip()) for item in confident]
        counts = {int(match[1]) for match in matches if match is not None}
        # A thresholded selected-card border can read as a standalone line.
        # Only those explicit line shapes are decoration; every other
        # unparsed high-confidence read still makes the quantity uncertain.
        decorative_lines = {"一", "-", "—", "─"}
        ambiguous = any(match is None and item["text"].strip() not in decorative_lines
                        for item, match in zip(confident, matches, strict=True))
        count = next(iter(counts)) if len(counts) == 1 and not ambiguous else None
        confidence = (max((item["confidence"] for item, match in zip(confident, matches, strict=True)
                           if match is not None and int(match[1]) == count), default=0.0)
                      if count is not None else max((item["confidence"] for item in confident), default=0.0))
        return {"count": count, "confidence": confidence,
                "frame": str(path), "slot_bbox": list(bbox), "roi": list(roi), "readings": reads,
                "count_bbox": list(count_bbox) if count_bbox is not None else None, "line_roi": list(line_roi),
                "reason": "explicit_header_count" if count is not None else "header_count_unverified"}

    def _in_observed_roi(self, item: OCRText, roi: tuple[int, int, int, int]) -> bool:
        if item.bbox is None:
            return False
        box = scale_box(roi, from_resolution=(1280, 720), to_resolution=self.baseline_resolution)
        cx, cy = (item.bbox[0] + item.bbox[2]) / 2, (item.bbox[1] + item.bbox[3]) / 2
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    def _request_dialog_appearance(self, path: Path) -> dict[str, object]:
        import cv2
        import numpy as np

        image = cv2.resize(read_frame(path, cv2.IMREAD_COLOR), (1280, 720), interpolation=cv2.INTER_AREA)
        panel_coverage = {}
        # These empty margins are opaque cream in every observed settled dialog.
        # During fade-in, map/chat pixels remain visible and contaminate body OCR.
        for name, (left, top, right, bottom) in {
            "left": (334, 110, 354, 548), "right": (925, 110, 945, 548),
            "bottom": (345, 555, 935, 565),
        }.items():
            patch = image[top:bottom, left:right].astype(np.int16)
            panel_coverage[name] = float(np.all(np.abs(patch - (225, 234, 234)) <= 3, axis=2).mean())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        border_coverage = {}
        for name, (left, top, right, bottom), axis in (
            ("left", (368, 314, 375, 419), 1), ("right", (907, 314, 914, 419), 1),
            ("top", (380, 300, 905, 310), 0), ("bottom", (380, 423, 905, 431), 0),
        ):
            border_coverage[name] = float((gray[top:bottom, left:right].min(axis=axis) < 70).mean())
        body = image[312:421, 378:906].astype(np.int16)
        white_fraction = float(((body.min(axis=2) >= 240) & (np.ptp(body, axis=2) <= 12)).mean())
        # Reserve ample foreground for multiple request-text lines; no line is cropped.
        ready = min(panel_coverage.values()) >= .98 and min(border_coverage.values()) >= .98 and white_fraction >= .70
        return {"ready": ready, "method": "opaque_panel_and_closed_input_border",
                "panel_coverage": panel_coverage, "input_border_coverage": border_coverage,
                "input_white_fraction": white_fraction, "coordinate_resolution": [1280, 720]}

    def _enemy_resources(
        self, path: Path, texts: list[OCRText], resolution: tuple[int, int],
    ) -> tuple[dict[str, int | None], dict[str, object]]:
        import cv2
        import numpy as np

        image = cv2.resize(read_frame(path, cv2.IMREAD_COLOR), (1280, 720), interpolation=cv2.INTER_AREA)
        values: dict[str, int | None] = {"gold": None, "elixir": None, "dark_elixir": None, "gems": None}
        evidence: dict[str, object] = {}
        # These offsets follow the actual icon crops, including the smaller dark
        # elixir crop. They locate text relative to the icon, not a fixed row.
        for name, offsets, center_offset in (
            ("gold", (29, -3, 193, 23), 12),
            ("elixir", (29, -3, 193, 23), 12),
            ("dark_elixir", (26, -6, 190, 20), 9),
        ):
            template_path = Path(__file__).resolve().parents[2] / "assets/templates" / f"enemy_{name}.png"
            if not template_path.is_file():
                continue
            template = read_template(template_path)
            if template is None:
                continue
            scores = cv2.matchTemplate(image[85:220, 20:72], template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            if not math.isfinite(score) or score < .9:
                continue
            x, y = location
            other_scores = scores.copy()
            other_scores[max(0, y - 15):y + 16, :] = -1
            if float(other_scores.max()) >= .9:
                continue
            left, top = x + 20, y + 85
            icon = {"bbox_at_1280x720": (left, top, left + template.shape[1], top + template.shape[0]),
                    "confidence": float(score), "template": str(template_path)}
            roi = (left + offsets[0], top + offsets[1], left + offsets[2], top + offsets[3])
            candidates = []
            for item in texts:
                if item.bbox is None or parse_resource_number(item.text) is None:
                    continue
                box = scale_box(item.bbox, from_resolution=self.baseline_resolution, to_resolution=(1280, 720))
                l, t, r, b = box
                # Defender levels are smaller and can sit just below the amount.
                # A center-in-broad-ROI test admits them even at higher confidence.
                if (math.isfinite(item.confidence) and self.ocr_config.confidence_threshold <= item.confidence <= 1
                        and roi[0] - 3 <= l < r <= roi[2] and 20 <= b - t <= 34
                        and abs((t + b) / 2 - (top + center_offset)) <= 4):
                    candidates.append(item)
            source = "full_ocr"
            if not candidates:
                read_line = getattr(self.provider, "recognize_line", None)
                if read_line is not None:
                    native_roi = scale_box(roi, from_resolution=(1280, 720), to_resolution=resolution)
                    candidates = [self._baseline_result(item, resolution) for item in
                                  filter_ocr_results(read_line(path, native_roi), max(0.9, self.ocr_config.confidence_threshold))
                                  if math.isfinite(item.confidence) and item.confidence <= 1
                                  and parse_resource_number(item.text) is not None]
                    source = "line_roi"
            # Conflicting readings or separated digit fragments remain unknown;
            # neither magnitude nor OCR confidence determines which is the loot.
            if len(candidates) == 1:
                item = candidates[0]
                if source == "full_ocr":
                    box = scale_box(item.bbox, from_resolution=self.baseline_resolution, to_resolution=(1280, 720))
                    if box[0] > roi[0] + 10:
                        continue
                values[name] = parse_resource_number(item.text)
                evidence[name] = {**asdict(item), "source": source, "resource_icon": icon,
                                  "line_roi_at_1280x720": roi}
        return values, evidence

    def _settlement_observation(
        self, path: Path, texts: list[OCRText], resolution: tuple[int, int],
    ) -> dict[str, object]:
        import cv2
        import numpy as np

        image = cv2.resize(read_frame(path, cv2.IMREAD_COLOR), (1280, 720), interpolation=cv2.INTER_AREA)
        resources = ("gold", "elixir", "dark_elixir")
        loot = {name: None for name in resources}
        bonus = {name: None for name in resources}
        evidence: dict[str, object] = {"loot": {}, "bonus": None, "percentage": None, "stars": None}
        icons = {}
        icon_scores = {}
        for name in resources:
            template_path = Path(__file__).resolve().parents[2] / "assets/templates" / f"settlement_{name}.png"
            if not template_path.is_file():
                continue
            template = read_template(template_path)
            if template is None:
                continue
            mask = None
            if name == "dark_elixir":
                # This observed 41x45 drop has transparent corners over the map.
                # Compare its opaque body, preserving the full box for row geometry.
                mask = np.zeros(template.shape[:2], dtype=np.uint8)
                cv2.ellipse(mask, (21, 26), (14, 16), 0, 0, 360, 255, -1)
            scores = cv2.matchTemplate(image[290:455, 655:735], template, cv2.TM_CCOEFF_NORMED, mask=mask)
            scores[~np.isfinite(scores)] = -1
            _, score, _, location = cv2.minMaxLoc(scores)
            icon_scores[name] = float(score)
            if score < .9:
                continue
            x, y = location
            # Multiple separated matches could be a bonus/changed layout. Refuse an
            # ambiguous icon rather than bind its numeric value to the wrong row.
            other_scores = scores.copy()
            other_scores[max(0, y - 25):y + 26, :] = -1
            if float(other_scores.max()) >= .9:
                continue
            box = (x + 655, y + 290, x + 655 + template.shape[1], y + 290 + template.shape[0])
            icons[name] = {"bbox_at_1280x720": box, "confidence": float(score), "template": str(template_path),
                           "comparison": "opaque_drop_body" if mask is not None else "full_template"}
        for name, icon in icons.items():
            left, top, right, bottom = icon["bbox_at_1280x720"]
            roi = (left - 195, top - 4, left - 2, bottom + 2)

            def amount(item: OCRText) -> int | None:
                if (not math.isfinite(item.confidence) or not max(.9, self.ocr_config.confidence_threshold) <= item.confidence <= 1
                        or not self._in_observed_roi(item, roi)):
                    return None
                return parse_resource_number(item.text)

            candidates = [item for item in texts if amount(item) is not None]
            source = "full_ocr"
            if not candidates:
                read_line = getattr(self.provider, "recognize_line", None)
                if read_line is not None:
                    native_roi = scale_box(roi, from_resolution=(1280, 720), to_resolution=resolution)
                    line = [self._baseline_result(item, resolution) for item in read_line(path, native_roi)]
                    candidates = [item for item in line if amount(item) is not None]
                    source = "line_roi"
                    if not candidates:
                        from .settlement import locate_number_line

                        tight_roi = locate_number_line(image, roi)
                        if tight_roi is not None:
                            native_roi = scale_box(tight_roi, from_resolution=(1280, 720), to_resolution=resolution)
                            line = [self._baseline_result(item, resolution) for item in read_line(path, native_roi)]
                            candidates = [item for item in line if amount(item) is not None]
                            source = "glyph_line_roi"
            if len(candidates) == 1:
                item = candidates[0]
                loot[name] = amount(item)
                evidence["loot"][name] = {**asdict(item), "source": source, "resource_icon": icon}
        percentages = [item for item in texts if self._in_observed_roi(item, (570, 95, 715, 145))
                       and re.fullmatch(r"\s*\d{1,3}\s*[%％]\s*", item.text)]
        percentage = None
        if percentages:
            item = max(percentages, key=lambda candidate: candidate.confidence)
            value = int(re.sub(r"\D", "", item.text))
            if 0 <= value <= 100:
                percentage, evidence["percentage"] = value, asdict(item)
        anchors = {}
        for name, labels, roi in (
            ("defeat", {"失败", "defeat"}, (545, 175, 745, 260)),
            ("victory", {"胜利", "victory"}, (545, 175, 745, 260)),
            ("received", {"您得到了", "lootgained"}, (555, 275, 730, 330)),
            ("losses", {"损耗的部队", "损耗部队"}, (550, 450, 735, 495)),
            ("return", {"回营", "returnhome"}, (520, 570, 760, 675)),
        ):
            matches = [item for item in texts if normalize_label(item.text) in labels and self._in_observed_roi(item, roi)]
            if matches:
                anchors[name] = asdict(max(matches, key=lambda item: item.confidence))
        stars = 0 if "defeat" in anchors and "victory" not in anchors and percentage is not None and percentage < 50 else None
        if stars == 0:
            evidence["stars"] = {"source": "explicit_defeat_result", "result": anchors["defeat"]}
        if "victory" in anchors and "defeat" not in anchors:
            from .settlement import recognize_earned_stars

            star_result = recognize_earned_stars(image)
            stars = star_result["count"]
            evidence["stars"] = {"source": "explicit_victory_star_shapes", "result": anchors["victory"],
                                 **star_result["evidence"]}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        separator_rows = []
        # The one-pixel translucent separators fragment under Canny on busy maps.
        # Verify their long, locally brighter horizontal ridge directly instead.
        for y in range(322, 457):
            line = gray[y:y + 2, 340:940].max(axis=0)
            above = gray[y - 4:y - 2, 340:940].mean(axis=0)
            below = gray[y + 3:y + 5, 340:940].mean(axis=0)
            if float(((line - np.maximum(above, below)) > 40).mean()) >= .75:
                separator_rows.append(y)
        row_groups = []
        for y in sorted(set(separator_rows)):
            if row_groups and y - row_groups[-1][-1] <= 5:
                row_groups[-1].append(y)
            else:
                row_groups.append([y])
        separators = [sum(group) / len(group) for group in row_groups]
        ordered_icons = sorted(icons, key=lambda name: icons[name]["bbox_at_1280x720"][1])
        icon_rows = [icons[name]["bbox_at_1280x720"] for name in ordered_icons]
        rows_complete = len(separators) == len(icon_rows) and all(
            abs(row[3] - separator) <= 5 for row, separator in zip(icon_rows, separators)
        )
        bonus_marker = any(any(word in item.text.lower() for word in ("奖励", "bonus", "额外")) for item in texts)
        layout = None
        complete_result = len(anchors) == 4 and stars == 0 and rows_complete and not bonus_marker
        if complete_result and ordered_icons == list(resources) and all(value is not None for value in loot.values()):
            layout = "regular_defeat_three_rows_v1"
        elif (complete_result and ordered_icons == ["gold", "elixir"]
              and all(loot[name] is not None for name in ("gold", "elixir"))
              and icon_scores.get("dark_elixir", 1) < .75
              and abs(icon_rows[0][1] - 332) <= 4 and abs(icon_rows[1][1] - 385) <= 4):
            layout = "regular_defeat_gold_elixir_only_v1"
            # The settled two-row layout omits unearned dark elixir. This provisional
            # zero is valid for success only after combat's exact inventory reconciliation.
            loot["dark_elixir"] = 0
            evidence["loot"]["dark_elixir"] = {
                "source": "verified_absent_resource_row_layout", "layout": layout,
                "requires_inventory_reconciliation": True, "resource_icon_scores": icon_scores,
                "anchors": anchors, "separator_rows": separators,
            }
        if layout is not None:
            bonus = {name: 0 for name in resources}
            evidence["bonus"] = {"source": "verified_no_bonus_layout", "layout": layout,
                                 "requires_inventory_reconciliation": True,
                                 "anchors": anchors, "separator_rows": separators}
        if {"victory", "received", "losses", "return"} <= anchors.keys() and "defeat" not in anchors:
            bonus, bonus_evidence = self._settlement_bonus(path, image, texts, resolution)
            evidence["bonus"] = bonus_evidence
            if (stars is not None and rows_complete and ordered_icons == list(resources)
                    and all(value is not None for value in (*loot.values(), *bonus.values()))):
                layout = "victory_with_bonus_three_rows_v1"
        evidence["resource_icons"] = icons
        evidence["layout"] = layout
        return {"loot": loot, "bonus": bonus, "percentage": percentage, "stars": stars, "evidence": evidence}

    def _settlement_bonus(self, path: Path, image, texts: list[OCRText], resolution: tuple[int, int]) -> tuple[dict, dict]:
        from .settlement import recognize_bonus_icons

        values = {name: None for name in ("gold", "elixir", "dark_elixir")}
        evidence = {"source": "explicit_bonus_panel", "marker": None, "resources": {}}
        markers = [item for item in texts if self._in_observed_roi(item, (895, 305, 1025, 350))
                   and math.isfinite(item.confidence) and .9 <= item.confidence <= 1
                   and re.fullmatch(r"(?:[0-9]{1,3}[%％])?(?:奖励|bonus)", normalize_label(item.text))]
        if len(markers) != 1:
            return values, evidence
        evidence["marker"] = asdict(markers[0])
        for name, icon in recognize_bonus_icons(image).items():
            left, top, right, bottom = icon["bbox_at_1280x720"]
            roi = (left - 130, top - 3, left - 1, bottom + 2)

            def amount(item: OCRText) -> int | None:
                if (not math.isfinite(item.confidence) or not .9 <= item.confidence <= 1
                        or not self._in_observed_roi(item, roi)):
                    return None
                text = item.text.strip()
                return parse_resource_number(text[1:].strip()) if text.startswith(("+", "＋")) else None

            candidates = [item for item in texts if amount(item) is not None]
            source = "full_ocr"
            if not candidates:
                read_line = getattr(self.provider, "recognize_line", None)
                if read_line is not None:
                    native_roi = scale_box(roi, from_resolution=(1280, 720), to_resolution=resolution)
                    line = [self._baseline_result(item, resolution) for item in read_line(path, native_roi)]
                    candidates = [item for item in line if amount(item) is not None]
                    source = "line_roi"
            if len(candidates) == 1:
                item = candidates[0]
                values[name] = amount(item)
                evidence["resources"][name] = {**asdict(item), "source": source, "resource_icon": icon}
        return values, evidence

    def _battle_observation(self, path: Path, texts: list[OCRText], scene: str) -> dict[str, object]:
        import cv2
        import numpy as np

        timers = [item for item in texts if self._in_observed_roi(item, (530, 30, 750, 95)) and parse_countdown(item.text) is not None]
        timer = max(timers, key=lambda item: item.confidence) if timers else None
        image = cv2.resize(read_frame(path, cv2.IMREAD_COLOR), (1280, 720), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([75, 65, 60]), np.array([170, 255, 255]))
        contours, _ = cv2.findContours(mask[585:720], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        edge_gradient = cv2.Sobel(cv2.cvtColor(image[625:695], cv2.COLOR_BGR2GRAY),
                                  cv2.CV_32F, 1, 0, ksize=3)
        edge_strength = np.median(np.abs(edge_gradient), axis=0)
        edge_direction = np.median(edge_gradient, axis=0)

        def supported_sides(left: int, right: int) -> bool:
            if left < 3 or right > 1277:
                return False
            # The first battle card's left edge meets the screen-side map
            # strip. Its cyan border loses a few pixels against the map in
            # some camera positions, although the independent quantity,
            # edge gradients and closed bottom still locate that card.
            minimum = .70 if left < 30 else .75
            return all(max(float((mask[625:695, start + shift:start + shift + 3] > 0).mean())
                           for shift in (-2, -1, 0, 1, 2)) >= minimum
                       for start in (left, right - 3))

        def colored_bottom(left: int, right: int) -> int | None:
            rows = (mask[699:716, left + 7:right - 7] > 0).mean(axis=1)
            return next((y for y in range(706, 716) if rows[y - 699] < .3
                         and np.all(rows[y - 702:y - 699] >= .8)), None)

        def edge_backed_card(left: int, right: int) -> bool:
            """Recover a bordered card whose moving art masks cyan side pixels.

            This stricter alternative needs both partial colour sides, its own
            opposite-directed long edges, a closed bottom and one exact count
            on the current frame. The count never supplies card identity.
            """
            if left < 3 or right > 1277 or not 80 <= right - left <= 105:
                return False
            sides = [max(float((mask[625:695, start + shift:start + shift + 3] > 0).mean())
                         for shift in (-2, -1, 0, 1, 2)) for start in (left, right - 3)]
            if min(sides) < .4 or edge_direction[left] < 250 or edge_direction[right - 1] > -250:
                return False
            if colored_bottom(left, right) is None:
                return False
            counts = []
            for item in texts:
                if (item.bbox is None or not math.isfinite(item.confidence)
                        or not max(.9, self.ocr_config.confidence_threshold) <= item.confidence <= 1
                        or not re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())):
                    continue
                box = scale_box(item.bbox, from_resolution=self.baseline_resolution,
                                to_resolution=(1280, 720))
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if left <= cx <= right and 588 <= cy <= 627:
                    counts.append(item)
            if len(counts) == 1:
                return True
            if counts:
                return False
            # A transient low-confidence quantity should not erase a card
            # with its own borders and a verified versioned portrait. Its
            # quantity remains unknown until a later fresh OCR observation.
            from .unit_catalog import recognize_card_identity

            identities = [recognize_card_identity(
                image, (left, 595, right, 711), kind, surface="battle",
                client_version=self.client_version) for kind in ("troop", "spell")]
            return sum(identity["unit_id"] is not None and
                       isinstance(identity["confidence"], (int, float)) and
                       identity["confidence"] >= .97 for identity in identities) == 1

        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if not (80 <= width <= 105 and 105 <= height <= 125 and
                    (supported_sides(left, left + width) or edge_backed_card(left, left + width))):
                continue
            boxes.append((left, max(590, top + 585), left + width, min(715, top + 585 + height)))

        # Map colors can join multiple cards below their headers. Recover actual
        # header edges and the bottom border; never divide a connected width into cards.
        headers, _ = cv2.findContours(mask[595:619], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # A cyan siege header can connect to purple hero headers through map
        # colors. Its observed hue separates that header, still requiring borders.
        siege_header = cv2.inRange(hsv[595:619], np.array([75, 65, 60]), np.array([98, 255, 255]))
        siege_header = cv2.morphologyEx(siege_header, cv2.MORPH_OPEN, np.ones((8, 1), dtype=np.uint8))
        siege_headers, _ = cv2.findContours(siege_header, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        headers = list(headers) + list(siege_headers)
        for contour in headers:
            left, top, width, height = cv2.boundingRect(contour)
            right = left + width
            if not (80 <= width <= 105 and height >= 12 and supported_sides(left, right)):
                continue
            if width >= 96:
                # Map colour can extend a header a few pixels into the next
                # card. Use its own strong right-edge gradient and long side
                # border to recover the actual card boundary before overlap
                # filtering, rather than dropping two valid neighbours.
                refined = min(range(max(left + 80, right - 20), right),
                              key=lambda x: edge_direction[x - 1])
                if (refined <= right - 5 and edge_direction[refined - 1] < -150
                        and supported_sides(left, refined)):
                    right = refined
            if any(abs(left - box[0]) <= 4 and abs(right - box[2]) <= 4 for box in boxes):
                continue
            bottom = colored_bottom(left, right)
            if bottom is not None:
                boxes.append((left, top + 595, right, bottom))

        from .hero_state import _match

        hero_templates = sorted((Path(__file__).resolve().parents[2] / "assets/templates").glob("battle_hero_pet_*.png"))

        # A known pet icon can locate a header obscured by map colors, but the
        # card still needs its own two long side borders and a complete bottom.
        # Blue map backgrounds can join both card bodies and headers. Explicit
        # quantities provide independent anchors, still requiring actual borders.
        for item in texts:
            if (not self._in_observed_roi(item, (60, 588, 1230, 627))
                    or not math.isfinite(item.confidence) or not max(.9, self.ocr_config.confidence_threshold) <= item.confidence <= 1
                    or not re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())):
                continue
            text_box = scale_box(item.bbox, from_resolution=self.baseline_resolution, to_resolution=(1280, 720))
            center = (text_box[0] + text_box[2]) / 2
            if any(box[0] <= center <= box[2] for box in boxes):
                continue
            right = max(range(max(4, text_box[2] - 4), min(1277, text_box[2] + 7)),
                        key=lambda x: -edge_direction[x - 1])
            left = max(range(max(3, right - 105), right - 79), key=lambda x: edge_direction[x])
            if not left < center < right or edge_direction[left] < 150 or edge_direction[right - 1] > -150:
                continue
            if not supported_sides(left, right):
                continue
            bottom = colored_bottom(left, right)
            if bottom is not None:
                boxes.append((left, 595, right, bottom))

        for template_path in hero_templates:
            match = _match(image, template_path, (0, 595, 1280, 646), mask_kind="pet")
            if match is None:
                continue
            anchor = match["bbox"][0]
            if not 8 <= anchor <= 1180:
                continue
            left = max((x for x in range(anchor - 8, anchor) if (mask[625:695, x:x + 3] > 0).mean() >= .75),
                       key=lambda x: edge_strength[x], default=None)
            right = max((x for x in range(anchor + 76, anchor + 94) if (mask[625:695, x - 3:x] > 0).mean() >= .75),
                        key=lambda x: edge_strength[x - 1], default=None)
            if left is None or right is None or not 80 <= right - left <= 105:
                continue
            if min(edge_strength[left], edge_strength[right - 1]) < 150:
                continue
            if any(abs(left - box[0]) <= 4 and abs(right - box[2]) <= 4 for box in boxes):
                continue
            bottom = colored_bottom(left, right)
            if bottom is not None:
                boxes.append((left, 595, right, bottom))

        # Selection enlarges a card and replaces its colored side border
        # with a closed white outline. This locates a card, not a deployed hero.
        white = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([180, 65, 255]))
        outlines, _ = cv2.findContours(white[570:720], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in outlines:
            left, top, width, height = cv2.boundingRect(contour)
            top += 570
            right, bottom = left + width, top + height
            if not (80 <= width <= 110 and 105 <= height <= 140 and bottom <= 716):
                continue
            # Anti-aliasing can leave a one-pixel white side. Require vertical
            # continuity per row instead of two fully white columns.
            sides = [(white[625:695, x:x + 2] > 0).any(axis=1).mean() for x in (left, right - 2)]
            top_edge = max((white[y, left + 12:right - 12] > 0).mean() for y in range(top, top + 3))
            bottom_edge = max((white[y, left + 12:right - 12] > 0).mean() for y in range(bottom - 3, bottom))
            if min(sides) < .75 or min(top_edge, bottom_edge) < .8:
                continue
            if any(min(right, box[2]) > max(left, box[0]) for box in boxes):
                continue
            boxes.append((left, top, right, colored_bottom(left, right) or bottom))

        # This event troop is added by the battle screen, outside My Army. Its
        # gray card has no blue/purple header; identify the sampled portrait.
        event_portrait = _match(image, Path(__file__).resolve().parents[2] / "assets/templates/battle_event_super_pekka.png",
                                (20, 620, 1250, 695), (1.0, .95, 1.05, 1.1))
        event_box = None
        if event_portrait is not None and event_portrait["confidence"] >= .95:
            left = round(event_portrait["bbox"][0] - 6 * event_portrait["scale"])
            right = round(left + 90 * event_portrait["scale"])
            if (3 <= left < right <= 1277 and
                    any(self._in_observed_roi(item, (left, 585, right, 630))
                        and item.confidence >= .9 and re.fullmatch(r"[xX×]\s*\d+", item.text.strip()) for item in texts)):
                event_box = (left, 595, right, 711)
                if not any(min(right, box[2]) > max(left, box[0]) for box in boxes):
                    boxes.append(event_box)

        # Deployed or used hero cards can lose the coloured header and their
        # pet icon. A versioned, hero-specific face proposes a card, but its
        # own opposing side edges and bottom boundary must also be present.
        from .hero_state import HERO_PHASE_TEMPLATE_IDS
        from .unit_catalog import CATALOG_ROOT, recognize_card_identity, template_manifest

        phase_cards = []
        catalog = template_manifest()
        if self.client_version == catalog["client"]:
            # Duke's unequipped card can lose the top-left coloured icon, which
            # splits its contour. Its versioned face only proposes a location;
            # both long card sides and the closed bottom must confirm it.
            duke_entries = [entry for entry in catalog["templates"]
                            if entry.get("unit_id") == "dragon_duke"
                            and entry.get("surface") == "battle"
                            and entry.get("client_version") == self.client_version]
            for entry in duke_entries:
                relative = entry.get("path")
                if not isinstance(relative, str):
                    continue
                template_path = (CATALOG_ROOT / relative).resolve()
                if not template_path.is_relative_to(CATALOG_ROOT.resolve()) or not template_path.is_file():
                    continue
                portrait = read_template(template_path)
                if portrait is None:
                    continue
                region = image[625:690]
                score_map = cv2.matchTemplate(region, portrait, cv2.TM_CCOEFF_NORMED)
                score_map[~np.isfinite(score_map)] = -1
                _, score, _, point = cv2.minMaxLoc(score_map)
                anchor_x, anchor_y = point[0], point[1] + 625
                if score < .97 or not 632 <= anchor_y <= 640 or anchor_x < 20:
                    continue
                # The sampled face begins about 20 px inside its own card.
                left = max(range(max(3, anchor_x - 24), anchor_x - 16),
                           key=lambda x: edge_direction[x])
                right = left + 90
                box = (left, 595, right, 711)
                if not (80 <= right - left <= 105 and right <= 1277
                        and edge_direction[left] >= 250
                        and min(edge_direction[right - 4:right]) <= -150
                        and supported_sides(left, right)
                        and colored_bottom(left, right) == 711):
                    continue
                identity = recognize_card_identity(image, box, "hero", surface="battle",
                                                   client_version=self.client_version)
                if identity["unit_id"] != "dragon_duke" or identity["confidence"] < .97:
                    continue
                if not any(min(right, old[2]) > max(left, old[0]) for old in boxes):
                    boxes.append(box)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for hero_id, template_hero in HERO_PHASE_TEMPLATE_IDS.items():
                for phase in ("ready", "used"):
                    portrait = _match(
                        image, Path(__file__).resolve().parents[2] / "assets/templates" /
                        f"hero_{template_hero}_{phase}_portrait.png", (20, 590, 1250, 650))
                    if portrait is None or not 590 <= portrait["bbox"][1] <= 599:
                        continue
                    left = portrait["bbox"][0] - 40
                    right = left + 89
                    if not (3 <= left < right <= 1277
                            and max(edge_direction[left - 2:left + 3]) >= 150
                            and min(edge_direction[right - 4:right]) <= -150
                            and gray[709:711, left + 8:right - 8].mean()
                            - gray[712, left + 8:right - 8].mean() >= 40):
                        continue
                    phase_cards.append({"hero": hero_id, "phase": phase,
                                        "portrait": portrait, "box": (left, 595, right, 711)})
            for candidate in phase_cards:
                left, _, right, _ = candidate["box"]
                if any(abs(left - other["box"][0]) <= 4 and other["hero"] != candidate["hero"]
                       for other in phase_cards):
                    continue
                if not any(abs(left - box[0]) <= 4 and abs(right - box[2]) <= 4 for box in boxes):
                    if not any(min(right, box[2]) > max(left, box[0]) for box in boxes):
                        boxes.append(candidate["box"])

        # An unresolved overlap is not two independently located cards.
        boxes = [box for index, box in enumerate(boxes) if not any(
            min(box[2], other[2]) > max(box[0], other[0])
            for other_index, other in enumerate(boxes) if index != other_index)]
        badge_path = Path(__file__).resolve().parents[2] / "assets/templates/battle_clan_badge.png"
        badge = read_template(badge_path) if badge_path.is_file() else None
        slots = []
        for index, box in enumerate(sorted(boxes)):
            baseline_box = scale_box(box, from_resolution=(1280, 720), to_resolution=self.baseline_resolution)
            count_roi = (box[0], 588, box[2], 627)
            counts = [item for item in texts if self._in_observed_roi(item, count_roi)
                      and math.isfinite(item.confidence) and max(.9, self.ocr_config.confidence_threshold) <= item.confidence <= 1
                      and re.fullmatch(r"[xX×]\s*[0-9]+", item.text.strip())]
            count = counts[0] if len(counts) == 1 else None
            strip = hsv[597:603, box[0] + 4:box[2] - 4]
            selected = (strip[:, :, 1] > 60) & (strip[:, :, 0] > 70) & (strip[:, :, 0] < 170)
            hue = float(np.median(strip[:, :, 0][selected])) if selected.any() else None
            kind = "unknown"
            hero_evidence = None
            if hue is not None:
                if hue < 99:
                    kind = "siege"
                elif hue < 118:
                    kind = "troop"
                elif count is not None:
                    kind = "spell"
                else:
                    for template_path in hero_templates:
                        match = _match(image, template_path, (box[0], max(560, box[1]), box[2], 646),
                                       (1.0, 1.05, 1.1), mask_kind="pet")
                        if match is not None and (hero_evidence is None or match["confidence"] > hero_evidence["confidence"]):
                            kind, hero_evidence = "hero", match
            clan_evidence = None
            is_event = event_box is not None and abs(box[0] - event_box[0]) < 10
            if is_event:
                kind = "troop"
            if kind in {"troop", "spell"} and badge is not None:
                header = image[588:628, box[0]:min(box[0] + 45, box[2])]
                for scale in (1.0, 1 / 1.1, 1.1):
                    candidate = badge if scale == 1 else cv2.resize(badge, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    score = cv2.minMaxLoc(cv2.matchTemplate(header, candidate, cv2.TM_CCOEFF_NORMED))[1]
                    if score >= .95 and (clan_evidence is None or score > clan_evidence["confidence"]):
                        clan_evidence = {"template": str(badge_path), "confidence": score, "scale": scale}
            from .unit_catalog import recognize_card_identity

            identity = recognize_card_identity(image, box, kind, surface="battle",
                                               client_version=self.client_version)
            if kind == "unknown":
                spell_identity = recognize_card_identity(
                    image, box, "spell", surface="battle",
                    client_version=self.client_version)
                # A hero's pet and equipment can change independently of the
                # hero. A sampled face may establish hero identity even when
                # the old pet-icon classifier has no matching template.
                hero_identity = recognize_card_identity(image, box, "hero", surface="battle",
                                                        client_version=self.client_version)
                if (spell_identity["unit_id"] is not None
                        and spell_identity["confidence"] >= .97
                        and hero_identity["unit_id"] is None):
                    kind, identity = "spell", spell_identity
                elif hero_identity["unit_id"] is not None:
                    kind, identity = "hero", hero_identity
            phases = [candidate for candidate in phase_cards
                      if abs(box[0] - candidate["box"][0]) <= 4
                      and abs(box[2] - candidate["box"][2]) <= 4]
            if (count is None and clan_evidence is None and not is_event and phases
                    and len({candidate["hero"] for candidate in phases}) == 1
                    and (identity["unit_id"] is None or identity["unit_id"] == phases[0]["hero"])):
                phase = max(phases, key=lambda candidate: candidate["portrait"]["confidence"])
                kind = "hero"
                identity = {"unit_id": phase["hero"],
                            "confidence": phase["portrait"]["confidence"],
                            "reason": "versioned_hero_phase_portrait",
                            "evidence": [phase["portrait"]], "version_verified": True}
            if is_event and identity["unit_id"] is None:
                # The legacy event portrait is not in the versioned unit
                # manifest. It can classify a card as event-sourced, but must
                # never bypass client-version or named-identity verification.
                identity = {"unit_id": None, "confidence": event_portrait["confidence"],
                            "reason": "unversioned_event_portrait_anonymous",
                            "evidence": [event_portrait], "version_verified": False}
            slots.append({
                "index": index, "kind": kind, "bbox": list(baseline_box),
                "unit_id": identity["unit_id"], "confidence": identity["confidence"],
                "source": "event" if is_event else "clan_reinforcement" if clan_evidence is not None else "army",
                "point": [(baseline_box[0] + baseline_box[2]) // 2, (baseline_box[1] + baseline_box[3]) // 2],
                "count": int(re.sub(r"\D", "", count.text)) if count is not None else None,
                "level": None, "available": None,
                "evidence": {"method": "independent_card_border", "header_hue": hue,
                             "hero_pet_icon": hero_evidence,
                             "clan_badge": clan_evidence,
                             "event_portrait": event_portrait if is_event else None,
                             "count": asdict(count) if count is not None else None},
            })
            slots[-1]["evidence"]["identity"] = identity
        return {"phase": "scout" if scene == SCENE_ENEMY_VILLAGE else "active",
                "countdown_seconds": parse_countdown(timer.text) if timer is not None else None,
                "countdown_evidence": asdict(timer) if timer is not None else None,
                "slots": slots, "missing_slot_means": "unknown_not_zero"}

    def _army_capacities(
        self, path: Path, texts: list[OCRText], source_resolution: tuple[int, int],
    ) -> dict[str, object]:
        # Numeric headers in the observed "My Army" panel, not troop-card levels/counts.
        rois = {
            "heroes": (40, 145, 103, 180),
            "troops": (577, 152, 667, 182),
            "spells": (577, 325, 638, 362),
            "siege": (1033, 330, 1081, 362),
            "clan_troops": (586, 520, 651, 548),
            "clan_spells": (709, 518, 756, 550),
            "clan_siege": (799, 520, 842, 548),
        }
        values: dict[str, object] = {name: None for name in rois}
        values["clan_received_unknown"] = True
        values["heroes_available"] = None
        for name, original_roi in rois.items():
            roi = scale_box(original_roi, from_resolution=(1280, 720), to_resolution=self.baseline_resolution)
            candidates = [item for item in texts if item.bbox is not None and parse_capacity(item.text) is not None
                          and roi[0] <= (item.bbox[0] + item.bbox[2]) / 2 <= roi[2]
                          and roi[1] <= (item.bbox[1] + item.bbox[3]) / 2 <= roi[3]]
            source = "full_ocr"
            if not candidates:
                source_roi = scale_box(roi, from_resolution=self.baseline_resolution, to_resolution=source_resolution)
                read_line = getattr(self.provider, "recognize_line", None)
                if read_line is None:
                    continue
                line_results = filter_ocr_results(read_line(path, source_roi), self.ocr_config.confidence_threshold)
                candidates = [self._baseline_result(item, source_resolution) for item in line_results if parse_capacity(item.text) is not None]
                source = "line_roi"
            if candidates:
                item = max(candidates, key=lambda candidate: candidate.confidence)
                used, capacity = parse_capacity(item.text)
                meaning = "requested_or_configured_capacity" if name.startswith("clan_") else "displayed_capacity_ratio"
                if name == "heroes":
                    meaning = "selected_hero_slots"
                values[name] = {"used": used, "capacity": capacity, "meaning": meaning,
                                "evidence": {**asdict(item), "source": source}}
        return values

    def _request_dialog_button(self, name: str, box: tuple[int, int, int, int]) -> bool:
        if name not in {"cancel", "send"}:
            return False
        width, height = self.baseline_resolution
        cx, cy = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
        if not 0.62 <= cy <= 0.78:
            return False
        return 0.3 <= cx <= 0.49 if name == "cancel" else 0.52 <= cx <= 0.71

    def _chat_navigation(self, name: str, path: Path, resolution: tuple[int, int]) -> dict[str, object] | None:
        filename, region = {
            "request_open": ("chat_request.png", (374, 640, 484, 715)),
            "chat_latest": ("chat_latest.png", (0, 530, 100, 650)),
        }[name]
        template = Path(__file__).resolve().parents[2] / "assets/templates" / filename
        if not template.is_file():
            return None
        roi = scale_box(region, from_resolution=(1280, 720), to_resolution=self.baseline_resolution)
        try:
            match = find_template(path, template, threshold=0.9, roi=roi,
                                  roi_base_resolution=self.baseline_resolution, template_base_resolution=(1280, 720))
        except LocatorError:
            return None
        if match.bounds is None:
            return None
        bounds = match.bounds
        box = scale_box((bounds.left, bounds.top, bounds.right, bounds.bottom),
                        from_resolution=resolution, to_resolution=self.baseline_resolution)
        return {"name": name, "text": {"request_open": "请求增援（图标）", "chat_latest": "回到最新消息（图标）"}[name],
                "method": "template", "semantic": "navigation_only",
                "bbox": list(box), "point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                "confidence": match.confidence, "template": str(template)}

    def _village_resources(
        self, path: Path, results: list[OCRText], source_resolution: tuple[int, int],
    ) -> tuple[dict[str, int | None], dict[str, object]]:
        width, height = self.baseline_resolution
        # These HUD bands are proportional to the screenshot, independent of village pan/zoom.
        bands = {"gold": (0.025, 0.085), "elixir": (0.12, 0.18), "dark_elixir": (0.215, 0.28), "gems": (0.305, 0.365)}
        line_rois = {
            "gold": (1008, 19, 1210, 60), "elixir": (1008, 87, 1210, 128),
            "dark_elixir": (1064, 153, 1210, 195), "gems": (1130, 220, 1210, 262),
        }
        values: dict[str, int | None] = {name: None for name in bands}
        evidence: dict[str, object] = {}
        for name, (top, bottom) in bands.items():
            candidates = []
            source = "full_ocr"
            for item in results:
                if item.bbox is None:
                    continue
                left, y1, right, y2 = item.bbox
                value = parse_resource_number(item.text)
                if value is None and name != "gems":
                    ratio = parse_capacity(item.text)
                    value = ratio[0] if ratio is not None else None
                if value is not None and left >= width * 0.78 and top * height <= (y1 + y2) / 2 <= bottom * height:
                    candidates.append((item.confidence, value, item))
            if not candidates:
                read_line = getattr(self.provider, "recognize_line", None)
                if read_line is not None:
                    source_roi = scale_box(line_rois[name], from_resolution=(1280, 720), to_resolution=source_resolution)
                    for result in filter_ocr_results(read_line(path, source_roi), max(0.9, self.ocr_config.confidence_threshold)):
                        value = parse_resource_number(result.text)
                        if value is None and name != "gems":
                            ratio = parse_capacity(result.text)
                            value = ratio[0] if ratio is not None else None
                        if value is not None:
                            candidates.append((result.confidence, value, self._baseline_result(result, source_resolution)))
                    source = "line_roi"
            if candidates:
                _, value, item = max(candidates, key=lambda candidate: candidate[0])
                values[name] = value
                evidence[name] = {**asdict(item), "source": source}
        return values, evidence

    def _village_resource_capacities(self, texts: list[OCRText]) -> dict[str, int | None]:
        """Only a current, unambiguous HUD numerator/denominator proves capacity."""
        width, height = self.baseline_resolution
        bands = {"gold": (.025, .085), "elixir": (.12, .18), "dark_elixir": (.215, .28)}
        values: dict[str, int | None] = {name: None for name in bands}
        for name, (top, bottom) in bands.items():
            candidates = [parse_capacity(item.text) for item in texts
                          if item.bbox is not None and math.isfinite(item.confidence)
                          and .9 <= item.confidence <= 1 and item.bbox[0] >= width * .78
                          and top * height <= (item.bbox[1] + item.bbox[3]) / 2 <= bottom * height
                          and parse_capacity(item.text) is not None]
            if len(candidates) == 1:
                values[name] = candidates[0][1]
        return values

    @staticmethod
    def _resolution(path: Path) -> tuple[int, int]:
        try:
            import cv2
            import numpy as np

            image = read_frame(path, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise SceneError(f"Unable to read screenshot {path}: {exc}") from exc
        if image is None:
            raise SceneError(f"Unable to decode screenshot: {path}")
        height, width = image.shape[:2]
        return width, height
