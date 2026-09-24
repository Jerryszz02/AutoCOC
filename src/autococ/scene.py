"""Scene snapshot and lightweight scene classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from .errors import SceneError
from .locator import RecognitionResult, XMLLocator


SCENE_STARTING = "starting"
SCENE_VILLAGE = "village"
SCENE_TRAINING = "training"
SCENE_SEARCH = "search"
SCENE_ENEMY_VILLAGE = "enemy_village"
SCENE_BATTLE = "battle"
SCENE_SETTLEMENT = "settlement"
SCENE_DISCONNECTED = "disconnected"
SCENE_MAINTENANCE = "maintenance"
SCENE_POPUP = "popup"
SCENE_CLAN_CHAT = "clan_chat"
SCENE_DONATION = "donation"
SCENE_REQUEST = "request"
SCENE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class SceneSnapshot:
    scene: str
    confidence: float
    screenshot_path: Path
    observations: dict[str, object] = field(default_factory=dict)


def detect_scene_from_xml(
    xml_text: str,
    *,
    screenshot_path: str | Path,
    recognition_results: list[RecognitionResult] | None = None,
) -> SceneSnapshot:
    try:
        locator = XMLLocator.from_string(xml_text)
    except Exception as exc:
        raise SceneError(f"Unable to analyze scene XML: {exc}") from exc

    tokens: list[str] = []
    for node in locator.nodes:
        tokens.extend([node.text, node.content_desc])
    haystack = " ".join(token.lower() for token in tokens if token)
    scene, confidence, reasons = classify_scene_text(haystack)
    observations: dict[str, object] = {
        "node_count": len(locator.nodes),
        "matched_reasons": reasons,
    }
    if recognition_results:
        observations["recognition_results"] = recognition_results
    return SceneSnapshot(
        scene=scene,
        confidence=confidence,
        screenshot_path=Path(screenshot_path),
        observations=observations,
    )


def classify_scene_text(text: str) -> tuple[str, float, list[str]]:
    normalized = re.sub(r"\s+", "", text.lower())
    village_keywords = ("builder", "shop", "attack", "商店", "进攻", "建筑工人")
    village_reasons = [keyword for keyword in village_keywords if keyword in normalized]
    return_controls = ("returnhome", "返回村庄", "回到村庄", "回营")
    result_markers = ("victory", "defeat", "lootgained", "totaldamage", "胜利", "失败", "摧毁")
    scout_markers = ("战斗开始倒计时", "开战倒计时", "battlestartsin")
    battle_timer_markers = ("战斗结束倒计时", "离战斗结束还有")
    next_controls = ("下一个", "下个对手", "next")
    loot_markers = ("availableloot", "可掠夺战利品", "可获得战利品", "可获得的战利品")
    checks: list[tuple[str, float, tuple[str, ...]]] = [
        (SCENE_DISCONNECTED, 0.95, ("connection lost", "reconnect", "连接中断", "连接丢失", "重新连接", "连接错误", "已断开连接", "重新载入游戏", "另一台设备", "有人正在")),
        (SCENE_MAINTENANCE, 0.95, ("maintenance", "server break", "维护中", "服务器维护")),
        (SCENE_SETTLEMENT, 0.95, ("return home", "loot gained", "返回村庄", "回到村庄", "回营")),
        (SCENE_ENEMY_VILLAGE, 0.92, ("下一个", "下个对手", "战斗开始倒计时", "开战倒计时", "battle starts in", "available loot")),
        (SCENE_BATTLE, 0.92, ("end battle", "surrender", "结束战斗", "投降", "战斗结束倒计时", "离战斗结束还有", "放弃")),
        (SCENE_SEARCH, 0.9, ("find a match", "searching", "multiplayer", "next opponent", "搜索对手", "寻找对手", "联机模式", "正在搜索")),
        (SCENE_DONATION, 0.9, ("donate troops", "捐赠部队", "捐赠法术", "捐赠攻城机器")),
        (SCENE_REQUEST, 0.9, ("request reinforcements", "请求增援", "请求援军", "请求部队")),
        (SCENE_TRAINING, 0.88, ("train troops", "train army", "训练部队", "军队配置", "编辑军队", "我的军队", "军队配方")),
        (SCENE_CLAN_CHAT, 0.88, ("clan chat", "部落聊天", "友谊战", "捐赠", "请求已发送")),
        (SCENE_POPUP, 0.8, ("confirm", "okay", "close", "确定", "确认", "取消", "稍后", "知道了")),
        (SCENE_STARTING, 0.65, ("loading", "logging in", "载入中", "正在载入", "正在加载", "正在登录")),
    ]
    for scene, confidence, keywords in checks:
        if scene == SCENE_SETTLEMENT and not (
            any(word in normalized for word in return_controls)
            and any(word in normalized for word in result_markers)
        ):
            continue
        if scene == SCENE_ENEMY_VILLAGE and any(word in normalized for word in battle_timer_markers):
            # The Next control can remain visible for the first deployed frame.
            continue
        if scene == SCENE_ENEMY_VILLAGE and not (
            any(word in normalized for word in scout_markers)
            or (any(word in normalized for word in next_controls)
                and any(word in normalized for word in loot_markers))
        ):
            continue
        # A clan castle can display "request reinforcements" above it in the village.
        # That label alone must not override both home HUD controls.
        if scene == SCENE_REQUEST:
            has_cancel = any(word in normalized for word in ("取消", "cancel"))
            has_send = any(word in normalized for word in ("发送", "send"))
            if not (has_cancel and has_send):
                continue
        reasons = [keyword for keyword in keywords if keyword.replace(" ", "") in normalized]
        if reasons:
            return scene, confidence, reasons
    if len(village_reasons) >= 2:
        return SCENE_VILLAGE, 0.9, village_reasons
    return SCENE_UNKNOWN, 0.0, []
