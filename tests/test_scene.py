import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from autococ.ocr import OCRText
from autococ.scene import SCENE_SETTLEMENT, SCENE_UNKNOWN, SCENE_VILLAGE, classify_scene_text, detect_scene_from_xml
from autococ.vision import ScreenshotRecognizer, detect_collectibles, parse_capacity, parse_countdown, parse_resource_number


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy>
  <node text="Attack" resource-id="" class="android.widget.Button"
        content-desc="" clickable="true" enabled="true" bounds="[10,20][110,70]" />
  <node text="Builder" resource-id="" class="android.widget.TextView"
        content-desc="" clickable="false" enabled="true" bounds="[5,100][80,140]" />
</hierarchy>
"""


class SceneTests(unittest.TestCase):
    def test_classifies_village_text(self) -> None:
        scene, confidence, reasons = classify_scene_text("gold elixir builder attack")

        self.assertEqual(scene, SCENE_VILLAGE)
        self.assertGreater(confidence, 0)
        self.assertIn("builder", reasons)

    def test_idle_disconnect_takes_priority_over_visible_home_controls(self) -> None:
        for message in ("因为太久没有进行操作，您已断开连接。", "重新载入游戏"):
            with self.subTest(message=message):
                scene, _, _ = classify_scene_text(f"进攻！ 商店 还在吗？ {message}")
                self.assertEqual(scene, "disconnected")

    def test_reload_button_requires_confident_text_in_central_dialog(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        background = [OCRText("已断开连接", .99, (660, 668, 1296, 716)),
                      OCRText("进攻！", .99, (108, 1342, 220, 1396)),
                      OCRText("商店", .99, (2354, 1338, 2438, 1394))]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            for confidence, box, expected in ((.99, (664, 816, 862, 858), ["retry"]),
                                               (.7, (664, 816, 862, 858), []),
                                               (.99, (20, 1300, 218, 1342), [])):
                with self.subTest(confidence=confidence, box=box):
                    provider.recognize.return_value = background + [OCRText("重新载入游戏", confidence, box)]
                    snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
                    self.assertEqual(snapshot.scene, "disconnected")
                    self.assertEqual([button["name"] for button in snapshot.observations["buttons"]], expected)
                    self.assertIsNone(snapshot.observations["resource_source"])
                    self.assertEqual(snapshot.observations["collectibles"], [])

    def test_recorded_idle_disconnect_frames_no_longer_expose_attack(self) -> None:
        import json

        root = Path(__file__).resolve().parents[1]
        events = root / "reports/20260924-031134-809295-96af5b0a/events.jsonl"
        if not events.is_file():
            self.skipTest("Optional live idle-disconnection fixtures are absent")
        provider = Mock()
        observations = [item for line in events.read_text(encoding="utf-8").splitlines()
                        if (item := json.loads(line))["kind"] == "observation"]
        self.assertEqual(len(observations), 5)
        for event in observations:
            with self.subTest(frame=event["frame"]):
                provider.recognize.return_value = [
                    OCRText(item["text"], item["confidence"], tuple(value * 2 for value in item["bbox"]))
                    for item in event["observations"]["ocr"]]
                snapshot = ScreenshotRecognizer(provider=provider).recognize(root / event["frame"])
                self.assertEqual(snapshot.scene, "disconnected")
                self.assertEqual([button["name"] for button in snapshot.observations["buttons"]], ["retry"])
                self.assertIsNone(snapshot.observations["resource_source"])

    def test_detects_scene_from_xml(self) -> None:
        snapshot = detect_scene_from_xml(SAMPLE_XML, screenshot_path="screen.png")

        self.assertEqual(snapshot.scene, SCENE_VILLAGE)
        self.assertEqual(snapshot.observations["node_count"], 2)

    def test_legible_battle_text_during_whiteout_does_not_enable_actions(self) -> None:
        import cv2
        import numpy as np

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "whiteout.png"
            cv2.imwrite(str(path), np.full((720, 1280, 3), 240, dtype=np.uint8))
            for timer in ("开战倒计时", "离战斗结束还有"):
                with self.subTest(timer=timer):
                    provider = Mock()
                    provider.recognize.return_value = [OCRText(timer, .99, (580, 30, 705, 50)),
                                                       OCRText("结束战斗", .99, (20, 530, 150, 560))]
                    snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
                    self.assertEqual(snapshot.scene, "unknown")
                    self.assertEqual(snapshot.observations["buttons"], [])
                    self.assertIsNone(snapshot.observations["resource_source"])

    def test_recorded_cloud_transitions_do_not_expose_battle_actions(self) -> None:
        import json

        root = Path(__file__).resolve().parents[1]
        report = root / "reports/vision-review-20260924/before.json"
        if not report.is_file():
            self.skipTest("Optional cloud-transition replay fixtures are absent")
        samples = {item["id"]: item for item in json.loads(report.read_text(encoding="utf-8"))["samples"]}
        for item in json.loads((report.parent / "sample-index.json").read_text(encoding="utf-8"))["samples"]:
            samples[item["id"]]["path"] = root / item["path"].replace("\\", "/")
        for sample_id in ("V37", "V43", "V44"):
            with self.subTest(sample=sample_id):
                sample = samples[sample_id]
                provider = Mock()
                provider.recognize.return_value = [
                    OCRText(item["text"], item["confidence"], tuple(value * 2 for value in item["bbox"]))
                    for item in sample["observations"]["ocr"]]
                snapshot = ScreenshotRecognizer(provider=provider).recognize(sample["path"])
                self.assertEqual(snapshot.scene, "unknown")
                self.assertEqual(snapshot.observations["buttons"], [])
                self.assertIsNone(snapshot.observations["resource_source"])
                self.assertIsNone(snapshot.observations["battle"])
                self.assertTrue(snapshot.observations["cloud_cover"]["obscured"])

    def test_recorded_clear_scouts_and_battles_retain_scene_and_controls(self) -> None:
        import json

        root = Path(__file__).resolve().parents[1]
        report = root / "reports/vision-review-20260924/before.json"
        if not report.is_file():
            self.skipTest("Optional clear battle replay fixtures are absent")
        samples = {item["id"]: item for item in json.loads(report.read_text(encoding="utf-8"))["samples"]}
        for item in json.loads((report.parent / "sample-index.json").read_text(encoding="utf-8"))["samples"]:
            samples[item["id"]]["path"] = root / item["path"].replace("\\", "/")
        for sample_id in ("V12", "V13", "V19", "V24", "V28"):
            with self.subTest(sample=sample_id):
                sample = samples[sample_id]
                provider = Mock()
                provider.recognize.return_value = [
                    OCRText(item["text"], item["confidence"], tuple(value * 2 for value in item["bbox"]))
                    for item in sample["observations"]["ocr"]]
                provider.recognize_line.return_value = []
                snapshot = ScreenshotRecognizer(provider=provider).recognize(sample["path"])
                self.assertEqual(snapshot.scene, sample["expected_scene"])
                self.assertFalse(snapshot.observations["cloud_cover"]["obscured"])
                controls = {button["name"] for button in snapshot.observations["buttons"]}
                self.assertIn("next" if snapshot.scene == "enemy_village" else "end_battle", controls)

    def test_settlement_takes_priority_over_village(self) -> None:
        scene, _, _ = classify_scene_text("victory return home gold")

        self.assertEqual(scene, SCENE_SETTLEMENT)

    def test_active_battle_timer_takes_priority_over_lingering_next_control(self) -> None:
        for timer in ("离战斗结束还有", "战斗结束倒计时"):
            with self.subTest(timer=timer):
                scene, _, _ = classify_scene_text(f"可获得的战利品 下一个 1600 放弃 {timer} 3分钟0秒")
                self.assertEqual(scene, "battle")

    def test_recorded_first_deployment_does_not_expose_next_opponent(self) -> None:
        import json

        root = Path(__file__).resolve().parents[1]
        directory = root / "reports/20260923-023613-726879-9389d9e7"
        events = directory / "events.jsonl"
        frame = directory / "frames/00011-deploy-consumption.png"
        if not events.is_file() or not frame.is_file():
            self.skipTest("Optional first-deployment fixture is absent")
        recorded = next(item for line in events.read_text(encoding="utf-8").splitlines()
                        if (item := json.loads(line)).get("kind") == "observation"
                        and item["frame"].replace("\\", "/").endswith("/00011-deploy-consumption.png"))
        provider = Mock()
        provider.recognize.return_value = [
            OCRText(item["text"], item["confidence"], tuple(value * 2 for value in item["bbox"]))
            for item in recorded["observations"]["ocr"]]
        provider.recognize_line.return_value = []
        snapshot = ScreenshotRecognizer(provider=provider).recognize(frame)
        self.assertEqual(snapshot.scene, "battle")
        self.assertEqual(snapshot.observations["resource_source"], "enemy_remaining")
        self.assertNotIn("next", {button["name"] for button in snapshot.observations["buttons"]})
        self.assertIn("end_battle", {button["name"] for button in snapshot.observations["buttons"]})

    def test_resource_id_is_not_scene_evidence(self) -> None:
        snapshot = detect_scene_from_xml(
            '<hierarchy><node resource-id="com.supercell.clashofclans:id/stage" /></hierarchy>',
            screenshot_path="screen.png",
        )
        self.assertEqual(snapshot.scene, SCENE_UNKNOWN)

    def test_recorded_bright_and_dim_supercell_logos_are_nonactionable_startup(self) -> None:
        import json

        root = Path(__file__).resolve().parents[1]
        index = root / "reports/vision-review-20260924/sample-index.json"
        if not index.is_file():
            self.skipTest("Optional startup-logo fixtures are absent")
        paths = {row["id"]: root / row["path"].replace("\\", "/")
                 for row in json.loads(index.read_text(encoding="utf-8"))["samples"]}
        provider = Mock()
        provider.recognize.return_value = []
        for sample_id in ("V03", "V17", "V32"):
            with self.subTest(sample=sample_id):
                snapshot = ScreenshotRecognizer(provider=provider).recognize(paths[sample_id])
                self.assertEqual(snapshot.scene, "starting")
                self.assertLess(snapshot.confidence, .8)
                self.assertEqual(snapshot.observations["buttons"], [])
                self.assertIsNone(snapshot.observations["resource_source"])
                self.assertIsNotNone(snapshot.observations["startup_logo"])

    def test_black_screen_partial_or_displaced_logo_and_nonblack_background_remain_unknown(self) -> None:
        import cv2
        import numpy as np

        root = Path(__file__).resolve().parents[1]
        template = cv2.imdecode(np.fromfile(root / "assets/templates/startup_supercell.png", dtype=np.uint8),
                                cv2.IMREAD_COLOR)
        height, width = template.shape[:2]
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        displaced = black.copy()
        displaced[40:40+height, 40:40+width] = template
        partial = black.copy()
        partial[230:230+height, 485:485+width//2] = template[:, :width//2]
        other_background = np.full_like(black, 35)
        other_background[230:230+height, 485:485+width] = template
        rectangle = black.copy()
        rectangle[250:460, 510:770] = 255
        provider = Mock()
        provider.recognize.return_value = []
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-startup.png"
            for name, image in (("black", black), ("displaced", displaced), ("partial", partial),
                                ("background", other_background), ("rectangle", rectangle)):
                with self.subTest(case=name):
                    cv2.imwrite(str(path), image)
                    snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
                    self.assertEqual(snapshot.scene, "unknown")
                    self.assertIsNone(snapshot.observations["startup_logo"])
                    self.assertEqual(snapshot.observations["buttons"], [])

    def test_chinese_scenes(self) -> None:
        cases = {
            "进攻！ 商店": "village",
            "进攻！ 商店 请求增援": "village",
            "商店": "unknown",
            "战斗开始倒计时 下一个 可掠夺战利品": "enemy_village",
            "战斗开始倒计时 下一个 结束战斗": "enemy_village",
            "战斗结束倒计时 结束战斗": "battle",
            "离战斗结束还有 2分钟44秒 放弃 可获得的战利品": "battle",
            "可掠夺战利品 战斗结束倒计时 结束战斗": "battle",
            "available loot end battle": "battle",
            "available loot": "unknown",
            "enemy village": "unknown",
            "可获得战利品 下一个": "enemy_village",
            "开战倒计时 28秒 可获得的战利品 下一个 结束战斗": "enemy_village",
            "胜利！ 返回村庄": "settlement",
            "失败 摧毁率2% 您得到了 回营": "settlement",
            "正在载入": "starting",
            "返回村庄": "unknown",
            "loot gained": "unknown",
            "请求失败": "unknown",
            "搜索对手 联机模式": "search",
            "请求增援": "unknown",
            "请求增援 友谊战 部落消息 商店": "clan_chat",
            "请求增援 取消 发送": "request",
            "捐赠部队": "donation",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify_scene_text(text)[0], expected)

    def test_screenshot_observations_are_scaled_and_resource_backed(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("进攻！", 0.99, (100, 1340, 220, 1400)),
            OCRText("商店", 0.99, (2350, 1340, 2440, 1400)),
            OCRText("10 435 275", 0.99, (2160, 52, 2410, 110)),
            OCRText("12 276 804", 0.99, (2160, 189, 2410, 242)),
            OCRText("102-559", 0.94, (2236, 320, 2416, 375)),
            OCRText("465", 0.99, (2314, 451, 2417, 504)),
            OCRText("请求增援", 0.99, (1000, 200, 1130, 245)),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(snapshot.scene, SCENE_VILLAGE)
        self.assertEqual(snapshot.observations["buttons"][0]["point"], [80, 685])
        self.assertEqual([button["name"] for button in snapshot.observations["buttons"]], ["attack", "shop"])
        self.assertEqual(snapshot.observations["resources"], {
            "gold": 10435275, "elixir": 12276804, "dark_elixir": 102559, "gems": 465,
        })
        self.assertEqual(snapshot.observations["resource_source"], "village_inventory")

    def test_timers_and_counters_are_not_resource_amounts(self) -> None:
        for text in ("0/5", "4小时41分", "-100", "100 gold"):
            self.assertIsNone(parse_resource_number(text))

    def test_army_headers_use_roi_fallback_and_preserve_unknown(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("我的军队", 0.99, (440, 120, 590, 180)),
            OCRText("330/330", 0.99, (1156, 316, 1324, 360)),
            OCRText("50/50", 0.99, (1178, 1046, 1300, 1092)),
            OCRText("3/3", 0.99, (1420, 1040, 1508, 1096)),
            OCRText("2/2", 0.99, (1598, 1044, 1682, 1092)),
        ]
        def line_result(path, roi):
            if roi == (1154, 650, 1276, 724):
                return [OCRText("11/11", 0.95, roi)]
            return []
        provider.recognize_line.side_effect = line_result
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        army = snapshot.observations["army"]
        self.assertEqual(army["troops"]["used"], 330)
        self.assertEqual(army["spells"]["used"], 11)
        self.assertEqual(army["spells"]["evidence"]["source"], "line_roi")
        self.assertEqual(army["clan_troops"]["capacity"], 50)
        self.assertEqual(army["clan_spells"]["capacity"], 3)
        self.assertEqual(army["clan_siege"]["capacity"], 2)
        self.assertIsNone(army["heroes"])
        self.assertIsNone(army["siege"])
        self.assertTrue(army["clan_received_unknown"])
        self.assertIsNone(army["heroes_available"])
        self.assertEqual(army["clan_troops"]["meaning"], "requested_or_configured_capacity")

    def test_request_modal_excludes_title_and_background_buttons(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("请求增援", 0.99, (1160, 135, 1400, 200)),
            OCRText("请求", 0.99, (80, 675, 150, 710)),
            OCRText("商店", 0.99, (2350, 1340, 2440, 1400)),
            OCRText("取消", 0.99, (934, 960, 1052, 1040)),
            OCRText("发送", 0.99, (1504, 964, 1624, 1044)),
            OCRText("发送", 0.99, (10, 1250, 100, 1300)),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            screen = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.rectangle(screen, (328, 40), (953, 571), (225, 234, 234), -1)
            cv2.rectangle(screen, (370, 305), (912, 428), (255, 255, 255), -1)
            cv2.rectangle(screen, (370, 305), (912, 428), (0, 0, 0), 3)
            cv2.imwrite(str(path), cv2.resize(screen, (2560, 1440), interpolation=cv2.INTER_NEAREST))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(snapshot.scene, "request")
        self.assertEqual([button["name"] for button in snapshot.observations["buttons"]], ["cancel", "send"])

    def test_capacity_rejects_ambiguous_or_impossible_counts(self) -> None:
        for text in ("11/1", "11/11 gold", "IL/", "3 3", "x10"):
            self.assertIsNone(parse_capacity(text))
        self.assertEqual(parse_capacity("0/50"), (0, 50))

    def test_request_fade_in_cannot_expose_send_even_when_ocr_geometry_matches(self) -> None:
        root = Path(__file__).resolve().parents[1]
        fixtures = (
            ("reports/20260923-012518-864019-0bcf5449/frames/00012-request-dialog.png", "unknown"),
            ("reports/20260923-005254-798133-183c43c2/frames/00026-request-dialog.png", "request"),
            ("reports/live-20260922/request-send-probe/frames/00001-before-send.png", "request"),
        )
        provider = Mock()
        provider.recognize.return_value = [
            OCRText("请求增援", .99, (1160, 135, 1400, 200)),
            OCRText("取消", .99, (934, 960, 1052, 1040)),
            OCRText("发送", .99, (1504, 964, 1624, 1044)),
            OCRText("救命啊！我需要增援！", .99, (760, 626, 1260, 675)),
            OCRText("副首领", .99, (828, 696, 930, 736)),
        ]
        for relative, expected in fixtures:
            with self.subTest(fixture=relative):
                path = root / relative
                if not path.is_file():
                    self.skipTest("Optional request-animation fixture is absent")
                snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
                self.assertEqual(snapshot.scene, expected)
                self.assertEqual(snapshot.observations["request_dialog"]["ready"], expected == "request")
                if expected == "unknown":
                    self.assertEqual(snapshot.observations["buttons"], [])
                    self.assertIn("request_dialog_opacity_unverified", snapshot.observations["matched_reasons"])

    def test_opaque_request_allows_multiline_body_without_truncating_ocr(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("请求增援", .99, (1160, 135, 1400, 200)),
            OCRText("取消", .99, (934, 960, 1052, 1040)),
            OCRText("发送", .99, (1504, 964, 1624, 1044)),
            OCRText("第一行请求", .99, (760, 626, 1200, 676)),
            OCRText("第二行请求", .99, (760, 702, 1200, 752)),
            OCRText("第三行请求", .99, (760, 778, 1200, 828)),
        ]
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (328, 40), (953, 571), (225, 234, 234), -1)
        cv2.rectangle(image, (370, 305), (912, 428), (255, 255, 255), -1)
        cv2.rectangle(image, (370, 305), (912, 428), (0, 0, 0), 3)
        for y in (334, 372, 410):
            cv2.putText(image, "Multiple request lines 123456", (380, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 45, 45), 2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.png"
            cv2.imwrite(str(path), cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_NEAREST))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
            self.assertEqual(snapshot.scene, "request")
            self.assertEqual([item["text"] for item in snapshot.observations["ocr"][-3:]], ["第一行请求", "第二行请求", "第三行请求"])
            # A white body with a missing modal border is still not actionable.
            image[300:311, 380:906] = (255, 255, 255)
            cv2.imwrite(str(path), cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_NEAREST))
            incomplete = ScreenshotRecognizer(provider=provider).recognize(path)
            self.assertEqual(incomplete.scene, "unknown")
            self.assertEqual(incomplete.observations["buttons"], [])

    def test_inventory_invalid_text_uses_fresh_line_ocr_without_guessing_letters(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("进攻！", 0.99, (100, 1340, 220, 1400)),
            OCRText("商店", 0.99, (2350, 1340, 2440, 1400)),
            OCRText("13583388", 0.99, (2160, 52, 2410, 110)),
            OCRText("S956SE", 0.799, (2228, 188, 2416, 244)),
            OCRText("137559", 0.99, (2236, 320, 2416, 375)),
            OCRText("323", 0.99, (2314, 451, 2417, 504)),
        ]
        provider.recognize_line.return_value = [OCRText("359565", 0.995, (2016, 174, 2420, 256))]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
            self.assertEqual(snapshot.observations["resources"]["elixir"], 359565)
            self.assertEqual(snapshot.observations["resource_evidence"]["elixir"]["source"], "line_roi")
            provider.recognize_line.return_value = [OCRText("S956SE", 0.99, (2016, 174, 2420, 256))]
            unknown = ScreenshotRecognizer(provider=provider).recognize(path)
            self.assertIsNone(unknown.observations["resources"]["elixir"])

    def test_regular_search_is_distinguished_from_ranked_and_background(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("联机模式", 0.99, (268, 92, 420, 144)),
            OCRText("常规战", 0.99, (348, 892, 516, 966)),
            OCRText("搜索对手", 0.99, (350, 1004, 522, 1068)),
            OCRText("搜索对手", 0.99, (1056, 1016, 1224, 1076)),
            OCRText("1600", 0.99, (312, 1064, 468, 1134)),
            OCRText("5/12", 0.99, (1072, 1076, 1212, 1146)),
            OCRText("商店", 0.99, (2352, 1336, 2438, 1392)),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(snapshot.scene, "search")
        self.assertEqual(snapshot.observations["mode"], "regular")
        self.assertEqual(snapshot.observations["search_cost_gold"], 1600)
        self.assertEqual([button["name"] for button in snapshot.observations["buttons"]], ["find_match_regular"])
        self.assertEqual(snapshot.observations["buttons"][0]["point"], [218, 518])

    def test_scout_resources_and_card_counts_are_separate_from_inventory_and_levels(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/scout-from-army-235605/frames/00007-scout-from-army.png"
        if not path.is_file():
            self.skipTest("Optional live scout fixture is absent")
        provider = Mock()
        values = [
            ("开战倒计时", (580, 12, 680, 40)), ("28秒", (592, 36, 682, 78)),
            ("306 567", (62, 98, 157, 124)), ("563 945", (60, 134, 159, 164)),
            ("915", (61, 174, 104, 202)), ("13 581788", (1124, 24, 1226, 52)),
            ("x10", (130, 592, 180, 622)), ("x1", (244, 594, 276, 621)),
            ("x2", (334, 592, 374, 621)), ("x1", (436, 592, 470, 622)),
            ("x6", (1044, 592, 1086, 622)), ("x5", (1142, 592, 1184, 622)),
            ("80", (892, 678, 922, 700)),
        ]
        provider.recognize.return_value = [OCRText(text, .99, tuple(v * 2 for v in box)) for text, box in values]
        snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(snapshot.scene, "enemy_village")
        self.assertEqual(snapshot.observations["resource_source"], "enemy_available")
        self.assertEqual(snapshot.observations["resources"]["gold"], 306567)
        self.assertEqual(snapshot.observations["resources"]["elixir"], 563945)
        battle = snapshot.observations["battle"]
        self.assertEqual(battle["countdown_seconds"], 28)
        self.assertEqual([slot["count"] for slot in battle["slots"] if slot["kind"] == "troop"], [10, 1, 2, 1])
        self.assertEqual([slot["count"] for slot in battle["slots"] if slot["kind"] == "spell"], [6, 5])
        self.assertEqual([slot["count"] for slot in battle["slots"] if slot["kind"] == "hero"], [None] * 4)

    def test_countdown_parsing_rejects_inventory_and_capacity(self) -> None:
        self.assertEqual(parse_countdown("2分59秒"), 179)
        self.assertEqual(parse_countdown("2:59"), 179)
        for text in ("330/330", "20分钟后", "359565", "1:99"):
            self.assertIsNone(parse_countdown(text))

    def test_disconnect_exposes_only_dialog_retry_not_background_actions(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("连接中断", .99, (660, 588, 852, 646)),
            OCRText("重试", .99, (670, 812, 750, 860)),
            OCRText("下一个", .99, (2304, 964, 2460, 1036)),
            OCRText("结束战斗", .99, (120, 1056, 248, 1100)),
            OCRText("开战倒计时", .99, (1160, 24, 1360, 80)),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
            self.assertEqual(snapshot.scene, "disconnected")
            self.assertEqual([button["name"] for button in snapshot.observations["buttons"]], ["retry"])
            self.assertEqual(snapshot.observations["buttons"][0]["point"], [355, 418])
            provider.recognize.return_value[0] = OCRText("服务器维护", .99, (660, 588, 852, 646))
            maintenance = ScreenshotRecognizer(provider=provider).recognize(path)
            self.assertEqual(maintenance.observations["buttons"], [])

    def test_background_request_label_does_not_override_clan_chat(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("请求增援", .99, (980, 240, 1112, 290)),
            OCRText("友谊战", .99, (572, 1344, 670, 1390)),
            OCRText("部落消息", .99, (48, 1340, 222, 1392)),
            OCRText("商店", .99, (2354, 1336, 2438, 1392)),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(snapshot.scene, "clan_chat")
        self.assertNotIn("request", [button["name"] for button in snapshot.observations["buttons"]])

    def test_sliding_chat_panel_does_not_enable_navigation_early(self) -> None:
        import cv2
        import numpy as np

        provider = Mock()
        provider.recognize.return_value = [
            OCRText("友谊战", .99, (480, 1344, 580, 1390)),
            OCRText("捐赠", .99, (760, 680, 850, 730)),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), dtype=np.uint8))
            snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(snapshot.scene, "unknown")
        self.assertEqual(snapshot.observations["buttons"], [])
        self.assertIn("clan_chat_panel_position_unverified", snapshot.observations["matched_reasons"])

    def test_observed_settlement_has_loot_zero_stars_and_layout_backed_zero_bonus(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/settlement-010735/frames/00008-end-confirmation.png"
        if not path.is_file():
            self.skipTest("Optional real settlement fixture is absent")
        values = [
            ("失败", (608, 196, 671, 236)), ("2%", (618, 102, 663, 132)),
            ("您得到了：", (599, 286, 680, 312)), ("67585", (562, 314, 670, 350)),
            ("95858", (558, 361, 670, 397)), ("1405", (581, 406, 672, 446)),
            ("损耗的部队", (590, 462, 676, 485)), ("回营", (623, 608, 660, 631)),
            ("13782480", (1119, 24, 1226, 50)), ("商店", (1176, 668, 1219, 696)),
        ]
        provider = Mock()
        provider.recognize.return_value = [OCRText(text, .99, tuple(v * 2 for v in box)) for text, box in values]
        snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(snapshot.scene, "settlement")
        self.assertEqual(snapshot.observations["resource_source"], "settlement_gained")
        self.assertEqual(snapshot.observations["settlement"]["loot"], {"gold": 67585, "elixir": 95858, "dark_elixir": 1405})
        self.assertEqual(snapshot.observations["settlement"]["bonus"], {"gold": 0, "elixir": 0, "dark_elixir": 0})
        self.assertEqual(snapshot.observations["settlement"]["percentage"], 2)
        self.assertEqual(snapshot.observations["settlement"]["stars"], 0)
        self.assertEqual(snapshot.observations["settlement"]["evidence"]["bonus"]["source"], "verified_no_bonus_layout")
        self.assertEqual([item["name"] for item in snapshot.observations["buttons"]], ["return_home"])
        self.assertEqual(snapshot.observations["buttons"][0]["point"], [641, 619])
        import cv2
        import numpy as np

        with TemporaryDirectory() as tmp:
            no_layout = Path(tmp) / "no-layout.png"
            cv2.imwrite(str(no_layout), np.zeros((1440, 2560, 3), dtype=np.uint8))
            absent_separators = ScreenshotRecognizer(provider=provider).recognize(no_layout)
        self.assertEqual(absent_separators.observations["settlement"]["bonus"], {"gold": None, "elixir": None, "dark_elixir": None})
        # A bonus label means another layout; absent readable bonus values stay unknown.
        provider.recognize.return_value.append(OCRText("额外奖励", .99, (1400, 650, 1570, 700)))
        with_bonus = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(with_bonus.observations["settlement"]["bonus"], {"gold": None, "elixir": None, "dark_elixir": None})
        provider.recognize.return_value = provider.recognize.return_value[:-1]
        provider.recognize.return_value[0] = OCRText("胜利", .99, (1216, 392, 1342, 472))
        victory = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(victory.scene, "settlement")
        self.assertIsNone(victory.observations["settlement"]["stars"])
        self.assertEqual(victory.observations["settlement"]["bonus"], {"gold": None, "elixir": None, "dark_elixir": None})
        provider.recognize.return_value = provider.recognize.return_value[1:]
        unknown_result = ScreenshotRecognizer(provider=provider).recognize(path)
        # Neither unknown result nor victory is silently assigned zero stars/bonus.
        self.assertEqual(unknown_result.scene, "unknown")

    def test_settlement_dark_icon_ignores_map_background_but_rejects_duplicates(self) -> None:
        import json
        import cv2
        import numpy as np

        root = Path(__file__).resolve().parents[1]
        run = root / "reports/20260924-001324-488783-7b354746"
        path = run / "frames/00048-battle-settlement-reread.png"
        if not path.is_file():
            self.skipTest("Optional varied-background settlement fixture is absent")
        events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observed = next(item for item in events if item["kind"] == "observation" and Path(item["frame"]).name == path.name)
        provider = Mock()
        provider.recognize.return_value = [OCRText(item["text"], item["confidence"], tuple(v * 2 for v in item["bbox"]))
                                           for item in observed["observations"]["ocr"]]
        provider.recognize_line.return_value = []
        result = ScreenshotRecognizer(provider=provider).recognize(path).observations["settlement"]
        self.assertEqual(result["loot"], {"gold": 75774, "elixir": 41131, "dark_elixir": 838})
        self.assertEqual(result["bonus"], {"gold": 0, "elixir": 0, "dark_elixir": 0})
        self.assertEqual(result["evidence"]["layout"], "regular_defeat_three_rows_v1")
        icon = result["evidence"]["resource_icons"]["dark_elixir"]
        self.assertGreaterEqual(icon["confidence"], .99)
        self.assertEqual(icon["comparison"], "opaque_drop_body")
        # The same icon in two separated rows cannot identify one loot amount.
        image = cv2.resize(cv2.imread(str(path)), (1280, 720), interpolation=cv2.INTER_AREA)
        template = cv2.imread(str(root / "assets/templates/settlement_dark_elixir.png"))
        image[310:355, 674:715] = template
        with TemporaryDirectory() as tmp:
            duplicate = Path(tmp) / "duplicate.png"
            cv2.imwrite(str(duplicate), cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_NEAREST))
            ambiguous = ScreenshotRecognizer(provider=provider).recognize(duplicate).observations["settlement"]
            self.assertIsNone(ambiguous["loot"]["dark_elixir"])
            self.assertIsNone(ambiguous["evidence"]["layout"])
            blank = Path(tmp) / "blank.png"
            cv2.imwrite(str(blank), np.zeros((1440, 2560, 3), dtype=np.uint8))
            empty = ScreenshotRecognizer(provider=provider).recognize(blank).observations["settlement"]
            self.assertTrue(all(value is None for value in empty["loot"].values()))

    def test_chat_latest_requires_visible_template_and_confirmed_chat(self) -> None:
        import cv2
        import numpy as np

        path = Path(__file__).resolve().parents[1] / "reports/20260923-011419-927154-a8f7a5a8/frames/00012-request-verify.png"
        template = Path(__file__).resolve().parents[1] / "assets/templates/chat_latest.png"
        if not path.is_file() or not template.is_file():
            self.skipTest("Optional chat-latest fixture/template is absent")
        provider = Mock()
        provider.recognize.return_value = [OCRText("友谊战", .99, (572, 1344, 670, 1390))]
        snapshot = ScreenshotRecognizer(provider=provider).recognize(path)
        buttons = [item for item in snapshot.observations["buttons"] if item["name"] == "chat_latest"]
        self.assertEqual(len(buttons), 1)
        self.assertEqual(buttons[0]["text"], "回到最新消息（图标）")
        self.assertGreaterEqual(buttons[0]["confidence"], .9)
        self.assertTrue(0 <= buttons[0]["point"][0] <= 100 and 530 <= buttons[0]["point"][1] <= 650)
        with TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.png"
            cv2.imwrite(str(empty), np.zeros((1440, 2560, 3), dtype=np.uint8))
            no_arrow = ScreenshotRecognizer(provider=provider).recognize(empty)
        self.assertNotIn("chat_latest", [item["name"] for item in no_arrow.observations["buttons"]])
        provider.recognize.return_value = []
        unknown_scene = ScreenshotRecognizer(provider=provider).recognize(path)
        self.assertEqual(unknown_scene.observations["buttons"], [])

    def test_two_row_settlement_anchors_resources_to_icons_and_zero_requires_complete_layout(self) -> None:
        import cv2

        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/battle-current-013325/frames/00001-before.png"
        if not path.is_file():
            self.skipTest("Optional two-row settlement fixture is absent")
        values = [
            ("失败", (608, 196, 671, 236)), ("30%", (610, 102, 672, 132)),
            ("您得到了", (600, 304, 670, 327)), ("653057", (540, 328, 670, 373)),
            ("396 241", (548, 387, 672, 426)), ("损耗的部队：", (592, 464, 686, 485)),
            ("回营", (623, 608, 660, 631)), ("141420", (1152, 110, 1226, 132)),
        ]
        provider = Mock()
        provider.recognize.return_value = [OCRText(text, .99, tuple(v * 2 for v in box)) for text, box in values]
        provider.recognize_line.return_value = []
        result = ScreenshotRecognizer(provider=provider).recognize(path).observations["settlement"]
        self.assertEqual(result["loot"], {"gold": 653057, "elixir": 396241, "dark_elixir": 0})
        self.assertEqual(result["percentage"], 30)
        self.assertEqual(result["stars"], 0)
        self.assertTrue(result["evidence"]["loot"]["dark_elixir"]["requires_inventory_reconciliation"])
        self.assertEqual(result["evidence"]["layout"], "regular_defeat_gold_elixir_only_v1")
        self.assertEqual(result["evidence"]["loot"]["elixir"]["resource_icon"]["bbox_at_1280x720"][1], 385)
        # Missing digits on an otherwise visible resource row are unknown, not zero.
        saved = provider.recognize.return_value
        provider.recognize.return_value = [item for item in saved if item.text != "653057"]
        unreadable = ScreenshotRecognizer(provider=provider).recognize(path).observations["settlement"]
        self.assertIsNone(unreadable["loot"]["gold"])
        self.assertIsNone(unreadable["loot"]["dark_elixir"])
        self.assertTrue(all(value is None for value in unreadable["bonus"].values()))
        provider.recognize.return_value = saved
        with TemporaryDirectory() as tmp:
            missing_line = Path(tmp) / "missing-line.png"
            image = cv2.imread(str(path))
            image[854:872, 680:1310] = 0
            cv2.imwrite(str(missing_line), image)
            incomplete = ScreenshotRecognizer(provider=provider).recognize(missing_line).observations["settlement"]
        self.assertEqual(incomplete["loot"]["elixir"], 396241)
        self.assertIsNone(incomplete["loot"]["dark_elixir"])
        self.assertIsNone(incomplete["evidence"]["layout"])

    def test_missing_icon_does_not_relabel_a_number_from_its_vertical_position(self) -> None:
        import cv2

        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/settlement-010735/frames/00008-end-confirmation.png"
        if not path.is_file():
            self.skipTest("Optional three-row settlement fixture is absent")
        values = [
            ("失败", (608, 196, 671, 236)), ("2%", (618, 102, 663, 132)),
            ("您得到了：", (599, 286, 680, 312)), ("67585", (562, 314, 670, 350)),
            ("95858", (558, 361, 670, 397)), ("1405", (581, 406, 672, 446)),
            ("损耗的部队", (590, 462, 676, 485)), ("回营", (623, 608, 660, 631)),
        ]
        provider = Mock()
        provider.recognize.return_value = [OCRText(text, .99, tuple(v * 2 for v in box)) for text, box in values]
        provider.recognize_line.return_value = []
        with TemporaryDirectory() as tmp:
            missing_icon = Path(tmp) / "missing-icon.png"
            image = cv2.imread(str(path))
            image[808:904, 1344:1434] = 0
            cv2.imwrite(str(missing_icon), image)
            result = ScreenshotRecognizer(provider=provider).recognize(missing_icon).observations["settlement"]
        self.assertEqual(result["loot"]["elixir"], 95858)
        self.assertIsNone(result["loot"]["dark_elixir"])
        self.assertIsNone(result["evidence"]["layout"])


class CollectionVisionTests(unittest.TestCase):
    def test_collectibles_scale_and_exclude_hud_and_building_menu(self) -> None:
        import cv2
        import numpy as np

        root = Path(__file__).resolve().parents[1]
        template = cv2.imread(str(root / "assets/templates/collect_gold.png"))
        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        for left, top, scale in ((400, 200, 1.0), (600, 300, 1.2), (1100, 100, 1.0), (400, 580, 1.0)):
            item = cv2.resize(template, (round(40 * scale), round(36 * scale)))
            screen[top:top + item.shape[0], left:left + item.shape[1]] = item
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.png"
            # A two-times screenshot must produce the same baseline positions.
            cv2.imwrite(str(path), cv2.resize(screen, (2560, 1440), interpolation=cv2.INTER_NEAREST))
            matches = detect_collectibles(path)
        self.assertEqual(len(matches), 2)
        self.assertEqual([item["resource"] for item in matches], ["gold", "gold"])
        self.assertEqual([item["point"] for item in matches], [[420, 218], [624, 321]])

    def test_observed_empty_village_has_no_collectible_false_positives(self) -> None:
        path = Path(__file__).resolve().parents[1] / "screenshots/current.png"
        if not path.is_file():
            self.skipTest("Optional original village fixture is absent")
        self.assertEqual(detect_collectibles(path), [])

    def test_observed_resource_bubbles_include_all_three_resources(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/request-open.png"
        if not path.is_file():
            self.skipTest("Optional live village fixture is absent")
        matches = detect_collectibles(path)
        self.assertEqual(len(matches), 11)
        self.assertEqual({item["resource"] for item in matches}, {"gold", "elixir", "dark_elixir"})


if __name__ == "__main__":
    unittest.main()
