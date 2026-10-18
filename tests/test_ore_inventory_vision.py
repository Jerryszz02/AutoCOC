"""Inventory page recognition must expose only its verified close control."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures" / "hero_equipment_18_647_2"


def readings(name="first"):
    return [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
            for item in json.loads((FIXTURES / f"{name}.ocr.json").read_text(encoding="utf8"))]


def recognize(texts, *, name="first", version="18.647.2"):
    provider = Mock()
    provider.recognize.return_value = texts
    provider.recognize_line.return_value = []
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = version
    return recognizer.recognize(FIXTURES / f"{name}.png", battle_details=False)


class OreInventoryVisionTests(unittest.TestCase):
    def test_both_actual_page_fixtures_expose_balances_and_only_close(self):
        for name in ("first", "second"):
            with self.subTest(name=name):
                result = recognize(readings(name), name=name)
                self.assertEqual(result.scene, "popup")
                self.assertEqual(result.observations["ore_inventory"]["balances"],
                                 {"shiny": 15662, "glowy": 1225, "starry": 77})
                self.assertEqual(result.observations["ore_inventory"]["capacities"],
                                 {"shiny": 50000, "glowy": 5000, "starry": 1000})
                self.assertEqual([button["name"] for button in result.observations["buttons"]],
                                 ["close_ore_inventory"])
                self.assertTrue(all(value is None for value in result.observations["resources"].values()))
                self.assertIsNone(result.observations["resource_source"])

    def test_missing_page_or_balance_evidence_does_not_expose_inventory_close(self):
        for omitted in ("英雄装备", "拖动图标为英雄配置装备", "X", "15662/50000", "1225/5000", "77/1000"):
            with self.subTest(omitted=omitted):
                result = recognize([item for item in readings() if item.text != omitted])
                self.assertIsNone(result.observations["ore_inventory"])
                self.assertNotIn("close_ore_inventory", [b["name"] for b in result.observations["buttons"]])

    def test_other_client_versions_never_use_the_sampled_ore_page(self):
        for version in (None, "18.647.1", "18.647.3"):
            with self.subTest(version=version):
                result = recognize(readings(), version=version)
                self.assertIsNone(result.observations["ore_inventory"])

    def test_account_ownership_dialog_takes_priority_over_background_inventory(self):
        message = OCRText("另一个设备正在连接这个村庄。", .999, (330, 334, 576, 358))
        complete = [OCRText("连接断开", .999, (330, 294, 426, 323)), message,
                    OCRText("重新载入", .999, (330, 408, 399, 430))]
        for foreground, scene in ((complete, "account_in_use"), ([message], "unknown")):
            with self.subTest(scene=scene):
                result = recognize(readings() + foreground)
                self.assertEqual(result.scene, scene)
                self.assertIsNone(result.observations["ore_inventory"])
                self.assertEqual(result.observations["buttons"], [])

    def test_recognized_inventory_does_not_expose_background_reward_controls(self):
        result = recognize(readings() + [OCRText("领取", .999, (930, 500, 980, 530))])
        self.assertEqual(result.scene, "popup")
        self.assertEqual([button["name"] for button in result.observations["buttons"]],
                         ["close_ore_inventory"])

    def test_current_event_panel_has_priority_over_background_inventory_anchors(self):
        event = [OCRText("活动", .999, (600, 42, 667, 68)),
                 OCRText("今日活动", .999, (210, 110, 290, 135)),
                 OCRText("X", .999, (1040, 35, 1070, 65))]
        result = recognize(readings() + event)
        self.assertEqual(result.scene, "popup")
        self.assertIsNotNone(result.observations["event_panel"])
        self.assertIsNone(result.observations["ore_inventory"])
        self.assertNotIn("close_ore_inventory", [b["name"] for b in result.observations["buttons"]])


if __name__ == "__main__":
    unittest.main()
