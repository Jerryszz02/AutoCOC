from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from autococ.errors import FlowError
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.session import GameSession
from autococ.social import donate_troops, request_reinforcements


BODY = "救命啊！我需要增援！"


def text(value: str, box: list[int]) -> dict:
    return {"text": value, "bbox": box, "confidence": 0.99}


def frame(index: int, scene: str, *, cooldown: str | None = None, record: bool = False,
          age: str | None = "刚刚", gems: int = 323, extra: list | None = None, donate: bool = False,
          older_messages: bool = False) -> SceneSnapshot:
    ocr = [text(str(gems), [1160, 226, 1208, 252])]
    buttons = []
    if scene == "request":
        ocr += [text(BODY, [377, 311, 613, 339]), text("50/50", [445, 124, 520, 150])]
        buttons += [{"name": "send", "point": [782, 502], "text": "发送"},
                    {"name": "send", "point": [80, 680], "text": "聊天发送"}]
    if scene == "clan_chat":
        ocr += [text("部落成员的可见聊天", [50, 100, 250, 130])]
    if record:
        ocr += [text(BODY, [37, 374, 240, 398]), text("请求：", [44, 431, 90, 453]),
                text("0/50", [124, 574, 175, 595]), text("0/3", [266, 573, 306, 595])]
        if age is not None:
            ocr += [text(age, [430, 620, 465, 640])]
    if cooldown is not None:
        ocr += [text(cooldown, [427, 681, 473, 708]), text("6", [430, 655, 446, 674])]
    if donate:
        buttons += [{"name": "donate", "point": [420, 450], "text": "捐赠"}]
    if older_messages:
        buttons += [{"name": "chat_latest", "point": [42, 617], "text": "Latest messages"}]
    ocr += extra or []
    return SceneSnapshot(scene, 0.95, Path(f"social-{index}.png"), {"ocr": ocr, "buttons": buttons})


class FakeRecognizer:
    def __init__(self, region_results: list[OCRText] | None = None,
                 line_results: list[OCRText] | None = None) -> None:
        self.region_results = region_results or []
        self.line_results = line_results or []
        self.calls = []

    def recognize_region(self, path: Path, roi: tuple[int, int, int, int], *, single_line=False) -> list[OCRText]:
        self.calls.append((path, roi, single_line))
        return self.line_results if single_line else self.region_results


class FakeSession:
    buttons = staticmethod(GameSession.buttons)
    click = GameSession.click

    def __init__(self, frames: list[SceneSnapshot], *, region_results: list[OCRText] | None = None,
                 line_results: list[OCRText] | None = None) -> None:
        self.frames = iter(frames)
        self.recognizer = FakeRecognizer(region_results, line_results)
        self.config = SimpleNamespace(runtime=SimpleNamespace(poll_interval_sec=0))
        self.taps, self.template_clicks, self.waits, self.events = [], [], [], []
        self.swipes = []
        self.last_snapshot = None

    def check_deadline(self) -> None:
        pass

    def observe(self, label: str = "observe") -> SceneSnapshot:
        try:
            self.last_snapshot = next(self.frames)
        except StopIteration:
            raise FlowError("Test observation deadline exceeded")
        return self.last_snapshot

    def wait_for(self, scenes: set[str], *, timeout_sec: float = 30, label: str = "wait") -> SceneSnapshot:
        self.waits.append((scenes, timeout_sec))
        for _ in range(10):
            snapshot = self.observe(label)
            if snapshot.scene in scenes:
                return snapshot
        raise FlowError("Test scene deadline exceeded")

    def tap(self, snapshot: SceneSnapshot, point: list[int], *, reason: str) -> None:
        if snapshot is not self.last_snapshot:
            raise AssertionError("Stale screenshot action")
        self.taps.append((snapshot.screenshot_path, point))

    def click_template(self, snapshot: SceneSnapshot, name: str, *, roi: tuple[int, int, int, int]) -> None:
        self.template_clicks.append((name, roi))

    def swipe(self, snapshot: SceneSnapshot, start: tuple[int, int], end: tuple[int, int], *,
              duration_ms: int, reason: str) -> None:
        if snapshot is not self.last_snapshot or snapshot.scene != "clan_chat":
            raise AssertionError("Unverified chat swipe")
        self.swipes.append((snapshot.screenshot_path, start, end, duration_ms))

    def event(self, kind: str, **data: object) -> None:
        self.events.append((kind, data))


class SocialTests(unittest.TestCase):
    def setUp(self) -> None:
        sleep = patch("autococ.social.time.sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_request_restores_latest_chat_before_opening_request(self) -> None:
        session = FakeSession([
            frame(0, "clan_chat", older_messages=True),
            frame(1, "clan_chat", older_messages=True), frame(2, "clan_chat"),
            frame(3, "request"), frame(4, "clan_chat", cooldown="59秒", record=True),
        ])
        result = request_reinforcements(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(session.taps, [(Path("social-0.png"), [42, 617]), (Path("social-3.png"), [782, 502])])
        self.assertEqual(result.metrics["send_actions"], 1)

    def test_request_dialog_opened_from_old_chat_restores_messages_after_send(self) -> None:
        session = FakeSession([
            frame(0, "request"), frame(1, "clan_chat", cooldown="59秒", older_messages=True),
            frame(2, "clan_chat", cooldown="58秒", record=True),
        ])
        result = request_reinforcements(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["send_actions"], 1)
        self.assertEqual(session.taps[-1], (Path("social-1.png"), [42, 617]))

    def test_latest_navigation_without_progress_never_sends_request(self) -> None:
        session = FakeSession([frame(i, "clan_chat", older_messages=True) for i in range(4)])
        result = request_reinforcements(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["send_actions"], 0)
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(session.template_clicks, [])

    def test_request_waits_for_dialog_and_confirms_record_cooldown_and_unchanged_gems(self) -> None:
        session = FakeSession([
            frame(0, "village"), frame(1, "clan_chat"), frame(2, "clan_chat"),
            frame(3, "request"), frame(4, "request"), frame(5, "clan_chat", cooldown="1分钟", record=True),
        ])
        result = request_reinforcements(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["request_submitted"], 1)
        self.assertEqual(result.metrics["request_cost_gems"], 0)
        self.assertEqual(result.metrics["cooldown_seconds"], 60)
        self.assertEqual(session.taps, [(Path("social-3.png"), [782, 502])])
        self.assertEqual([item[0] for item in session.template_clicks], ["hud_chat", "chat_request"])
        self.assertIn(({"request"}, 45), session.waits)

    def test_request_configuration_capacity_is_not_received_capacity(self) -> None:
        session = FakeSession([frame(0, "request"), frame(1, "clan_chat", cooldown="1分钟", record=True)])
        self.assertEqual(request_reinforcements(session).status, "succeeded")
        self.assertEqual(len(session.taps), 1)

    def test_initial_cooldown_skips_without_spending_gems_or_clicking(self) -> None:
        session = FakeSession([frame(0, "clan_chat", cooldown="59秒")])
        result = request_reinforcements(session)
        self.assertEqual((result.status, result.reason), ("skipped", "request_cooldown"))
        self.assertEqual(session.taps, [])
        self.assertEqual(session.template_clicks, [])

    def test_request_record_without_cooldown_is_not_success(self) -> None:
        session = FakeSession([frame(0, "request")] + [frame(i, "clan_chat", record=True) for i in range(1, 4)])
        result = request_reinforcements(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["request_submitted"], 0)
        self.assertEqual(len(session.taps), 1)

    def test_cooldown_without_new_request_record_is_not_success(self) -> None:
        session = FakeSession([frame(0, "request")] + [frame(i, "clan_chat", cooldown="1分钟") for i in range(1, 4)])
        self.assertEqual(request_reinforcements(session).status, "failed")
        self.assertEqual(len(session.taps), 1)

    def test_old_request_and_new_cooldown_do_not_satisfy_confirmation(self) -> None:
        session = FakeSession([frame(0, "request")] + [
            frame(i, "clan_chat", cooldown="1分钟", record=True, age="14小时52分钟") for i in range(1, 4)
        ])
        self.assertEqual(request_reinforcements(session).status, "failed")

    def test_small_timestamp_uses_region_ocr_when_full_frame_misses_it(self) -> None:
        session = FakeSession([
            frame(0, "request"), frame(1, "clan_chat", cooldown="1分钟", record=True, age=None),
        ], region_results=[OCRText("刚刚", 0.99, (430, 620, 465, 640))])
        self.assertEqual(request_reinforcements(session).status, "succeeded")
        self.assertEqual(session.recognizer.calls[0][1], (350, 595, 489, 650))

    def test_tiny_gray_timestamp_uses_anchored_line_when_detection_misses_it(self) -> None:
        session = FakeSession([
            frame(0, "request"), frame(1, "clan_chat", cooldown="47秒", record=True, age=None),
        ], line_results=[OCRText("刚刚", 0.9999, (420, 607, 485, 646))])
        result = request_reinforcements(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(session.recognizer.calls[-1], (Path("social-1.png"), (420, 607, 485, 646), True))
        self.assertEqual(result.metrics["request_record"]["timestamp_evidence"]["source"], "timestamp_line_ocr")

    def test_line_ocr_cannot_accept_old_or_uncertain_timestamp(self) -> None:
        for value, confidence in (("刚刚", 0.89), ("刚刚", float("nan")), ("2分钟", 0.99), ("14小时", 0.99)):
            with self.subTest(value=value, confidence=confidence):
                session = FakeSession([frame(0, "request")] + [
                    frame(i, "clan_chat", cooldown="47秒", record=True, age=None) for i in range(1, 4)
                ], line_results=[OCRText(value, confidence, (420, 607, 485, 646))])
                self.assertEqual(request_reinforcements(session).status, "failed")

    def test_known_old_timestamp_is_not_overridden_by_line_fallback(self) -> None:
        session = FakeSession([frame(0, "request")] + [
            frame(i, "clan_chat", cooldown="47秒", record=True, age="2分钟") for i in range(1, 4)
        ], line_results=[OCRText("刚刚", 0.99, (420, 607, 485, 646))])
        self.assertEqual(request_reinforcements(session).status, "failed")
        self.assertEqual(session.recognizer.calls, [])

    def test_gem_price_bare_number_is_not_a_cooldown(self) -> None:
        session = FakeSession([frame(0, "request")] + [
            frame(i, "clan_chat", cooldown="6", record=True) for i in range(1, 4)
        ])
        self.assertEqual(request_reinforcements(session).status, "failed")

    def test_unrelated_building_timer_is_not_request_cooldown(self) -> None:
        session = FakeSession([frame(0, "request")] + [
            frame(i, "clan_chat", record=True, extra=[text("5分钟", [1125, 446, 1181, 468])])
            for i in range(1, 4)
        ])
        self.assertEqual(request_reinforcements(session).status, "failed")

    def test_gem_change_fails_even_if_request_has_appeared(self) -> None:
        session = FakeSession([frame(0, "request"), frame(1, "clan_chat", cooldown="1分钟", record=True, gems=317)])
        result = request_reinforcements(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["request_cost_gems"], 6)

    def test_request_dialog_timeout_never_clicks_send_on_chat(self) -> None:
        session = FakeSession([frame(0, "clan_chat"), frame(1, "clan_chat")])
        self.assertEqual(request_reinforcements(session).status, "failed")
        self.assertEqual(session.taps, [])

    def test_wrong_scene_does_not_generate_social_actions(self) -> None:
        for handler in (request_reinforcements, donate_troops):
            with self.subTest(handler=handler):
                session = FakeSession([frame(0, "battle")])
                self.assertEqual(handler(session).status, "failed")
                self.assertEqual(session.taps, [])
                self.assertEqual(session.template_clicks, [])

    def test_no_visible_donation_requests_is_explicit_skip(self) -> None:
        session = FakeSession([frame(i, "clan_chat") for i in range(4)])
        result = donate_troops(session)
        self.assertEqual((result.status, result.reason), ("skipped", "no_donation_requests_in_scanned_chat"))
        self.assertEqual(result.metrics["pages_scanned"], 2)
        self.assertEqual(result.metrics["unique_pages_scanned"], 1)
        self.assertEqual(result.metrics["scan_stop_reason"], "repeated_page")
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(len(result.evidence), 4)

    def test_visible_request_opens_dialog_but_uncalibrated_unit_grid_stays_failed(self) -> None:
        session = FakeSession([frame(0, "clan_chat", donate=True), frame(1, "donation")])
        result = donate_troops(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("donation_layout_unverified", result.reason)
        self.assertEqual(result.metrics["visible_donation_requests"], 1)
        self.assertEqual(session.taps, [(Path("social-0.png"), [420, 450])])
        self.assertEqual(result.metrics["donated_units"], 0)
        self.assertEqual(result.evidence[-1], Path("social-1.png"))

    def test_unreadable_chat_cannot_claim_no_requests(self) -> None:
        snapshot = frame(0, "clan_chat")
        snapshot.observations["ocr"] = []
        self.assertEqual(donate_troops(FakeSession([snapshot])).status, "failed")

    def test_own_latest_request_does_not_skip_older_donation_opportunity(self) -> None:
        session = FakeSession([frame(0, "clan_chat", record=True), frame(1, "clan_chat", donate=True), frame(2, "donation")])
        result = donate_troops(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("donation_layout_unverified", result.reason)
        self.assertEqual(result.metrics["pages_scanned"], 2)
        self.assertEqual(result.metrics["scan_stop_reason"], "donation_request_found")
        self.assertEqual(session.swipes[0][1:3], ((180, 200), (180, 600)))
        self.assertEqual(session.taps, [(Path("social-1.png"), [420, 450])])

    def test_four_distinct_pages_are_scanned_without_clicking_messages(self) -> None:
        session = FakeSession([
            frame(i, "clan_chat", extra=[text(f"玩家{i}的旧消息", [50, 220, 250, 250])]) for i in range(4)
        ])
        result = donate_troops(session)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metrics["pages_scanned"], 4)
        self.assertEqual(result.metrics["unique_pages_scanned"], 4)
        self.assertEqual(result.metrics["scan_stop_reason"], "page_limit")
        self.assertEqual(result.metrics["scan_scope"], "current_clan_chat_and_up_to_three_older_pages")
        self.assertEqual(len(result.metrics["scan_pages"]), 4)
        self.assertEqual(len(session.swipes), 3)
        self.assertEqual(len([e for e in session.events if e[0] == "donation_scan_page"]), 4)
        self.assertEqual(session.taps, [])

    def test_last_scan_page_donation_is_not_discarded_by_page_limit(self) -> None:
        session = FakeSession([
            frame(i, "clan_chat", donate=i == 3, extra=[text(f"旧消息{i}", [50, 220, 250, 250])])
            for i in range(4)
        ])
        result = donate_troops(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["visible_donation_requests"], 1)
        self.assertEqual(result.metrics["pages_scanned"], 4)

    def test_old_frame_after_swipe_is_reobserved_without_second_swipe(self) -> None:
        session = FakeSession([frame(0, "clan_chat"), frame(1, "clan_chat"), frame(2, "clan_chat", donate=True), frame(3, "donation")])
        result = donate_troops(session)
        self.assertIn("donation_layout_unverified", result.reason)
        self.assertEqual(len(session.swipes), 1)
        self.assertEqual(len(result.evidence), 4)

    def test_timer_hud_animation_and_ocr_order_do_not_create_new_chat_page(self) -> None:
        frames = [frame(i, "clan_chat", record=True, age=f"{30+i}分钟", gems=323+i,
                        cooldown=f"{60-i}秒") for i in range(4)]
        frames[2].observations["ocr"].reverse()
        session = FakeSession(frames)
        result = donate_troops(session)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metrics["unique_pages_scanned"], 1)
        self.assertEqual(result.metrics["scan_stop_reason"], "repeated_page")
        self.assertEqual(len(session.swipes), 1)

    def test_unreadable_scrolled_page_fails_instead_of_skipping(self) -> None:
        unreadable = frame(1, "clan_chat")
        unreadable.observations["ocr"] = []
        session = FakeSession([frame(0, "clan_chat"), unreadable])
        self.assertEqual(donate_troops(session).status, "failed")
        self.assertEqual(len(session.swipes), 1)

    def test_revisited_page_ends_scan_even_if_previous_page_was_different(self) -> None:
        session = FakeSession([frame(0, "clan_chat"),
                               frame(1, "clan_chat", extra=[text("另一页", [50, 220, 250, 250])]),
                               frame(2, "clan_chat")])
        result = donate_troops(session)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.metrics["scan_stop_reason"], "repeated_page")
        self.assertEqual(result.metrics["unique_pages_scanned"], 2)
        self.assertEqual(len(session.swipes), 2)


if __name__ == "__main__":
    unittest.main()
