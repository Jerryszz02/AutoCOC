from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from autococ.army_editor import recognize_army_editor
from autococ.army_navigation import open_current_army, swipe_current_siege_picker
from autococ.errors import FlowError
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot


class CurrentSiegeNavigationTests(TestCase):
    def test_new_client_opens_edit_army_without_search_purchase(self):
        home = SceneSnapshot("village", .95, Path("home.png"))
        menu = SceneSnapshot("search", .95, Path("menu.png"))
        army = SceneSnapshot("training", .95, Path("army.png"))
        session = Mock(client_version="18.647.2")
        session.wait_for.side_effect = [menu, army]
        evidence = []
        self.assertIs(open_current_army(session, home, evidence), army)
        self.assertEqual([call.args[1] for call in session.click.call_args_list], ["attack", "edit_army"])
        self.assertEqual(evidence, [menu.screenshot_path])
        session.click_template.assert_not_called()

    def test_missing_edit_button_never_falls_back_to_find_match(self):
        home = SceneSnapshot("village", .95, Path("home.png"))
        session = Mock(client_version="18.647.2")
        session.wait_for.return_value = SceneSnapshot("search", .95, Path("menu.png"))
        session.click.side_effect = [None, FlowError("edit_army missing")]
        with self.assertRaises(FlowError):
            open_current_army(session, home, [])
        self.assertEqual([call.args[1] for call in session.click.call_args_list], ["attack", "edit_army"])
        session.click_template.assert_not_called()

    def test_real_siege_tray_allows_navigation_but_no_recipe_mutation(self):
        path = Path(__file__).parent / "fixtures" / "siege_picker_navigation_18_647_2.png"
        texts = [OCRText("强化军队", .99, (892, 45, 950, 66)),
                 OCRText("强化英雄", .99, (1070, 46, 1128, 66)),
                 OCRText("4/4", .99, (50, 64, 98, 90)),
                 OCRText("340/340", .99, (580, 70, 655, 90)),
                 OCRText("11/11", .99, (591, 250, 635, 270))]
        editor = recognize_army_editor(path, None, texts, client_version="18.647.2")
        self.assertEqual(editor["surface"], "current_picker")
        self.assertEqual(editor["editing_kind"], "siege")
        self.assertTrue(editor["ready"])
        self.assertFalse(editor["complete"])
        self.assertEqual([c["action"] for c in editor["controls"]], ["close_picker"])
        for version in (None, "unknown", "18.600.7"):
            with self.subTest(version=version):
                unknown = recognize_army_editor(path, None, texts, client_version=version)
                self.assertFalse(unknown["ready"])
                self.assertEqual(unknown["controls"], [])

    def test_swipe_invalidates_frame_and_never_clicks_unit(self):
        session, frame = self.session()
        swipe_current_siege_picker(session, frame, direction="next")
        session.context.swipe.assert_called_once_with(980, 560, 330, 560, 550)
        session.context.tap.assert_not_called()
        self.assertIsNone(session.last_snapshot)
        self.assertEqual(session.action_count, 1)

    def test_no_input_for_unverified_or_unrelated_surface(self):
        for change in ("version", "frame", "surface", "kind", "direction", "stale"):
            with self.subTest(change=change):
                session, frame = self.session()
                editor = frame.observations["army_editor"]
                direction = "next"
                if change == "version": session.client_version = "unknown"
                if change == "frame": editor["frame"] = "previous.png"
                if change == "surface": editor["surface"] = "current"
                if change == "kind": editor["editing_kind"] = "hero"
                if change == "direction": direction = "up"
                if change == "stale": session._validate_snapshot.side_effect = FlowError("stale")
                with self.assertRaises(FlowError):
                    swipe_current_siege_picker(session, frame, direction=direction)
                session.context.swipe.assert_not_called()
                self.assertEqual(session.action_count, 0)

    def session(self):
        frame = SceneSnapshot("training", .95, Path("current.png"), {"army_editor": {
            "frame": "current.png", "surface": "current_picker", "ready": True, "editing_kind": "siege"}})
        session = SimpleNamespace(_validate_snapshot=Mock(), client_version="18.647.2",
            config=SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720))),
            context=Mock(), event=Mock(), action_count=0, last_snapshot=frame)
        return session, frame
