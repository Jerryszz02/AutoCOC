from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from autococ.gui import AutoCOCApp


class GUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interpreter = tk.Tk()
        cls.interpreter.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.interpreter.destroy()

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.toml"
        self.path.write_text("", encoding="utf-8")
        self.root = tk.Toplevel(self.interpreter)
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.controller = Mock()
        self.controller.running = False
        self.app = AutoCOCApp(self.root, self.path, controller=self.controller)

    def test_opening_window_does_not_start_or_connect(self):
        self.controller.start.assert_not_called()
        self.assertEqual(self.app.mode.get(), "离线预演")
        self.assertEqual(self.app.start_button.cget("text"), "开始预演")

    def test_selected_preset_is_passed_to_background_worker(self):
        self.app.preset.set("仅对战")
        self.app._preset_changed()
        self.app.start()
        options = self.controller.start.call_args.args[1]
        self.assertEqual(options.tasks, ("launch", "battle"))
        self.assertTrue(options.dry_run)
        self.assertTrue(self.app.start_button.instate(["disabled"]))
        self.app.stop()
        self.controller.stop.assert_called_once_with()

    def test_invalid_settings_never_start_worker(self):
        self.app.values["minutes"].set("nan")
        with patch("autococ.gui.messagebox.showerror") as error:
            self.app.start()
        error.assert_called_once()
        self.controller.start.assert_not_called()

    def test_gui_completion_distinguishes_simulation_from_success(self):
        self.app._event({"kind": "finished", "report": None,
                         "summary": {"mode": "dry-run", "successes": 0, "failures": 0,
                                     "skipped": 0, "simulated": 5, "stop_reason": "dry-run planning complete"}})
        self.assertEqual(self.app.counters["successes"].get(), "0")
        self.assertEqual(self.app.counters["simulated"].get(), "5")
        self.assertIn("报告未保存", self.app.status.get())

    def test_close_requests_stop_and_keeps_window_until_worker_exits(self):
        self.controller.running = True
        self.app.close()
        self.controller.stop.assert_called_once_with()
        self.assertTrue(self.app.closing)
        self.assertTrue(self.root.winfo_exists())

    def test_finished_interruption_does_not_leave_task_running(self):
        self.app.start()
        self.app._event({"kind": "task_started", "task": "launch"})
        self.app._event({"kind": "finished", "report": None, "summary": {
            "mode": "live", "successes": 0, "failures": 1, "stop_reason": "interrupted by user",
            "task_results": [{"task": "launch", "status": "failed", "reason": "interrupted by user"}]}})
        self.assertEqual(self.app.task_table.item("launch", "values")[1], "失败")
        self.assertEqual(self.app.task_table.item("battle", "values")[1], "未执行")
