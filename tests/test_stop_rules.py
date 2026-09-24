import time
import unittest

from autococ.config import StopConfig
from autococ.stop_rules import StopController


class StopControllerTests(unittest.TestCase):
    def test_stops_at_max_runs(self) -> None:
        controller = StopController(StopConfig(max_runs=2, max_duration_sec=100, max_failures=10))
        controller.record_success()
        self.assertFalse(controller.check().should_stop)
        controller.record_failure()

        decision = controller.check()

        self.assertTrue(decision.should_stop)
        self.assertIn("max_runs", decision.reason)

    def test_stops_at_max_failures(self) -> None:
        controller = StopController(StopConfig(max_runs=10, max_duration_sec=100, max_failures=1))
        controller.record_failure()

        decision = controller.check()

        self.assertTrue(decision.should_stop)
        self.assertIn("max_failures", decision.reason)

    def test_stops_at_max_duration(self) -> None:
        controller = StopController(
            StopConfig(max_runs=10, max_duration_sec=1, max_failures=10),
            started_monotonic=time.monotonic() - 2,
        )

        decision = controller.check()

        self.assertTrue(decision.should_stop)
        self.assertIn("max_duration_sec", decision.reason)


if __name__ == "__main__":
    unittest.main()
