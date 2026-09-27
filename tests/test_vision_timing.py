import math
import unittest

from autococ.vision_timing import ActionTiming, summarize_action_timings


class ActionTimingTests(unittest.TestCase):
    def test_stage_order_and_finite_values_are_required(self) -> None:
        with self.assertRaises(ValueError):
            ActionTiming("spell", "warm", "failed", None)
        invalid = [
            {"t_pixels": -1},
            {"t_pixels": math.nan},
            {"t_pixels": math.inf},
            {"t_pixels": True},
            {"t_state": 1},
            {"t_pixels": 2, "t_state": 1},
            {"t_input_done": 1},
        ]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                ActionTiming("spell", "warm", "timeout", 0, **fields)
        with self.assertRaises(ValueError):
            ActionTiming("spell", "warm", "verified", 0)
        with self.assertRaises(ValueError):
            ActionTiming("spell", "warm", "input_sent", 0)

    def test_full_action_records_request_to_input_and_later_effect(self) -> None:
        action = ActionTiming("spell", "warm", "verified", 10, 10.1, 10.2, 10.3, 10.4, 10.9)
        self.assertAlmostEqual(action.input_latency_sec, .4)
        self.assertAlmostEqual(action.effect_latency_sec, .9)

    def test_failed_rejected_fallback_and_timeout_remain_in_denominator(self) -> None:
        records = [
            ActionTiming("spell", "cold", "verified", 0, .1, .2, .3, .4, .7),
            ActionTiming("spell", "warm", "input_sent", 1, 1.1, 1.2, 1.3, 1.8),
            ActionTiming("spell", "warm", "failed", 2, 2.1),
            ActionTiming("spell", "warm", "rejected", 3),
            ActionTiming("spell", "warm", "fallback", 4, 4.1, 4.2),
            ActionTiming("spell", "warm", "timeout", 5, 5.1, 5.2, 5.3),
            ActionTiming("troop", "warm", "input_sent", 6, 6.1, 6.2, 6.3, 6.5),
        ]
        report = summarize_action_timings(records)
        spell = report["by_action_type"]["spell"]["all"]
        self.assertEqual(spell["attempts"], 6)
        self.assertEqual(spell["input_completed_count"], 2)
        self.assertEqual(spell["unknown_input_latency_count"], 4)
        self.assertEqual(spell["over_target_count"], 1)
        self.assertEqual(spell["over_target_ratio"], 1 / 6)
        self.assertEqual(spell["not_on_time_count"], 5)
        self.assertAlmostEqual(spell["input_latency_sec"]["p50"], .6)
        self.assertAlmostEqual(spell["input_latency_sec"]["p95"], .78)
        self.assertEqual(report["by_action_type"]["spell"]["cold"]["attempts"], 1)
        self.assertEqual(report["by_action_type"]["troop"]["warm"]["attempts"], 1)
        self.assertEqual(report["overall"]["attempts"], 7)

    def test_no_samples_are_unknown_and_target_must_be_valid(self) -> None:
        report = summarize_action_timings([])
        self.assertIsNone(report["overall"]["input_latency_sec"]["p95"])
        self.assertIsNone(report["overall"]["over_target_ratio"])
        for target in (0, -1, True, math.inf, math.nan):
            with self.subTest(target=target), self.assertRaises(ValueError):
                summarize_action_timings([], target_sec=target)


if __name__ == "__main__":
    unittest.main()
