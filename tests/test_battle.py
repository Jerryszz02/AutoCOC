import unittest

from autococ.battle import BattleTarget, parse_resource_number, score_target
from autococ.config import BattleConfig


class BattleTests(unittest.TestCase):
    def test_scores_resource_first_target(self) -> None:
        config = BattleConfig(min_expected_resources=300000)
        target = BattleTarget(
            gold=200000,
            elixir=150000,
            dark_elixir=10000,
        )

        score = score_target(target, config)

        self.assertTrue(score.should_attack)
        self.assertEqual(score.gold_elixir_total, 350000)
        self.assertFalse(score.can_search_next)
        self.assertIn("resource floor met", score.reasons)

    def test_rejects_target_below_resource_floor(self) -> None:
        config = BattleConfig(min_expected_resources=300000)

        score = score_target(BattleTarget(gold=50000, elixir=0), config)

        self.assertFalse(score.should_attack)
        self.assertTrue(score.can_search_next)

    def test_dark_elixir_does_not_fill_gold_elixir_threshold(self) -> None:
        target = BattleTarget(gold=100000, elixir=100000, dark_elixir=1000000)
        score = score_target(target, BattleConfig(min_expected_resources=300000))
        self.assertFalse(score.should_attack)
        self.assertEqual(score.gold_elixir_total, 200000)
        self.assertEqual(score.target.dark_elixir, 1000000)

    def test_unreadable_or_invalid_resources_are_not_zero(self) -> None:
        for value in (None, -1, True, 150000.0, float("nan")):
            with self.subTest(value=value):
                score = score_target(BattleTarget(gold=value, elixir=500000), BattleConfig())
                self.assertFalse(score.should_attack)
                self.assertIsNone(score.gold_elixir_total)

    def test_exact_threshold_and_unknown_dark_elixir_are_accepted(self) -> None:
        score = score_target(BattleTarget(gold=150000, elixir=150000), BattleConfig())
        self.assertTrue(score.should_attack)
        self.assertIsNone(score.target.dark_elixir)

    def test_zero_threshold_still_requires_readable_amounts(self) -> None:
        config = BattleConfig(min_expected_resources=0)
        self.assertTrue(score_target(BattleTarget(gold=0, elixir=0), config).should_attack)
        self.assertFalse(score_target(BattleTarget(), config).should_attack)

    def test_search_budget_never_relaxes_resource_floor(self) -> None:
        config = BattleConfig(max_searches=2)
        poor = BattleTarget(gold=100000, elixir=100000)
        self.assertTrue(score_target(poor, config, searches=1).can_search_next)
        final = score_target(poor, config, searches=2)
        self.assertFalse(final.should_attack)
        self.assertFalse(final.can_search_next)
        self.assertIn("search limit reached", final.reasons)

    def test_last_allowed_candidate_may_be_attacked_but_over_budget_may_not(self) -> None:
        config = BattleConfig(max_searches=2)
        rich = BattleTarget(gold=300000, elixir=0)
        self.assertTrue(score_target(rich, config, searches=2).should_attack)
        self.assertFalse(score_target(rich, config, searches=3).should_attack)

    def test_search_counter_is_one_based(self) -> None:
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                score_target(BattleTarget(gold=300000, elixir=0), BattleConfig(), searches=count)

    def test_parses_resource_suffixes(self) -> None:
        self.assertEqual(parse_resource_number("1.2M"), 1_200_000)
        self.assertEqual(parse_resource_number("1,2M"), 1_200_000)
        self.assertEqual(parse_resource_number("450K"), 450_000)
        self.assertEqual(parse_resource_number("300,000"), 300_000)

    def test_unreadable_resource_text_is_unknown(self) -> None:
        for value in ("", "unreadable", "?", "1..2M", "9" * 400):
            with self.subTest(value=value):
                self.assertIsNone(parse_resource_number(value))


if __name__ == "__main__":
    unittest.main()
