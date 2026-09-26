"""Task defaults are the same for direct construction and saved routines."""

import unittest

from autococ.errors import ConfigError
from autococ.routine_config import ResourceFilter, TaskSpec, routine_from_dict


class RoutineFilterDefaultsTests(unittest.TestCase):
    def test_battle_task_defaults_match_direct_and_parsed_construction(self):
        expected = {
            "resources": ResourceFilter(min_total=300000),
            "event": ResourceFilter(enabled=False),
            "clan_games": ResourceFilter(enabled=False),
        }
        for kind, rule in expected.items():
            with self.subTest(kind=kind):
                direct = TaskSpec(kind, kind)
                parsed = routine_from_dict({"tasks": [{"id": kind, "kind": kind}]}).tasks[0]
                self.assertEqual(direct.resource_filter, rule)
                self.assertEqual(parsed.resource_filter, rule)
                direct.validate()

    def test_explicit_conditions_are_not_augmented_by_defaults(self):
        for kind in ("resources", "event", "clan_games"):
            with self.subTest(kind=kind):
                selected = ResourceFilter(enabled=True, min_dark_elixir=9000)
                direct = TaskSpec(kind, kind, resource_filter=selected)
                parsed = routine_from_dict({"tasks": [{"id": kind, "kind": kind,
                    "resource_filter": {"enabled": True, "min_dark_elixir": 9000}}]}).tasks[0]
                self.assertEqual(direct.resource_filter, selected)
                self.assertEqual(parsed.resource_filter, selected)
                self.assertIsNone(parsed.resource_filter.min_total)

    def test_explicit_invalid_filter_still_fails_validation(self):
        for value in (None, {"enabled": False}, "disabled"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    TaskSpec("event", "event", resource_filter=value).validate()
        with self.assertRaises(ConfigError):
            routine_from_dict({"tasks": [{"id": "event", "kind": "event",
                "resource_filter": {"enabled": "false"}}]})


if __name__ == "__main__":
    unittest.main()
