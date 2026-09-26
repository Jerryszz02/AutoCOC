from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autococ.config import BattleConfig, load_config
from autococ.errors import ConfigError


class ConfigTests(unittest.TestCase):
    def test_native_transport_is_opt_in_and_requires_an_explicit_instance(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("", encoding="utf-8")
            self.assertIsNone(load_config(path).mumu)
            section = f"[mumu]\ninstall_dir = '{Path(tmp).as_posix()}'\n"
            path.write_text(section + "instance_index = 1\n", encoding="utf-8")
            self.assertEqual(load_config(path).mumu.instance_index, 1)
            for text in (section, section + "instance_index = -1\n", section + "instance_index = true\n",
                         "[mumu]\ninstall_dir = 'relative'\ninstance_index = 1\n"):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(path)

    def test_loads_defaults_for_missing_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("", encoding="utf-8")

            config = load_config(path)

            self.assertEqual(config.adb.scan_hosts, ("127.0.0.1",))
            self.assertEqual(config.runtime.log_level, "INFO")
            self.assertEqual(config.game.baseline_resolution, (1280, 720))
            self.assertEqual(config.vision.template_threshold, 0.8)
            self.assertEqual(config.ocr.provider, "auto")
            self.assertIn("core-loop", config.profiles)
            self.assertEqual(config.battle.max_searches, 30)
            self.assertTrue(config.reporting.write_markdown)
            self.assertEqual(config.stop.max_runs, 10)

    def test_rejects_invalid_log_level(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[runtime]\nlog_level = "NOPE"\n', encoding="utf-8")

            with self.assertRaises(ConfigError):
                load_config(path)

    def test_loads_profile_and_battle_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "\n".join(
                    [
                        "[profiles.custom]",
                        'enabled_tasks = ["launch", "battle"]',
                        "[battle]",
                        "min_expected_resources = 500000",
                        "max_searches = 12",
                        "[reporting]",
                        "write_markdown = false",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.profiles["custom"].enabled_tasks, ("launch", "battle"))
            self.assertEqual(config.battle.min_expected_resources, 500000)
            self.assertEqual(config.battle.max_searches, 12)
            self.assertFalse(config.reporting.write_markdown)

    def test_rejects_unimplemented_battle_objective(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\nobjective = "trophies"\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "only 'resources'"):
                load_config(path)

    def test_two_edge_strategy_loads_and_unknown_strategy_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\nstrategy = "two_edge"\n', encoding="utf-8")
            self.assertEqual(load_config(path).battle.strategy, "two_edge")
            path.write_text('[battle]\nstrategy = "guess"\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_legacy_weights_are_explicitly_ignored_for_existing_local_configs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\ntraining_cost_weight = 999\nresource_weight = 0\n', encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, "Ignored obsolete battle weights"):
                config = load_config(path)
            self.assertEqual(config.battle, BattleConfig())

    def test_direct_battle_config_validates_supported_goal_and_limits(self) -> None:
        for kwargs in ({"objective": "win_rate"}, {"min_expected_resources": -1},
                       {"max_searches": 0}, {"deploy_timeout_sec": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                BattleConfig(**kwargs)

    def test_rejects_unknown_profile_task(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                '[profiles.bad]\nenabled_tasks = ["launch", "unknown"]\n',
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_config(path)

    def test_rejects_invalid_resolution(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[game]\nbaseline_resolution = [0, 720]\n", encoding="utf-8")

            with self.assertRaises(ConfigError):
                load_config(path)

    def test_rejects_invalid_port(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[adb]\nscan_ports = [0]\n", encoding="utf-8")

            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
