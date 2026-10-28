from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autococ.config import BattleConfig, OCRConfig, VisionAgentConfig, load_config
from autococ.errors import ConfigError


class ConfigTests(unittest.TestCase):
    def test_vision_agent_is_explicit_and_validated(self):
        self.assertFalse(VisionAgentConfig().enabled)
        for kwargs in ({"enabled": 1}, {"preparation_reserve_sec": float("nan")},
                       {"preparation_reserve_sec": 0}, {"jev_timeout_sec": float("inf")},
                       {"mode": "magic"}, {"evidence_limit_mb": False}, {"jev_model": "jev-latest"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                VisionAgentConfig(**kwargs)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[runtime]\npng_compression_level = 0\n[vision_agent]\nenabled = true\nmodel_dir = "models/current"\n', encoding="utf-8")
            config = load_config(path)
            self.assertTrue(config.vision_agent.enabled)
            self.assertEqual(config.runtime.png_compression_level, 0)
            path.write_text('[vision_agent]\nunknown_option = true\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)
            for value in ("-1", "10", "true", "1.5"):
                path.write_text(f'[runtime]\npng_compression_level = {value}\n', encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(path)

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
            self.assertIsNone(config.ocr.max_side_len)
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

    def test_optional_ocr_side_limit_is_bounded_and_typed(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("[ocr]\nmax_side_len = 1280\n", encoding="utf-8")
            self.assertEqual(load_config(path).ocr.max_side_len, 1280)
            for value in ("511", "2001", "true", "1280.0", '"1280"'):
                with self.subTest(value=value):
                    path.write_text(f"[ocr]\nmax_side_len = {value}\n", encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(path)
        for value in (511, 2001, True, 1280.0, "1280"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                OCRConfig(max_side_len=value)

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
            with self.assertRaisesRegex(ConfigError, "resources, event or clan_games"):
                load_config(path)

    def test_two_edge_strategy_loads_and_unknown_strategy_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\nstrategy = "two_edge"\n', encoding="utf-8")
            self.assertEqual(load_config(path).battle.strategy, "two_edge")
            path.write_text('[battle]\nstrategy = "guess"\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_continuous_line_is_explicit_and_scoped_to_edrag_line(self):
        self.assertEqual(BattleConfig().line_execution_mode, "verified")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\nstrategy = "edrag_line"\nline_execution_mode = "continuous"\n', encoding="utf-8")
            self.assertEqual(load_config(path).battle.line_execution_mode, "continuous")
        for kwargs in ({"line_execution_mode": "continuous"},
                       {"strategy": "two_edge", "line_execution_mode": "continuous"},
                       {"strategy": "edrag_line", "line_execution_mode": "guess"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                BattleConfig(**kwargs)

    def test_two_edge_modes_require_explicit_continuous_edrag_line(self):
        self.assertEqual(BattleConfig().line_tactics, "single_edge")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            for tactics in ("two_edge_split", "two_edge_probed_anchors",
                            "two_edge_furnace_mixed_probes"):
                with self.subTest(tactics=tactics):
                    path.write_text('[battle]\nstrategy="edrag_line"\n'
                                    'line_execution_mode="continuous"\n'
                                    f'line_tactics="{tactics}"\n', encoding="utf-8")
                    self.assertEqual(load_config(path).battle.line_tactics, tactics)
                for strategy in ("verified", "edrag_line", "two_edge"):
                    with self.subTest(tactics=tactics, strategy=strategy), self.assertRaises(ConfigError):
                        BattleConfig(strategy=strategy, line_tactics=tactics)
        with self.assertRaises(ConfigError):
            BattleConfig(strategy="edrag_line", line_execution_mode="continuous", line_tactics="unknown")

    def test_post_deployment_skill_is_explicit_scoped_and_bounded(self):
        self.assertEqual(BattleConfig().post_deployment_hero_skill, "")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\nstrategy="edrag_line"\nline_execution_mode="continuous"\n'
                            'post_deployment_hero_skill="grand_warden"\nhero_skill_delay_sec=20\n', encoding="utf-8")
            config = load_config(path).battle
            self.assertEqual(config.post_deployment_hero_skill, "grand_warden")
            self.assertEqual(config.hero_skill_delay_sec, 20)
        for changes in ({"strategy": "verified", "line_execution_mode": "verified"},
                        {"line_execution_mode": "verified"}, {"strategy_file": "custom.toml"},
                        {"post_deployment_hero_skill": "archer_queen"},
                        {"post_deployment_hero_skill": None}, {"post_deployment_hero_skill": []},
                        {"hero_skill_delay_sec": -1}, {"hero_skill_delay_sec": 31},
                        {"hero_skill_delay_sec": True}, {"hero_skill_delay_sec": 1.5}):
            values = {"strategy": "edrag_line", "line_execution_mode": "continuous",
                      "post_deployment_hero_skill": "grand_warden"}
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                BattleConfig(**values)

    def test_legacy_weights_are_explicitly_ignored_for_existing_local_configs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\ntraining_cost_weight = 999\nresource_weight = 0\n', encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, "Ignored obsolete battle weights"):
                config = load_config(path)
            self.assertEqual(config.battle, BattleConfig())

    def test_direct_battle_config_validates_supported_goal_and_limits(self) -> None:
        for kwargs in ({"objective": "win_rate"}, {"min_expected_resources": -1},
                       {"max_searches": 0}, {"deploy_timeout_sec": False},
                       {"settlement_timeout_sec": 0}, {"settlement_timeout_sec": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                BattleConfig(**kwargs)

    def test_settlement_wait_has_independent_default_and_preserves_explicit_values(self) -> None:
        self.assertEqual(BattleConfig().settlement_timeout_sec, 240)
        self.assertEqual(BattleConfig(deploy_timeout_sec=90).settlement_timeout_sec, 240)
        self.assertIsNone(BattleConfig(settlement_timeout_sec=None).settlement_timeout_sec)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[battle]\ndeploy_timeout_sec = 90\n', encoding="utf-8")
            battle = load_config(path).battle
            self.assertEqual((battle.deploy_timeout_sec, battle.settlement_timeout_sec), (90, 240))
            path.write_text('[battle]\ndeploy_timeout_sec = 90\nsettlement_timeout_sec = 120\n',
                            encoding="utf-8")
            self.assertEqual(load_config(path).battle.settlement_timeout_sec, 120)
            for value in ("0", "-1", "true", "1.5"):
                with self.subTest(value=value):
                    path.write_text(f'[battle]\nsettlement_timeout_sec = {value}\n', encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(path)

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
