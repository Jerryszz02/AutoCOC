from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from autococ.config import ProfileConfig, load_config
from autococ.reporting import DecisionTrace, RunStats, TaskResult, capture_run_provenance, resource_metrics, summarize_run, write_report


def success(task: str, **metrics: object) -> TaskResult:
    return TaskResult(task, "succeeded", "verified postcondition", evidence=[Path("after.png")], metrics=metrics)


class ReportingTests(unittest.TestCase):
    def test_success_requires_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence"):
            TaskResult("battle", "succeeded", "no exception")

    def test_dry_run_never_counts_simulation_as_real_success(self) -> None:
        stats = RunStats(mode="dry-run")
        stats.record_task(success("battle", loot_gold=100))
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.attempts, 0)
        self.assertEqual(stats.simulated, 1)
        self.assertEqual(stats.task_results[0].status, "simulated")
        self.assertIsNone(resource_metrics(stats)["resources"]["gold"]["battle_loot"])

    def test_report_rechecks_mode_when_caller_appends_results_directly(self) -> None:
        stats = RunStats(mode="dry-run", successes=8, task_results=[success("battle", loot_gold=100)])
        payload = summarize_run(stats)
        self.assertEqual(payload["successes"], 0)
        self.assertEqual(payload["simulated"], 1)
        self.assertEqual(payload["task_results"][0]["status"], "simulated")

    def test_skips_and_simulations_do_not_inflate_attempts(self) -> None:
        stats = RunStats(cycles=1)
        stats.record_task(TaskResult("donate", "skipped", "no matching requests"))
        stats.record_task(TaskResult("request", "simulated", "offline plan"))
        stats.record_task(TaskResult("collect", "failed", "postcondition not confirmed"))
        stats.record_task(success("launch"))
        payload = summarize_run(stats)
        self.assertEqual((stats.attempts, stats.successes, stats.failures, stats.skipped, stats.simulated), (2, 1, 1, 1, 1))
        self.assertEqual(payload["cycles"], 1)
        self.assertEqual(payload["attempts"], 2)

    def test_missing_numbers_are_unknown_not_zero(self) -> None:
        stats = RunStats(task_results=[success("battle", loot_gold=100), success("battle")])
        gold = resource_metrics(stats)["resources"]["gold"]
        self.assertIsNone(gold["battle_loot"])
        self.assertIsNone(gold["operating_net"])
        self.assertIsNone(gold["battle_gross_per_hour"])

    def test_unrelated_launch_does_not_contaminate_income(self) -> None:
        stats = RunStats(task_results=[success("battle", loot_gold=100, bonus_gold=20), success("launch")])
        gold = resource_metrics(stats, elapsed_sec=1800)["resources"]["gold"]
        self.assertEqual(gold["battle_loot"], 100)
        self.assertEqual(gold["battle_gross_per_hour"], 240)

    def test_resource_ledger_separates_gross_collection_and_costs(self) -> None:
        stats = RunStats(task_results=[
            success("battle", loot_gold=100, bonus_gold=20, search_cost_gold=3,
                    loot_elixir=200, bonus_elixir=40, search_cost_elixir=0),
            success("collect", collected_gold=50, collected_elixir=60),
            success("donate", donation_cost_gold=10, donation_cost_elixir=30),
            TaskResult("battle", "simulated", "not real", metrics={"loot_gold": 90000}),
        ])
        metrics = resource_metrics(stats, elapsed_sec=1800)
        gold = metrics["resources"]["gold"]
        self.assertEqual(gold["battle_gross"], 120)
        self.assertEqual(gold["operating_net"], 157)
        self.assertEqual(gold["operating_net_per_hour"], 314)
        self.assertEqual(metrics["battle_gold_elixir_per_hour"], 720)
        self.assertEqual(metrics["net_gold_elixir_per_hour"], 854)
        self.assertIsNone(metrics["resources"]["dark_elixir"]["battle_loot"])

    def test_verified_costs_of_failed_donation_are_preserved(self) -> None:
        stats = RunStats(task_results=[TaskResult("donate", "failed", "partial donation", metrics={"donation_cost_elixir": 30})])
        self.assertEqual(resource_metrics(stats)["resources"]["elixir"]["donation_spend"], 30)

    def test_absent_cost_events_are_zero(self) -> None:
        stats = RunStats(task_results=[
            success("collect", collected_gold=100, collected_elixir=0, collected_dark_elixir=0),
            TaskResult("donate", "skipped", "no visible donation requests"),
        ])
        for values in resource_metrics(stats)["resources"].values():
            self.assertEqual(values["donation_spend"], 0)
            self.assertEqual(values["search_spend"], 0)

    def test_observed_cost_events_with_missing_amounts_remain_unknown(self) -> None:
        stats = RunStats(task_results=[
            TaskResult("donate", "failed", "cost unreadable", metrics={"donation_cost_elixir": None}),
            TaskResult("battle", "failed", "search cost unreadable"),
        ])
        values = resource_metrics(stats)["resources"]["elixir"]
        self.assertIsNone(values["donation_spend"])
        self.assertIsNone(values["search_spend"])

    def test_verified_partial_collection_is_credited_without_counting_task_success(self) -> None:
        failed = TaskResult("collect", "failed", "second click could not be verified", metrics={
            "collected_gold": 9999,
            "verified_collections": [
                {"resource": "gold", "increase": 50},
                {"resource": "elixir", "increase": 20},
            ],
        })
        stats = RunStats(task_results=[success("collect", collected_gold=10, collected_elixir=0, collected_dark_elixir=0), failed])
        metrics = resource_metrics(stats)
        self.assertEqual(metrics["resources"]["gold"]["collected_credited"], 60)
        self.assertEqual(metrics["resources"]["elixir"]["collected_credited"], 20)
        self.assertEqual(metrics["resources"]["dark_elixir"]["collected_credited"], 0)
        self.assertEqual(summarize_run(stats)["successes"], 1)
        self.assertEqual(summarize_run(stats)["failures"], 1)

    def test_failed_collection_without_verification_cannot_claim_revenue(self) -> None:
        result = TaskResult("collect", "failed", "unverified", metrics={"collected_gold": 5000})
        self.assertIsNone(resource_metrics(RunStats(task_results=[result]))["resources"]["gold"]["collected_credited"])

    def test_invalid_partial_collection_amount_is_unknown(self) -> None:
        result = TaskResult("collect", "failed", "unverified amount", metrics={
            "verified_collections": [{"resource": "gold", "increase": None}],
        })
        self.assertIsNone(resource_metrics(RunStats(task_results=[result]))["resources"]["gold"]["collected_credited"])

    def test_simulation_cannot_contribute_partial_income_or_costs(self) -> None:
        result = TaskResult("collect", "failed", "offline replay", metrics={
            "verified_collections": [{"resource": "gold", "increase": 50}],
        })
        stats = RunStats(mode="dry-run", task_results=[result, success("donate", donation_cost_gold=30)])
        values = resource_metrics(stats)["resources"]["gold"]
        self.assertIsNone(values["collected_credited"])
        self.assertEqual(values["donation_spend"], 0)

    def test_no_samples_and_invalid_numbers_are_unknown(self) -> None:
        for value in (None, "100", True, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                stats = RunStats(task_results=[success("battle", loot_gold=value)])
                self.assertIsNone(resource_metrics(stats)["resources"]["gold"]["battle_loot"])
        self.assertIsNone(resource_metrics(RunStats())["battle_gold_elixir_per_hour"])

    def test_markdown_and_json_preserve_unknown_and_hide_disabled_trace(self) -> None:
        stats = RunStats(mode="dry-run", task_results=[TaskResult("collect", "simulated", "offline")])
        stats.decision_traces.append(DecisionTrace(1, "collect", "village", 1, "private trace", (), None))
        with TemporaryDirectory() as temp:
            report = write_report(Path(temp), stats, save_decision_trace=False)
            document = report.read_text(encoding="utf-8")
            payload = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertIn("Mode: dry-run", document)
            self.assertIn("unknown", document)
            self.assertNotIn("Decision Trace", document)
            self.assertNotIn("private trace", document)
            self.assertNotIn("decision_traces", payload)
            self.assertIsNone(payload["resource_metrics"]["resources"]["gold"]["battle_loot"])

    def test_concurrent_runs_with_identical_clock_do_not_overwrite(self) -> None:
        with TemporaryDirectory() as temp, patch("autococ.reporting.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 22, 12, 0, 0, 123456)
            stats = [RunStats() for _ in range(8)]
            self.assertEqual(len({item.run_id for item in stats}), 8)
        with TemporaryDirectory() as temp:
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(lambda item: write_report(Path(temp), item), stats))
            self.assertEqual(len(set(paths)), 8)
            self.assertEqual(len(list(Path(temp).glob("*.json"))), 8)
            with self.assertRaises(FileExistsError):
                write_report(Path(temp), stats[0])


class ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "src/autococ/reporting.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source at run start")
        self.template = self.root / "assets/templates/icon.png"
        self.template.parent.mkdir(parents=True)
        self.template.write_bytes(b"template at run start")
        config_path = self.root / "config.toml"
        config_path.write_text("", encoding="utf-8")
        loaded = load_config(config_path)
        self.config = replace(loaded, profiles={**loaded.profiles, "core": ProfileConfig(("launch", "collect", "battle"))})
        patcher = patch("autococ.reporting.__file__", str(self.source))
        patcher.start()
        self.addCleanup(patcher.stop)

    def capture(self, config=None):
        return capture_run_provenance(config or self.config, "core")

    def test_exact_hashes_cover_only_source_and_template_files(self) -> None:
        for relative in ("reports/previous.json", ".venv/ignored.py", "src/autococ/notes.txt"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"excluded")
        provenance = self.capture()
        self.assertEqual(provenance["files_sha256"], {
            "src/autococ/reporting.py": hashlib.sha256(b"source at run start").hexdigest(),
            "assets/templates/icon.png": hashlib.sha256(b"template at run start").hexdigest(),
        })
        self.assertEqual(len(provenance["source_fingerprint_sha256"]), 64)
        self.assertEqual(provenance["client_version"], "unknown")
        self.assertEqual(provenance["package_name"], self.config.game.package_name)
        self.assertEqual(provenance["baseline_resolution"], [1280, 720])
        self.assertEqual(provenance["tasks"], list(self.config.profiles["core"].enabled_tasks))
        self.assertEqual(provenance["errors"], {})

    def test_config_hash_is_order_stable_and_reflects_effective_overrides(self) -> None:
        initial = self.capture()
        reordered = replace(self.config, profiles=dict(reversed(list(self.config.profiles.items()))),
                            source_path=self.root / "another-config.toml")
        self.assertEqual(initial["effective_config_sha256"], self.capture(reordered)["effective_config_sha256"])
        once = replace(self.config, stop=replace(self.config.stop, max_runs=1))
        dry = replace(self.config, runtime=replace(self.config.runtime, dry_run=True))
        for config in (once, dry):
            with self.subTest(config=config):
                self.assertNotEqual(initial["effective_config_sha256"], self.capture(config)["effective_config_sha256"])

    def test_file_edits_and_additions_change_new_snapshot_not_saved_report(self) -> None:
        initial = self.capture()
        self.source.write_bytes(b"changed while run is active")
        changed = self.capture()
        self.assertNotEqual(initial["source_fingerprint_sha256"], changed["source_fingerprint_sha256"])
        (self.source.parent / "new_module.py").write_bytes(b"new source")
        self.assertNotEqual(changed["source_fingerprint_sha256"], self.capture()["source_fingerprint_sha256"])
        stats = RunStats(provenance=initial)
        report = write_report(self.root / "reports", stats)
        payload = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(payload["provenance"], initial)
        self.assertIn(initial["source_fingerprint_sha256"], report.read_text(encoding="utf-8"))

    def test_unreadable_file_preserves_partial_hashes_without_complete_fingerprint(self) -> None:
        original_read = Path.read_bytes

        def read(path):
            if path == self.template:
                raise PermissionError("sensitive diagnostic must not be persisted")
            return original_read(path)

        with patch.object(Path, "read_bytes", read):
            provenance = self.capture()
        self.assertIsNone(provenance["source_fingerprint_sha256"])
        self.assertIsNone(provenance["files_sha256"]["assets/templates/icon.png"])
        self.assertEqual(provenance["errors"]["assets/templates/icon.png"], "PermissionError")
        self.assertIsNotNone(provenance["files_sha256"]["src/autococ/reporting.py"])
        self.assertNotIn("sensitive diagnostic", json.dumps(provenance))

    def test_missing_templates_and_unserializable_config_remain_unknown(self) -> None:
        self.template.unlink()
        self.template.parent.rmdir()
        config = replace(self.config, runtime=replace(self.config.runtime, poll_interval_sec=float("nan")))
        provenance = self.capture(config)
        self.assertIsNone(provenance["source_fingerprint_sha256"])
        self.assertIsNone(provenance["effective_config_sha256"])
        self.assertIn("assets/templates", provenance["errors"])
        self.assertIn("effective_config", provenance["errors"])

    def test_report_contains_config_hash_but_no_config_values_or_credentials(self) -> None:
        config = replace(self.config, adb=replace(self.config.adb, manual_serial="private-config-sentinel"))
        provenance = self.capture(config)
        report = write_report(self.root / "reports", RunStats(provenance=provenance))
        for path in (report, report.with_suffix(".json")):
            document = path.read_text(encoding="utf-8")
            self.assertIn(provenance["effective_config_sha256"], document)
            self.assertNotIn("private-config-sentinel", document)
            self.assertNotIn("manual_serial", document)


if __name__ == "__main__":
    unittest.main()
