import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

from autococ.config import load_config
from autococ.daily import validate_routine_files
from autococ.daily_evidence import audit_daily_report
from autococ.routine_config import routine_from_dict


ROOT = Path(__file__).resolve().parents[1]


def test_current_client_combined_routine_is_loadable_and_ordered():
    data = tomllib.loads((ROOT / "routine.current-18-647-2.toml").read_text(encoding="utf-8"))
    routine = routine_from_dict(data["routine"])
    assert [(task.kind, task.enabled) for task in routine.tasks] == [
        (name, True) for name in ("collect", "request", "donate", "train", "event", "resources")]
    assert routine.tasks[3].kind == "train"  # Inspect only; battle preflight prepares/validates army.
    assert routine.tasks[4].goal.adapter_path.endswith("rift_event_18_647_2.adapter.toml")
    config = load_config(ROOT / "config.example.toml")
    validate_routine_files(config, routine)


def receipt(kind, status="succeeded", *, evidence=None, reason=None, **metrics):
    if reason is None:
        reason = {"collect": "collected_resources",
                  "request": "request_record_and_cooldown_verified",
                  "donate": "donation_units_and_cost_verified"}.get(kind, "verified")
    return {"task": "rift_event" if kind == "event" else kind, "status": status,
            "reason": reason, "evidence": ["frame.png"] if evidence is None else evidence,
            "metrics": {"receipt_kind": kind, **metrics}}


def report(*, required_checks=True):
    rows = [receipt(kind, evidence=[f"one-run/check-{kind}-0.png"] if kind != "train" else None)
            for kind in ("collect", "request", "donate", "train")]
    for index in range(10):
        battle = receipt("battle", battle_id=f"battle-{index}",
                            objective="event" if index == 0 else "resources", returned_home=True,
                            rounds_completed=1, settlement_observed={"stars": 1}, victory=index == 0,
                            deployment={"completed": True, "verified": True})
        battle["task"] = "rift_event" if index == 0 else "resources"
        rows.append(battle)
    if required_checks:
        rows.extend(receipt(kind, evidence=[f"one-run/check-{kind}-{index}.png"])
                    for kind, count in (("collect", 3), ("request", 2), ("donate", 3))
                    for index in range(1, count))
    rows.append({"task": "rift_event", "status": "succeeded", "reason": "goal_progress_observed",
                 "evidence": ["progress.png"], "metrics": {"receipt_kind": "goal_progress",
                 "battle_id": "battle-0", "before": {"current": 1}, "after": {"current": 2}}})
    return {"profile": "daily", "mode": "live", "run_id": "one-run", "elapsed_sec": 3600,
            "battles_completed": 10, "battles_won": 1, "task_results": rows}


def audit(payload, *, copied_evidence=()):
    with TemporaryDirectory() as temp:
        path = Path(temp) / "run.json"
        (Path(temp) / "frame.png").write_bytes(b"verified fixture")
        (Path(temp) / "progress.png").write_bytes(b"verified fixture")
        (Path(temp) / "one-run").mkdir()
        for kind, count in (("collect", 3), ("request", 2), ("donate", 3)):
            for index in range(count):
                name = f"check-{kind}-{index}.png"
                (Path(temp) / "one-run" / name).write_bytes(
                    b"same copied image" if name in copied_evidence else name.encode())
        path.write_text(json.dumps(payload), encoding="utf-8")
        return audit_daily_report(path)


def test_one_run_window_pass_requires_every_verified_receipt():
    result = audit(report())
    assert result["status"] == "PASS"
    assert result["complete_battles"] == result["best_complete_streak"] == 10
    assert result["task_state"]["train"] == "army_inspected_only"
    assert result["scope"] == "single_run_workflow_audit"
    assert result["formal_acceptance"] is False
    assert result["maintenance_checks"] == {"collect": 3, "donate": 3, "request": 2}
    assert result["maintenance_successes"] == result["maintenance_checks"]


def test_one_check_per_maintenance_kind_does_not_satisfy_run_01():
    result = audit(report(required_checks=False))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["maintenance_checks"] == {"collect": 1, "donate": 1, "request": 1}
    assert not result["checks"]["three_collect_checks"]
    assert not result["checks"]["three_donation_checks"]
    assert not result["checks"]["two_request_checks"]


def test_distinct_no_opportunity_checks_count_without_becoming_function_successes():
    sample = report()
    for row in sample["task_results"]:
        kind = row["metrics"].get("receipt_kind")
        if kind == "donate":
            row.update(status="skipped", reason="no_donation_requests_in_scanned_chat")
    sample["task_results"][0].update(status="skipped", reason="storage_full")
    sample["task_results"][1].update(status="skipped", reason="request_cooldown")
    result = audit(sample)
    assert result["maintenance_checks"] == {"collect": 3, "donate": 3, "request": 2}
    assert result["maintenance_successes"] == {"collect": 2, "donate": 0, "request": 1}
    assert all(result["checks"][key] for key in
               ("three_collect_checks", "three_donation_checks", "two_request_checks"))
    assert result["task_state"]["donate"] == "unverified_or_unavailable"
    assert result["formal_acceptance"] is False


def test_failed_unsupported_and_reused_evidence_do_not_count_as_new_checks():
    sample = report()
    sample["task_results"][14]["status"] = "failed"
    sample["task_results"][15].update(status="skipped", reason="not_supported")
    sample["task_results"][16]["evidence"] = sample["task_results"][0]["evidence"]
    sample["task_results"][18]["evidence"] = [
        "frame.png", "one-run/check-donate-2.png"]  # Mixed external/current evidence cannot be combined.
    result = audit(sample)
    assert result["maintenance_checks"] == {"collect": 1, "donate": 2, "request": 1}
    assert result["status"] == "INSUFFICIENT_DATA"
    copied = report()
    result = audit(copied, copied_evidence={"check-collect-0.png", "check-request-0.png"})
    assert result["maintenance_checks"]["request"] == 1  # Different paths, identical pixels.
    unsupported_success = report()
    unsupported_success["task_results"][14]["reason"] = "unverified"
    assert audit(unsupported_success)["maintenance_checks"]["collect"] == 2


def test_settled_failed_battle_does_not_extend_complete_streak_or_count_as_win():
    failed = report()
    failed["task_results"][8]["status"] = "failed"
    failed["task_results"][8]["metrics"]["deployment"]["completed"] = False
    result = audit(failed)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["settled_battles"] == 10
    assert result["complete_battles"] == 9
    assert result["best_complete_streak"] < 10
    assert result["verified_wins"] == 1


def test_aggregate_counter_conflict_or_duplicate_battle_id_fails():
    inconsistent = report()
    inconsistent["battles_completed"] = 11
    assert audit(inconsistent)["status"] == "FAIL"
    duplicate = report()
    duplicate["task_results"][6]["metrics"]["battle_id"] = "battle-0"
    assert audit(duplicate)["status"] == "FAIL"


def test_simulated_short_and_unavailable_daily_receipts_do_not_pass():
    sample = report()
    sample["mode"] = "dry-run"
    assert audit(sample)["status"] == "INSUFFICIENT_DATA"
    sample = report()
    sample["elapsed_sec"] = 3599
    for row in sample["task_results"]:
        if row["metrics"].get("receipt_kind") == "donate":
            row.update(status="skipped", reason="no_donation_requests_in_scanned_chat", evidence=[])
    result = audit(sample)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert not result["checks"]["elapsed_60_minutes"]
    assert result["task_state"]["donate"] == "unverified_or_unavailable"


def test_event_progress_must_follow_the_same_successful_event_battle():
    mismatched = report()
    mismatched["task_results"][-1]["metrics"]["battle_id"] = "battle-1"  # resource objective
    assert not audit(mismatched)["checks"]["event_progress_gain"]
    failed = report()
    failed["task_results"][4]["status"] = "failed"  # event battle settled but tactic failed
    assert not audit(failed)["checks"]["event_progress_gain"]
    arbitrary = report()
    arbitrary["task_results"][4]["task"] = "moon_madness"
    arbitrary["task_results"][-1]["task"] = "moon_madness"
    assert audit(arbitrary)["checks"]["event_progress_gain"]


def test_missing_source_screenshot_cannot_count_as_verified_battle():
    missing = report()
    missing["task_results"][4]["evidence"] = ["absent.png"]
    result = audit(missing)
    assert result["settled_battles"] == 9
    assert result["status"] == "FAIL"  # aggregate claimed ten settled receipts
