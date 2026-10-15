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


def receipt(kind, status="succeeded", **metrics):
    return {"task": "rift_event" if kind == "event" else kind, "status": status,
            "reason": "verified", "evidence": ["frame.png"],
            "metrics": {"receipt_kind": kind, **metrics}}


def report():
    rows = [receipt(kind) for kind in ("collect", "request", "donate", "train")]
    for index in range(10):
        battle = receipt("battle", battle_id=f"battle-{index}",
                            objective="event" if index == 0 else "resources", returned_home=True,
                            rounds_completed=1, settlement_observed={"stars": 1}, victory=index == 0,
                            deployment={"completed": True, "verified": True})
        battle["task"] = "rift_event" if index == 0 else "resources"
        rows.append(battle)
    rows.append({"task": "rift_event", "status": "succeeded", "reason": "goal_progress_observed",
                 "evidence": ["progress.png"], "metrics": {"receipt_kind": "goal_progress",
                 "battle_id": "battle-0", "before": {"current": 1}, "after": {"current": 2}}})
    return {"profile": "daily", "mode": "live", "run_id": "one-run", "elapsed_sec": 3600,
            "battles_completed": 10, "battles_won": 1, "task_results": rows}


def audit(payload):
    with TemporaryDirectory() as temp:
        path = Path(temp) / "run.json"
        (Path(temp) / "frame.png").write_bytes(b"verified fixture")
        (Path(temp) / "progress.png").write_bytes(b"verified fixture")
        path.write_text(json.dumps(payload), encoding="utf-8")
        return audit_daily_report(path)


def test_one_run_window_pass_requires_every_verified_receipt():
    result = audit(report())
    assert result["status"] == "PASS"
    assert result["complete_battles"] == result["best_complete_streak"] == 10
    assert result["task_state"]["train"] == "army_inspected_only"
    assert result["scope"] == "single_run_workflow_audit"
    assert result["formal_acceptance"] is False


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
    sample["task_results"][2] = receipt("donate", "skipped")
    sample["task_results"][2]["evidence"] = []
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
