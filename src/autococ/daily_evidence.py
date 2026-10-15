"""Conservative read-only audit of one daily run report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


REQUIRED_TASKS = ("collect", "request", "donate", "train", "event", "resources")


def audit_daily_report(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict) or not isinstance(report.get("task_results"), list):
        raise ValueError("Expected one run report with task_results")
    results = report["task_results"]
    if any(not isinstance(row, dict) for row in results):
        raise ValueError("Invalid task receipt")
    live = report.get("profile") == "daily" and report.get("mode") == "live"
    elapsed = report.get("elapsed_sec")
    elapsed = elapsed if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0 else None
    task_receipts = {kind: [] for kind in REQUIRED_TASKS}
    battles = []
    goal_progress = []
    reached_goals = set()

    def evidence_present(row: dict) -> bool:
        evidence = row.get("evidence")
        return (isinstance(evidence, list) and bool(evidence) and
                all(isinstance(item, str) and
                    (Path(item) if Path(item).is_absolute() else path.parent / item).is_file()
                    for item in evidence))

    for row in results:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        kind = metrics.get("receipt_kind")
        if kind in task_receipts:
            task_receipts[kind].append({"task": row.get("task"), "status": row.get("status"),
                                        "reason": row.get("reason"), "evidence": evidence_present(row)})
        if row.get("reason") == "goal_reached" and row.get("status") == "succeeded" and evidence_present(row):
            reached_goals.add(row.get("task"))
        if kind == "goal_progress" and row.get("status") == "succeeded" and evidence_present(row):
            before, after = metrics.get("before"), metrics.get("after")
            gain = (after["current"] - before["current"] if isinstance(before, dict) and
                    isinstance(after, dict) and type(before.get("current")) is int and
                    type(after.get("current")) is int else None)
            goal_progress.append({"task": row.get("task"), "battle_id": metrics.get("battle_id"),
                                  "verified_gain": gain if gain is not None and gain > 0 else None})
        if kind != "battle":
            continue
        deployment = metrics.get("deployment") if isinstance(metrics.get("deployment"), dict) else {}
        settled_home = (metrics.get("returned_home") is True and
                        type(metrics.get("rounds_completed")) is int and metrics["rounds_completed"] == 1 and
                        isinstance(metrics.get("settlement_observed"), dict) and evidence_present(row))
        complete = (row.get("status") == "succeeded" and settled_home and
                    deployment.get("completed") is True and deployment.get("verified") is True)
        battles.append({"battle_id": metrics.get("battle_id"), "task": row.get("task"),
                        "objective": metrics.get("objective"),
                        "status": row.get("status"), "settled_home": settled_home,
                        "complete": complete, "victory": metrics.get("victory") is True,
                        "reason": row.get("reason")})
    ids = [item["battle_id"] for item in battles]
    distinct = all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids)
    streak = best_streak = 0
    for item in battles:
        streak = streak + 1 if item["complete"] else 0
        best_streak = max(best_streak, streak)
    settled_count = sum(item["settled_home"] for item in battles)
    count_consistent = type(report.get("battles_completed")) is int and report["battles_completed"] == settled_count
    won_count = sum(item["settled_home"] and item["victory"] for item in battles)
    win_consistent = type(report.get("battles_won")) is int and report["battles_won"] == won_count
    contradictions = []
    if live and (not distinct or not count_consistent or not win_consistent):
        contradictions.append("battle receipt IDs or aggregate counters conflict")
    task_state = {}
    for kind, rows in task_receipts.items():
        if not rows:
            task_state[kind] = "missing"
        elif any(row["status"] == "failed" for row in rows):
            task_state[kind] = "failed"
        elif any(row["status"] == "succeeded" and row["evidence"] for row in rows):
            task_state[kind] = "observed"
        else:
            task_state[kind] = "unverified_or_unavailable"
    task_ids_by_objective = {kind: {b["task"] for b in battles if b["objective"] == kind}
                             for kind in ("event", "resources")}
    for kind in ("event", "resources"):
        task_state[kind] = ("observed" if reached_goals & task_ids_by_objective[kind] or
                            any(b["objective"] == kind and b["complete"] for b in battles)
                            else "unverified_or_unavailable")
    # Army inspection is only an observation, never proof that training was changed.
    if task_state["train"] == "observed":
        task_state["train"] = "army_inspected_only"
    completed_event_battles = {b["battle_id"]: b["task"] for b in battles
                               if b["complete"] and b["objective"] == "event"}
    event_gain = (bool(reached_goals & task_ids_by_objective["event"]) or
                  any(item["verified_gain"] is not None and
                      completed_event_battles.get(item["battle_id"]) == item["task"]
                      for item in goal_progress))
    checks = {"one_live_daily_run": live, "elapsed_60_minutes": elapsed is not None and elapsed >= 3600,
              "ten_consecutive_complete_battles": best_streak >= 10 and distinct,
              "at_least_one_verified_win": won_count >= 1,
              "event_progress_gain": event_gain,
              "army_inspected": task_state["train"] == "army_inspected_only",
              "all_tasks_observed": all(task_state[k] == "observed" for k in
                                        ("collect", "request", "donate", "event", "resources"))}
    return {"schema_version": 1, "scope": "single_run_workflow_audit",
            "formal_acceptance": False,
            "unassessed_acceptance": ["REV gross/net resource ledger thresholds",
                                      "zero gem spending across audited actions",
                                      "DON repeated successful donation opportunities",
                                      "model quality, released strategy and sustained win criteria"],
            "source_report": str(path.resolve()),
            "source_sha256": hashlib.sha256(raw).hexdigest(), "run_id": report.get("run_id"),
            "profile": report.get("profile"), "mode": report.get("mode"), "elapsed_sec": elapsed,
            "status": "FAIL" if contradictions else "PASS" if all(checks.values()) else "INSUFFICIENT_DATA",
            "checks": checks, "contradictions": contradictions,
            "task_state": task_state, "battle_receipts": battles,
            "settled_battles": settled_count, "complete_battles": sum(b["complete"] for b in battles),
            "best_complete_streak": best_streak, "verified_wins": won_count,
            "goal_progress": goal_progress,
            "limits": ["PASS is limited to the listed single-run workflow checks; formal daily acceptance remains false.",
                       "Training task only inspects the army; it does not prove a recipe change.",
                       "A goal reached without a battle objective receipt cannot be attributed to an arbitrary task ID.",
                       "This report audits one run only; external or later recovery cannot fill missing receipts."]}
