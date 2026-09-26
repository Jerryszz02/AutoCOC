"""Run resource battles and verify deployment, settlement, and return inventory."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
import time
import unicodedata

from .battle import BattleTarget, score_target
from .errors import AutoCOCError, DeploymentError, FlowError
from .flow import return_to_village
from .reporting import TaskResult
from .scene import SceneSnapshot
from .session import GameSession
from .strategies import is_line_strategy


RESOURCES = ("gold", "elixir", "dark_elixir")


def inspect_army(session: GameSession) -> TaskResult:
    """Inspect the displayed recipe; clan capacity does not prove received troops."""
    started_at, started = datetime.now(), time.monotonic()
    evidence: list[Path] = []
    metrics: dict[str, object] = {}
    status, reason = "failed", "army_inspection_incomplete"
    try:
        home = return_to_village(session)
        evidence.append(home.screenshot_path)
        session.click_template(home, "hud_army", roi=(0, 450, 140, 600))
        army_frame = session.wait_for({"training"}, timeout_sec=30, label="inspect-army")
        evidence.append(army_frame.screenshot_path)
        metrics.update(_read_army(army_frame))
        home = return_to_village(session)
        evidence.append(home.screenshot_path)
        status, reason = "succeeded", "army_recipe_inspected_and_returned_home"
    except AutoCOCError as exc:
        reason = str(exc)
        _retain_last_frame(session, evidence)
    return TaskResult("train", status, reason, started_at, time.monotonic() - started, evidence, metrics)


def run_battle(session: GameSession) -> TaskResult:
    started_at, started = datetime.now(), time.monotonic()
    evidence: list[Path] = []
    searches: list[dict[str, object]] = []
    metrics: dict[str, object] = {
        "searches": searches, "search_count": 0,
        "search_cost_gold": 0, "search_cost_elixir": 0, "search_cost_dark_elixir": 0,
        "deployment": None, "settlement_observed": None, "inventory_reconciled": False,
    }
    status, reason = "failed", "battle_verification_incomplete"
    settlement_error: str | None = None
    line_mode = is_line_strategy(session.config.battle.strategy)
    metrics["strategy"] = session.config.battle.strategy
    try:
        if line_mode and session.native is None:
            raise FlowError("Two-edge minimum zoom requires the MuMu native transport")
        home = return_to_village(session)
        evidence.append(home.screenshot_path)
        metrics["resources_before"] = home.observations.get("resources")
        before = home.observations.get("resources", {}) if line_mode else _inventory(home)
        session.click(home, "attack")
        menu = session.wait_for({"search"}, timeout_sec=30, label="battle-menu")
        evidence.append(menu.screenshot_path)
        pending_cost = _search_price(menu)
        session.click(menu, "find_match_regular")
        army_frame = session.wait_for({"training"}, timeout_sec=45, label="battle-army-confirmation")
        evidence.append(army_frame.screenshot_path)
        army = _read_army(army_frame)
        metrics["army"] = army
        if not army["troops_full"] or not line_mode and not army["spells_full"]:
            raise FlowError("Army troop/spell recipe is not full")

        total_search_cost = 0
        metrics["search_cost_gold"] = None
        session.click(army_frame, "attack")
        scout = _next_candidate(session)
        while True:
            evidence.append(scout.screenshot_path)
            total_search_cost += pending_cost
            metrics["search_cost_gold"] = total_search_cost
            available = scout.observations.get("resources", {})
            if not isinstance(available, dict) or not line_mode and scout.observations.get("resource_source") != "enemy_available":
                raise FlowError("Enemy available resources were not identified")
            target = BattleTarget(gold=available.get("gold"), elixir=available.get("elixir"),
                                  dark_elixir=available.get("dark_elixir"))
            decision = score_target(target, session.config.battle, searches=len(searches) + 1)
            search = {"candidate": len(searches) + 1, "frame": str(scout.screenshot_path),
                      "search_cost_gold": pending_cost, "decision": asdict(decision)}
            searches.append(search)
            metrics["search_count"] = len(searches)
            session.event("battle_candidate", **search)
            if decision.should_attack:
                metrics["selected_target"] = asdict(target)
                break
            if not decision.can_search_next:
                raise FlowError("Search budget exhausted without a qualifying target")
            pending_cost = _search_price(scout)
            metrics["search_cost_gold"] = None
            session.click(scout, "next")
            scout = _next_candidate(session, previous=scout)

        from .deployment import deploy_army

        scout.observations["expected_army_manifest"] = army["recipe"]["manifest"]
        try:
            deployed = deploy_army(session, scout)
        except DeploymentError as exc:
            last = session.last_snapshot
            units = exc.partial_receipt.get("deployed_units")
            issued = exc.partial_receipt.get("issued_placements", 0)
            verified_partial = exc.partial_receipt.get("verified") is True and type(units) is int and units > 0
            attempted_line = line_mode and type(issued) is int and issued > 0
            if (last is None or not (verified_partial or attempted_line)
                    or last.scene not in {"enemy_village", "battle", "settlement"}):
                raise
            # A partial attack still needs its result recorded and a safe return
            # home. Do not replay deployment actions after an uncertain result.
            deployed = last
            deployed.observations["deployment"] = exc.partial_receipt
            metrics["deployment_error"] = str(exc)
        evidence.append(deployed.screenshot_path)
        deployment = deployed.observations.get("deployment")
        metrics["deployment"] = deployment
        _require_scene(deployed, {"enemy_village", "battle", "settlement"})
        cleanup_only = line_mode and "deployment_error" in metrics
        if not isinstance(deployment, dict) or not cleanup_only and deployment.get("verified") is not True:
            raise FlowError("Deployment has no verified receipt")
        if (type(deployment.get("deployed_units")) is not int or deployment["deployed_units"] < 0
                or deployment["deployed_units"] == 0 and not cleanup_only):
            raise FlowError("No positive troop deployment was verified")
        deployment_evidence = deployment.get("evidence")
        if not isinstance(deployment_evidence, list) or not deployment_evidence:
            raise FlowError("Deployment receipt has no screenshot evidence")
        evidence.extend(Path(path) for path in deployment_evidence)

        settlement_frame = deployed if deployed.scene == "settlement" else session.wait_for(
            {"settlement"}, timeout_sec=session.config.battle.deploy_timeout_sec + (35 if cleanup_only else 0), label="battle-settlement",
            **({"purpose": "settlement", "poll_interval_sec": 5} if line_mode else {}))
        settlement = None
        reads: list[dict[str, object]] = []
        metrics["settlement_reads"] = reads
        read_started = time.monotonic()
        read_deadline = read_started + 10
        for attempt in range(1 if line_mode else 4):  # Resource mode allows at most three fresh rereads.
            if settlement_frame.screenshot_path not in evidence:
                evidence.append(settlement_frame.screenshot_path)
            observed = settlement_frame.observations.get("settlement")
            metrics["settlement_observed"] = observed
            reading = {"frame": str(settlement_frame.screenshot_path), "scene": settlement_frame.scene,
                       "settlement": observed, "elapsed_sec": time.monotonic() - read_started,
                       "valid": False, "error": None}
            reads.append(reading)
            try:
                session.check_deadline()
                if attempt and time.monotonic() >= read_deadline:
                    raise FlowError("Settlement reread window exceeded")
                settlement = _settlement(settlement_frame)
            except AutoCOCError as exc:
                reading["error"] = str(exc)
                settlement_error = settlement_error or str(exc)
            else:
                reading["valid"] = True
                settlement_error = None
                break
            if line_mode:
                break
            if attempt == 3 or time.monotonic() >= read_deadline:
                break
            session.check_deadline()
            time.sleep(min(session.config.runtime.poll_interval_sec, 1, max(0, read_deadline - time.monotonic())))
            session.check_deadline()
            if time.monotonic() >= read_deadline:
                break
            settlement_frame = session.observe("battle-settlement-reread")
        metrics["settlement_validation_error"] = settlement_error
        # An unsupported result can still be left safely. Never act on an older
        # settlement if the newest observation is unknown or another scene.
        session.check_deadline()
        _require_scene(settlement_frame, {"settlement"})
        session.click(settlement_frame, "return_home")
        home_after = session.wait_for({"village", "popup"}, timeout_sec=45, label="battle-return-home")
        if home_after.scene == "popup":
            _require_scene(home_after, {"popup"})
            reward = [item for item in home_after.observations.get("ocr", [])
                      if item["text"].replace(" ", "").rstrip("!！") == "已收到胜利之星奖励"
                      and math.isfinite(item.get("confidence", 0)) and .95 <= item["confidence"] <= 1]
            if len(reward) != 1:
                raise FlowError("Unrecognized popup after battle return")
            evidence.append(home_after.screenshot_path)
            metrics["star_bonus_popup_frame"] = str(home_after.screenshot_path)
            # This receipt acknowledges rewards already received; never confirm
            # an arbitrary popup, or repeat the click while it animates away.
            session.click(home_after, "confirm")
            home_after = session.wait_for({"village"}, timeout_sec=20, label="battle-reward-return-home")
        evidence.append(home_after.screenshot_path)
        metrics["resources_after"] = home_after.observations.get("resources")
        if line_mode:
            observed = metrics["settlement_observed"] or {}
            for group, prefix in (("loot", "loot"), ("bonus", "bonus")):
                for resource in RESOURCES:
                    proof = (observed.get("evidence") or {}).get(group) or {}
                    proof = proof.get(resource, proof)
                    metrics[f"{prefix}_{resource}"] = (None if proof.get("requires_inventory_reconciliation") else
                                                       (observed.get(group) or {}).get(resource))
            stars = observed.get("stars")
            metrics.update(stars=stars, percentage=observed.get("percentage"),
                           victory=stars > 0 if type(stars) is int else None,
                           returned_home=True, rounds_completed=1)
            after = home_after.observations.get("resources") or {}
            if type(before.get("gems")) is int and type(after.get("gems")) is int and before["gems"] != after["gems"]:
                # Village counters animate from zero on arrival. Confirm a
                # discrepancy from fresh frames before reporting a balance change.
                reads = [{"frame": str(home_after.screenshot_path), "resources": after}]
                metrics["home_resource_reads"] = reads
                for _ in range(3):
                    session.check_deadline()
                    time.sleep(min(session.config.runtime.poll_interval_sec, 1))
                    home_after = session.observe("battle-home-resources")
                    evidence.append(home_after.screenshot_path)
                    _require_scene(home_after, {"village"})
                    after = home_after.observations.get("resources") or {}
                    reads.append({"frame": str(home_after.screenshot_path), "resources": after})
                    metrics["resources_after"] = after
                    if after.get("gems") == before["gems"]:
                        break
                else:
                    raise FlowError("Gem balance changed during battle or could not be reverified")
            if deployment.get("completed") is not True:
                raise FlowError(metrics.get("deployment_error") or "Line troop/hero deployment is incomplete")
            session.event("battle_round_verified", settlement_frame=str(settlement_frame.screenshot_path),
                          home_frame=str(home_after.screenshot_path), victory=metrics["victory"])
            return TaskResult("battle", "succeeded", f"{session.config.battle.strategy}_troops_settlement_and_return_verified",
                              started_at, time.monotonic() - started, evidence, metrics)
        if settlement_error is not None:
            raise FlowError(settlement_error)
        after = _inventory(home_after)
        metrics["inventory_delta"] = {name: after[name] - before[name] for name in (*RESOURCES, "gems")}
        for resource in RESOURCES:
            metrics[f"loot_{resource}"] = settlement["loot"][resource]
            metrics[f"bonus_{resource}"] = settlement["bonus"][resource]
        metrics.update({"stars": settlement["stars"], "percentage": settlement["percentage"],
                        "victory": settlement["stars"] > 0})
        expected = {resource: settlement["loot"][resource] + settlement["bonus"][resource]
                    - (total_search_cost if resource == "gold" else 0) for resource in RESOURCES}
        metrics["expected_inventory_delta"] = expected
        if after["gems"] != before["gems"]:
            raise FlowError("Gem balance changed during battle")
        if any(after[resource] - before[resource] != expected[resource] for resource in RESOURCES):
            raise FlowError("Battle settlement does not reconcile with return-home inventory")
        metrics["inventory_reconciled"] = True
        if deployment.get("completed") is not True:
            raise FlowError(metrics.get("deployment_error") or "Army or hero ability deployment is incomplete")
        session.event("battle_verified", settlement_frame=str(settlement_frame.screenshot_path),
                      home_frame=str(home_after.screenshot_path), inventory_delta=metrics["inventory_delta"])
        status, reason = "succeeded", "deployment_settlement_and_return_inventory_verified"
    except AutoCOCError as exc:
        reason = settlement_error or str(exc)
        if settlement_error is not None:
            metrics["settlement_validation_error"] = settlement_error
            if str(exc) != settlement_error:
                metrics["settlement_followup_error"] = str(exc)
        if isinstance(exc, DeploymentError):
            metrics["deployment"] = exc.partial_receipt
            for path in exc.partial_receipt.get("evidence", []):
                if Path(path) not in evidence:
                    evidence.append(Path(path))
        _retain_last_frame(session, evidence)
    return TaskResult("battle", status, reason, started_at, time.monotonic() - started, evidence, metrics)


def _require_scene(snapshot: SceneSnapshot, scenes: set[str]) -> None:
    if snapshot.scene not in scenes or not math.isfinite(snapshot.confidence) or snapshot.confidence < 0.8:
        raise FlowError(f"Expected battle scene {sorted(scenes)}, observed {snapshot.scene}")


def _inventory(snapshot: SceneSnapshot) -> dict[str, int]:
    _require_scene(snapshot, {"village"})
    values = snapshot.observations.get("resources")
    if snapshot.observations.get("resource_source") != "village_inventory" or not isinstance(values, dict):
        raise FlowError("Village inventory source is not verified")
    if any(type(values.get(name)) is not int or values[name] < 0 for name in (*RESOURCES, "gems")):
        raise FlowError("Village inventory is incomplete")
    return {name: values[name] for name in (*RESOURCES, "gems")}


def _read_army(snapshot: SceneSnapshot) -> dict[str, object]:
    _require_scene(snapshot, {"training"})
    army = snapshot.observations.get("army")
    if not isinstance(army, dict):
        raise FlowError("Army recipe is unreadable")
    full = {}
    for name in ("troops", "spells"):
        ratio = army.get(name)
        if not isinstance(ratio, dict) or any(type(ratio.get(key)) is not int for key in ("used", "capacity")):
            raise FlowError(f"Army {name} capacity is unreadable")
        if not 0 <= ratio["used"] <= ratio["capacity"] or name == "troops" and ratio["capacity"] == 0:
            raise FlowError(f"Army {name} capacity is invalid")
        full[f"{name}_full"] = ratio["used"] == ratio["capacity"]
    manifest = army.get("manifest")
    if (not isinstance(manifest, dict) or manifest.get("complete") is not True
            or manifest.get("supported_layout") is not True
            or manifest.get("frame") != str(snapshot.screenshot_path)):
        raise FlowError("Independent army card manifest is incomplete")
    return {"recipe": army, **full, "clan_received_unknown": True, "heroes_available": None}


def _search_price(snapshot: SceneSnapshot) -> int:
    value = snapshot.observations.get("search_cost_gold")
    if type(value) is not int or value < 0:
        raise FlowError("Search gold cost is unreadable")
    return value


def _candidate_identity(snapshot: SceneSnapshot) -> tuple:
    values = snapshot.observations.get("resources", {})
    if not isinstance(values, dict):
        raise FlowError("Candidate resources are unreadable")
    headings = set()
    for item in snapshot.observations.get("ocr", []):
        box = item.get("bbox")
        confidence = item.get("confidence", 0)
        if (box and 40 <= box[0] < 360 and 0 <= (box[1] + box[3]) / 2 <= 40
                and math.isfinite(confidence) and .85 <= confidence <= 1):
            # Only the player name, excluding clan and badge labels. OCR case
            # and spacing jitter do not prove that a paid search has completed.
            name = "".join(unicodedata.normalize("NFKC", item["text"]).casefold().split())
            if name:
                headings.add(name)
    return tuple(values.get(resource) for resource in RESOURCES), tuple(headings) if len(headings) == 1 else ()


def _candidate_changed(previous: SceneSnapshot, current: SceneSnapshot) -> bool:
    before, before_names = _candidate_identity(previous)
    after, after_names = _candidate_identity(current)
    changed_amount = any(type(old) is int and old >= 0 and type(new) is int and new >= 0 and old != new
                         for old, new in zip(before, after))
    return changed_amount or bool(before_names and after_names and before_names != after_names)


def _next_candidate(session: GameSession, previous: SceneSnapshot | None = None) -> SceneSnapshot:
    # Observed cloud transitions plus capture/OCR can take over 35 seconds.
    # wait_for still enforces the shorter session/task deadline.
    deadline = time.monotonic() + 60
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FlowError("Next target did not become distinguishable from the previous candidate")
        snapshot = session.wait_for({"enemy_village"}, timeout_sec=remaining, label="battle-candidate")
        _require_scene(snapshot, {"enemy_village"})
        if is_line_strategy(session.config.battle.strategy) and previous is None:
            return _wait_for_clear_scout(session, snapshot, deadline)
        values = snapshot.observations.get("resources", {})
        readable = (isinstance(values, dict) and snapshot.observations.get("resource_source") == "enemy_available"
                    and all(type(values.get(name)) is int and values[name] >= 0 for name in ("gold", "elixir")))
        if readable and (previous is None or _candidate_changed(previous, snapshot)):
            snapshot = _wait_for_clear_scout(session, snapshot, deadline)
            values = snapshot.observations.get("resources", {})
            if (snapshot.observations.get("resource_source") == "enemy_available" and isinstance(values, dict)
                    and all(type(values.get(name)) is int and values[name] >= 0 for name in ("gold", "elixir"))
                    and (previous is None or _candidate_changed(previous, snapshot))):
                return snapshot
        session.check_deadline()
        time.sleep(min(session.config.runtime.poll_interval_sec, 1.0))


def _wait_for_clear_scout(session: GameSession, snapshot: SceneSnapshot, deadline: float) -> SceneSnapshot:
    from .terrain import measure_camera_motion, measure_cloud_cover

    previous_clear = None
    while True:
        session.check_deadline()
        if time.monotonic() >= deadline:
            raise FlowError("Scout camera deadline exceeded before a clear stable map")
        _require_scene(snapshot, {"enemy_village"})
        cloud = measure_cloud_cover(snapshot.screenshot_path)
        motion = None
        if not cloud["obscured"] and previous_clear is not None:
            motion = measure_camera_motion(previous_clear.screenshot_path, snapshot.screenshot_path)
        ready = motion is not None and motion["stationary"] is True
        session.event("scout_camera_readiness", frame=str(snapshot.screenshot_path), cloud=cloud,
                      previous_frame=str(previous_clear.screenshot_path) if previous_clear is not None else None,
                      motion=motion, ready=ready)
        session.check_deadline()
        if time.monotonic() >= deadline:
            raise FlowError("Scout camera deadline exceeded before a clear stable map")
        if ready:
            return snapshot
        previous_clear = None if cloud["obscured"] else snapshot
        time.sleep(min(session.config.runtime.poll_interval_sec, 1, max(0, deadline-time.monotonic())))
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise FlowError("Scout camera deadline exceeded before a clear stable map")
        snapshot = session.wait_for({"enemy_village"}, timeout_sec=remaining, label="scout-camera-ready")


def _settlement(snapshot: SceneSnapshot) -> dict:
    _require_scene(snapshot, {"settlement"})
    result = snapshot.observations.get("settlement")
    if not isinstance(result, dict) or not result.get("evidence"):
        raise FlowError("Settlement has no recognition evidence")
    for group in ("loot", "bonus"):
        values = result.get(group)
        if not isinstance(values, dict) or any(type(values.get(resource)) is not int or values[resource] < 0 for resource in RESOURCES):
            raise FlowError(f"Settlement {group} resources are incomplete")
    for name, maximum in (("percentage", 100), ("stars", 3)):
        value = result.get(name)
        if type(value) is not int or not 0 <= value <= maximum:
            raise FlowError(f"Settlement {name} is invalid")
    return result


def _retain_last_frame(session: GameSession, evidence: list[Path]) -> None:
    if session.last_snapshot is not None and session.last_snapshot.screenshot_path not in evidence:
        evidence.append(session.last_snapshot.screenshot_path)
