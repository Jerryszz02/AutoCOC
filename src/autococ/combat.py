"""Run resource battles and verify deployment, settlement, and return inventory."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import math
from pathlib import Path
import time
import unicodedata

from .battle import BattleTarget, score_target
from .errors import AutoCOCError, CapabilityUnavailable, DeploymentError, FlowError
from .flow import return_to_village
from .reporting import TaskResult
from .resource_capacity import read_resource_capacity
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
        from .army_navigation import open_current_army
        army_frame = open_current_army(session, home, evidence)
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
    prior_battle_reward_receipt = getattr(session, "battle_reward_receipt", None)
    session._battle_reward_run_initial_receipt = prior_battle_reward_receipt
    evidence: list[Path] = []
    searches: list[dict[str, object]] = []
    metrics: dict[str, object] = {
        "searches": searches, "search_count": 0,
        "search_cost_gold": 0, "confirmed_search_cost_gold": 0,
        "search_cost_elixir": 0, "search_cost_dark_elixir": 0,
        "deployment": None, "settlement_observed": None, "inventory_reconciled": False,
        "preflight_rejections": [],
    }
    status, reason = "failed", "battle_verification_incomplete"
    settlement_error: str | None = None
    vision_mode = bool(getattr(session.config, "vision_agent", None) and session.config.vision_agent.enabled)
    custom_mode = bool(session.config.battle.strategy_file)
    line_mode = vision_mode or custom_mode or is_line_strategy(session.config.battle.strategy)
    # Freeze the opt-in skill choice before any search or deployment input.
    hero_skill_id = getattr(session.config.battle, "post_deployment_hero_skill", "")
    hero_skill_delay = getattr(session.config.battle, "hero_skill_delay_sec", 20)
    if hero_skill_id:
        metrics["hero_skill_plan"] = {"hero_id": hero_skill_id, "delay_sec": hero_skill_delay}
        session.post_deployment_skill_receipt = None
    vision_model = None
    vision_decider = None
    vision_guides = ()
    if vision_mode:
        session.prepared_battle_receipt = None
        metrics["vision_agent"] = {"mode": session.config.vision_agent.mode, "status": "not_started",
                                   "plan_ready": False, "input_sent": False,
                                   "consumption_verified": False, "burst_elapsed_sec": None,
                                   "burst_limit_sec": None, "input_count": 0,
                                   "burst_pass": False, "attempted_placements": 0}
    definition = None
    prepared_army = None
    task_buildings = {session.config.battle.target_building} if session.config.battle.target_building else set()
    required_buildings = set(task_buildings)
    # A new battle must never inherit a frozen continuous-deployment receipt
    # from a prior attempt if preparation is interrupted before deployment.
    session.frozen_line_receipt = None
    def building_types(targets):
        if "" in targets:
            raise CapabilityUnavailable("Targeted strategy needs an explicit building objective")
        return {name.split(":", 1)[0] for name in targets if not name.startswith("relative:")}
    def check_building_coverage(targets):
        if targets:
            from .building_vision import building_coverage
            coverage = building_coverage(client_version=getattr(session, "client_version", "unknown"))
            for target in targets:
                if coverage.get(target, {}).get("recognition", "unavailable") == "unavailable":
                    raise CapabilityUnavailable(f"Building recognition samples unavailable: {target}")
    def progress(phase):
        callback = getattr(session, "progress_callback", None)
        if callable(callback):
            callback(phase)
    metrics["strategy"] = session.config.battle.strategy
    try:
        if hero_skill_id and (vision_mode or custom_mode
                              or session.config.battle.strategy != "edrag_line"
                              or session.config.battle.line_execution_mode != "continuous"):
            raise CapabilityUnavailable("Post-deployment hero skill requires built-in model-free continuous edrag_line")
        if vision_mode:
            if session.config.vision_agent.mode != "continuous":
                raise CapabilityUnavailable("Battle enhancement mode has not passed action-specific acceptance")
            from .model_vision import BuildingModel, ModelUnavailable
            from .jev import JevDecider, load_guides
            setting = session.config.vision_agent
            directory = Path(setting.model_dir)
            if not setting.model_dir:
                raise CapabilityUnavailable("Building model directory is not configured")
            if not directory.is_absolute():
                directory = session.config.source_path.resolve().parent / directory
            try:
                if getattr(session, "vision_model_error", None):
                    raise ModelUnavailable(session.vision_model_error)
                vision_model = getattr(session, "prepared_building_model", None)
                if vision_model is None:
                    vision_model = BuildingModel(directory / "building", client_version=session.client_version)
                from .model_release import require_released_model
                require_released_model(vision_model)
                # Load and warm local inference before starting enemy search.
                import numpy as np
                applicability = vision_model.metadata["applicability"]
                source_width = max(1, int(applicability["source_width_range"][0]))
                source_height = max(1, int(applicability["source_height_range"][0]))
                vision_model.detect(np.zeros((source_height, source_width, 3), dtype=np.uint8),
                                    captured_at=time.monotonic())
            except (ModelUnavailable, OSError, ValueError) as exc:
                raise CapabilityUnavailable(f"Building model unavailable: {exc}") from exc
            from .prepared_battle import validate_prepared_capability
            validate_prepared_capability(session, vision_model)
            vision_decider = JevDecider(enabled=setting.jev_enabled, model=setting.jev_model,
                                       timeout_sec=setting.jev_timeout_sec)
            if setting.guide_file:
                guide_path = Path(setting.guide_file)
                if not guide_path.is_absolute():
                    guide_path = session.config.source_path.resolve().parent / guide_path
                vision_guides = load_guides(guide_path)
        if custom_mode:
            from .strategy_config import load_strategy
            path = Path(session.config.battle.strategy_file)
            if not path.is_absolute():
                path = session.config.source_path.resolve().parent / path
            definition = load_strategy(path)
            metrics["strategy"] = definition.id
        check_building_coverage(required_buildings)
        needs_terrain = definition is None or any(step.action in {"deploy_troop", "deploy_hero", "deploy_siege"}
                          or step.action == "cast_spell" and step.target.startswith("relative:") for step in definition.steps)
        if (line_mode and needs_terrain or required_buildings) and session.native is None:
            raise FlowError("Two-edge minimum zoom requires the MuMu native transport")
        home = return_to_village(session)
        if home.observations.get("village_type") == "builder_base":
            raise CapabilityUnavailable("Main Home Village is required; Builder Base is outside this task")
        if definition is not None and definition.recipe is not None:
            from .army_control import ensure_army
            progress("应用并核对配兵")
            prepared = ensure_army(session, definition.recipe)
            evidence.extend(prepared.evidence)
            metrics["army_preparation"] = {"status": prepared.status, "reason": prepared.reason, **prepared.metrics}
            if prepared.status == "skipped":
                mutations = prepared.metrics.get("recipe_mutations")
                if type(mutations) is not int or mutations != 0:
                    raise FlowError("Unavailable army result followed an uncertain recipe mutation")
                home = return_to_village(session)
                evidence.append(home.screenshot_path)
                metrics["returned_home"] = True
                return TaskResult("battle", "skipped", prepared.reason,
                                  started_at, time.monotonic() - started, evidence, metrics)
            if prepared.status != "succeeded":
                if prepared.status == "not_supported":
                    raise CapabilityUnavailable(prepared.reason)
                raise FlowError(prepared.reason)
            prepared_army = {"status": prepared.status,
                             "observed": prepared.metrics.get("observed"),
                             "frame": str(prepared.evidence[-1])}
            home = return_to_village(session)
        if definition is not None:
            from .strategy_execution import declared_building_targets
            targets = declared_building_targets(definition, session.config.battle.target_building,
                                                prepared_army=prepared_army)
            required_buildings = building_types(task_buildings | targets)
            check_building_coverage(required_buildings)
            if required_buildings and session.native is None:
                raise FlowError("Targeted strategy minimum zoom requires the MuMu native transport")
        evidence.append(home.screenshot_path)
        metrics["resources_before"] = home.observations.get("resources")
        verified_before = None
        if line_mode:
            before = home.observations.get("resources")
            if not isinstance(before, dict):
                before = {}
            try:
                verified_before = _inventory(home)
            except FlowError:
                pass
        else:
            before = _inventory(home)
        session.click(home, "attack")
        menu = session.wait_for({"search"}, timeout_sec=30, label="battle-menu")
        evidence.append(menu.screenshot_path)
        pending_cost = _search_price(menu)
        session.click(menu, "find_match_regular")
        army_frame = session.wait_for({"training"}, timeout_sec=45, label="battle-army-confirmation")
        evidence.append(army_frame.screenshot_path)
        army = _read_army(army_frame)
        if getattr(session.config.battle, "line_execution_mode", "verified") == "continuous":
            army_frame, army = _read_continuous_line_army(session, army_frame, evidence)
        metrics["army"] = army
        if definition is None and (not army["troops_full"] or not line_mode and not army["spells_full"]):
            raise FlowError("Army troop/spell recipe is not full")

        total_search_cost = 0
        metrics["search_cost_gold"] = None
        session.click(army_frame, "attack")
        progress("正在筛选对手")
        scout = _next_candidate(session)
        probed_rescout = (not vision_mode and definition is None
                          and session.config.battle.strategy == "edrag_line"
                          and session.config.battle.line_execution_mode == "continuous"
                          and session.config.battle.line_tactics == "two_edge_probed_anchors")
        cached_deployment = None
        cached_deployment_error = None
        while True:
            if definition is not None and prepared_army is not None:
                targets = declared_building_targets(definition, session.config.battle.target_building,
                                                    prepared_army=prepared_army, scout=scout)
                try:
                    required_buildings = building_types(task_buildings | targets)
                    check_building_coverage(required_buildings)
                    if required_buildings and session.native is None:
                        raise CapabilityUnavailable("Targeted strategy minimum zoom requires the MuMu native transport")
                except CapabilityUnavailable as exc:
                    # A battle-only card can contradict the earlier home-army
                    # absence proof. Leave this unstarted match before reporting it.
                    _leave_scout(session, scout)
                    home = return_to_village(session)
                    evidence.append(home.screenshot_path)
                    metrics["returned_home"] = True
                    return TaskResult("battle", "not_supported", str(exc),
                                      started_at, time.monotonic() - started, evidence, metrics)
            if required_buildings:
                scout = _prepare_building_view(session, scout)
            evidence.append(scout.screenshot_path)
            total_search_cost += pending_cost
            metrics["search_cost_gold"] = total_search_cost
            metrics["confirmed_search_cost_gold"] = total_search_cost
            available = scout.observations.get("resources", {})
            if not isinstance(available, dict) or not line_mode and scout.observations.get("resource_source") != "enemy_available":
                raise FlowError("Enemy available resources were not identified")
            target = BattleTarget(gold=available.get("gold"), elixir=available.get("elixir"),
                                  dark_elixir=available.get("dark_elixir"))
            decision = score_target(target, session.config.battle, searches=len(searches) + 1)
            rule = session.config.battle.resource_filter
            if rule is not None and rule.enabled and scout.observations.get("resource_source") != "enemy_available":
                decision = replace(decision, should_attack=False,
                    can_search_next=len(searches) + 1 < session.config.battle.max_searches,
                    reasons=("enemy available resource source unverified",))
            for required_building in required_buildings:
                buildings = scout.observations.get("buildings", [])
                matches = [b for b in buildings if b.get("type") == required_building and b.get("state") == "alive"
                           and b.get("frame") == str(scout.screenshot_path)
                           and isinstance(b.get("confidence"), (int, float)) and b["confidence"] >= .9]
                if not matches:
                    decision = replace(decision, should_attack=False,
                        can_search_next=len(searches) + 1 < session.config.battle.max_searches,
                        reasons=decision.reasons + ("required building not positively identified",))
            search = {"candidate": len(searches) + 1, "frame": str(scout.screenshot_path),
                      "search_cost_gold": pending_cost, "decision": asdict(decision)}
            searches.append(search)
            metrics["search_count"] = len(searches)
            session.event("battle_candidate", **search)
            if decision.should_attack:
                metrics["selected_target"] = asdict(target)
                if probed_rescout:
                    # The paid search loop owns the bounded next-target action.
                    # Invoke deployment once per candidate and retain its result
                    # or error, so the shared cleanup path never repeats it.
                    scout.observations["expected_army_manifest"] = army["recipe"]["manifest"]
                    scout.observations["expected_hero_manifest"] = army["recipe"].get("hero_evidence")
                    goal_progress = getattr(session, "goal_progress", None)
                    scout.observations["target_progress"] = goal_progress if isinstance(goal_progress, dict) else {}
                    progress("执行打法")
                    from .deployment import deploy_army
                    try:
                        cached_deployment = deploy_army(session, scout)
                    except DeploymentError as exc:
                        cached_deployment_error = exc
                        if not _retryable_boundary_preflight(session, exc):
                            break
                        rejection = {"candidate": len(searches), "frame": str(scout.screenshot_path),
                                     "reason": str(exc), "phase": exc.partial_receipt["phase"],
                                     "search_cost_gold": pending_cost,
                                     "evidence": list(exc.partial_receipt.get("evidence", []))}
                        search["preflight_rejection"] = rejection
                        metrics["preflight_rejections"].append(rejection)
                        session.event("battle_preflight_rejected", **rejection)
                        if len(searches) >= session.config.battle.max_searches:
                            break  # Keep the last deployment failure and its cleanup.
                        try:
                            fresh = _fresh_boundary_retry_preview(session, evidence)
                            if fresh is None:
                                break
                            pending_cost = _search_price(fresh)
                        except AutoCOCError as retry_error:
                            metrics["preflight_retry_error"] = str(retry_error)
                            if "account_in_use" in str(retry_error):
                                raise
                            break
                        metrics["search_cost_gold"] = None
                        metrics["next_search_cost_unconfirmed"] = True
                        session.click(fresh, "next")
                        scout = _next_candidate(session, previous=fresh)
                        cached_deployment_error = None
                        metrics["next_search_cost_unconfirmed"] = False
                        metrics.pop("selected_target", None)
                        continue
                break
            if not decision.can_search_next:
                _leave_scout(session, scout)
                home = return_to_village(session)
                evidence.append(home.screenshot_path)
                metrics["returned_home"] = True
                return TaskResult("battle", "limited", "search_limit_reached_without_target",
                                  started_at, time.monotonic() - started, evidence, metrics)
            pending_cost = _search_price(scout)
            metrics["search_cost_gold"] = None
            session.click(scout, "next")
            scout = _next_candidate(session, previous=scout)

        from .deployment import deploy_army

        scout.observations["expected_army_manifest"] = army["recipe"]["manifest"]
        scout.observations["expected_hero_manifest"] = army["recipe"].get("hero_evidence")
        goal_progress = getattr(session, "goal_progress", None)
        scout.observations["target_progress"] = goal_progress if isinstance(goal_progress, dict) else {}
        progress("执行打法")
        try:
            if vision_mode:
                from .prepared_battle import run_prepared_battle
                try:
                    deployed = run_prepared_battle(session, scout, model=vision_model,
                                                   decider=vision_decider, guides=vision_guides)
                except AutoCOCError as exc:
                    if session.prepared_battle_receipt is None:
                        # Preparation may have changed the camera or exhausted
                        # the preview timer. Never exit using the old scout.
                        metrics["preparation_error"] = str(exc)
                        try:
                            current = session.observe("preparation-failure-check")
                            evidence.append(current.screenshot_path)
                            if current.scene == "enemy_village":
                                _require_scene(current, {"enemy_village"})
                                _leave_scout(session, current)
                                home = return_to_village(session)
                            else:
                                _require_scene(current, {"battle", "settlement"})
                                timeout = (session.config.battle.settlement_timeout_sec
                                           or session.config.battle.deploy_timeout_sec)
                                settled = current if current.scene == "settlement" else session.wait_for(
                                    {"settlement"}, timeout_sec=timeout + 35,
                                    label="preparation-failure-settlement",
                                    purpose="settlement", poll_interval_sec=5)
                                _require_scene(settled, {"settlement"})
                                evidence.append(settled.screenshot_path)
                                metrics["settlement_observed"] = settled.observations.get("settlement")
                                home = return_to_village(session, initial_snapshot=settled)
                                stars = _verified_stars(metrics["settlement_observed"])
                                metrics.update(rounds_completed=1,
                                               victory=stars > 0 if stars is not None else None)
                            evidence.append(home.screenshot_path)
                            metrics["returned_home"] = True
                        except AutoCOCError as recovery_error:
                            metrics["preparation_recovery_error"] = str(recovery_error)
                    # Preserve the preparation failure even after a safe return.
                    # StopRequested bypasses both handlers and sends no input.
                    raise
            elif definition is None:
                if cached_deployment_error is not None:
                    raise cached_deployment_error
                deployed = cached_deployment if cached_deployment is not None else deploy_army(session, scout)
            else:
                from .strategy_execution import execute_strategy
                deployed = execute_strategy(session, scout, definition,
                                            prepared_army=prepared_army)
        except DeploymentError as exc:
            metrics["deployment"] = exc.partial_receipt
            if exc.partial_receipt.get("interruption") == "account_in_use":
                raise
            last = session.last_snapshot
            units = exc.partial_receipt.get("deployed_units")
            issued = exc.partial_receipt.get("issued_placements", 0)
            offensive = exc.partial_receipt.get("offensive_actions", units)
            verified_partial = exc.partial_receipt.get("verified") is True and type(offensive) is int and offensive > 0
            attempted = exc.partial_receipt.get("attempted_placements", 0)
            attempted_line = line_mode and (type(issued) is int and issued > 0 or
                                           type(attempted) is int and attempted > 0)
            if attempted_line and last is None and exc.partial_receipt.get("interruption") != "account_in_use":
                # A failed capture clears GameSession.last_snapshot. The old
                # preview must never authorize recovery input after a probe.
                # Keep the original partial receipt and never retry placement.
                for recovery_attempt in range(2):
                    try:
                        session.check_deadline()
                    except AutoCOCError as recovery_error:
                        metrics["deployment_recovery_error"] = str(recovery_error)
                        break
                    if time.monotonic() >= min(session.deadline, session.task_deadline):
                        break
                    try:
                        fresh = session.observe("partial-deployment-recovery")
                        evidence.append(fresh.screenshot_path)
                        session.event("partial_deployment_recovery", attempt=recovery_attempt + 1,
                                      frame=str(fresh.screenshot_path), scene=fresh.scene)
                    except AutoCOCError as recovery_error:
                        metrics["deployment_recovery_error"] = str(recovery_error)
                        session.event("partial_deployment_recovery", attempt=recovery_attempt + 1,
                                      error=str(recovery_error))
                        if "account_in_use" in str(recovery_error):
                            raise exc
                        continue
                    if fresh.scene == "account_in_use":
                        metrics["deployment_recovery_error"] = "account_in_use"
                        raise exc
                    if (fresh.scene in {"battle", "settlement", "enemy_village"} and
                            math.isfinite(fresh.confidence) and fresh.confidence >= .8):
                        last = fresh
                        break
                if last is not None and last.scene == "enemy_village":
                    # Only this newly recognized preview authorizes the same
                    # bounded scout-exit path used by zero-input failures.
                    try:
                        _require_scene(last, {"enemy_village"})
                        _leave_scout(session, last)
                        home = return_to_village(session)
                        evidence.append(home.screenshot_path)
                        metrics["returned_home"] = True
                    except AutoCOCError as recovery_error:
                        metrics["deployment_recovery_error"] = str(recovery_error)
                    metrics["deployment_error"] = str(exc)
                    raise exc
            unstarted_continuous_line = (
                line_mode and not vision_mode
                and getattr(session.config.battle, "line_execution_mode", "verified") == "continuous"
                and type(issued) is int and issued == 0
                and type(attempted) is int and attempted == 0
                and type(units) is int and units == 0
                and type(offensive) is int and offensive == 0
                and exc.partial_receipt.get("input_sent") is not True
                and not verified_partial)
            if unstarted_continuous_line:
                # Freezing may consume the preview without placing anything.
                # The old scout cannot authorize an exit after that delay.
                metrics["deployment_error"] = str(exc)
                try:
                    current = session.observe("continuous-line-failure-check")
                    evidence.append(current.screenshot_path)
                    if current.scene == "enemy_village":
                        _require_scene(current, {"enemy_village"})
                        _leave_scout(session, current)
                        home = return_to_village(session)
                        evidence.append(home.screenshot_path)
                        metrics["returned_home"] = True
                    elif current.scene == "battle":
                        _require_scene(current, {"battle"})
                        last = current
                    else:
                        raise FlowError("Continuous line failure scene is not verified for cleanup")
                except AutoCOCError as recovery_error:
                    metrics["deployment_recovery_error"] = str(recovery_error)
                    raise exc
                if metrics.get("returned_home") is True:
                    raise
            # Preparation can consume the preview timer before any placement.
            # A positively observed battle still needs settlement/home cleanup;
            # it never makes this failed deployment a successful receipt.
            expired_preview = (line_mode and last is not None and last.scene == "battle"
                               and math.isfinite(last.confidence) and last.confidence >= .8)
            if (last is None or not (verified_partial or attempted_line or expired_preview)
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
        if vision_mode and isinstance(deployment, dict):
            metrics["vision_agent"].update(
                status="verified" if deployment.get("verified") is True else "incomplete",
                plan_ready=bool(deployment.get("plan_id")),
                input_sent=deployment.get("input_sent") is True,
                consumption_verified=deployment.get("verified") is True,
                burst_pass=deployment.get("burst_pass") is True,
                attempted_placements=deployment.get("attempted_placements", 0),
                burst_elapsed_sec=deployment.get("burst_elapsed_sec"),
                burst_limit_sec=deployment.get("burst_limit_sec"),
                input_count=deployment.get("input_count", 0))
            metrics["vision_agent"]["decision"] = {
                "candidate_id": deployment.get("candidate_id"),
                "source": deployment.get("decision_source"),
                "reason": deployment.get("decision_reason"),
                "model": deployment.get("decision_model"),
                "elapsed_sec": deployment.get("decision_elapsed_sec"),
                "confidence": deployment.get("decision_confidence")}
        _require_scene(deployed, {"enemy_village", "battle", "settlement"})
        cleanup_only = line_mode and ("deployment_error" in metrics or vision_mode and
                                      isinstance(deployment, dict) and deployment.get("verified") is not True)
        if not isinstance(deployment, dict) or not cleanup_only and deployment.get("verified") is not True:
            raise FlowError("Deployment has no verified receipt")
        positive_actions = deployment.get("offensive_actions") if custom_mode else deployment.get("deployed_units")
        if (type(positive_actions) is not int or positive_actions < 0
                or positive_actions == 0 and not cleanup_only):
            raise FlowError("No positive troop deployment was verified")
        deployment_evidence = deployment.get("evidence")
        if not isinstance(deployment_evidence, list) or not deployment_evidence:
            raise FlowError("Deployment receipt has no screenshot evidence")
        evidence.extend(Path(path) for path in deployment_evidence)

        skill_failure_needs_settlement_wait = False
        if hero_skill_id:
            if not cleanup_only and deployment.get("completed") is True:
                progress("核验并使用英雄技能")
                try:
                    from .post_deployment_skill import execute_post_deployment_hero_skill
                    deployed, skill_receipt = execute_post_deployment_hero_skill(
                        session, deployed, hero_id=hero_skill_id, delay_sec=hero_skill_delay)
                except Exception as exc:
                    # A failed optional skill cannot abandon an already deployed
                    # army. StopRequested derives from KeyboardInterrupt and
                    # bypasses this handler; account conflicts also stop inputs.
                    previous = getattr(session, "post_deployment_skill_receipt", None)
                    skill_receipt = dict(previous) if isinstance(previous, dict) else {}
                    skill_receipt.update(hero_id=hero_skill_id, verified=False, completed=False,
                                         status="failed", reason="unexpected_skill_error",
                                         error=f"{type(exc).__name__}: {exc}")
                    skill_receipt.setdefault("evidence", [])
                    session.post_deployment_skill_receipt = skill_receipt
                    metrics["hero_skill"] = skill_receipt
                    latest = session.last_snapshot
                    if "account_in_use" in str(exc) or (latest is not None and latest.scene == "account_in_use"):
                        skill_receipt.update(reason="account_in_use", interruption="account_in_use")
                        raise FlowError("Game interruption: account_in_use") from exc
                    session.check_deadline()
                    # Capture failure may have cleared last_snapshot. Never use
                    # an old settlement frame to authorize the return input.
                    skill_failure_needs_settlement_wait = True
                metrics["hero_skill"] = skill_receipt
                evidence.extend(Path(path) for path in skill_receipt.get("evidence", []))
            else:
                metrics["hero_skill"] = {"hero_id": hero_skill_id, "verified": False,
                                         "completed": False, "status": "skipped",
                                         "reason": "deployment_incomplete", "evidence": []}

        progress("等待结算" if deployment.get("completed") is True
                 else "投放核验失败，等待结算")
        settlement_timeout = session.config.battle.settlement_timeout_sec or session.config.battle.deploy_timeout_sec
        settlement_frame = deployed if deployed.scene == "settlement" and not skill_failure_needs_settlement_wait else session.wait_for(
            {"settlement"}, timeout_sec=settlement_timeout + (35 if cleanup_only else 0), label="battle-settlement",
            **({"purpose": "settlement", "poll_interval_sec": 5} if line_mode else {}))
        settlement = None
        reads: list[dict[str, object]] = []
        metrics["settlement_reads"] = reads
        read_started = time.monotonic()
        read_deadline = read_started + (35 if line_mode else 10)
        prior_line_values = None
        stable_line_result = False
        max_settlement_reads = 12 if line_mode else 4
        for attempt in range(max_settlement_reads):  # Every reread must be a new saved frame.
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
                settlement_error = str(exc) if line_mode else settlement_error or str(exc)
                prior_line_values = None
            else:
                reading["valid"] = True
                if not line_mode:
                    settlement_error = None
                    break
                # The result screen counts up its percentage and loot after
                # appearing. A single valid OCR read can still be transitional.
                values = (settlement["percentage"], settlement["stars"],
                          tuple(settlement[group][name] for group in ("loot", "bonus")
                                for name in RESOURCES))
                if (prior_line_values is not None and prior_line_values[0] == values
                        and prior_line_values[1] != str(settlement_frame.screenshot_path)):
                    stable_line_result = True
                    settlement_error = None
                    break
                prior_line_values = (values, str(settlement_frame.screenshot_path))
                settlement_error = "Settlement values have not stabilized"
            if attempt == max_settlement_reads - 1 or time.monotonic() >= read_deadline:
                break
            session.check_deadline()
            time.sleep(min(session.config.runtime.poll_interval_sec, 1, max(0, read_deadline - time.monotonic())))
            session.check_deadline()
            if time.monotonic() >= read_deadline:
                break
            settlement_frame = session.observe("battle-settlement-reread",
                                               **({"purpose": "settlement"} if line_mode else {}))
        metrics["settlement_validation_error"] = settlement_error
        # An unsupported or unstable result can still be left safely through
        # the current positively recognized settlement's return control. It
        # never authorizes a result, reward, or successful task. Never act on
        # an older settlement if the newest observation is another scene.
        session.check_deadline()
        _require_scene(settlement_frame, {"settlement"})
        progress("结算回村")
        if isinstance(settlement_frame.observations.get("battle_reward_claim"), dict):
            # A chest-bearing result has a distinct claim control.  Its free
            # chain must finish on new evidence before the old battle ledger
            # can be reconciled; failed deployment remains failed below.
            if settlement_error is not None or settlement is None or line_mode and not stable_line_result:
                raise FlowError(settlement_error or "Battle chest settlement was not stable")
            from .battle_rewards import finish_claimed_battle_reward
            home_after, metrics["battle_reward"] = finish_claimed_battle_reward(
                session, settlement_frame, evidence)
        else:
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
        home_after = return_to_village(session, initial_snapshot=home_after)
        if home_after.screenshot_path not in evidence:
            evidence.append(home_after.screenshot_path)
        if isinstance(metrics.get("battle_reward"), dict) and metrics["battle_reward"].get("returned_home") is True:
            metrics["returned_home"] = True
        if line_mode and not stable_line_result:
            metrics.update(returned_home=True, rounds_completed=1,
                           resources_after=home_after.observations.get("resources"), victory=None)
            raise FlowError(settlement_error or "Settlement values could not be confirmed stable")
        # A finished battle is independent of whether every planned action or
        # resource reconciliation succeeded. Both its settlement and safe home
        # are now observed; keep that fact even when later checks fail.
        provisional_outcome = line_mode and _settlement_requires_inventory_reconciliation(settlement)
        observed_stars = None if provisional_outcome else _verified_stars(metrics["settlement_observed"])
        metrics.update(returned_home=True, rounds_completed=1, victory=None)
        if vision_mode:
            progress("战斗完成")
        metrics["resources_after"] = home_after.observations.get("resources")
        if line_mode:
            observed = metrics["settlement_observed"] or {}
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
            if settlement_error is None and settlement is not None and verified_before is not None:
                # A failed deployment can still have an independently verified
                # result and home inventory. Record that ledger evidence before
                # preserving the deployment failure below. Provisional zeros
                # from an absent settlement row become reportable only after
                # the exact resource delta agrees.
                try:
                    verified_after = _inventory(home_after)
                except FlowError:
                    pass
                else:
                    ledger, home_after = _reconcile_return_inventory(
                        session, home_after, verified_before, verified_after, settlement,
                        total_search_cost, evidence, allow_capacity=not provisional_outcome)
                    metrics.update(ledger)
            if not metrics["inventory_reconciled"]:
                raise FlowError("Provisional settlement rewards need exact inventory reconciliation" if provisional_outcome
                                else "Battle settlement does not reconcile with return-home inventory")
            for group, prefix in (("loot", "loot"), ("bonus", "bonus")):
                for resource in RESOURCES:
                    proof = (observed.get("evidence") or {}).get(group) or {}
                    proof = proof.get(resource, proof)
                    metrics[f"{prefix}_{resource}"] = (None if not proof or
                                                       proof.get("requires_inventory_reconciliation") and
                                                       not metrics["inventory_reconciled"] else
                                                       (observed.get(group) or {}).get(resource))
            stars = _verified_stars(observed) if provisional_outcome else observed_stars
            metrics.update(stars=stars, percentage=observed.get("percentage"),
                           victory=stars > 0 if type(stars) is int else None)
            if deployment.get("completed") is not True:
                raise FlowError(metrics.get("deployment_error") or "Line troop/hero deployment is incomplete")
            if hero_skill_id and (metrics["hero_skill"].get("verified") is not True
                                  or metrics["hero_skill"].get("completed") is not True):
                raise FlowError("Configured post-deployment hero skill was not verified: "
                                + str(metrics["hero_skill"].get("reason", "unknown")))
            if vision_mode:
                metrics["vision_agent"]["status"] = "battle_completed"
            session.event("battle_round_verified", settlement_frame=str(settlement_frame.screenshot_path),
                          home_frame=str(home_after.screenshot_path), victory=metrics["victory"])
            completed_reason = ("strategy_actions_settlement_and_return_verified" if custom_mode else
                                f"{session.config.battle.strategy}_troops_settlement_and_return_verified")
            return TaskResult("battle", "succeeded", completed_reason,
                              started_at, time.monotonic() - started, evidence, metrics)
        if settlement_error is not None:
            raise FlowError(settlement_error)
        after = _inventory(home_after)
        for resource in RESOURCES:
            metrics[f"loot_{resource}"] = settlement["loot"][resource]
            metrics[f"bonus_{resource}"] = settlement["bonus"][resource]
        metrics.update({"stars": settlement["stars"], "percentage": settlement["percentage"],
                        "victory": settlement["stars"] > 0})
        ledger, home_after = _reconcile_return_inventory(
            session, home_after, before, after, settlement, total_search_cost, evidence, allow_capacity=True)
        metrics.update(ledger)
        if after["gems"] != before["gems"]:
            raise FlowError("Gem balance changed during battle")
        if not metrics["inventory_reconciled"]:
            raise FlowError("Battle settlement does not reconcile with return-home inventory")
        if deployment.get("completed") is not True:
            raise FlowError(metrics.get("deployment_error") or "Army or hero ability deployment is incomplete")
        session.event("battle_verified", settlement_frame=str(settlement_frame.screenshot_path),
                      home_frame=str(home_after.screenshot_path), inventory_delta=metrics["inventory_delta"])
        metrics.update(returned_home=True, rounds_completed=1)
        status, reason = "succeeded", "deployment_settlement_and_return_inventory_verified"
    except (AutoCOCError, OSError) as exc:
        # Report-storage failure must not discard an already observed settlement
        # and return. The caller still stops on failure while retaining its count.
        if isinstance(exc, CapabilityUnavailable):
            status = "not_supported"
        reason = metrics.get("deployment_error") or settlement_error or str(exc)
        chest_receipt = getattr(session, "battle_reward_receipt", None)
        if (chest_receipt is not prior_battle_reward_receipt and isinstance(chest_receipt, dict)
                and chest_receipt.get("kind") == "battle_chest_ui"):
            metrics["battle_reward"] = chest_receipt
        if vision_mode and metrics["vision_agent"]["status"] == "not_started":
            metrics["vision_agent"]["status"] = "unavailable" if isinstance(exc, CapabilityUnavailable) else "failed"
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


def _retryable_boundary_preflight(session: GameSession, error: DeploymentError) -> bool:
    """Only a failed boundary scan proves no probe or placement was attempted."""
    receipt = error.partial_receipt
    return (str(error) == "Two current deployment boundaries could not be verified"
            and receipt.get("phase") == "boundary_scan"
            and all(type(receipt.get(name)) is int and receipt[name] == 0 for name in
                    ("attempted_placements", "issued_placements", "offensive_actions", "deployed_units"))
            and receipt.get("consumption_events") == []
            and receipt.get("terrain_evidence") == []
            and getattr(session, "frozen_line_receipt", None) is None)


def _fresh_boundary_retry_preview(session: GameSession, evidence: list[Path]) -> SceneSnapshot | None:
    """A new full frame must still prove a live, unstarted preview and Next."""
    session.check_deadline()
    current = session.observe("boundary-rescout-check")
    evidence.append(current.screenshot_path)
    if current.scene == "account_in_use":
        raise FlowError("Game interruption: account_in_use")
    if current.scene != "enemy_village" or not math.isfinite(current.confidence) or current.confidence < .8:
        return None
    battle = current.observations.get("battle")
    if not isinstance(battle, dict) or battle.get("phase") != "scout":
        return None
    from .prepared_battle import _remaining_seconds
    try:
        seconds = _remaining_seconds(current)
    except FlowError:
        return None
    captured_at = current.observations.get("observed_at_monotonic")
    if (type(captured_at) not in (int, float) or not math.isfinite(captured_at)
            or time.monotonic() + 2 >= captured_at + seconds - 1):
        return None
    buttons = current.observations.get("buttons")
    if (not isinstance(buttons, list) or
            any(sum(isinstance(item, dict) and item.get("name") == name for item in buttons) != 1
                for name in ("next", "end_battle"))):
        return None
    session.check_deadline()
    return current


def _leave_scout(session, scout):
    """Exit an unstarted match only through recognized navigation and an anchored prompt."""
    _require_scene(scout, {"enemy_village"})
    session.click(scout, "end_battle")
    current = session.wait_for({"village", "search", "popup"}, timeout_sec=45, label="search-limit-exit")
    if current.scene == "popup":
        prompts = [item for item in current.observations.get("ocr", [])
                   if item.get("confidence", 0) >= .95 and
                   item.get("text", "").replace(" ", "").rstrip("?？!！") in
                   {"确定要放弃战斗吗", "确定要结束战斗吗", "结束战斗", "放弃战斗"}]
        if len(prompts) != 1:
            raise FlowError("Unrecognized search exit prompt")
        session.click(current, "confirm")
        session.wait_for({"village", "search"}, timeout_sec=20, label="search-exit-home")


def _prepare_building_view(session, current):
    """Calibrated building samples require a stable minimum-scale enemy map."""
    from .terrain import measure_camera_motion
    _require_scene(current, {"enemy_village"})
    for _ in range(3):
        session._validate_snapshot(current)
        session.native.zoom_out()
        session.action_count += 1
        time.sleep(.5)
    time.sleep(1)
    before = session.observe("building-zoom-before")
    _require_scene(before, {"enemy_village"})
    session._validate_snapshot(before)
    session.native.zoom_out()
    session.action_count += 1
    time.sleep(1)
    current = session.observe("building-zoom-verified")
    _require_scene(current, {"enemy_village"})
    motion = measure_camera_motion(before.screenshot_path, current.screenshot_path)
    session.event("building_minimum_zoom", frame=str(current.screenshot_path), motion=motion)
    if motion.get("stationary") is not True:
        raise FlowError("Building search camera did not stabilize at minimum zoom")
    return current


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


def _reconcile_return_inventory(
    session: GameSession, home: SceneSnapshot, before: dict[str, int], after: dict[str, int],
    settlement: dict, search_cost: int, evidence: list[Path], *, allow_capacity: bool,
) -> tuple[dict[str, object], SceneSnapshot]:
    """Use a capacity only for a positive shortfall confirmed by its own HUD detail."""
    gross = {name: settlement["loot"][name] + settlement["bonus"][name] for name in RESOURCES}
    base = {name: before[name] - (search_cost if name == "gold" else 0) for name in RESOURCES}
    expected = {name: base[name] + gross[name] for name in RESOURCES}
    delta = {name: after[name] - before[name] for name in (*RESOURCES, "gems")}
    result: dict[str, object] = {
        "inventory_delta": delta,
        "expected_inventory_delta": {name: expected[name] - before[name] for name in RESOURCES},
        "inventory_reconciled": False,
    }
    if delta["gems"] != 0:
        return result, home
    mismatched = [name for name in RESOURCES if after[name] != expected[name]]
    capacities: dict[str, int] = {}
    capacity_proofs: dict[str, object] = {}
    if mismatched and allow_capacity and getattr(session, "client_version", None) == "18.647.2" and all(
        gross[name] > 0 and base[name] <= after[name] < expected[name] for name in mismatched
    ):
        for name in mismatched:
            proof, home, frames = read_resource_capacity(session, home, name)
            evidence.extend(frames)
            capacities[name] = proof["capacity"]
            capacity_proofs[name] = proof
        result["resource_capacity_evidence"] = capacity_proofs
        result["resources_after"] = home.observations.get("resources")
    if any(after[name] != min(expected[name], capacities.get(name, expected[name])) for name in RESOURCES):
        return result, home
    if any(capacities[name] < base[name] or capacities[name] >= expected[name] for name in capacities):
        return result, home
    result["inventory_reconciled"] = True
    result["inventory_reconciliation_method"] = "capacity_capped" if capacities else "exact"
    for name in RESOURCES:
        credited = after[name] - base[name]
        result[f"battle_credited_{name}"] = credited
        result[f"overflow_{name}"] = gross[name] - credited
    return result, home


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


def _verified_army_identities(rows, kind, client_version):
    if (not isinstance(client_version, str) or client_version in {"", "unknown"}
            or not isinstance(rows, list) or not rows):
        return False
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        unit_id = row.get("unit_id")
        evidence = row.get("evidence")
        identity = evidence.get("identity") if isinstance(evidence, dict) else None
        if (not isinstance(unit_id, str) or not unit_id or unit_id in seen
                or row.get("kind") != kind or row.get("source") != "army"
                or not isinstance(identity, dict)
                or identity.get("unit_id") != unit_id
                or identity.get("version_verified") is not True
                or identity.get("template_client_version") != client_version
                or type(identity.get("confidence")) not in (int, float)
                or not math.isfinite(identity["confidence"])
                or not .9 <= identity["confidence"] <= 1):
            return False
        seen.add(unit_id)
    return True


def _read_continuous_line_army(session, current, evidence):
    """Resolve troop and hero identities before paying, using fresh pixels only."""
    for attempt in range(3):
        army = _read_army(current)
        manifest = army["recipe"].get("manifest")
        troops = manifest.get("troops") if isinstance(manifest, dict) else None
        heroes = army["recipe"].get("hero_evidence")
        rows = heroes.get("cards") if isinstance(heroes, dict) else None
        if (isinstance(heroes, dict) and heroes.get("complete") is True and not heroes.get("unknowns")
                and heroes.get("frame") == str(current.screenshot_path)
                and _verified_army_identities(troops, "troop", session.client_version)
                and _verified_army_identities(rows, "hero", session.client_version)):
            return current, army
        if attempt < 2:
            session.check_deadline()
            current = session.observe("battle-army-roster-reread")
            evidence.append(current.screenshot_path)
    home = return_to_village(session, initial_snapshot=current)
    evidence.append(home.screenshot_path)
    raise FlowError("Continuous line troop or hero roster is incomplete; no paid search started")


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
    independent_filter = (is_line_strategy(session.config.battle.strategy)
                          or session.config.battle.resource_filter is not None
                          or bool(session.config.battle.strategy_file))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FlowError("Next target did not become distinguishable from the previous candidate")
        snapshot = session.wait_for({"enemy_village"}, timeout_sec=remaining, label="battle-candidate")
        _require_scene(snapshot, {"enemy_village"})
        if independent_filter and (previous is None or _candidate_changed(previous, snapshot)):
            stable = _wait_for_clear_scout(session, snapshot, deadline)
            # Only enabled thresholds decide which resources must be readable.
            # A newly read field alone does not prove a paid target transition.
            if previous is None or _candidate_changed(previous, stable):
                return stable
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
    if _verified_stars(result) is None:
        raise FlowError("Settlement stars have no independent recognition evidence")
    return result


def _settlement_requires_inventory_reconciliation(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return False
    bonus = evidence.get("bonus")
    loot = evidence.get("loot")
    return any(
        isinstance(group, dict) and (group.get("requires_inventory_reconciliation") is True or
                                     any(isinstance(group.get(resource), dict) and
                                         group[resource].get("requires_inventory_reconciliation") is True
                                         for resource in RESOURCES))
        for group in (loot, bonus))


def _verified_stars(result: dict | None) -> int | None:
    if not isinstance(result, dict):
        return None
    stars = result.get("stars")
    proof = (result.get("evidence") or {}).get("stars")
    if type(stars) is not int or not 0 <= stars <= 3 or not isinstance(proof, dict):
        return None
    label = proof.get("result")
    confidence = label.get("confidence") if isinstance(label, dict) else None
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not .9 <= confidence <= 1:
        return None
    source = proof.get("source")
    if stars == 0:
        percentage = result.get("percentage")
        return 0 if (source == "explicit_defeat_result" and type(percentage) is int and
                     0 <= percentage < 50) else None
    shapes = proof.get("shapes")
    if (source == "explicit_victory_star_shapes" and isinstance(shapes, list) and
            len(shapes) == stars and all(isinstance(shape, dict) and shape.get("verified") is True for shape in shapes)):
        return stars
    return None


def _retain_last_frame(session: GameSession, evidence: list[Path]) -> None:
    if session.last_snapshot is not None and session.last_snapshot.screenshot_path not in evidence:
        evidence.append(session.last_snapshot.screenshot_path)
