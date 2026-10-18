from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from autococ.deployment import _deploy_continuous_line, deploy_army
from autococ.combat import _read_continuous_line_army
from autococ.errors import DeploymentError, FlowError, StopRequested
from autococ.scene import SceneSnapshot


def setup_line():
    frame = SceneSnapshot("enemy_village", .99, Path("selected.png"), {
        "observed_at_monotonic": 100.0,
        "battle": {"countdown_seconds": 7, "countdown_evidence": {
            "source": "current_frame_ocr", "frame": "selected.png", "confidence": .99}},
    })
    before = SceneSnapshot("enemy_village", .99, Path("toolbar.png"))
    heroes = {"complete": True, "cards": [{"unit_id": "archer_queen"}]}
    receipt = {"evidence": ["scout.png", "toolbar.png", "selected.png"],
               "terrain_evidence": [{"edge": 0}], "expected_army_manifest": {"complete": True},
               "minimum_zoom": {"verified": True, "frame": "toolbar.png"}}
    return Mock(last_snapshot=frame), frame, before, heroes, receipt


def test_continuous_line_uses_capture_time_and_preserves_preparation_evidence():
    session, frame, before, heroes, receipt = setup_line()
    session.config.battle.line_tactics = "two_edge_split"
    post = SceneSnapshot("battle", .99, Path("post.png"), {"deployment": {
        "verified": True, "completed": True, "deployed_units": 14,
        "evidence": ["selected.png", "post.png"]}})
    plan = SimpleNamespace(plan_id="plan", plan_sha256="hash", actions=(1, 2),
                           burst_limit_sec=10, line_tactics="two_edge_split")
    with patch("autococ.frozen_line.freeze_model_free_line", return_value=plan) as freeze, \
            patch("autococ.frozen_line.execute_frozen_line", return_value=post) as execute:
        result = _deploy_continuous_line(session, frame, receipt, {"toolbar_before": before}, heroes, 200)
    kwargs = freeze.call_args.kwargs
    assert kwargs["latest_start"] == 106.0  # seven observed seconds, one-second reserve
    assert kwargs["toolbar_before"] is before
    assert kwargs["hero_manifest"] is heroes
    assert kwargs["hero_ids"] == ["archer_queen"]
    assert kwargs["line_tactics"] == "two_edge_split"
    execute.assert_called_once_with(session, plan)
    assert result.observations["deployment"]["evidence"] == ["scout.png", "toolbar.png", "selected.png", "post.png"]


@pytest.mark.parametrize("problem", ["active_battle", "timer_missing", "heroes_missing"])
def test_continuous_line_does_not_freeze_without_preview_contract(problem):
    session, frame, before, heroes, receipt = setup_line()
    if problem == "active_battle":
        frame = SceneSnapshot("battle", .99, frame.screenshot_path, frame.observations)
    elif problem == "timer_missing":
        frame.observations["battle"] = {}
    else:
        heroes = None
    with patch("autococ.frozen_line.freeze_model_free_line") as freeze, \
            patch("autococ.frozen_line.execute_frozen_line") as execute, pytest.raises(FlowError):
        _deploy_continuous_line(session, frame, receipt, {"toolbar_before": before}, heroes, 200)
    freeze.assert_not_called()
    execute.assert_not_called()


def test_partial_continuous_receipt_survives_outer_deployment_error():
    session, frame, _, _, _ = setup_line()
    partial = {"deployed_units": 0, "attempted_placements": 1, "issued_placements": 0,
               "plan_id": "one-shot", "evidence": ["selected.png"], "actions": [{"status": "uncertain"}]}
    with patch("autococ.deployment._deploy_army", side_effect=DeploymentError("transport", partial_receipt=partial)), \
            pytest.raises(DeploymentError) as error:
        deploy_army(session, frame)
    assert error.value.partial_receipt["attempted_placements"] == 1
    assert error.value.partial_receipt["plan_id"] == "one-shot"
    assert error.value.partial_receipt["actions"] == [{"status": "uncertain"}]


def test_user_stop_remains_cooperative_and_is_not_a_deployment_failure():
    session, frame, _, _, _ = setup_line()
    with patch("autococ.deployment._deploy_army", side_effect=StopRequested("stop")), pytest.raises(StopRequested):
        deploy_army(session, frame)


def test_presearch_roster_uses_fresh_frame_and_nested_recipe():
    first = SceneSnapshot("training", .99, Path("first.png"))
    fresh = SceneSnapshot("training", .99, Path("fresh.png"))
    manifest = {"complete": True, "frame": "fresh.png", "cards": [{
        "kind": "hero", "source": "army", "unit_id": "archer_queen", "evidence": {"identity": {
            "unit_id": "archer_queen", "version_verified": True,
            "template_client_version": "18.647.2", "confidence": .99}}}]}
    session = Mock(client_version="18.647.2")
    session.observe.return_value = fresh
    evidence = []
    troop = {"kind": "troop", "source": "army", "unit_id": "archer", "count": 3,
             "evidence": {"identity": {"unit_id": "archer", "version_verified": True,
                 "template_client_version": "18.647.2", "confidence": .99}}}
    # Spell identities do not satisfy or defeat the new troop-only gate. The
    # frozen plan retains its separate complete-manifest contract downstream.
    recipe = {"hero_evidence": manifest, "manifest": {"troops": [troop],
              "spells": [{"unit_id": None}], "identity_complete": False}}
    with patch("autococ.combat._read_army", side_effect=[{"recipe": {}}, {"recipe": recipe}]):
        frame, army = _read_continuous_line_army(session, first, evidence)
    assert frame is fresh
    assert army["recipe"]["hero_evidence"] is manifest
    assert evidence == [fresh.screenshot_path]
    session.click.assert_not_called()


def test_incomplete_presearch_roster_returns_home_without_paid_search():
    current = SceneSnapshot("training", .99, Path("army.png"))
    home = SceneSnapshot("village", .99, Path("home.png"))
    session = Mock(client_version="18.647.2")
    session.observe.return_value = current
    evidence = []
    with patch("autococ.combat._read_army", return_value={"recipe": {}}), \
            patch("autococ.combat.return_to_village", return_value=home) as recover, \
            pytest.raises(FlowError, match="no paid search"):
        _read_continuous_line_army(session, current, evidence)
    assert session.observe.call_count == 2
    recover.assert_called_once_with(session, initial_snapshot=current)
    session.click.assert_not_called()
