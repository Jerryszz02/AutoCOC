"""Current-client real-image regression for the observed free battle chest chain."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import pytest

from autococ.battle_rewards import (finish_claimed_battle_reward,
                                    observe_battle_chest, observe_battle_chest_currency_banner,
                                    observe_battle_reward_claim)
from autococ.errors import FlowError
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.session import GameSession
from autococ.vision import ScreenshotRecognizer


FIXTURES = Path(__file__).parent / "fixtures" / "battle_rewards"
PROVENANCE = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))["fixtures"]


def recognize(name: str):
    provider = Mock()
    provider.recognize.return_value = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                                       for item in PROVENANCE[name]["ocr_from_saved_event_or_summary"]]
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    return recognizer.recognize(FIXTURES / f"{name}.png")


def test_claim_settlement_is_positive_and_preserves_battle_ledger():
    snapshot = recognize("settlement_claim")
    assert snapshot.scene == "settlement"
    assert [button["name"] for button in snapshot.observations["buttons"]] == ["claim_battle_reward"]
    assert snapshot.observations["settlement"]["loot"] == {
        "gold": 77944, "elixir": 1000109, "dark_elixir": 1770}
    assert snapshot.observations["settlement"]["bonus"] == {
        "gold": 281250, "elixir": 281250, "dark_elixir": 2081}
    assert snapshot.observations["settlement"]["stars"] == 1


@pytest.mark.parametrize("name,rarity,hammers", [
    ("chest_ordinary_4", "ordinary", 4),
    ("chest_rare_3", "rare", 3),
    ("chest_rare_2", "rare", 2),
    ("chest_rare_1", "rare", 1),
])
def test_closed_chest_state_is_bound_to_real_pixels(name, rarity, hammers):
    snapshot = recognize(name)
    assert snapshot.scene == "battle_reward"
    reward = snapshot.observations["battle_reward"]
    assert reward["stage"] == "closed_chest"
    assert reward["rarity"] == rarity
    assert reward["hammers_remaining"] == hammers
    assert [button["name"] for button in snapshot.observations["buttons"]] == ["strike_battle_chest"]


def test_observed_hero_potion_page_is_not_currency_loot():
    snapshot = recognize("hero_potion_temporary_storage")
    assert snapshot.scene == "battle_reward"
    reward = snapshot.observations["battle_reward"]
    assert reward["stage"] == "hero_potion_temporary_storage"
    assert reward["reward_item"] == "hero_potion"
    assert reward["reward_amount"] is None
    assert [button["name"] for button in snapshot.observations["buttons"]] == ["continue_battle_reward"]


def test_anonymized_fixture_hashes_and_fast_settlement_path():
    for name, source in PROVENANCE.items():
        assert hashlib.sha256((FIXTURES / f"{name}.png").read_bytes()).hexdigest() == source["fixture_sha256"]
        assert source["human_review"] is False
    provider = Mock()
    provider.recognize.return_value = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                                       for item in PROVENANCE["settlement_claim"]["ocr_from_saved_event_or_summary"]]
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    snapshot = recognizer.recognize_battle(FIXTURES / "settlement_claim.png", purpose="settlement")
    assert snapshot.scene == "settlement"
    assert snapshot.observations["fast_fallback"] is True


def test_claim_and_chest_reject_wrong_version_payment_and_fake_hammer(tmp_path):
    claim = FIXTURES / "settlement_claim.png"
    claim_ocr = PROVENANCE["settlement_claim"]["ocr_from_saved_event_or_summary"]
    assert observe_battle_reward_claim(claim, claim_ocr, client_version="18.600.7") is None
    assert observe_battle_reward_claim(claim, claim_ocr + [{"text": "购买", "confidence": 1.0,
                                                            "bbox": [10, 10, 70, 30]}],
                                       client_version="18.647.2") is None
    chest = FIXTURES / "chest_rare_2.png"
    chest_ocr = PROVENANCE["chest_rare_2"]["ocr_from_saved_event_or_summary"]
    image = cv2.imread(str(chest))
    image[584:619, 695:741] = 0
    modified = tmp_path / "fake-hammer.png"
    assert cv2.imwrite(str(modified), image)
    assert observe_battle_chest(modified, chest_ocr, client_version="18.647.2") is None
    assert observe_battle_chest(chest, chest_ocr + [{"text": "史诗", "confidence": .99,
                                                   "bbox": [100, 100, 150, 125]}],
                                client_version="18.647.2") is None


class ChainSession:
    buttons = staticmethod(GameSession.buttons)
    click = GameSession.click

    def __init__(self, stages, *, fail_on: str | None = None):
        self.stages = list(stages)
        self.last_snapshot = None
        self.client_version = "18.647.2"
        self.config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720)),
                                      runtime=SimpleNamespace(step_timeout_sec=5))
        import time
        self.deadline = self.task_deadline = time.monotonic() + 60
        self.actions = []
        self.events = []
        self.fail_on = fail_on

    def check_deadline(self):
        pass

    def observe(self, label, *, purpose="full"):
        assert purpose == "full"
        if not self.stages:
            raise FlowError("No more staged fresh frames")
        current = self.stages.pop(0)
        self.last_snapshot = current
        return current

    def tap(self, snapshot, point, *, reason):
        assert snapshot is self.last_snapshot
        self.actions.append(reason)
        if self.fail_on and self.fail_on in reason:
            raise FlowError("Transport result uncertain")

    def event(self, kind, **data):
        self.events.append((kind, data))


def _captured(name, index):
    import time
    snapshot = recognize(name)
    snapshot.observations.update(observed_at_monotonic=time.monotonic(),
                                 requested_at_monotonic=time.monotonic(),
                                 frame_id=f"fixture-{index}")
    return snapshot


def _chain_session(*, fail_on=None):
    names = ["chest_ordinary_4", "chest_rare_3", "chest_rare_2", "chest_rare_1",
             "hero_potion_temporary_storage"]
    stages = [_captured(name, index) for index, name in
              enumerate(name for name in names for _ in range(2))]
    for index in range(2):
        import time
        stages.append(SceneSnapshot("village", .9, FIXTURES / "home_after.png", {
            "buttons": [{"name": "attack"}, {"name": "shop"}],
            "village_type": "home", "resource_source": "village_inventory",
            "resources": {"gold": 19244661, "elixir": 5316226,
                          "dark_elixir": 387254, "gems": 217},
            "frame_id": f"home-{index}", "observed_at_monotonic": time.monotonic(),
            "requested_at_monotonic": time.monotonic()}))
    session = ChainSession(stages, fail_on=fail_on)
    settlement = _captured("settlement_claim", 100)
    session.last_snapshot = settlement
    return session, settlement


def test_single_shot_chain_requires_each_fresh_stage_and_returns_home():
    session, settlement = _chain_session()
    evidence = []
    with patch("autococ.battle_rewards._require_foreground"), patch("autococ.battle_rewards.time.sleep"):
        home, receipt = finish_claimed_battle_reward(session, settlement, evidence)
    assert home.scene == "village"
    assert receipt["status"] == "ui_complete_home_verified"
    assert receipt["returned_home"] is True
    assert receipt["reward_item"] == "hero_potion"
    assert receipt["reward_amount"] is None
    assert receipt["reward_inventory_verified"] is False
    assert len(receipt["input_attempts"]) == 6
    assert all(item["input_attempted"] and item["input_returned"] for item in receipt["input_attempts"])
    assert [stage["hammers_remaining"] for stage in receipt["stages"]] == [4, 3, 2, 1]
    assert len(evidence) == 12


def test_uncertain_chest_input_is_never_replayed():
    session, settlement = _chain_session(fail_on="strike_battle_chest")
    with patch("autococ.battle_rewards._require_foreground"), patch("autococ.battle_rewards.time.sleep"):
        with pytest.raises(FlowError, match="uncertain"):
            finish_claimed_battle_reward(session, settlement, [])
        with pytest.raises(FlowError, match="already attempted|incomplete"):
            finish_claimed_battle_reward(session, settlement, [])
    assert len(session.actions) == 2
    assert session.battle_reward_receipt["input_attempts"][-1]["input_returned"] is False


def test_mutated_button_point_rejects_input_before_reservation():
    session, settlement = _chain_session()
    settlement.observations["buttons"][0]["point"] = [1, 1]
    with patch("autococ.battle_rewards._require_foreground"):
        with pytest.raises(FlowError, match="revalidation|differs from verified"):
            finish_claimed_battle_reward(session, settlement, [])
    assert session.actions == []
    assert session.battle_reward_receipt["input_attempts"] == []


def test_two_home_frames_must_have_same_complete_resources():
    session, settlement = _chain_session()
    session.stages[-1].observations["resources"]["gold"] += 1
    with patch("autococ.battle_rewards._require_foreground"), patch("autococ.battle_rewards.time.sleep"):
        with pytest.raises(FlowError):
            finish_claimed_battle_reward(session, settlement, [])
    assert session.battle_reward_receipt["returned_home"] is False


NEW_PROVENANCE = json.loads((FIXTURES / "083520-provenance.json").read_text(encoding="utf-8"))


def _new_snapshot(name, index):
    import time
    source = NEW_PROVENANCE["fixtures"][name]
    provider = Mock()
    provider.recognize.return_value = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                                       for item in source["recorded_ocr"]]

    def line(_path, roi):
        if name == "defeat_claim" and tuple(roi) == (547, 310, 670, 351):
            return [OCRText("70924", .99894, tuple(roi))]
        if name == "defeat_claim" and tuple(roi) == (554, 366, 666, 391):
            return [OCRText("237792", .99999, tuple(roi))]
        return []

    provider.recognize_line.side_effect = line
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    snapshot = recognizer.recognize(FIXTURES / source["fixture"])
    snapshot.observations.update(observed_at_monotonic=time.monotonic(),
                                 requested_at_monotonic=time.monotonic(),
                                 frame_id=f"083520-{name}-{index}")
    return snapshot


def _new_home(name, index):
    import time
    source = NEW_PROVENANCE["fixtures"][name]
    path = FIXTURES / source["fixture"]
    return SceneSnapshot("village", .9, path, {
        "buttons": [{"name": "attack"}, {"name": "shop"}],
        "village_type": "home", "resource_source": "village_inventory",
        "resources": {"gold": 23692114, "elixir": 11228617,
                      "dark_elixir": 442729, "gems": 231},
        "ocr": source["recorded_ocr"], "frame_id": f"083520-home-{index}",
        "observed_at_monotonic": time.monotonic(), "requested_at_monotonic": time.monotonic(),
    })


def test_anonymized_defeat_claim_and_free_skip_states_are_source_bound():
    for name, record in NEW_PROVENANCE["fixtures"].items():
        assert hashlib.sha256((FIXTURES / record["fixture"]).read_bytes()).hexdigest() == record["fixture_sha256"]
        assert record["human_ground_truth"] is False
    result = _new_snapshot("defeat_claim", 1)
    assert result.scene == "settlement"
    assert result.observations["battle_reward_claim"]["result"] == "defeat"
    assert result.observations["settlement"]["loot"] == {
        "gold": 70924, "elixir": 237792, "dark_elixir": 777}
    assert result.observations["settlement"]["bonus"] == {
        "gold": 0, "elixir": 0, "dark_elixir": 0}
    assert result.observations["settlement"]["event_reward"]["amount"] == 267
    assert result.observations["settlement"]["evidence"]["layout"] == "defeat_with_claim_three_rows_v1"
    chest = _new_snapshot("ordinary_skip", 2)
    assert chest.scene == "battle_reward"
    assert chest.observations["battle_reward"]["button"]["name"] == "skip_battle_chest_animation"
    confirmation = _new_snapshot("skip_confirmation", 3)
    assert confirmation.scene == "battle_reward"
    assert confirmation.observations["battle_reward"]["stage"] == "skip_confirmation"
    assert confirmation.observations["battle_reward"]["button"]["name"] == "confirm_free_battle_chest_skip"


def test_current_fixture_masks_and_confirm_template_trace_to_original_when_available():
    external = Path("E:/projects/AutoCOC/reports/completion-20261015")
    if not external.is_dir():
        pytest.skip("Original local TRAIN and assisted source is unavailable")
    for record in NEW_PROVENANCE["fixtures"].values():
        source = external / record["source_frame_relative_to_external"]
        events = external / record["source_events_relative_to_external"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_sha256"]
        assert hashlib.sha256(events.read_bytes()).hexdigest() == record["source_events_sha256"]
        baseline = cv2.resize(cv2.imread(str(source)), (1280, 720), interpolation=cv2.INTER_AREA)
        expected = baseline * 0
        for x1, y1, x2, y2 in record["retained_rectangles_at_1280x720"]:
            expected[y1:y2, x1:x2] = baseline[y1:y2, x1:x2]
        assert (expected == cv2.imread(str(FIXTURES / record["fixture"]))).all()
    asset = FIXTURES.parents[2] / "assets/templates/battle_chest_skip_confirm_18_647_2.png"
    provenance = json.loads(asset.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    source = external / provenance["source_frame_relative_to_external"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["source_sha256"]
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == provenance["template_sha256"]
    baseline = cv2.resize(cv2.imread(str(source)), (1280, 720), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = provenance["crop_at_baseline"]
    assert (baseline[y1:y2, x1:x2] == cv2.imread(str(asset))).all()


def test_skip_requires_exact_free_text_current_pixels_and_version(tmp_path):
    chest = NEW_PROVENANCE["fixtures"]["ordinary_skip"]
    confirm = NEW_PROVENANCE["fixtures"]["skip_confirmation"]
    chest_path, confirm_path = (FIXTURES / chest["fixture"], FIXTURES / confirm["fixture"])
    assert observe_battle_chest(chest_path, chest["recorded_ocr"], client_version="18.600.7") is None
    assert observe_battle_chest(confirm_path, confirm["recorded_ocr"], client_version="18.600.7") is None
    for text in ("您将支付奖励，但会跳过开箱动画。仍要跳过", "跳过开箱动画并购买奖励"):
        changed = [dict(item, text=text) if item["text"].startswith("您将获得奖励") else item
                   for item in confirm["recorded_ocr"]]
        assert observe_battle_chest(confirm_path, changed, client_version="18.647.2") is None
    duplicate = confirm["recorded_ocr"] + [next(item for item in confirm["recorded_ocr"] if item["text"] == "是")]
    assert observe_battle_chest(confirm_path, duplicate, client_version="18.647.2") is None
    modified = cv2.imread(str(confirm_path))
    modified[440:495, 750:810] = 0
    altered = tmp_path / "missing-confirm-button.png"
    assert cv2.imwrite(str(altered), modified)
    assert observe_battle_chest(altered, confirm["recorded_ocr"], client_version="18.647.2") is None


def test_skip_chain_one_click_each_and_current_banner_credit_only():
    stages = [_new_snapshot("ordinary_skip", i) for i in (1, 2)]
    stages += [_new_snapshot("skip_confirmation", i) for i in (3, 4)]
    stages += [_new_home("home_banner", 5), _new_home("home_stable", 6)]
    session = ChainSession(stages)
    settlement = _new_snapshot("defeat_claim", 0)
    session.last_snapshot = settlement
    with patch("autococ.battle_rewards._require_foreground"), patch("autococ.battle_rewards.time.sleep"):
        home, receipt = finish_claimed_battle_reward(session, settlement, [])
    assert home.scene == "village"
    assert receipt["status"] == "ui_complete_home_currency_observed"
    assert receipt["reward_item"] == "elixir" and receipt["reward_amount"] == 962000
    assert receipt["currency_reward_observed"] is True
    assert receipt["reward_inventory_verified"] is False
    assert receipt["formal_acceptance"] is False
    assert [action["action"] for action in receipt["input_attempts"]] == [
        "claim_battle_reward", "skip_battle_chest_animation", "confirm_free_battle_chest_skip"]
    assert len(session.actions) == 3


@pytest.mark.parametrize("failed_action,expected_inputs", [
    ("skip_battle_chest_animation", 2),
    ("confirm_free_battle_chest_skip", 3),
])
def test_uncertain_skip_input_is_reserved_once_and_never_retried(failed_action, expected_inputs):
    stages = [_new_snapshot("ordinary_skip", i) for i in (1, 2)]
    stages += [_new_snapshot("skip_confirmation", i) for i in (3, 4)]
    session = ChainSession(stages, fail_on=failed_action)
    settlement = _new_snapshot("defeat_claim", 0)
    session.last_snapshot = settlement
    with patch("autococ.battle_rewards._require_foreground"), patch("autococ.battle_rewards.time.sleep"):
        with pytest.raises(FlowError, match="uncertain"):
            finish_claimed_battle_reward(session, settlement, [])
        with pytest.raises(FlowError, match="already attempted|incomplete"):
            finish_claimed_battle_reward(session, settlement, [])
    assert len(session.battle_reward_receipt["input_attempts"]) == expected_inputs
    assert session.battle_reward_receipt["input_attempts"][-1]["input_returned"] is False


@pytest.mark.parametrize("guard", ["_require_foreground", "_deadline"])
def test_claim_rechecks_foreground_and_deadline_after_pixel_validation_before_input(guard):
    session = ChainSession([])
    settlement = _new_snapshot("defeat_claim", 0)
    session.last_snapshot = settlement
    calls = 0

    def reject_third(*_args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise FlowError("guard changed immediately before input")

    with patch(f"autococ.battle_rewards.{guard}", side_effect=reject_third):
        if guard == "_deadline":
            foreground = patch("autococ.battle_rewards._require_foreground")
        else:
            foreground = patch("autococ.battle_rewards._deadline")
        with foreground:
            with pytest.raises(FlowError, match="immediately before input"):
                finish_claimed_battle_reward(session, settlement, [])
    assert calls == 3
    assert session.actions == []
    assert len(session.battle_reward_receipt["input_attempts"]) == 1
    assert session.battle_reward_receipt["input_attempts"][0]["input_returned"] is False


def test_skip_account_conflict_stops_without_confirming():
    stages = [_new_snapshot("ordinary_skip", i) for i in (1, 2)]
    stages.append(SceneSnapshot("account_in_use", 1.0, FIXTURES / "083520-skip_confirmation.png", {}))
    session = ChainSession(stages)
    settlement = _new_snapshot("defeat_claim", 0)
    session.last_snapshot = settlement
    with patch("autococ.battle_rewards._require_foreground"), patch("autococ.battle_rewards.time.sleep"):
        with pytest.raises(FlowError, match="interruption"):
            finish_claimed_battle_reward(session, settlement, [])
    assert [item["action"] for item in session.battle_reward_receipt["input_attempts"]] == [
        "claim_battle_reward", "skip_battle_chest_animation"]


def test_missing_or_conflicting_banner_never_invents_currency(tmp_path):
    source = NEW_PROVENANCE["fixtures"]["home_banner"]
    path = FIXTURES / source["fixture"]
    assert observe_battle_chest_currency_banner(path, [], client_version="18.647.2") is None
    assert observe_battle_chest_currency_banner(path, source["recorded_ocr"], client_version="18.600.7") is None
    too_confident = [dict(item, confidence=1.01) if "一个宝箱并获得了" in item["text"] else item
                     for item in source["recorded_ocr"]]
    assert observe_battle_chest_currency_banner(path, too_confident, client_version="18.647.2") is None
    conflicting = source["recorded_ocr"] + [dict(
        next(item for item in source["recorded_ocr"] if "一个宝箱并获得了" in item["text"]),
        text="一个宝箱并获得了123X金币")]
    assert observe_battle_chest_currency_banner(path, conflicting, client_version="18.647.2") is None
    stages = [_new_snapshot("ordinary_skip", i) for i in (1, 2)]
    stages += [_new_snapshot("skip_confirmation", i) for i in (3, 4)]
    first_home = _new_home("home_stable", 5)
    second_home = _new_home("home_stable", 6)
    fresh_path = tmp_path / "second-home-capture.png"
    fresh_path.write_bytes(second_home.screenshot_path.read_bytes())
    second_home = SceneSnapshot(second_home.scene, second_home.confidence, fresh_path,
                                second_home.observations)
    stages += [first_home, second_home]
    session = ChainSession(stages)
    settlement = _new_snapshot("defeat_claim", 0)
    session.last_snapshot = settlement
    with patch("autococ.battle_rewards._require_foreground"), patch("autococ.battle_rewards.time.sleep"):
        with pytest.raises(FlowError, match="currency reward"):
            finish_claimed_battle_reward(session, settlement, [])
    assert session.battle_reward_receipt["status"] == "incomplete"
    assert session.battle_reward_receipt["currency_reward_observed"] is False
