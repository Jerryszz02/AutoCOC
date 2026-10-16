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
                                    observe_battle_chest, observe_battle_reward_claim)
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
