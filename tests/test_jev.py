import json
from pathlib import Path
from threading import Event
import time
from unittest.mock import patch

import pytest

from autococ.errors import CapabilityUnavailable, FlowError
from autococ.jev import JevDecider, load_guides, relevant_guides


CANDIDATES = [{"id": "east", "reason": "near core"}, {"id": "west", "reason": "clear edge"}]


def answer(**changes):
    value = {"type": "choice", "choice": "west", "confidence": .9,
             "probabilities": {"east": .1, "west": .9}}
    value.update(changes)
    return {"model": "jev-1.13.0", "answers": {"candidate": value}}


def test_jev_disabled_and_missing_credentials_make_no_request():
    def forbidden(*args):
        pytest.fail("unexpected external call")
    for enabled, reason in [(False, "jev_disabled"), (True, "jev_credentials_unavailable")]:
        with patch.dict("os.environ", {}, clear=True):
            result = JevDecider(enabled=enabled, transport=forbidden).select({}, CANDIDATES,
                                                                          deadline=time.monotonic() + 5)
        assert result.candidate_id == "east" and result.source == "local" and result.reason == reason


def test_valid_choice_uses_official_wire_shape_and_only_finite_candidates():
    sent = []
    def transport(payload, timeout, key):
        sent.append((payload, timeout, key))
        return answer()
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-credential"}):
        result = JevDecider(enabled=True, transport=transport).select({"core": [500, 300]}, CANDIDATES,
                                                                   deadline=time.monotonic() + 5)
    assert result.candidate_id == "west" and result.source == "jev"
    assert sent[0][0]["questions"]["candidate"]["criteria"] == {item["id"]: item for item in CANDIDATES}
    assert "test-credential" not in repr(result) and "test-credential" not in json.dumps(sent[0][0])


@pytest.mark.parametrize("response", [answer(choice="arbitrary"), answer(confidence=.2),
    answer(confidence=float("nan")), answer(probabilities={"east": .9, "west": .1}),
    answer(probabilities={"west": 1}), answer(probabilities={"east": True, "west": False}),
    {"model": "wrong", "answers": answer()["answers"]}, None, {}])
def test_invalid_or_low_confidence_response_only_uses_local_candidate(response):
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"}):
        result = JevDecider(enabled=True, transport=lambda *_: response).select({}, CANDIDATES,
                                                                               deadline=time.monotonic() + 5)
    assert result.source == "local" and result.candidate_id == "east"


def test_timeout_does_not_mutate_choice_or_start_overlapping_request():
    release, entered = Event(), Event()
    calls = []
    def slow(*_):
        calls.append(1)
        entered.set()
        release.wait(2)
        return answer()
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"}):
        client = JevDecider(enabled=True, transport=slow, timeout_sec=.02)
        try:
            first = client.select({}, CANDIDATES, deadline=time.monotonic() + 5)
            second = client.select({}, CANDIDATES, deadline=time.monotonic() + 5)
            assert entered.is_set() and len(calls) == 1
            assert first.reason == "jev_timeout" and second.reason == "jev_previous_request_pending"
            assert first.candidate_id == "east"
        finally:
            release.set()
            client._worker.join(1)
    assert first.candidate_id == "east"


def test_expired_preparation_cannot_fallback_or_accept_late_response():
    with pytest.raises(FlowError):
        JevDecider().select({}, CANDIDATES, deadline=time.monotonic() - 1)
    clock = [100.0]
    def late(*_):
        clock[0] = 103
        return answer()
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test"}), pytest.raises(FlowError):
        JevDecider(enabled=True, transport=late, clock=lambda: clock[0]).select({}, CANDIDATES, deadline=102)


def test_no_candidate_is_not_invented():
    with pytest.raises(CapabilityUnavailable):
        JevDecider().select({}, [], deadline=time.monotonic() + 5)


def test_guide_schema_and_applicability(tmp_path: Path):
    path = tmp_path / "guides.json"
    rule = dict(id="air", units=["dragon"], reason="test", order="funnel first", failure_conditions="unknown core")
    path.write_text(json.dumps([rule]), encoding="utf-8")
    guides = load_guides(path)
    assert relevant_guides(guides, {"dragon"}) == (rule,)
    assert relevant_guides(guides, {"barbarian"}) == ()
    path.write_text(json.dumps([rule, rule]), encoding="utf-8")
    with pytest.raises(CapabilityUnavailable):
        load_guides(path)
