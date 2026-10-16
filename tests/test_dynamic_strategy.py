"""Synthetic contract tests; no R/U model or game input is exercised."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from autococ import dynamic_strategy
from autococ.dynamic_strategy import (propose_configured_hero_ready_wait,
                                      propose_configured_hero_ready_wait_from_sources)
from autococ.dynamic_tracking import DynamicTracker, HeroDeploymentEvidence
from autococ.dynamic_vision import CARD_STATES, DynamicFrame, DynamicModel, DynamicObservation
from autococ.strategy_config import StrategyDefinition, StrategyStep


ROI = (10, 10, 40, 40)
MODEL_SHA = "a" * 64
CONTRACT_SHA = "b" * 64


def _frame(index: int, sha256: str, *, camera: str = "camera-1") -> DynamicFrame:
    return DynamicFrame("battle-1", camera, f"frame-{index}", float(index),
                        "18.647.2", 100, 100, True, True, sha256)


def _observation(frame: DynamicFrame) -> DynamicObservation:
    return DynamicObservation("r_card", "available", ROI, .99, frame.session_id,
                              frame.camera_epoch, frame.frame_id,
                              frame.captured_at_monotonic, ROI,
                              "hero_skill:archer_queen", MODEL_SHA, CONTRACT_SHA,
                              frame.source_sha256)


def _hero(frame: DynamicFrame) -> HeroDeploymentEvidence:
    return HeroDeploymentEvidence("archer_queen", True, .99, frame.session_id,
                                  frame.camera_epoch, frame.frame_id,
                                  frame.captured_at_monotonic, frame.client_version,
                                  frame.source_sha256, ROI,
                                  "friendly_battle_card_portrait_and_enclosed_hp")


def _definition() -> StrategyDefinition:
    return StrategyDefinition("queen-skill", "Queen skill", None,
                              (StrategyStep("wait", unit_id="archer_queen", target="hero_ready"),
                               StrategyStep("activate_ability", unit_id="archer_queen")))


def _inputs(tmp_path) -> dict:
    paths = []
    frames = []
    for index in (1, 2):
        # Synthetic pixels exercise byte/dimension binding, not model quality.
        pixels = np.full((100, 100, 3), index * 30, dtype=np.uint8)
        path = tmp_path / f"synthetic-{index}.png"
        path.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
        paths.append(path)
        frames.append(_frame(index, hashlib.sha256(path.read_bytes()).hexdigest()))
    first_frame, confirmed_frame = frames
    tracker = DynamicTracker(max_gap_seconds=2)
    first_hero, confirmed_hero = _hero(first_frame), _hero(confirmed_frame)
    first_step = tracker.update(first_frame, [_observation(first_frame)],
                                hero_deployments=[first_hero], offline_replay=True)
    confirmed_step = tracker.update(confirmed_frame, [_observation(confirmed_frame)],
                                    hero_deployments=[confirmed_hero], offline_replay=True)
    assert len(confirmed_step.events) == 1  # Real tracker contract, synthetic inputs.
    return dict(definition=_definition(), wait_step_index=0,
                first_source_path=paths[0], first_frame=first_frame,
                first_step=first_step, first_hero=first_hero,
                confirmed_source_path=paths[1], confirmed_frame=confirmed_frame,
                confirmed_step=confirmed_step,
                confirmed_hero=confirmed_hero,
                model=SimpleNamespace(kind="r_card", client_version="18.647.2",
                                      model_sha256=MODEL_SHA, contract_sha256=CONTRACT_SHA,
                                      event_rules={"max_gap_seconds": 2},
                                      validation={"status": "evaluated", "prototype_only": False,
                                                  "report_id": "synthetic-evaluation"}))


class _Session:
    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def run(self, _outputs, inputs):
        self.calls.append(inputs["image"].copy())
        return [np.array([[.98, .01, .005, .005]], dtype=np.float32)]


def _loaded_r_card_model(tmp_path):
    root = tmp_path / "r-card-package"
    root.mkdir()
    weight = root / "model.onnx"
    weight.write_bytes(b"synthetic-r-card-weights")
    manifest = {
        "schema_version": 1, "kind": "r_card", "classes": list(CARD_STATES),
        "client_versions": ["18.647.2"], "model_file": "model.onnx",
        "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "max_frame_age_seconds": 1.0,
        "event_rules": {"consecutive_frames": 2, "min_iou": .4,
                         "max_gap_seconds": 2.0, "max_missed_frames": 1,
                         "event_min_confidence": .95, "max_frame_age_seconds": .5},
        "applicability": {"roi_kind": "r_card", "source_width_range": [100, 100],
                          "source_height_range": [100, 100], "requires_stable_camera": True,
                          "requires_verified_zoom": False},
        "preprocess": {"width": 224, "height": 224, "color": "RGB", "resize": "stretch",
                       "scale": 1 / 255, "mean": [0, 0, 0], "std": [1, 1, 1]},
        "output": {"format": "probabilities", "min_confidence": .7,
                   "max_detections": 10},
        "validation": {"status": "evaluated", "prototype_only": False,
                       "report_id": "synthetic-evaluation",
                       "snapshot_sha256": "a" * 64, "split_sha256": "b" * 64},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    session = _Session()
    return DynamicModel(root, client_version="18.647.2",
                        session_factory=lambda _: session), session


def _source_inputs(tmp_path):
    paths = []
    frames = []
    for index in (1, 2):
        pixels = np.full((100, 100, 3), index * 30, dtype=np.uint8)
        path = tmp_path / f"source-{index}.png"
        path.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
        paths.append(path)
        # Hash is deliberately absent: the source API must compute it itself.
        frames.append(_frame(index, None))
    return {"definition": _definition(), "wait_step_index": 0,
            "first_source_path": paths[0], "first_frame": frames[0],
            "confirmed_source_path": paths[1], "confirmed_frame": frames[1],
            "roi_bbox": ROI}


def test_two_verified_hero_frames_only_propose_configured_wait_offline(tmp_path):
    params = _inputs(tmp_path)
    result = propose_configured_hero_ready_wait(**params)
    assert result.reason == "offline_candidate_only_release_unverified"
    assert result.execution_authorized is False
    candidate = result.candidate
    assert candidate is not None
    assert (candidate.hero_id, candidate.wait_step_index, candidate.following_step_index) == (
        "archer_queen", 0, 1)
    assert (candidate.first_frame_id, candidate.confirmed_frame_id) == ("frame-1", "frame-2")
    assert (candidate.model_sha256, candidate.contract_sha256) == (MODEL_SHA, CONTRACT_SHA)
    assert candidate.roi_bbox == ROI
    assert candidate.source_paths == (str(params["first_source_path"].resolve()),
                                      str(params["confirmed_source_path"].resolve()))
    assert not hasattr(candidate, "point")  # Cannot be passed as an input action.


def test_only_explicit_same_hero_wait_and_ability_pair_can_match(tmp_path):
    params = _inputs(tmp_path)
    alternatives = (
        replace(params["definition"], steps=(StrategyStep("activate_ability", unit_id="archer_queen"),
                                              StrategyStep("activate_ability", unit_id="archer_queen"))),
        replace(params["definition"], steps=(StrategyStep("wait", unit_id="archer_queen",
                                                           target="hero_ready"),
                                              StrategyStep("activate_ability", unit_id="grand_warden"))),
        replace(params["definition"], steps=(StrategyStep("wait", unit_id="electro_dragon",
                                                           target="hero_ready"),
                                              StrategyStep("activate_ability", unit_id="electro_dragon"))),
    )
    for definition in alternatives:
        result = propose_configured_hero_ready_wait(**{**params, "definition": definition})
        assert result.candidate is None
        assert result.reason == "configured_hero_ability_pair_missing"


def test_model_and_camera_are_independently_bound(tmp_path):
    params = _inputs(tmp_path)
    model_variant = lambda **changes: SimpleNamespace(**{**vars(params["model"]), **changes})
    for model in (
        model_variant(validation={"status": "unvalidated", "prototype_only": True}),
        model_variant(kind="u_position"),
        model_variant(client_version="18.600.7"),
    ):
        result = propose_configured_hero_ready_wait(**{**params, "model": model})
        assert result.candidate is None
        assert result.reason == "formal_r_card_evaluation_claim_missing"
    wrong_model = model_variant(model_sha256="d" * 64)
    assert propose_configured_hero_ready_wait(**{**params, "model": wrong_model}).reason == (
        "tracked_card_or_model_binding_invalid")
    for frame in (replace(params["confirmed_frame"], camera_epoch="camera-2"),
                  replace(params["confirmed_frame"], frame_id="frame-1"),
                  replace(params["confirmed_frame"], camera_stable=False),
                  replace(params["confirmed_frame"], source_sha256="0" * 64)):
        result = propose_configured_hero_ready_wait(**{**params, "confirmed_frame": frame})
        assert result.candidate is None
        assert result.reason in {"frame_or_camera_binding_invalid",
                                 "tracked_card_or_model_binding_invalid",
                                 "original_source_unverified"}
    late = replace(params["confirmed_frame"], captured_at_monotonic=10.0)
    assert propose_configured_hero_ready_wait(**{**params, "confirmed_frame": late}).reason == (
        "frame_or_camera_binding_invalid")


def test_missing_or_wrong_friendly_hero_proof_breaks_candidate(tmp_path):
    params = _inputs(tmp_path)
    for proof in (None,
                  replace(params["confirmed_hero"], hero_id="grand_warden"),
                  replace(params["confirmed_hero"], deployed=False),
                  replace(params["confirmed_hero"], confidence=.8),
                  replace(params["confirmed_hero"], frame_id="frame-1"),
                  replace(params["confirmed_hero"], source_sha256="0" * 64),
                  replace(params["confirmed_hero"], roi_bbox=(1, 1, 30, 30))):
        result = propose_configured_hero_ready_wait(**{**params, "confirmed_hero": proof})
        assert result.candidate is None
        assert result.reason == "friendly_hero_deployment_unverified"


def test_event_cannot_borrow_another_track_or_roi(tmp_path):
    params = _inputs(tmp_path)
    bad_event = replace(params["confirmed_step"].events[0], first_frame_id="frame-0")
    step = replace(params["confirmed_step"], events=(bad_event,))
    assert propose_configured_hero_ready_wait(**{**params, "confirmed_step": step}).reason == (
        "unique_two_frame_skill_event_missing")
    bad_track = replace(params["confirmed_step"].tracked[0],
                        observation=replace(params["confirmed_step"].tracked[0].observation,
                                            roi_bbox=(5, 5, 30, 30)))
    step = replace(params["confirmed_step"], tracked=(bad_track,))
    assert propose_configured_hero_ready_wait(**{**params, "confirmed_step": step}).reason == (
        "tracked_card_or_model_binding_invalid")


def test_originals_must_exist_match_bytes_and_decode_at_declared_size(tmp_path):
    params = _inputs(tmp_path)
    changed = tmp_path / "changed.png"
    changed.write_bytes(cv2.imencode(".png", np.ones((100, 100, 3), np.uint8))[1].tobytes())
    wrong_size = tmp_path / "wrong-size.png"
    wrong_size.write_bytes(cv2.imencode(".png", np.zeros((90, 100, 3), np.uint8))[1].tobytes())
    missing = tmp_path / "missing.png"
    for field, path in (("first_source_path", missing),
                        ("first_source_path", changed),
                        ("confirmed_source_path", changed)):
        result = propose_configured_hero_ready_wait(**{**params, field: path})
        assert result.candidate is None
        assert result.reason == "original_source_unverified"
    # A self-consistent SHA cannot disguise an incompatible image size.
    resized_frame = replace(params["first_frame"],
                            source_sha256=hashlib.sha256(wrong_size.read_bytes()).hexdigest())
    result = propose_configured_hero_ready_wait(**{**params, "first_source_path": wrong_size,
                                                  "first_frame": resized_frame})
    assert result.candidate is None
    assert result.reason == "original_source_unverified"
    params["confirmed_source_path"].write_bytes(changed.read_bytes())
    assert propose_configured_hero_ready_wait(**params).reason == "original_source_unverified"
    params["first_source_path"].unlink()
    assert propose_configured_hero_ready_wait(**params).reason == "original_source_unverified"


def test_source_entry_runs_loaded_model_and_one_tracker_on_same_hashed_sources(tmp_path, monkeypatch):
    params = _source_inputs(tmp_path)
    model, session = _loaded_r_card_model(tmp_path)
    source_paths = (params["first_source_path"].resolve(),
                    params["confirmed_source_path"].resolve())
    expected_hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in source_paths)
    verified_snapshots = []
    readonly_inputs = []

    def verify_snapshot(path, frame, roi_bbox, hero_id):
        path = Path(path)
        assert path not in source_paths
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame.source_sha256
        assert roi_bbox == ROI
        assert hero_id == "archer_queen"
        verified_snapshots.append((path, frame.frame_id, frame.source_sha256))
        return _hero(frame)

    original_infer = DynamicModel.infer

    def read_only_infer(self, image, *args, **kwargs):
        readonly_inputs.append(image.flags.writeable)
        return original_infer(self, image, *args, **kwargs)

    monkeypatch.setattr(dynamic_strategy, "read_hero_deployment_evidence", verify_snapshot)
    monkeypatch.setattr(DynamicModel, "infer", read_only_infer)
    result = propose_configured_hero_ready_wait_from_sources(**params, model=model)

    assert result.reason == "offline_candidate_only_release_unverified"
    assert result.execution_authorized is False
    assert result.candidate is not None
    assert result.candidate.source_paths == tuple(map(str, source_paths))
    assert (result.candidate.first_source_sha256,
            result.candidate.confirmed_source_sha256) == expected_hashes
    assert len(verified_snapshots) == 2
    assert readonly_inputs == [False, False]
    assert len(session.calls) == 2  # DynamicModel.infer ran once for each source.
    assert not np.array_equal(session.calls[0], session.calls[1])


def test_source_entry_rejects_unbound_hash_and_unverified_current_hero(tmp_path, monkeypatch):
    params = _source_inputs(tmp_path)
    model, session = _loaded_r_card_model(tmp_path)
    conflicting_frame = replace(params["first_frame"], source_sha256="0" * 64)
    result = propose_configured_hero_ready_wait_from_sources(
        **{**params, "first_frame": conflicting_frame}, model=model)
    assert result.candidate is None
    assert result.reason == "original_source_unverified"
    assert session.calls == []

    # The real verifier sees these synthetic images and fails closed: it must
    # not turn model output alone into current-frame friendly-hero evidence.
    result = propose_configured_hero_ready_wait_from_sources(**params, model=model)
    assert result.candidate is None
    assert result.reason == "friendly_hero_deployment_unverified"
    assert session.calls == []


def test_source_entry_rejects_nonpackage_models_and_mutated_inference_pixels(tmp_path, monkeypatch):
    params = _source_inputs(tmp_path)
    legacy = _inputs(tmp_path)
    result = propose_configured_hero_ready_wait_from_sources(
        **params, model=legacy["model"])
    assert result.candidate is None
    assert result.reason == "formal_r_card_evaluation_claim_missing"

    model, _ = _loaded_r_card_model(tmp_path)
    model.validation["prototype_only"] = True
    result = propose_configured_hero_ready_wait_from_sources(**params, model=model)
    assert result.candidate is None
    assert result.reason == "formal_r_card_evaluation_claim_missing"

    model.validation["prototype_only"] = False
    monkeypatch.setattr(dynamic_strategy, "read_hero_deployment_evidence",
                        lambda _path, frame, _roi, hero_id: _hero(frame))
    original_infer = DynamicModel.infer

    def mutate_then_infer(self, image, *args, **kwargs):
        image.setflags(write=True)
        image[0, 0, 0] ^= np.uint8(1)
        return original_infer(self, image, *args, **kwargs)

    monkeypatch.setattr(DynamicModel, "infer", mutate_then_infer)
    result = propose_configured_hero_ready_wait_from_sources(**params, model=model)
    assert result.candidate is None
    assert result.reason == "source_model_modified_input"

    def broken_infer(self, image, *args, **kwargs):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(DynamicModel, "infer", broken_infer)
    result = propose_configured_hero_ready_wait_from_sources(**params, model=model)
    assert result.candidate is None
    assert result.reason == "source_replay_failed"
