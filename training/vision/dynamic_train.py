"""Offline R-state ROI classifier training from hash-frozen event windows."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .data import sha256
from .dynamic_snapshot import verify_dynamic_snapshot
from .snapshot import TrainingError
from .train import (_checkpoint_improvement_reason, _checkpoint_score_details,
                    _classification_validation_diagnostics, _environment, _metric,
                    _restore_rng, _rng_state, _scene_model, _seed, _signature,
                    _advance_scene_patience,
                    _set_scene_training_phase, SCENE_TRAINER_RECIPE)


R_PREPROCESS = {"width": 224, "height": 224, "color": "RGB", "resize": "stretch",
                "scale": 1 / 255, "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]}


def roi_tensor(root: Path, frame: dict, preprocess: dict = R_PREPROCESS) -> np.ndarray:
    """Match DynamicModel.infer's source-frame ROI transform exactly."""
    relative = (frame["image"] if frame["sha256"] == frame["parent_image_sha256"]
                else frame["parent_image"])
    image = cv2.imdecode(np.fromfile(root / relative, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise TrainingError(f"Cannot decode frozen dynamic source: {relative}")
    x0, y0, x1, y1 = frame["roi_bbox"]
    crop = cv2.resize(image[y0:y1, x0:x1], (preprocess["width"], preprocess["height"]),
                      interpolation=cv2.INTER_LINEAR)
    if preprocess["color"] == "RGB":
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mean = np.asarray(preprocess["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(preprocess["std"], dtype=np.float32).reshape(1, 1, 3)
    value = (crop.astype(np.float32) * preprocess["scale"] - mean) / std
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def _rows(snapshot: dict) -> dict[str, list[dict]]:
    result = {"train": [], "val": [], "test": []}
    seen = {}
    for window in snapshot["windows"]:
        component = snapshot["inventory"]["window_components"][window["window_id"]]
        for frame in window["frames"]:
            # One classifier input/target supervision has one sampling weight,
            # even when adjacent reviewed event windows overlap in time.
            key = (frame["parent_image_sha256"], tuple(frame["roi_bbox"]), frame["state_target_id"])
            prior = seen.get(key)
            supervision = (frame["state_label"], window["split"], component)
            if prior is not None:
                if prior != supervision:
                    raise TrainingError("Conflicting or cross-split repeated R supervision")
                continue
            seen[key] = supervision
            result[window["split"]].append({"frame": frame, "label": frame["state_label"],
                                            "component": component})
    return result


def _config(snapshot: dict, config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = snapshot["metadata"]
    if (not isinstance(config, dict) or config.get("kind") != metadata["kind"] or
            config.get("kind") not in {"r_building", "r_card"} or
            config.get("classes") != metadata["classes"] or
            config.get("client_versions") != metadata["client_versions"] or
            config.get("prototype_only") is not metadata["prototype_only"]):
        raise TrainingError("R config kind/classes/client versions/review policy differ from frozen snapshot")
    if config.get("preprocess") != R_PREPROCESS or config.get("architecture") != "mobilenet_v3_small":
        raise TrainingError("R config must declare the fixed 224x224 MobileNetV3-small ROI contract")
    for key in ("applicability", "output", "event_rules"):
        if not isinstance(config.get(key), dict):
            raise TrainingError(f"R config missing explicit {key}")
    if (config["applicability"].get("roi_kind") != config["kind"] or
            config["output"].get("format") != "logits" or
            not isinstance(config.get("code_version"), str) or not config["code_version"].strip()):
        raise TrainingError("R applicability/output/code version is incomplete")
    confidence = config["output"].get("min_confidence")
    age = config.get("max_frame_age_seconds")
    if (type(confidence) not in (int, float) or not math.isfinite(confidence) or
            not 0 <= confidence <= 1 or type(age) not in (int, float) or
            not math.isfinite(age) or not 0 < age <= 10):
        raise TrainingError("R output threshold/frame-age contract is invalid")
    application = config["applicability"]
    for key in ("source_width_range", "source_height_range"):
        bounds = application.get(key)
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(type(value) is not int or value < 1 for value in bounds) or
                bounds[0] > bounds[1]):
            raise TrainingError(f"R applicability {key} is invalid")
    if (type(application.get("requires_stable_camera")) is not bool or
            type(application.get("requires_verified_zoom")) is not bool or
            config["kind"] == "r_building" and not application["requires_stable_camera"]):
        raise TrainingError("R camera/zoom applicability is invalid")
    rules = config["event_rules"]
    try:
        from autococ.dynamic_tracking import DynamicTracker
        if rules.get("consecutive_frames") != 2:
            raise ValueError("two consecutive frames required")
        tracker = DynamicTracker(**{key: rules[key] for key in
            ("min_iou", "max_gap_seconds", "max_missed_frames",
             "event_min_confidence", "max_frame_age_seconds")})
        if tracker.max_frame_age_seconds > age:
            raise ValueError("event age exceeds model age")
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError("R event rules are invalid") from exc
    root = Path(snapshot["root"])
    seen = set()
    for window in snapshot["windows"]:
        for frame in window["frames"]:
            source = (frame["image"] if frame["sha256"] == frame["parent_image_sha256"]
                      else frame["parent_image"])
            if source in seen:
                continue
            seen.add(source)
            image = cv2.imdecode(np.fromfile(root / source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or not (application["source_width_range"][0] <= image.shape[1] <= application["source_width_range"][1]
                                     and application["source_height_range"][0] <= image.shape[0] <= application["source_height_range"][1]):
                raise TrainingError("R source image outside declared applicability")
    return config


def train_dynamic_state(snapshot_root: Path, config_path: Path, output: Path, *,
                        device: str, seed: int = 42, resume: Path | None = None,
                        max_epochs: int = 35, deadline_seconds: float = 4 * 3600,
                        pretrained: Path | None = None) -> dict:
    if not 1 <= max_epochs <= 35 or deadline_seconds <= 0:
        raise TrainingError("R training requires 1..35 epochs and a positive deadline")
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    if device.isdigit():
        device = f"cuda:{device}"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    snapshot = verify_dynamic_snapshot(snapshot_root, kind=raw_config.get("kind"))
    root = Path(snapshot["root"])
    config = _config(snapshot, config_path)
    if not snapshot["metadata"]["prototype_only"] and pretrained is None:
        raise TrainingError("Formal R training requires local ImageNet pretrained weights")
    splits = _rows(snapshot)
    if not splits["train"] or not splits["val"]:
        raise TrainingError("R training needs nonempty frozen train and validation splits")
    classes = config["classes"]
    if not config["prototype_only"] and any(set(classes) - {row["label"] for row in splits[name]}
                                               for name in ("train", "val")):
        raise TrainingError("Formal R training needs every state in train and validation")
    batch = int(config.get("batch", 8))
    if not 1 <= batch <= 32:
        raise TrainingError("R batch must be within 1..32")
    output = output.resolve()
    if output.exists() and resume is None:
        raise TrainingError(f"Output already exists: {output}")
    _seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    environment = _environment()
    config.update({"snapshot_sha256": snapshot["snapshot_sha256"],
                   "split_sha256": snapshot["split_sha256"], "seed": seed,
                   "device": device, "batch": batch, "max_epochs": max_epochs,
                   "deadline_seconds": deadline_seconds,
                   "pretrained_sha256": sha256(pretrained) if pretrained else None,
                   "torch_version": environment["packages"]["torch"],
                   "torchvision_version": environment["packages"]["torchvision"],
                   **SCENE_TRAINER_RECIPE})
    identity = {key: value for key, value in config.items() if key != "deadline_seconds"}
    signature = _signature(identity)
    checkpoint = None
    if resume is not None:
        if not resume.is_file() or resume.resolve() != output / "last.pt":
            raise TrainingError("R resume requires this run's last.pt")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("signature") != signature:
            raise TrainingError("R resume config/snapshot/pretrained/version differs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    model, _ = _scene_model(classes, pretrained)
    model.to(device)

    class ROIs(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            return torch.from_numpy(roi_tensor(root, row["frame"])), classes.index(row["label"])

    train_rows, val_rows = splits["train"], splits["val"]
    group_counts = {key: sum(row["component"] == key for row in train_rows)
                    for key in {row["component"] for row in train_rows}}
    class_counts = {key: sum(row["label"] == key for row in train_rows)
                    for key in {row["label"] for row in train_rows}}
    weights = torch.as_tensor([1 / (group_counts[row["component"]] * class_counts[row["label"]])
                               for row in train_rows], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(ROIs(train_rows), batch_size=batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(ROIs(val_rows), batch_size=batch, shuffle=False, num_workers=0)
    start, best_score, best_epoch, bad_epochs, history = 0, (-1.0, -1.0, None), None, 0, []
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        start = checkpoint["epoch"] + 1
        best_score = tuple(checkpoint["best_score"])
        best_epoch = checkpoint["best_epoch"]
        bad_epochs = checkpoint["bad_epochs"]
        history = checkpoint["history"]
        _restore_rng(checkpoint["rng"])
    head_epochs = min(5, max_epochs)
    began = time.monotonic()
    optimizer = scheduler = None
    for epoch in range(start, max_epochs):
        if time.monotonic() - began >= deadline_seconds:
            break
        phase = "head" if epoch < head_epochs else "finetune"
        _set_scene_training_phase(model, phase)
        if optimizer is None or epoch == head_epochs:
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                          lr=.001 if phase == "head" else .0001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)
            if checkpoint and epoch == start and checkpoint["phase"] == phase:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise TrainingError("Non-finite R training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        truth, predictions, validation_logits = [], [], []
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device)).cpu().numpy()
                validation_logits.append(logits)
                predictions.extend(logits.argmax(axis=1).tolist())
                truth.extend(labels.tolist())
        metrics = _metric(truth, predictions, classes)
        diagnostics = _classification_validation_diagnostics(
            np.concatenate(validation_logits) if validation_logits else
            np.empty((0, len(classes)), dtype=np.float32), truth, classes)
        score = (metrics["macro_f1"], metrics["minimum_recall"],
                 diagnostics["val_cross_entropy"])
        previous_best_score = best_score
        scheduler.step()
        best_score, bad_epochs, improved = _advance_scene_patience(
            score, best_score, bad_epochs, epoch=epoch, head_epochs=head_epochs,
            nll_min_delta=config["validation_nll_min_delta"])
        if improved:
            best_epoch = epoch
        history.append({"epoch": epoch, "phase": phase, "loss": sum(losses) / len(losses),
                        "validation": metrics, "validation_diagnostics": diagnostics,
                        "checkpoint_selection": {
                            "improved": improved,
                            "reason": _checkpoint_improvement_reason(score, previous_best_score, improved),
                            "current_score": _checkpoint_score_details(score),
                            "best_score": _checkpoint_score_details(best_score),
                            "validation_nll_min_delta": config["validation_nll_min_delta"]},
                        "elapsed_seconds": time.monotonic() - began})
        state = {"signature": signature, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "phase": phase, "best_score": best_score, "best_epoch": best_epoch,
                 "bad_epochs": bad_epochs, "history": history, "rng": _rng_state()}
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if phase == "finetune" and bad_epochs >= SCENE_TRAINER_RECIPE["finetune_patience"]:
            break
    report = {"kind": config["kind"], "prototype_only": config["prototype_only"],
              "train_frames": len(train_rows), "validation_frames": len(val_rows),
              "overlap_frames_removed": sum(len(window["frames"]) for window in snapshot["windows"]) -
                                        sum(len(rows) for rows in splits.values()),
              "train_groups": len({row["component"] for row in train_rows}),
              "validation_groups": len({row["component"] for row in val_rows}),
              "epochs_completed": len(history), "best_score": best_score, "best_epoch": best_epoch,
              "snapshot_sha256": snapshot["snapshot_sha256"],
              "status": "prototype_only" if config["prototype_only"] else "trained_unevaluated",
              "checkpoint": str(output / "best.pt"), "last_checkpoint": str(output / "last.pt")}
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
