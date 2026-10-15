"""Reproducible, local scene and building training from reviewed snapshots.

Training dependencies belong in a separate environment. This module does not
download pretrained weights or import training libraries during product startup.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np

from .data import read_jsonl, sha256


SCENE_PREPROCESS = {"width": 320, "height": 192, "color": "RGB", "resize": "letterbox",
                    "scale": 1 / 255, "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225], "pad_value": 114}
BUILDING_PREPROCESS = {"width": 640, "height": 640, "color": "RGB", "resize": "letterbox",
                       "scale": 1 / 255, "mean": [0, 0, 0], "std": [1, 1, 1],
                       "pad_value": 114}


from .snapshot import TrainingError, verify_snapshot


def prepare_bgr(image: np.ndarray, preprocess: dict) -> np.ndarray:
    """Match src/autococ/model_vision.py::_prepare exactly."""
    height, width = image.shape[:2]
    out_w, out_h = preprocess["width"], preprocess["height"]
    factor = min(out_w / width, out_h / height)
    scaled_w, scaled_h = max(1, round(width * factor)), max(1, round(height * factor))
    dx, dy = (out_w - scaled_w) // 2, (out_h - scaled_h) // 2
    canvas = np.full((out_h, out_w, 3), preprocess["pad_value"], dtype=np.uint8)
    canvas[dy:dy + scaled_h, dx:dx + scaled_w] = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    mean = np.asarray(preprocess["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(preprocess["std"], dtype=np.float32).reshape(1, 1, 3)
    return np.ascontiguousarray(((canvas.astype(np.float32) * preprocess["scale"] - mean) / std).transpose(2, 0, 1))


def _config(snapshot: dict, task: str, config_path: Path, *, prototype: bool) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("task") != task:
        raise TrainingError("Training config task mismatch")
    for key in ("client_versions", "source_width_range", "source_height_range", "zoom", "data_version", "code_version", "output"):
        if not config.get(key):
            raise TrainingError(f"Training config requires {key}")
    metadata = snapshot["metadata"]
    if config["data_version"] != metadata["data_version"] or sorted(config["client_versions"]) != sorted(metadata["client_versions"]):
        raise TrainingError("Config data/client version differs from snapshot")
    if metadata["prototype_only"] and not prototype:
        raise TrainingError("Prototype snapshot cannot be used for formal training")
    if config.get("prototype_only") is not True and prototype:
        config["prototype_only"] = True
    if not prototype and config.get("prototype_only") is True:
        raise TrainingError("Prototype config cannot run as formal training")
    if task == "scene":
        classes = config.get("classes")
        taxonomy = json.loads((Path(snapshot["root"]) / "taxonomy.json").read_text(encoding="utf-8"))
        if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes) or not set(classes) <= set(taxonomy["scene_classes"]):
            raise TrainingError("Scene classes must be an explicit taxonomy subset of at least two")
        config["preprocess"] = SCENE_PREPROCESS
    else:
        mapping = json.loads((Path(snapshot["root"]) / "class_mapping.json").read_text(encoding="utf-8"))["output_indices"]
        config["classes"] = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
        config["preprocess"] = BUILDING_PREPROCESS
    return config


def _scene_rows(root: Path, snapshot: dict, classes: list[str]) -> dict[str, list[dict]]:
    rows = read_jsonl(root / "reviewed-scenes.jsonl")
    by_image = {row["image"]: row for row in rows}
    if len(by_image) != len(rows):
        raise TrainingError("Duplicate reviewed image")
    splits = {}
    unknown = {}
    for name, paths in snapshot["paths"].items():
        selected, rejected = [], []
        for path in paths:
            row = by_image.get(path)
            if row is None:
                raise TrainingError(f"Missing or out-of-scope reviewed scene: {path}")
            if row.get("label") == "unknown":
                rejected.append(row)
            elif row.get("label") in classes:
                selected.append(row)
            else:
                raise TrainingError(f"Out-of-scope reviewed scene: {path}")
        splits[name] = selected
        unknown[name] = rejected
    train_groups = {r["group_id"] for r in splits["train"] + unknown["train"]}
    val_groups = {r["group_id"] for r in splits["val"] + unknown["val"]}
    test_groups = {r["group_id"] for r in splits["test"] + unknown["test"]}
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise TrainingError("Leakage groups cross splits")
    splits["unknown"] = unknown
    return splits


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment() -> dict:
    import importlib.metadata
    import platform
    import sys
    names = ("torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "numpy", "opencv-python")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    installed = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
                       if dist.metadata.get("Name"))
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "installed_versions": installed}


def _signature(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _offline_ultralytics(config_dir: Path):
    """Keep Ultralytics settings in this run and disable outbound logging."""
    import socket
    settings_dir = config_dir / "Ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    disabled_keys = ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")
    # Preseed before first import: installed Ultralytics migrates this partial
    # settings file while preserving valid false values. Import defaults to
    # sync=true if no settings file exists.
    (settings_dir / "settings.json").write_text(json.dumps({key: False for key in disabled_keys}), encoding="utf-8")
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["WANDB_MODE"] = "disabled"
    def blocked_connect(*_args, **_kwargs):
        raise OSError("AutoCOC vision training forbids network connections")
    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect
    from ultralytics import YOLO, settings
    disabled = {name: False for name in disabled_keys if name in settings}
    settings.update(disabled)
    return YOLO


def _rng_state() -> dict:
    import torch
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict) -> None:
    import torch
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _scene_model(classes: list[str], pretrained: Path | None):
    import torch
    from torchvision.models import mobilenet_v3_small
    model = mobilenet_v3_small(weights=None)
    source_hash = None
    if pretrained is not None:
        if not pretrained.is_file():
            raise TrainingError(f"Local pretrained weights missing: {pretrained}")
        source_hash = sha256(pretrained)
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    return model, source_hash


def _metric(truth: list[int], predicted: list[int], classes: list[str]) -> dict:
    recalls, f1s = [], []
    per_class = {}
    for index, name in enumerate(classes):
        tp = sum(a == index and b == index for a, b in zip(truth, predicted))
        fp = sum(a != index and b == index for a, b in zip(truth, predicted))
        fn = sum(a == index and b != index for a, b in zip(truth, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
        if tp + fn:
            recalls.append(recall)
            f1s.append(f1)
    return {"macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "minimum_recall": min(recalls) if recalls else 0.0, "per_class": per_class}


def _validate_scene_mode(prototype: bool, development: bool, max_epochs: int | None) -> None:
    if max_epochs is not None and max_epochs <= 0:
        raise TrainingError("Epoch cap must be positive")
    if development and not prototype:
        raise TrainingError("Development training requires a prototype snapshot and --prototype")
    if development and (max_epochs is None or max_epochs > 35):
        raise TrainingError("Development training requires an explicit epoch cap of at most 35")
    if prototype and not development and max_epochs is not None and max_epochs > 2:
        raise TrainingError("Prototype smoke is limited to two epochs")


def _stage_building_inputs(root: Path, output: Path, paths: list[str]) -> list[Path]:
    """Keep Ultralytics label caches beside disposable run copies, not the snapshot."""
    staged = output / "staged"
    image_paths = []
    for path in paths:
        relative = Path(path)
        if relative.parts[0] != "images" or ".." in relative.parts:
            raise TrainingError(f"Invalid frozen image path: {path}")
        label = Path("labels") / relative.relative_to("images").with_suffix(".txt")
        for source, target in ((root / relative, staged / relative),
                               (root / label, staged / label)):
            if not source.is_file():
                raise TrainingError(f"Frozen input missing: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha256(target) != sha256(source):
                    raise TrainingError(f"Staged training input differs from snapshot: {target}")
            else:
                shutil.copyfile(source, target)
        image_paths.append(staged / relative)
    return image_paths


def train_scene(snapshot_root: Path, config_path: Path, output: Path, *, device: str, seed: int,
                resume: Path | None, prototype: bool, max_epochs: int | None,
                deadline_seconds: float, pretrained: Path | None,
                development: bool = False) -> dict:
    _validate_scene_mode(prototype, development, max_epochs)
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    if device.isdigit():
        device = f"cuda:{device}"
    snapshot = verify_snapshot(snapshot_root, "scene")
    root = Path(snapshot["root"])
    config = _config(snapshot, "scene", config_path, prototype=prototype)
    if not prototype and pretrained is None:
        raise TrainingError("Formal scene training requires local ImageNet pretrained weights")
    classes = config["classes"]
    splits = _scene_rows(root, snapshot, classes)
    if not prototype and set(classes) - set(row["label"] for row in splits["train"]):
        raise TrainingError("Every enabled class needs training support")
    if not prototype and (set(classes) - set(row["label"] for row in splits["val"])):
        raise TrainingError("Every enabled class needs validation support")
    _seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = output.resolve()
    if output.exists() and resume is None:
        raise TrainingError(f"Output already exists: {output}")
    config.update({"seed": seed, "device": device, "snapshot_sha256": snapshot["snapshot_sha256"],
                   "split_sha256": snapshot["split_sha256"], "pretrained_sha256": sha256(pretrained) if pretrained else None,
                   "architecture": "mobilenet_v3_small", "prototype_only": prototype,
                   "development": development,
                   "max_epochs": max_epochs or (2 if prototype else 35), "deadline_seconds": deadline_seconds})
    resume_identity = {key: value for key, value in config.items() if key != "deadline_seconds"}
    signature = hashlib.sha256(json.dumps(resume_identity, sort_keys=True).encode()).hexdigest()
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("signature") != signature:
            raise TrainingError("Resume config or snapshot differs from checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(_environment(), indent=2), encoding="utf-8")
    model, _ = _scene_model(classes, pretrained)
    model.to(device)

    class Images(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            path = root / row["image"]
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise TrainingError(f"Cannot decode {path}")
            return torch.from_numpy(prepare_bgr(image, SCENE_PREPROCESS)), classes.index(row["label"])

    train_rows, val_rows = splits["train"], splits["val"]
    if prototype and not development:
        train_rows = train_rows[:32]
        val_rows = val_rows[:32]
    batch = int(config.get("batch", 8 if prototype else 32))
    if batch <= 0:
        raise TrainingError("Batch must be positive")
    # A single sampler addresses both class imbalance and adjacent frames in
    # one leakage group; do not stack a second class-weighted loss on top.
    group_counts = {group: sum(row["group_id"] == group for row in train_rows)
                    for group in {row["group_id"] for row in train_rows}}
    class_counts = {label: sum(row["label"] == label for row in train_rows)
                    for label in {row["label"] for row in train_rows}}
    sample_weights = torch.as_tensor([1 / (group_counts[row["group_id"]] * class_counts[row["label"]])
                                      for row in train_rows], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(Images(train_rows), batch_size=batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(Images(val_rows), batch_size=batch, shuffle=False, num_workers=0)
    head_epochs = 5 if development or not prototype else 0
    total_epochs = int(config["max_epochs"])
    optimizer, scheduler = None, None
    start_epoch, best_score, bad_epochs, history = 0, (-1.0, -1.0), 0, []
    if resume:
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = tuple(checkpoint["best_score"])
        bad_epochs = checkpoint["bad_epochs"]
        history = checkpoint["history"]
        _restore_rng(checkpoint["rng"])
    began = time.monotonic()
    for epoch in range(start_epoch, total_epochs):
        if time.monotonic() - began >= deadline_seconds:
            break
        phase = "head" if epoch < head_epochs else "finetune"
        for parameter in model.features.parameters():
            parameter.requires_grad = phase == "finetune"
        if optimizer is None or epoch == head_epochs:
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                          lr=0.001 if phase == "head" else 0.0001, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            if resume and epoch == start_epoch and checkpoint.get("phase") == phase:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise TrainingError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        truth, predicted = [], []
        with torch.inference_mode():
            for images, labels in val_loader:
                result = model(images.to(device)).argmax(1).cpu().tolist()
                predicted.extend(result)
                truth.extend(labels.tolist())
        metrics = _metric(truth, predicted, classes)
        score = (metrics["macro_f1"], metrics["minimum_recall"])
        scheduler.step()
        better = score > best_score
        best_score, bad_epochs = (score, 0) if better else (best_score, bad_epochs + 1)
        record = {"epoch": epoch, "phase": phase, "train_loss": sum(losses) / len(losses),
                  "validation": metrics, "elapsed_seconds": time.monotonic() - began}
        history.append(record)
        state = {"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "phase": phase, "best_score": best_score,
                 "bad_epochs": bad_epochs, "history": history, "rng": _rng_state()}
        torch.save(state, output / "last.pt")
        if better:
            torch.save(state, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if (development or not prototype) and phase == "finetune" and bad_epochs >= 7:
            break
    result = {"task": "scene", "prototype_only": prototype, "epochs_completed": len(history),
              "train_images": len(train_rows), "validation_images": len(val_rows),
              "unknown_train_excluded": len(splits["unknown"]["train"]),
              "unknown_validation_reserved": len(splits["unknown"]["val"]),
              "unknown_test_reserved": len(splits["unknown"]["test"]),
              "t2_smoke_complete": bool(prototype and not development and len(train_rows) == 32 and len(history) == 2),
              "development": development,
              "best_score": best_score, "checkpoint": str(output / "best.pt"),
              "last_checkpoint": str(output / "last.pt"), "snapshot_sha256": snapshot["snapshot_sha256"],
              "status": "prototype_only" if prototype else "trained_unevaluated"}
    (output / "training_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_building(snapshot_root: Path, config_path: Path, output: Path, *, device: str,
                   seed: int, resume: Path | None, prototype: bool, max_epochs: int | None,
                   deadline_seconds: float, pretrained: Path | None,
                   development: bool = False) -> dict:
    if development and not prototype:
        raise TrainingError("Building development requires a prototype snapshot and --prototype")
    if development and (max_epochs is None or max_epochs <= 0 or max_epochs > 100):
        raise TrainingError("Building development requires an explicit epoch cap of 1..100")
    snapshot = verify_snapshot(snapshot_root, "building")
    root = Path(snapshot["root"])
    config = _config(snapshot, "building", config_path, prototype=prototype)
    if pretrained is None or not pretrained.is_file():
        raise TrainingError("Building training requires a local pretrained YOLO checkpoint")
    if config.get("architecture") not in {"yolo11n", "yolo26n"}:
        raise TrainingError("Building architecture must be yolo11n or yolo26n")
    output = output.resolve()
    if output.exists() and resume is None:
        raise TrainingError(f"Output already exists: {output}")
    config.update({"seed": seed, "device": device, "snapshot_sha256": snapshot["snapshot_sha256"],
                   "split_sha256": snapshot["split_sha256"], "pretrained_sha256": sha256(pretrained),
                   "prototype_only": prototype, "development": development,
                   "max_epochs": max_epochs or (2 if prototype else 100),
                   "deadline_seconds": deadline_seconds})
    if prototype and not development and config["max_epochs"] > 2:
        raise TrainingError("Prototype smoke is limited to two epochs")
    if development and int(config.get("batch", 2)) > 8:
        raise TrainingError("Building development batch must be at most 8")
    environment = _environment()
    recipe = {"imgsz": 640, "batch": int(config.get("batch", 2 if prototype else 8)),
              "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005,
              "patience": 20, "amp": False, "workers": 0, "cache": False,
              "rect": False, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
              "degrees": 0.0, "translate": 0.0, "scale": 0.0,
              "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
              "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0,
              "copy_paste": 0.0, "save": True, "val": True,
              "deterministic": True}
    # The resume state is only reusable with the exact frozen data, model,
    # classes, seed, optimizer settings and trainer version. Deadline is a
    # run-time cap rather than a training hyperparameter.
    identity = {"config": {key: value for key, value in config.items() if key != "deadline_seconds"},
                "recipe": recipe,
                "ultralytics_version": environment["packages"]["ultralytics"],
                "training_recipe_version": 2}
    signature = _signature(identity)
    resumable = output / "resume.pt"
    resume_metadata = output / "resume.metadata.json"
    if resume:
        resume = resume.resolve()
        if resume != resumable or not resume.is_file() or not resume_metadata.is_file():
            raise TrainingError("Building resume requires this run's resume.pt and metadata")
        saved = json.loads(resume_metadata.read_text(encoding="utf-8"))
        if (saved.get("signature") != signature or saved.get("checkpoint_sha256") != sha256(resume)):
            raise TrainingError("Building resume snapshot/config/version or checkpoint differs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    YOLO = _offline_ultralytics(output / "ultralytics_config")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    checkpoint = resume or pretrained
    model = YOLO(str(checkpoint))
    data_yaml = output / "data.yaml"
    mapping = json.loads((root / "class_mapping.json").read_text(encoding="utf-8"))["output_indices"]
    # Ultralytics writes labels.cache beside labels even with cache=False.
    # Stage byte-identical inputs inside the run so the frozen snapshot stays
    # read-only; absolute paths avoid working-directory interpretation.
    selected_counts = {}
    for name in ("train", "val"):
        paths = snapshot["paths"][name][:32] if prototype and not development else snapshot["paths"][name]
        selected_counts[name] = len(paths)
        staged_paths = _stage_building_inputs(root, output, paths)
        (output / f"{name}.txt").write_text("".join(str(path) + "\n" for path in staged_paths), encoding="utf-8")
    data_yaml.write_text("path: " + json.dumps(str(output / "staged")) + "\ntrain: " + json.dumps(str(output / "train.txt")) +
                         "\nval: " + json.dumps(str(output / "val.txt")) + "\nnames:\n" +
                         "".join(f"  {idx}: {name}\n" for name, idx in sorted(mapping.items(), key=lambda item: item[1])), encoding="utf-8")
    args = {"data": str(data_yaml), "epochs": int(config["max_epochs"]), "device": device,
            "project": str(output), "name": "run", "exist_ok": bool(resume),
            "resume": bool(resume), "seed": seed, **recipe}
    (output / "effective_train_args.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
    began = time.monotonic()
    def preserve_resume_state(trainer):
        # Final Ultralytics cleanup strips optimizer from best.pt/last.pt.
        # on_model_save runs immediately after the unstripped last.pt write.
        shutil.copyfile(trainer.last, resumable)
        resume_metadata.write_text(json.dumps({"signature": signature,
                                               "checkpoint_sha256": sha256(resumable),
                                               "snapshot_sha256": snapshot["snapshot_sha256"],
                                               "split_sha256": snapshot["split_sha256"],
                                               "trainer_version": environment["packages"]["ultralytics"]},
                                              indent=2), encoding="utf-8")
    def stop_after_deadline(trainer):
        # Ultralytics' `time=` overrides `epochs`, so enforce both bounds at
        # the fit-epoch boundary after its checkpoint has been saved.
        if time.monotonic() - began >= deadline_seconds:
            trainer.stop = True
    model.add_callback("on_model_save", preserve_resume_state)
    model.add_callback("on_fit_epoch_end", stop_after_deadline)
    model.train(**args)
    weights = output / "run" / "weights"
    results_csv = output / "run" / "results.csv"
    epochs_completed = max(0, len(results_csv.read_text(encoding="utf-8").splitlines()) - 1) if results_csv.is_file() else 0
    result = {"task": "building", "prototype_only": prototype, "development": development,
              "train_images": selected_counts["train"], "validation_images": selected_counts["val"],
              "epochs_completed": epochs_completed,
              "t2_smoke_complete": bool(prototype and not development and selected_counts["train"] == 32 and epochs_completed == 2),
              "best_checkpoint": str(weights / "best.pt"), "last_checkpoint": str(weights / "last.pt"),
              "resume_checkpoint": str(resumable),
              "snapshot_sha256": snapshot["snapshot_sha256"],
              "status": "prototype_only" if prototype else "trained_unevaluated"}
    (output / "training_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
