"""Train S/B vision models from a reviewed, frozen local snapshot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.vision.train import TrainingError, train_building, train_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("scene", "building"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Explicit torch/Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pretrained", type=Path, help="Local weights only; never downloaded")
    parser.add_argument("--prototype", action="store_true", help="32 training/validation images, <=2 epochs")
    parser.add_argument("--development", action="store_true", help="Prototype only: full frozen train/val; scene <=35 or building <=100 epochs; never formal")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--deadline-hours", type=float, default=4)
    args = parser.parse_args()
    if args.max_epochs is not None and args.max_epochs <= 0 or args.deadline_hours <= 0:
        parser.error("epochs and deadline must be positive")
    if args.development and (not args.prototype or args.max_epochs is None or
                             args.max_epochs > (35 if args.task == "scene" else 100)):
        parser.error("--development requires --prototype and --max-epochs within task limit")
    trainer = train_scene if args.task == "scene" else train_building
    try:
        report = trainer(args.snapshot, args.config, args.output, device=args.device,
                         seed=args.seed, resume=args.resume, prototype=args.prototype,
                         max_epochs=args.max_epochs, deadline_seconds=args.deadline_hours * 3600,
                         pretrained=args.pretrained, development=args.development)
    except (TrainingError, OSError, ValueError) as exc:
        parser.exit(2, f"training failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
