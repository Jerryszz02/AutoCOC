"""Train the independent U-position detector from a reviewed frozen dynamic snapshot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.vision.snapshot import TrainingError
from training.vision.u_train import train_u_position


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--pretrained", type=Path, required=True, help="Local yolo26n.pt; no download")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--deadline-hours", type=float, default=4)
    args = parser.parse_args()
    try:
        report = train_u_position(args.snapshot, args.config, args.output, device=args.device,
                                  pretrained=args.pretrained, seed=args.seed, resume=args.resume,
                                  max_epochs=args.max_epochs, deadline_seconds=args.deadline_hours * 3600)
    except (TrainingError, OSError, ValueError) as exc:
        parser.exit(2, f"U training failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
