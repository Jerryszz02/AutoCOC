"""Export a trained S/B checkpoint as a loadable local CPU FP32 ONNX package."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.vision.export import export_package
from training.vision.train import TrainingError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    try:
        report = export_package(args.training_dir, args.snapshot, args.package,
                                checkpoint=args.checkpoint, opset=args.opset,
                                atol=args.atol, rtol=args.rtol)
    except (TrainingError, OSError, ValueError) as exc:
        parser.exit(2, f"export failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
