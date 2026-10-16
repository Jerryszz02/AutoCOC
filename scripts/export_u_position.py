"""Export a trained U-position detector to CPU FP32 ONNX with real-loader parity."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.vision.snapshot import TrainingError
from training.vision.u_export import export_u_position


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    try:
        report = export_u_position(args.training_dir, args.snapshot, args.package, opset=args.opset)
    except (TrainingError, OSError, ValueError) as exc:
        parser.exit(2, f"U export failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
