"""Export a trained offline R-state checkpoint to a CPU FP32 ONNX package."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.vision.dynamic_export import export_dynamic_state
from training.vision.snapshot import TrainingError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    try:
        report = export_dynamic_state(args.training_dir, args.snapshot, args.package,
                                      opset=args.opset)
    except (TrainingError, OSError, ValueError) as exc:
        parser.exit(2, f"R export failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
