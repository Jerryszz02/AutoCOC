"""Create a new source-bound, prototype-only U capture metadata queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.vision.u_adjudication import adjudicate_capture_metadata
from training.vision.u_candidates import CandidateError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["prototype"], required=True)
    args = parser.parse_args(argv)
    try:
        result = adjudicate_capture_metadata(args.parent_queue, args.decisions, args.output)
    except (CandidateError, OSError, ValueError) as exc:
        print(f"u-capture-adjudication: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
