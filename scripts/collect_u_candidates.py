"""Build a hash-bound U candidate source pool from preplanned live captures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.vision.u_candidates import CandidateError, build_candidate_queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True,
                        help="u_candidate_batch_v1 JSON with predeclared plans")
    parser.add_argument("--output", type=Path, required=True,
                        help="New external directory; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_candidate_queue(args.batch, args.output),
                         ensure_ascii=False, indent=2))
        return 0
    except (CandidateError, OSError, ValueError) as exc:
        print(f"u-candidates: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
