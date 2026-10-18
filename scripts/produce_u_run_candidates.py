"""Produce a pending-only U queue from a prospectively scoped normal CLI run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.vision.u_candidates import CandidateError
from training.vision.u_run_capture import produce_run_candidate_queue


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New external directory; existing paths are refused")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Write source-bound plan/capture/batch without copying PNGs")
    args = parser.parse_args(argv)
    try:
        result = produce_run_candidate_queue(
            args.report, args.launch, args.binding, args.prospective, args.output,
            manifest_only=args.manifest_only,
        )
    except (CandidateError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"u-run-candidates: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
