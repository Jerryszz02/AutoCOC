"""Read-only acceptance evidence summary for one AutoCOC daily run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autococ.daily_evidence import audit_daily_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="One original run-*.json report")
    parser.add_argument("--output", type=Path, help="New summary JSON path; never overwrites")
    args = parser.parse_args()
    summary = audit_daily_report(args.report)
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
