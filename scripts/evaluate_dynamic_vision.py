"""Evaluate a separate R/U package on reviewed event windows, offline only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autococ.dynamic_vision import DynamicModel, DynamicUnavailable
from training.vision.data import DatasetError, read_jsonl, sha256
from training.vision.dynamic_evaluation import evaluate_dynamic


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("package", type=Path)
    cli.add_argument("--client-version", required=True)
    cli.add_argument("--source-root", type=Path, required=True)
    cli.add_argument("--windows", type=Path, required=True)
    cli.add_argument("--split", choices=("train", "val", "test"), default="test")
    cli.add_argument("--prototype-only", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    try:
        if args.output.exists():
            raise DatasetError(f"Refusing to overwrite dynamic report: {args.output}")
        model = DynamicModel(args.package, client_version=args.client_version)
        report = evaluate_dynamic(model, args.source_root, read_jsonl(args.windows),
                                  prototype_only=args.prototype_only, split=args.split)
        report["reviewed_windows_sha256"] = sha256(args.windows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": report["status"],
                          "metrics": report["metrics"]}, ensure_ascii=False))
        return 0
    except (DatasetError, DynamicUnavailable, OSError, ValueError) as exc:
        print(f"dynamic-evaluation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
