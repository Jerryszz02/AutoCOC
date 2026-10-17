"""Inspect, validate, split and export reviewed vision data without touching originals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.vision.data import (DatasetError, export, import_annotations,
                                  import_legacy, inspect, load_taxonomy, read_jsonl,
                                  split, stage_sources, validate, write_jsonl)
from training.vision.picker_candidates import import_picker_candidates


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("import-legacy", "import-annotations", "import-picker-candidates", "stage", "inspect", "validate", "split", "export"):
        command = commands.add_parser(name)
        if name == "import-picker-candidates":
            command.add_argument("--source-run", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            continue
        if name == "stage":
            command.add_argument("--source", nargs=2, metavar=("ROOT", "MANIFEST"),
                                 type=Path, action="append", required=True)
            command.add_argument("--output", type=Path, required=True)
            continue
        if name == "import-annotations":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--base", type=Path, help="Existing annotations JSONL")
            command.add_argument("--incoming", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            continue
        command.add_argument("--root", type=Path, required=True, help="Original image root (read only)")
        if name == "import-legacy":
            command.add_argument("--legacy", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            continue
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--annotations", type=Path, required=True)
        command.add_argument("--taxonomy", type=Path)
        if name in ("inspect", "validate", "split"):
            command.add_argument("--output", type=Path, help="Write JSON report without replacing existing file")
        if name == "inspect":
            command.add_argument("--near-duplicates", action="store_true")
        if name == "validate":
            command.add_argument("--skip-pixel-hash", action="store_true", help="Fast metadata check only")
        if name in ("split", "export"):
            command.add_argument("--task", choices=("scene", "building"), required=True)
        if name == "split":
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--classes", nargs="+", help="Building classes reviewed exhaustively in every selected image; fixes output order")
            command.add_argument("--prototype-only", action="store_true",
                                 help="Allow reviewed but version-unknown images for pipeline smoke tests")
            command.add_argument("--near-duplicate-review", type=Path,
                                 help="JSON list of source-bound visual verdicts for dHash candidates")
        if name == "export":
            command.add_argument("--split", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--data-version", required=True)
            command.add_argument("--classes", nargs="+", help="Building output classes in explicit index order")
            command.add_argument("--prototype-only", action="store_true",
                                 help="Mark exported snapshot as incompatible with formal model acceptance")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "import-legacy":
            report = import_legacy(args.root, args.legacy, args.output)
        elif args.command == "import-picker-candidates":
            report = import_picker_candidates(args.source_run, args.output)
        elif args.command == "stage":
            report = stage_sources(args.source, args.output)
        elif args.command == "import-annotations":
            existing = read_jsonl(args.base) if args.base else []
            merged = import_annotations(existing, read_jsonl(args.incoming), read_jsonl(args.manifest))
            if args.output.exists():
                raise DatasetError(f"Refusing to overwrite annotations: {args.output}")
            write_jsonl(args.output, merged)
            report = {"annotation_count": len(merged), "output": str(args.output)}
        else:
            manifest, annotations = read_jsonl(args.manifest), read_jsonl(args.annotations)
            taxonomy = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
            if args.command == "inspect":
                report = inspect(args.root, manifest, annotations, near_duplicates=args.near_duplicates)
            elif args.command == "validate":
                report = validate(args.root, manifest, annotations, taxonomy=taxonomy,
                                  verify_pixels=not args.skip_pixel_hash)
            elif args.command == "split":
                checks = validate(args.root, manifest, annotations, taxonomy=taxonomy, verify_pixels=True)
                if not checks["valid"]:
                    raise DatasetError("Validation failed: " + "; ".join(checks["errors"][:10]))
                report = split(args.root, manifest, annotations, task=args.task, seed=args.seed,
                               prototype_only=args.prototype_only,
                               classes=args.classes, taxonomy=taxonomy,
                               near_duplicate_reviews=(json.loads(args.near_duplicate_review.read_text(encoding="utf-8"))
                                                       if args.near_duplicate_review else None))
            else:
                checks = validate(args.root, manifest, annotations, taxonomy=taxonomy, verify_pixels=True)
                if not checks["valid"]:
                    raise DatasetError("Validation failed: " + "; ".join(checks["errors"][:10]))
                split_doc = json.loads(args.split.read_text(encoding="utf-8"))
                report = export(args.root, manifest, annotations, split_doc, args.output,
                                task=args.task, data_version=args.data_version,
                                classes=args.classes, taxonomy=taxonomy,
                                prototype_only=args.prototype_only)
        if args.command in ("inspect", "validate", "split") and args.output:
            if args.output.exists():
                raise DatasetError(f"Refusing to overwrite report: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if args.command == "validate" and not report["valid"] else 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"vision-data: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
