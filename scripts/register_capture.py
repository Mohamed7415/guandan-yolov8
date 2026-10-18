#!/usr/bin/env python3
"""Register existing, real PNG/JPEG files without capturing or re-encoding images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from capture_common import FIELDS, IMAGE_FORMATS, inspect_image, manifest_path, parse_scenes, read_manifest, validate_timestamp

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="Existing PNG/JPG/JPEG paths; no image is generated or re-encoded.")
    parser.add_argument("--session-id", required=True, help="One complete game/round identifier.")
    parser.add_argument("--group-id", required=True, help="One nearby-frame group; use a session prefix.")
    parser.add_argument("--scene", required=True, help="One or more scene names separated by |.")
    parser.add_argument("--captured-at", help="ISO 8601 with timezone; default: file mtime, explicitly reported as inferred.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/capture_manifest.csv")
    parser.add_argument("--copy-to-raw", action="store_true", help="Copy external files into raw_dir using original filenames; never overwrite.")
    args = parser.parse_args()
    try:
        if not args.session_id.strip() or not args.group_id.strip():
            raise ValueError("session_id and group_id must not be empty")
        scene = "|".join(parse_scenes(args.scene))
        explicit_time = validate_timestamp(args.captured_at) if args.captured_at else None
        raw_dir = args.raw_dir.resolve()
        rows, errors = read_manifest(args.manifest)
        if errors and errors != ["manifest_missing"]:
            raise ValueError(", ".join(errors))
        by_name = {}
        for row in rows:
            filename = row.get("filename") or ""
            manifest_path(raw_dir, filename)
            if filename in by_name:
                raise ValueError(f"duplicate manifest filename: {filename}")
            by_name[filename] = {field: row.get(field, "") for field in FIELDS}
        pending = []
        for source_arg in args.images:
            source = source_arg.resolve(strict=True)
            if not source.is_file() or source.suffix.lower() not in IMAGE_FORMATS:
                raise ValueError(f"expected an existing PNG/JPG/JPEG file: {source}")
            destination = source if source.is_relative_to(raw_dir) else raw_dir / source.name
            if source != destination and not args.copy_to_raw:
                raise ValueError(f"{source} is outside raw_dir; save there or use --copy-to-raw")
            if source != destination and destination.exists():
                raise ValueError(f"refusing to overwrite {destination}")
            if any(item[1] == destination for item in pending):
                raise ValueError(f"same destination appears twice: {destination}")
            info = inspect_image(source)
            if info["error"]:
                raise ValueError(f"{source}: {info['error']}")
            filename = destination.relative_to(raw_dir).as_posix()
            for other_name, other_row in by_name.items():
                if other_name != filename and other_row["sha256"] == info["sha256"]:
                    raise ValueError(f"exact duplicate: {filename} matches {other_name}")
            row = dict(filename=filename, session_id=args.session_id.strip(), group_id=args.group_id.strip(),
                       scene=scene, captured_at=explicit_time or datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                       sha256=info["sha256"], width=str(info["width"]), height=str(info["height"]))
            by_name[filename] = row
            pending.append((source, destination, row))
        # Validate the complete batch before any copies or manifest writes.
        raw_dir.mkdir(parents=True, exist_ok=True)
        for source, destination, _ in pending:
            if source != destination:
                shutil.copy2(source, destination)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=args.manifest.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(by_name[name] for name in sorted(by_name))
        os.replace(temporary, args.manifest)
        print(json.dumps({"registered": len(pending), "manifest": str(args.manifest.resolve()),
                          "filenames": [row["filename"] for _, _, row in pending],
                          "timestamp_source": "provided" if explicit_time else "file_mtime_inferred"}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Registration failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
