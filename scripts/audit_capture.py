#!/usr/bin/env python3
"""Audit original PNG/JPEG files; only unique valid registered captures count."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from capture_common import IMAGE_FORMATS, SCENES, inspect_image, manifest_path, parse_scenes, read_manifest, validate_timestamp

ROOT = Path(__file__).resolve().parents[1]


def audit(raw_dir: Path, manifest: Path, target: int = 200, near_distance: int = 6, max_pairs: int = 1000) -> dict:
    raw_dir = raw_dir.resolve()
    rows, manifest_errors = read_manifest(manifest)
    files = []
    for path in sorted(raw_dir.rglob("*")) if raw_dir.exists() else []:
        if not path.is_file() or path.suffix.lower() not in IMAGE_FORMATS:
            continue
        name = path.relative_to(raw_dir).as_posix()
        if not path.resolve().is_relative_to(raw_dir):
            files.append({"filename": name, "error": "symlink_outside_raw_dir"})
            continue
        try:
            files.append({"filename": name, **inspect_image(path)})
        except OSError as exc:
            files.append({"filename": name, "error": f"read_error:{type(exc).__name__}"})
    by_name = {item["filename"]: item for item in files}
    valid_images = [item for item in files if not item["error"]]
    hash_groups = defaultdict(list)
    for item in valid_images:
        hash_groups[item["pixel_sha256"]].append(item["filename"])
    exact_duplicates = [{"pixel_sha256": sha, "filenames": names,
                         "file_sha256": [by_name[name]["sha256"] for name in names]}
                        for sha, names in sorted(hash_groups.items()) if len(names) > 1]
    row_names = Counter(row.get("filename") or "" for row in rows)
    manifest_issues = []
    eligible = []
    covered = set()
    for index, row in enumerate(rows, start=2):
        filename = row.get("filename") or ""
        issues = []
        try:
            manifest_path(raw_dir, filename)
            covered.add(filename)
        except ValueError as exc:
            issues.append(str(exc))
        if row_names[filename] > 1:
            issues.append("duplicate_manifest_row")
        item = by_name.get(filename)
        if item is None:
            issues.append("file_missing")
        elif item["error"]:
            issues.append(item["error"])
        else:
            for field in ("sha256", "width", "height"):
                if str(row.get(field, "")) != str(item[field]):
                    issues.append(f"{field}_mismatch")
        for field in ("session_id", "group_id"):
            if not (row.get(field) or "").strip():
                issues.append(f"{field}_missing")
        try:
            scenes = parse_scenes(row.get("scene") or "")
        except ValueError:
            scenes = []
            issues.append("scene_invalid_or_missing")
        try:
            validate_timestamp(row.get("captured_at") or "")
        except (ValueError, TypeError):
            issues.append("captured_at_invalid_or_missing")
        if issues:
            manifest_issues.append({"row": index, "filename": filename, "issues": issues})
        elif not manifest_errors:
            eligible.append((item, row, scenes))
    unique_registered = {}
    for item, row, scenes in eligible:
        unique_registered.setdefault(item["pixel_sha256"], (item, row, scenes))
    session_counts = Counter()
    scene_counts = Counter()
    session_scene_counts = defaultdict(Counter)
    for _, row, scenes in unique_registered.values():
        session_counts[row["session_id"]] += 1
        for scene in scenes:
            scene_counts[scene] += 1
            session_scene_counts[row["session_id"]][scene] += 1
    near_pairs = []
    near_total = 0
    # dHash is an advisory review signal: matching table backgrounds can look alike.
    representatives = [by_name[names[0]] for names in hash_groups.values()]
    for index, left in enumerate(representatives):
        for right in representatives[index + 1:]:
            distance = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
            if distance <= near_distance:
                near_total += 1
                if len(near_pairs) < max_pairs:
                    near_pairs.append({"left": left["filename"], "right": right["filename"], "distance": distance})
    unique_count = len(unique_registered)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir), "manifest": str(manifest.resolve()), "target": target,
        "counts": {"image_files": len(files), "valid_nonblank_images": len(valid_images),
                   "unique_valid_images": len(hash_groups), "registered_unique_valid_images": unique_count,
                   "remaining": max(0, target - unique_count), "invalid_images": len(files) - len(valid_images),
                   "exact_duplicate_extra_files": sum(len(names) - 1 for names in hash_groups.values()),
                   "unregistered_files": len(set(by_name) - covered), "near_duplicate_pairs": near_total},
        "target_reached": unique_count >= target,
        "session_counts": dict(sorted(session_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "missing_scene_coverage": [scene for scene in SCENES if not scene_counts[scene]],
        "session_scene_counts": {session: dict(sorted(counts.items())) for session, counts in sorted(session_scene_counts.items())},
        "manifest_errors": manifest_errors, "manifest_issues": manifest_issues,
        "unregistered_files": sorted(set(by_name) - covered),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": {"method": "64-bit dHash", "max_distance": near_distance,
                            "reported_pairs": len(near_pairs), "truncated": near_total > len(near_pairs),
                            "advisory_only": True, "pairs": near_pairs},
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/capture_manifest.csv")
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--near-distance", type=int, default=6)
    parser.add_argument("--max-near-pairs", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/capture_audit.json")
    parser.add_argument("--require-target", action="store_true", help="Exit 1 if target not reached or manifest/file errors exist.")
    args = parser.parse_args()
    if args.target < 1 or not 0 <= args.near_distance <= 64 or args.max_near_pairs < 0:
        parser.error("target must be positive, near-distance in [0,64], max-near-pairs nonnegative")
    try:
        report = audit(args.raw_dir, args.manifest, args.target, args.near_distance, args.max_near_pairs)
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "target_reached": report["target_reached"],
                      "session_counts": report["session_counts"], "scene_counts": report["scene_counts"],
                      "missing_scene_coverage": report["missing_scene_coverage"],
                      "manifest_errors": report["manifest_errors"], "report": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    errors = (not report["target_reached"] or report["manifest_errors"] or report["manifest_issues"]
              or report["counts"]["invalid_images"] or report["counts"]["unregistered_files"])
    return int(bool(args.require_target and errors))


if __name__ == "__main__":
    raise SystemExit(main())
