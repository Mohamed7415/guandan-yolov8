#!/usr/bin/env python3
"""Build a YOLO dataset, keeping every game/sequence group in one split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from card_common import CLASS_NAMES, manifest_rows, read_labels


def prepare(manifest: Path, output: Path, ratios: list[float], seed: int) -> dict:
    from PIL import Image

    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("Train/val/test fractions must be positive and sum to 1")
    rows = manifest_rows(manifest)
    has_sessions = any((row.get("session_id") or "").strip() for row in rows)
    if has_sessions and any(not (row.get("session_id") or "").strip() for row in rows):
        raise ValueError("When session_id is used, every reviewed row must have one")
    groups = defaultdict(list)
    for row in rows:
        read_labels(Path(row["label"]))
        with Image.open(row["image"]) as image:
            image.verify()
        split_key = row["session_id"].strip() if has_sessions else row["group_id"]
        groups[split_key].append(row)
    if len(groups) < 3:
        raise ValueError("Need at least three independent game/sequence groups for train, val, test; do not split adjacent frames")
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; use a new output directory to preserve the existing split: {output}")
    splits = ("train", "val", "test")
    assignments = {}
    counts = dict.fromkeys(splits, 0)
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda group: len(groups[group]), reverse=True)
    targets = dict(zip(splits, [len(rows) * ratio for ratio in ratios]))
    for index, group in enumerate(ordered):
        empty = [split for split in splits if counts[split] == 0]
        choices = empty if len(ordered) - index == len(empty) else list(splits)
        split = max(choices, key=lambda name: (targets[name] - counts[name]) / targets[name])
        assignments[group] = split
        counts[split] += len(groups[group])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for split in splits:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        prepared_rows = []
        for row in rows:
            split_key = row["session_id"].strip() if has_sessions else row["group_id"]
            split = assignments[split_key]
            source = Path(row["image"])
            identifier = hashlib.sha256((row["group_id"] + "\0" + row["image"]).encode()).hexdigest()[:16]
            stem = f"card_{identifier}"
            image_rel = Path("images") / split / (stem + source.suffix.lower())
            label_rel = Path("labels") / split / (stem + ".txt")
            shutil.copy2(source, staging / image_rel)
            shutil.copy2(row["label"], staging / label_rel)
            prepared_rows.append({"image": image_rel.as_posix(), "label": label_rel.as_posix(), "group_id": row["group_id"], "session_id": (row.get("session_id") or "").strip(), "scene": row["scene"], "reviewed": "true", "split": split, "source_image": row["image"], "source_label": row["label"]})
        with (staging / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prepared_rows[0]))
            writer.writeheader()
            writer.writerows(prepared_rows)
        config = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "nc": 54, "names": CLASS_NAMES}
        (staging / "data.yaml").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from validate_dataset import validate
        report = validate(staging / "data.yaml", dataset_root=staging)
        if report["errors"]:
            raise ValueError("Dataset validation failed: " + "; ".join(report["errors"][:12]))
        report["data"] = str(output / "data.yaml")
        (staging / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "split_info.json").write_text(json.dumps({"seed": seed, "requested_ratios": dict(zip(splits, ratios)), "source_manifest": str(manifest.resolve()), "split_key": "session_id" if has_sessions else "group_id", "counts": counts, "group_assignments": assignments}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"data": str(output / "data.yaml"), "images": counts, "groups": len(groups), "warnings": report["warnings"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.manifest, args.output, args.ratios, args.seed), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Dataset preparation stopped: {error}\n")


if __name__ == "__main__":
    main()
