#!/usr/bin/env python3
"""Validate labels, class order, group isolation, image integrity and coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from card_common import CLASS_NAMES, IMAGE_EXTENSIONS, manifest_rows, read_labels


def validate(data: Path, *, dataset_root: Path | None = None, full: bool = False) -> dict:
    import yaml
    from PIL import Image

    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = dataset_root or Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    errors, warnings = [], []
    names = config.get("names")
    if isinstance(names, dict):
        names = [names.get(index, names.get(str(index))) for index in range(54)]
    if names != CLASS_NAMES or config.get("nc", 54) != 54:
        errors.append("data.yaml must use all 54 names in MODEL_SPEC index order")
    report = {"data": str(data.resolve()), "images": {}, "background_images": {}, "groups": {}, "instances": {}, "class_image_counts": {}, "errors": errors, "warnings": warnings, "valid": False}
    manifest = root / "split_manifest.csv"
    if not manifest.is_file():
        errors.append("split_manifest.csv is required to verify game/sequence isolation")
        return report
    try:
        rows = manifest_rows(manifest)
    except (ValueError, OSError) as error:
        errors.append(str(error))
        return report
    group_splits, image_splits, session_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    instances = {split: Counter() for split in ("train", "val", "test")}
    image_classes = {split: Counter() for split in instances}
    hashes = {}
    fingerprint_entries = []
    tiny_boxes = 0
    for split in instances:
        declared = config.get(split)
        expected = f"images/{split}"
        if declared != expected:
            errors.append(f"{split}: this pipeline expects {expected}, got {declared!r}")
        split_rows = [row for row in rows if row.get("split") == split]
        report["images"][split] = len(split_rows)
        report["background_images"][split] = 0
        report["groups"][split] = len({row["group_id"] for row in split_rows})
        if not split_rows:
            errors.append(f"{split}: no reviewed images")
        actual = {str(path.resolve()) for path in (root / "images" / split).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS}
        listed = {row["image"] for row in split_rows}
        if actual != listed:
            errors.append(f"{split}: image directory and split manifest differ ({len(actual)} files, {len(listed)} listed)")
        expected_labels = {str((root / "labels" / split / (Path(path).stem + ".txt")).resolve()) for path in listed}
        if expected_labels != {row["label"] for row in split_rows}:
            errors.append(f"{split}: labels must match images by stem under labels/{split}")
        for row in split_rows:
            group_splits[row["group_id"]].add(split)
            if row.get("session_id"):
                session_splits[row["session_id"]].add(split)
            image_splits[row.get("source_image", row["image"])].add(split)
            try:
                labels = read_labels(Path(row["label"]))
                with Image.open(row["image"]) as original:
                    image = original.convert("RGB")
                    width, height = image.size
                    digest = hashlib.sha256(f"{width},{height}:".encode() + image.tobytes()).hexdigest()
                fingerprint_entries.append([split, row["image"], row["group_id"], row.get("session_id", ""), digest, hashlib.sha256(Path(row["label"]).read_bytes()).hexdigest()])
                if digest in hashes:
                    other_split, other_path = hashes[digest]
                    if split != other_split:
                        errors.append(f"Duplicate image pixels across {other_split}/{split}: {other_path}, {row['image']}")
                    else:
                        warnings.append(f"Repeated image pixels within {split}: {row['image']}")
                else:
                    hashes[digest] = (split, row["image"])
                if not labels:
                    report["background_images"][split] += 1
                for cls, cx, cy, box_width, box_height in labels:
                    instances[split][cls] += 1
                    scale = 640 / max(width, height)
                    if min(box_width * width * scale, box_height * height * scale) < 4:
                        tiny_boxes += 1
                image_classes[split].update({label[0] for label in labels})
            except (OSError, ValueError) as error:
                errors.append(str(error))
        report["instances"][split] = {name: instances[split][index] for index, name in enumerate(CLASS_NAMES)}
        report["class_image_counts"][split] = {name: image_classes[split][index] for index, name in enumerate(CLASS_NAMES)}
    unknown = sorted({row.get("split", "") for row in rows} - set(instances))
    if unknown:
        errors.append(f"Unknown splits in manifest: {unknown}")
    for group, splits in group_splits.items():
        if len(splits) != 1:
            errors.append(f"Game/sequence group leaks across splits: {group}")
    for session, splits in session_splits.items():
        if len(splits) != 1:
            errors.append(f"Game session leaks across splits: {session}")
    for source, splits in image_splits.items():
        if len(splits) != 1:
            errors.append(f"Source image leaks across splits: {source}")
    if not sum(instances["train"].values()):
        errors.append("Train split has no labeled cards; refusing a background-only training set")
    if not sum(instances["val"].values()):
        errors.append("Val split has no labeled cards; detection quality cannot be validated")
    if not sum(instances["test"].values()):
        errors.append("Test split has no labeled cards; keep reviewed positives in held-out data")
    for split in instances:
        missing = [name for index, name in enumerate(CLASS_NAMES) if not instances[split][index]]
        if missing:
            (errors if full else warnings).append(f"{split} missing classes ({len(missing)}): {', '.join(missing)}")
    if full:
        below = {name: sum(instances[split][index] for split in instances) for index, name in enumerate(CLASS_NAMES) if sum(instances[split][index] for split in instances) < 50}
        if below:
            errors.append(f"Full dataset needs at least 50 labeled instances per class; below target: {below}")
    if tiny_boxes:
        warnings.append(f"{tiny_boxes} boxes have a side under 4 pixels after 640 letterbox; inspect crop/visibility and resolution")
    for split, count in report["images"].items():
        if count and report["background_images"][split] / count > 0.4:
            warnings.append(f"{split}: more than 40% are background images; review the scene balance")
    report["valid"] = not errors
    report["dataset_fingerprint"] = hashlib.sha256(json.dumps(sorted(fingerprint_entries), ensure_ascii=False).encode()).hexdigest()
    report["full_coverage_required"] = full
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/data.yaml"))
    parser.add_argument("--full", action="store_true", help="Require every class in every split and >=50 instances per class overall")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.data.resolve(), full=args.full)
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"Dataset validation stopped: {error}\n")
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
