#!/usr/bin/env python3
"""Apply reviewed additive label corrections to a new frozen dataset version.

The plan must pin original image, annotation and YOLO-label hashes. Existing
datasets and annotations are never edited. Image hardlinks are read-only inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from card_common import CLASS_NAMES
from convert_annotations import convert_one
from atomic_io import atomic_json
from validate_dataset import validate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotation_digest(path: Path) -> str:
    """Pin annotation meaning across portable imagePath/provenance rewrites."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    value = {"imageWidth": doc["imageWidth"], "imageHeight": doc["imageHeight"],
             "shapes": [{key: s.get(key) for key in ("label", "shape_type", "points", "flags")}
                        for s in doc["shapes"]]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def apply_revision(source: Path, annotations: Path, plan_path: Path, output: Path) -> dict:
    source, annotations, output = source.resolve(), annotations.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Revision output already exists; preserve frozen versions")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("reviewed") is not True or not plan.get("reason"):
        raise ValueError("A reviewed plan with an evidence-based reason is required")
    changes = plan.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("Revision must contain explicit reviewed changes")
    by_name = {}
    for change in changes:
        name = change["source_name"]
        if Path(name).name != name or name in by_name or not change.get("evidence"):
            raise ValueError("Revision needs unique source filenames and review evidence")
        by_name[name] = change
    with (source / "split_manifest.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    found = set()
    for row in rows:
        name = Path(row["source_image"]).name
        if name not in by_name:
            continue
        change = by_name[name]
        annotation = annotations / (Path(name).stem + ".json")
        expected = {"image_sha256": source / row["image"], "label_sha256": source / row["label"]}
        if "annotation_sha256" in change:
            expected["annotation_sha256"] = annotation
        elif annotation_digest(annotation) != change.get("annotation_semantic_sha256"):
            raise ValueError(f"Original annotation meaning changed: {name}")
        for key, path in expected.items():
            if sha(path) != change[key]:
                raise ValueError(f"Original evidence changed: {name}: {key}")
        found.add(name)
    if found != set(by_name):
        raise ValueError("Revision contains images absent from the frozen split")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    audit = []
    try:
        prepared = []
        for row in rows:
            row = dict(row)
            name = Path(row["source_image"]).name
            old_image, old_label = source / row["image"], source / row["label"]
            new_image, new_label = staging / row["image"], staging / row["label"]
            new_image.parent.mkdir(parents=True, exist_ok=True)
            new_label.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(old_image, new_image)
            except OSError:
                shutil.copy2(old_image, new_image)
            old_json = annotations / (Path(name).stem + ".json")
            doc = json.loads(old_json.read_text(encoding="utf-8"))
            new_json = staging / "annotations" / old_json.name
            new_json.parent.mkdir(exist_ok=True)
            doc["imagePath"] = os.path.relpath(new_image, new_json.parent)
            doc["imageData"] = None
            if name in by_name:
                change = by_name[name]
                before = len(doc["shapes"])
                doc["shapes"].extend(change["add_shapes"])
                doc["revision_provenance"] = {"reason": plan["reason"], "evidence": change["evidence"],
                                              "after_model_evaluation": True,
                                              "independent_human_review": False}
                atomic_json(new_json, doc)
                conversion_row = {**row, "image": str(new_image), "label": str(new_json)}
                _, _, _, lines, _ = convert_one(conversion_row, staging, {c: i for i, c in enumerate(CLASS_NAMES)})
                old_lines = old_label.read_text(encoding="utf-8").splitlines()
                # This command supports additions only: prior reviewed boxes
                # must remain exactly the same after deterministic conversion.
                if lines[:len(old_lines)] != old_lines:
                    raise ValueError(f"Existing JSON and frozen YOLO labels disagree: {name}")
                new_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
                audit.append({"source_name": name, "before_boxes": before, "after_boxes": len(lines),
                              "old_label_sha256": sha(old_label), "new_label_sha256": sha(new_label),
                              "added_shapes": change["add_shapes"], "evidence": change["evidence"]})
            else:
                atomic_json(new_json, doc)
                shutil.copy2(old_label, new_label)
            row["source_label"] = str(output / row["label"])
            row["source_annotation"] = "annotations/" + old_json.name
            prepared.append(row)
        with (staging / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prepared[0]))
            writer.writeheader()
            writer.writerows(prepared)
        import yaml
        config = yaml.safe_load((source / "data.yaml").read_text())
        config["path"] = str(output)
        atomic_json(staging / "data.yaml", config)
        check = validate(staging / "data.yaml", dataset_root=staging, full=True)
        if check["errors"]:
            raise ValueError("Revised data validation failed: " + "; ".join(check["errors"][:10]))
        atomic_json(staging / "revision.json", {"plan_sha256": sha(plan_path), "reason": plan["reason"],
                    "source_dataset": str(source), "changes": audit,
                    "test_is_retrospective_regression": True, "source_files_modified": False})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # The fingerprint intentionally contains resolved paths; calculate it at
    # the committed location, rather than publishing the staging fingerprint.
    report = validate(output / "data.yaml", full=True)
    atomic_json(output / "dataset_report.json", report)
    return {"output": str(output), "changed_images": len(audit),
            "added_boxes": sum(r["after_boxes"] - r["before_boxes"] for r in audit),
            "valid": report["valid"], "dataset_fingerprint": report["dataset_fingerprint"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(apply_revision(args.source, args.annotations, args.plan, args.output), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Revision stopped: {error}\n")


if __name__ == "__main__":
    main()
