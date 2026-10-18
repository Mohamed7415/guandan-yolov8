"""Preserve frozen originals and require explicit negative labels."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from card_common import CLASS_NAMES
from revise_dataset import apply_revision, annotation_digest
from validate_dataset import validate


def fixture(directory):
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        image = directory / f"images/{split}/card{index}.png"
        label = directory / f"labels/{split}/card{index}.txt"
        image.parent.mkdir(parents=True); label.parent.mkdir(parents=True)
        Image.new("RGB", (16, 16), (index * 60, 80, 120)).save(image)
        label.write_text("0 0.25000000 0.25000000 0.37500000 0.37500000\n")
        rows.append(dict(image=f"images/{split}/{image.name}", label=f"labels/{split}/{label.name}",
                         group_id=split, session_id=split, scene="sorted", reviewed="true", split=split,
                         source_image=str(image), source_label=str(label)))
    manifest(directory, rows)
    (directory / "data.yaml").write_text(json.dumps({"path": str(directory), "names": CLASS_NAMES,
               "nc": 54, **{s: "images/" + s for s in ("train", "val", "test")}}))
    return rows


def manifest(directory, rows):
    with (directory / "split_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


class DatasetChecks(unittest.TestCase):
    def test_portable_path_rewrite_preserves_semantics_but_label_change_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "annotation.json"
            doc = {"imageWidth": 100, "imageHeight": 100, "imagePath": "/old/private/path.png",
                   "shapes": [{"label": "8♠", "shape_type": "rectangle", "points": [[1, 1], [20, 20]], "flags": {}}]}
            path.write_text(json.dumps(doc)); original = annotation_digest(path)
            doc["imagePath"] = "../images/test/portable.png"
            path.write_text(json.dumps(doc)); self.assertEqual(annotation_digest(path), original)
            doc["shapes"][0]["label"] = "8♣"
            path.write_text(json.dumps(doc)); self.assertNotEqual(annotation_digest(path), original)

    def test_accidentally_empty_positive_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); rows = fixture(directory)
            # Keep a positive train frame as well so the expected failure is
            # specifically an unreviewed background, not background-only data.
            original = dict(rows[0]); original["image"] = "images/train/extra.png"; original["label"] = "labels/train/extra.txt"
            Image.new("RGB", (16, 16), (200, 200, 200)).save(directory / original["image"])
            (directory / original["label"]).write_text("")
            rows.append(original); manifest(directory, rows)
            report = validate(directory / "data.yaml")
            self.assertFalse(report["valid"])
            self.assertTrue(any("explicit negative" in error for error in report["errors"]))

    def test_negative_with_boxes_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); rows = fixture(directory)
            rows[0]["scene"] = "negative"; manifest(directory, rows)
            self.assertFalse(validate(directory / "data.yaml")["valid"])

    def test_reviewed_empty_negative_is_accepted_with_other_positives(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); rows = fixture(directory)
            row = dict(rows[0]); row.update(image="images/train/negative.png", label="labels/train/negative.txt", scene="negative")
            Image.new("RGB", (16, 16), (200, 200, 200)).save(directory / row["image"])
            (directory / row["label"]).write_text("")
            rows.append(row); manifest(directory, rows)
            report = validate(directory / "data.yaml")
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["background_images"]["train"], 1)

    def test_revision_preserves_original_bytes_and_refuses_hash_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); source = directory / "source"; source.mkdir()
            rows = fixture(source); annotations = directory / "annotations"; annotations.mkdir()
            for row in rows:
                doc = {"imageWidth": 16, "imageHeight": 16, "flags": {"sorted": True},
                       "imagePath": str(source / row["image"]), "shapes": [
                           {"label": "3♠", "shape_type": "rectangle", "points": [[1, 1], [7, 7]], "flags": {}}]}
                (annotations / (Path(row["image"]).stem + ".json")).write_text(json.dumps(doc))
            row = rows[0]; image, label = source / row["image"], source / row["label"]
            annotation = annotations / (image.stem + ".json")
            old = {p: p.read_bytes() for p in (image, label, annotation)}
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            change = {"source_name": image.name, "image_sha256": sha(image), "label_sha256": sha(label),
                      "annotation_sha256": sha(annotation), "evidence": {"original": "visually reviewed"},
                      "add_shapes": [{"label": "3♥", "shape_type": "rectangle", "points": [[10, 10], [14, 14]], "flags": {}}]}
            plan = directory / "plan.json"; plan.write_text(json.dumps({"reviewed": True, "reason": "missing visible card", "changes": [change]}))
            with patch("revise_dataset.validate", return_value={"errors": [], "valid": True, "dataset_fingerprint": "test"}):
                result = apply_revision(source, annotations, plan, directory / "revised")
            self.assertEqual(result["added_boxes"], 1)
            for path, content in old.items(): self.assertEqual(path.read_bytes(), content)
            self.assertEqual(len((directory / "revised" / row["label"]).read_text().splitlines()), 2)
            label.write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                apply_revision(source, annotations, plan, directory / "rejected")
            self.assertFalse((directory / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
