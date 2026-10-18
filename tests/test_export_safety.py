"""Exercise file-placement guards without installing the training framework."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_onnx as exporter
from atomic_io import publish_new_file
from card_common import CLASS_NAMES


class FakeModel:
    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = Path(checkpoint)
        self.names = dict(enumerate(CLASS_NAMES))

    def export(self, **kwargs):
        path = self.checkpoint.with_suffix(".onnx")
        path.write_bytes(b"verified ONNX bytes")
        return str(path)


def setup(folder):
    root = Path(folder); run = root / "run"; weights = run / "weights"; weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"; checkpoint.write_bytes(b"audited weights")
    data = root / "dataset"; (data / "images/val").mkdir(parents=True)
    (data / "images/val/sample.png").write_bytes(b"sample placeholder for mocked parity only")
    (data / "split_manifest.csv").write_text("mocked data validator\n")
    config = data / "data.yaml"; config.write_text(json.dumps({"path": str(data)}))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (run / "training_record.json").write_text(json.dumps({"completed": True, "stage": "full", "class_names": CLASS_NAMES,
        "checkpoint_sha256": sha(checkpoint), "data_yaml": str(config), "data_yaml_sha256": sha(config),
        "dataset_fingerprint": "frozen", "split_manifest_sha256": sha(data / "split_manifest.csv")}))
    return root, checkpoint


def run_export(checkpoint, output, *, parity_error=None):
    with patch.object(sys, "argv", ["export_onnx", "--checkpoint", str(checkpoint), "--output", str(output)]), \
         patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeModel)}), \
         patch.object(exporter, "validate", return_value={"errors": [], "dataset_fingerprint": "frozen"}), \
         patch.object(exporter, "require_yolov8n"), patch.object(exporter, "scrub_metadata"), \
         patch.object(exporter, "verify_onnx", return_value={"checked": True}, side_effect=parity_error):
        exporter.main()


class ExportSafetyTests(unittest.TestCase):
    def test_new_output_does_not_overwrite_checkpoint_sibling(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = setup(folder); sibling = checkpoint.with_suffix(".onnx")
            sibling.write_bytes(b"previous export")
            output = root / "delivery/new.onnx"
            run_export(checkpoint, output)
            self.assertEqual(sibling.read_bytes(), b"previous export")
            self.assertEqual(output.read_bytes(), b"verified ONNX bytes")
            self.assertTrue(output.with_suffix(".verification.json").exists())

    def test_explicit_sibling_output_does_not_raise_same_file_error(self):
        with tempfile.TemporaryDirectory() as folder:
            _, checkpoint = setup(folder); output = checkpoint.with_suffix(".onnx")
            run_export(checkpoint, output)
            self.assertEqual(output.read_bytes(), b"verified ONNX bytes")

    def test_parity_failure_publishes_nothing_and_preserves_old_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = setup(folder); sibling = checkpoint.with_suffix(".onnx")
            sibling.write_bytes(b"previous export"); output = root / "delivery/new.onnx"
            with self.assertRaisesRegex(ValueError, "parity failed"):
                run_export(checkpoint, output, parity_error=ValueError("parity failed"))
            self.assertEqual(sibling.read_bytes(), b"previous export")
            self.assertFalse(output.exists())

    def test_existing_report_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = setup(folder); output = root / "new.onnx"
            report = output.with_suffix(".verification.json"); report.write_text("previous audit")
            with self.assertRaises(SystemExit): run_export(checkpoint, output)
            self.assertEqual(report.read_text(), "previous audit")
            self.assertFalse(output.exists())

    def test_publish_race_never_replaces_an_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "new", Path(folder) / "old"
            source.write_bytes(b"new"); target.write_bytes(b"keep")
            with self.assertRaises(FileExistsError): publish_new_file(source, target)
            self.assertEqual(target.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
