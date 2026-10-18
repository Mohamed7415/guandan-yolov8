"""Check interruption safety without loading torch or running real training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import training_recovery as recovery
from card_common import CLASS_NAMES


class FakeTrainer:
    def __init__(self, folder):
        self.wdir = folder
        self.last, self.best = folder / "last.pt", folder / "best.pt"

    def save_model(self):
        self.last.write_bytes(b"new epoch with optimizer")
        self.best.write_bytes(b"new best")


class TrainingRecoveryTests(unittest.TestCase):
    def test_failed_publish_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.pt", Path(folder) / "last.pt"
            source.write_bytes(b"new"); target.write_bytes(b"last valid epoch")
            with patch.object(recovery.os, "replace", side_effect=OSError("power interruption")):
                with self.assertRaises(OSError):
                    recovery.publish_file(source, target)
            self.assertEqual(target.read_bytes(), b"last valid epoch")

    def test_durable_save_preserves_optimizer_copy_and_previous_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "weights"; directory.mkdir()
            (directory / "resume.pt").write_bytes(b"previous valid epoch")
            trainer = recovery.durable_trainer(FakeTrainer)(directory)
            original_paths = trainer.wdir, trainer.last, trainer.best
            with patch.object(recovery, "inspect_checkpoint", return_value={"epoch": 1}):
                trainer.save_model()
            self.assertEqual((directory / "resume.previous.pt").read_bytes(), b"previous valid epoch")
            self.assertEqual((directory / "resume.pt").read_bytes(), b"new epoch with optimizer")
            self.assertEqual((trainer.wdir, trainer.last, trainer.best), original_paths)
            # Upstream final_eval strips last/best; the recovery copy survives.
            trainer.last.write_bytes(b"stripped inference weights")
            self.assertEqual((directory / "resume.pt").read_bytes(), b"new epoch with optimizer")

    def test_failed_serialization_restores_paths_and_preserves_checkpoint(self):
        class FailingTrainer(FakeTrainer):
            def save_model(self):
                self.last.write_bytes(b"partial")
                raise OSError("disk interruption")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "resume.pt").write_bytes(b"valid")
            trainer = recovery.durable_trainer(FailingTrainer)(directory)
            original_paths = trainer.wdir, trainer.last, trainer.best
            with self.assertRaises(OSError):
                trainer.save_model()
            self.assertEqual((trainer.wdir, trainer.last, trainer.best), original_paths)
            self.assertEqual((directory / "resume.pt").read_bytes(), b"valid")

    def test_data_drift_rejected_before_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data.yaml"; data.write_text("{}")
            contract = {"stage": "full", "class_names": CLASS_NAMES, "dataset_fingerprint": "old"}
            with patch.object(recovery, "inspect_checkpoint") as inspect:
                with self.assertRaisesRegex(ValueError, "Dataset changed"):
                    recovery.verify_resume_contract(Path("resume.pt"), contract, {"dataset_fingerprint": "new"}, data, "full")
                inspect.assert_not_called()

    def test_corrupt_latest_falls_back_to_previous_epoch(self):
        with patch.object(recovery, "inspect_checkpoint", side_effect=[ValueError("truncated"), {"epoch": 4}]):
            path, info = recovery.choose_resume_checkpoint(Path("run/weights/resume.pt"))
        self.assertEqual(path.name, "resume.previous.pt")
        self.assertEqual(info["epoch"], 4)


if __name__ == "__main__":
    unittest.main()
