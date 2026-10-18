"""Atomic epoch checkpoints and a verified optimizer-preserving resume path."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from card_common import CLASS_NAMES
from atomic_io import atomic_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    """Load only locally trusted project checkpoints; reject stripped weights."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 0:
        raise ValueError("Checkpoint is stripped/completed or has no recoverable epoch")
    if not checkpoint.get("optimizer") or not checkpoint.get("train_args"):
        raise ValueError("Checkpoint lacks optimizer/training state; --weights is a new run, not resume")
    network = checkpoint.get("ema") or checkpoint.get("model")
    names = getattr(network, "names", None)
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != CLASS_NAMES:
        raise ValueError("Resume checkpoint does not contain the 54 required classes")
    args = checkpoint["train_args"]
    return {"epoch": checkpoint["epoch"], "next_epoch": checkpoint["epoch"] + 1,
            "planned_epochs": int(args["epochs"]), "data": args["data"],
            "imgsz": args["imgsz"], "batch": args["batch"], "optimizer_present": True,
            "sha256": digest(path)}


def publish_file(source: Path, target: Path) -> None:
    """Fsync a complete replacement before atomically publishing it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + target.name + ".", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as original:
            shutil.copyfileobj(original, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
        if os.name == "posix":
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def choose_resume_checkpoint(requested: Path) -> tuple[Path, dict]:
    """Fall back to the previous epoch if the latest recovery file is damaged."""
    candidates = [requested]
    if requested.name == "resume.pt":
        candidates.append(requested.with_name("resume.previous.pt"))
    failures = []
    for path in candidates:
        try:
            return path, inspect_checkpoint(path)
        except Exception as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
    raise ValueError("No valid recoverable checkpoint: " + "; ".join(failures))


def durable_trainer(base_class=None):
    """Wrap the installed trainer, without modifying the installed package."""
    if base_class is None:
        from ultralytics.models.yolo.detect import DetectionTrainer
        base_class = DetectionTrainer

    class DurableDetectionTrainer(base_class):
        def save_model(self):
            original = self.wdir, self.last, self.best
            directory = Path(self.wdir)
            directory.mkdir(parents=True, exist_ok=True)
            # Upstream serializes all state (including optimizer) but writes
            # directly to final filenames. Let it write into private staging.
            with tempfile.TemporaryDirectory(prefix=".epoch-", dir=directory) as temporary:
                staging = Path(temporary)
                try:
                    self.wdir = staging
                    self.last, self.best = staging / "last.pt", staging / "best.pt"
                    super().save_model()
                finally:
                    self.wdir, self.last, self.best = original
                info = inspect_checkpoint(staging / "last.pt")
                recoverable = directory / "resume.pt"
                if recoverable.exists():
                    # Preserve the previous good epoch. If current was damaged
                    # externally, do not overwrite an older valid backup.
                    try:
                        inspect_checkpoint(recoverable)
                    except Exception:
                        pass
                    else:
                        publish_file(recoverable, directory / "resume.previous.pt")
                publish_file(staging / "last.pt", recoverable)
                for path in staging.glob("*.pt"):
                    publish_file(path, directory / path.name)
                atomic_json(directory.parent / "training_progress.json", {
                    **info, "checkpoint": str(recoverable.resolve()),
                    "previous_checkpoint": str((directory / "resume.previous.pt").resolve()),
                    "status": "epoch_saved", "run_directory": str(directory.parent.resolve()),
                })

    return DurableDetectionTrainer


def verify_resume_contract(checkpoint: Path, contract: dict, report: dict, data: Path, stage: str) -> dict:
    """Reject data drift before loading a model or restoring an optimizer."""
    import yaml

    if contract.get("stage") != stage or contract.get("class_names") != CLASS_NAMES:
        raise ValueError("Resume stage/classes differ from the original training contract")
    if report["dataset_fingerprint"] != contract["dataset_fingerprint"] or digest(data) != contract["data_yaml_sha256"]:
        raise ValueError("Dataset changed since the run started; create a new run instead of resume")
    config = yaml.safe_load(data.read_text())
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = data.parent / root
    if digest(root / "split_manifest.csv") != contract["split_manifest_sha256"]:
        raise ValueError("Dataset split changed since the run started")
    info = inspect_checkpoint(checkpoint)
    if Path(info["data"]).resolve() != data.resolve():
        raise ValueError("Checkpoint data path differs from the verified contract")
    if info["next_epoch"] >= info["planned_epochs"]:
        raise ValueError("All planned epochs are saved; recover final evaluation/export, do not restart training")
    return info
