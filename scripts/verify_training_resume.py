#!/usr/bin/env python3
"""Integration check: abruptly exit after epoch 1, then resume real YOLO training.

Uses three genuine, existing images and an isolated 64px CPU smoke run. This is
recovery verification, not a production candidate or an accuracy benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from card_common import CLASS_NAMES
from atomic_io import atomic_json
from training_recovery import digest, inspect_checkpoint, choose_resume_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHILD = '''
import os, runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]).parent))
import torch
torch.set_num_threads(2)
import ultralytics.utils.checks as checks
checks.check_pip_update_available = lambda: None
import training_recovery
from atomic_io import atomic_json
from verify_training_resume import optimizer_signature
original = training_recovery.durable_trainer
interrupt = sys.argv[1] == "interrupt"
def verification_trainer(*args, **kwargs):
    base = original(*args, **kwargs)
    class VerifiedTrainer(base):
        def _setup_train(self):
            super()._setup_train()
            if self.resume:
                atomic_json(Path(self.save_dir) / "restored_optimizer.json", optimizer_signature(self.optimizer.state_dict()))
        def save_model(self):
            super().save_model()
            if interrupt and self.epoch == 0:
                os._exit(75)
    return VerifiedTrainer
training_recovery.durable_trainer = verification_trainer
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def optimizer_step(path: Path) -> float:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return max(float(s["step"]) for s in ckpt["optimizer"]["state"].values() if "step" in s)


def optimizer_signature(state: dict) -> dict:
    import torch
    entries = []
    for index, entry in sorted(state["state"].items()):
        values = {}
        for key, value in sorted(entry.items()):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().float().contiguous()
                values[key] = {"shape": list(value.shape), "sha256": __import__("hashlib").sha256(value.numpy().tobytes()).hexdigest()}
            else:
                values[key] = value
        entries.append({"index": index, "values": values})
    # JSON represents tuples (e.g. Adam betas) as lists. Normalize both sides
    # before comparing the on-disk snapshot to the saved optimizer.
    return json.loads(json.dumps({"state": entries, "param_groups": state["param_groups"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/card_model.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output, weights = args.source.resolve(), args.output.resolve(), args.weights.resolve()
    if output.exists():
        parser.error("Use a fresh output directory; previous evidence is preserved")
    output.mkdir(parents=True)
    before = digest(weights)
    with (source / "split_manifest.csv").open() as stream:
        source_rows = list(csv.DictReader(stream))
    rows = []
    data = output / "dataset"
    for split in ("train", "val", "test"):
        row = dict(next(r for r in source_rows if r["split"] == split
                        and (source / r["label"]).read_text().strip()))
        for key in ("image", "label"):
            target = data / row[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row[key], target)
        rows.append(row)
    with (data / "split_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(data / "data.yaml", {"path": str(data), "nc": 54, "names": CLASS_NAMES,
                **{split: f"images/{split}" for split in ("train", "val", "test")}})
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "False"}
    train = ROOT / "scripts/train.py"
    command = [sys.executable, "-c", CHILD, "interrupt", str(train), "--data", str(data / "data.yaml"),
               "--weights", str(weights), "--epochs", "3", "--batch", "1", "--imgsz", "64", "--device", "cpu",
               "--project", str(output / "runs"), "--name", "smoke"]
    with (output / "interrupted.log").open("w") as log:
        first = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if first.returncode != 75:
        raise RuntimeError(f"Expected abrupt exit 75, got {first.returncode}; inspect interrupted.log")
    run = output / "runs/smoke"
    checkpoint = run / "weights/resume.pt"
    first_info, first_step = inspect_checkpoint(checkpoint), optimizer_step(checkpoint)
    import torch
    expected_optimizer = optimizer_signature(torch.load(checkpoint, map_location="cpu", weights_only=False)["optimizer"])
    atomic_json(output / "expected_optimizer.json", expected_optimizer)
    if first_info["epoch"] != 0 or (run / "training_record.json").exists():
        raise RuntimeError("Interrupted process did not preserve exactly the first unfinished epoch")
    command = [sys.executable, "-c", CHILD, "resume", str(train), "--resume", str(checkpoint), "--device", "cpu"]
    with (output / "resumed.log").open("w") as log:
        second = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if second.returncode:
        raise RuntimeError(f"Resume failed with {second.returncode}; inspect resumed.log")
    final_info, final_step = inspect_checkpoint(checkpoint), optimizer_step(checkpoint)
    record = json.loads((run / "training_record.json").read_text())
    restored_optimizer = json.loads((run / "restored_optimizer.json").read_text())
    if restored_optimizer != expected_optimizer:
        raise RuntimeError("Restored optimizer tensors/groups differ from the saved state")
    with (run / "results.csv").open() as stream:
        epochs = [int(float(r["epoch"])) for r in csv.DictReader(stream)]
    if epochs != [1, 2, 3] or final_info["epoch"] != 2 or final_step <= first_step:
        raise RuntimeError("Resume repeated/lost an epoch or reset optimizer state")
    # Damage a separate copy; never damage the actual successful smoke run.
    damaged = output / "damaged_checkpoint_copy"
    damaged.mkdir()
    (damaged / "resume.pt").write_bytes(b"truncated checkpoint")
    shutil.copy2(run / "weights/resume.previous.pt", damaged / "resume.previous.pt")
    fallback, fallback_info = choose_resume_checkpoint(damaged / "resume.pt")
    report = {"passed": record["completed"] and digest(weights) == before,
              "verification_only": True, "production_accuracy_claim": False,
              "interruption": "os._exit(75) immediately after atomic epoch save; no graceful finalization",
              "first_epoch": first_info["epoch"], "final_epoch": final_info["epoch"],
              "logged_epochs": epochs, "optimizer_step_before_resume": first_step,
              "optimizer_step_after_resume": final_step,
              "optimizer_preserved": restored_optimizer == expected_optimizer,
              "optimizer_tensors_verified_before_next_step": True,
              "corrupt_latest_fallback": fallback.name, "fallback_epoch": fallback_info["epoch"],
              "baseline_weights_unchanged": digest(weights) == before,
              "actual_power_loss_tested": False}
    atomic_json(output / "result.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
