#!/usr/bin/env python3
"""Train YOLOv8n only after reviewed, independent data pass validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from card_common import CLASS_NAMES, require_yolov8n
from atomic_io import atomic_json
from validate_dataset import validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Default: data/prepared/data.yaml, or the verified resume contract")
    parser.add_argument("--stage", choices=("pilot", "full"), help="Default: pilot, or the original resume stage")
    default_weights = "models/yolov8n.pt" if Path("models/yolov8n.pt").is_file() else "yolov8n.pt"
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--weights", default=default_weights, help="YOLOv8n initialization; starts a new optimizer/run")
    initialization.add_argument("--resume", type=Path, help="Verified weights/resume.pt; preserves optimizer and epoch")
    parser.add_argument("--epochs", type=int, help="Default: pilot 60, full 150")
    parser.add_argument("--batch", type=int, help="Default: 8, or the original batch on resume")
    parser.add_argument("--imgsz", type=int, help="Default: 640; a new input resolution is a separate experiment")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or CUDA device index")
    parser.add_argument("--workers", type=int, default=0, help="0 is reliable for macOS and avoids copied worker state")
    parser.add_argument("--cache-ram", action="store_true", help="Cache resized training images in RAM")
    parser.add_argument("--project", type=Path, default=Path("runs/cards"))
    parser.add_argument("--name", help="Default: pilot or full; Ultralytics increments an existing run name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.batch is not None and args.batch < 1) or args.workers < 0 or (args.epochs is not None and args.epochs < 1):
        parser.error("batch/epochs must be positive and workers must be nonnegative")
    if args.imgsz is not None and (args.imgsz < 32 or args.imgsz % 32):
        parser.error("imgsz must be a positive multiple of 32")
    contract = None
    try:
        if args.resume:
            args.resume = args.resume.resolve()
            from training_recovery import choose_resume_checkpoint
            args.resume, _ = choose_resume_checkpoint(args.resume)
            contract = json.loads((args.resume.parent.parent / "recovery_contract.json").read_text())
        args.data = (args.data or Path(contract["data_yaml"] if contract else "data/prepared/data.yaml")).resolve()
        args.stage = args.stage or (contract["stage"] if contract else "pilot")
        report = validate(args.data, full=args.stage == "full")
        from training_recovery import verify_resume_contract
        resume_info = verify_resume_contract(args.resume, contract, report, args.data, args.stage) if args.resume else None
        if resume_info:
            if args.epochs is not None and args.epochs != resume_info["planned_epochs"]:
                raise ValueError("Resume must retain the original epoch target; start a new run to change it")
            if args.imgsz is not None and args.imgsz != resume_info["imgsz"]:
                raise ValueError("Resume must retain the original input resolution")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as error:
        parser.exit(2, f"Training stopped before loading any model: {error}\n")
    if report["errors"]:
        parser.exit(2, "Training stopped before loading any model:\n" + "\n".join(report["errors"]) + "\n")
    for warning in report["warnings"]:
        print(f"Data warning: {warning}")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    import torch
    from ultralytics import YOLO
    import ultralytics

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps":
        from mps_compat import install_safe_assignment
        install_safe_assignment()
        print("MPS compatibility: task-aligned assignment on CPU; network and gradients on MPS")
    data = args.data.resolve()
    epochs = resume_info["planned_epochs"] if resume_info else (args.epochs or (60 if args.stage == "pilot" else 150))
    batch = args.batch or (resume_info["batch"] if resume_info else 8)
    imgsz = resume_info["imgsz"] if resume_info else (args.imgsz or 640)
    model = YOLO(str(args.resume) if args.resume else args.weights, task="detect")
    try:
        require_yolov8n(model)
    except ValueError as error:
        parser.exit(2, f"Training stopped: {error}\n")
    import yaml
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    new_contract = {
        "stage": args.stage, "architecture": "YOLOv8n", "class_names": CLASS_NAMES,
        "data_yaml": str(data), "data_yaml_sha256": sha256(data),
        "split_manifest_sha256": sha256(root / "split_manifest.csv"),
        "dataset_fingerprint": report["dataset_fingerprint"],
        "planned_epochs": epochs, "imgsz": imgsz, "seed": args.seed,
        "initial_weights": str(Path(args.weights).resolve()),
    }

    def before_training(trainer):
        path = Path(trainer.save_dir) / "recovery_contract.json"
        if args.resume:
            if path.resolve() != (args.resume.parent.parent / "recovery_contract.json").resolve():
                raise ValueError("Resume must continue the original run directory")
        else:
            if path.exists():
                raise ValueError("A recovery contract already exists; use a new run name")
            atomic_json(path, new_contract)

    model.add_callback("on_pretrain_routine_start", before_training)
    from training_recovery import durable_trainer
    trainer_type = durable_trainer()
    # Card rank and suit orientation/color carry semantic information. Keep flips,
    # color shifts and aggressive geometric synthesis out of this screenshot task.
    if args.resume:
        model.train(trainer=trainer_type, resume=True, device=device, batch=batch)
    else:
        model.train(
            trainer=trainer_type, data=str(data), imgsz=imgsz, epochs=epochs, batch=batch, device=device,
            workers=args.workers, project=str(args.project.resolve()), name=args.name or args.stage,
            seed=args.seed, deterministic=True, patience=25, optimizer="AdamW", lr0=0.001,
            cos_lr=True, pretrained=True, val=True, plots=True, cache="ram" if args.cache_ram else False, save=True,
            amp=False, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
            degrees=0.0, translate=0.03, scale=0.1, shear=0.0, perspective=0.0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        )
    run = Path(model.trainer.save_dir).resolve()
    best = Path(model.trainer.best).resolve()
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt; no completed training record will be written")
    record = {
        "completed": True, "stage": args.stage, "architecture": "YOLOv8n", "class_names": CLASS_NAMES,
        "data_yaml": str(data), "data_yaml_sha256": sha256(data),
        "split_manifest_sha256": sha256(root / "split_manifest.csv"),
        "dataset_fingerprint": report["dataset_fingerprint"],
        "checkpoint": str(best), "checkpoint_sha256": sha256(best),
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        "device": device, "requested_epochs": epochs, "seed": int(model.trainer.args.seed),
        "initial_weights": contract["initial_weights"] if contract else new_contract["initial_weights"],
        "resumed_from": str(args.resume) if args.resume else None,
        "input_size": imgsz,
        "mps_assignment_policy": "CPU task-aligned assignment" if device == "mps" else "native",
        "dataset_report": report,
        "results_csv": str(run / "results.csv"),
        "note": "Pilot models are exploratory and may lack class coverage; metrics are in results.csv, not fabricated here.",
    }
    atomic_json(run / "training_record.json", record)
    print(f"Trained checkpoint: {best}")
    print(f"Training record: {run / 'training_record.json'}")
    print("Export after reviewing validation images: .venv/bin/python scripts/export_onnx.py --checkpoint " + str(best))


if __name__ == "__main__":
    main()
