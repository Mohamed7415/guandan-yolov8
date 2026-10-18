#!/usr/bin/env python3
"""Export an audited 54-class YOLOv8n checkpoint and verify ONNX parity."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from card_common import CLASS_NAMES, require_yolov8n
from validate_dataset import validate
from infer_onnx import candidate_count
from atomic_io import publish_new_file


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_onnx(onnx_path: Path, model, sample_images: list[Path], input_size: int = 640) -> dict:
    """Test exact graph contract and compare raw outputs on held-out real images."""
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics.data.augment import LetterBox

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    opset = next(item.version for item in graph.opset_import if item.domain in ("", "ai.onnx"))
    if opset != 12:
        raise ValueError(f"Expected ONNX opset 12, got {opset}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    expected_output = [1, 58, candidate_count(input_size)]
    if len(inputs) != 1 or inputs[0].shape != [1, 3, input_size, input_size] or inputs[0].type != "tensor(float)":
        raise ValueError(f"Unexpected model input: {[(item.shape, item.type) for item in inputs]}")
    if len(outputs) != 1 or outputs[0].shape != expected_output:
        raise ValueError(f"Expected {expected_output}, got {[item.shape for item in outputs]}")
    metadata = session.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(metadata.get("names", "{}"))
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if names != CLASS_NAMES:
        raise ValueError("ONNX metadata class order differs from MODEL_SPEC")
    model.model.to("cpu").float().eval()
    letterbox = LetterBox(new_shape=(input_size, input_size), auto=False, stride=32)
    comparisons = []
    for image_path in sample_images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read held-out image: {image_path}")
        image = letterbox(image=image)
        tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        actual = session.run(None, {inputs[0].name: tensor})[0]
        with torch.inference_mode():
            reference = model.model(torch.from_numpy(tensor))
        if isinstance(reference, (tuple, list)):
            reference = reference[0]
        reference = reference.detach().cpu().numpy()
        if actual.shape != tuple(expected_output) or not np.isfinite(actual).all():
            raise ValueError("ONNX output shape/finite-value check failed")
        if not ((actual[:, 4:, :] >= -1e-6) & (actual[:, 4:, :] <= 1 + 1e-6)).all():
            raise ValueError("ONNX class scores are outside [0,1]")
        # Pixel coordinates and small class probabilities need separate tolerances.
        np.testing.assert_allclose(actual[:, :4], reference[:, :4], atol=0.05, rtol=1e-4)
        np.testing.assert_allclose(actual[:, 4:], reference[:, 4:], atol=1e-4, rtol=1e-3)
        comparisons.append({"image": str(image_path), "max_coordinate_error_px": float(np.max(np.abs(actual[:, :4] - reference[:, :4]))), "max_score_error": float(np.max(np.abs(actual[:, 4:] - reference[:, 4:])))})
    if not comparisons:
        raise ValueError("At least one held-out real image is needed for export verification")
    return {"opset": opset, "input_shape": inputs[0].shape, "output_shape": outputs[0].shape, "providers": session.get_providers(), "raw_parity": comparisons, "coordinate_format": f"cx cy w h, pixels in the {input_size}x{input_size} letterboxed input; external confidence filtering and class-aware NMS required"}


def scrub_metadata(path: Path, input_size: int) -> None:
    import onnx
    graph = onnx.load(str(path))
    for entry in graph.metadata_props:
        if entry.key == "description":
            entry.value = f"YOLOv8n 54-class playing-card detector, static {input_size}x{input_size}."
        if any(prefix in entry.value for prefix in ("/Users/", "/home/", "\\Users\\")):
            raise ValueError("Export metadata still contains a local user directory; refusing publication")
    onnx.save(graph, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--record", type=Path, help="Default: checkpoint's run directory/training_record.json")
    parser.add_argument("--output", type=Path, help="Default: artifacts/card_model.onnx (full) or card_model_pilot.onnx")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--imgsz", type=int, help="Default: audited training input size, or 640 for legacy records")
    parser.add_argument("--evaluate-test", action="store_true", help="Once model choices are frozen, evaluate final PyTorch and ONNX against held-out test labels")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    checkpoint = args.checkpoint.resolve()
    record_path = args.record or checkpoint.parent.parent / "training_record.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("completed") is not True or record.get("class_names") != CLASS_NAMES:
            raise ValueError("A completed training record for these 54 classes is required")
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Checkpoint does not match the audited training record")
        data = Path(record["data_yaml"])
        if sha256(data) != record["data_yaml_sha256"]:
            raise ValueError("Dataset configuration changed since training")
        report = validate(data, full=record["stage"] == "full")
        if report["errors"]:
            raise ValueError("Dataset validation failed: " + "; ".join(report["errors"][:12]))
        if report["dataset_fingerprint"] != record["dataset_fingerprint"]:
            raise ValueError("Dataset image pixels or labels changed since training")
        import yaml
        config = yaml.safe_load(data.read_text(encoding="utf-8"))
        root = Path(config.get("path", data.parent))
        if not root.is_absolute():
            root = (data.parent / root).resolve()
        if sha256(root / "split_manifest.csv") != record["split_manifest_sha256"]:
            raise ValueError("Dataset split changed since training")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Export stopped before loading any model: {error}\n")
    output_name = "card_model.onnx" if record["stage"] == "full" else "card_model_pilot.onnx"
    output = (args.output or Path("artifacts") / output_name).resolve()
    report_path = output.with_suffix(".verification.json")
    input_size = args.imgsz if args.imgsz is not None else int(record.get("input_size", 640))
    try: candidate_count(input_size)
    except ValueError as error: parser.error(str(error))
    if output.suffix.lower() != ".onnx": parser.error("output must end with .onnx")
    if output.exists() or report_path.exists():
        parser.exit(2, f"Output exists; choose a new path to preserve it: {output}\n")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from ultralytics import YOLO

    # Ultralytics writes beside its loaded checkpoint. A private copy prevents
    # an export from silently replacing an earlier sibling ONNX in the run.
    with tempfile.TemporaryDirectory(prefix="guandan-export-") as temporary:
        staging = Path(temporary).resolve()
        staged_checkpoint = staging / "checkpoint.pt"
        shutil.copy2(checkpoint, staged_checkpoint)
        model = YOLO(str(staged_checkpoint), task="detect")
        require_yolov8n(model)
        if list(model.names.values()) != CLASS_NAMES:
            parser.exit(2, "Checkpoint names are not the required 54 classes\n")
        exported = Path(model.export(format="onnx", imgsz=input_size, batch=1, opset=12, simplify=True, dynamic=False, half=False, nms=False, device="cpu"))
        if not exported.resolve().is_relative_to(staging):
            raise ValueError("Exporter did not keep intermediates in its private directory")
        scrub_metadata(exported, input_size)
        samples = sorted(path for path in (root / "images" / "val").iterdir() if path.is_file())[:args.samples]
        verification = verify_onnx(exported, YOLO(str(checkpoint), task="detect"), samples, input_size)
        if args.evaluate_test:
            pt_metrics = YOLO(str(checkpoint)).val(data=str(data), split="test", imgsz=input_size, batch=1, rect=False, device="cpu", workers=0, plots=True, project=str(output.parent / "evaluation"), name="pytorch_test")
            onnx_metrics = YOLO(str(exported), task="detect").val(data=str(data), split="test", imgsz=input_size, batch=1, rect=False, device="cpu", workers=0, plots=True, project=str(output.parent / "evaluation"), name="onnx_test")
            verification["test_metrics"] = {"pytorch": {key: float(value) for key, value in pt_metrics.results_dict.items()}, "onnx": {key: float(value) for key, value in onnx_metrics.results_dict.items()}}
        verification.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "onnx_sha256": sha256(exported), "training_record": str(record_path.resolve()), "stage": record["stage"], "class_names": CLASS_NAMES,
                             "training_input_size": record.get("input_size", 640), "export_input_size": input_size,
                             "metadata_description_sanitized": True})
        staged_report = staging / "verification.json"
        staged_report.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        publish_new_file(exported, output)
        publish_new_file(staged_report, report_path)
    print(f"Verified {record['stage']} ONNX: {output}")
    print(f"Verification report: {output.with_suffix('.verification.json')}")


if __name__ == "__main__":
    main()
