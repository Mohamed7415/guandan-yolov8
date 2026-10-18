"""Shared data contract for the 54-class card detector."""
from __future__ import annotations

import csv
import math
from pathlib import Path

CLASS_NAMES = [rank + suit for rank in ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"] for suit in ["♠", "♥", "♦", "♣"]] + ["SJ", "BJ"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TRUE_VALUES = {"true", "yes", "1"}


def manifest_rows(path: Path) -> list[dict]:
    """Resolve paths at the manifest, never at the current working directory."""
    path = path.resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "label", "group_id", "scene", "reviewed"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        rows = []
        seen = set()
        for number, row in enumerate(reader, 2):
            if not any(row.values()):
                continue
            for key in required:
                row[key] = (row.get(key) or "").strip()
            if row["reviewed"].lower() not in TRUE_VALUES:
                raise ValueError(f"Manifest line {number}: reviewed must be true after human review")
            if not row["group_id"] or not row["scene"]:
                raise ValueError(f"Manifest line {number}: group_id and scene are required")
            for key in ("image", "label"):
                if not row[key]:
                    raise ValueError(f"Manifest line {number}: {key} path is missing")
                candidate = Path(row[key]).expanduser()
                resolved = candidate.resolve() if candidate.is_absolute() else (path.parent / candidate).resolve()
                if not resolved.is_file():
                    raise ValueError(f"Manifest line {number}: missing {key}: {resolved}")
                row[key] = str(resolved)
            if row["image"] in seen:
                raise ValueError(f"Manifest line {number}: repeated image path: {row['image']}")
            seen.add(row["image"])
            if Path(row["image"]).suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Manifest line {number}: unsupported image extension")
            if Path(row["label"]).suffix.lower() != ".txt":
                raise ValueError(f"Manifest line {number}: expected YOLO .txt; convert X-AnyLabeling JSON first")
            rows.append(row)
    if not rows:
        raise ValueError("Manifest has no reviewed images; no dataset or model will be generated")
    return rows


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO normalized cx cy w h labels; explicit empty files are reviewed negatives."""
    boxes = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{number}: expected class_id cx cy w h (five values)")
        try:
            cls = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{number}: invalid class or coordinates") from error
        values = (cx, cy, width, height)
        if not 0 <= cls < len(CLASS_NAMES) or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{number}: class must be 0..53 and coordinates finite")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{number}: coordinates must be normalized; width and height positive")
        if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{number}: box extends outside the image")
        box = (cls, cx, cy, width, height)
        if box in seen:
            raise ValueError(f"{path}:{number}: duplicated bounding box")
        seen.add(box)
        boxes.append(box)
    return boxes


def require_yolov8n(model) -> None:
    """Verify architecture metadata and actual tensor shapes against official v8n."""
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    network = model.model
    if type(network) is not DetectionModel or model.task != "detect":
        raise ValueError("This pipeline requires the YOLOv8n DetectionModel architecture")
    expected = yaml_model_load("yolov8n.yaml")
    actual = network.yaml
    nano_scale = actual.get("scale") == "n"
    if not actual.get("scale"):
        # Official v8.3.0 release assets retain the original 2023 config format.
        nano_scale = actual.get("depth_multiple") == 0.33 and actual.get("width_multiple") == 0.25
    if not nano_scale or any(actual.get(key) != expected.get(key) for key in ("backbone", "head")):
        raise ValueError("Checkpoint is not the official YOLOv8 nano architecture (yolov8n)")
    classes = network.model[-1].nc
    canonical = DetectionModel(expected, ch=3, nc=classes, verbose=False)
    expected_shapes = {name: tuple(tensor.shape) for name, tensor in canonical.state_dict().items()}
    actual_shapes = {name: tuple(tensor.shape) for name, tensor in network.state_dict().items()}
    # A fused checkpoint has no batch-norm parameters. Accept it only if its
    # fused tensor layout also matches a fused official nano model.
    if actual_shapes != expected_shapes:
        canonical.fuse(verbose=False)
        expected_shapes = {name: tuple(tensor.shape) for name, tensor in canonical.state_dict().items()}
    if actual_shapes != expected_shapes:
        raise ValueError("Checkpoint tensor shapes do not match the YOLOv8n architecture")
