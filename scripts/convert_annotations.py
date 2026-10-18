#!/usr/bin/env python3
"""Validate reviewed X-AnyLabeling rectangles, then emit YOLO labels and a manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = {"self_play", "others_play", "pass", "fan", "sorted", "negative"}
FIELDS = ["image", "label", "group_id", "scene", "reviewed", "session_id"]
TRUE = {"true", "yes", "1"}


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: 缺少列 {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: 清单为空；请先采集真实截图")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: CSV 某行列数不匹配")
    return rows


def resolve(base: Path, value: str) -> Path:
    if not value.strip():
        raise ValueError("图片/标注路径不能为空")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def initialize(args: argparse.Namespace) -> None:
    """Build pending records without fabricating annotation JSON or negative labels."""
    capture = args.init_from_capture.resolve()
    target = args.manifest.resolve()
    if target.exists() and not args.overwrite:
        raise ValueError(f"{target} 已存在；请保留人工审核状态，或明确使用 --overwrite")
    rows = read_csv(capture, {"filename", "session_id", "group_id", "scene"})
    images = args.images.resolve()
    annotations = args.annotations.resolve()
    output = []
    seen = set()
    for i, row in enumerate(rows, 2):
        relative = Path(row["filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"第 {i} 行 filename 必须是 raw 内的相对路径")
        image = (images / relative).resolve()
        if not image.is_file():
            raise ValueError(f"第 {i} 行截图不存在: {image}")
        if image in seen:
            raise ValueError(f"第 {i} 行重复截图: {image}")
        seen.add(image)
        if not row["session_id"].strip() or not row["group_id"].strip():
            raise ValueError(f"第 {i} 行必须填写真实 session_id 与 group_id")
        output.append({
            "image": str(image),
            "label": str(annotations / relative.with_suffix(".json")),
            "group_id": row["group_id"],
            "scene": row["scene"],
            "reviewed": "false",
            "session_id": row["session_id"],
        })
    write_csv(target, output, FIELDS)
    print(f"已建立 {len(output)} 条待标注记录: {target}（未生成标注或训练负样本）")


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("需要 Pillow；请使用 .venv-label/bin/python 或训练环境运行") from exc
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
        if orientation not in (None, 1):
            raise ValueError("图片含旋转 EXIF，请统一方向后重新标注")
    if width <= 0 or height <= 0:
        raise ValueError("无效图片尺寸")
    return width, height


def scenes_for(row: dict[str, str], doc: dict) -> list[str]:
    scenes = [s.strip() for s in re.split(r"[|;]", row["scene"]) if s.strip()]
    if not scenes:
        flags = doc.get("flags") or {}
        scenes = [s for s in sorted(SCENES) if flags.get(s) is True]
    if not scenes or not set(scenes) <= SCENES:
        raise ValueError(f"scene 须属于 {', '.join(sorted(SCENES))}，多个用 | 分隔")
    return list(dict.fromkeys(scenes))


def number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("矩形坐标必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("矩形坐标不可包含 NaN/Infinity")
    return result


def convert_one(row: dict[str, str], base: Path, classes: dict[str, int]):
    image = resolve(base, row["image"])
    annotation = resolve(base, row["label"])
    if annotation.suffix.lower() != ".json":
        raise ValueError("输入 label 必须是 X-AnyLabeling JSON")
    if not row["group_id"].strip():
        raise ValueError("group_id 不能为空，不能随机拆分临近帧")
    if not image.is_file() or not annotation.is_file():
        raise ValueError(f"缺少图片或人工标注: {image} / {annotation}")
    doc = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("JSON 根节点必须是对象")
    if doc.get("flags") is not None and not isinstance(doc["flags"], dict):
        raise ValueError("JSON flags 必须为对象或 null")
    width, height = image_size(image)
    if doc.get("imageWidth") != width or doc.get("imageHeight") != height:
        raise ValueError(f"JSON 尺寸与实际图片 {width}×{height} 不一致")
    image_path = doc.get("imagePath")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("JSON imagePath 缺失")
    if resolve(annotation.parent, image_path) != image:
        raise ValueError("JSON imagePath 与清单 image 不一致，可能标注了错误图片")
    shapes = doc.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("JSON shapes 必须是数组；未标注文件不能当负样本")
    scenes = scenes_for(row, doc)
    flags = doc.get("flags") or {}
    negative = "negative" in scenes or flags.get("negative") is True
    if negative and shapes:
        raise ValueError("negative 图片含目标，请复核场景或标注")
    if not shapes and not negative:
        raise ValueError("空 shapes 必须显式标记 negative，且经人工审核")
    lines = []
    counts = Counter()
    boxes_seen = set()
    for i, shape in enumerate(shapes, 1):
        if not isinstance(shape, dict) or shape.get("shape_type") != "rectangle":
            raise ValueError(f"目标 {i}: 只接受 rectangle")
        label = shape.get("label")
        if label not in classes:
            raise ValueError(f"目标 {i}: 未知类别 {label!r}")
        points = shape.get("points")
        if (not isinstance(points, list) or len(points) not in (2, 4)
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise ValueError(f"目标 {i}: rectangle 应有两个对角点或四个轴对齐顶点")
        coordinates = [(number(p[0]), number(p[1])) for p in points]
        x1, x2 = min(p[0] for p in coordinates), max(p[0] for p in coordinates)
        y1, y2 = min(p[1] for p in coordinates), max(p[1] for p in coordinates)
        if len(coordinates) == 4:
            corners = {(x1, y1), (x2, y1), (x2, y2), (x1, y2)}
            if len(corners) != 4 or set(coordinates) != corners:
                raise ValueError(f"目标 {i}: 四点 rectangle 须包含轴对齐矩形的四个不同顶点；不接受旋转/斜四边形")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"目标 {i}: 零面积或越界矩形，须在原图内修正")
        key = (label, x1, y1, x2, y2)
        if key in boxes_seen:
            raise ValueError(f"目标 {i}: 完全重复标注")
        boxes_seen.add(key)
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        lines.append(f"{classes[label]} " + " ".join(f"{v:.8f}" for v in values))
        counts[label] += 1
    return image, annotation, scenes, lines, counts


def convert(args: argparse.Namespace) -> None:
    source = args.manifest.resolve()
    output = args.output.resolve()
    if source == output / "manifest.csv":
        raise ValueError("输入清单不可与输出清单相同")
    names = args.classes.read_text(encoding="utf-8-sig").splitlines()
    expected = [r+s for r in ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")
                for s in ("♠", "♥", "♦", "♣")] + ["SJ", "BJ"]
    if names != expected:
        raise ValueError("classes.txt 必须严格保持 MODEL_SPEC 的 54 类及顺序")
    classes = {name: index for index, name in enumerate(names)}
    rows = read_csv(source, {"image", "label", "group_id", "scene", "reviewed"})
    errors, prepared, seen, counts = [], [], set(), Counter()
    skipped = 0
    for line, row in enumerate(rows, 2):
        if row["reviewed"].strip().lower() not in TRUE:
            if args.skip_unreviewed:
                skipped += 1
                continue
            errors.append(f"第 {line} 行: reviewed 非 true；请审核或使用 --skip-unreviewed")
            continue
        try:
            image, annotation, scenes, labels, per_class = convert_one(row, source.parent, classes)
            if image in seen:
                raise ValueError("清单含重复图片")
            seen.add(image)
            # Hash the full path to avoid collisions between same-named source images.
            stem = image.stem + "__" + hashlib.sha256(str(image).encode()).hexdigest()[:12]
            target = output / "labels" / f"{stem}.txt"
            record = {"image": str(image), "label": str(target), "group_id": row["group_id"],
                      "scene": "|".join(scenes), "reviewed": "true",
                      "session_id": row.get("session_id", "").strip(),
                      "source_annotation": str(annotation)}
            prepared.append((target, labels, record))
            counts.update(per_class)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            errors.append(f"第 {line} 行: {exc}")
    if errors:
        raise ValueError("转换前校验失败，未写训练标签：\n" + "\n".join(errors[:30])
                         + (f"\n另有 {len(errors)-30} 项错误" if len(errors) > 30 else ""))
    if not prepared:
        raise ValueError("没有已审核样本；不会生成空训练集")
    targets = [p[0] for p in prepared] + [output / "manifest.csv", output / "conversion_report.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise ValueError("输出已存在；请使用新目录或明确 --overwrite（原始 JSON 不会被更改）")
    for target, labels, _ in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    write_csv(output / "manifest.csv", [p[2] for p in prepared], FIELDS + ["source_annotation"])
    report = {"source_manifest": str(source), "images": len(prepared), "skipped_unreviewed": skipped,
              "objects": sum(counts.values()), "class_counts": {name: counts[name] for name in names},
              "missing_classes": [name for name in names if not counts[name]],
              "below_50_objects": [name for name in names if counts[name] < 50]}
    (output / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"已转换 {len(prepared)} 张真实图片、{sum(counts.values())} 个目标；跳过未审核 {skipped} 张")
    print(f"训练清单: {output / 'manifest.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/converted")
    parser.add_argument("--classes", type=Path, default=ROOT / "configs/classes.txt")
    parser.add_argument("--init-from-capture", type=Path, help="仅由真实采集清单建立待标注 manifest")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/annotations")
    parser.add_argument("--skip-unreviewed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        initialize(args) if args.init_from_capture else convert(args)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
