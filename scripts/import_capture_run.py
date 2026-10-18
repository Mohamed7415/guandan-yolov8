#!/usr/bin/env python3
"""Import explicitly approved native screenshots; annotations stay unreviewed."""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import tempfile
from pathlib import Path

from capture_common import FIELDS, inspect_image, parse_scenes, read_manifest, validate_timestamp

ROOT = Path(__file__).resolve().parents[1]
LABEL_FIELDS = ["image", "label", "group_id", "scene", "reviewed", "session_id"]


def csv_bytes(rows: list[dict], fields: list[str] | tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        f.write(text)
    os.replace(temp, path)


def import_run(directory: Path, project: Path, apply: bool) -> int:
    directory, project = directory.resolve(), project.resolve()
    with (directory / "captures.csv").open(encoding="utf-8", newline="") as f:
        captures = {r["filename"]: r for r in csv.DictReader(f)}
    with (directory / "review.csv").open(encoding="utf-8-sig", newline="") as f:
        review = list(csv.DictReader(f))
    rows, errors = read_manifest(project / "data/capture_manifest.csv")
    if errors and errors != ["manifest_missing"]:
        raise ValueError(str(errors))
    label_path = project / "data/manifest.csv"
    if label_path.exists():
        with label_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != LABEL_FIELDS:
                raise ValueError("标注清单列不匹配，拒绝覆盖。")
            label_rows = list(reader)
    else:
        label_rows = []
    by_name = {r["filename"]: r for r in rows}
    pixel_hashes = set()
    for row in rows:
        image = project / "data/raw" / row["filename"]
        info = inspect_image(image)
        if info["error"]:
            raise ValueError(f"现有原图无效，需先修复：{image}")
        pixel_hashes.add(info["pixel_sha256"])
    pending, seen = [], set()
    skipped = 0
    for row in review:
        if row.get("approved", "").strip().lower() not in {"true", "1", "yes"}:
            continue
        name = row["filename"]
        if name in seen or name not in captures or Path(name).name != name:
            raise ValueError(f"复核清单图片重复或路径无效：{name}")
        seen.add(name)
        capture = captures[name]
        source = directory / name
        info = inspect_image(source)
        if info["error"] or info["sha256"] != capture["sha256"]:
            raise ValueError(f"候选原图损坏或被改动：{name}")
        if name in by_name:
            if by_name[name]["sha256"] != info["sha256"]:
                raise ValueError(f"同名原图冲突：{name}")
            skipped += 1
            continue
        if info["pixel_sha256"] in pixel_hashes:
            skipped += 1
            continue
        scene = "|".join(parse_scenes(row.get("scene", "")))
        session, group = row.get("session_id", "").strip(), row.get("group_id", "").strip()
        if not session or not group:
            raise ValueError(f"缺少真实局号/序列分组：{name}")
        destination = project / "data/raw" / name
        if destination.exists():
            raise ValueError(f"不覆盖未登记的已有文件：{destination}")
        annotation = project / "data/annotations" / Path(name).with_suffix(".json")
        if annotation.exists():
            raise ValueError(f"存在同名标注，需先确认对应关系：{annotation}")
        rows.append(dict(filename=name, session_id=session, group_id=group, scene=scene,
                         captured_at=validate_timestamp(capture["captured_at"]), sha256=info["sha256"],
                         width=info["width"], height=info["height"]))
        label_rows.append(dict(image=str(destination), label=str(annotation), group_id=group,
                               scene=scene, reviewed="false", session_id=session))
        pixel_hashes.add(info["pixel_sha256"])
        pending.append((source, destination))
    print(f"已明确批准且通过完整性检查：{len(pending)}；重复/已导入跳过：{skipped}。")
    if not apply:
        print("当前为预览。检查 review.csv 后加 --apply 导入；不会自动标注或训练。")
        return len(pending)
    if not pending:
        return 0
    # Validate the whole batch before copies or writes. Keep candidate originals.
    (project / "data/raw").mkdir(parents=True, exist_ok=True)
    for source, destination in pending:
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    atomic_write(project / "data/capture_manifest.csv", csv_bytes(rows, FIELDS))
    atomic_write(label_path, csv_bytes(label_rows, LABEL_FIELDS))
    print(f"已导入 {len(pending)} 张，全部保持待标注/待复核；原有复核状态已保留。")
    return len(pending)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--apply", action="store_true", help="导入 review.csv 中 approved=true 的图片")
    args = p.parse_args()
    try:
        import_run(args.run_dir, ROOT, args.apply)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f"导入失败：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
