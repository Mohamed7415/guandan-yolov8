#!/usr/bin/env python3
"""Passive macOS window capture. Never sends input or starts a game."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ["filename", "captured_at", "sha256", "width", "height", "window_id",
              "sequence_id", "source", "scene", "reviewed"]
WINDOW_TITLE = "天天爱掼蛋"
OWNERS = {"WeChat", "Weixin", "微信"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def choose_window(windows: list[dict], title: str, window_id: int | None = None) -> dict:
    matches = []
    for window in windows:
        bounds = window.get("kCGWindowBounds", {})
        if (str(window.get("kCGWindowOwnerName", "")) in OWNERS
                and str(window.get("kCGWindowName", "")).strip() == title
                and int(window.get("kCGWindowLayer", -1)) == 0
                and bounds.get("Width", 0) >= 400 and bounds.get("Height", 0) >= 250
                and (window_id is None or int(window["kCGWindowNumber"]) == window_id)):
            matches.append(window)
    if not matches:
        raise LookupError(f"找不到可见的微信窗口「{title}」。请打开小程序，并保持窗口未最小化。")
    if len(matches) != 1:
        ids = ", ".join(str(w["kCGWindowNumber"]) for w in matches)
        raise LookupError(f"同名小程序窗口不止一个，请用 --window-id 指定：{ids}")
    return matches[0]


class MacWindowCapture:
    def __init__(self, title: str, window_id: int | None):
        if sys.platform != "darwin":
            raise RuntimeError("此采集器使用 macOS 窗口截图功能，仅支持 Mac。")
        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError("缺少截图依赖。请运行 .venv/bin/python -m pip install -r requirements-capture.txt") from exc
        self.quartz = Quartz
        self.title = title
        self.window_id = window_id
        if not Quartz.CGPreflightScreenCaptureAccess():
            raise RuntimeError(
                "macOS 尚未授予当前启动应用屏幕录制权限。请在「系统设置 → 隐私与安全性 → "
                "屏幕与系统音频录制」中，手动允许启动脚本的应用（双击启动通常是终端），"
                "按系统要求重新打开该应用，再启动本脚本。脚本不会自行修改权限。")

    def locate(self) -> dict:
        q = self.quartz
        windows = q.CGWindowListCopyWindowInfo(
            q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements,
            q.kCGNullWindowID)
        return choose_window(list(windows or []), self.title, self.window_id)

    def capture(self, directory: Path) -> tuple[bytes, int]:
        window = self.locate()
        window_id = int(window["kCGWindowNumber"])
        with tempfile.TemporaryDirectory(prefix=".frame-", dir=directory) as temporary:
            path = Path(temporary) / "window.png"
            result = subprocess.run(
                ["/usr/sbin/screencapture", "-x", "-o", "-t", "png", "-l", str(window_id), str(path)],
                capture_output=True, timeout=15, check=False)
            if result.returncode or not path.exists():
                raise RuntimeError("窗口截图失败。请确认屏幕录制权限及小程序窗口状态；不会回退为全屏截图。")
            # Confirm the same target still exists after capture. Do not accept a reused ID.
            current = self.locate()
            if int(current["kCGWindowNumber"]) != window_id:
                raise LookupError("截图过程中小程序窗口已改变，本帧已丢弃。")
            return path.read_bytes(), window_id


def frame_features(data: bytes) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float]:
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG":
            raise ValueError("截图并非原生 PNG。")
        image.load()
        if min(image.size) < 250:
            raise ValueError("截图尺寸不足。")
        rgb = image.convert("RGB")
        if max(ImageStat.Stat(rgb).stddev) < 3:
            raise ValueError("截图为空白、黑屏或未获授权。")
        # Fixed relative ROI excludes the title bar, avatars at the sides, and bottom HUD.
        # This is only a candidate filter, not card detection or class annotation.
        w, h = rgb.size
        region = rgb.crop((round(w * .12), round(h * .25), round(w * .88), round(h * .91)))
        small = np.asarray(region.resize((192, 128), Image.Resampling.BILINEAR)).astype(np.int16)
        white = (small.min(axis=2) > 172) & ((small.max(axis=2) - small.min(axis=2)) < 65)
        return small, white, rgb.size, float(white.mean())


def card_change(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    a, am = left
    b, bm = right
    shape_delta = float(np.not_equal(am, bm).mean())
    union = am | bm
    if not union.any():
        return shape_delta, 0.0
    color_delta = float((np.abs(a - b).max(axis=2)[union] > 24).mean())
    return shape_delta, color_delta


def looks_like_green_table(data: bytes) -> bool:
    """Conservative collection filter; not a detector or an annotation."""
    with Image.open(io.BytesIO(data)) as image:
        w, h = image.size
        region = image.convert("RGB").crop((round(w * .2), round(h * .26), round(w * .8), round(h * .7)))
        pixels = np.asarray(region.resize((128, 80))).astype(np.int16)
    green = ((pixels[:, :, 1] > pixels[:, :, 0] * 1.25)
             & (pixels[:, :, 1] > pixels[:, :, 2] * .8)
             & (pixels[:, :, 0] < 135))
    return float(green.mean()) > .45


def load_run(directory: Path) -> tuple[list[dict], list[tuple[np.ndarray, np.ndarray]], set[str]]:
    rows, history, hashes = [], [], set()
    path = directory / "captures.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_FIELDS:
                raise ValueError("已有采集清单格式不匹配，拒绝覆盖。")
            rows = list(reader)
        for row in rows:
            name = Path(row["filename"])
            if name.is_absolute() or len(name.parts) != 1:
                raise ValueError("已有清单图片路径不合法。")
            data = (directory / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"已有截图 hash 不匹配：{name}")
            colors, mask, _, _ = frame_features(data)
            history.append((colors, mask))
            hashes.add(row["sha256"])
    # A crash between image save and CSV append leaves an orphan. Keep it and refuse
    # resume until reviewed; never overwrite or silently pretend it was registered.
    known = {row["filename"] for row in rows}
    orphans = {p.name for p in directory.glob("*.png")} - known
    if orphans:
        raise ValueError(f"发现未登记图片，需先检查或改用新批次：{sorted(orphans)[:3]}")
    return rows, history, hashes


def write_review_queue(directory: Path, rows: list[dict]) -> None:
    path = directory / "review.csv"
    fields = ["filename", "session_id", "group_id", "scene", "approved"]
    existing = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            existing = {r["filename"]: r for r in csv.DictReader(stream)}
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(existing.get(row["filename"], dict(
                filename=row["filename"], session_id=row["sequence_id"],
                group_id=row["sequence_id"], scene="", approved="false")))
    temporary.replace(path)


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须是有限的正数")
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="只截微信指定小程序窗口；不点击、不出牌、不开新局。")
    p.add_argument("--target", type=int, default=200, help="本批候选截图目标数（默认 200，不是已复核训练样本数）")
    p.add_argument("--interval", type=positive_float, default=2.0, help="采样间隔秒数，至少 0.5")
    p.add_argument("--max-minutes", type=positive_float, default=60, help="最长运行分钟数，默认 60")
    p.add_argument("--run-dir", type=Path, help="恢复已有批次；默认在 data/candidates 下新建目录")
    p.add_argument("--sequence-id", help="同一连续采集序列 ID；恢复时必须相同，不自动推断新一局")
    p.add_argument("--window-title", default=WINDOW_TITLE)
    p.add_argument("--window-id", type=int, help="存在多个同名窗口时指定窗口 ID")
    p.add_argument("--change-threshold", type=positive_float, default=.004, help="牌面白色区域变化阈值，默认 0.004")
    p.add_argument("--include-other-screens", action="store_true", help="也收集无明显白色牌面的候选，仍需复核")
    p.add_argument("--gameplay-only", action="store_true", help="只收集含绿色牌桌的候选，跳过大厅/广告/结算")
    p.add_argument("--check", action="store_true", help="仅检查权限和目标窗口，不截图、不发送输入")
    return p


def run(args: argparse.Namespace) -> int:
    if args.target < 1 or args.target > 20000:
        raise ValueError("--target 必须在 1..20000 之间。")
    if args.interval < .5 or args.change_threshold >= .1:
        raise ValueError("--interval 至少 0.5；--change-threshold 必须小于 0.1。")
    capture = MacWindowCapture(args.window_title, args.window_id)
    if args.check:
        w = capture.locate()
        print(json.dumps({"ready": True, "title": w["kCGWindowName"],
                          "window_id": int(w["kCGWindowNumber"]), "sends_input": False}, ensure_ascii=False, indent=2))
        return 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    directory = (args.run_dir or ROOT / "data/candidates" / run_id).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # One writer per batch; OS releases the lock on interruption/crash.
    import fcntl
    lock = (directory / ".capture.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError("这个批次已在另一个采集进程中运行。") from exc
    metadata_path = directory / "run.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sequence_id = metadata["sequence_id"]
        if args.sequence_id and args.sequence_id != sequence_id:
            raise ValueError("恢复批次的 sequence-id 不一致。")
        if metadata["window_title"] != args.window_title:
            raise ValueError("恢复批次的窗口标题不一致。")
    else:
        sequence_id = args.sequence_id or "capture_" + run_id
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", sequence_id):
            raise ValueError("sequence-id 只接受 1..100 个英文、数字、连字符或下划线。")
        metadata = {"created_at": now(), "sequence_id": sequence_id, "window_title": args.window_title,
                    "source": "current_wechat_miniprogram_window", "sends_input": False,
                    "candidate_only": True, "reviewed": False,
                    "grouping_note": "A whole capture sequence stays together unless actual round boundaries are reviewed. New process is not evidence of a new round."}
    rows, history, hashes = load_run(directory)
    history_white = [float(mask.mean()) for _, mask in history]
    write_review_queue(directory, rows)
    counts = {"saved": len(rows), "duplicate": 0, "unstable": 0, "no_cardlike_area": 0, "not_gameplay": 0, "window_waits": 0}
    metadata.pop("error", None)
    metadata.pop("finished_at", None)
    metadata.update(started_at=now(), target=args.target, interval=args.interval, status="running", counts=counts)
    atomic_json(metadata_path, metadata)
    print(f"采集目录：{directory}\n只观察窗口，不操作游戏。Ctrl+C 安全停止。")
    print(f"已有 {len(rows)} 张候选，目标 {args.target}；候选图片须复核后才计入训练。", flush=True)
    previous = None
    previous_size = None
    started = time.monotonic()
    last_heartbeat = started
    status, exit_code, error = "time_limit", 0, None
    try:
        while len(rows) < args.target and time.monotonic() - started < args.max_minutes * 60:
            sampled = time.monotonic()
            try:
                data, window_id = capture.capture(directory)
            except LookupError as exc:
                previous = None
                counts["window_waits"] += 1
                if counts["window_waits"] == 1 or time.monotonic() - last_heartbeat >= 20:
                    print(f"等待：{exc}", flush=True)
                    last_heartbeat = time.monotonic()
                time.sleep(min(args.interval, 5))
                continue
            timestamp = now()
            colors, mask, size, white_fraction = frame_features(data)
            current = (colors, mask)
            if args.gameplay_only and not looks_like_green_table(data):
                counts["not_gameplay"] += 1
                previous = None
            elif previous is None or previous_size != size:
                previous, previous_size = current, size
                counts["unstable"] += 1
            else:
                # Accept only states stable across consecutive sampling observations.
                delta, _ = card_change(previous, current)
                previous = current
                if delta > .007:
                    counts["unstable"] += 1
                elif not args.include_other_screens and not .008 <= white_fraction <= .65:
                    counts["no_cardlike_area"] += 1
                else:
                    digest = hashlib.sha256(data).hexdigest()
                    duplicate = digest in hashes
                    if not duplicate:
                        for old, old_white in zip(history, history_white):
                            # Mask disagreement is at least the difference in
                            # white coverage. Skip impossible duplicates without
                            # comparing every pixel of every previous screenshot.
                            if abs(old_white-white_fraction) >= args.change_threshold:
                                continue
                            shape_delta, color_delta = card_change(old, current)
                            if shape_delta < args.change_threshold and color_delta < .08:
                                duplicate = True
                                break
                    if duplicate:
                        counts["duplicate"] += 1
                    else:
                        filename = f"{sequence_id}_{len(rows) + 1:05d}.png"
                        target = directory / filename
                        # Original native PNG bytes, no resizing, crop, annotation, or rewrite.
                        with target.open("xb") as stream:
                            stream.write(data)
                            stream.flush()
                            os.fsync(stream.fileno())
                        row = dict(filename=filename, captured_at=timestamp, sha256=digest,
                                   width=size[0], height=size[1], window_id=window_id,
                                   sequence_id=sequence_id, source="current_wechat_miniprogram_window",
                                   scene="", reviewed="false")
                        csv_path = directory / "captures.csv"
                        first = not csv_path.exists()
                        with csv_path.open("a", encoding="utf-8", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                            if first:
                                writer.writeheader()
                            writer.writerow(row)
                            stream.flush()
                            os.fsync(stream.fileno())
                        rows.append(row)
                        write_review_queue(directory, rows)
                        history.append(current)
                        history_white.append(white_fraction)
                        hashes.add(digest)
                        counts["saved"] = len(rows)
                        print(f"候选 {len(rows)}/{args.target}  {filename}  {size[0]}×{size[1]}", flush=True)
            if time.monotonic() - last_heartbeat >= 20:
                print(f"观察中：已存 {len(rows)}；跳过重复 {counts['duplicate']}、变化中 {counts['unstable']}。", flush=True)
                atomic_json(directory / "progress.json", {"updated_at": now(), **counts})
                last_heartbeat = time.monotonic()
            time.sleep(max(0, args.interval - (time.monotonic() - sampled)))
        if len(rows) >= args.target:
            status = "candidate_target_reached"
    except KeyboardInterrupt:
        status = "stopped_by_user"
    except Exception as exc:
        status, exit_code, error = "error", 2, str(exc)
    finally:
        metadata.update(status=status, finished_at=now(), counts=counts, error=error)
        atomic_json(metadata_path, metadata)
        atomic_json(directory / "progress.json", {"updated_at": now(), **counts})
        lock.close()
    print(f"已停止：{status}；本批 {len(rows)} 张候选。原始训练数据未改动。")
    if error:
        print(f"错误：{error}", file=sys.stderr)
    return exit_code


def stop_capture(_signal, _frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    # The double-click launcher runs this process in the background. Handle both
    # signals explicitly so launcher cleanup always flushes status and exits.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, stop_capture)
    args = parser().parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, LookupError) as exc:
        print(f"无法启动：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
