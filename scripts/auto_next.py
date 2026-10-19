#!/usr/bin/env python3
"""Continue a known ordinary-game settlement; no lobby navigation or purchases."""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from auto_capture import MacWindowCapture, ROOT, frame_features, now, positive_float
from atomic_io import atomic_json


class SettlementDetector:
    def __init__(self, directory: Path = ROOT / "configs/ui_templates"):
        self.layout = json.loads((directory / "layout.json").read_text(encoding="utf-8"))
        self.templates = {
            name: np.asarray(Image.open(directory / f"{name}.png").convert("RGB"))
            for name in self.layout["regions"]}
        self.variants = {name: [np.asarray(Image.open(p).convert("RGB"))
                                for p in sorted(directory.glob(f"{name}__*.png"))]
                         for name in self.templates}
        extra = directory / 'extra_actions.json'
        self.extra_actions = json.loads(extra.read_text()) if extra.exists() else {}
        self.extra_templates = {kind:{name:np.asarray(Image.open(directory/spec['directory']/f'{name}.png').convert('RGB'))
                                      for name in spec['regions']}
                                for kind,spec in self.extra_actions.items()}
        self.popup_actions = ['level_up','free_gold_base',*self.extra_actions]
        self.level_regions = {'header': (640,177,1010,304),
                              'reward_label': (672,337,936,379),
                              'confirm': (505,813,792,909)}
        self.level_templates = {name: np.asarray(Image.open(directory/'level_up'/f'{name}.png').convert('RGB'))
                                for name in self.level_regions
                                if (directory/'level_up'/f'{name}.png').is_file()}
        self.gold_regions = {'header': (110,190,570,284), 'coin': (763,455,864,533),
                             'base': (490,753,783,852)}
        self.gold_templates = {name: np.asarray(Image.open(directory/'free_gold_base'/f'{name}.png').convert('RGB'))
                               for name in self.gold_regions
                               if (directory/'free_gold_base'/f'{name}.png').is_file()}
        self.wait_regions = {'center': (714,329,892,384),
                             'left': (221,443,401,499), 'self': (750,694,868,754)}
        self.wait_templates = {name: np.asarray(Image.open(directory/'waiting_ready'/f'{name}.png').convert('RGB'))
                               for name in self.wait_regions
                               if (directory/'waiting_ready'/f'{name}.png').is_file()}

    @staticmethod
    def popup_matches(rgb, templates, regions):
        if len(templates) != 3: return False
        for name, template in templates.items():
            x1,y1,x2,y2=regions[name]
            region=rgb[y1:y2,x1:x2]
            corr=float(cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED)[0,0])
            error=float(np.abs(region.astype(np.int16)-template.astype(np.int16)).mean())
            if not math.isfinite(corr) or corr < .93 or error > 20: return False
        return True

    def inspect(self, data: bytes) -> dict:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            w, h = image.size
            rw, rh = self.layout["reference_size"]
            if abs((w / h) / (rw / rh) - 1) > .025:
                return {"settlement": False, "gameplay": False, "reason": "window_aspect_changed"}
            rgb = np.asarray(image.convert("RGB").resize((rw, rh), Image.Resampling.LANCZOS))
        scores, errors = {}, {}
        # All three independent, fixed-position anchors must match. Correlation
        # alone accepts dimmed dialogs; absolute RGB error also blocks overlays.
        for name, (x1, y1, x2, y2) in self.layout["regions"].items():
            region = rgb[y1:y2, x1:x2]
            options = []
            for template in [self.templates[name], *self.variants[name]]:
                score = float(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                score = score if math.isfinite(score) else 0
                error = float(np.abs(region.astype(np.int16) - template.astype(np.int16)).mean())
                passed = score >= self.layout["minimum_correlation"] and error <= self.layout["maximum_color_error"]
                options.append((passed, score - error / 255, score, error))
            _, _, scores[name], errors[name] = max(options)
        settlement = all(scores[n] >= self.layout["minimum_correlation"]
                         and errors[n] <= self.layout["maximum_color_error"] for n in scores)
        # A green empty matchmaking table also contains white HUD icons. Require
        # visible neutral card surfaces above the bottom toolbar/hosting banner
        # before counting a dealt game. This UI guard does not classify cards.
        table = rgb[260:700, 320:1300].astype(np.int16)
        green = float(((table[:, :, 1] > table[:, :, 0] * 1.25)
                       & (table[:, :, 1] > table[:, :, 2] * .8)
                       & (table[:, :, 0] < 135)).mean())
        _, _, _, white = frame_features(data)
        hand = rgb[700:810, 230:1515].astype(np.int16)
        hand_white = float(((hand.min(axis=2) > 150)
                            & (hand.max(axis=2) - hand.min(axis=2) < 45)).mean())
        gameplay = (not settlement and green > .45 and .012 < white < .55
                    and hand_white > .025)
        level_up = self.popup_matches(rgb,self.level_templates,self.level_regions)
        free_gold_base = self.popup_matches(rgb,self.gold_templates,self.gold_regions)
        extras = {kind:self.popup_matches(rgb,self.extra_templates[kind],spec['regions'])
                  for kind,spec in self.extra_actions.items()}
        waiting_ready = (not settlement and green > .45
                         and self.popup_matches(rgb, self.wait_templates, self.wait_regions))
        if level_up or free_gold_base or any(extras.values()):
            gameplay = waiting_ready = False
        if waiting_ready: gameplay = False
        return dict(settlement=settlement, gameplay=gameplay, scores=scores,
                    color_errors=errors, green_fraction=green, white_fraction=white,
                    hand_white_fraction=hand_white,
                    waiting_ready=bool(waiting_ready),
                    level_up=bool(level_up), free_gold_base=bool(free_gold_base), **extras)

    def target_point(self, window: dict, kind: str = 'settlement') -> tuple[float, float]:
        b = window["kCGWindowBounds"]
        x, y = {'level_up':(648,861),'free_gold_base':(636,802)}.get(kind,self.layout["next_button_center"])
        if kind in self.extra_actions: x,y=self.extra_actions[kind]['point']
        w, h = self.layout["reference_size"]
        return b["X"] + x * b["Width"] / w, b["Y"] + y * b["Height"] / h


class ContinueState:
    """One click per settlement, then require an observed game before re-arming."""
    def __init__(self):
        self.pending_since = None
        self.stable_settlement = 0
        self.game_observations = 0
        self.completed_transitions = 0
        self.known_ready_wait_observed = False

    def observe(self, state: dict, current_time: float) -> str:
        if self.pending_since is not None:
            # A native ready screen can display a90-second countdown. A known
            # three-anchor match grants one fixed120-second total deadline,
            # never a sliding extension or another click. Unknown states alone
            # retain the45-second stop; only a dealt hand confirms completion.
            if state.get('waiting_ready'):
                self.known_ready_wait_observed = True
            deadline = 120 if self.known_ready_wait_observed else 45
            if current_time - self.pending_since >= deadline:
                return "transition_failed_stop"
            self.game_observations = self.game_observations + 1 if state["gameplay"] else 0
            if self.game_observations >= 3:
                self.completed_transitions += 1
                self.pending_since = None
                self.game_observations = self.stable_settlement = 0
                return "new_game_observed"
            return "awaiting_game"
        self.stable_settlement = self.stable_settlement + 1 if state["settlement"] else 0
        return "ready_to_continue" if self.stable_settlement >= 2 else "observe"

    def clicked(self, current_time: float) -> None:
        self.pending_since = current_time
        self.stable_settlement = 0
        self.game_observations = 0
        self.known_ready_wait_observed = False


def accessibility_allowed() -> bool:
    library = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
    library.AXIsProcessTrusted.restype = ctypes.c_bool
    return bool(library.AXIsProcessTrusted())


def window_point_clear(windows: list[dict], target_id: int, point: tuple[float, float]) -> bool:
    ordinary = [w for w in windows if int(w.get("kCGWindowLayer", -1)) == 0
                and w.get("kCGWindowAlpha", 1) > 0
                and w.get("kCGWindowBounds", {}).get("Width", 0) > 1]
    if not ordinary or int(ordinary[0]["kCGWindowNumber"]) != target_id:
        return False
    target_bounds = ordinary[0]["kCGWindowBounds"]
    for item in windows:
        if int(item["kCGWindowNumber"]) == target_id:
            return True
        # Computer Use's cursor visualization follows the last clicked point;
        # it is an input-transparent overlay, not another application's dialog.
        if (item.get("kCGWindowOwnerName") == "ChatGPT Computer Use"
                and item.get("kCGWindowName") == "Software Cursor"
                and int(item.get("kCGWindowLayer", 0)) == 102):
            continue
        if (item.get("kCGWindowOwnerName") == "Window Server"
                and item.get("kCGWindowName") == "Cursor"
                and int(item.get("kCGWindowLayer", 0)) == 2147483630):
            continue
        b = item.get("kCGWindowBounds", {})
        # macOS retains full-display transparent compositing windows for these
        # system surfaces even when their visible controls do not cover the app.
        # Keep checking actual small panels and every other application's window.
        persistent_compositor = (item.get("kCGWindowOwnerName") in {
            "Window Server", "程序坞", "Dock", "通知中心", "Notification Center", "ChatGPT Computer Use"}
            and int(item.get("kCGWindowLayer", 0)) >= 20
            and b.get("Width", 0) >= target_bounds.get("Width", 0) * 1.5
            and b.get("Height", 0) >= target_bounds.get("Height", 0) * 1.5)
        if persistent_compositor:
            continue
        if (item.get("kCGWindowAlpha", 1) > 0
                and b.get("X", 0) <= point[0] < b.get("X", 0) + b.get("Width", 0)
                and b.get("Y", 0) <= point[1] < b.get("Y", 0) + b.get("Height", 0)):
            return False
    return False


def frontmost(window: dict, point: tuple[float, float]) -> bool:
    from AppKit import NSWorkspace
    import Quartz as q
    active = NSWorkspace.sharedWorkspace().frontmostApplication()
    if active is None or int(active.processIdentifier()) != int(window["kCGWindowOwnerPID"]):
        return False
    windows = q.CGWindowListCopyWindowInfo(
        q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements, q.kCGNullWindowID)
    return window_point_clear(list(windows or []), int(window["kCGWindowNumber"]), point)


def activate_game_window(window: dict) -> bool:
    """Raise only the already located game window, retaining the foreground guard."""
    from AppKit import (NSRunningApplication, NSApplicationActivateIgnoringOtherApps,
                        NSApplicationActivateAllWindows)
    from ApplicationServices import (AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
        AXUIElementPerformAction, AXUIElementSetAttributeValue, kAXWindowsAttribute,
        kAXTitleAttribute, kAXRaiseAction, kAXFrontmostAttribute)

    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(window["kCGWindowOwnerPID"]))
    if app is None or not app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps | NSApplicationActivateAllWindows):
        return False
    ax_app = AXUIElementCreateApplication(int(window["kCGWindowOwnerPID"]))
    err, windows = AXUIElementCopyAttributeValue(ax_app, kAXWindowsAttribute, None)
    matching = []
    for candidate in windows or []:
        error, title = AXUIElementCopyAttributeValue(candidate, kAXTitleAttribute, None)
        if not error and title == "天天爱掼蛋":
            matching.append(candidate)
    if err or len(matching) != 1 or AXUIElementPerformAction(matching[0], kAXRaiseAction):
        return False
    if AXUIElementSetAttributeValue(ax_app, kAXFrontmostAttribute, True) != 0:
        return False
    # Activation can still be pending when AX returns. Process its events
    # before the caller rechecks all foreground/window guards.
    from Foundation import NSDate, NSRunLoop
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(.25))
    return True


def same_window(before: dict, after: dict, captured_id: int) -> bool:
    return (int(before["kCGWindowNumber"]) == captured_id == int(after["kCGWindowNumber"])
            and int(before["kCGWindowOwnerPID"]) == int(after["kCGWindowOwnerPID"])
            and dict(before["kCGWindowBounds"]) == dict(after["kCGWindowBounds"]))


def click_verified(capture: MacWindowCapture, detector: SettlementDetector, output: Path, targeted: bool = False, kind: str = 'settlement', activate_before_click: bool = False) -> bool:
    if kind not in {'settlement',*detector.popup_actions}:
        raise ValueError('Unrecognized input action')
    if not accessibility_allowed():
        raise RuntimeError("缺少辅助功能权限，未发送点击。请在系统设置中手动允许启动脚本的应用后重开脚本。")
    action_name={'level_up':'known_level_up_confirm',
                 'free_gold_base':'known_free_virtual_gold_base_claim',
                 'settlement':'ordinary_settlement_next_round',
                 **{k:v['action'] for k,v in detector.extra_actions.items()}}[kind]
    # A brief animation may hide then restore a modal. Its reappearance is not
    # evidence that a repeated input is needed. Preserve limits across restarts.
    history=output/'actions.jsonl'
    if history.exists():
        moments=[datetime.fromisoformat(row['time']).timestamp()
                 for line in history.read_text().splitlines()
                 if (row:=json.loads(line)).get('action')==action_name]
        current=datetime.now(timezone.utc).timestamp()
        cooldown=180 if kind=='free_video_gold' else 30 if kind.startswith('server_retry') else 20
        if moments and current-max(moments)<cooldown: return False
        if sum(current-t<300 for t in moments)>=3:
            raise RuntimeError('同一提示在五分钟内已处理三次，停止并保留证据，避免反复点击。')
    before = capture.locate()
    if not targeted and not frontmost(before, detector.target_point(before,kind)):
        if not activate_before_click:
            return False
        # Prove the exact known screen before bringing anything to the front.
        # Raising a window never substitutes for a fresh foreground check.
        probe, probe_id = capture.capture(output)
        observed = capture.locate()
        if (not same_window(before, observed, probe_id)
                or not detector.inspect(probe).get(kind, False)):
            return False
        if (not activate_game_window(observed)
                or not frontmost(observed, detector.target_point(observed, kind))):
            return False
        before = observed
    # Re-capture after any activation; never click from the earlier probe.
    data, window_id = capture.capture(output)
    state = detector.inspect(data)
    after = capture.locate()
    if (not same_window(before, after, window_id)
            or not state.get(kind,False)
            or (not targeted and not frontmost(after, detector.target_point(after,kind)))):
        return False
    q = capture.quartz
    # The pointer in the upper-left corner is an additional emergency stop.
    pointer = q.CGEventGetLocation(q.CGEventCreate(None))
    if pointer.x <= 3 and pointer.y <= 3:
        raise RuntimeError("鼠标位于左上角，已触发紧急停止。")
    point = detector.target_point(after,kind)
    proof = output / f"before_next_{time.time_ns()}.png"
    proof.write_bytes(data)
    down = q.CGEventCreateMouseEvent(None, q.kCGEventLeftMouseDown, point, q.kCGMouseButtonLeft)
    up = q.CGEventCreateMouseEvent(None, q.kCGEventLeftMouseUp, point, q.kCGMouseButtonLeft)
    if down is None or up is None:
        raise RuntimeError("无法建立鼠标事件，未发送点击。")
    if targeted:
        # Public Core Graphics per-process routing. Never inject a global click
        # into the foreground app when the mini-program is behind another app.
        for index, event in enumerate((down, up)):
            q.CGEventSetIntegerValueField(event, q.kCGMouseEventWindowUnderMousePointer, window_id)
            q.CGEventSetIntegerValueField(event, q.kCGMouseEventWindowUnderMousePointerThatCanHandleThisEvent, window_id)
            q.CGEventSetIntegerValueField(event, q.kCGMouseEventClickState, 1)
            q.CGEventPostToPid(int(after["kCGWindowOwnerPID"]), event)
            if index == 0:
                time.sleep(.15)
    else:
        # Proof-file I/O and event construction can take time. Check again at
        # the final boundary so a new foreground app never receives this click.
        current = capture.locate()
        if not same_window(after, current, window_id) or not frontmost(current, point):
            return False
        q.CGEventPost(q.kCGHIDEventTap, down)
        time.sleep(.15)
        q.CGEventPost(q.kCGHIDEventTap, up)
    with (output / "actions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(time=now(), action=action_name,
                                     window_id=window_id, proof=proof.name,
                                     routing="targeted_process" if targeted else "foreground", mouse_hold_ms=150), ensure_ascii=False) + "\n")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="普通场续局及已核对的等级确认、免费金蛋单倍领取；不购买、不进入赛事。")
    p.add_argument("--dry-run", action="store_true", help="只识别记录，完全不点击")
    p.add_argument("--once", action="store_true", help="仅识别一次，完全不点击")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--max-minutes", type=positive_float, default=60)
    p.add_argument("--interval", type=positive_float, default=2)
    p.add_argument("--activate-on-start", action="store_true", help="启动时将已确认的小程序进程置前一次")
    p.add_argument("--activate-before-click", action="store_true", help="仅在重新确认已知结算/提示后置前游戏；仍重新截图并验证前台和无遮挡，可能切换当前前台应用")
    p.add_argument("--targeted-input", action="store_true", help="只向小程序进程及窗口发送事件，不向其他前台应用发送点击")
    p.add_argument("--capture-run-dir", type=Path, help="关联采集批次；采集器停止后同步结束续局")
    p.add_argument("--output", type=Path, default=ROOT / "reports/auto_next")
    args = p.parse_args()
    if args.activate_before_click and args.targeted_input:
        p.error("--activate-before-click 与 --targeted-input 不可同时使用")
    if args.max_rounds < 1 or args.interval < 1:
        p.error("max-rounds 至少 1；interval 至少 1 秒")
    args.output.mkdir(parents=True, exist_ok=True)
    input_lock = None
    try:
        if not (args.dry_run or args.once):
            import fcntl
            lock_path = ROOT / "reports/.auto_next.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            input_lock = lock_path.open("a")
            try:
                fcntl.flock(input_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("已有一个续局脚本在运行，拒绝重复启动。") from exc
        capture = MacWindowCapture("天天爱掼蛋", None)
        detector, machine = SettlementDetector(), ContinueState()
        if not (args.dry_run or args.once) and not accessibility_allowed():
            raise RuntimeError("尚无辅助功能权限。请手动授权启动脚本的应用；脚本不会自行修改权限。")
        if args.activate_on_start and not (args.dry_run or args.once):
            if not activate_game_window(capture.locate()):
                raise RuntimeError("不能唯一激活已确认的小程序窗口，未发送点击。")
        print("仅用于已进入的普通金蛋场。请先开启游戏自带托管，再保持小程序在前台。", flush=True)
        started, last_known, attempts = time.monotonic(), time.monotonic(), 0
        level_stable, level_pending, popup_name = 0, None, None
        missing_since = None
        loss_offer_pending_since = None
        while time.monotonic() - started < args.max_minutes * 60:
            if not (args.dry_run or args.once):
                pointer = capture.quartz.CGEventGetLocation(capture.quartz.CGEventCreate(None))
                if pointer.x <= 3 and pointer.y <= 3:
                    raise RuntimeError("鼠标移至屏幕左上角，已触发紧急停止。")
            if args.capture_run_dir:
                run_file = args.capture_run_dir / "run.json"
                if run_file.exists():
                    run = json.loads(run_file.read_text(encoding="utf-8"))
                    if run.get("status") != "running":
                        print(f"采集器已停止（{run.get('status')}），结束续局。")
                        return 0
                elif time.monotonic() - started > 15:
                    raise RuntimeError("关联的采集器未启动。")
            try:
                data, window_id = capture.capture(args.output)
            except LookupError as exc:
                current = time.monotonic()
                if args.once:
                    print(f"目标窗口不可用，停止：{exc}")
                    return 2
                if missing_since is None:
                    missing_since = current
                    print(f"目标窗口暂不可用，被动等待最多30秒，不发送输入：{exc}", flush=True)
                # A gap breaks consecutive observations, but must not reset a
                # pending click or extend its original45/120-second deadline.
                machine.stable_settlement = machine.game_observations = 0
                level_stable = 0
                if current - missing_since >= 30:
                    print(f"目标窗口连续30秒不可用，停止：{exc}", flush=True)
                    return 2
                time.sleep(min(args.interval, 30 - (current - missing_since)))
                continue
            if capture.window_id is None:
                capture.window_id = window_id
            if missing_since is not None:
                print("同一目标窗口已恢复，重新核验当前界面。", flush=True)
                missing_since = None
            state = detector.inspect(data)
            atomic_json(args.output / "latest_state.json", dict(time=now(), window_id=window_id, **state))
            if args.once:
                (args.output / "inspection.png").write_bytes(data)
                print(json.dumps(state, ensure_ascii=False, indent=2))
                return 0
            if state["gameplay"] or state["settlement"] or state.get('waiting_ready') or any(state.get(k) for k in detector.popup_actions):
                last_known = time.monotonic()
            elif time.monotonic() - last_known > 60:
                (args.output/'unrecognized_stop.png').write_bytes(data)
                raise RuntimeError("超过 60 秒未识别到普通对局或其结算；可能是弹窗、金蛋不足、断线或布局改变，已停止。")
            popup=next((k for k in detector.popup_actions if state.get(k)),None)
            if popup:
                if popup_name != popup:
                    level_stable, level_pending, popup_name = 0, None, popup
                if popup == 'close_loss_offer' and loss_offer_pending_since is not None:
                    level_pending = loss_offer_pending_since
                level_stable += 1
                # A known timed paid offer can ignore its close button until
                # expiry. Wait passively after ONE close; keep the original
                # timestamp even if an animation briefly breaks the match.
                wait_limit = 300 if popup == 'close_loss_offer' and machine.pending_since is None else 20
                if machine.pending_since is not None:
                    deadline = 120 if machine.known_ready_wait_observed else 45
                    if time.monotonic() - machine.pending_since >= deadline:
                        raise RuntimeError(f"续局后 {deadline} 秒未观察到新对局，停止；不会再次盲点。")
                if level_pending is not None:
                    if time.monotonic()-level_pending >= wait_limit:
                        raise RuntimeError(f'已知提示关闭后{wait_limit}秒仍未消失，停止；不会重复点击。')
                elif level_stable >= 2:
                    if args.dry_run:
                        if level_stable == 2: print(f'识别到已知提示 {popup}；试运行不点击。',flush=True)
                    elif click_verified(capture,detector,args.output,targeted=args.targeted_input,kind=popup,activate_before_click=args.activate_before_click):
                        level_pending=time.monotonic()
                        if popup == 'close_loss_offer':
                            loss_offer_pending_since = level_pending
                        print(f'已处理已知提示 {popup}，被动等待最多{wait_limit}秒，不重复点击。',flush=True)
                time.sleep(args.interval)
                continue
            level_stable, level_pending, popup_name = 0, None, None
            action = machine.observe(state, time.monotonic())
            if action == "transition_failed_stop":
                deadline = 120 if machine.known_ready_wait_observed else 45
                raise RuntimeError(f"续局后 {deadline} 秒未观察到新对局，停止；不会再次盲点。")
            if action == "new_game_observed":
                loss_offer_pending_since = None
                with (args.output / "transitions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(time=now(), event=action,
                        completed_transitions=machine.completed_transitions)) + "\n")
                print(f"已观察到新对局：{machine.completed_transitions}。", flush=True)
            if action == "ready_to_continue":
                if args.dry_run:
                    print("已识别普通场结算与下一局按钮；试运行不点击。", flush=True)
                elif attempts >= args.max_rounds:
                    print("达到续局次数上限，停止。")
                    return 0
                elif click_verified(capture, detector, args.output, targeted=args.targeted_input,activate_before_click=args.activate_before_click):
                    attempts += 1
                    machine.clicked(time.monotonic())
                    print(f"已点普通场「再来一局」({attempts}/{args.max_rounds})，等待实际新局确认。", flush=True)
            time.sleep(args.interval)
        print("达到运行时限，停止。")
        return 0
    except KeyboardInterrupt:
        print("已手动停止。")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"已停止：{exc}", file=sys.stderr)
        return 2
    finally:
        if input_lock is not None:
            input_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
