#!/usr/bin/env python3
"""Keep the goal's progress durable and reopen its desktop host after login.

This service never invokes a model, spends credits, starts training, or clicks
the game. The app's thread heartbeat owns subsequent goal work.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import plistlib
import subprocess
import time
from datetime import datetime, timezone

from atomic_io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "reports" / "continuous_goal"
APP = Path("/Applications/ChatGPT.app")
SERVICE = "com.guandan.yolov8.goal-recovery"
TERMINAL_STAGES = {"completed", "stopped"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def locked(directory: Path):
    """The kernel releases this lock on exit or power loss; no stale PID lock."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "recovery.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_state(directory: Path) -> dict:
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("enabled"), bool):
        raise ValueError("Goal state must explicitly contain boolean enabled")
    return state


def record_progress(directory: Path, changes: dict, event: str) -> dict:
    """Serialize updates with a previous-state copy and a durable event log."""
    with locked(directory):
        path = directory / "state.json"
        old = read_state(directory) if path.exists() else {}
        state = {**old, **changes, "updated_at": now()}
        if not isinstance(state.get("enabled"), bool):
            raise ValueError("Specify enabled when creating goal state")
        if old:
            atomic_json(directory / "state.previous.json", old)
        atomic_json(path, state)
        with (directory / "journal.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"at": now(), "event": event, "state": state}, ensure_ascii=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
        return state


def app_running(app: Path) -> bool:
    with (app / "Contents" / "Info.plist").open("rb") as stream:
        executable = plistlib.load(stream)["CFBundleExecutable"]
    main = str(app / "Contents" / "MacOS" / executable)
    processes = subprocess.run(["/bin/ps", "-axo", "args="], check=True,
                               capture_output=True, text=True).stdout.splitlines()
    return any(line.strip() == main or line.strip().startswith(main + " ")
               for line in processes)


def tick(directory: Path, app: Path = APP) -> dict:
    with locked(directory):
        # Missing or corrupt progress fails closed; it must not reopen an app
        # after the user has removed/disabled the goal's recovery state.
        state = read_state(directory)
        if not state["enabled"] or state.get("stage") in TERMINAL_STAGES:
            return {"at": now(), "action": "disabled"}
        if not app.is_dir():
            raise FileNotFoundError(f"Desktop application is unavailable: {app}")
        if app_running(app):
            result = {"at": now(), "action": "already_running"}
        else:
            subprocess.run(["/usr/bin/open", "-g", "-j", str(app)], check=True, timeout=30)
            result = {"at": now(), "action": "launch_requested"}
        atomic_json(directory / "recovery_health.json", result)
        return result


def supervise(directory: Path, app: Path = APP) -> None:
    """Hold the user's scoped sleep assertions while goal recovery is enabled."""
    state = read_state(directory)
    if not state["enabled"] or state.get("stage") in TERMINAL_STAGES:
        return
    # Display/battery idle-sleep prevention is opt-in. Watching this parent PID
    # releases assertions even if launchd terminates the supervisor abruptly.
    flags = ["-d", "-i", "-s"] if state.get("recovery", {}).get("keep_display_and_system_awake") is True else ["-s"]
    awake = subprocess.Popen(["/usr/bin/caffeinate", *flags, "-w", str(os.getpid())])
    try:
        while True:
            if tick(directory, app)["action"] == "disabled":
                return
            if awake.poll() is not None:
                raise RuntimeError("The scoped sleep assertion exited")
            time.sleep(300)
    finally:
        awake.terminate()
        try:
            awake.wait(timeout=5)
        except subprocess.TimeoutExpired:
            awake.kill()
            awake.wait(timeout=5)


def agent_spec(python: Path, script: Path, directory: Path) -> dict:
    return {
        "Label": SERVICE,
        "ProgramArguments": [str(python), str(script), "--state-dir", str(directory), "supervise"],
        "RunAtLoad": True,
        "StartInterval": 300,
        "LimitLoadToSessionType": "Aqua",
        "ProcessType": "Background",
        "StandardOutPath": str(directory / "recovery.stdout.log"),
        "StandardErrorPath": str(directory / "recovery.stderr.log"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("tick")
    commands.add_parser("supervise")
    update = commands.add_parser("update")
    update.add_argument("--changes", required=True, help="JSON object merged into existing progress")
    update.add_argument("--event", required=True)
    args = parser.parse_args()
    try:
        if args.command == "supervise":
            supervise(args.state_dir)
            return
        if args.command == "tick":
            result = tick(args.state_dir)
        elif args.command == "status":
            result = read_state(args.state_dir)
        else:
            changes = json.loads(args.changes)
            if not isinstance(changes, dict):
                raise ValueError("changes must be an object")
            result = record_progress(args.state_dir, changes, args.event)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        return
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Recovery stopped: {error}\n")


if __name__ == "__main__":
    main()
