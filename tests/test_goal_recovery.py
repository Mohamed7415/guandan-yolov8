"""Recovery checks use temporary state and mocked desktop calls only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import goal_recovery as recovery


class GoalRecoveryTests(unittest.TestCase):
    def test_interrupted_publish_keeps_previous_valid_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            recovery.atomic_json(path, {"enabled": True, "stage": "review"})
            with patch.object(recovery.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    recovery.atomic_json(path, {"enabled": False})
            self.assertEqual(json.loads(path.read_text())["stage"], "review")

    def test_disabled_or_completed_does_not_open_app(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(recovery.subprocess, "run") as run:
            directory = Path(folder)
            for state in [{"enabled": False}, {"enabled": True, "stage": "completed"}]:
                recovery.atomic_json(directory / "state.json", state)
                self.assertEqual(recovery.tick(directory)["action"], "disabled")
            run.assert_not_called()

    def test_missing_or_corrupt_state_does_not_open_app(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(recovery.subprocess, "run") as run:
            directory = Path(folder)
            with self.assertRaises(FileNotFoundError):
                recovery.tick(directory)
            (directory / "state.json").write_text("{")
            with self.assertRaises(ValueError):
                recovery.tick(directory)
            run.assert_not_called()

    def test_running_app_is_not_launched_again(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            recovery.atomic_json(directory / "state.json", {"enabled": True})
            with patch.object(recovery, "app_running", return_value=True), patch.object(recovery.subprocess, "run") as run:
                result = recovery.tick(directory, directory)
            self.assertEqual(result["action"], "already_running")
            run.assert_not_called()

    def test_app_recovery_only_requests_background_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            recovery.atomic_json(directory / "state.json", {"enabled": True})
            with patch.object(recovery, "app_running", return_value=False), patch.object(recovery.subprocess, "run") as run:
                result = recovery.tick(directory, directory)
            self.assertEqual(result["action"], "launch_requested")
            run.assert_called_once_with(["/usr/bin/open", "-g", "-j", str(directory)], check=True, timeout=30)

    def test_progress_update_preserves_prior_evidence_and_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            recovery.record_progress(directory, {"enabled": True, "stage": "review", "baseline": "v1"}, "created")
            recovery.record_progress(directory, {"stage": "train"}, "review_finished")
            self.assertEqual(recovery.read_state(directory)["baseline"], "v1")
            self.assertEqual(json.loads((directory / "state.previous.json").read_text())["stage"], "review")
            entries = [json.loads(line) for line in (directory / "journal.jsonl").read_text().splitlines()]
            self.assertEqual([entry["event"] for entry in entries], ["created", "review_finished"])

    def test_supervisor_releases_sleep_assertion_on_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            recovery.atomic_json(directory / "state.json", {"enabled": True})
            with patch.object(recovery.subprocess, "Popen") as popen, patch.object(recovery, "tick", return_value={"action": "already_running"}), patch.object(recovery.time, "sleep", side_effect=KeyboardInterrupt):
                child = popen.return_value
                child.poll.return_value = None
                with self.assertRaises(KeyboardInterrupt):
                    recovery.supervise(directory)
                popen.assert_called_once_with(["/usr/bin/caffeinate", "-s"])
                child.terminate.assert_called_once()
                child.wait.assert_called_once_with(timeout=5)

    def test_disabled_supervisor_does_not_change_sleep(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            recovery.atomic_json(directory / "state.json", {"enabled": False})
            with patch.object(recovery.subprocess, "Popen") as popen:
                recovery.supervise(directory)
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
