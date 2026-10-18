"""Recover interrupted commits without accepting unregistered images/drafts."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import auto_capture as capture
import watch_proposals as proposals


def frame():
    image = Image.new("RGB", (810, 500), (20, 130, 100))
    ImageDraw.Draw(image).rectangle((300, 270, 450, 440), fill="white")
    stream = io.BytesIO(); image.save(stream, format="PNG"); data = stream.getvalue()
    row = dict(filename="native_00001.png", captured_at="2026-10-19T00:00:00+00:00",
               sha256=hashlib.sha256(data).hexdigest(), width=810, height=500,
               window_id=1, sequence_id="native_run", source="current_wechat_miniprogram_window",
               scene="", reviewed="false")
    return data, row


class CaptureRecoveryTests(unittest.TestCase):
    def test_crash_after_png_before_manifest_is_recovered_once(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); data, row = frame()
            with patch.object(capture, "write_capture_manifest", side_effect=OSError("power loss")):
                with self.assertRaises(OSError):
                    capture.commit_frame(directory, data, row, [])
            self.assertEqual((directory / row["filename"]).read_bytes(), data)
            rows, _, _ = capture.load_run(directory)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reviewed"], "false")
            self.assertEqual(len(capture.load_run(directory)[0]), 1)

    def test_crash_after_manifest_before_journal_cleanup_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); data, row = frame()
            capture.commit_frame(directory, data, row, [])
            capture.atomic_json(directory / ".pending_frame.json", {"row": row})
            self.assertEqual(len(capture.load_run(directory)[0]), 1)

    def test_changed_orphan_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); data, row = frame()
            capture.atomic_json(directory / ".pending_frame.json", {"row": row})
            (directory / row["filename"]).write_bytes(data + b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                capture.load_run(directory)
            self.assertTrue((directory / row["filename"]).exists())

    def test_uncommitted_png_is_not_queued_for_proposals(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); data, row = frame()
            (directory / row["filename"]).write_bytes(data)
            self.assertEqual(proposals.pending_images(directory, directory / "proposals"), [])

    def test_watcher_drains_frame_committed_during_terminal_check(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); path = directory / "last.png"
            (directory / "run.json").write_text(json.dumps({"status": "candidate_target_reached"}))
            with patch.object(proposals, "pending_images", side_effect=[[], [path], [path], []]), patch.object(proposals.subprocess, "run") as run:
                proposals.watch(directory, directory / "proposals")
            self.assertEqual(run.call_count, 1)
            self.assertIn(str(path), run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
