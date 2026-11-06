"""Full-session mode must keep hard/transient states, not filter by card area."""
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


def test_png(color):
    stream = io.BytesIO()
    image = Image.new("RGB", (810, 500), color)
    ImageDraw.Draw(image).rectangle((100, 100, 600, 400), fill=(180, 70, 40))
    image.save(stream, format="PNG")
    return stream.getvalue()


class FullSessionCaptureTests(unittest.TestCase):
    def test_transient_and_no_cards_saved_exact_duplicates_only_referenced(self):
        # In-memory test fixtures only; never native data or acceptance credit.
        first, second = test_png("black"), test_png("white")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            args = capture.parser().parse_args([
                "--run-dir", str(directory), "--sequence-id", "fixture_only",
                "--target", "2", "--interval", "0.5", "--retain-all-native-frames"])
            with patch.object(capture, "MacWindowCapture") as factory, patch.object(capture.time, "sleep"):
                factory.return_value.capture.side_effect = [(first, 7), (first, 7), (second, 7)]
                self.assertEqual(capture.run(args), 0)
            observations = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in observations], ["saved_unfiltered_native", "exact_duplicate", "saved_unfiltered_native"])
            self.assertEqual(observations[0]["filename"], observations[1]["filename"])
            for row in observations:
                self.assertEqual(hashlib.sha256((directory / row["filename"]).read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(json.loads((directory / "run.json").read_text())["counts"]["unstable"], 0)

    def test_full_session_rejects_filtering_settlement(self):
        args = capture.parser().parse_args(["--retain-all-native-frames", "--gameplay-only"])
        with self.assertRaisesRegex(ValueError, "不能排除"):
            capture.run(args)

    def test_resume_rejects_changed_retention_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "run.json").write_text(json.dumps({"sequence_id": "old", "window_title": capture.WINDOW_TITLE}))
            args = capture.parser().parse_args(["--run-dir", folder, "--retain-all-native-frames"])
            with patch.object(capture, "MacWindowCapture"), self.assertRaisesRegex(ValueError, "保留策略"):
                capture.run(args)


if __name__ == "__main__":
    unittest.main()
