"""Exercise controller recovery without a desktop or mouse events."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import io
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import auto_next


class WindowGapChecks(unittest.TestCase):
    def run_controller(self, observations, machine=None, once=False):
        clock = SimpleNamespace(value=0.)
        capture = MagicMock()
        capture.window_id = None
        remaining = iter(observations)
        def frame(_):
            item = next(remaining, LookupError('missing'))
            if isinstance(item, Exception):
                raise item
            return item, 71
        capture.capture.side_effect = frame
        capture.quartz.CGEventGetLocation.return_value = SimpleNamespace(x=100, y=100)
        detector = MagicMock()
        detector.popup_actions = []
        detector.inspect.side_effect = lambda data: dict(
            settlement=data == b'settlement', gameplay=data == b'hand')
        machine = machine or auto_next.ContinueState()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            argv = ['auto_next', '--output', folder, '--interval', '2', '--max-minutes', '2']
            if once:
                argv += ['--once']
            with patch.object(sys, 'argv', argv), \
                 patch.object(auto_next, 'MacWindowCapture', return_value=capture), \
                 patch.object(auto_next, 'SettlementDetector', return_value=detector), \
                 patch.object(auto_next, 'ContinueState', return_value=machine), \
                 patch.object(auto_next, 'accessibility_allowed', return_value=True), \
                 patch.object(auto_next, 'click_verified', return_value=False) as click, \
                 patch.object(auto_next.time, 'monotonic', side_effect=lambda: clock.value), \
                 patch.object(auto_next.time, 'sleep', side_effect=lambda dt: setattr(clock, 'value', clock.value + dt)), \
                 redirect_stdout(output), redirect_stderr(output):
                result = auto_next.main()
            return result, clock.value, capture, detector, click.call_count, output.getvalue()

    def test_transient_loss_requires_two_fresh_settlements_before_click(self):
        result, _, capture, detector, clicks, output = self.run_controller(
            [b'settlement', LookupError('missing'), b'settlement', b'settlement'])
        self.assertEqual(result, 2)  # Later permanent loss still stops.
        self.assertEqual(clicks, 1)
        self.assertEqual(detector.inspect.call_count, 3)
        self.assertEqual(capture.window_id, 71)
        self.assertIn('同一目标窗口已恢复', output)

    def test_permanent_loss_stops_at_fixed_deadline_without_inspection_or_click(self):
        result, elapsed, _, detector, clicks, _ = self.run_controller([])
        self.assertEqual((result, elapsed, clicks), (2, 30., 0))
        detector.inspect.assert_not_called()

    def test_gap_breaks_hand_streak_and_keeps_pending_click(self):
        machine = auto_next.ContinueState()
        machine.clicked(0)
        self.run_controller([b'hand', b'hand', LookupError('missing'), b'hand', b'hand'], machine)
        self.assertEqual(machine.completed_transitions, 0)
        self.assertEqual(machine.pending_since, 0)

    def test_gap_does_not_extend_original_transition_deadline(self):
        machine = auto_next.ContinueState()
        machine.clicked(-44)
        result, _, _, _, clicks, output = self.run_controller([LookupError('missing'), b'hand'], machine)
        self.assertEqual((result, clicks, machine.completed_transitions), (2, 0, 0))
        self.assertIn('45 秒未观察到新对局', output)

    def test_once_remains_single_observation_without_retry(self):
        result, elapsed, capture, _, clicks, _ = self.run_controller([], once=True)
        self.assertEqual((result, elapsed, clicks), (2, 0., 0))
        self.assertEqual(capture.capture.call_count, 1)


if __name__ == '__main__':
    unittest.main()
