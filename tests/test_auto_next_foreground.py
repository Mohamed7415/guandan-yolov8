"""Foreground recovery must never turn a stale screen into mouse input."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from auto_next import click_verified, now


def fixtures(folder):
    window=dict(kCGWindowNumber=7,kCGWindowOwnerPID=123,kCGWindowBounds=dict(X=20,Y=30,Width=810,Height=500))
    capture=MagicMock();capture.locate.return_value=window
    capture.capture.side_effect=[(b'old-probe',7),(b'fresh-proof',7)]
    capture.quartz.CGEventGetLocation.return_value=SimpleNamespace(x=100,y=100)
    detector=MagicMock();detector.popup_actions=[];detector.extra_actions={}
    detector.target_point.return_value=(350,450);detector.inspect.return_value={'settlement':True}
    return capture,detector,window


class ForegroundChecks(unittest.TestCase):
    def test_default_never_raises_a_background_app(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,detector,_=fixtures(folder)
            with patch('auto_next.accessibility_allowed',return_value=True),patch('auto_next.frontmost',return_value=False),patch('auto_next.activate_game_window') as activate:
                self.assertFalse(click_verified(capture,detector,Path(folder)))
            activate.assert_not_called();capture.capture.assert_not_called();capture.quartz.CGEventPost.assert_not_called()

    def test_recent_action_cooldown_blocks_focus_change_too(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,detector,_=fixtures(folder)
            (Path(folder)/'actions.jsonl').write_text(json.dumps({'time':now(),'action':'ordinary_settlement_next_round'})+'\n')
            with patch('auto_next.accessibility_allowed',return_value=True),patch('auto_next.frontmost',return_value=False),patch('auto_next.activate_game_window') as activate:
                self.assertFalse(click_verified(capture,detector,Path(folder),activate_before_click=True))
            activate.assert_not_called();capture.capture.assert_not_called();capture.quartz.CGEventPost.assert_not_called()

    def test_unknown_screen_cannot_activate_or_click(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,detector,_=fixtures(folder);detector.inspect.return_value={'settlement':False}
            with patch('auto_next.accessibility_allowed',return_value=True),patch('auto_next.frontmost',return_value=False),patch('auto_next.activate_game_window') as activate:
                self.assertFalse(click_verified(capture,detector,Path(folder),activate_before_click=True))
            activate.assert_not_called();capture.quartz.CGEventPost.assert_not_called()

    def test_known_screen_is_recaptured_after_activation_and_before_input(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,detector,_=fixtures(folder);events=[]
            detector.inspect.side_effect=lambda data:events.append(('inspect',data)) or {'settlement':True}
            capture.quartz.CGEventPost.side_effect=lambda *args:events.append(('post',))
            with patch('auto_next.accessibility_allowed',return_value=True),patch('auto_next.frontmost',side_effect=[False,True,True,True]),patch('auto_next.activate_game_window',side_effect=lambda w:events.append(('activate',)) or True),patch('auto_next.time.sleep'):
                self.assertTrue(click_verified(capture,detector,Path(folder),activate_before_click=True))
            self.assertEqual(events,[('inspect',b'old-probe'),('activate',),('inspect',b'fresh-proof'),('post',),('post',)])
            self.assertEqual(next(Path(folder).glob('before_next_*.png')).read_bytes(),b'fresh-proof')
            action=json.loads((Path(folder)/'actions.jsonl').read_text());self.assertEqual(action['routing'],'foreground')

    def test_focus_lost_at_final_boundary_cannot_receive_input(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,detector,_=fixtures(folder)
            with patch('auto_next.accessibility_allowed',return_value=True),patch('auto_next.frontmost',side_effect=[False,True,True,False]),patch('auto_next.activate_game_window',return_value=True):
                self.assertFalse(click_verified(capture,detector,Path(folder),activate_before_click=True))
            capture.quartz.CGEventPost.assert_not_called();self.assertFalse((Path(folder)/'actions.jsonl').exists())

    def test_changed_window_identity_blocks_activation(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,detector,window=fixtures(folder);capture.locate.side_effect=[window,{**window,'kCGWindowOwnerPID':999}]
            with patch('auto_next.accessibility_allowed',return_value=True),patch('auto_next.frontmost',return_value=False),patch('auto_next.activate_game_window') as activate:
                self.assertFalse(click_verified(capture,detector,Path(folder),activate_before_click=True))
            activate.assert_not_called();capture.quartz.CGEventPost.assert_not_called()

if __name__=='__main__':unittest.main()
