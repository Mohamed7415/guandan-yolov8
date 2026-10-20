"""Background reads retain identity and never route clicks to the movie app."""
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import io
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import auto_capture
import auto_next


def window(**changes):
    return dict(kCGWindowOwnerName='微信', kCGWindowName='天天爱掼蛋',
                kCGWindowNumber=71, kCGWindowOwnerPID=123, kCGWindowLayer=0,
                kCGWindowBounds=dict(X=20,Y=30,Width=810,Height=500), **changes)


class BackgroundChecks(unittest.TestCase):
    def test_explicit_id_required_before_background_access(self):
        with self.assertRaisesRegex(ValueError, 'window-id'):
            auto_capture.MacWindowCapture('天天爱掼蛋', None, allow_background_window=True)

    def test_other_space_window_selected_only_when_explicitly_enabled(self):
        c=object.__new__(auto_capture.MacWindowCapture)
        q=MagicMock(kCGWindowListOptionAll=0,kCGWindowListOptionOnScreenOnly=1,
                    kCGWindowListExcludeDesktopElements=16,kCGNullWindowID=0)
        q.CGWindowListCopyWindowInfo.side_effect=lambda options,_: [window()] if options==16 else []
        c.quartz=q;c.title='天天爱掼蛋';c.window_id=71;c.allow_background_window=False
        with self.assertRaises(LookupError): c.locate()
        c.allow_background_window=True
        self.assertEqual(c.locate()['kCGWindowNumber'],71)
        c.window_id=72
        with self.assertRaises(LookupError): c.locate()

    def test_reused_background_id_in_a_new_process_is_rejected(self):
        c=object.__new__(auto_capture.MacWindowCapture)
        c.quartz=MagicMock(kCGWindowListOptionAll=0,kCGWindowListOptionOnScreenOnly=1,
                          kCGWindowListExcludeDesktopElements=16,kCGNullWindowID=0)
        c.quartz.CGWindowListCopyWindowInfo.side_effect=[[window()],[{**window(),'kCGWindowOwnerPID':456}]]
        c.title='天天爱掼蛋';c.window_id=71;c.allow_background_window=True
        c.locate()
        with self.assertRaises(LookupError):c.locate()

    def test_reused_window_id_or_moved_window_discards_frame(self):
        for changed in [dict(kCGWindowOwnerPID=456),dict(kCGWindowBounds=dict(X=21,Y=30,Width=810,Height=500))]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                c=object.__new__(auto_capture.MacWindowCapture);first=window();c.locate=MagicMock(side_effect=[first,{**first,**changed}])
                def screenshot(args,**kw): Path(args[-1]).write_bytes(b'native-png');return SimpleNamespace(returncode=0)
                with patch.object(auto_capture.subprocess,'run',side_effect=screenshot):
                    with self.assertRaises(LookupError): c.capture(Path(folder))

    def test_background_cli_rejects_global_input_and_activation(self):
        cases=[[],['--window-id','71'],['--window-id','71','--targeted-input','--activate-on-start'],['--window-id','71','--once','--activate-before-click']]
        for extra in cases:
            with self.subTest(extra=extra),patch.object(sys,'argv',['next','--allow-background-window',*extra]),redirect_stderr(io.StringIO()),patch.object(auto_next,'MacWindowCapture') as capture:
                with self.assertRaises(SystemExit): auto_next.main()
                capture.assert_not_called()

    def targeted(self,known=True,changed=False):
        with tempfile.TemporaryDirectory() as folder:
            c=MagicMock();c.locate.side_effect=[window(),{**window(),'kCGWindowOwnerPID':456} if changed else window()]
            c.capture.return_value=(b'fresh-native',71)
            c.quartz.CGEventGetLocation.return_value=SimpleNamespace(x=100,y=100)
            d=MagicMock();d.popup_actions=[];d.extra_actions={};d.target_point.return_value=(350,450);d.inspect.return_value={'settlement':known}
            with patch.object(auto_next,'accessibility_allowed',return_value=True),patch.object(auto_next,'frontmost') as front,patch.object(auto_next,'activate_game_window') as activate,patch.object(auto_next.time,'sleep'):
                sent=auto_next.click_verified(c,d,Path(folder),targeted=True)
            front.assert_not_called();activate.assert_not_called();c.quartz.CGEventPost.assert_not_called()
            return sent,c.quartz.CGEventPostToPid.call_count

    def test_background_targeted_route_never_posts_global_input(self):
        self.assertEqual(self.targeted(),(True,2))

    def test_background_unknown_screen_cannot_receive_input(self):
        self.assertEqual(self.targeted(known=False),(False,0))

    def test_background_identity_change_cannot_receive_input(self):
        self.assertEqual(self.targeted(changed=True),(False,0))
