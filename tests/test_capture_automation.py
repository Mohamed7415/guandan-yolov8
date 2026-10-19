"""Offline regression checks. No mouse input, live windows, or training data writes."""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from auto_capture import choose_window, frame_features, card_change, write_review_queue, load_run, looks_like_green_table
from auto_next import SettlementDetector, ContinueState, window_point_clear


def observed_source(path):
    """Use local capture evidence or the checked-in regression fixture."""
    path = Path(path)
    if path.exists():
        return path
    name = path.parent.name + '_inspection.png' if path.name == 'inspection.png' else path.name
    return ROOT / 'tests/fixtures' / name


def png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class CaptureChecks(unittest.TestCase):
    def test_capture_green_table_gate(self):
        self.assertFalse(looks_like_green_table(png(Image.new("RGB", (810, 500), (120, 140, 225)))))
        self.assertTrue(looks_like_green_table(png(Image.new("RGB", (810, 500), (20, 130, 100)))))
    def test_never_selects_another_app_with_same_title(self):
        window = dict(kCGWindowOwnerName="Safari", kCGWindowName="天天爱掼蛋",
                      kCGWindowLayer=0, kCGWindowNumber=1, kCGWindowBounds={"Width": 810, "Height": 500})
        with self.assertRaises(LookupError):
            choose_window([window], "天天爱掼蛋")
        window["kCGWindowOwnerName"] = "微信"
        self.assertEqual(choose_window([window], "天天爱掼蛋")["kCGWindowNumber"], 1)
        with self.assertRaises(LookupError):
            choose_window([window, dict(window, kCGWindowNumber=2)], "天天爱掼蛋")

    def test_title_hud_animation_does_not_become_new_card_layout(self):
        image = Image.new("RGB", (810, 500), (20, 130, 100))
        draw = ImageDraw.Draw(image)
        draw.rectangle((240, 250, 400, 420), fill="white")
        other = image.copy()
        ImageDraw.Draw(other).rectangle((0, 0, 200, 70), fill="red")
        a, am, *_ = frame_features(png(image))
        b, bm, *_ = frame_features(png(other))
        self.assertEqual(card_change((a, am), (b, bm)), (0, 0))
        ImageDraw.Draw(other).rectangle((410, 250, 500, 420), fill="white")
        b, bm, *_ = frame_features(png(other))
        self.assertGreater(card_change((a, am), (b, bm))[0], .004)

    def test_review_queue_preserves_review_without_auto_approving_new_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = [dict(filename="one.png", sequence_id="same_round")]
            write_review_queue(directory, rows)
            with (directory / "review.csv").open(encoding="utf-8-sig") as f:
                review = list(csv.DictReader(f))
            review[0].update(approved="true", scene="sorted")
            with (directory / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(review[0])); w.writeheader(); w.writerows(review)
            write_review_queue(directory, rows + [dict(filename="two.png", sequence_id="same_round")])
            with (directory / "review.csv").open(encoding="utf-8-sig") as f:
                result = list(csv.DictReader(f))
            self.assertEqual(result[0]["approved"], "true")
            self.assertEqual(result[1]["approved"], "false")
            self.assertEqual(result[1]["scene"], "")
            self.assertEqual(result[1]["session_id"], "same_round")

    def test_resume_refuses_orphan_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "orphan.png").write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "未登记"):
                load_run(directory)
            self.assertEqual((directory / "orphan.png").read_bytes(), b"original")


class ContinueChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = SettlementDetector()
        source = observed_source(ROOT / "data/raw/game002_settlement0001.jpg")
        cls.real = Image.open(source).convert("RGB") if source.exists() else None

    def test_real_settlement_detected_across_retina_scaling(self):
        if self.real is None: self.skipTest("Local genuine screenshot unavailable")
        self.assertTrue(self.detector.inspect(png(self.real))["settlement"])
        resized = self.real.resize((3240, 2000))
        self.assertTrue(self.detector.inspect(png(resized))["settlement"])

    def test_dark_overlay_blocks_correlation_false_positive(self):
        if self.real is None: self.skipTest("Local genuine screenshot unavailable")
        dimmed = ImageEnhance.Brightness(self.real).enhance(.5)
        state = self.detector.inspect(png(dimmed))
        self.assertFalse(state["settlement"])
        self.assertTrue(any(e > 20 for e in state["color_errors"].values()))

    def test_observed_winning_settlement_header_variant(self):
        if self.real is None: self.skipTest("Local genuine screenshot unavailable")
        variants = self.detector.variants["settlement_label"]
        if not variants: self.skipTest("Winning variant not installed")
        image = self.real.copy()
        image.paste(Image.fromarray(variants[0]), (522, 279))
        self.assertTrue(self.detector.inspect(png(image))["settlement"])

    def test_yellow_next_button_alone_is_insufficient(self):
        image = Image.new("RGB", (1620, 1000), (130, 150, 220))
        region = self.detector.layout["regions"]["next_round"]
        image.paste(Image.fromarray(self.detector.templates["next_round"]), region[:2])
        self.assertFalse(self.detector.inspect(png(image))["settlement"])
        self.assertFalse(self.detector.inspect(png(image.resize((1000, 1000))))["settlement"])

    def test_no_repeated_click_without_actual_new_game(self):
        machine = ContinueState()
        settle = dict(settlement=True, gameplay=False)
        game = dict(settlement=False, gameplay=True)
        unknown = dict(settlement=False, gameplay=False)
        self.assertEqual(machine.observe(settle, 0), "observe")
        self.assertEqual(machine.observe(settle, 2), "ready_to_continue")
        machine.clicked(2)
        self.assertEqual(machine.observe(settle, 4), "awaiting_game")
        self.assertEqual(machine.observe(unknown, 8), "awaiting_game")
        self.assertEqual(machine.observe(game, 10), "awaiting_game")
        self.assertEqual(machine.observe(game, 12), "awaiting_game")
        self.assertEqual(machine.observe(game, 14), "new_game_observed")
        self.assertEqual(machine.completed_transitions, 1)
        self.assertEqual(machine.observe(settle, 20), "observe")
        self.assertEqual(machine.observe(settle, 22), "ready_to_continue")

    def test_failed_transition_stops_without_retry(self):
        machine = ContinueState()
        machine.clicked(5)
        self.assertEqual(machine.observe(dict(settlement=True, gameplay=False), 50), "transition_failed_stop")
        self.assertEqual(machine.completed_transitions, 0)

    def test_other_wechat_window_in_front_blocks_input(self):
        bounds = dict(X=0, Y=0, Width=810, Height=500)
        target = dict(kCGWindowNumber=7, kCGWindowLayer=0, kCGWindowBounds=bounds)
        chat = dict(kCGWindowNumber=8, kCGWindowLayer=0, kCGWindowBounds=bounds)
        self.assertFalse(window_point_clear([chat, target], 7, (350, 450)))
        self.assertTrue(window_point_clear([target, chat], 7, (350, 450)))

    def test_floating_overlay_over_button_blocks_input(self):
        target = dict(kCGWindowNumber=7, kCGWindowLayer=0,
                      kCGWindowBounds=dict(X=0, Y=0, Width=810, Height=500))
        popup = dict(kCGWindowNumber=9, kCGWindowLayer=3,
                     kCGWindowBounds=dict(X=300, Y=400, Width=150, Height=80))
        self.assertFalse(window_point_clear([popup, target], 7, (350, 450)))
        self.assertTrue(window_point_clear([popup, target], 7, (100, 450)))

    def test_persistent_full_display_compositor_is_not_an_app_panel(self):
        target=dict(kCGWindowNumber=7,kCGWindowLayer=0,kCGWindowBounds=dict(X=0,Y=0,Width=810,Height=500))
        dock=dict(kCGWindowNumber=18,kCGWindowOwnerName='程序坞',kCGWindowLayer=20,
                  kCGWindowBounds=dict(X=0,Y=0,Width=1512,Height=982))
        self.assertTrue(window_point_clear([dock,target],7,(350,450)))

    def test_visual_mouse_cursors_do_not_occlude_input(self):
        target=dict(kCGWindowNumber=7,kCGWindowLayer=0,kCGWindowBounds=dict(X=0,Y=0,Width=810,Height=500))
        cursors=[dict(kCGWindowNumber=8,kCGWindowOwnerName='Window Server',kCGWindowName='Cursor',
                      kCGWindowLayer=2147483630,kCGWindowBounds=dict(X=350,Y=450,Width=32,Height=32)),
                 dict(kCGWindowNumber=9,kCGWindowOwnerName='ChatGPT Computer Use',kCGWindowName='Software Cursor',
                      kCGWindowLayer=102,kCGWindowBounds=dict(X=300,Y=400,Width=126,Height=126))]
        self.assertTrue(window_point_clear(cursors+[target],7,(350,450)))
        cursors[1]['kCGWindowName']='Unrecognized panel'
        self.assertFalse(window_point_clear(cursors+[target],7,(350,450)))

    def test_known_level_popup_requires_all_three_anchors(self):
        p=observed_source(ROOT/'reports/auto_next/popup_levelup6/inspection.png')
        if not p.exists(): self.skipTest('Local observed popup unavailable')
        original=Image.open(p).convert('RGB')
        state=self.detector.inspect(png(original))
        self.assertTrue(state['level_up']);self.assertFalse(state['settlement'])
        original.paste((0,0,0),(505,813,792,909))
        self.assertFalse(self.detector.inspect(png(original))['level_up'])

    def test_free_gold_claim_never_matches_a_dimmed_or_changed_button(self):
        p=observed_source(ROOT/'reports/auto_next/popup_free_gold/inspection.png')
        if not p.exists(): self.skipTest('Local observed popup unavailable')
        original=Image.open(p).convert('RGB')
        self.assertTrue(self.detector.inspect(png(original))['free_gold_base'])
        original.paste((0,0,0),(490,753,783,852))
        self.assertFalse(self.detector.inspect(png(original))['free_gold_base'])

    def test_observed_connection_and_offer_require_three_anchors(self):
        for kind,spec in self.detector.extra_actions.items():
            source=observed_source(ROOT/spec['source'])
            if not source.exists(): self.skipTest('Observed popup source unavailable')
            image=Image.open(source).convert('RGB')
            state=self.detector.inspect(png(image))
            self.assertTrue(state[kind],kind)
            self.assertFalse(state['settlement'])
            self.assertFalse(state['gameplay'])
            button=next(spec['regions'][key] for key in ('close','retry','claim','confirm') if key in spec['regions'])
            image.paste((0,0,0),button)
            self.assertFalse(self.detector.inspect(png(image))[kind],kind)

    def test_r_timeout_retry_requires_every_native_anchor(self):
        source=ROOT/'tests/fixtures/server_retry_r_inspection.png'
        original=Image.open(source).convert('RGB')
        self.assertTrue(self.detector.inspect(png(original))['server_retry_r'])
        spec=self.detector.extra_actions['server_retry_r']
        for name,box in spec['regions'].items():
            with self.subTest(anchor=name):
                changed=original.copy();changed.paste((0,0,0),box)
                self.assertFalse(self.detector.inspect(png(changed))['server_retry_r'])
        dimmed=ImageEnhance.Brightness(original).enhance(.5)
        self.assertFalse(self.detector.inspect(png(dimmed))['server_retry_r'])


if __name__ == "__main__":
    unittest.main()
