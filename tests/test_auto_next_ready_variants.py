"""Offline UI guards; these constructed inputs are never dataset candidates."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from auto_next import ContinueState, SettlementDetector


def png(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


class ReadyVariantChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = SettlementDetector()

    def ready_screen(self, regions, anchors, background=(20, 130, 100)):
        image = Image.new('RGB', (1620, 1000), background)
        # Neutral surfaces reproduce the old hand-white false positive without
        # including any actual card, rank, suit, avatar or private screenshot.
        image.paste((210, 210, 210), (650, 700, 1050, 810))
        for name, box in regions.items():
            tile = Image.fromarray(anchors[name])
            image.paste(tile, box[:2], tile.getchannel('A'))
        return image

    def test_native_variants_block_leftover_hand_and_remain_bounded(self):
        self.assertEqual(len(self.detector.wait_variants), 3)
        for regions, anchors in self.detector.wait_variants:
            with self.subTest(regions=regions):
                state = self.detector.inspect(png(self.ready_screen(regions, anchors)))
                self.assertGreater(state['hand_white_fraction'], .025)
                self.assertTrue(state['waiting_ready'])
                self.assertFalse(state['gameplay'])
                machine = ContinueState()
                machine.clicked(0)
                for t in (2, 46, 90, 119):
                    self.assertEqual(machine.observe(state, t), 'awaiting_game')
                self.assertEqual(machine.observe(state, 120), 'transition_failed_stop')
                self.assertEqual(machine.completed_transitions, 0)

    def test_every_variant_requires_all_three_intact_anchors(self):
        for regions, anchors in self.detector.wait_variants:
            original = self.ready_screen(regions, anchors)
            for name, box in regions.items():
                with self.subTest(variant=regions, removed=name):
                    damaged = original.copy()
                    damaged.paste((0, 0, 0), box)
                    self.assertFalse(self.detector.ready_variant_matches(
                        np.asarray(damaged), regions, anchors))

    def test_background_pixels_do_not_control_ready_match(self):
        for regions, anchors in self.detector.wait_variants:
            for color in ((0, 0, 0), (255, 255, 255)):
                image = self.ready_screen(regions, anchors, color)
                self.assertTrue(self.detector.ready_variant_matches(
                    np.asarray(image), regions, anchors))

    def test_dimmed_overlay_and_flat_orange_patches_are_rejected(self):
        for regions, anchors in self.detector.wait_variants:
            original = self.ready_screen(regions, anchors)
            dimmed = ImageEnhance.Brightness(original).enhance(.5)
            self.assertFalse(self.detector.ready_variant_matches(
                np.asarray(dimmed), regions, anchors))
            flat = original.copy()
            for box in regions.values():
                flat.paste((220, 140, 30), box)
            self.assertFalse(self.detector.ready_variant_matches(
                np.asarray(flat), regions, anchors))

    def test_ui_assets_exclude_all_non_glyph_pixels(self):
        for _, anchors in self.detector.wait_variants:
            for tile in anchors.values():
                self.assertTrue(np.all(tile[:, :, :3][tile[:, :, 3] == 0] == 0))
                self.assertGreater((tile[:, :, 3] == 255).sum(), 64)
                self.assertLess((tile[:, :, 3] == 255).mean(), .3)

    def test_blank_ready_locations_do_not_block_actual_hand_confirmation(self):
        image = Image.new('RGB', (1620, 1000), (20, 130, 100))
        image.paste((210, 210, 210), (650, 700, 1050, 810))
        state = self.detector.inspect(png(image))
        self.assertFalse(state['waiting_ready'])
        self.assertTrue(state['gameplay'])
        machine = ContinueState()
        machine.clicked(0)
        for t in (2, 4, 6):
            result = machine.observe(state, t)
        self.assertEqual(result, 'new_game_observed')


if __name__ == '__main__':
    unittest.main()
