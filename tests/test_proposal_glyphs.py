"""Regression checks against genuine local card glyphs; no draft approval."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from propose_card_annotations import bank_from_reviewed,propose

class GlyphChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.bank=bank_from_reviewed()
 def cards(self,index):
  source=ROOT/f'tests/fixtures/glyph_{index}.png'
  if not source.exists():
   source=ROOT/f'data/candidates/live_20261017_02/capture_20261017_182041_fefea2_{index:05d}.png'
  if not source.exists():self.skipTest('Genuine local regression screenshot unavailable')
  return propose(source,self.bank)[1]
 def test_border_beside_j_does_not_hide_spade_jack(self):
  cards=self.cards(383)
  self.assertTrue(any(c['label']=='J♠' and 600<c['x']<640 and 590<c['y']<630 for c in cards))
 def test_real_ten_still_merges(self):
  cards=self.cards(383)
  self.assertTrue(any(c['label']=='10♠' and c['own'] for c in cards))
 def test_border_beside_q_preserves_both_club_queens(self):
  cards=self.cards(708)
  self.assertEqual(sum(c['label']=='Q♣' and c['own'] for c in cards),2)
 def test_all_27_starting_cards_survive_border_filter(self):
  cards=self.cards(2)
  self.assertEqual(len(cards),27)

if __name__=='__main__':unittest.main()
