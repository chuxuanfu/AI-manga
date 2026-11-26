import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from brief_check import confirm_contact_gap

class ContactGapTests(unittest.TestCase):
    def test_requires_visible_separation_at_the_same_two_points(self):
        im=Image.new('RGB',(100,100))
        first=dict(relation='separated',both_visible=True,nearest_relevant_target=True,
                   actor_edge=[220,450],target_edge=[358,574],intervening_surface='turquoise desk paint')
        for second,expected in [(first,True),({**first,'relation':'occluded'},False),
                                ({**first,'target_edge':[780,840]},False),({**first,'nearest_relevant_target':False},False)]:
            calls=[json.dumps(first),json.dumps(second)]
            with patch('brief_check.ask',side_effect=calls):
                answer=confirm_contact_gap(None,im,im,'turns the knob','first','second')
            self.assertEqual(answer['confirmed'],expected)
        with patch('brief_check.ask',side_effect=AssertionError('No physical contact specified')):
            self.assertIsNone(confirm_contact_gap(None,im,im,'watches a butterfly','first','second'))
