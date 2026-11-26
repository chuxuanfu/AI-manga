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

    def test_disputed_band_labels_require_separate_visual_readings(self):
        from brief_check import resolve_mixed_bands
        im=Image.new('RGB',(400,650))
        paired=dict(visible=True,item_band=14,landmark_band=14,
                    trace='The paper is at band labeled 14. The upper chest is in band 8.')
        points=[dict(visible=True,band=14,trace='The paper center aligns with 14.'),
                dict(visible=True,band=8,trace='The lapel center aligns with 8.')]
        with patch('brief_check.ask',side_effect=map(json.dumps,points)) as reader:
            result=resolve_mixed_bands(None,im,'reader','paper','upper chest',paired)
        self.assertEqual((result['item_band'],result['landmark_band']),(14,8))
        self.assertNotIn('upper chest',reader.call_args_list[0].args[2])
        inconsistent=dict(visible=True,band=5,trace='Ambiguous point near the top.')
        with patch('brief_check.ask',return_value=json.dumps(inconsistent)):
            self.assertFalse(resolve_mixed_bands(None,im,'reader','paper','chest',paired)['visible'])

    def test_full_scene_contact_binds_the_working_item_to_its_surface(self):
        from brief_check import check_working_contact
        proof={'confirmed':True,'observations':[
            {'actor_edge':[300,500],'target_edge':[680,510],'intervening_surface':'meadow'},
            {'actor_edge':[310,496],'target_edge':[675,505],'intervening_surface':'grass and sky'}]}
        im=Image.new('RGB',(600,400),'green')
        wording='She holds one short brush against a painted canvas.'
        with patch('brief_check.model_identity',return_value='installed'),patch('brief_check.ask',return_value=json.dumps(
                dict(required=True,acting_item='brush',receiving_surface='painted canvas'))),patch('brief_check.confirm_contact_gap',return_value=proof) as reader:
            result=check_working_contact('local',im,wording,'first','second')
        self.assertEqual(reader.call_args.kwargs['endpoints'],['brush','painted canvas'])
        self.assertGreater(result['minimum_gap'],300)
        self.assertEqual(result['box'],[265,461,715,545])
        with patch('brief_check.ask') as reader:
            self.assertIsNone(check_working_contact('local',im,'A landscape in morning light.','first','second'))
            reader.assert_not_called()
