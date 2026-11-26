import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from print_judge import verify_depicted_connection


class ConnectionTraceTests(unittest.TestCase):
    def test_contradictory_inventory_needs_visible_trace_and_independent_brief_check(self):
        image=Image.new('RGB',(256,256),'gold')
        for fault,authority,expected in [('visible gap',None,False),('','visible_fault',False),
                                         ('','only_unseen_internal_hardware',True)]:
            replies=[dict(condition='visible_break',trace='The strap meets the rail with no visible gap.'),
                     dict(condition='visible_coherent_contact',entry_surface='strap on rail',end_or_occluder='rear rail edge',
                          trace='The strap crosses the top and passes behind the upper rail.',visible_fault=fault)]
            if authority:replies.append(dict(basis=authority,trace='The metal rail hides the rear fastener.'))
            with patch('print_judge.model_identity',return_value='installed'),patch('print_judge.ask',side_effect=map(json.dumps,replies)):
                result=verify_depicted_connection(None,image,[240,200,440,350],'No visible bolt at the rail.','One strap wraps the rail.')
            self.assertEqual(result['disproved'],expected)
            self.assertIn('contradiction_check',result)

class ContradictoryActionTests(unittest.TestCase):
    def test_disagreements_reenter_a_visual_check(self):
        from print_judge import disavowed_action_fault
        for finding,expected in [('None. The head and its contours are smooth.',True),
                                 ('No visible physical defect at this joint.',True),
                                 ('No visible shape contradiction. The track passes behind the rail.',True),
                                 ('One visible joint is disconnected.',False),
                                 ('The shape is nonetheless broken.',False)]:
            self.assertEqual(disavowed_action_fault(dict(condition='visible_fault',problem=finding)),expected)
            self.assertFalse(disavowed_action_fault(dict(condition='insufficient',problem=finding)))

    def test_trace_must_show_both_disputed_parts(self):
        from print_judge import inspect_disputed_action
        claim='The railing appears to merge with the tracks.'
        initial=dict(condition='visible_fault',visible_shapes='Rail and train.',depicted_action='none',
                     problem='No visible shape contradiction.')
        trace=dict(first_landmark='cat ear',second_landmark='shoulder',connection='unbroken contour',
                   repair='none',verdict='continuous_and_plausible',disputed_parts_in_view=False)
        with patch('print_judge.model_identity',return_value='installed'),patch('print_judge.ask',side_effect=map(json.dumps,[initial,trace])) as review:
            result=inspect_disputed_action(None,Image.new('RGB',(450,350)),[0,400,500,900],'A train scene.',claim)
        self.assertEqual(result['condition'],'insufficient')
        self.assertNotIn(claim,review.call_args_list[0].args[2])
        self.assertIn(claim,review.call_args_list[-1].args[2])
        self.assertEqual(result['consistency_check']['tested_claim'],claim)


class PairedContourTests(unittest.TestCase):
    def test_only_reopens_a_disavowed_fusion_claim_about_the_named_parts(self):
        from paired_contours import needs_pair_trace
        check=dict(tested_claim='The fawn forelegs are fused.',verdict='specific_visible_error',
                   disputed_parts_in_view=True,repair='No correction needed; separate joints.',first_landmark='Hoof A',
                   second_landmark='Hoof B',connection='Distinct outlines reach the body.')
        self.assertTrue(needs_pair_trace(dict(condition='insufficient',consistency_check=check)))
        self.assertFalse(needs_pair_trace(dict(condition='insufficient',consistency_check=dict(check,repair='Separate the broken legs.'))))

    def test_independent_attachments_and_mirrored_tip_locations_must_agree(self):
        from paired_contours import inspect_pair
        original=dict(first_tip=[220,810],second_tip=[410,820],
                      first_trace='The left outline joins the shoulder.',second_trace='The far outline continues to the chest.',
                      between='A narrow grass interval separates them.',both_named_parts_visible=True,
                      separate_attachments_visible=True,shared_or_broken_interval=False)
        reverse=dict(original,first_tip=[589,820],second_tip=[782,810])
        for second,expected in ((reverse,True),(dict(reverse,first_tip=[300,650]),False),
                                (dict(reverse,shared_or_broken_interval=True),False),(dict(reverse,separate_attachments_visible=False),False)):
            with patch('paired_contours.ask',side_effect=map(json.dumps,(original,second))):
                result=inspect_pair(None,Image.new('RGB',(1000,900)),[550,500,800,900],'fused forelegs',('one','two'))
            self.assertEqual(result['confirmed'],expected)


class OpticalTieBreakTests(unittest.TestCase):
    def test_disputed_panel_requires_two_independent_optical_observations(self):
        from PIL import ImageDraw,ImageFilter
        from print_judge import independent_detail_check
        from construction_review import LARGE
        sharp=Image.new('RGB',(192,192),'white')
        pen=ImageDraw.Draw(sharp)
        for x in range(7,170,9):pen.line((x,10,x+13,180),width=2,fill='black')
        weak=sharp.filter(ImageFilter.GaussianBlur(2))
        first=dict(trace='Twin ink strokes',improvement='Ends separately visible.',resolvability='resolved',
                   defined_finer_forms=True,new_artifact=False)
        vote=dict(finer='B',location='middle',evidence='Separated tips.',new_artifact=False)
        loupe=dict(finer_version='second',kind='resolved_material_or_contour',location='tips',evidence='Separated outlined mark.')
        for damaged in [False,True]:
            calls=[first,dict(vote,new_artifact=damaged),loupe,dict(loupe,finer_version='first')]
            with patch('print_judge.ask',side_effect=map(json.dumps,calls)) as judge,patch('print_judge.model_identity',return_value='local'),patch('print_judge.resize_only',return_value={'condition':'distinct_signal'}):
                result=independent_detail_check(None,[weak,sharp],sharp,'ink illustration')
            self.assertEqual(result['confirmed'],not damaged)
            self.assertEqual(judge.call_count,2 if damaged else 4)
            if not damaged:
                self.assertEqual([c.kwargs['tag'] for c in judge.call_args_list[-2:]],[LARGE,LARGE])
