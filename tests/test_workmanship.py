import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/studio_pipeline'))


class WorkmanshipTests(unittest.TestCase):
    def test_consistent_evidence_in_both_orders(self):
        from reference_review import order_verdict
        def decision(a,b,text='two separate marks'):
            return order_verdict({'form':dict(level=a,evidence=text)},
                                 {'form':dict(level=b,evidence=text)},['form'])['decision']
        self.assertEqual(decision('left','right'),'below')
        self.assertEqual(decision('left','left'),'review')
        self.assertEqual(decision('tie','tie',''),'review')
        self.assertEqual(decision('right','left'),'candidate')

    def test_final_workmanship_grades_both_orders_with_the_print_model(self):
        from contextlib import nullcontext
        from evidence import TAG
        from print_judge import MODEL
        from reference_review import AXES, compare
        board=Image.new('RGB',(1152,1104),'white')
        grades={axis:dict(level='tie',evidence='The rope and fabric each show distinct directional marks.') for axis in AXES}
        with patch('reference_review.available',return_value=True),patch('reference_review.comparison',return_value=board), \
                patch('reference_review.backend',side_effect=lambda tag:nullcontext('local')), \
                patch('reference_review.ask',return_value=json.dumps(grades)) as reader:
            for final,tag in ((False,TAG),(True,MODEL)):
                result=compare(board,{},None,brief='A moored boat in woodcut.',final=final)
                self.assertEqual((result['decision'],result['assessment_model']),('candidate',tag))
                self.assertEqual([call.kwargs['tag'] for call in reader.call_args_list],[tag,tag])
                reader.reset_mock()

    def test_coordinates_without_a_name_use_observed_part_and_wider_pixels(self):
        from construction_review import verify_repair_location
        from print_judge import MODEL
        im=Image.new('RGB',(1200,1600),'white')
        absent=dict(state='target_absent_or_unclear',target_visible=False,evidence='Contour clipped.')
        trace=dict(part_visible=True,landmark_one='cuff',landmark_two='thumb',
                   trace='Cuff continues to the thumb.',finding='visible_and_coherent')
        reads=[absent,dict(left=300,top=420,right=370,bottom=450),absent,dict(part='right thumb'),
               dict(located=True,first_boundary='thumb outline',second_boundary='cuff opening'),trace,trace]
        result=dict(target='[20,20,50,50]',observation='The right thumb may be detached.',fix='',box=[20,20,50,50])
        with patch('construction_review.available',return_value=True),patch('construction_review.ask',side_effect=map(json.dumps,reads)) as reader:
            verify_repair_location('local',im,result,'independent-model')
        self.assertEqual(result['verdict'],'coherent')
        self.assertEqual(result['location_verification'][-1]['extracted_part'],'right thumb')
        self.assertEqual(result['proposed_finding']['observation'],'The right thumb may be detached.')
        for call in reader.call_args_list[-3:]:
            first,wide=call.args[1]
            self.assertGreater(wide.width,first.width)
            self.assertEqual(call.kwargs['tag'],MODEL)

    def test_an_undecided_reference_comparison_reopens_the_exact_master(self):
        from production import workmanship_confirmed
        def row(digest,grade):
            return dict(stage='final-art',hash=digest,review={'status':'pass','参考比较':{'decision':grade}})
        record={'工作流':'reference-illustration-v1','reviews':[row('old','candidate'),row('new','review')]}
        self.assertFalse(workmanship_confirmed(record,'new'))
        record['reviews'].append(row('new','candidate'))
        self.assertTrue(workmanship_confirmed(record,'new'))
