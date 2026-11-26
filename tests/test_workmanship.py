import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/studio_pipeline'))


class WorkmanshipTests(unittest.TestCase):
    def test_landscape_and_portrait_receive_the_same_full_picture_area(self):
        from reference_review import artwork_sheet
        areas=[]
        for width,height in ((1920,1080),(1080,1920),(1024,1024)):
            sheet=artwork_sheet(Image.new('RGB',(width,height),'red'),[])
            areas.append(dict((color,count) for count,color in sheet.getcolors())[255,0,0])
        self.assertLess(max(areas)-min(areas),2000)
        self.assertGreater(min(areas),398000)

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
        grades={axis:dict(level='tie',evidence='Each material has clear directional marks.',left_detail='Twisted strands around the mooring ring.',right_detail='Sleeve seams follow the folded elbow.') for axis in AXES}
        with patch('reference_review.available',return_value=True),patch('reference_review.comparison',return_value=board), \
                patch('reference_review.artwork_pair',return_value=[board,board.copy()]), \
                patch('reference_review.backend',side_effect=lambda tag:nullcontext('local')), \
                patch('reference_review.ask',return_value=json.dumps(grades)) as reader:
            for final,tag in ((False,TAG),(True,MODEL)):
                result=compare(board,{},None,brief='A moored boat in woodcut.',final=final)
                self.assertEqual((result['decision'],result['assessment_model']),('candidate',tag))
                self.assertEqual([call.kwargs['tag'] for call in reader.call_args_list],[tag,tag])
                if final:
                    self.assertNotIn('A moored boat in woodcut.',reader.call_args.args[2])
                    self.assertIn('left_detail',reader.call_args.args[3]['properties']['lighting']['required'])
                reader.reset_mock()

    def test_repaint_uses_both_candidate_observations_and_excludes_reference_subjects(self):
        from reference_review import redraw_direction,last_finish_direction
        grade=dict(decision='below',consistent_deficits=['material_construction'],
                   reference_on_left={'material_construction':{'left_detail':'A robot paw','right_detail':'A flat yellow cape'}},
                   reference_on_right={'material_construction':{'right_detail':'The umbrella ribs','left_detail':'A yellow collar'}})
        edit='Fold the cape collar from a distinct layer of yellow paper.'
        with patch('reference_review.ask',return_value=json.dumps({'edit':edit})) as writer:
            grade['redraw_direction']=redraw_direction(grade,'A girl watches a lamb.',address='local')
            prompt=writer.call_args.args[2]
            self.assertIn('flat yellow cape',prompt)
            self.assertIn('yellow collar',prompt)
            self.assertNotIn('robot',prompt)
            self.assertNotIn('umbrella',prompt)
        record={'reviews':[{'review':{'参考比较':grade}}]}
        self.assertEqual(last_finish_direction(record),edit)
        record['reviews'].append({'review':{'参考比较':{'decision':'candidate'}}})
        self.assertEqual(last_finish_direction(record),'')

    def test_shared_unlocated_reference_evidence_requires_another_reader(self):
        from reference_review import order_verdict
        weak=dict(level='tie',evidence='Both use paper.',left_detail='Both use layered paper.',right_detail='Both use layered paper.')
        self.assertEqual(order_verdict({'axis':weak},{'axis':weak},['axis'],sided=True)['decision'],'review')

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
        self.assertFalse(workmanship_confirmed(record,'new'))
        from reference_review import AXES
        from print_judge import MODEL
        readings={axis:dict(level='tie',evidence='Different materials retain distinct edge structures.',
                     left_detail='Knit stitches along the lifted wrist.',right_detail='Cut edges around the sheep fleece.') for axis in AXES}
        grade=record['reviews'][-1]['review']['参考比较']
        grade.update(assessment_model=MODEL,separate_material_evidence=True,reference_on_left=readings,reference_on_right=readings)
        self.assertTrue(workmanship_confirmed(record,'new'))
        grade['reference_on_right']={}
        self.assertFalse(workmanship_confirmed(record,'new'))
