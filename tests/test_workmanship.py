import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/studio_pipeline'))


class WorkmanshipTests(unittest.TestCase):
    def test_replacement_geometry_is_grounded_in_its_repair_field(self):
        from illustration import construction_guide
        from masks import frame_supported_object
        guide=construction_guide('Draw wound thread between the two ends of the sewing spool.')
        self.assertEqual(guide[1]['placement'],'supported_object')
        with Image.open(guide[0]) as source:
            plate,box=frame_supported_object(source,(256,192),(55,78,145,140))
            self.assertEqual(plate.size,(256,192))
            self.assertEqual(plate.getchannel('A').getbbox(),box)
            self.assertEqual(box[3],140)
            self.assertLessEqual(abs((box[0]+box[2])/2-100),1)
            with self.assertRaises(ValueError):
                frame_supported_object(source,(256,192),(400,100,425,120))
        self.assertIsNone(construction_guide('Preserve the sewing spool and the wound thread.'))

    def test_paper_relief_preserves_surface_tones_in_the_print_pass(self):
        from illustration import continuous_tone
        for medium in ('layered cut-paper diorama', 'paper relief', '立体剪纸'):
            self.assertTrue(continuous_tone(medium))
        for medium in ('flat black paper silhouette', 'manga line art on paper', ''):
            self.assertFalse(continuous_tone(medium))

    def test_landscape_and_portrait_receive_the_same_full_picture_area(self):
        from reference_review import artwork_sheet
        areas=[]
        for width,height in ((1920,1080),(1080,1920),(1024,1024)):
            sheet=artwork_sheet(Image.new('RGB',(width,height),'red'),[])
            areas.append(dict((color,count) for count,color in sheet.getcolors())[255,0,0])
        self.assertLess(max(areas)-min(areas),2000)
        self.assertGreater(min(areas),398000)

    def test_technique_studies_follow_the_requested_medium(self):
        from illustration import medium_study
        self.assertTrue(medium_study('Japanese shin-hanga woodblock print').is_file())
        self.assertTrue(medium_study('layered cut-paper relief').is_file())
        self.assertIsNone(medium_study('Botanical watercolor on rough paper'))
        self.assertIsNone(medium_study('Avoid woodcut'))
        self.assertIsNone(medium_study('Mixed cut-paper and woodcut'))

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

    def test_finish_material_regression_requires_a_second_reading_of_corresponding_crops(self):
        from contextlib import nullcontext
        from tempfile import TemporaryDirectory
        from reference_review import review_finish_changes
        from construction_review import LARGE
        first=dict(verdict='local',box=[240,580,360,660],target='sleeve',observation='New transparent band.',
                   fix='Restore the paper cuff.')
        second=dict(verdict='unintended_change',original_feature='Folded opaque paper.',
                    repainted_feature='A transparent rectangular band.',restore='Restore the opaque folded cuff.')
        with TemporaryDirectory() as tmp:
            old,new=Path(tmp)/'old.png',Path(tmp)/'new.png'
            Image.new('RGB',(800,600),'gold').save(old)
            Image.new('RGB',(1600,1200),'olive').save(new)
            with patch('reference_review.backend',return_value=nullcontext('local')), \
                    patch('reference_review.available',return_value=True), \
                    patch('reference_review.ask',side_effect=[json.dumps(first),json.dumps(second)]) as reader:
                result=review_finish_changes(old,new,'Layered paper illustration.')
                self.assertEqual(result['fix'],second['restore'])
                self.assertEqual(reader.call_args.kwargs['tag'],LARGE)
                pair=reader.call_args.args[1]
                self.assertEqual(pair[0].size,pair[1].size)
                self.assertEqual(pair[0].getpixel((5,5)),(255,215,0))
            with patch('reference_review.backend',return_value=nullcontext('local')), \
                    patch('reference_review.ask',return_value=json.dumps({**first,'box':[0,0,1000,1000]})) as reader:
                self.assertEqual(review_finish_changes(old,new,'A paper study.')['verdict'],'unclear')
                self.assertEqual(reader.call_count,1)

    def test_finish_repaint_uses_approved_scene_and_only_confirmed_final_feedback(self):
        from illustration import finish_repaint_graph
        from reference_review import final_repaint_direction
        from print_judge import MODEL
        grade=dict(decision='below',consistent_deficits=['lighting'],assessment_model=MODEL,
                   separate_material_evidence=True,redraw_direction='Ground the existing figures with cast shadows.')
        direction=final_repaint_direction(grade)
        self.assertEqual(direction,grade['redraw_direction'])
        for override in ({'decision':'review'},{'assessment_model':'draft judge'},{'separate_material_evidence':False}):
            self.assertFalse(final_repaint_direction({**grade,**override}))
        graph=finish_repaint_graph('A lamb beside a gate.',direction,42,'actual.png','revised','local.safetensors',
                                  study='paper.webp',full=True)
        nodes=list(graph.api.values())
        self.assertEqual([n['inputs']['image'] for n in nodes if n['class_type']=='LoadImage'],['actual.png','paper.webp'])
        self.assertEqual(next(n['inputs']['steps'] for n in nodes if n['class_type']=='KSampler'),40)
        self.assertIn(direction,next(n['inputs']['prompt'] for n in nodes if n['class_type']=='TextEncodeQwenImageEditPlus'))
        quick=finish_repaint_graph('A cat at dusk.',direction,42,'cat.png','revised','local.safetensors')
        self.assertEqual([n['inputs']['image'] for n in quick.api.values() if n['class_type']=='LoadImage'],['cat.png'])
        self.assertEqual(next(n['inputs']['steps'] for n in quick.api.values() if n['class_type']=='KSampler'),4)

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
