"""Check safe relocation of independent illustrated subjects."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'runtime/studio_pipeline'), str(ROOT/'extensions')]
import placement


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (400, 200), 'ivory')
        self.plan = dict(standalone=True, label='vase', source=[100, 200, 300, 800],
                         destination_foot=[760, 700], scale=.5, revealed_surface='wall',
                         requirement_quote='on the distant shelf')

    def test_plan_requires_explicit_brief_support(self):
        instruction = 'Move the vase farther back toward the gallery wall.'
        self.assertTrue(placement.placement_requested(instruction))
        self.assertFalse(placement.placement_requested('Repair the left rim of the vase.'))
        with patch.object(placement,'MODEL') as weights, patch.object(placement,'backend',return_value=nullcontext('url')), patch.object(placement,'ask',return_value=json.dumps(self.plan)):
            weights.is_file.return_value=True
            self.assertEqual(placement.placement_plan(self.image,instruction,'A vase on the distant shelf.'), self.plan)
            self.assertIsNone(placement.placement_plan(self.image,instruction,'A vase resting on a table.'))

    def test_bad_coordinates_do_not_start_segmentation(self):
        with patch.object(placement,'probabilities',side_effect=AssertionError('unusable plan')):
            for key,value in [('source',None),('source',[3]*5),('destination_foot',[0,float('nan')]),
                              ('scale','small'),('scale',True),('label',None),('source',[0,0,1000,1000])]:
                with self.subTest(key=key,value=value):
                    self.assertIsNone(placement.reposition(self.image,{**self.plan,key:value}))

    def test_closes_matching_paint_without_filling_openings(self):
        rgb=Image.new('RGB',(80,80),'ivory')
        ImageDraw.Draw(rgb).rectangle((41,32,47,38),fill='black')
        alpha=Image.new('L',rgb.size)
        draw=ImageDraw.Draw(alpha)
        draw.rectangle((8,8,73,73),fill=255)
        for x in (26,41):draw.rectangle((x,32,x+6,38),fill=0)
        fixed=placement.mend_color_matched_holes(rgb,alpha)
        self.assertEqual(fixed.getpixel((29,35)),255)
        self.assertEqual(fixed.getpixel((44,35)),0)
        self.assertEqual(fixed.getpixel((3,35)),0)

    def test_actor_moves_once_and_unrelated_paint_is_preserved(self):
        ink=(200,24,62)
        ImageDraw.Draw(self.image).rectangle((43,43,116,156),fill=ink)
        self.image.putpixel((390,8),(27,44,189))
        def precise(crop,rough):
            return Image.fromarray(np.uint8(np.all(np.asarray(crop)==np.array(ink),axis=2))*255)
        def fill(image,mask):
            result=image.copy()
            result.paste('ivory',(0,0,image.width,image.height),mask)
            return result
        with patch.object(placement,'probabilities',return_value=[np.ones((28,28))]),patch.object(placement,'refine_subject_mask',side_effect=precise),patch.object(placement,'erase_region',side_effect=fill):
            after,areas=placement.reposition(self.image,self.plan)
            self.assertEqual(after.getpixel((305,110)),ink)
            self.assertEqual(after.getpixel((80,80)),(255,255,240))
            self.assertEqual(after.getpixel((390,8)),(27,44,189))
            self.assertLess(areas['source_subject'][2],areas['destination'][0])
            self.assertIsNone(placement.reposition(self.image,{**self.plan,'destination_foot':[200,700]}))

    def test_registered_placement_preserves_the_exact_masked_reconstruction(self):
        from tempfile import TemporaryDirectory
        from PIL import ImageChops
        from run import repair_placement
        with TemporaryDirectory() as tmp, patch('placement.reposition',return_value=(self.image,{'inspection':[50,100,800,900]})), \
                patch('run.repaint',side_effect=AssertionError('An unrestricted second drawing can duplicate the moved animal')):
            path,inspection=repair_placement(self.image,{},'one dog',3,8193,Path(tmp),'dog')
            with Image.open(path) as actual:self.assertIsNone(ImageChops.difference(actual,self.image).getbbox())
            self.assertEqual(inspection,[50,100,800,900])

    def test_carried_parts_must_be_present_and_touch_the_assembly(self):
        plan={**self.plan,'parts':['saucer','fruit']}
        body=np.zeros((72,72),dtype=float);body[42:48,16:55]=.99
        fruit=np.zeros_like(body);fruit[32:44,26:46]=.99
        distant=np.zeros_like(body);distant[5:10,8:16]=.99
        with patch.object(placement,'refine_subject_mask',side_effect=lambda rgb,rough:rough.copy()), \
                patch.object(placement,'erase_region',side_effect=lambda rgb,mask:rgb.copy()):
            for other,expected in ((fruit,True),(fruit*.1,False),(distant,False)):
                with patch.object(placement,'probabilities',return_value=[body,other]):
                    self.assertEqual(placement.reposition(self.image,{**plan,'scale':1}) is not None,expected)
        self.assertTrue(placement.placement_requested('Rest the filled saucer on top of the lid.'))
        self.assertTrue(placement.placement_requested('把盛好的盘子放到篮子上面。'))

    def test_recomposition_requires_a_confirmed_gap_and_draws_markers_only_on_the_guide(self):
        from run import contact_layout_graph
        statement='The pen touches the paper.'
        issue=dict(confirmed=True,minimum_gap=250,items=['pen','paper'],requirement=statement,
                   observations=[dict(relation='separated',actor_edge=[100,400],target_edge=[350,400])])
        frames=[]
        def capture(picture):
            frames.append(picture.copy())
            return 'view-'+str(len(frames))+'.png'
        with patch('run.installed_editor',return_value='editor'),patch('run.local_image',side_effect=capture):
            for override in ({'minimum_gap':10},{'confirmed':False},{'requirement':'another scene'}):
                self.assertIsNone(contact_layout_graph(self.image,{**issue,**override},statement,2,'local'))
            graph=contact_layout_graph(self.image,issue,statement,2,'local')
            self.assertIsNotNone(graph)
            with self.subTest('the source contains the original pixels'):
                self.assertEqual(frames[0].tobytes(),self.image.tobytes())
                self.assertNotEqual(frames[0].tobytes(),frames[1].tobytes())
            loaders=[v['inputs']['image'] for v in graph.api.values() if v['class_type']=='LoadImage']
            self.assertCountEqual(loaders,['view-1.png','view-2.png'])
