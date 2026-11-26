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
