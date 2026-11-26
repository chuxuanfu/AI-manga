import sys
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime/studio_pipeline'))
from registered_edit import CONTROL, registered_design, controlled_drawing, drawing_inputs


class RegisteredEditTests(unittest.TestCase):
    def test_correct_shape_uses_corresponding_context_and_preserves_surroundings(self):
        rng = np.random.default_rng(57)
        drawing = Image.new('RGB', (240, 220), '#e6d5b4')
        ink = ImageDraw.Draw(drawing)
        for x, y, size in rng.integers((5, 5, 3), (225, 205, 13), (200, 3)):
            ink.ellipse((x, y, x+size, y+size), fill='#59472e')
        ink.ellipse((86, 65, 143, 150), fill='orange')
        moved = Image.fromarray(cv2.warpAffine(np.asarray(drawing), np.float32([[1, 0, 5], [0, 1, -4]]),
                                               drawing.size, borderMode=cv2.BORDER_REFLECT_101))
        current = moved.copy()
        ImageDraw.Draw(current).rectangle((86, 60, 151, 151), fill='navy')
        mask = Image.new('L', drawing.size)
        ImageDraw.Draw(mask).rectangle((82, 54, 159, 159), fill=255)
        result, evidence = registered_design(current, drawing, mask)
        self.assertGreaterEqual(evidence['context_matches'], 7)
        self.assertEqual(result.getpixel((115, 105)), moved.getpixel((115, 105)))
        self.assertTrue(np.array_equal(np.asarray(result)[np.asarray(mask)==0], np.asarray(current)[np.asarray(mask)==0]))
        self.assertIsNone(registered_design(current, Image.new('RGB', drawing.size), mask))

    def test_masks_and_guide_reach_both_controlled_conditioning_inputs(self):
        image = Image.new('RGB', (128, 160), 'white')
        ImageDraw.Draw(image).ellipse((18, 38, 100, 119), fill='black')
        mask = Image.new('L', image.size)
        ImageDraw.Draw(mask).rectangle((30, 33, 114, 136), fill=255)
        prepared, edges = drawing_inputs(image, mask, (768, 960))
        self.assertEqual(prepared.getpixel((0, 0))[-1], 255)
        self.assertEqual(prepared.getpixel((255, 255))[-1], 0)
        self.assertEqual(edges.convert('L').getextrema(), (0, 255))
        graph = controlled_drawing('prepared.png', 'ink.png', 'Woodcut rope and joined iron base.', 33, 'test')
        nodes = graph.api
        ids = {v['class_type']: k for k,v in nodes.items()}
        sampler = nodes[ids['KSampler']]['inputs']
        self.assertEqual(sampler['negative'], [ids['ControlNetApplyAdvanced'], 1])
        self.assertEqual(sampler['positive'], [ids['ControlNetApplyAdvanced'], 0])
        self.assertEqual(sampler['latent_image'], [ids['SetLatentNoiseMask'], 0])
        self.assertEqual(nodes[ids['ControlNetLoader']]['inputs']['control_net_name'], CONTROL)
