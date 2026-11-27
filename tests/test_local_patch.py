from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/studio_pipeline'), str(ROOT / 'extensions')]
from local_patch import confine_edit


class ConfinedEditTests(unittest.TestCase):
    def test_global_proposal_changes_only_interior(self):
        with TemporaryDirectory() as folder:
            source, proposal, result = [Path(folder) / f'{name}.png' for name in ('source', 'proposal', 'bounded')]
            for mode in ('RGB', 'RGBA', 'L'):
                with self.subTest(mode=mode):
                    before = Image.new(mode, (100, 100))
                    after = Image.new(mode, before.size, 'white')
                    before.save(source)
                    after.save(proposal)
                    confine_edit(source, proposal, [250, 200, 750, 800], result)
                    with Image.open(result) as bounded:
                        observed = np.array(bounded)
                    exterior = np.ones((100, 100), dtype=bool)
                    exterior[20:80, 25:75] = False
                    np.testing.assert_array_equal(observed[exterior], np.array(before)[exterior])
                    np.testing.assert_array_equal(observed[20, 25], np.array(before)[20, 25])
                    np.testing.assert_array_equal(observed[50, 50], np.array(after)[50, 50])

    def test_different_canvas_rejected(self):
        with TemporaryDirectory() as folder:
            source, proposal, result = [Path(folder) / f'{name}.png' for name in ('source', 'proposal', 'bounded')]
            Image.new('RGB', (100, 100)).save(source)
            Image.new('RGB', (101, 100)).save(proposal)
            with self.assertRaises(ValueError):
                confine_edit(source, proposal, [0, 0, 1000, 1000], result)
            self.assertFalse(result.exists())

    def test_mask_protects_holes_alpha_and_soft_edges_after_another_edit(self):
        from PIL import ImageDraw
        from local_patch import merge_selection, selection_mask
        with TemporaryDirectory() as folder:
            root=Path(folder)
            original=root/'original.png'
            proposal=root/'proposal.png'
            Image.new('RGBA',(100,100),(12,18,30,100)).save(original)
            Image.new('RGBA',(100,100),(200,220,240,230)).save(proposal)
            mask=Image.new('L',(100,100))
            draw=ImageDraw.Draw(mask)
            draw.ellipse((5,5,90,90),fill=150)
            draw.rectangle((30,30,65,65),fill=0)
            mask.convert('RGBA').save(root/'mask.png')
            mask=selection_mask(root/'mask.png',(100,100))
            chosen=merge_selection(original,proposal,mask,root/'selected.png')
            with Image.open(chosen) as im:
                self.assertEqual(im.getpixel((45,45)),(12,18,30,100))
                self.assertTrue(12 < im.getpixel((20,40))[0] < 200)
            protected=merge_selection(original,proposal,mask,root/'protected.png',protect_only=True)
            with Image.open(protected) as im:
                self.assertEqual(im.getpixel((45,45)),(12,18,30,100))
                self.assertEqual(im.getpixel((20,40)),(200,220,240,230))
