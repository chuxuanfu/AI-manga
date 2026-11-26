import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/studio_pipeline'))
from framing import paper_bounds, prepare_full_bleed


class FullBleedTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB',(300,450),'white')
        ImageDraw.Draw(self.image).rectangle((15,19,284,428),fill='#334466')

    def test_only_requested_bleed_and_retains_native_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'print.png'
            self.image.save(path)
            for brief in ('a pencil sketch','not full-bleed','full bleed with a white border'):
                self.assertEqual(prepare_full_bleed(path,brief),path)
            corrected = prepare_full_bleed(path,'a full-bleed wallpaper')
            with Image.open(corrected) as result:
                self.assertAlmostEqual(result.width/result.height,2/3,places=2)
                self.assertEqual(result.getextrema(),((51,51),(68,68),(102,102)))
                self.assertLess(result.width,self.image.width)

    def test_preserves_projecting_marks_and_nonrectangular_negative_space(self):
        draw = ImageDraw.Draw(self.image)
        draw.line((12,7,16,25),fill='black',width=3)
        self.assertIsNone(paper_bounds(self.image))
        wash=Image.new('RGB',(300,450),'white')
        ImageDraw.Draw(wash).ellipse((14,20,285,429),fill='#334466')
        self.assertIsNone(paper_bounds(wash))
