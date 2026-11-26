"""Verify localized cleanup keeps the source geometry and the surrounding print."""
import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageChops

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from focal_edit import repair_reported_speckles
from print_artifacts import suppress_isolated_marks
from print_judge import fingerprint


class PrintSpeckleTests(unittest.TestCase):
    def test_registered_highlights_and_unbroken_ink_are_preserved(self):
        base=Image.new('RGB',(128,128),'navy')
        draw=ImageDraw.Draw(base)
        draw.line((6,50,119,50),fill='gold',width=3)
        draw.ellipse((47,36,51,40),fill='white')
        noisy=base.copy()
        added=[(x,y) for x in (20,40,60,80) for y in (73,95)]
        for point in added:noisy.putpixel(point,(245,245,245))
        fixed,count=suppress_isolated_marks(base,noisy)
        self.assertEqual(count,len(added))
        self.assertIsNone(ImageChops.difference(fixed,base).getbbox())

    def test_binds_evidence_and_source_before_repair(self):
        with TemporaryDirectory() as tmp:
            d=Path(tmp);base=d/'base.png';master=d/'art.png'
            Image.new('RGB',(64,64),'ivory').save(base)
            im=Image.new('RGB',(128,128),'ivory')
            for x in range(24,88,8):im.putpixel((x,41),(30,30,30))
            im.putpixel((120,120),(5,5,5));im.save(master)
            view=dict(view='comparison',pixels=[8,8,109,110],
                      blind_detail_check=dict(trace=dict(new_artifact=True,improvement='New speckles cover the glaze.')))
            report=dict(status='local',sha256=fingerprint(master),comparison_sha256=fingerprint(base),
                        size=list(im.size),views=[view])
            target,rect=repair_reported_speckles(master,base,report,d,'trial')
            with Image.open(target) as fixed:
                self.assertEqual(fixed.getpixel((120,120)),(5,5,5))
                self.assertEqual(fixed.getpixel((24,41)),(255,255,240))
            self.assertIsNone(repair_reported_speckles(master,base,{**report,'sha256':'unrelated'},d,'trial'))
            view['blind_detail_check']['trace']['new_artifact']=False
            self.assertIsNone(repair_reported_speckles(master,base,report,d,'trial'))

class OpticalShapeTests(unittest.TestCase):
    def test_fine_surface_marks_survive_but_a_new_silhouette_returns_to_drawing(self):
        from PIL import ImageDraw
        from print_artifacts import optical_plane_change
        source=Image.new('RGB',(360,320),'#78b6a6')
        ink=ImageDraw.Draw(source)
        ink.rectangle((95,20,155,265),fill='#193a48')
        ink.rectangle((0,265,359,268),fill='#ccab75')
        fine=source.copy()
        drawing=ImageDraw.Draw(fine)
        for y in range(36,217,8):
            drawing.line((109,y,120,y+2),fill='#899a99',width=1)
        self.assertFalse(optical_plane_change(source,fine)['hold'])
        drawing.polygon([(120,190),(220,146),(252,301),(199,304)],fill='#17384b')
        self.assertTrue(optical_plane_change(source,fine)['hold'])
