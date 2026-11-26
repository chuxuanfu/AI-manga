"""A borrowed documentary pose must not introduce a second painting medium."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from masks import pose_outline


class PoseOutlineTests(unittest.TestCase):
    def test_outline_discards_photographic_color_and_background(self):
        photo=Image.new('RGB',(100,120),'blue')
        mask=Image.new('L',photo.size)
        ImageDraw.Draw(mask).ellipse((16,12,79,111),fill=255)
        photo.paste('orange',(0,0,*photo.size),mask)
        with patch('masks.probabilities',return_value=[np.ones((32,32))]),patch('masks.refine_subject_mask',return_value=mask):
            guide=pose_outline(photo,'a standing figure')
        rgb=np.asarray(guide)
        self.assertTrue(np.array_equal(rgb[:,:,0],rgb[:,:,2]))
        self.assertLess(guide.width,photo.width)
        self.assertEqual(guide.getpixel((0,0)),(247,247,247))
        self.assertLess(rgb.min(),100)

    def test_unlocated_pose_does_not_fall_back_to_a_photograph(self):
        with patch('masks.probabilities',return_value=[np.zeros((32,32))]):
            self.assertIsNone(pose_outline(Image.new('RGB',(60,60),'red'),'animal'))
