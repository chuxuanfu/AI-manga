import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from retouch import Editor, drawing_mask, image_info


class BrushEditorTests(unittest.TestCase):
    def test_native_strokes_preserve_eraser_hole(self):
        strokes=[dict(tool='paint',width=41,points=[[2,15],[140,15]]),
                 dict(tool='erase',width=9,points=[[70,15]])]
        mask=drawing_mask((3840,2160),strokes)
        self.assertEqual(mask.size,(3840,2160))
        self.assertEqual([mask.getpixel((x,15)) for x in (2,70,140,200)],[255,0,255,0])
        with self.assertRaises(ValueError):
            drawing_mask(mask.size,[strokes[-1]])

    def test_8k_submission_preserves_size_and_deduplicates_retries(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp)
            file=folder/'input.png'
            Image.new('L',(7680,512)).save(file)
            self.assertEqual(image_info(file),(7680,512))
            with Editor(('127.0.0.1',0),folder) as server, patch('retouch.IMAGES',folder/'images'), \
                    patch('retouch.settings',return_value={}), patch('retouch.subprocess.Popen',return_value=Mock()) as launch:
                src=server.add_source(file,file)
                token='a'*32
                data=dict(request=token,source=src['id'],change='修正袖口',
                          strokes=[dict(tool='paint',width=24,points=[[500,300]])])
                self.assertEqual(server.submit(data),token)
                self.assertEqual(server.submit(data),token)
                launch.assert_called_once()
                args=launch.call_args.args[0]
                self.assertIn('--keep-native',args)
                with Image.open(args[args.index('--selection-mask')+1]) as im:
                    self.assertEqual(im.size,(7680,512))
                    self.assertEqual(im.getpixel((500,300)),255)
