import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/studio_pipeline'))
from palm_review import inspect_attachments, stable_palms


class PalmEvidenceTests(unittest.TestCase):
    def test_mirrored_detections_pair_once(self):
        image=Image.new('RGB',(150,140))
        direct=[(21,17,22,20),(25,80,25,20)]
        reflected=[(107,17,22,20),(101,80,25,20),(10,10,20,20)]
        with patch('palm_review.palms',side_effect=[direct,reflected]):
            self.assertEqual(stable_palms(image),direct)

    def test_visual_readers_must_resolve_every_individual_wrist(self):
        image=Image.new('RGB',(320,320))
        boxes=[(20,30,150,135),(15,140,155,270)]
        hand=dict(attachment='intact',wrist='cuff rim',finger_or_occluder='edge of card',visible_contour='palm reaches cuff',object_boundary='thumb overlaps card corner')
        good=dict(hands=[hand,hand],all_disputed_parts_visible=True,other_unattached_fragment=False)
        for other,expected in [(good,'coherent'),(dict(good,hands=[hand]),'unresolved'),
                               (dict(good,other_unattached_fragment=True),'unresolved')]:
            with patch('palm_review.localized_palms',return_value=((0,0,320,320),boxes)),patch('palm_review.model_identity',return_value='installed'),patch('palm_review.ask',side_effect=map(json.dumps,[good,other])) as judge:
                result=inspect_attachments(None,image,[0,0,800,950],'Verify each wrist attachment.',('first','second'))
            self.assertEqual(result['decision'],expected)
            self.assertEqual([row['order'] for row in result['readings']],[[0,1],[1,0]])
            self.assertEqual([call.kwargs['tag'] for call in judge.call_args_list],['first','second'])

class EditionContextTests(unittest.TestCase):
    def test_first_picture_and_locator_are_from_the_same_revision(self):
        from contextlib import nullcontext
        from construction_review import assess_local_edit
        from PIL import ImageDraw
        original=Image.new('RGB',(400,350),'white')
        candidate=original.copy()
        ImageDraw.Draw(candidate).rectangle((160,120,250,220),fill='navy')
        judgement={'verdict':'new_defect','evidence':'A new navy shape crosses the white margin.','next_edit':'Retain the original stroke.'}
        with patch('construction_review.backend',return_value=nullcontext(None)),patch('construction_review.ask',return_value=json.dumps(judgement)) as judge:
            out=assess_local_edit(original,candidate,[350,300,680,680],'Repair one local connection.')
        self.assertEqual(out['verdict'],'new_defect')
        frames=judge.call_args_list[0].args[1]
        self.assertEqual(len(frames),2)
        self.assertEqual(frames[1].getpixel((205,165)),(0,0,128))
