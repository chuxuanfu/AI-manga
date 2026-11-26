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
