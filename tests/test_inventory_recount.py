from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/studio_pipeline'))
from evidence import recount_apparent_absences


class InventoryRecountTests(unittest.TestCase):
    def recount(self,conflict,boxes,points,certainty='clear'):
        reading={'certainty':certainty,'subjects':[{'appearance':'visible figure','center':point} for point in points]}
        with patch('evidence.model_identity',return_value='installed'),patch('evidence.ask',return_value=json.dumps(reading)) as reader:
            seen,receipts=recount_apparent_absences('localhost',object(),{'person':conflict[0]},
                [dict(object='person',observed=conflict[0],requested=conflict[1],detected=len(boxes))],
                [dict(name='person',box=box) for box in boxes])
        return seen['person'],receipts,reader.call_count

    def test_text_absence_requires_independent_localization(self):
        box=[[429,97,840,990]]
        self.assertEqual(self.recount((0,1),box,[[640,500]])[0],1)
        self.assertEqual(self.recount((0,1),box,[[100,100]])[0],0)
        self.assertEqual(self.recount((0,1),box,[[640,500]],'uncertain')[0],0)
        self.assertEqual(self.recount((2,1),box,[[640,500]]),(2,{},0))

    def test_duplicate_centers_cannot_hide_a_missing_subject(self):
        boxes=[[0,0,300,900],[500,0,950,900]]
        self.assertEqual(self.recount((0,2),boxes,[[100,300],[110,320]])[0],0)
        self.assertEqual(self.recount((0,2),boxes,[[100,300],[700,300]])[0],2)
