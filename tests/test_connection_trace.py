import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from print_judge import verify_depicted_connection


class ConnectionTraceTests(unittest.TestCase):
    def test_contradictory_inventory_needs_visible_trace_and_independent_brief_check(self):
        image=Image.new('RGB',(256,256),'gold')
        for fault,authority,expected in [('visible gap',None,False),('','visible_fault',False),
                                         ('','only_unseen_internal_hardware',True)]:
            replies=[dict(condition='visible_break',trace='The strap meets the rail with no visible gap.'),
                     dict(condition='visible_coherent_contact',entry_surface='strap on rail',end_or_occluder='rear rail edge',
                          trace='The strap crosses the top and passes behind the upper rail.',visible_fault=fault)]
            if authority:replies.append(dict(basis=authority,trace='The metal rail hides the rear fastener.'))
            with patch('print_judge.model_identity',return_value='installed'),patch('print_judge.ask',side_effect=map(json.dumps,replies)):
                result=verify_depicted_connection(None,image,[240,200,440,350],'No visible bolt at the rail.','One strap wraps the rail.')
            self.assertEqual(result['disproved'],expected)
            self.assertIn('contradiction_check',result)
