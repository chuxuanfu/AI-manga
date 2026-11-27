import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from run import record_redraw


class RedrawBudgetTests(unittest.TestCase):
    def test_resume_keeps_local_and_whole_frame_attempts_in_one_budget(self):
        with TemporaryDirectory() as td:
            file=Path(td)/'record.json'
            record={}
            record_redraw(None,file,record,'untouched')
            self.assertFalse(file.exists())
            record_redraw(2,file,record,'local-1')
            record_redraw(2,file,json.loads(file.read_text()),'recompose-1')
            with self.assertRaises(SystemExit) as stopped:
                record_redraw(2,file,json.loads(file.read_text()),'final-edit')
            self.assertEqual(stopped.exception.code,2)
            saved=json.loads(file.read_text())
            self.assertEqual(saved['redraw_attempts'],['local-1','recompose-1'])
            self.assertTrue(saved['status'].startswith('未达标'))
