from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from print_judge import store


class ParallelReceiptsTests(unittest.TestCase):
    def test_concurrent_writers_replace_only_fully_written_receipts(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json'
            def write(i):
                store(path,{'index':i,'stamp':str(i)*4096})
                row=json.loads(path.read_text())
                self.assertEqual(row['stamp'],str(row['index'])*4096)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(write,range(80)))
            self.assertEqual(list(path.parent.iterdir()),[path])

    def test_finish_reuses_one_gpu_lease_for_structure_and_print(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        from PIL import Image
        from evidence import backend
        from print_judge import grade_finished_print
        loads=[]
        @contextmanager
        def serve(model,timings):
            loads.append(model)
            yield 'loopback'
        def coarse():
            with backend('a') as addr:self.assertEqual(addr,'loopback')
            return {'status':'pass'}
        def native(*args):
            with backend('b') as addr:self.assertEqual(addr,'loopback')
            return {'status':'pass'}
        with TemporaryDirectory() as tmp, patch('evidence._local_service',side_effect=serve),patch('evidence.model_identity',return_value='installed'),patch('print_judge._inspect_print',side_effect=native) as scan:
            root=Path(tmp);im=root/'print.png';Image.new('RGB',(18,18)).save(im)
            args=dict(small=None,focus=None,draft=None,source_comparison=False)
            final,report,_=grade_finished_print(im,'A figure',root,coarse,**args)
            self.assertEqual((final['status'],report['status'],loads),('pass','pass',['a']))
            _,report,_=grade_finished_print(im,'A figure',root,lambda:{'status':'global'},**args)
            self.assertIsNone(report)
            self.assertEqual(scan.call_count,1)
