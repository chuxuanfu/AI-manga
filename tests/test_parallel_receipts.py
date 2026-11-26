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

    def test_text_extraction_reuses_exact_local_request_and_images_still_run(self):
        import io
        from unittest.mock import patch
        from PIL import Image
        import evidence
        schema={'type':'object','properties':{'count':{'type':'integer'}},'required':['count']}
        packet=json.dumps({'message':{'content':'{"count":1}'},'done_reason':'stop'}).encode()
        with TemporaryDirectory() as tmp, patch.object(evidence,'RUNTIME',Path(tmp)), \
                patch.object(evidence,'model_identity',return_value='weights-1') as identity, \
                patch.object(evidence.shutil,'which',return_value=sys.executable), \
                patch.object(evidence.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(packet)) as query:
            url='http://127.0.0.1:12345/api/chat'
            with evidence.review_session() as timing:
                for _ in range(2):
                    self.assertEqual(json.loads(evidence.ask(url,[],'one bird',schema)),{'count':1})
                self.assertEqual(timing['reused_text_readings'],1)
            self.assertEqual(query.call_count,1)
            evidence.ask(url,[],'two birds',schema)
            identity.return_value='weights-2'
            evidence.ask(url,[],'one bird',schema)
            for _ in range(2):
                evidence.ask(url,[Image.new('RGB',(12,12))],'one bird',schema)
            self.assertEqual(query.call_count,5)
            for receipt in Path(tmp).rglob('*.json'):
                receipt.write_text('incomplete')
            evidence.ask(url,[],'one bird',schema)
            self.assertEqual(query.call_count,6)

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

    def test_gallery_and_publication_require_the_brief_and_every_native_field(self):
        from print_judge import REVISION, complete_print_review, print_windows
        size=[3840,2160]
        views=[dict(view='whole',pixels=[0,0,*size],observation={'decision':'local'},
                    independent_context={'decision':'clear'})]
        views.extend(dict(view=kind,pixels=list(rect),observation={'decision':'clear'})
                     for kind,rect in print_windows(tuple(size),{}) if kind=='native')
        report=dict(size=size,views=views,status='pass',revision=REVISION,planned=len(views))
        self.assertFalse(complete_print_review(report))
        views[0]['brief_checked']=True
        self.assertTrue(complete_print_review(report))
        views.pop()
        report['planned']=len(views)
        self.assertFalse(complete_print_review(report))
