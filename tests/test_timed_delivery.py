from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'scripts')]
import deadlines


class TimedDeliveryTests(unittest.TestCase):
    def test_latest_repair_and_two_fixed_windows_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)
            folder=job/'rounds/0001'
            folder.mkdir(parents=True)
            clock=deadlines.begin(job,'Moonlit garden','16:9')
            draft=folder/'draft.png'
            Image.new('RGB',(128,72),'blue').save(draft)
            deadlines.checkpoint(folder,draft,'draft')
            self.assertEqual(deadlines.boundary(job,clock)[0],clock['started']+900)
            self.assertIsNone(deadlines.read(job/deadlines.LATEST)['first_4k_at'])
            four=folder/'painted.png'
            Image.new('RGB',(3840,2160),'red').save(four)
            ready=time.time()
            deadlines.checkpoint(folder,four,'first-print',print_ready=True,observed_at=ready)
            repair=folder/'repair.png'
            Image.new('RGB',(3840,2160),'green').save(repair)
            deadlines.checkpoint(folder,repair,'repair',print_ready=True)
            resumed=deadlines.begin(job,'Moonlit garden','16:9')
            self.assertEqual(deadlines.boundary(job,resumed)[0],ready+900)
            state={}
            self.assertEqual(deadlines.manual_delivery(job,state,'review timeout'),3)
            self.assertEqual(state['status'],deadlines.MANUAL)
            receipt=state['manual_delivery']
            self.assertEqual(receipt['source'],str(repair.resolve()))
            self.assertFalse(receipt['approved'])
            with Image.open(job/deadlines.MANUAL_FILE) as im:
                self.assertEqual(im.size,(3840,2160))
                self.assertEqual(im.getpixel((99,99)),(0,128,0))
                self.assertIn(deadlines.MANUAL,im.info['Title'])
            self.assertFalse((job/'4K.png').exists())
            self.assertEqual(len(list((job/'delivery-cache').iterdir())),1)

    def test_recovered_draft_discloses_dimensions_and_empty_job_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)
            folder=job/'rounds/0001'
            folder.mkdir(parents=True)
            deadlines.begin(job,'Umbrella','9:16')
            self.assertEqual(deadlines.manual_delivery(job,{},'timeout'),2)
            self.assertFalse((job/deadlines.MANUAL_FILE).exists())
            source=folder/'proof.png'
            Image.new('RGB',(90,160)).save(source)
            deadlines.checkpoint(folder,source,'draft')
            state={}
            self.assertEqual(deadlines.manual_delivery(job,state,'deadline'),3)
            self.assertEqual(state['manual_delivery']['size'],[2160,3840])
            self.assertEqual(state['manual_delivery']['source_size'],[90,160])

    def test_wedged_worker_is_terminated(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)
            folder=job/'rounds/0001'
            folder.mkdir(parents=True)
            clock={'run':'test','started':time.time(),'seconds':.45}
            canary=job/'late'
            command=[sys.executable,'-c',"import time, pathlib, sys; time.sleep(1); pathlib.Path(sys.argv[1]).touch()",str(canary)]
            with patch.object(deadlines,'stop_graphs') as stop:
                code,reason=deadlines.run_worker(command,job,folder,8193,clock)
            self.assertIsNone(code)
            self.assertIn('4K',reason)
            stop.assert_called_once_with(folder,8193)
            time.sleep(.65)
            self.assertFalse(canary.exists())

    def test_device_slot_excludes_another_process_and_allows_nesting(self):
        import model_slot
        with tempfile.TemporaryDirectory() as tmp, patch.object(model_slot,'RUNTIME',Path(tmp)):
            probe=Path(tmp)/'acquired'
            script="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import model_slot
model_slot.RUNTIME=Path(sys.argv[2])
with model_slot.slot():Path(sys.argv[3]).touch()
"""
            with model_slot.slot(),model_slot.slot():
                proc=subprocess.Popen([sys.executable,'-c',script,str(ROOT/'runtime/studio_pipeline'),tmp,str(probe)])
                with self.assertRaises(subprocess.TimeoutExpired):proc.wait(timeout=.25)
                self.assertFalse(probe.exists())
            self.assertEqual(proc.wait(timeout=5),0)
            self.assertTrue(probe.is_file())

    def test_album_counts_only_hash_verified_four_k_and_labels_manual_art(self):
        import batch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            job=root/'jobs/one'
            folder=job/'rounds/0001'
            for directory in (folder,root/'finals',root/'thumbnails'):directory.mkdir(parents=True)
            case={'id':'one','title':'Moon','prompt':'Moon','aspect':'16:9'}
            source=folder/'draft.png'
            Image.new('RGB',(256,144),'blue').save(source)
            deadlines.begin(job,'Moon','16:9')
            deadlines.checkpoint(folder,source,'draft')
            self.assertIsNone(batch.collect(root,case,job))
            state={}
            deadlines.manual_delivery(job,state,'timeout')
            deadlines.save(job/'production.json',state)
            row=batch.collect(root,case,job)
            batch.gallery(root,[case],{'one':row})
            progress=deadlines.read(root/'progress.json')
            self.assertEqual((progress['delivered'],progress['manual'],progress['approved']),(1,1,0))
            self.assertIn('Moon · '+deadlines.MANUAL,(root/'index.html').read_text())
            receipt=deadlines.read(job/'manual-review.json')
            receipt['sha256']='wrong'
            deadlines.save(job/'manual-review.json',receipt)
            self.assertIsNone(batch.collect(root,case,job))

    def test_album_has_fifty_unique_prompts_and_output_names(self):
        entries=deadlines.read(ROOT/'scripts/anime-portraits.json')
        self.assertEqual(len(entries),50)
        for field in ('id','prompt','title','seed'):
            self.assertEqual(len({x[field] for x in entries}),50)
        self.assertEqual({x['aspect'] for x in entries},{'16:9','9:16'})

    def test_resume_retries_only_exhausted_jobs_with_fresh_directories(self):
        import batch
        entries=[dict(id='x',prompt='night',title='Night',aspect='16:9',seed=1),
                 dict(id='y',prompt='day',title='Day',aspect='16:9',seed=2)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            deadlines.save(root/'prompts.json',entries)
            deadlines.save(root/'progress.json',{'cases':{
                'x':{'status':'failed','attempt':2,'job':'jobs/x-retry'},
                'y':{'status':'approved','attempt':1,'job':'jobs/y','file':'y.png','thumbnail':'y.jpg'}}})
            def complete(folder,cases,rows,case,attempt):
                return dict(status='manual',file='x.png',thumbnail='x.jpg')
            with patch.object(batch,'ALBUM',root),patch.object(batch,'collect',return_value={'status':'approved'}), \
                    patch.object(batch,'produce',side_effect=complete) as produce:
                batch.run()
            self.assertEqual(produce.call_count,1)
            self.assertEqual(produce.call_args.args[-1],3)
            self.assertEqual(deadlines.read(root/'progress.json')['delivered'],2)
