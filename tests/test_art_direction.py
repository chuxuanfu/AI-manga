import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from contextlib import nullcontext

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/studio_pipeline'))
import art_direction as art
from production import workmanship_confirmed

class ArtDirectionTests(unittest.TestCase):
    def setUp(self):
        self.evaluation={'policy':art.POLICY,'reference_sha256':art.hashlib.sha256(art.REFERENCE.read_bytes()).hexdigest(),
                         'model_id':'local','reading':dict(brief_and_structure='sound',overall_finish='exhibition_ready',
                         preference=91,structure_evidence='One continuous figure by the window.',cohesion='Window light reaches the sleeve.',
                         focal_hierarchy='Quiet edges behind the subject.',surface_and_edges='Fabric absorbs light.',direction='')}

    def test_automatic_anime_paint_respects_the_requested_medium(self):
        self.assertTrue(art.anime_finish(''))
        self.assertTrue(art.anime_finish('Japanese anime illustration'))
        self.assertFalse(art.anime_finish('woodblock print'))
        self.assertFalse(art.anime_finish('watercolor anime illustration'))

    def test_global_grade_and_publication_are_bound_to_a_sound_exact_master(self):
        def evaluate():
            result=art.assessment_result(self.evaluation,'one girl in a station')
            record={'art_policy':art.POLICY,'工作流':'reference-illustration-v1',
                    'reviews':[dict(hash='abc',stage='final-art',review=result)]}
            self.assertFalse(workmanship_confirmed(record,'different'))
            return result['status'],workmanship_confirmed(record,'abc')
        self.assertEqual(evaluate(),('pass',True))
        for field,value in [('overall_finish','needs_art_direction'),('brief_and_structure','major_failure'),
                            ('brief_and_structure','uncertain'),('cohesion','')]:
            with patch.dict(self.evaluation['reading'],{field:value}):
                self.assertEqual(evaluate(),('global',False))
        with patch.dict(self.evaluation,{'reference_sha256':'stale'}):
            self.assertEqual(evaluate(),('global',False))

    def test_major_faults_remain_blocking_and_uncertainty_is_not_cosmetic(self):
        import json
        report=dict(status='local',issue=dict(operation='redraw',box=[100,200,400,500],symptom='Check the rim of the cup'))
        with TemporaryDirectory() as root:
            path=Path(root)/'master.png';Image.new('RGB',(200,200)).save(path)
            with patch('art_direction.backend',return_value=nullcontext('local')),patch('art_direction.ask') as reader:
                for impact in ('major','uncertain','cosmetic'):
                    reader.return_value=json.dumps(dict(impact=impact,evidence='A single stray brushstroke.'))
                    note=art.retouch_note(path,'A cafe',report)
                    self.assertEqual(note['impact'] if note else None,None if impact=='major' else impact)
                    self.assertEqual(len(reader.call_args.args[1]),2)
                with patch.dict(report['issue'],{'operation':'remove_forbidden'}):
                    self.assertIsNone(art.retouch_note(path,'A cafe',report))
                self.assertEqual(reader.call_count,3)
                reader.side_effect=RuntimeError('reader timed out')
                self.assertEqual(art.retouch_note(path,'A cafe',report)['impact'],'uncertain')

    def test_uncertain_focus_crop_reaches_reviewer_without_an_issue_box(self):
        report=dict(status='uncertain',size=[800,400],
                    issue=dict(category='unresolved_focus',operation='none',symptom='Soft reflection'),
                    views=[dict(view='comparison',pixels=[320,100,480,200])])
        with TemporaryDirectory() as root:
            path=Path(root)/'master.png'
            image=Image.new('RGB',(800,400))
            image.paste('red',(320,100,480,200))
            image.save(path)
            with patch('art_direction.backend',return_value=nullcontext('local')),patch('art_direction.ask') as reader:
                reader.return_value='{"impact":"cosmetic","evidence":"Intentional soft reflection."}'
                note=art.retouch_note(path,'Reflections in rain',report)
                self.assertEqual(note['box'],[400,250,600,500])
                self.assertEqual(reader.call_args.args[1][1].getextrema(),((255,255),(0,0),(0,0)))
                with patch.dict(report,{'size':[400,800]}):
                    self.assertIsNone(art.retouch_note(path,'Reflections in rain',report))
                with patch.dict(report,{'status':'global'}):
                    self.assertIsNone(art.retouch_note(path,'Reflections in rain',report))
                reader.assert_called_once()

    def test_retouch_delivery_preserves_pixels_and_explains_why_in_receipt(self):
        import json
        from PIL import ImageChops
        from deadlines import MANUAL_FILE, RETOUCH, begin, checkpoint, manual_delivery
        with TemporaryDirectory() as root:
            job=Path(root)
            folder=job/'rounds/0001';folder.mkdir(parents=True)
            master=folder/'master.png'
            Image.new('RGB',(3840,2160),'steelblue').save(master)
            begin(job,'Afternoon by the sea','16:9')
            checkpoint(folder,master,'final-art',print_ready=True)
            note=dict(box=[450,500,510,540],impact='cosmetic',evidence='袖口有短杂线')
            self.assertEqual(manual_delivery(job,{},RETOUCH,retouch=note),3)
            receipt=json.loads((job/'manual-review.json').read_text())
            self.assertEqual(receipt['retouch'],note)
            self.assertFalse(receipt['approved'])
            with Image.open(master) as original,Image.open(job/MANUAL_FILE) as delivered:
                self.assertIsNone(ImageChops.difference(original,delivered).getbbox())
