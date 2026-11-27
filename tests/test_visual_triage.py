import json
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/studio_pipeline'),str(ROOT / 'extensions')]
from visual_triage import review_image


class OpenReviewTests(unittest.TestCase):
    def exercise(self, votes, verification):
        image = Image.new('RGB',(900,1400))
        def reading(vote):
            label, severity = vote
            return json.dumps(dict(observation=label,consequence='test',severity=severity,
                                   left=200,top=100,right=800,bottom=900))
        replies = [reading(v) for v in votes]
        with TemporaryDirectory() as folder, patch('visual_triage.model_identity',return_value='local'), \
                patch('visual_triage.backend',return_value=nullcontext('local')), \
                patch('visual_triage.ask',side_effect=replies) as ask, \
                patch('visual_triage.verify',return_value=verification) as verify:
            first=review_image(image,'brief','abc',Path(folder))
            self.assertEqual(first,review_image(image,'brief','abc',Path(folder)))
            self.assertEqual(ask.call_count,7)
            return first,verify.call_count

    def test_late_unanticipated_blocker_outranks_earlier_cosmetic_vote(self):
        votes=[('minor lettering','minor')]+[('clear','clear')]*5+[('clock shows mutually incompatible shadows','major')]
        verdict=dict(observed='sunlit shadows diverge',deciding_visual_cue='sun is single',
                     decision='substantiated',fix='Align the cast shadows to the same light.')
        result,count=self.exercise(votes,verdict)
        self.assertEqual(count,1)
        self.assertEqual(result['issue']['target'],votes[-1][0])
        self.assertEqual(result['issue']['box'],[568,676,892,964])

    def test_explained_stylization_allows_normal_review_to_continue(self):
        votes=[('impossible outline','major')]+[('clear','clear')]*6
        result,_=self.exercise(votes,dict(decision='explained'))
        self.assertEqual(result['status'],'clear')
        self.assertIsNone(result['issue'])

    def test_uncorroborated_issue_is_held_without_editing(self):
        votes=[('possible contradiction','uncertain')]+[('clear','clear')]*6
        result,_=self.exercise(votes,dict(decision='insufficient',fix='none',observed='unclear',deciding_visual_cue='occluded'))
        self.assertEqual(result['status'],'uncertain')
        self.assertEqual(result['issue']['operation'],'reinspect')


class ConfirmedEvidenceTests(unittest.TestCase):
    def test_unrelated_retouch_cannot_discharge_a_confirmed_issue(self):
        from confirmed_issues import confirmed_issue, ledger_version, register
        from print_judge import fingerprint
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/'issues'
            art=Image.effect_noise((200,200),50).convert('RGB')
            source=root/'source.png'
            art.save(source)
            before=ledger_version(folder)
            register(source,[100,200,450,650],'join reads inconsistently','Clarify both visible surfaces.',folder)
            self.assertNotEqual(before,ledger_version(folder))
            self.assertIsNotNone(confirmed_issue(art,fingerprint(source),folder))
            art.putpixel((180,180),(255,0,0))
            self.assertIsNotNone(confirmed_issue(art,'changed elsewhere',folder))
            art.paste((255,0,0),(25,48,65,105))
            self.assertIsNone(confirmed_issue(art,'changed inside',folder))
