import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import evaluate


class BenchmarkRotationTests(unittest.TestCase):
    def test_failed_round_yields_to_unseen_cases_and_exhaustion_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root/'suite.json'
            suite.write_text(json.dumps([dict(id=name, prompt=name, seed=5, aspect='1:1')
                                         for name in ('one','two','three')]))
            calls = []
            def complete(command, **kwargs):
                job = Path(command[command.index('--job')+1])
                self.assertEqual(command[command.index('--max-rounds')+1],'1')
                manifest = job/'production.json'
                count = len(evaluate.read(manifest)['rounds']) if manifest.exists() else 0
                self.assertEqual('--prompt' in command, count == 0)
                calls.append(job.name)
                passed = job.name=='two'
                state = dict(status='已自动定稿' if passed else '达到指定轮数，本次未产生新定稿',
                             rounds=[dict(status='已通过离线终审' if passed else '未达标')] * (count+1))
                manifest.write_text(json.dumps(state))
                return evaluate.subprocess.CompletedProcess(command, 0 if passed else 2)
            with (patch.object(evaluate,'ROOT',root),
                  patch.object(evaluate,'pipeline_identity',return_value='fixture'),
                  patch.object(evaluate.subprocess,'run',side_effect=complete),
                  patch.object(evaluate,'verified_release',return_value=True) as verify):
                report = evaluate.run(suite,'rotation',2)
                self.assertEqual(calls,['one','two','three','one','three'])
                cases=evaluate.read(report)['cases']
                self.assertEqual([case['status'] for case in cases.values()],
                                 ['budget_exhausted','passed','budget_exhausted'])
                verify.assert_called_once_with(report.parent/'two','two')
