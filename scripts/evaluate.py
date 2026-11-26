"""Run an unseen prompt suite through Studio.command and verify each released 4K file."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'runtime/studio_pipeline'), str(ROOT/'extensions')]


def read(path):
    return json.loads(path.read_text())


def verified_release(job, prompt):
    from PIL import Image
    from production import workmanship_confirmed
    from print_judge import complete_print_review, fingerprint
    production = read(job/'production.json')
    if production.get('status')!='已自动定稿' or production.get('prompt')!=prompt:
        return False
    record = read(job/production['selected']/'review.json')
    receipt = read(job/'final/review-receipt.json')
    report_file = job/'final/tile-inspection.json'
    report = read(report_file)
    master = job/'4K.png'
    with Image.open(master) as image:
        size = image.size
    sha = fingerprint(master)
    return (record.get('执行模式')=='离线自动定稿' and max(size)==3840
            and tuple(report.get('size', ()))==size and sha==receipt['4K指纹']==report['sha256']
            and fingerprint(report_file)==receipt['报告指纹'] and complete_print_review(report)
            and workmanship_confirmed(record,sha))


def run(suite, label, budget):
    from print_judge import store
    cases = read(suite)
    ids = [case['id'] for case in cases]
    if (not cases or len(set(ids))!=len(ids) or not all(re.fullmatch('[a-zA-Z0-9_-]+',x) for x in [label,*ids])
            or budget<1):
        raise ValueError('Choose a positive round budget and unique filesystem-safe case IDs.')
    for case in cases:
        if (not isinstance(case['prompt'],str) or not case['prompt'].strip() or type(case['seed']) is not int
                or case['aspect'] not in ('2:3','3:2','16:9','9:16','1:1')):
            raise ValueError('Each case needs a prompt, integer seed, and supported aspect ratio.')
    folder = ROOT/'images/benchmarks'/label
    folder.mkdir(parents=True,exist_ok=True)
    report_file = folder/'results.json'
    key = hashlib.sha256(suite.read_bytes()).hexdigest()
    with (folder/'.evaluation-lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result = read(report_file) if report_file.exists() else dict(suite_sha256=key,max_rounds=budget,cases={})
        if (result['suite_sha256'],result['max_rounds'])!=(key,budget):
            raise ValueError('Use a new run name when the suite or round budget changes.')
        for case in cases:
            name=case['id']; prior=result['cases'].get(name,{})
            if prior.get('status') in ('passed','budget_exhausted'):
                continue
            job=folder/name;job.mkdir(exist_ok=True)
            state=read(job/'production.json') if (job/'production.json').exists() else {}
            rounds=state.get('rounds',[])
            used=sum(row['status'] not in ('运行中','运行中断') for row in rounds)
            started=prior.get('started_at',time.time())
            result['cases'][name]=dict(status='running',started_at=started,job=name)
            store(report_file,result)
            rc=None
            if used<budget and state.get('status')!='已自动定稿':
                cmd=[str(ROOT/'Studio.command'),'--job',str(job),'--max-rounds',str(budget-used)]
                if not state:
                    cmd+=['--prompt',case['prompt'],'--seed',str(case['seed']),'--aspect',case['aspect']]
                with (job/'benchmark.log').open('ab') as log:
                    rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            state=read(job/'production.json') if (job/'production.json').exists() else {}
            passed=state.get('status')=='已自动定稿' and verified_release(job,case['prompt'])
            exhausted=state.get('status')=='达到指定轮数，本次未产生新定稿'
            status='passed' if passed else ('budget_exhausted' if exhausted else 'error')
            result['cases'][name]=dict(status=status,job=name,started_at=started,elapsed_seconds=round(time.time()-started,1),
                                       rounds=len(state.get('rounds',[])),exit_code=rc,
                                       last_finding=(state.get('rounds') or [{}])[-1].get('原因',''))
            store(report_file,result)
    return report_file


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,default=ROOT/'tests/fixtures/unseen-scenes.json')
    parser.add_argument('--run',required=True)
    parser.add_argument('--rounds',type=int,default=1)
    args=parser.parse_args()
    print(run(args.suite,args.run,args.rounds))
