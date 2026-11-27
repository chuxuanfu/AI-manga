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


def pipeline_identity():
    digest=hashlib.sha256()
    files=[ROOT/'Studio.command',ROOT/'scripts/studio.py',ROOT/'scripts/evaluate.py',ROOT/'scripts/models.json']
    for directory in ('runtime/studio_pipeline','extensions/manga_materials'):
        files.extend(sorted((ROOT/directory).glob('*.py')))
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode()+b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


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


def run(suite, label, budget, *, resume_after_update=False):
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
        version=pipeline_identity()
        upgrading=result.setdefault('pipeline_sha256',version)!=version
        if upgrading:
            if not resume_after_update:
                raise ValueError('The workflow changed. Start a new named run for a clean comparison, or use --resume-after-update to continue saved work.')
            history=dict(pipeline_sha256=result['pipeline_sha256'],cases=result['cases'],ended_at=time.time())
            result.setdefault('version_history',[]).append(history)
            result['pipeline_sha256']=version
            result['cases']={}
            for case in cases:
                name=case['id']
                prior=history['cases'].get(name,{})
                if prior.get('status')=='passed' and verified_release(folder/name,case['prompt']):
                    result['cases'][name]=dict(prior,verified_pipeline_sha256=prior.get('verified_pipeline_sha256',history['pipeline_sha256']))
                    continue
                state_file=folder/name/'production.json'
                state=read(state_file) if state_file.is_file() else {}
                finished=sum(r['status'] not in ('运行中','运行中断') for r in state.get('rounds',[]))
                result['cases'][name]=dict(status='queued',job=name,completed_before_update=finished)
        for case in cases:
            prior=result['cases'].setdefault(case['id'],dict(status='queued',job=case['id']))
            if prior['status']=='error':
                prior['status']='queued'
        store(report_file,result)
        if upgrading:
            def priority(case):
                from production import held_workmanship_pair
                job=folder/case['id']
                rounds=sorted(job.glob('rounds/[0-9]*'))
                if rounds and held_workmanship_pair(rounds[-1],case['prompt']):return 0
                state=history['cases'].get(case['id'],{}).get('status','queued')
                return 1 if state=='running' else (2 if state=='queued' else 3)
            cases=sorted(cases,key=priority)
        for turn in range(budget):
            for case in cases:
                name=case['id']; prior=result['cases'].get(name,{})
                if prior.get('status') in ('passed','budget_exhausted','error'):
                    continue
                job=folder/name;job.mkdir(exist_ok=True)
                state=read(job/'production.json') if (job/'production.json').exists() else {}
                rounds=state.get('rounds',[])
                used=sum(row['status'] not in ('运行中','运行中断') for row in rounds)
                baseline=prior.get('completed_before_update',0)
                used-=baseline
                started=prior.get('started_at',time.time())
                result['cases'][name]=dict(status='running',started_at=started,job=name,completed_before_update=baseline)
                store(report_file,result)
                rc=None
                if used<budget and state.get('status')!='已自动定稿':
                    cmd=[str(ROOT/'Studio.command'),'--job',str(job),'--max-rounds','1']
                    if not state:
                        cmd+=['--prompt',case['prompt'],'--seed',str(case['seed']),'--aspect',case['aspect']]
                    with (job/'benchmark.log').open('ab') as log:
                        rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
                state=read(job/'production.json') if (job/'production.json').exists() else {}
                passed=state.get('status')=='已自动定稿' and verified_release(job,case['prompt'])
                exhausted=state.get('status')=='达到指定轮数，本次未产生新定稿'
                used=sum(row['status'] not in ('运行中','运行中断') for row in state.get('rounds',[]))
                status='passed' if passed else 'error'
                if exhausted:
                    status='budget_exhausted' if used-baseline>=budget else 'retry_pending'
                result['cases'][name]=dict(status=status,job=name,started_at=started,elapsed_seconds=round(time.time()-started,1),
                                           rounds=len(state.get('rounds',[])),exit_code=rc,completed_before_update=baseline,
                                           verified_pipeline_sha256=version if passed else None,
                                           last_finding=(state.get('rounds') or [{}])[-1].get('原因',''))
                if pipeline_identity()!=version:
                    result['cases'][name]['status']='workflow_changed'
                    store(report_file,result)
                    raise RuntimeError('Workflow source changed during execution. Use a new run for a clean comparison.')
                store(report_file,result)
    return report_file


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,default=ROOT/'tests/fixtures/unseen-scenes.json')
    parser.add_argument('--run',required=True)
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--resume-after-update',action='store_true',help='Keep locally generated checkpoints and record the previous workflow version. Omit for a single-version benchmark.')
    args=parser.parse_args()
    report=run(args.suite,args.run,args.rounds,resume_after_update=args.resume_after_update)
    print(report, flush=True)
    sys.exit(0 if all(case['status']=='passed' for case in read(report)['cases'].values()) else 2)
