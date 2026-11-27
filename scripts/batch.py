"""Run a restartable album on one device and publish only verified batch artifacts."""
import fcntl
import hashlib
import html
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote

from studio import ROOT,PYTHON,IMAGES
from evaluate import verified_release
from deadlines import MANUAL,MANUAL_FILE,LATEST,read,save
from PIL import Image
from illustration import KEY_VISUAL

ALBUM=IMAGES/'anime-wallpapers-50'
CATALOG=ROOT/'scripts/anime-portraits.json'


def gallery(folder, cases, rows):
    counts={label:sum(r.get('status')==label for r in rows.values()) for label in ('approved','manual','failed','running')}
    tiles=[]
    for case in cases:
        row=rows.get(case['id'],{})
        title=case['title']
        if row.get('status')=='manual':title+=' · '+MANUAL
        text=html.escape(title)
        if row.get('file'):
            thumb=quote(row['thumbnail'])
            tiles.append(f'<a href="{quote(row["file"])}"><img loading="lazy" src="{thumb}" alt=""><strong>{text}</strong></a>')
        elif row.get('status')=='running':
            stage=html.escape(row.get('stage','开始本地制作'))
            tiles.append(f'<p><strong>{text}</strong> · {stage}</p>')
    finished=counts['approved']+counts['manual']
    report=dict(total=len(cases),delivered=finished,**counts,updated=time.time(),cases=rows)
    save(folder/'progress.json',report)
    refresh='<meta http-equiv="refresh" content="30">' if finished+counts['failed']<len(cases) else ''
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}<meta name="viewport" content="width=device-width"><title>少女插画壁纸 · 50 张</title>
<style>body{{margin:36px auto;padding:0 22px;max-width:1150px;font:16px/1.6 system-ui;background:#f5f4ef;color:#24372d}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:18px}}a{{color:#245746}}img{{width:100%;height:180px;object-fit:contain;background:#e5e6de;display:block;margin-bottom:6px}}a:focus-visible{{outline:3px solid #c86}}</style>
<h1>少女插画壁纸</h1><p>已交付 {finished}/{len(cases)} · 通过终审 {counts["approved"]} · 需要人工 review {counts["manual"]} · 失败 {counts["failed"]}</p><p><a href="prompts.json">50 条提示词</a> · <a href="progress.json">逐张状态</a></p><main>{''.join(tiles)}</main></html>'''
    temp=folder/'index.tmp';temp.write_text(page);temp.replace(folder/'index.html')


def collect(folder, case, job):
    state=read(job/'production.json')
    manual=state.get('status')==MANUAL
    if manual:
        receipt=read(job/'manual-review.json')
        art=job/MANUAL_FILE
        if (receipt.get('approved') is not False or receipt.get('file')!=MANUAL_FILE
                or receipt.get('status')!=MANUAL or not art.is_file()):return None
        with art.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=receipt.get('sha256'):return None
    else:
        if state.get('status')!='已自动定稿' or not verified_release(job,case['prompt']+' · '+KEY_VISUAL):return None
        art=job/'4K.png'
    a,b=map(int,case['aspect'].split(':'))
    name=case['id']+('-needs-human-review' if manual else '')
    final=folder/'finals'/(name+'.png')
    thumb=folder/'thumbnails'/(name+'.jpg')
    with Image.open(art) as im:
        if im.size!=(round(3840*a/max(a,b)),round(3840*b/max(a,b))):return None
        im.thumbnail((540,400))
        temp=thumb.with_suffix('.tmp')
        im.convert('RGB').save(temp,format='JPEG',quality=85);temp.replace(thumb)
    temp=final.with_suffix('.tmp')
    temp.unlink(missing_ok=True)
    os.link(art.resolve(),temp);temp.replace(final)
    return dict(status='manual' if manual else 'approved',file=str(final.relative_to(folder)),
                thumbnail=str(thumb.relative_to(folder)),reason=state.get('delivery_reason',''))


def monitor(folder, cases, rows, key, job):
    """Expose meaningful stage changes even while a detached producer owns the job."""
    def update():
        state=read(job/'production.json')
        rounds=state.get('rounds') or []
        review=read(job/rounds[-1]['directory']/'review.json') if rounds else {}
        latest=read(job/LATEST)
        stage=review.get('status') or state.get('status') or '正在启动'
        stamp=dict(stage=stage,latest_canvas=latest.get('stage'),first_4k_at=latest.get('first_4k_at'))
        if any(rows[key].get(k)!=v for k,v in stamp.items()):
            rows[key].update(stamp)
            gallery(folder,cases,rows)
    return update


def produce(folder, cases, rows, case, attempt):
    key=case['id']
    suffix='' if attempt==1 else ('-retry' if attempt==2 else f'-retry-{attempt}')
    job=folder/'jobs'/(key+suffix)
    job.mkdir(exist_ok=True)
    prev=rows.get(key,{})
    started=prev.get('started',time.time()) if prev.get('status')=='running' else time.time()
    rows[key]=dict(status='running',attempt=attempt,job=str(job.relative_to(folder)),started=started)
    update=monitor(folder,cases,rows,key,job)
    update()
    guard=job/'.production-lock'
    if guard.exists():
        with guard.open('a') as existing:
            while True:
                try:
                    fcntl.flock(existing,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    update()
                    time.sleep(3)
    result=collect(folder,case,job)
    if result:return result
    command=[str(PYTHON),str(ROOT/'scripts/studio.py'),'--job',str(job),'--port','8193','--max-rounds','1',
             '--max-redraws','2','--candidates','1','--aspect',case['aspect'],'--seed',str(case['seed']),
             '--style','anime-key-visual','--prompt',case['prompt']]
    with (job/'run.log').open('ab') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            update()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:pass
    return collect(folder,case,job)


def run():
    folder=ALBUM
    for sub in ('jobs','finals','thumbnails'):(folder/sub).mkdir(parents=True,exist_ok=True)
    with (folder/'.batch-lock').open('a') as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('画集正在本地串行制作',folder/'index.html',flush=True)
            return
        suite=folder/'prompts.json'
        if not suite.exists():shutil.copyfile(CATALOG,suite)
        cases=read(suite)
        rows=read(folder/'progress.json').get('cases',{})
        for case in cases:
            row=rows.get(case['id'],{})
            if row.get('status') in ('manual','approved'):
                try:verified=collect(folder,case,folder/row['job'])
                except (KeyError,ValueError,OSError):verified=None
                if not verified:rows[case['id']]=dict(status='retry',attempt=1)
        gallery(folder,cases,rows)
        # A new invocation can retry jobs that previously exhausted both attempts.
        # Give them fresh deadlines and keep every earlier job directory for diagnosis.
        starts={}
        for case in cases:
            row=rows.get(case['id'],{})
            prior=row.get('attempt',1)
            starts[case['id']]=prior+1 if row.get('status')=='failed' else 1+2*((prior-1)//2)
        interrupted=[(case,rows[case['id']]['attempt']) for case in cases
                     if rows.get(case['id'],{}).get('status')=='running']
        for case,attempt in interrupted+[(case,starts[case['id']]+offset) for offset in (0,1) for case in cases]:
            key=case['id']
            prev=rows.get(key,{})
            if prev.get('status') in ('approved','manual'):continue
            if prev.get('attempt',0)>attempt or (prev.get('attempt')==attempt and prev.get('status') in ('retry','failed')):continue
            try:
                result=produce(folder,cases,rows,case,attempt)
            except (OSError,ValueError,KeyError) as exc:
                result=None
                rows.setdefault(key,{})['error']=str(exc)
            rows[key].update(result or dict(status='retry' if attempt==starts[key] else 'failed'),ended=time.time())
            gallery(folder,cases,rows)
            if not result and not (folder/rows[key]['job']/'production.json').exists():
                print(key,'启动失败，见该任务 run.log；暂停一分钟后继续。',flush=True)
                time.sleep(60)
        print('画集任务结束',folder/'index.html',flush=True)


if __name__=='__main__':run()
