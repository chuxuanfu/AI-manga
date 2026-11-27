"""Two job-wide delivery deadlines and the latest complete full-frame recovery image."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib import request
import uuid

from PIL import Image, ImageOps, PngImagePlugin

SECONDS=15*60
MANUAL='需要人工 review'
MANUAL_FILE='4K-needs-human-review.png'
CLOCK='delivery-window.json'
LATEST='delivery-candidate.json'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def save(path,data):
    temp=path.with_name('.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def begin(job, prompt, aspect, *, reset=False, submitted=None):
    if not reset and (job/CLOCK).exists():
        return read(job/CLOCK)
    started=time.time() if submitted is None else min(time.time(),submitted)
    clock=dict(started=started,seconds=SECONDS,prompt=prompt,aspect=aspect,run=uuid.uuid4().hex)
    save(job/CLOCK,clock)
    return clock


def checkpoint(folder, path, stage, *, print_ready=False, observed_at=None):
    # Only complete camera canvases reach this hook. Detail crops, locator sheets,
    # comparison montages and partially written PNGs are never delivery candidates.
    job=folder.parent.parent
    clock=read(job/CLOCK)
    if not clock:return
    previous=read(job/LATEST)
    same_run=previous.get('run')==clock['run']
    ready=time.time() if observed_at is None else observed_at
    first=previous.get('first_4k_at') if same_run else None
    if ready>(first or clock['started'])+clock['seconds']:return
    path=Path(path)
    stat=path.stat()
    identity=[str(path.resolve()),stat.st_mtime_ns,stat.st_size]
    if same_run and previous.get('identity')==identity and (first or not print_ready):return
    with Image.open(path) as source:
        source.load()
        size=source.size
        if print_ready and max(size)<3840:
            raise ValueError('成品候选长边不足 3840')
        aspect=clock.get('aspect','auto')
        w,h=size if aspect=='auto' else map(int,aspect.split(':'))
        target=(round(w*3840/max(w,h)),round(h*3840/max(w,h)))
        art=source.convert('RGB')
        if art.size!=target:art=ImageOps.fit(art,target,method=Image.Resampling.LANCZOS)
    cache=job/'delivery-cache'
    cache.mkdir(exist_ok=True)
    name=uuid.uuid4().hex+'.png'
    out=cache/name
    info=PngImagePlugin.PngInfo()
    title=clock['prompt']+' · '+MANUAL
    info.add_text('Title',title)
    art.save(out,pnginfo=info,compress_level=1)
    made=time.time()
    if made>(first or clock['started'])+clock['seconds']:
        out.unlink();return
    current=dict(run=clock['run'],file=str(out.relative_to(job)),identity=identity,stage=stage,
                 source_size=size,size=target,title=title,prepared_at=made,first_4k_at=first or (ready if print_ready else None))
    save(job/LATEST,current)
    if previous.get('file') and (job/previous['file']).parent==cache:
        (job/previous['file']).unlink(missing_ok=True)


def boundary(job,clock):
    latest=read(job/LATEST)
    first=latest.get('first_4k_at') if latest.get('run')==clock['run'] else None
    return (first or clock['started'])+clock['seconds'], ('终审超过 15 分钟' if first else '生成 4K 超过 15 分钟')


def stop_graphs(folder,port):
    url=f'http://127.0.0.1:{port}'
    root=Path(__file__).resolve().parents[2]/'images'
    prefix=folder.relative_to(root).as_posix()+'/'
    try:
        with request.urlopen(url+'/queue',timeout=2) as reply:queue=json.load(reply)
        for kind,items in queue.items():
            for row in items:
                if not any(str(n['inputs'].get('filename_prefix','')).startswith(prefix) for n in row[2].values()):continue
                uri,body=('/interrupt',{'prompt_id':row[1]}) if kind=='queue_running' else ('/queue',{'delete':[row[1]]})
                req=request.Request(url+uri,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with request.urlopen(req,timeout=2):pass
    except OSError:
        pass


def stop_group(pid,folder,port):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        try:os.killpg(pid,sig)
        except ProcessLookupError:break
        except PermissionError:
            # Some macOS execution contexts disallow group signals but allow
            # signaling the specific child and its tools in the same session.
            import psutil
            for child in psutil.pids():
                try:
                    if os.getpgid(child)==pid:os.kill(child,sig)
                except (PermissionError,ProcessLookupError):continue
        if sig==signal.SIGTERM:time.sleep(.2)
    stop_graphs(folder,port)


def run_worker(command, job, folder, port, clock):
    """The parent stays responsive during a blocked GPU call, lock wait or slow VLM request."""
    until,reason=boundary(job,clock)
    if time.time()>=until:return None,reason
    proc=subprocess.Popen(command,start_new_session=True)
    try:
        while True:
            until,reason=boundary(job,clock)
            left=until-time.time()
            if left<=0:return None,reason
            try:
                code=proc.wait(timeout=min(.25,left))
            except subprocess.TimeoutExpired:continue
            until,reason=boundary(job,clock)
            return (code,None) if time.time()<=until else (None,reason)
    finally:
        if proc.poll() is None:
            stop_group(proc.pid,folder,port)
            proc.wait()


def manual_delivery(job,state,reason):
    from delivery import export_files
    clock=read(job/CLOCK)
    last=read(job/LATEST)
    state['delivery_reason']=reason
    if not last.get('file') or last.get('run')!=clock.get('run') or not (job/last['file']).is_file():
        state.update(status='已结束 · 没有可交付的完整画面')
        return 2
    source=job/last['file']
    dest=job/MANUAL_FILE
    temp=job/('.'+uuid.uuid4().hex+'.png')
    try:
        os.link(source,temp);temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    with dest.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=dict(status=MANUAL,title=last['title'],file=MANUAL_FILE,source=last['identity'][0],source_size=last['source_size'],
                 size=last['size'],sha256=digest,reason=reason,stage=last['stage'],started=clock['started'],
                 first_4k_at=last['first_4k_at'],delivered_at=time.time(),approved=False)
    save(job/'manual-review.json',receipt)
    state.update(status=MANUAL,manual_delivery=receipt)
    if state.get('output_dir'):
        state['exported_to']=export_files(state['output_dir'],job,{MANUAL_FILE:dest,'manual-review.json':job/'manual-review.json'})
    return 3
