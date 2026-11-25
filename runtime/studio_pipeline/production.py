"""Own unattended job retries and publish only an immutable reviewed image pair."""
import fcntl
import html
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from urllib.parse import quote
from print_judge import MODEL, fingerprint, model_identity, print_windows, store


def link(path,target):
    tmp=path.with_name('.link-'+uuid.uuid4().hex)
    try:
        tmp.symlink_to(target)
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)


def publish(job, source, record):
    master=source/record['8K']
    small=source/record['4K']
    checked=record['自动定稿']
    if fingerprint(master)!=checked['sha256'] or fingerprint(small)!=checked['4K指纹']:
        raise ValueError('定稿文件在终审后发生变化')
    report_file=source/checked['终审报告']
    report=json.loads(report_file.read_text())
    expected={rect for kind,rect in print_windows(tuple(report['size']),{}) if kind=='native'}
    inspected={tuple(row['pixels']) for row in report['views'] if row['view']=='native'}
    unresolved=any((row.get('independent_context') or row.get('corroboration') or row['observation']).get('decision')!='clear' for row in report['views'])
    if (unresolved or report['status']!='pass' or report['sha256']!=checked['sha256'] or
        report['comparison_sha256']!=checked['4K指纹'] or report['model_id']!=checked['模型指纹'] or
        len(report['views'])!=report['planned'] or expected!=inspected):
        raise ValueError('终审凭据未能验证此图的完整覆盖')
    receipt=fingerprint(report_file)
    selected=job/'approved-versions'/(checked['sha256']+'-'+receipt[:12])
    selected.parent.mkdir(exist_ok=True)
    if not selected.is_dir():
        staging=selected.parent/('.'+uuid.uuid4().hex)
        staging.mkdir()
        try:
            for path,name in ((master,'8K.png'),(small,'4K.png')):
                shutil.copyfile(path,staging/name)
            if fingerprint(staging/'8K.png')!=checked['sha256'] or fingerprint(staging/'4K.png')!=checked['4K指纹']:
                raise ValueError('定稿写入校验失败')
            shutil.copyfile(report_file,staging/'tile-inspection.json')
            store(staging/'review-receipt.json',{**checked,'终审报告':'tile-inspection.json','报告指纹':receipt})
            staging.rename(selected)
        finally:
            if staging.exists():shutil.rmtree(staging)
    if fingerprint(selected/'8K.png')!=checked['sha256'] or fingerprint(selected/'4K.png')!=checked['4K指纹'] or fingerprint(selected/'tile-inspection.json')!=receipt:
        raise ValueError('原存档内容与通过终审的版本不一致')
    link(job/'final',selected.relative_to(job))
    for name in ('4K.png','8K.png'):link(job/name,Path('final')/name)


def export_release(job,state):
    if state.get('output_dir'):
        from delivery import export_files
        files={name:job/'final'/name for name in ('4K.png','8K.png','tile-inspection.json','review-receipt.json')}
        state['exported_to']=export_files(state['output_dir'],job,files)


def dashboard(job,state):
    parts=['<!doctype html><meta charset="utf-8"><title>离线出片</title>',
           '<style>body{max-width:58rem;margin:4rem auto;padding:0 1rem;font:18px/1.65 system-ui;background:#f7f5ef;color:#272923}a{color:#305647}li{margin:1rem 0}</style>',
           '<h1>离线出片</h1><p>'+html.escape(state['prompt'])+'</p><strong>'+html.escape(state['status'])+'</strong>']
    if (job/'final').exists():
        parts.append('<p><a href="4K.png">打开 4K</a>　<a href="8K.png">打开 8K</a>　<a href="final/review-receipt.json">终审凭据</a>　<a href="final/tile-inspection.json">逐块目检记录</a></p>')
    parts.append('<ol>')
    for row in state['rounds']:
        loc=Path(row['directory'])
        item=html.escape(row['status'])
        if row.get('原因'):item+=' · '+html.escape(row['原因'])
        if (job/loc/'proofs.html').is_file():
            item+='　<a href="'+quote((loc/'proofs.html').as_posix())+'">检查记录与局部对照</a>'
        parts.append('<li>'+item+'</li>')
    tmp=job/'proofs.tmp'
    tmp.write_text(''.join(parts)+'</ol>')
    tmp.replace(job/'proofs.html')


def save(job,state):
    store(job/'production.json',state)
    dashboard(job,state)
    from library import IMAGES,refresh
    if job.parent==IMAGES/'new-jobs':
        try:refresh()
        except OSError as exc:print('图片库刷新失败',exc,flush=True)


def run(args,job,entry):
    """Each round has its own graph cache and seed; an interrupted round resumes in place."""
    job.mkdir(parents=True,exist_ok=True)
    with (job/'.production-lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('这个离线出片任务正在运行。')
        manifest=job/'production.json'
        if manifest.exists():
            state=json.loads(manifest.read_text())
            if args.prompt and args.prompt.strip()!=state['prompt']:raise SystemExit('新提示词需要独立任务目录。')
        else:
            if not args.prompt or not args.prompt.strip():raise SystemExit('新任务需要 --prompt')
            options=[]
            for key in ('native_4k','full_8k_diffusion','keep_native'):
                if getattr(args,key):options.append('--'+key.replace('_','-'))
            for key in ('reviewed_image','candidate_image','selection_mask'):
                if getattr(args,key):options.extend(['--'+key.replace('_','-'),str(getattr(args,key).resolve())])
            for source in args.from_image:options.extend(['--from-image',str(source.resolve())])
            if args.repair_box:options+=['--repair-box',*map(str,args.repair_box),'--repair-text',args.repair_text]
            if args.object:options+=['--object',args.object]
            if getattr(args,'aspect','auto')!='auto':options+=['--aspect',args.aspect]
            if args.candidates:options+=['--candidates',str(args.candidates)]
            state=dict(prompt=args.prompt.strip(),seed=args.seed if args.seed is not None else secrets.randbelow(2**45),
                       options=options,rounds=[],status='准备离线出片',port=args.port)
        if getattr(args,'output_dir',None):state['output_dir']=str(args.output_dir)
        if args.accept_final:raise SystemExit('自动任务只接收本机模型通过的版本。')
        if not model_identity(MODEL):raise SystemExit('本机缺少终审模型 '+MODEL)
        edit=args.skip_focus is not None
        if edit and not state['rounds']:raise SystemExit('还没有可调整的精绘区域')
        if state.get('status')=='已自动定稿' and not edit:
            folder=job/state['selected']
            record=json.loads((folder/'review.json').read_text())
            publish(job,folder,record)
            export_release(job,state);save(job,state)
            print('已自动定稿',job/'final',flush=True);return 0
        print('离线出片目录',job,flush=True)
        attempts=0
        while args.max_rounds==0 or attempts<args.max_rounds:
            rounds=state['rounds']
            if edit or (rounds and rounds[-1]['status'] in ('运行中','运行中断')):
                row=rounds[-1]
            else:
                index=len(rounds)+1
                row=dict(directory=f'rounds/{index:04}',seed=(state['seed']+10007*(index-1))%(2**63-1))
                rounds.append(row)
            folder=job/row['directory']
            options=state['options']
            if edit:row['skip_focus']=args.skip_focus
            row['status']='运行中';state['status']='离线审图与返修中'
            save(job,state)
            command=[sys.executable,str(entry),'--round-worker','--job',str(folder),'--prompt',state['prompt'],
                     '--port',str(state['port']),'--seed',str(row['seed']),*options]
            if len(rounds)>1 and rounds[-2].get('原因'):
                command+=['--retry-observation',rounds[-2]['原因'][:700]]
            if 'skip_focus' in row:command+=['--skip-focus',*map(str,row['skip_focus'])]
            for retry in range(3):
                if retry:
                    row['status']='运行中';state['status']='离线审图与返修中'
                    save(job,state)
                code=subprocess.run(command).returncode
                outcome=folder/'review.json'
                record=json.loads(outcome.read_text()) if outcome.is_file() else {}
                if code==0 and record.get('status')=='已通过离线终审':break
                if code==2 and record.get('status','').startswith('未达标'):break
                row['status']='运行中断';state['status']='本地运行中断，正在恢复' if retry<2 else '本地运行故障，已保存进度'
                save(job,state)
                if retry==2:return 1
                time.sleep(3*(retry+1))
            row['status']=record['status']
            if record.get('终审退回原因'):row['原因']=record['终审退回原因']
            if code==0:
                publish(job,folder,record)
                export_release(job,state)
                state.update(status='已自动定稿',selected=row['directory'],sha256=record['自动定稿']['sha256'])
                save(job,state);print('已自动定稿',job/'final',flush=True);return 0
            attempts+=1;edit=False
            state['status']='本轮未达标，自动重试'
            save(job,state)
        state['status']='达到指定轮数，本次未产生新定稿'
        save(job,state);print(state['status'],flush=True);return 2
