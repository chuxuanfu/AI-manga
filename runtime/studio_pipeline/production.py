"""Own unattended job retries and publish only an immutable reviewed release."""
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
from print_judge import MODEL, REVISION, fingerprint, model_identity, complete_print_review, store


def link(path,target):
    tmp=path.with_name('.link-'+uuid.uuid4().hex)
    try:
        tmp.symlink_to(target)
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)


def workmanship_confirmed(record, digest):
    """Bind a reference-style release to the final assessment on these exact pixels."""
    if record.get('工作流')!='reference-illustration-v1':
        return True
    for row in reversed(record.get('reviews',[])):
        if row.get('hash')==digest and row.get('stage','').endswith('final-art'):
            review=row.get('review',{})
            from reference_review import confirmed_final_comparison
            return review.get('status')=='pass' and confirmed_final_comparison(review.get('参考比较') or {})
    return False


def publish(job, source, record):
    checked=record['自动定稿']
    if not workmanship_confirmed(record,checked['sha256']):
        raise ValueError('此母版的参考画质对照尚未通过')
    outputs={'4K.png':(source/record['4K'],checked['4K指纹'])}
    if record.get('8K'):
        outputs['8K.png']=(source/record['8K'],checked['sha256'])
    elif checked['sha256']!=checked['4K指纹']:
        raise ValueError('4K 母版指纹与终审记录不一致')
    if any(fingerprint(path)!=digest for path,digest in outputs.values()):
        raise ValueError('定稿文件在终审后发生变化')
    aspect=record.get('aspect','auto')
    if aspect and aspect!='auto':
        from PIL import Image
        a,b=map(int,aspect.split(':'))
        for name,(path,_) in outputs.items():
            edge=3840 if name=='4K.png' else 7680
            wanted=(round(a*edge/max(a,b)),round(b*edge/max(a,b)))
            with Image.open(path) as actual:
                if actual.size!=wanted:raise ValueError('定稿尺寸与指定画幅不符')
    report_file=source/checked['终审报告']
    report=json.loads(report_file.read_text())
    if max(report['size'])!=(7680 if '8K.png' in outputs else 3840):
        raise ValueError('终审像素尺寸与发布规格不符')
    if (not complete_print_review(report) or report['sha256']!=checked['sha256'] or
        report['comparison_sha256']!=checked.get('对照指纹',checked['4K指纹']) or report['model_id']!=checked['模型指纹']):
        raise ValueError('终审凭据未能验证此图的完整覆盖')
    receipt=fingerprint(report_file)
    selected=job/'approved-versions'/(checked['sha256']+'-'+receipt[:12])
    selected.parent.mkdir(exist_ok=True)
    if not selected.is_dir():
        staging=selected.parent/('.'+uuid.uuid4().hex)
        staging.mkdir()
        try:
            for name,(path,digest) in outputs.items():
                shutil.copyfile(path,staging/name)
            if any(fingerprint(staging/name)!=digest for name,(_,digest) in outputs.items()):
                raise ValueError('定稿写入校验失败')
            shutil.copyfile(report_file,staging/'tile-inspection.json')
            store(staging/'review-receipt.json',{**checked,'终审报告':'tile-inspection.json','报告指纹':receipt})
            staging.rename(selected)
        finally:
            if staging.exists():shutil.rmtree(staging)
    if any(fingerprint(selected/name)!=digest for name,(_,digest) in outputs.items()) or fingerprint(selected/'tile-inspection.json')!=receipt:
        raise ValueError('原存档内容与通过终审的版本不一致')
    link(job/'final',selected.relative_to(job))
    for name in outputs:link(job/name,Path('final')/name)


def previous_print_pair(job,source,record):
    """Use an immutable release and the exact lower-resolution image it was judged against."""
    receipt=record['自动定稿']
    baseline=Path(record.get('底图') or '')
    if not baseline.is_absolute():baseline=source/baseline
    printed=(job/'final/4K.png').resolve()
    if (not baseline.is_file() or not printed.is_file() or fingerprint(printed)!=receipt.get('sha256')
            or fingerprint(baseline)!=receipt.get('对照指纹')):
        return None
    from focus_print import validate_print_pair
    try:validate_print_pair(baseline,printed)
    except (ValueError,OSError):return None
    return [str(baseline.resolve()),str(printed)]


def held_workmanship_pair(folder, brief):
    """Reinspect exact saved pixels when the subject-detail comparison changed."""
    from reference_review import DETAIL_REVIEW_VERSION
    file=folder/'review.json'
    if not file.exists():return None
    record=json.loads(file.read_text())
    if (record.get('prompt')!=brief or record.get('delivery_edge')!=3840
            or record.get('status')!='未达标 · 终审退回' or not record.get('4K') or not record.get('底图')):
        return None
    master=(folder/record['4K']).resolve()
    baseline=(folder/record['底图']).resolve()
    if (not master.is_relative_to(folder.resolve()) or not master.is_file() or not baseline.is_file()
            or fingerprint(baseline)!=record.get('当前4K重建底图指纹')):return None
    digest=fingerprint(master)
    assessment=next((row['review'] for row in reversed(record.get('reviews',[]))
                     if row['stage'].endswith('final-art') and row['hash']==digest),{})
    comparison=assessment.get('参考比较') or {}
    if (comparison.get('decision') not in ('below','review') or comparison.get('detail_review_version')==DETAIL_REVIEW_VERSION
            or not assessment.get('note') or assessment['note']!=record.get('终审退回原因')):return None
    return [str(baseline),str(master)]


def export_release(job,state):
    if state.get('output_dir'):
        from delivery import export_files
        files={name:job/'final'/name for name in ('4K.png','8K.png','tile-inspection.json','review-receipt.json') if (job/'final'/name).is_file()}
        state['exported_to']=export_files(state['output_dir'],job,files)


def dashboard(job,state):
    from library import IMAGES, published_cover, publication_label
    art=published_cover(job,state)
    parts=['<!doctype html><meta charset="utf-8"><title>离线出片</title>',
           '<style>body{max-width:58rem;margin:4rem auto;padding:0 1rem;font:18px/1.65 system-ui;background:#f7f5ef;color:#272923}a{color:#305647}li{margin:1rem 0}</style>',
           '<h1>离线出片</h1><p>'+html.escape(state['prompt'])+'</p><strong>'+html.escape(publication_label(state,art))+'</strong>']
    if not state.get('rechecking_approved') and art is not None:
        eight='<a href="8K.png">打开 8K</a>　' if (job/'final/8K.png').is_file() else ''
        edit='<a href="http://127.0.0.1:8195/?image='+quote(str(art.relative_to(IMAGES)),safe='')+'">涂抹微调</a>　'
        parts.append('<p>'+edit+'<a href="4K.png">打开 4K</a>　'+eight+'<a href="final/review-receipt.json">终审凭据</a>　<a href="final/tile-inspection.json">逐块目检记录</a></p>')
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


def local_retry_source(previous, brief, *, interrupted=False, review_upgrade=False):
    """Resume only a localized 4K rejection whose image and baseline still match the review."""
    record_file=previous/'review.json'
    if not record_file.is_file():return None
    record=json.loads(record_file.read_text())
    status=record.get('status','')
    eligible=status.startswith('未达标') or (interrupted and status in ('精绘中','离线逐块终审中'))
    if (record.get('delivery_edge')!=3840 or record.get('prompt')!=brief
            or not eligible or not record.get('4K') or not record.get('底图')):
        return None
    entry=record.get('逐块终审',{})
    if entry.get('状态') not in ('local','uncertain') or not entry.get('报告'):return None
    report_file=previous/entry['报告']
    if not report_file.is_file():return None
    report=json.loads(report_file.read_text())
    image=previous/record['4K']
    candidates=[record['底图'],record.get('高清底图返工',{}).get('重绘')]
    issue=report.get('issue',{})
    from print_judge import REVISION, actionable_edit, boxes_agree, map_box, needs_perspective_review
    recheck=(review_upgrade and report.get('revision') and report['revision']!=REVISION
             and entry['状态']=='uncertain' and (issue.get('operation') not in ('redraw','remove_forbidden')
             or not actionable_edit(issue.get('fix'))))
    if review_upgrade and needs_perspective_review(report):
        recheck=True
        if not issue.get('box'):
            rect=report['views'][-1].get('pixels')
            if rect and report.get('size'):
                issue=dict(issue,box=[round(1000*v/n) for v,n in zip(rect,report['size']*2)])
    if (review_upgrade and report.get('revision') and report['revision']!=REVISION
            and entry['状态']=='local' and issue.get('operation')=='redraw' and not recheck):
        issue=dict(issue,verify_part_identity=True)
    if recheck:
        issue=dict(issue,operation='reinspect',fix=issue.get('fix') or 'Inspect this previously held region.')
        views=report.get('views',[])
        if views:
            original=views[-1].get('observation',{})
            if original.get('operation')=='remove_forbidden':
                issue['unresolved_exclusion']=True
            if original.get('evidence') and all(k in original for k in ('left','top','right','bottom')):
                location=map_box(original,views[-1]['pixels'],report['size'])
                if location:
                    issue.update(box=location,symptom=original['evidence'])
    if review_upgrade and report.get('revision')!=REVISION and report.get('views'):
        last=report['views'][-1]
        if last.get('view')=='previous_local':
            known=dict((tuple(rect),data) for rect,data in report.get('previous_regions',[]))
            expected=known.get(tuple(last.get('pixels',[])),{})
            if expected.get('box') and issue.get('box') and not boxes_agree(expected['box'],issue['box']):
                recheck=True
                issue=dict(issue,operation='reinspect',box=expected['box'],symptom=expected.get('symptom',''),
                           fix='Recheck the originally reported location on the retained file.')
    if (review_upgrade and report.get('revision')!=REVISION and
            report.get('issue',{}).get('operation')=='remove_forbidden' and 'visual_triage' not in report):
        recheck=True
        issue=dict(report['issue'],operation='reinspect',
                   fix='Check the main visual problems on this master before revisiting decorative marks.')
    if (report.get('brief')!=brief or report.get('status')!=entry['状态']
            or (not recheck and issue.get('operation') not in ('redraw','remove_forbidden'))
            or (not recheck and not actionable_edit(issue.get('fix'))) or not issue.get('box') or not image.is_file()
            or fingerprint(image)!=report.get('sha256')):
        return None
    for name in dict.fromkeys(candidates):
        if name:
            draft=Path(name)
            if draft.is_file() and fingerprint(draft)==report.get('comparison_sha256'):
                return image,draft,issue
    return None


def same_repair_region(box, other):
    if not box or not other or len(box)!=4 or len(other)!=4:return False
    l,t,r,b=box
    x,y,u,v=other
    area=min((r-l)*(b-t),(u-x)*(v-y))
    return area>0 and max(0,min(r,u)-max(l,x))*max(0,min(b,v)-max(t,y))>=.25*area


def failed_scene_connection(directory, record, finding):
    scene=record.get('高清底图返工',{})
    if not scene.get('模型') or scene.get('保留原构图') is False or finding.get('operation')!='redraw':return False
    prior=scene.get('目标范围')
    if prior:return same_repair_region(prior,finding.get('box'))
    # Earlier scene-rebuild records kept the disputed box in their signed print report.
    entry=record.get('逐块终审',{})
    if not entry.get('报告'):return False
    path=directory/entry['报告']
    if not path.is_file():return False
    report=json.loads(path.read_text())
    if (entry.get('sha256')!=report.get('sha256') or report.get('brief')!=record.get('prompt')
            or report.get('status') not in ('local','uncertain')):return False
    return any(same_repair_region(item.get('box'),finding.get('box'))
               for _,item in report.get('previous_regions',[]))


def previous_scene_candidate(folder, stage, reviews, output_root, *, structural_reports=None):
    """Reopen a locally generated scene only when the render and review hashes agree."""
    receipts=[folder/(stage+'-task.json'),*sorted((folder/'render-history').glob(stage+'-*.json'))]
    tasks=[json.loads(p.read_text()) for p in receipts if p.is_file()]
    for row in reversed(reviews):
        if row.get('stage')!=stage:continue
        status=row.get('review',{}).get('status')
        if status!='uncertain':
            if status!='pass' or not structural_reports:continue
            report_name=structural_reports.get(row['file'])
            if not report_name:continue
            report_path=(folder/report_name).resolve()
            if not report_path.is_relative_to(folder.resolve()) or not report_path.is_file():continue
            judged=json.loads(report_path.read_text())
            from print_judge import REVISION
            if (judged.get('revision') in (None,REVISION) or judged.get('sha256')!=row.get('hash')
                    or judged.get('status') not in ('local','uncertain')):continue
        path=(folder/row['file']).resolve()
        if path.parent!=folder.resolve() or not path.is_relative_to(output_root.resolve()) or not path.is_file():continue
        key=path.relative_to(output_root.resolve()).as_posix()
        for task in tasks:
            if task.get('renderer')!='ComfyUI' or not task.get('model_files'):continue
            if task.get('output_sha256',{}).get(key)==row.get('hash')==fingerprint(path):
                return path,task['model_files']
    return None


def previous_scene_on_master(previous, source, brief, output_root):
    """An earlier guided composition is only a candidate on the same authenticated master."""
    path=previous/'review.json'
    if not path.is_file():return None
    saved=json.loads(path.read_text())
    if saved.get('prompt')!=brief or not saved.get('4K'):return None
    retained=previous/saved['4K']
    digest=saved.get('逐块终审',{}).get('sha256')
    if not digest or not retained.is_file() or fingerprint(retained)!=digest or fingerprint(source)!=digest:return None
    for row in reversed(saved.get('reviews',[])):
        stage=row.get('stage','')
        if not stage.endswith('-scene'):continue
        candidate=previous_scene_candidate(previous,stage,[row],output_root,
                                           structural_reports=saved.get('底图结构终审'))
        if candidate:return candidate
    return None


def earlier_local_version(folder, current, baseline, box, brief, revision):
    """Find a checked earlier master differing only in this damaged neighborhood."""
    from PIL import Image, ImageChops
    if folder.parent.name!='rounds' or not box:
        return None
    rounds=folder.parent.resolve()
    current_sha=fingerprint(current)
    baseline_sha=fingerprint(baseline)
    seen={current_sha}
    for receipt in rounds.glob('*/prior-version-*-review.json'):
        verdict=json.loads(receipt.read_text())
        if verdict.get('current')==current_sha and verdict.get('revision')==revision:
            seen.add(verdict.get('earlier'))
    # Prefer the earliest drawing, before repeated edits of one part accumulated.
    reports=sorted(rounds.glob('*/print-review/*.json'),key=lambda p:p.stat().st_mtime)
    with Image.open(current) as src:
        original=src.convert('RGB')
        left,top,right,bottom=(round(v*n/1000) for v,n in zip(box,original.size*2))
        pad=max(100,right-left,bottom-top)
        for report in reports:
            record=json.loads(report.read_text())
            digest=record.get('sha256')
            if (not digest or digest in seen or record.get('brief')!=brief or record.get('size')!=list(original.size)
                    or record.get('comparison_sha256')!=baseline_sha or record.get('status') not in ('local','uncertain')
                    or not same_repair_region(box,record.get('issue',{}).get('box'))):
                continue
            earlier=Path(record['source']).resolve()
            if not earlier.is_relative_to(rounds) or not earlier.is_file() or fingerprint(earlier)!=digest:
                continue
            seen.add(digest)
            with Image.open(earlier) as other:
                if other.size!=original.size:continue
                region=ImageChops.difference(original,other.convert('RGB')).getbbox()
            if not region:continue
            x,y,u,v=region
            if (x>=left-pad and y>=top-pad and u<=right+pad and v<=bottom+pad
                    and max(left,x)<min(right,u) and max(top,y)<min(bottom,v)):
                return earlier,dict(current=current_sha,earlier=digest,revision=revision,changed_pixels=list(region))
    return None


def repeated_rejection(directory, box, *, minimum=2):
    """Choose a different editor first when multiple nearby repairs actually failed."""
    left,top,right,bottom=box
    area=(right-left)*(bottom-top)
    if area<=0:return False
    failures=0
    for receipt in directory.glob('*-patch-review.json'):
        review=json.loads(receipt.read_text())
        if review.get('verdict') not in ('still_present','new_defect','uncertain','unchanged'):continue
        other=review.get('target_box')
        if not other:
            plan=directory/(receipt.name.removesuffix('-patch-review.json')+'-repair-plan.json')
            if plan.is_file():other=json.loads(plan.read_text()).get('box')
        if not isinstance(other,list) or len(other)!=4:continue
        l,t,r,b=other
        overlap=max(0,min(right,r)-max(left,l))*max(0,min(bottom,b)-max(top,t))
        if overlap>=.55*area:
            failures+=1
            if failures>=minimum:return True
    return False


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
            for key in ('reviewed_image','candidate_image','selection_mask','review_print'):
                if getattr(args,key):options.extend(['--'+key.replace('_','-'),str(getattr(args,key).resolve())])
            for source in args.from_image:options.extend(['--from-image',str(source.resolve())])
            if args.repair_box:options+=['--repair-box',*map(str,args.repair_box),'--repair-text',args.repair_text]
            if args.object:options+=['--object',args.object]
            if getattr(args,'aspect','auto')!='auto':options+=['--aspect',args.aspect]
            if args.base_model!='auto':options+=['--base-model',args.base_model]
            if args.print_method!='auto':options+=['--print-method',args.print_method]
            if args.candidates:options+=['--candidates',str(args.candidates)]
            state=dict(prompt=args.prompt.strip(),seed=args.seed if args.seed is not None else secrets.randbelow(2**45),
                       delivery_edge=7680 if args.eight_k else 3840,options=options,rounds=[],status='准备离线出片',port=args.port)
        if args.deep_repair:state['deep_repair']=True
        if args.eight_k and state.get('delivery_edge',7680)==3840:
            if args.skip_focus is not None:
                raise SystemExit('--skip-focus 需要已有 8K 精绘结果')
            # Keep the published 4K in place until the separately reviewed upgrade passes.
            state['delivery_edge']=7680
            if state['rounds'] and state['rounds'][-1]['status'] in ('运行中','运行中断'):
                state['rounds'][-1]['status']='已转入 8K 制作'
            if (job/'final/4K.png').is_file():
                state['options']=['--reviewed-image',str((job/'final/4K.png').resolve())]
            if args.full_8k_diffusion and '--full-8k-diffusion' not in state['options']:
                state['options'].append('--full-8k-diffusion')
            state['status']='准备 8K 制作'
        if getattr(args,'output_dir',None):state['output_dir']=str(args.output_dir)
        if args.accept_final:raise SystemExit('自动任务只接收本机模型通过的版本。')
        if not model_identity(MODEL):raise SystemExit('本机缺少终审模型 '+MODEL)
        edit=args.skip_focus is not None
        if edit and not state['rounds']:raise SystemExit('还没有可调整的精绘区域')
        if state.get('status')=='已自动定稿' and not edit:
            folder=job/state['selected']
            record=json.loads((folder/'review.json').read_text())
            receipt=record['自动定稿']
            stale=receipt.get('审图版本')!=REVISION or receipt.get('模型指纹')!=model_identity(MODEL)
            stale=stale or not workmanship_confirmed(record,receipt.get('sha256'))
            if not stale:
                inspected=folder/receipt['终审报告']
                stale=not inspected.is_file() or not complete_print_review(json.loads(inspected.read_text()))
            if not stale:
                publish(job,folder,record)
                export_release(job,state);save(job,state)
                print('已自动定稿',job/'final',flush=True);return 0
            if state.get('delivery_edge',7680)==3840:
                state['retained_print_pair']=previous_print_pair(job,folder,record)
            else:
                # An 8K receipt binds its 4K source as the comparison image.
                four=(job/'final/4K.png').resolve()
                if four.is_file() and fingerprint(four)==receipt.get('4K指纹')==receipt.get('对照指纹',receipt.get('4K指纹')):
                    state['retained_4k_source']=str(four)
            state['rechecking_approved']=True
            state['status']='按新版标准重新审查'
            save(job,state)
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
            options=list(state['options'])
            if len(rounds)>1 and '--review-print' in options:
                from print_judge import REPAIR_LAYOUT_REVISION
                prior=rounds[-2]
                prior_evidence=job/prior['directory']/'composition-evidence.json'
                prior_revision=(json.loads(prior_evidence.read_text()).get('revision')
                                if prior_evidence.is_file() else None)
                revised_check=(prior['status']=='未达标 · 修订稿与底图构图不一致'
                               and prior_revision!=REPAIR_LAYOUT_REVISION)
                if not revised_check:
                    previous=options.index('--review-print')
                    del options[previous:previous+2]
            pending_pair=state.pop('retained_print_pair',None)
            if pending_pair:row['reinspection_pair']=pending_pair
            retained_four=state.pop('retained_4k_source',None)
            if retained_four:row['reinspection_source']=retained_four
            if (len(rounds)>1 and '--from-image' not in options and not edit and state['delivery_edge']==3840
                    and not row.get('reinspection_pair') and not rounds[-2].get('reinspection_pair')
                    and not any(flag in options for flag in ('--reviewed-image','--review-print','--repair-box','--candidate-image'))):
                pair=held_workmanship_pair(job/rounds[-2]['directory'],state['prompt'])
                if pair:row['reinspection_pair']=pair
            if row.get('reinspection_source'):
                # Use only the previously checked 4K master as the native composition.
                aspect=options[options.index('--aspect')+1] if '--aspect' in options else 'auto'
                print_options=['--full-8k-diffusion'] if '--full-8k-diffusion' in options else []
                options=['--reviewed-image',row['reinspection_source'],'--aspect',aspect,*print_options]
            if row.get('reinspection_pair'):
                baseline,printed=row['reinspection_pair']
                aspect=options[options.index('--aspect')+1] if '--aspect' in options else 'auto'
                options=['--reviewed-image',baseline,'--review-print',printed,'--aspect',aspect]
            if edit:row['skip_focus']=args.skip_focus
            row['status']='运行中';state['status']='离线审图与返修中'
            save(job,state)
            command=[sys.executable,str(entry),'--round-worker','--job',str(folder),'--prompt',state['prompt'],
                     '--port',str(state['port']),'--seed',str(row['seed']),*options]
            if state.get('delivery_edge',7680)==7680:command.append('--8k')
            if state.get('deep_repair'):command.append('--deep-repair')
            if len(rounds)>1 and state.get('delivery_edge')==3840 and not edit:
                command+=['--retry-round',str(job/rounds[-2]['directory'])]
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
                state.pop('rechecking_approved',None)
                state.update(status='已自动定稿',selected=row['directory'],sha256=record['自动定稿']['sha256'])
                save(job,state);print('已自动定稿',job/'final',flush=True);return 0
            attempts+=1;edit=False
            state['status']='本轮未达标，自动重试'
            save(job,state)
        state['status']='达到指定轮数，本次未产生新定稿'
        save(job,state);print(state['status'],flush=True);return 2
