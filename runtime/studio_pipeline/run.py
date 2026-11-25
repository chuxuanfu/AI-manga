"""Local candidate selection, repeatable regional repair, and checked 4K/8K delivery."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
from urllib import error, request
import uuid

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch

from gallery import write_gallery, write_focus_proof
from graphs import draft, finish
from illustration import BRUSHWORK, REFERENCE, reference_draft, independent_draft, text_draft, print_graph
from reference_review import compare
from structure import inspect_structure
from construction_review import review as review_construction
from review import audit, reviewer, RUNTIME, CHECKS
from manga_materials.planner import plan_scene, requested_medium
from focus_print import PATCH, centers, combine_source, detail_graph, finish_neural, neural_double
from masks import object_mask
from erase import duplicate_box, masked_candidate, MODEL as ERASE_MODEL

STUDIO = RUNTIME.parent
OUTPUT = STUDIO / 'images'
INPUT = RUNTIME / 'ComfyUI/input'
Image.MAX_IMAGE_PIXELS = 180_000_000


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)
    if path.name=="review.json":write_gallery(path.parent,obj,STUDIO)


def deliver_file(source, dest):
    if dest.exists() and dest.samefile(source):
        return
    temporary=dest.with_name(uuid.uuid4().hex+'.png')
    try:
        try:os.link(source,temporary)
        except OSError:shutil.copyfile(source,temporary)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


def local_image(im):
    # Stable content-addressed inputs support editor replay and parallel jobs.
    key = hashlib.sha256(im.mode.encode() + str(im.size).encode() + im.tobytes()).hexdigest()[:24]
    name = 'studio-' + key + '.png'
    if not (INPUT/name).exists():
        temp = INPUT / (uuid.uuid4().hex + '.png')
        im.save(temp)
        temp.replace(INPUT/name)
    return name


def post_graph(api, port, directory, stage, workflow=None):
    cache = directory / (stage + '-task.json')
    url = 'http://127.0.0.1:' + str(port)
    request_data={'prompt':api}
    if workflow:request_data['extra_data']={'extra_pnginfo':{'workflow':workflow}}
    payload = json.dumps(request_data).encode()
    digest = hashlib.sha256(payload).hexdigest()
    job = json.loads(cache.read_text()) if cache.exists() else {}
    if job.get('digest') != digest:
        job = {}
    def get(path):
        with request.urlopen(url+path,timeout=30) as response:
            return json.load(response)
    if job:
        history = get('/history/'+job['id'])
        active = get('/queue')
        if job['id'] not in history and not any(row[1]==job['id'] for rows in active.values() for row in rows):
            job = {}
    if not job:
        req = request.Request(url+'/prompt', data=payload, headers={'Content-Type':'application/json'})
        try:
            with request.urlopen(req,timeout=45) as response:
                ident = json.load(response)['prompt_id']
        except error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
        job = {'id':ident,'digest':digest}
        save(cache,job)
    while True:
        hist = get('/history/'+job['id']).get(job['id'])
        if hist:
            if hist['status']['status_str']!='success':
                cache.unlink(missing_ok=True)
                raise RuntimeError(str(hist['status']['messages'][-2:])[:2300])
            return hist['outputs']
        time.sleep(4)


def output_image(record):
    image = record['images'][0]
    return OUTPUT / image['subfolder'] / image['filename']


def region(issue,assessment):
    def valid(box):
        return (isinstance(box,list) and len(box)==4 and all(isinstance(x,(int,float)) and 0<=x<=1000 for x in box)
                and box[2]-box[0]>=12 and box[3]-box[1]>=12 and (box[2]-box[0])*(box[3]-box[1])<650000)
    box = issue.get('box')
    if not valid(box):
        box = assessment.get('regions',{}).get(str(issue.get('region','')),{}).get('box')
    return box if valid(box) else None


def repaint(im, box, description, caption, seed, port, directory, stage, target=None, neighbours=(), reference_style=False):
    # Re-sample only the interior; the wider square gives diffusion scene context.
    w,h = im.size
    x0,y0,x1,y1 = [round(v*n/1000) for v,n in zip(box,(w,h,w,h))]
    span = max(x1-x0,y1-y0)
    side = min(max(w,h), max(192, round(span*1.65)))
    cw,ch = min(side,w),min(side,h)
    x = min(max(0,(x0+x1-cw)//2),w-cw)
    y = min(max(0,(y0+y1-ch)//2),h-ch)
    crop = im.crop((x,y,x+cw,y+ch))
    size = tuple(max(64,round(n*960/max(cw,ch)/8)*8) for n in (cw,ch))
    mask = Image.new('L',(cw,ch))
    margin = max(2,round(span*.035))
    rect = (x0-x-margin,y0-y-margin,x1-x+margin,y1-y+margin)
    ImageDraw.Draw(mask).rounded_rectangle(rect,radius=max(2,round(span*.05)),fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(2,round(span*.022))))
    prepared = crop.resize(size,Image.Resampling.LANCZOS)
    prepared.putalpha(ImageOps.invert(mask.resize(size,Image.Resampling.BILINEAR)))
    prefix = (directory.relative_to(OUTPUT)/stage).as_posix()
    if reference_style:
        size=tuple(max(64, round(n/16)*16) for n in size)
        layout=local_image(crop.resize(size,Image.Resampling.LANCZOS))
        text=description
        edit=reference_draft(text,seed,*size,prefix,reference=layout,edit=True)
        graph=edit.api
        output=str(len(edit.nodes))
    else:
        graph = draft(description+'. '+caption,seed,*size,prefix).api
        graph['4'] = {'class_type':'LoadImage','inputs':{'image':local_image(prepared)}}
        graph['8'] = {'class_type':'VAEEncode','inputs':{'vae':['1',2],'pixels':['4',0]}}
        graph['9'] = {'class_type':'SetLatentNoiseMask','inputs':{'samples':['8',0],'mask':['4',1]}}
        graph['5']['inputs'].update(latent_image=['9',0],steps=24,denoise=.81,cfg=5.5,sampler_name='dpmpp_2m')
        output='7'
    hist = post_graph(graph,port,directory,stage)
    with Image.open(output_image(hist[output])) as new:
        patched = im.copy()
        proposed=new.convert('RGB').resize((cw,ch),Image.Resampling.LANCZOS)
        patched.paste(proposed,(x,y),object_mask(crop,proposed,target,mask,neighbours))
    target = directory/(stage+'-composite.png')
    patched.save(target)
    return target


def dimensions(prompt):
    if any(s in prompt.lower() for s in ('16:9','16：9','横屏','横幅','landscape format','landscape orientation','widescreen')):
        return (1280,720)
    if any(s in prompt.lower() for s in ('9:16','9：16','手机屏','phone wallpaper')):
        return (720,1280)
    if any(s in prompt.lower() for s in ('1:1','方形','square format','square composition')):
        return (1024,1024)
    return (832,1248)


def main():
    parser = argparse.ArgumentParser(description='本地漫画壁纸选片、纠错、精绘')
    parser.add_argument('-p','--prompt',help='中文或英文画面描述')
    parser.add_argument('-o','--output-dir',type=Path,help='将选中的校样或通过终审的成品复制到此目录')
    parser.add_argument('--aspect',choices=('auto','2:3','9:16','16:9','1:1'),default='auto',help='画幅比例，默认从描述判断，未指定时为 2:3')
    parser.add_argument('--style',help='可选画材或画风，例如透明水彩')
    parser.add_argument('--accept-final',metavar='SHA256',help='目检后，按当前母版指纹标记定稿')
    parser.add_argument('--skip-focus',nargs='*',type=int,choices=(1,2,3),default=None,help='对照后撤回指定编号的8K局部采样；不带编号可全部恢复。')
    parser.add_argument('--full-8k-diffusion',action='store_true',help='4K、8K 两级均进行全幅重叠分块扩散')
    parser.add_argument('--native-4k',action='store_true',help='直接在长边 3840 的画布采样，耗时较长')
    parser.add_argument('--native-only',action='store_true',help='生成并保留各张原生底图，供原尺寸查看')
    parser.add_argument('--proof-only',action='store_true',help='先选片纠错并保存校样，稍后可继续精绘')
    parser.add_argument('--job',type=Path,help='继续已有任务目录')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--candidate-image',type=Path,help='已经重绘、与底图同尺寸的备选图')
    parser.add_argument('--selection-mask',type=Path,help='灰度或透明通道遮罩，白色或不透明表示采用备选')
    parser.add_argument('--keep-native',action='store_true',help='对已有长边 7680 的母版返修，保留选区外的原像素')
    parser.add_argument('--from-image',type=Path,action='append',default=[],help='先检查已有底图，可重复')
    parser.add_argument('--candidates',type=int,choices=range(1,9),default=None)
    parser.add_argument('--max-rounds',type=int,default=0,help='完整出片的最大轮数，默认 0 表示持续至通过')
    parser.add_argument('--retry-observation',default='',help=argparse.SUPPRESS)
    parser.add_argument('--round-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--port',type=int,choices=(8193,8194),default=8193)
    parser.add_argument('--reviewed-image',type=Path,help='使用已经人工审定的原图，仍进行最终检查')
    parser.add_argument('--repair-box',nargs=4,type=int,metavar=('L','T','R','B'),help='手动指出需要复绘的位置，0 到 1000')
    parser.add_argument('--object',help='可选，指定需要保护轮廓的对象，如 porcelain teapot')
    parser.add_argument('--repair-text',help='这个区域应当画成什么样')
    args=parser.parse_args()
    if args.style and args.prompt:
        args.prompt=args.prompt.rstrip()+' · '+args.style.strip()
    if args.repair_box and (not args.repair_text or not args.from_image or not region({'box':args.repair_box},{})):
        parser.error('--repair-box 需要有效坐标、--repair-text 和 --from-image')
    if bool(args.candidate_image)!=bool(args.selection_mask):parser.error('请同时传入 --candidate-image 与 --selection-mask')
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error('--seed must be between 0 and 9223372036854775807')
    if args.output_dir is not None:
        args.output_dir=args.output_dir.expanduser().resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(6,os.cpu_count() or 2))
    job = args.job.resolve() if args.job else OUTPUT/'new-jobs'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    job.relative_to(OUTPUT)
    if args.max_rounds<0:parser.error('--max-rounds 需要非负数')
    previous=job/'review.json'
    adopt_draft=False
    if not args.round_worker and not (args.proof_only or args.native_only or args.accept_final) and previous.is_file() and not (job/'production.json').exists():
        earlier=json.loads(previous.read_text())
        adopt_draft=earlier.get('status') in ('底图已选 · 待目检','待人工复核 · 尚未选出底图')
        if adopt_draft:
            if args.prompt and args.prompt.strip()!=earlier['prompt']:
                parser.error('请为新提示词建立独立任务。')
            args.prompt=earlier['prompt']
            if args.seed is None:args.seed=earlier.get('seed')
            if earlier.get('原生4K'):args.native_4k=True
            if earlier.get('输出模式')=='全幅双级扩散':args.full_8k_diffusion=True
            if not args.from_image and not args.reviewed_image:
                if earlier.get('底图'):args.reviewed_image=Path(earlier['底图'])
                elif earlier.get('输入底图'):args.from_image=[Path(x) for x in earlier['输入底图']]
    if not args.round_worker and not (args.proof_only or args.native_only) and (
        (job/'production.json').exists() or not previous.exists() or adopt_draft
    ):
        from production import run as supervise
        return supervise(args,job,Path(__file__))
    job.mkdir(parents=True,exist_ok=True)
    lock=(job/'.running').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('这个任务正由另一个本地进程运行')
    manifest = job/'review.json'
    rec = json.loads(manifest.read_text()) if manifest.exists() else {}
    if args.skip_focus is not None and (not rec or rec.get('输出模式')!='全幅4K扩散与8K焦点精修' or not rec.get('精绘区域报告',{}).get('8K局部')):
        parser.error('--skip-focus 需要使用已有focus-refinement结果的 --job')
    if args.skip_focus is not None:
        if any(i>len(rec['精绘区域报告']['8K局部']) for i in args.skip_focus):parser.error('编号超过这张图的实际焦点数')
        if args.accept_final or args.repair_box or args.keep_native or args.proof_only or args.native_only:parser.error('请分别执行局部取舍和其他修订操作')
        rec['撤回焦点']=sorted(set(args.skip_focus));rec.pop('目检定稿',None)
        rec['焦点取舍依据']={'4K指纹':hashlib.sha256((job/rec['4K']).read_bytes()).hexdigest(),'区域':rec['精绘区域报告']['8K局部']}
    if args.accept_final:
        if rec.get('执行模式')=='离线自动定稿':parser.error('此版本需要通过自动终审才能定稿。')
        if not args.job or not rec.get('8K'):parser.error('需要有 8K 结果的 --job')
        master=job/rec['8K'];small=job/rec['4K']
        fingerprint=hashlib.sha256(master.read_bytes()).hexdigest()
        if args.accept_final!=fingerprint:
            parser.error('本次选择与当前 8K 母版不符，请重新打开校样')
        if fingerprint in rec.get('退稿',{}):
            parser.error('这份母版已经退稿。修订图有独立的指纹。')
        release=job/'manual-release'/fingerprint[:12]
        release.mkdir(parents=True,exist_ok=True)
        for src,label in ((master,'8K'),(small,'4K')):
            dest=release/(label+'.png')
            if not dest.exists():os.link(src,dest)
        rec['目检定稿']={'sha256':fingerprint,'时间':time.strftime('%Y-%m-%d %H:%M:%S'),'8K':str((release/'8K.png').relative_to(job)),'4K':str((release/'4K.png').relative_to(job))}
        rec['状态在目检前']=rec.get('状态在目检前',rec['status']);rec['status']='已目视定稿'
        save(manifest,rec);print(rec['status'],release,flush=True);return 0
    if rec.get('退稿') and args.skip_focus is None and not any((args.reviewed_image,args.from_image,args.candidate_image,args.repair_box,args.prompt)):
        print(rec['status'],flush=True);return 2
    changing=args.skip_focus is not None or any((args.prompt,args.from_image,args.reviewed_image,args.repair_box,args.candidate_image,args.seed is not None,args.candidates))
    if rec.get('status')=='已目视定稿' and not changing:
        print(rec['status'],job/rec['目检定稿']['8K']);return 0
    if rec.get('选区返修') and not changing:
        prior=rec['选区返修']
        args.from_image=[Path(prior['原图'])];args.candidate_image=Path(prior['候选'])
        args.selection_mask=Path(prior['遮罩']);args.keep_native=True
    prompt = (args.prompt or rec.get('prompt') or '').strip()
    if not prompt: parser.error('需要 --prompt 或已有任务目录')
    if rec and rec['prompt']!=prompt: parser.error('请为新的 prompt 开始一个新任务')
    seed = rec.get('seed',args.seed if args.seed is not None else secrets.randbelow(2**45))
    print('任务目录',job,flush=True)
    if not rec:
        positive,raw=plan_scene(prompt)
        rec=dict(prompt=prompt,seed=seed,positive=positive,parts=json.loads(raw),reviews=[],status='选片',工作流='reference-illustration-v1',原生4K=(args.native_4k or any(t in prompt.lower().replace(' ','') for t in ('原生4k','native4k'))),画法参考=hashlib.sha256((INPUT/REFERENCE).read_bytes()).hexdigest())
        rec['画法自适应']=True
        rec['画材']=requested_medium(prompt)
        rec['输出模式']='全幅双级扩散' if args.full_8k_diffusion else '全幅4K扩散与8K焦点精修'
        rec['笔触参考']=hashlib.sha256((INPUT/BRUSHWORK).read_bytes()).hexdigest()
        save(manifest,rec)
    if args.aspect!='auto':rec['aspect']=args.aspect
    if args.round_worker:rec['执行模式']='离线自动定稿'
    reference_style=rec.get('工作流')=='reference-illustration-v1'
    if args.native_4k and not rec.get('原生4K'):
        parser.error('继续原任务时请保持原画幅，新画幅请建立新任务。')
    if reference_style and rec['画法参考']!=hashlib.sha256((INPUT/REFERENCE).read_bytes()).hexdigest():
        parser.error('该任务的画法参考已更换，请为新参考建立新任务。')
    args.candidates=args.candidates or rec.get('候选数',3)
    if args.from_image:
        rec['输入底图']=[str(p.resolve()) for p in args.from_image]
    else:
        args.from_image=[Path(p) for p in rec.get('输入底图',[])]
    rec['候选数']=args.candidates
    if not args.candidate_image and not args.reviewed_image and not args.repair_box and rec.get('底图人工审定'):
        args.reviewed_image=Path(rec['底图人工审定'])
    if args.repair_box:
        rec['指定修正']={'box':args.repair_box,'text':args.repair_text,'object':args.object}
    elif rec.get('指定修正') and not args.reviewed_image:
        correction=rec['指定修正']
        args.repair_box=correction['box'];args.repair_text=correction['text'];args.object=correction['object']
    prefix = job.relative_to(OUTPUT).as_posix()
    def build_draft(caption,draw_seed,label,index):
        from page_layout import retry_direction
        clean=retry_direction(args.retry_observation)
        if clean:caption += ' Preserve the brief and address this previous observation: '+clean
        if not any(v in prompt.lower() for v in ('分镜','分格','多格','四格','多联','双联','三联','两联','二联','分幅','拼贴','拼图','对照图','连环画','分屏','漫画页','panel','diptych','triptych','collage','contact sheet','comic page','storyboard','split screen','split view')):
            caption='Compose one continuous scene in an uninterrupted frame. '+caption
        aspect=rec.get('aspect','auto')
        width,height={'2:3':(832,1248),'9:16':(720,1280),'16:9':(1280,720),'1:1':(1024,1024)}.get(aspect,dimensions(prompt))
        name=prefix+'/candidates/'+label
        if not reference_style:return draft(caption,draw_seed,width,height,name)
        size={(832,1248):(1248,1872),(720,1280):(1152,2048),(1280,720):(2048,1152),(1024,1024):(1536,1536)}[width,height]
        if rec.get('原生4K'):
            size=tuple(round(v*3840/max(size)/16)*16 for v in size)
        if not prompt.isascii():
            caption+=' Original request (use its species, colours and counts): '+prompt
        # A single style reference can also copy the example pose. Give complete jobs
        # both an independent layout and a text-only alternative to select from.
        borrow=any(term in prompt.lower() for term in ('参考图','参考画','像图','reference image','reference illustration'))
        own_first=rec.get('画法自适应') and bool(rec.get('画材')) and not borrow and not rec.get('原生4K') and not args.native_only
        if (own_first and index==0) or (not own_first and index==2 and rec.get('画法自适应')):
            direction=('Follow the medium and palette in the request. When unspecified, draw a finished Japanese storybook illustration. '
                       'Place each principal subject once. At every overlap, keep the nearer contour legible. Show a plausible support or occluding surface for visible figures, tools and animals. Preserve all requested interactions. ')
            graph=text_draft(direction+caption,draw_seed,*size,name)
        elif index==1 and not rec.get('原生4K'):
            graph=independent_draft(caption,draw_seed,*size,name)
        elif args.native_only:
            graph=reference_draft(caption,draw_seed,*size,name)
        else:
            graph=reference_draft(caption,draw_seed,*size,name,layout=None if index==2 and not own_first else True)
        save(job/(label+'workflow.json'),graph.ui())
        return graph
    if args.native_only:
        args.proof_only=True
    def check(path,stage,focus=None):
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        rubric=hashlib.sha256((CHECKS+('-construction-zoom-v10' if rec.get('画法自适应') else '')+'-independent-inventory-v4'+('body-support-v3-blind-pair'+rec['画法参考'] if reference_style else '')).encode()).hexdigest()[:12]
        from page_layout import gutters,unrequested_panels
        with Image.open(path) as probe:overview,gaps=gutters(probe)
        if gaps and args.round_worker:rubric+='-layout2'
        prior=next((a for a in rec['reviews'] if a['hash']==key and a.get('rubric')==rubric and a['stage']==stage),None)
        if prior:return prior['review']
        if gaps and args.round_worker:
            mismatch=unrequested_panels(overview,gaps,prompt)
            if mismatch:
                result=dict(status='global',score=35,note='画面被纸白间隔切成了多个独立画幅。'+mismatch['evidence'],
                            issues=[],regions={},layout=mismatch)
                rec['reviews'].append(dict(file=os.path.relpath(path,job),hash=key,rubric=rubric,stage=stage,review=result))
                save(manifest,rec);return result
        with Image.open(path) as image:
            rgb=image.convert('RGB')
            result=audit(rgb,prompt,rec['parts'],job/(stage+'-inspection.jpg'),focus=focus)
            if reference_style:
                geometry=inspect_structure(rgb,result.get('detector',[]),job/(stage+'-structure-and-support.jpg'),result.get('observed_counts'))
                result['结构审查']=geometry
                verdict=geometry.get('verdict','unclear')
                if verdict=='defective':
                    result.update(status='global',note='主体连接或承托有误。'+geometry.get('support_or_occlusion',''))
                elif verdict!='coherent' and result['status']=='pass':
                    result.update(status='uncertain',note='主体连接或承托无法确认。'+geometry.get('support_or_occlusion',''))
                if rec.get('画法自适应'):
                    detail=review_construction(rgb,prompt,result.get('regions',{}),job/(stage+'-contact-and-occlusion.jpg'),result.get('detector',()),result.get('observed_counts',{}).get('person',0))
                    result['局部连接复审']=detail
                    dv=detail['verdict']
                    if dv in ('local','global') and result['status']!='global':
                        result.update(status=dv,note=detail['observation'])
                        if dv=='local':
                            result['issues'].insert(0,dict(target=detail['target'],box=detail['box'],fix=detail['fix'],symptom=detail['observation']))
                    elif dv=='unclear' and result['status']=='pass':
                        result.update(status='uncertain',note=detail['observation'])
                comparison=compare(rgb,result.get('regions',{}),job/(stage+'-reference-comparison.jpg'),brief=prompt if rec.get('画法自适应') else '')
                result['参考比较']=comparison
                if result['status']=='pass' and comparison['decision']=='below':
                    result['status']='global'
                    result['note']+=' · 交换左右后仍发现画法差距'
                elif comparison['decision']=='review':
                    result['note']+=' · 画法对照需目检'
        rec['reviews'].append(dict(file=os.path.relpath(path,job),hash=key,rubric=rubric,stage=stage,review=result))
        save(manifest,rec)
        print(stage,result['status'],result['note'],flush=True)
        return result
    def usable(assessment):
        return assessment['status']=='pass' and assessment['score']>=75
    approved=None
    focus=args.repair_box
    def check_composition(path,label):
        result=check(path,label)
        duplicate=duplicate_box(result) if rec.get('画法自适应') and not args.repair_box and ERASE_MODEL.exists() else None
        if not duplicate and args.round_worker and ERASE_MODEL.is_file() and any(
            c.get('requested')==0 and (c.get('observed') or 0)>0 for c in result.get('count_conflicts',[])
        ):
            # The blind inventory saw a forbidden subject. Localize it on real pixels before erasing.
            from print_judge import inspect
            inspected=inspect(path,prompt,job,result)
            finding=inspected.get('issue',{})
            if inspected['status']=='local' and finding.get('operation')=='remove_forbidden':
                duplicate=region(finding,result)
        if duplicate:
            repaired=job/(label+'-removed-duplicate.png')
            with Image.open(path) as source:
                masked_candidate(source,duplicate).save(repaired)
            checked=check(repaired,label+'-count-repair',duplicate)
            if usable(checked):return repaired,checked
        return path,result

    if args.candidate_image:
        if not args.from_image:parser.error('选区合成需要 --from-image 提供原图')
        with Image.open(args.from_image[0]) as before, Image.open(args.candidate_image) as candidate, Image.open(args.selection_mask) as selection:
            if before.size!=candidate.size or selection.size!=before.size:parser.error('原图、候选和遮罩必须具有相同尺寸')
            mask=selection.getchannel('A') if 'A' in selection.getbands() else selection.convert('L')
            bounds=mask.getbbox()
            if not bounds:parser.error('选区为空')
            composite=before.convert('RGB')
            composite.paste(candidate.crop(bounds).convert('RGB'),bounds[:2],mask.crop(bounds))
            approved=job/'merged-selection.png'
            tmp=job/(uuid.uuid4().hex+'.png');composite.save(tmp);tmp.replace(approved)
            focus=[round(1000*z/n) for z,n in zip(bounds,(*before.size,*before.size))]
        rec['选区返修']={'原图':str(args.from_image[0]),'候选':str(args.candidate_image.resolve()),'遮罩':str(args.selection_mask.resolve()),'范围':focus}
    elif args.reviewed_image:
        approved=args.reviewed_image.resolve()
        rec['底图人工审定']=str(approved)
    else:
        paths=list(args.from_image) or [job/name for name in rec.get('原生底图',[]) if (job/name).is_file()]
        assessed=[]
        passing=[]
        for i in range(max(args.candidates,len(paths))):
            label=f'draft-{i+1}'
            if i<len(paths):
                path=paths[i].resolve()
            else:
                graph=build_draft(rec['positive'],seed+i,label,i)
                hist=post_graph(graph.api,args.port,job,label,graph.ui() if reference_style else None)
                path=output_image(hist[str(len(graph.nodes))])
                native=rec.setdefault('原生底图',[])
                local=str(path.relative_to(job))
                if local not in native:native.append(local)
                save(manifest,rec)
                print('原生底图',path,flush=True)
            path,result=check_composition(path,label)
            if args.repair_box and i==0:
                result={'status':'local','score':100,'issues':[{'box':args.repair_box,'fix':args.repair_text,'target':args.object}]}
            if usable(result) and not args.repair_box:
                passing.append((result['score'],path))
            if result['status']=='local':
                assessed.append((result,path,i))
        if passing:approved=max(passing,key=lambda item:item[0])[1]
        if not approved:
            for result,path,i in sorted(assessed,key=lambda x:-x[0]['score']):
                issue=next((j for j in result['issues'] if region(j,result) and j.get('fix')),None)
                if not issue:continue
                base_path=path
                for n in range(3):
                    label=f'region-repair-{i+1}-{n+1}'
                    with Image.open(base_path) as original:
                        patched=repaint(original.convert('RGB'),region(issue,result),issue['fix'],rec['positive'],seed+100+i*3+n,args.port,job,label,issue.get('target'),[o for o,_ in rec['parts']],reference_style)
                    next_review=check(patched,label,region(issue,result))
                    if usable(next_review):
                        approved=patched;break
                    if next_review['status']=='local' and next_review['score']>=result['score']:
                        next_issue=next((j for j in next_review['issues'] if region(j,next_review) and j.get('fix')),None)
                        if next_issue:
                            base_path,result,issue=patched,next_review,next_issue
                if approved:break
    if not approved and not args.from_image and not args.repair_box:
        # Allow fresh layouts for unattended production even when reviewers cannot localize an issue.
        conflicts=[c for row in rec['reviews'] for c in row['review'].get('count_conflicts',[])]
        terms={c['object']:c['requested'] for c in conflicts}
        if terms or not args.proof_only or any(a['review']['status']=='global' for a in rec['reviews']):
            constraint=', '.join(f"exactly {n} {name}" if n>0 else f"no {name}" for name,n in terms.items())
            instructions=list(dict.fromkeys(a['review'].get('局部连接复审',{}).get('fix','') for a in rec['reviews'] if a['review']['status'] in ('local','global')))
            positive=' '.join(filter(None,[constraint,rec['positive'],*instructions[-3:]]))
            for i in range(2):
                label=f'recompose-{i+1}'
                graph=build_draft(positive,seed+800+i,label,i)
                avoid=', '.join('duplicate '+o+', multiple '+o for o,n in terms.items() if n==1)
                if avoid and not reference_style:graph.api['3']['inputs']['text']+=', '+avoid
                hist=post_graph(graph.api,args.port,job,label,graph.ui() if reference_style else None)
                new_path=output_image(hist[str(len(graph.nodes))])
                new_path,result=check_composition(new_path,label)
                if usable(result):
                    approved=new_path;break
    if not approved:
        rec['终审退回原因']=next((v['review']['note'] for v in reversed(rec['reviews']) if v['review'].get('status')!='pass' and v['review'].get('note')), 'No draft cleared the composition review.')
        old_proof=job/'selected-proof.png'
        if old_proof.is_file():
            archive=job/'archived-proofs'
            archive.mkdir(exist_ok=True)
            key=hashlib.sha256(old_proof.read_bytes()).hexdigest()[:16]
            old_proof.replace(archive/(key+'.png'))
        rec.pop('底图',None)
        rec['status']='未达标 · 所有底图均未通过' if args.round_worker else '待人工复核 · 尚未选出底图';save(manifest,rec)
        print(rec['status'],flush=True);return 2
    rec['底图']=str(approved);rec['status']='精绘中';save(manifest,rec)
    if args.proof_only:
        proof=job/'selected-proof.png'
        deliver_file(approved,proof)
        if args.output_dir:
            from delivery import export_files
            export_files(args.output_dir,job,{'selected-proof.png':proof})
        rec['status']='底图已选 · 待目检';save(manifest,rec);print(rec['status'],proof,flush=True);return 0
    def render(source_path,stage):
        reviewer.cache_clear()
        if torch.backends.mps.is_available():torch.mps.empty_cache()
        with Image.open(source_path) as selected:
            composition=selected.convert('RGB')
            native_four=reference_style and max(composition.size)==3840
            if not native_four:
                bound=2304 if reference_style else 1536
                composition.thumbnail((bound,bound),Image.Resampling.LANCZOS)
            source=local_image(composition)
        focus_mode=rec.get('输出模式')=='全幅4K扩散与8K焦点精修'
        if reference_style:
            graph,ids=print_graph(source,prefix+'/pending-review/'+stage,seed+320,native_four=native_four,
                                  medium=rec.get('画材',''),four_steps=2 if focus_mode else 4,four_only=focus_mode)
        else:
            graph,ids=finish(source,prompt,prefix+'/pending-review/'+stage,seed+320)
        save(job/(stage+'workflow.json'),graph.ui())
        hist=post_graph(graph.api,args.port,job,stage,graph.ui() if reference_style else None)
        if focus_mode:
            small=output_image(hist[ids['small']])
            # Use actual landmarks from the approved image; there are no scene-specific coordinates.
            from evidence import ask, backend
            from construction_review import landmarks
            prior=rec.get('焦点取舍依据',{})
            reuse=bool(prior) and prior['4K指纹']==hashlib.sha256(small.read_bytes()).hexdigest()
            if not reuse:
                with backend() as address:
                    points=landmarks(address,composition)
            neural=small.with_name(small.stem+'-neural-2x.png')
            if not neural.exists():
                temp=neural.with_suffix('.partial.png')
                neural_double(small,temp);temp.replace(neural)
            with Image.open(neural) as sr, Image.open(small) as original:
                canvas=combine_source(original,sr,rec.get('画材',''))
                boxes=prior['区域'] if reuse else centers(points,canvas.size)
                rec['焦点取舍依据']={'4K指纹':hashlib.sha256(small.read_bytes()).hexdigest(),'区域':boxes}
                names=[local_image(canvas.crop((l,t,l+PATCH,t+PATCH)).convert('RGB')) for _,l,t in boxes]
            detail,nodes=detail_graph(names,rec.get('画材',''),seed+12020,prefix+'/pending-review/'+stage,roles=[name for name,_,_ in boxes])
            save(job/(stage+'-8K-detail-workflow.json'),detail.ui())
            patches=post_graph(detail.api,args.port,job,stage+'-native-detail',detail.ui())
            master=small.with_name('8K-focus-refinement.png')
            skipped=set(rec.get('撤回焦点',[])) if reuse else set()
            chosen=[(l,t,output_image(patches[k]),name) for i,((name,l,t),k) in enumerate(zip(boxes,nodes),1) if i not in skipped]
            finish_neural(small,neural,chosen,master,rec.get('画材',''))
            focus_sheet=job/(stage+'-detail-comparison.jpg')
            write_focus_proof(master,[INPUT/name for name in names],boxes,focus_sheet)
            rec['精绘区域报告']={'同位校样':focus_sheet.name,'覆盖':'4K全幅扩散或原生4K，8K全幅神经超分加焦点原像素扩散','8K图块':PATCH,'8K局部':boxes,'实际撤回的区域':sorted(skipped),'长边':[3840,7680]}
            return master,small
        if 'report' in ids:rec['精绘区域报告']=hist.get(ids['report'])
        else:rec['精绘区域报告']={'覆盖':'原生 4K、8K 分块扩散' if native_four else '两次全幅重叠分块扩散','图块':1152,'布局':'全幅等距，保形融合','长边':[3840,7680]}
        master=output_image(hist[ids['master']]);small=output_image(hist[ids['small']])
        for path,edge in ((master,7680),(small,3840)):
            with Image.open(path) as image:
                if max(image.size)!=edge:raise RuntimeError('尺寸校验失败 '+str(path))
        return master,small
    def native_pair(source_path,stage):
        dest=job/'pending-review'/stage
        dest.mkdir(parents=True,exist_ok=True)
        with Image.open(source_path) as im:
            if max(im.size)!=7680:raise ValueError('--keep-native 仅接受长边 7680 的既有母版')
            ratio=3840/max(im.size)
            scaled=im.convert('RGB').resize(tuple(round(v*ratio) for v in im.size),Image.Resampling.LANCZOS)
        master=dest/'8K.png';small=dest/'4K.png'
        deliver_file(source_path,master)
        temp=dest/(uuid.uuid4().hex+'.png');scaled.save(temp);temp.replace(small)
        return master,small
    master,small=native_pair(approved,'native-repair') if args.keep_native else render(approved,'refine')
    from print_judge import fingerprint, inspect as print_inspect, MODEL
    from production import publish
    # The model inspects the exact finished files, including every replacement.
    for attempt in range(3):
        label='final-art' if attempt==0 else f'revision-{attempt}-final-art'
        final=check(master,label,focus)
        rec['8K']=os.path.relpath(master,job);rec['4K']=os.path.relpath(small,job)
        rec['status']='离线逐块终审中';save(manifest,rec)
        # A decisive failed structural review must not be released on a second model's vote.
        if final['status']=='global':
            rec['终审退回原因']=final['note'];break
        judged=print_inspect(master,prompt,job,final,small=small,focus=rec.get('精绘区域报告',{}))
        rec['逐块终审']={'sha256':judged['sha256'],'状态':judged['status'],'已看视图':len(judged['views']),
                      '计划视图':judged['planned'],'原像素图块':judged['planned_native'],'报告':judged['report']}
        save(manifest,rec)
        if judged['status']=='pass' and usable(final):
            if judged['comparison_sha256']!=fingerprint(small):raise ValueError('4K 版本在对照审查后改变')
            with Image.open(master) as eight, Image.open(small) as four:
                if max(eight.size)!=7680 or max(four.size)!=3840 or any(abs(a-b*2)>1 for a,b in zip(eight.size,four.size)):
                    raise ValueError('定稿尺寸不一致')
            rec['自动定稿']={'model':MODEL,'模型指纹':judged['model_id'],'sha256':judged['sha256'],
                         '4K指纹':fingerprint(small),'终审报告':judged['report'],'审图版本':judged['revision'],
                         '原像素图块':judged['planned_native'],'已看视图':len(judged['views'])}
            rec.pop('终审退回原因',None)
            rec['status']='已通过离线终审'
            if not args.round_worker:publish(job,job,rec)
            save(manifest,rec);print(rec['status'],flush=True);return 0
        candidates=([judged['issue']] if judged['status']=='local' else final['issues'] if final['status']=='local' and judged['status']=='pass' else [])
        issue=next((item for item in candidates if region(item,final) and item.get('fix')),None)
        rec['终审退回原因']=judged.get('issue',{}).get('symptom') or final.get('note','')
        if not issue or attempt==2:break
        focus=region(issue,final)
        with Image.open(master) as final_image:
            native=args.keep_native or max((focus[2]-focus[0])*final_image.width,(focus[3]-focus[1])*final_image.height)/1000<=1300
        source=master if native else approved
        stage=f'print-repair-{attempt+1}'
        with Image.open(source) as original:
            if issue.get('operation')=='remove_forbidden' and ERASE_MODEL.is_file():
                revised=job/(stage+'-removed-unrequested-object.png')
                corrected=masked_candidate(original.convert('RGB'),focus)
                temp=revised.with_suffix('.partial.png')
                corrected.save(temp);temp.replace(revised)
            else:
                revised=repaint(original.convert('RGB'),focus,issue['fix'],rec['positive'],seed+650+attempt,args.port,
                                job,stage,issue.get('target'),[o for o,_ in rec['parts']],reference_style)
        if native:master,small=native_pair(revised,stage)
        else:
            approved=revised
            master,small=render(approved,stage)
    rec['status']='未达标 · 终审退回'
    save(manifest,rec);print(rec['status'],flush=True);return 2


if __name__=='__main__':
    sys.exit(main())
