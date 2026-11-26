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
from graphs import UP, draft, finish
from illustration import pigment_edit, has_drawing_model, text_composition, DRAW_UNET, installed_editor, instruction_edit, detail_canvas, NATURAL_UP, continuous_tone, continuous_tone_double, KEY_VISUAL, ANIME_PAINT, anime_color_pass, BRUSHWORK, REFERENCE, reference_draft, independent_draft, text_draft, print_graph, paint_key_visual
from reference_review import compare
from structure import inspect_structure
from construction_review import review as review_construction
from review import audit, reviewer, RUNTIME, CHECKS
from manga_materials.planner import plan_scene, requested_medium
from focus_print import PATCH, centers, sampled_surfaces, combine_source, detail_graph, finish_neural, neural_double, repair_print_pair, preserve_source_marks
from masks import object_mask, structural_composite
from lettering import remove_reviewed_feature
from erase import duplicate_box, erase_region, masked_candidate, MODEL as ERASE_MODEL

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
    # Unattended jobs share one graphics device. Rendering a large diffusion
    # canvas while another process loads the print judge can exhaust unified RAM.
    request_data={'prompt':api}
    if workflow:request_data['extra_data']={'extra_pnginfo':{'workflow':workflow}}
    payload=json.dumps(request_data).encode()
    digest=hashlib.sha256(payload).hexdigest()
    cache=directory/(stage+'-task.json')
    if cache.is_file():
        old=json.loads(cache.read_text())
        if (old.get('digest')==digest and old.get('renderer')=='ComfyUI' and old.get('output_sha256')
                and output_fingerprints(old.get('outputs',{}))==old['output_sha256']):
            return old['outputs']
    waiting=time.monotonic()
    with (RUNTIME/'vision-model-lock').open('a') as memory_lock:
        fcntl.flock(memory_lock,fcntl.LOCK_EX)
        queue_seconds=round(time.monotonic()-waiting,2)
        try:
            try:
                return _post_graph(payload,digest,port,directory,stage,queue_seconds)
            except (error.URLError,ConnectionError) as exc:
                cause=getattr(exc,'reason',exc)
                if not isinstance(cause,(ConnectionRefusedError,ConnectionResetError,BrokenPipeError)):raise
                # The local service can exit independently of a long-running job.
                # Only restart this project's offline server on its reserved port.
                sys.path.insert(0,str(STUDIO/'scripts'))
                try:
                    from studio import server
                    server(port)
                finally:
                    sys.path.pop(0)
                return _post_graph(payload,digest,port,directory,stage,queue_seconds)
        finally:
            release_idle_renderer(port)


def output_fingerprints(outputs):
    paths={}
    for record in outputs.values():
        for image in record.get('images',[]):
            if image.get('type')!='output':return None
            path=(OUTPUT/image.get('subfolder','')/image['filename']).resolve()
            if not path.is_relative_to(OUTPUT.resolve()) or not path.is_file():return None
            with path.open('rb') as stream:
                paths[str(path.relative_to(OUTPUT.resolve()))]=hashlib.file_digest(stream,'sha256').hexdigest()
    return paths or None


def _post_graph(payload, digest, port, directory, stage, queue_seconds):
    cache = directory / (stage + '-task.json')
    url = 'http://127.0.0.1:' + str(port)
    job = json.loads(cache.read_text()) if cache.exists() else {}
    if job.get('digest') != digest:
        if (job.get('renderer')=='ComfyUI' and job.get('output_sha256')
                and output_fingerprints(job.get('outputs',{}))==job['output_sha256']):
            # Another local graph may reuse the stage name. Its older pictures
            # remain eligible for later assessment only with a matching receipt.
            archive=directory/'render-history'
            archive.mkdir(exist_ok=True)
            save(archive/(stage+'-'+job['digest']+'.json'),job)
        job = {}
    cached=job.get('outputs')
    if isinstance(cached,dict) and job.get('output_sha256'):
        if output_fingerprints(cached)==job['output_sha256']:return cached
        # History authenticates the original render only. Save the regenerated
        # graph to a fresh filename so an altered or missing file cannot be adopted.
        submission=json.loads(payload)
        suffix='-restored-'+uuid.uuid4().hex[:10]
        for node in submission['prompt'].values():
            fields=node['inputs']
            if node['class_type']=='SaveImage' and 'filename_prefix' in fields:fields['filename_prefix']+=suffix
        payload=json.dumps(submission).encode()
        job={}
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
        job = {'id':ident,'digest':digest,'gpu_queue_seconds':queue_seconds}
        save(cache,job)
    while True:
        hist = get('/history/'+job['id']).get(job['id'])
        if hist:
            if hist['status']['status_str']!='success':
                cache.unlink(missing_ok=True)
                raise RuntimeError(str(hist['status']['messages'][-2:])[:2300])
            if 'renderer' not in job:
                # Keep the files referenced by Comfy's executed graph, including
                # optional local editors actually selected on this device.
                executed=hist.get('prompt',[])
                if len(executed)>2 and isinstance(executed[2],dict):
                    weights={v for node in executed[2].values() for k,v in node['inputs'].items()
                             if k in ('ckpt_name','unet_name','clip_name','lora_name','vae_name','model_name')
                             and isinstance(v,str)}
                    job.update(renderer='ComfyUI',endpoint=url,model_files=sorted(weights))
            timestamps={name:data['timestamp'] for name,data in hist['status'].get('messages',[])
                        if isinstance(data,dict) and isinstance(data.get('timestamp'),(int,float))}
            if 'execution_start' in timestamps and 'execution_success' in timestamps:
                job['renderer_seconds']=round((timestamps['execution_success']-timestamps['execution_start'])/1000,2)
            hashes=output_fingerprints(hist['outputs'])
            if hashes:job.update(outputs=hist['outputs'],output_sha256=hashes)
            save(cache,job)
            return hist['outputs']
        time.sleep(4)


def release_idle_renderer(port):
    """Return idle render caches to unified memory before the large visual reviewer starts."""
    address=f'http://127.0.0.1:{port}'
    try:
        with request.urlopen(address+'/queue',timeout=3) as reply:queue=json.load(reply)
        if not queue.get('queue_running') and not queue.get('queue_pending'):
            data=json.dumps({'unload_models':True,'free_memory':True}).encode()
            req=request.Request(address+'/free',data=data,headers={'Content-Type':'application/json'})
            with request.urlopen(req,timeout=3):pass
    except (OSError,ValueError):
        pass  # Cache release is optional. The normal checks still run.


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



def draft_repair_issue(result):
    """A localized, actionable draft finding can be repaired before paying for a print."""
    from print_judge import actionable_edit
    if result.get('status') not in ('local','uncertain'):return None
    return next((issue for issue in result.get('issues',[]) if region(issue,result)
                 and actionable_edit(issue.get('fix')) and (result['status']=='local'
                 or issue.get('operation') in ('redraw','remove_forbidden'))),None)


def repaint(im, box, description, caption, seed, port, directory, stage, target=None, neighbours=(), reference_style=False, rebuild_shape=False, reconstruction_attempt=0, allow_specialist=True):
    # Re-sample only the interior; the wider square gives diffusion scene context.
    original_request=(box,description,target)
    if reference_style and rebuild_shape:
        from construction_review import plan_local_repair
        box,target,description=plan_local_repair(im,box,target,description,requested_medium(caption) or '',directory,stage,caption)
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
    if reference_style:
        size=tuple(max(64,round(n/16)*16) for n in size)
    prepared = crop.resize(size,Image.Resampling.LANCZOS)
    prepared.putalpha(ImageOps.invert(mask.resize(size,Image.Resampling.BILINEAR)))
    prefix = (directory.relative_to(OUTPUT)/stage).as_posix()
    model_root=RUNTIME/'ComfyUI/models'
    recolor=pigment_edit(original_request[1])
    from illustration import construction_guide
    guide=construction_guide(original_request[1]) if reference_style and rebuild_shape else None
    scene_depth='whole scene as a depth reference' in original_request[1].casefold()
    specialist=(installed_editor(model_root) if allow_specialist and reference_style and rebuild_shape
                and (guide or scene_depth or recolor or reconstruction_attempt>=2) else None)
    if specialist:
        if reconstruction_attempt>=3 and not recolor and ERASE_MODEL.is_file():
            # A masked editor can still copy a malformed silhouette from its visual
            # conditioning. Give it the same surroundings on a locally cleared plate.
            clear=mask.point(lambda value:255 if value>=160 else 0)
            if clear.getbbox():
                plate=erase_region(crop,clear).resize(size,Image.Resampling.LANCZOS)
                plate.putalpha(prepared.getchannel('A'))
                prepared=plate
        source=local_image(prepared)
        direction=('Repaint the pigment and pattern of the specified surface. Keep its silhouette, pose, contact shadows '
                   'and linework. Make the requested colors and pattern clearly visible over the intended surface. '
                   if recolor else 'Reconstruct the specified part and its visible connections. ')
        reference=None
        if guide:
            file,credit=guide
            with Image.open(file) as photo:reference=local_image(photo.convert('RGB'))
            save(directory/(stage+'-reference.json'),{**credit,'input':reference})
            construction=credit['construction']
            direction+=('Image 2 shows only a physical relationship: '+construction+
                        ' Orient that relationship to the perspective and physical contact in image 1. '
                        'Keep the surrounding surfaces, scale, graphic style, and palette in image 1. ')
        elif scene_depth:
            overview=im.copy()
            overview.thumbnail((1024,1024),Image.Resampling.LANCZOS)
            reference=local_image(overview)
            save(directory/(stage+'-reference.json'),{'role':'whole composition and depth','input':reference,'target':box})
            direction+=('Image 2 is the entire same drawing. Locate the part in image 2 before revising '
                        'image 1. Keep the pose and distinguish a distant background from an actual support. ')
        edit=instruction_edit('Edit this same illustration close-up. '+direction+
                              'Preserve the surrounding objects, light and medium. Target: '+
                              str(target or 'the selected area')+'. '+description,seed,source,prefix,specialist,masked=True,
                              detail_reference=reference,full=reconstruction_attempt>=4)
        graph=edit.api
        output=str(len(edit.nodes))
    elif reference_style:
        guidance=crop
        text=description
        if rebuild_shape:
            # Preserve the surroundings, attachment and light while clearing the faulty drawing.
            subject=object_mask(crop,crop,target,mask,neighbours,confirmed_shape=True)
            if reconstruction_attempt and ERASE_MODEL.is_file():
                # Remove the previous silhouette from conditioning when diffusion has
                # already copied its defect. Retain the outer context and attachment.
                clear=subject if reconstruction_attempt==1 else mask
                # LaMa receives an explicit binary support, without Gaussian tails.
                clear=clear.point(lambda pixel: 255 if pixel>=64 else 0)
                if clear.getbbox():
                    guidance=erase_region(crop,clear)
                placeholder='locally cleared area'
            else:
                guidance=Image.composite(crop.filter(ImageFilter.GaussianBlur(max(5,span*.11))),crop,subject)
                placeholder='soft placeholder'
            named=target if target and target.casefold() not in ('local','global','none') else 'subject'
            text=('Draw a complete '+named+' in the '+placeholder+
                  '. Reconstruct its silhouette, internal parts and attachment in the local pose; '
                  'match the tapered contours, local colors, material and shadows of its surroundings. '+description)
        layout=local_image(guidance.resize(size,Image.Resampling.LANCZOS))
        edit=reference_draft(text,seed,*size,prefix,reference=layout,edit=True,
                             masked_source=local_image(prepared) if rebuild_shape else None,
                             preserve_layout=rebuild_shape and reconstruction_attempt==0)
        graph=edit.api
        output=str(len(edit.nodes))
    else:
        graph = draft(description+'. '+caption,seed,*size,prefix).api
        graph['4'] = {'class_type':'LoadImage','inputs':{'image':local_image(prepared)}}
        graph['8'] = {'class_type':'VAEEncode','inputs':{'vae':['1',2],'pixels':['4',0]}}
        graph['9'] = {'class_type':'SetLatentNoiseMask','inputs':{'samples':['8',0],'mask':['4',1]}}
        graph['5']['inputs'].update(latent_image=['9',0],steps=24,denoise=.81,cfg=5.5,sampler_name='dpmpp_2m')
        output='7'
    if specialist:
        try:
            hist=post_graph(graph,port,directory,stage)
        except RuntimeError as error:
            # The baseline editor can still run on a device/operator combination
            # that rejects the specialist graph. Its result receives the same review.
            save(directory/(stage+'-editor-fallback.json'),{'model':specialist,'reason':str(error)[:1500]})
            original_box,original_description,original_target=original_request
            return repaint(im,original_box,original_description,caption,seed,port,directory,stage,
                           original_target,neighbours,reference_style,rebuild_shape,reconstruction_attempt,False)
    else:
        hist=post_graph(graph,port,directory,stage)
    with Image.open(output_image(hist[output])) as new:
        patched = im.copy()
        proposed=new.convert('RGB').resize((cw,ch),Image.Resampling.LANCZOS)
        if rebuild_shape and reference_style:
            borders=(x==0,y==0,x+cw==w,y+ch==h)
            patched.paste(structural_composite(crop,proposed,mask,borders),(x,y))
        else:
            patched.paste(proposed,(x,y),object_mask(crop,proposed,target,mask,neighbours))
    target = directory/(stage+'-composite.png')
    patched.save(target)
    return target


def dimensions(prompt):
    if '3:2' in prompt.replace('：',':'):return (1248,832)
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
    parser.add_argument('--aspect',choices=('auto','2:3','3:2','9:16','16:9','1:1'),default='auto',help='画幅比例，默认从描述判断，未指定时为 2:3')
    parser.add_argument('--print-method',choices=('auto','restoration','tiled'),default='auto',
                        help='4K 初次精绘方式。自动模式在支持的设备上优先从底图用 SeedVR2 重建')
    parser.add_argument('--base-model',choices=('auto','qwen-image-2512','qwen-image-2512-full'),default='auto',help='纯文字底图绘制模型，默认自动搭配候选')
    parser.add_argument('--style',help='可选画材或画风，例如透明水彩')
    parser.add_argument('--accept-final',metavar='SHA256',help='目检后，按当前母版指纹标记定稿')
    parser.add_argument('--skip-focus',nargs='*',type=int,choices=(1,2,3),default=None,help='对照后撤回指定编号的8K局部采样；不带编号可全部恢复。')
    parser.add_argument('--8k',dest='eight_k',action='store_true',help='在默认 4K 成品之外生成并审查 8K')
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
    parser.add_argument('--retry-round',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--retry-observation',default='',help=argparse.SUPPRESS)
    parser.add_argument('--round-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--port',type=int,choices=(8193,8194),default=8193)
    parser.add_argument('--reviewed-image',type=Path,help='使用已经人工审定的原图，仍进行最终检查')
    parser.add_argument('--review-print',type=Path,help='对照 --reviewed-image 底图，重新验收已有的 4K 修订稿')
    parser.add_argument('--repair-box',nargs=4,type=int,metavar=('L','T','R','B'),help='手动指出需要复绘的位置，0 到 1000')
    parser.add_argument('--object',help='可选，指定需要保护轮廓的对象，如 porcelain teapot')
    parser.add_argument('--repair-text',help='这个区域应当画成什么样')
    args=parser.parse_args()
    if args.base_model!='auto':
        native_requested=any(term in (args.prompt or '').lower().replace(' ','') for term in ('原生4k','native4k'))
        if args.from_image or args.reviewed_image or args.native_4k or native_requested:
            parser.error('--base-model qwen-image-2512 适用于从文字起稿，请使用模型原生画幅')
        if not has_drawing_model(RUNTIME/'ComfyUI/models',full=args.base_model=='qwen-image-2512-full'):
            parser.error('缺少 Qwen-Image-2512 模型组件，请先运行安装器')
    args.eight_k=args.eight_k or args.full_8k_diffusion or args.keep_native or args.skip_focus is not None
    if args.print_method=='restoration':
        from restoration import available as has_restorer
        if args.eight_k:parser.error('--print-method restoration 仅适用于 4K')
        if not has_restorer(RUNTIME/'ComfyUI/models'):parser.error('缺少本地 SeedVR2，安装模型后可使用 --print-method restoration')
    if args.review_print:
        if (not args.reviewed_image or args.eight_k or args.from_image or args.repair_box or args.candidate_image
                or args.native_only or args.proof_only or args.accept_final):
            parser.error('--review-print 需要 --reviewed-image 和完整 4K 审查')
        from focus_print import validate_print_pair
        try:validate_print_pair(args.reviewed_image,args.review_print)
        except (OSError,ValueError) as exc:parser.error(str(exc))
    if args.prompt and not args.style and KEY_VISUAL not in args.prompt:
        if args.prompt.lstrip().startswith(('日系商业插画，','日系商业插画：','日系商业插画,','日系商业插画:')):
            args.style='anime-key-visual'
    if args.style and args.prompt:
        style=args.style.strip()
        if style.casefold() in ('anime-key-visual','日系商业插画'):style=KEY_VISUAL
        args.prompt=args.prompt.rstrip()+' · '+style
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
            args.eight_k=args.eight_k or earlier.get('delivery_edge',7680)==7680
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
        if not rec.get('焦点取舍依据'):
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
        rec['画材']=KEY_VISUAL if KEY_VISUAL in prompt else requested_medium(prompt)
        rec['delivery_edge']=7680 if args.eight_k else 3840
        rec['输出模式']=('全幅双级扩散' if args.full_8k_diffusion else '全幅4K扩散与8K焦点精修') if args.eight_k else '全幅4K精绘'
        rec['笔触参考']=hashlib.sha256((INPUT/BRUSHWORK).read_bytes()).hexdigest()
        save(manifest,rec)
    four_only=rec.get('delivery_edge',7680)==3840
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
    if args.base_model!='auto':rec['base_model']=args.base_model
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
        if clean:caption += ' '+clean
        if not any(v in prompt.lower() for v in ('分镜','分格','多格','四格','多联','双联','三联','两联','二联','分幅','拼贴','拼图','对照图','连环画','分屏','漫画页','panel','diptych','triptych','collage','contact sheet','comic page','storyboard','split screen','split view')):
            caption='Compose one continuous scene in an uninterrupted frame. '+caption
        aspect=rec.get('aspect','auto')
        width,height={'2:3':(832,1248),'3:2':(1248,832),'9:16':(720,1280),'16:9':(1280,720),'1:1':(1024,1024)}.get(aspect,dimensions(prompt))
        name=prefix+'/candidates/'+label
        if not reference_style:return draft(caption,draw_seed,width,height,name)
        size={(832,1248):(1248,1872),(1248,832):(1872,1248),(720,1280):(1152,2048),(1280,720):(2048,1152),(1024,1024):(1536,1536)}[width,height]
        if rec.get('原生4K'):
            size=tuple(round(v*3840/max(size)/16)*16 for v in size)
        if not prompt.isascii():
            caption+=' Original request (use its species, colours and counts): '+prompt
        # A single style reference can also copy the example pose. Give complete jobs
        # both an independent layout and a text-only alternative to select from.
        borrow=any(term in prompt.lower() for term in ('参考图','参考画','像图','reference image','reference illustration'))
        own_first=rec.get('画法自适应') and bool(rec.get('画材')) and not borrow and not rec.get('原生4K') and not args.native_only
        text_slot=(index==1 or (args.candidates==1 and args.retry_round and index==0))
        automatic_text=text_slot and not borrow and not rec.get('原生4K') and not args.native_only and not args.from_image
        requested=rec.get('base_model','auto')
        # After two rejected rounds, devote one independent text candidate to the base
        # denoiser when production actually requests a new composition.
        full=(requested=='qwen-image-2512-full' or (requested=='auto' and automatic_text
              and args.retry_round and args.retry_round.name.isdigit() and int(args.retry_round.name)>=2))
        if requested!='auto' or (automatic_text and has_drawing_model(RUNTIME/'ComfyUI/models',full=bool(full))):
            # The text generator can reconstruct the relationships directly from the
            # brief, independently of any earlier image or rejected silhouette.
            native={(832,1248):(1088,1632),(1248,832):(1632,1088),(720,1280):(1008,1792),(1280,720):(1792,1008),
                    (1024,1024):(1328,1328)}[width,height]
            direction=('Draw the finished illustration in the medium and palette specified by the author. '
                       'When unspecified, use a Japanese storybook illustration style. ')
            if full:
                native={(832,1248):(1056,1584),(1248,832):(1584,1056),(720,1280):(928,1664),
                        (1280,720):(1664,928),(1024,1024):(1328,1328)}[width,height]
            graph=text_composition(direction+caption,draw_seed,*native,name,full=bool(full))
        elif (own_first and index==0) or (not own_first and index==2 and rec.get('画法自适应')):
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
    def color_candidate(path,label,draw_seed):
        if KEY_VISUAL not in prompt or not reference_style or args.repair_box or args.from_image:
            return path
        with Image.open(path) as layout:
            try:
                workflow=json.loads(layout.info.get('prompt','{}'))
            except (TypeError,ValueError):
                workflow={}
            if isinstance(workflow,dict) and any(isinstance(node,dict) and node.get('inputs',{}).get('unet_name')==DRAW_UNET for node in workflow.values()):
                return path
            source=local_image(layout.convert('RGB'))
            size=layout.size
        destination=prefix+'/candidates/'+label+'-painted'
        if (RUNTIME/'models/checkpoints'/ANIME_PAINT).is_file() and max(size)<3840:
            if not rec.get('上色标签'):
                from evidence import ask,backend
                schema={'type':'object','properties':{'tags':{'type':'array','items':{'type':'string'},'minItems':6,'maxItems':50}},'required':['tags']}
                request_tags=('Translate the brief into concise English anime illustration tags, ordered by subject counts, identity, individual colors, action, materials, setting and lighting. '
                              'Keep each interaction intact, for example a named animal sitting on a named surface in one phrase. Follow the actual request. Preserve object-specific color names. '
                              'Do not introduce people into an empty scene. Brief: '+prompt.split(' · '+KEY_VISUAL)[0])
                with backend() as address:items=json.loads(ask(address,[],request_tags,schema))['tags']
                rec['上色标签']=', '.join(t.strip() for t in items if isinstance(t,str) and t.strip())
                save(manifest,rec)
            paint=anime_color_pass(rec['上色标签'],draw_seed,destination,source)
        else:
            paint=paint_key_visual(prompt,draw_seed,size,destination,source)
        stage=label+'-color'
        save(job/(stage+'workflow.json'),paint.ui())
        history=post_graph(paint.api,args.port,job,stage,paint.ui())
        return output_image(history[str(len(paint.nodes))])
    if args.native_only:
        args.proof_only=True
    def check(path,stage,focus=None):
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        rubric=hashlib.sha256((CHECKS+('-construction-zoom-v15' if rec.get('画法自适应') else '')+'-independent-inventory-v6'+('body-support-v5-grounded-figures'+rec['画法参考'] if reference_style else '')).encode()).hexdigest()[:12]
        from page_layout import gutters,unrequested_panels
        with Image.open(path) as probe:overview,gaps=gutters(probe)
        if gaps and args.round_worker:rubric+='-layout2'
        from review import inspection_environment,inspection_key
        context_key=inspection_key(rubric,prompt,rec['parts'],focus,rec.get('画法参考'),inspection_environment())
        from construction_review import held_pocket_reading
        def match(row):
            return row.get('hash')==key and row.get('inspection_key')==context_key and not held_pocket_reading(row['review'])
        prior=next((row for row in rec['reviews'] if match(row) and row['stage']==stage),None)
        if prior:return prior['review']
        if args.round_worker and args.retry_round and (args.retry_round/'review.json').is_file():
            previous=json.loads((args.retry_round/'review.json').read_text())
            prior=next((row for row in previous.get('reviews',[]) if match(row) and row['review']['status']=='pass'),None)
            if prior:
                for suffix in ('-inspection.jpg','-inspection-1x.png','-structure-and-support.jpg',
                               '-contact-and-occlusion.jpg','-reference-comparison.jpg'):
                    proof=args.retry_round/(prior['stage']+suffix)
                    if proof.is_file():deliver_file(proof,job/(stage+suffix))
                rec['reviews'].append(dict(prior,file=os.path.relpath(path,job),stage=stage,
                                           reused_from=str(args.retry_round),elapsed_seconds=0))
                save(manifest,rec)
                return prior['review']
        started=time.monotonic()
        if gaps and args.round_worker:
            mismatch=unrequested_panels(overview,gaps,prompt)
            if mismatch:
                result=dict(status='global',score=35,note='画面被纸白间隔切成了多个独立画幅。'+mismatch['evidence'],
                            issues=[],regions={},layout=mismatch)
                rec['reviews'].append(dict(file=os.path.relpath(path,job),hash=key,rubric=rubric,inspection_key=context_key,stage=stage,elapsed_seconds=round(time.monotonic()-started,2),review=result))
                save(manifest,rec);return result
        from evidence import review_session
        with Image.open(path) as image, review_session() as timings:
            rgb=image.convert('RGB')
            result=audit(rgb,prompt,rec['parts'],job/(stage+'-inspection.jpg'),focus=focus)
            if reference_style:
                geometry=inspect_structure(rgb,result.get('detector',[]),job/(stage+'-structure-and-support.jpg'),result.get('observed_counts'),prompt)
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
                # A rejected composition must be re-rendered and assessed again.
                # Assess workmanship on the actual version eligible to advance.
                if result['status']=='pass' or not args.round_worker:
                    comparison=compare(rgb,result.get('regions',{}),job/(stage+'-reference-comparison.jpg'),brief=prompt if rec.get('画法自适应') else '')
                    result['参考比较']=comparison
                    if result['status']=='pass' and comparison['decision']=='below':
                        result['status']='global'
                        result['note']+=' · 交换左右后仍发现画法差距'
                    elif comparison['decision']=='review':
                        result['note']+=' · 画法对照需目检'
        rec['reviews'].append(dict(file=os.path.relpath(path,job),hash=key,rubric=rubric,inspection_key=context_key,stage=stage,elapsed_seconds=round(time.monotonic()-started,2),device=timings,review=result))
        save(manifest,rec)
        print(stage,result['status'],result['note'],flush=True)
        return result
    def usable(assessment):
        return assessment['status']=='pass' and assessment['score']>=75
    approved=None
    focus=args.repair_box
    def check_composition(path,label):
        if not args.from_image and not args.repair_box:
            from lettering import clean_lettering
            path=clean_lettering(path,prompt,job,label)
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

    def check_draft_structure(path,result):
        if not usable(result) or not args.round_worker or args.proof_only:return result
        from print_judge import inspect
        report=inspect(path,prompt,job,result,construction_draft=True)
        rec.setdefault('底图结构终审',{})[os.path.relpath(path,job)]=report['report']
        save(manifest,rec)
        if report['status']=='pass':return result
        finding=report.get('issue',{})
        return dict(result,status=report['status'],issues=[finding] if finding else [],
                    note=finding.get('symptom') or '底图结构需要返修')

    retry_source=None
    retry_origin=None
    if args.round_worker and four_only:
        from production import local_retry_source
        retry_source=local_retry_source(job,prompt,interrupted=True,review_upgrade=True)
        if retry_source:
            retry_origin=job
        elif args.retry_round:
            retry_origin=args.retry_round
            retry_source=local_retry_source(retry_origin,prompt,review_upgrade=True)
        if retry_source and retry_origin==job and not (args.reviewed_image or args.from_image or args.candidate_image or args.repair_box):
            from production import failed_scene_connection
            if failed_scene_connection(job,rec,retry_source[2]):
                rec.update({'整图重绘后仍需换构图':True,'终审退回原因':retry_source[2]['symptom'],
                            'status':'未达标 · 整图重绘后仍需换构图'})
                save(manifest,rec)
                return 2
        if retry_source and retry_source[2]['operation']!='reinspect' and retry_origin!=job and not (args.reviewed_image or args.from_image or args.candidate_image or args.repair_box):
            # When a text-only job consumes a whole new round yet keeps exactly
            # the previous rejected master, let new seeds try a fresh composition.
            # User-supplied source images always keep their original composition.
            retained=retry_source[0].resolve()
            from production import failed_scene_connection
            previous=json.loads((args.retry_round/'review.json').read_text())
            if (not retained.is_relative_to(args.retry_round.resolve()) or previous.get('整图重绘后仍需换构图')
                    or failed_scene_connection(args.retry_round,previous,retry_source[2])):
                rec['重选构图原因']='上一轮局部重画未能改变被退回的母版'
                retry_source=None
    if retry_source:
        approved=retry_source[1]
    elif args.candidate_image:
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
        # A selection belongs to the supplied picture. New unrelated layouts
        # cannot establish whether a patch improves that selected area.
        count=len(paths) if args.repair_box else max(args.candidates,len(paths))
        for i in range(count):
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
            if args.repair_box and i==0:
                # The caller identified this source defect. Inspect the edited pixels
                # against the original, then apply every ordinary draft and print gate.
                result={'status':'local','score':100,'issues':[{'box':args.repair_box,'fix':args.repair_text,
                                                               'operation':'redraw','target':args.object}]}
            else:
                path=color_candidate(path,label,seed+373+i)
                path,result=check_composition(path,label)
            if usable(result) and not args.repair_box:
                passing.append((result,path,i))
            if draft_repair_issue(result):
                assessed.append((result,path,i))
        for result,path,i in sorted(passing,key=lambda item:-item[0]['score']):
            result=check_draft_structure(path,result)
            if usable(result):
                approved=path
                break
            if draft_repair_issue(result):assessed.append((result,path,i))
        if not approved:
            for result,path,i in sorted(assessed,key=lambda x:-x[0]['score']):
                issue=draft_repair_issue(result)
                if not issue:continue
                base_path=path
                for n in range(3):
                    label=f'region-repair-{i+1}-{n+1}'
                    with Image.open(base_path) as original:
                        if issue.get('operation')=='remove_forbidden' and ERASE_MODEL.is_file():
                            patched=job/(label+'-removed.png')
                            remove_reviewed_feature(original.convert('RGB'),region(issue,result),issue).save(patched)
                        else:
                            structural=reference_style and issue.get('operation')=='redraw'
                            patched=repaint(original.convert('RGB'),region(issue,result),issue['fix'],rec['positive'],
                                            seed+100+i*3+n,args.port,job,label,issue.get('target'),
                                            [o for o,_ in rec['parts']],reference_style,
                                            rebuild_shape=structural,reconstruction_attempt=n)
                            if structural:
                                from construction_review import assess_local_edit
                                with Image.open(patched) as candidate:
                                    judged=assess_local_edit(original,candidate,region(issue,result),issue['fix'],
                                                             issue.get('symptom'),prompt)
                                save(job/(label+'-draft-patch.json'),judged)
                                if judged.get('verdict')!='resolved':continue
                    next_review=check_draft_structure(patched,check(patched,label,region(issue,result)))
                    if usable(next_review):
                        approved=patched;break
                    if next_review['score']>=result['score']:
                        next_issue=draft_repair_issue(next_review)
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
                new_path=color_candidate(new_path,label,seed+1173+i)
                new_path,result=check_composition(new_path,label)
                if usable(check_draft_structure(new_path,result)):
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
    def reconstruct_print(source_path,stage,print_seed):
        from restoration import restoration_graph
        with Image.open(source_path) as original:
            input_name=local_image(original.convert('RGB'))
        graph,node=restoration_graph(input_name,prefix+'/pending-review/'+stage,3840,print_seed)
        save(job/(stage+'-workflow.json'),graph.ui())
        outputs=post_graph(graph.api,args.port,job,stage,graph.ui())
        return preserve_source_marks(source_path,output_image(outputs[node]))

    def render(source_path,stage):
        reviewer.cache_clear()
        if torch.backends.mps.is_available():torch.mps.empty_cache()
        with Image.open(source_path) as selected:
            composition=selected.convert('RGB')
            source_edge=max(composition.size)
            native_four=reference_style and source_edge>=3840
            if native_four:
                composition.thumbnail((3840,3840),Image.Resampling.LANCZOS)
            else:
                rebuilt_source=rec.get('高清底图返工',{}).get('重绘')==str(source_path)
                bound=3840 if rebuilt_source else (2304 if reference_style else 1536)
                composition.thumbnail((bound,bound),Image.Resampling.LANCZOS)
            source=local_image(composition)
        if four_only and stage=='refine' and source_edge<3840:
            from restoration import prefer_source_reconstruction
            # Resume a previously submitted graph with its own files and seed.
            if not (job/'refine-task.json').is_file() and prefer_source_reconstruction(
                    args.print_method,source_edge,RUNTIME/'ComfyUI/models'):
                master=reconstruct_print(source_path,'first-source-restoration',seed+28601)
                rec['初次4K方法']='SeedVR2'
                rec['精绘区域报告']={'覆盖':'全幅源图条件扩散重建','长边':[3840]}
                save(manifest,rec)
                return master,None
        focus_mode=rec.get('输出模式')=='全幅4K扩散与8K焦点精修'
        medium=rec.get('画材','')
        natural_sr=continuous_tone(medium) and (RUNTIME/'models/upscale_models'/NATURAL_UP).is_file()
        if reference_style or four_only:
            from focal_edit import print_constraints, print_subjects
            restrictions=print_constraints(prompt)
            scales=([3840] if not native_four else [])+([] if four_only or focus_mode else [7680])
            subjects={edge:print_subjects(composition,edge,job/(stage+'-tile-subjects-'+str(edge)+'.json'),constraints=restrictions) for edge in scales}
            graph,ids=print_graph(source,prefix+'/pending-review/'+stage,seed+320,native_four=native_four,
                                  medium=medium+' '+restrictions,four_steps=4,four_only=four_only or focus_mode,upscale_model=NATURAL_UP if natural_sr else UP,tile_subjects=subjects)
        else:
            graph,ids=finish(source,prompt,prefix+'/pending-review/'+stage,seed+320)
        save(job/(stage+'workflow.json'),graph.ui())
        hist=post_graph(graph.api,args.port,job,stage,graph.ui() if reference_style else None)
        if four_only:
            four=output_image(hist[ids['small']])
            rec['精绘区域报告']={'覆盖':'原生4K' if native_four else '4K全幅分块扩散','长边':[3840]}
            return preserve_source_marks(source_path,four),None
        if focus_mode:
            small=output_image(hist[ids['small']])
            # Use actual landmarks from the approved image; there are no scene-specific coordinates.
            from evidence import ask, backend
            from construction_review import landmarks
            prior=rec.get('焦点取舍依据',{})
            small_hash=hashlib.sha256(small.read_bytes()).hexdigest()
            reuse=bool(prior) and prior['4K指纹']==small_hash
            if not reuse:
                with backend() as address:
                    points=landmarks(address,composition)
            neural=small.with_name(small.stem+'-neural-2x.png')
            if natural_sr:
                with Image.open(small) as four:
                    sr_input=local_image(four.convert('RGB'))
                sr_graph,node=continuous_tone_double(sr_input,prefix+'/pending-review/'+stage+'/continuous-tone-2x')
                sr_hist=post_graph(sr_graph.api,args.port,job,stage+'-continuous-tone-2x',sr_graph.ui())
                neural=output_image(sr_hist[node])
            elif not neural.exists():
                temp=neural.with_suffix('.partial.png')
                neural_double(small,temp);temp.replace(neural)
            # Retain source frequencies above the 4K sampling limit for large imported art.
            base_guide=source_path if source_edge>3840 else small
            with Image.open(neural) as sr, Image.open(base_guide) as original:
                canvas=combine_source(original,sr,rec.get('画材',''))
                boxes=prior['区域'] if reuse else centers(points,canvas.size)
                rec['焦点取舍依据']={'4K指纹':small_hash,'区域':boxes}
                if not reuse:save(manifest,rec)
                surfaces=sampled_surfaces(original,canvas.size,boxes)
                redraw=[*boxes,*surfaces]
                names=[local_image(canvas.crop((l,t,l+PATCH,t+PATCH))) for _,l,t in redraw]
                from focal_edit import describe_tiles
                descriptions=describe_tiles(canvas,names,redraw,PATCH,INPUT,job/(stage+'-tile-subjects.json'))
            master=small.with_name('8K-focus-refinement.png')
            skipped=set(rec.get('撤回焦点',[])) if reuse else set()
            chosen=[]
            for offset in range(0,len(redraw),6):
                batch=redraw[offset:offset+6]
                batch_stage=stage+'-detail-'+str(offset)
                prefix8=prefix+'/pending-review/'+batch_stage
                detail,nodes=detail_graph(names[offset:offset+6],medium+' '+restrictions,seed+12020+offset,
                                          prefix8,roles=[role for role,_,_ in batch],descriptions=descriptions[offset:offset+6])
                save(job/(batch_stage+'-workflow.json'),detail.ui())
                patches=post_graph(detail.api,args.port,job,batch_stage,detail.ui())
                chosen.extend((l,t,output_image(patches[k]),role) for i,((role,l,t),k) in enumerate(zip(batch,nodes),offset+1) if i not in skipped)
            finish_neural(base_guide,neural,chosen,master,rec.get('画材',''))
            master=preserve_source_marks(source_path,master)
            small=preserve_source_marks(source_path,small)
            focus_sheet=job/(stage+'-detail-comparison.jpg')
            if boxes:write_focus_proof(master,[INPUT/name for name in names[:len(boxes)]],boxes,focus_sheet)
            rec['精绘区域报告']={'同位校样':focus_sheet.name,'覆盖':'4K全幅扩散或原生4K，8K全幅神经超分，焦点与有结构区域按原像素重绘','8K图块':PATCH,'8K局部':boxes,'全幅纹理选区':surfaces,'实际撤回的区域':sorted(skipped),'长边':[3840,7680]}
            return master,small
        if 'report' in ids:rec['精绘区域报告']=hist.get(ids['report'])
        else:rec['精绘区域报告']={'覆盖':'原生 4K、8K 分块扩散' if native_four else '两次全幅重叠分块扩散','图块':1152,'布局':'全幅等距，保形融合','长边':[3840,7680]}
        master=output_image(hist[ids['master']]);small=output_image(hist[ids['small']])
        for path,edge in ((master,7680),(small,3840)):
            with Image.open(path) as image:
                if max(image.size)!=edge:raise RuntimeError('尺寸校验失败 '+str(path))
        if reference_style:
            master=preserve_source_marks(source_path,master)
            small=preserve_source_marks(source_path,small)
        return master,small
    def native_pair(source_path,stage,prior=None):
        dest=job/'pending-review'/stage
        dest.mkdir(parents=True,exist_ok=True)
        with Image.open(source_path) as im:
            if max(im.size)!=7680:raise ValueError('--keep-native 仅接受长边 7680 的既有母版')
            ratio=3840/max(im.size)
            scaled=(repair_print_pair(*prior,im) if prior else
                    im.convert('RGB').resize(tuple(round(v*ratio) for v in im.size),Image.Resampling.LANCZOS))
        master=dest/'8K.png';small=dest/'4K.png'
        deliver_file(source_path,master)
        temp=dest/(uuid.uuid4().hex+'.png');scaled.save(temp);temp.replace(small)
        return master,small
    def correct_shape(source,issue,box,stage,edit_seed,previous=None):
        from construction_review import assess_local_edit
        instruction=issue['fix']
        with Image.open(source) as original:
            before=original.convert('RGB')
        from production import repeated_rejection
        recurring=bool(previous and repeated_rejection(previous,box))
        strategies=[2,0,1] if recurring else [0,1,2]
        for variant,strategy in enumerate(strategies):
            name=stage if variant==0 else f'{stage}-variant-{variant+1}'
            revised=repaint(before,box,instruction,rec['positive'],edit_seed+variant*1753,args.port,
                            job,name,issue.get('target'),[o for o,_ in rec['parts']],reference_style,rebuild_shape=True,
                            reconstruction_attempt=strategy)
            release_idle_renderer(args.port)
            with Image.open(revised) as proposed:
                verdict=assess_local_edit(before,proposed,box,issue['fix'],issue.get('symptom'),prompt)
            with source.open('rb') as original_file, revised.open('rb') as edited_file:
                verdict.update(source_sha256=hashlib.file_digest(original_file,'sha256').hexdigest(),
                               candidate_sha256=hashlib.file_digest(edited_file,'sha256').hexdigest(),
                               source_file=os.path.relpath(source,job),candidate_file=os.path.relpath(revised,job))
            verdict.update(target_box=box,reconstruction_attempt=strategy)
            if strategy==2 and verdict.get('verdict') in ('unchanged','still_present') and not pigment_edit(instruction):
                # The specialist reproduced a known geometric error. The next attempt
                # rebuilds the interior using a cleared visual reference.
                if 3 not in strategies:strategies.append(3)
            if (strategy==3 and recurring and verdict.get('verdict') in ('unchanged','still_present')
                    and installed_editor(RUNTIME/'ComfyUI/models')):
                # A sequence of rejected short edits can repeat the same geometry. Give
                # the full base editor one attempt before a complete scene redraw.
                strategies.append(4)
            save(job/(name+'-patch-review.json'),verdict)
            if verdict['verdict']=='resolved':
                return revised
            instruction=(issue['fix']+' Recompose the faulty local forms and their contact within the same framing. '
                         +str(verdict.get('next_edit') or 'Redraw the silhouette and its point of support.'))
        return None

    scene_tried=False
    def redraw_scene(source,issue,box,stage):
        nonlocal scene_tried
        if (scene_tried or not reference_style or args.keep_native or args.repair_box or args.selection_mask
                or not box or issue.get('operation')!='redraw'):
            return None
        editor=installed_editor(RUNTIME/'ComfyUI/models')
        if not editor:return None
        scene_tried=True
        guided=bool(args.reviewed_image or args.from_image or args.candidate_image)
        def screen_scene(path,label,known_box):
            reviewed=check_draft_structure(path,check(path,label,known_box))
            if usable(reviewed):return path
            finding=draft_repair_issue(reviewed)
            if not finding or finding.get('operation')!='redraw':return None
            # Keep a promising new composition. A single localized flaw can be
            # repaired at drawing scale before another full-frame print is built.
            repair_box=region(finding,reviewed)
            corrected=correct_shape(path,finding,repair_box,label+'-draft-repair',seed+8311)
            if not corrected:return None
            if guided:
                from print_judge import same_composition
                if not same_composition(source,corrected,prompt):return None
            final_draft=check_draft_structure(corrected,check(corrected,label+'-draft-corrected',repair_box))
            return corrected if usable(final_draft) else None

        from production import previous_scene_candidate, previous_scene_on_master
        from print_judge import same_composition
        prior=previous_scene_candidate(job,stage,rec['reviews'],OUTPUT,structural_reports=rec.get('底图结构终审'))
        if not prior and args.retry_round:
            prior=previous_scene_on_master(args.retry_round,source,prompt,OUTPUT)
        if prior:
            existing,weights=prior
            if not guided or same_composition(source,existing,prompt):
                reassessed=screen_scene(existing,stage+'-rescreen',box if guided else None)
                if reassessed:
                    rec['高清底图返工']={'源图':str(source),'重绘':str(reassessed),'依据':issue['fix'],'模型':weights,
                                     '保留原构图':guided,'目标范围':box if guided else None}
                    save(manifest,rec)
                    return reassessed
        with Image.open(source) as canvas:
            scale=min(1,(1536*1024/(canvas.width*canvas.height))**.5)
            size=tuple(max(32,round(n*scale/16)*16) for n in canvas.size)
            rgb=canvas.convert('RGB')
            full=rgb.resize(size,Image.Resampling.LANCZOS)
            # The editor internally reduces its semantic view to roughly 384x384.
            # Supply native neighboring contours for small defects in a wide composition.
            from print_judge import expanded
            pixels=tuple(round(v*n/1000) for v,n in zip(box,rgb.size*2))
            field=expanded(pixels,rgb.size)
            if guided:
                # A loose mask allows rerouting the attachment at scene scale while
                # anchoring the selected composition outside its surrounding field.
                field_px=tuple(round(v*target/origin) for v,target,origin in zip(field,size*2,rgb.size*2))
                feather=max(2,round(min(field_px[2]-field_px[0],field_px[3]-field_px[1])*.055))
                mask=Image.new('L',size)
                ImageDraw.Draw(mask).rounded_rectangle(field_px,radius=feather*2,fill=255)
                full.putalpha(ImageOps.invert(mask.filter(ImageFilter.GaussianBlur(feather))))
            reference=local_image(full)
            detail=rgb.crop(field)
            detail.thumbnail((1152,1152),Image.Resampling.LANCZOS)
            detail_reference=local_image(detail)
        center=((box[0]+box[2])/20,(box[1]+box[3])/20)
        text=('Edit image 1 as a complete illustration. Image 2 enlarges the defect in this same drawing. '
              'Use its surrounding contours to locate and repair that part in image 1. Output the complete '
              'scene as one uninterrupted image. Preserve its framing, '
              'subjects, identities, gestures, clothes, colors, lighting, and original drawing medium. '
              'The local crop editor could not construct this part with coherent connections to its '
              'surroundings. Redraw the part together with its connected structures in the full composition. '
              f'Location about {center[0]:.0f}% of width from left and {center[1]:.0f}% of height from top. '
              'Required correction: '+issue['fix']+' Original art direction: '+prompt)
        graph=instruction_edit(text,seed+1621,reference,prefix+'/'+stage,editor,masked=guided,
                               detail_reference=detail_reference)
        try:
            history=post_graph(graph.api,args.port,job,stage,graph.ui())
        except RuntimeError as exc:
            save(job/(stage+'-editor-failure.json'),{'model':editor,'error':str(exc)[:1200]})
            return None
        candidate=output_image(history[str(len(graph.nodes))])
        release_idle_renderer(args.port)
        from print_judge import same_composition
        aligned=same_composition(source,candidate,prompt)
        if not aligned and guided:return None
        # An unconstrained text brief may legitimately change layout. Evaluate it
        # like a new draft and locate its subjects afresh; old coordinates no longer apply.
        candidate=screen_scene(candidate,stage,box if aligned else None)
        if not candidate:return None
        rec['高清底图返工']={'源图':str(source),'重绘':str(candidate),'依据':issue['fix'],'模型':editor,
                         '保留原构图':aligned,'目标范围':box if aligned else None}
        save(manifest,rec)
        return candidate

    repair_check=None
    if retry_source:
        last,_,issue=retry_source
        focus=region(issue,{})
        # Check a reported border problem on the retained pixels before redrawing
        # a body or object that intentionally continues beyond the canvas.
        from print_judge import FRAME_CLAIM, verify_frame_exit, excluded_by_brief, exclusion_repair, MODEL as PRINT_MODEL
        # Earlier generated edits can introduce a seam that becomes part of the
        # next editor's conditioning. Test a previous version against current pixels
        # only when every pixel outside the damaged region is exactly the same.
        reverted=None
        if focus and issue.get('operation')=='redraw':
            from production import earlier_local_version, repeated_rejection
            from print_judge import REVISION
            if repeated_rejection(retry_origin,focus):
                alternative=earlier_local_version(job,last,retry_source[1],focus,prompt,REVISION)
                if alternative:
                    earlier,history=alternative
                    from construction_review import compare_prior_version
                    with Image.open(last) as current_art, Image.open(earlier) as older_art:
                        verdict=compare_prior_version(current_art,older_art,focus,issue['fix'],issue.get('symptom'),prompt)
                    save(job/('prior-version-'+history['current'][:12]+'-'+history['earlier'][:12]+'-review.json'),
                         {**history,'verdict':verdict})
                    if verdict.get('verdict')=='resolved' or verdict.get('eligible_for_full_review'):
                        reverted=earlier
        if not reverted and focus and issue.get('verify_part_identity'):
            from print_judge import verify_depicted_connection, verify_part_identity
            from evidence import backend
            with Image.open(last) as source,backend(PRINT_MODEL) as address:
                identity=verify_part_identity(address,source,focus,issue)
                contact=None
                if not (identity and identity['disproved']):
                    contact=verify_depicted_connection(address,source,focus,issue.get('symptom',''),prompt)
            if contact:
                save(job/'continued-visible-connection.json',contact)
                if contact['disproved']:
                    issue=dict(issue,operation='reinspect',symptom=contact['verification']['trace'])
            if identity:
                save(job/'continued-part-identity.json',identity)
                if identity['disproved']:
                    issue=dict(issue,operation='reinspect',symptom='Check the visible attachment. '+identity['verification']['trace'])
        if focus and issue.get('unresolved_exclusion'):
            from evidence import backend
            with Image.open(last) as source,backend(PRINT_MODEL) as address:
                plan=exclusion_repair(address,source,focus,prompt,excluded_by_brief(prompt),issue.get('symptom',''))
            if plan:
                save(job/'continued-exclusion-plan.json',plan)
                issue=dict(issue,operation='redraw' if plan['action']=='clear_surface' else 'remove_forbidden',
                           target=plan['part'],fix=plan['edit'])
        claim=issue.get('symptom','')+' '+issue['fix']
        framing=None
        if focus and FRAME_CLAIM.search(claim) and (min(focus[:2])<=12 or max(focus[2:])>=988):
            from evidence import backend
            with Image.open(last) as source, backend(PRINT_MODEL) as address:
                overview=source.copy();overview.thumbnail((1152,1152),Image.Resampling.LANCZOS)
                framing=verify_frame_exit(address,source,overview,focus,claim,prompt)
            if framing:save(job/'continued-framing-review.json',framing)
        scene=None
        if reverted:
            repaired=reverted
        elif issue.get('operation')=='reinspect' or (framing and framing['explained']):
            repaired=last
        else:
            stage='continued-repair'
            if retry_origin==job:
                with last.open('rb') as stream:
                    stage='resume-'+hashlib.file_digest(stream,'sha256').hexdigest()[:16]+'-repair'
            if focus and issue.get('operation')=='remove_forbidden' and ERASE_MODEL.is_file():
                repaired=job/(stage+'-removed-unrequested-object.png')
                with Image.open(last) as retained:remove_reviewed_feature(retained.convert('RGB'),focus,issue).save(repaired)
            else:
                from production import repeated_rejection
                if focus and repeated_rejection(retry_origin,focus,minimum=3):
                    scene=redraw_scene(last,issue,focus,'continued-scene')
                repaired=correct_shape(last,issue,focus,stage,seed+820,retry_origin) if focus and not scene else None
        if not repaired and not scene:scene=redraw_scene(last,issue,focus,'continued-scene')
        if repaired or scene:
            if scene:
                approved=scene
                master,small=render(approved,'continued-scene-print')
            else:
                master,small=repaired,None
            if scene and rec['高清底图返工'].get('保留原构图') is False:focus=None
            repair_check={'box':focus,'symptom':issue.get('symptom') or issue['fix']} if focus else None
            rec['续修来源']=str(last)
        else:
            # None of the new candidates improved the defect. The prior, hash-verified
            # negative receipt still describes these exact pixels and can be carried
            # forward without asking a vision model to regrade an unchanged failure.
            previous=json.loads((retry_origin/'review.json').read_text())
            entry=dict(previous['逐块终审'])
            entry['报告']=os.path.relpath((retry_origin/entry['报告']).resolve(),job)
            rec.update({'4K':os.path.relpath(last,job),'逐块终审':entry,
                        '终审退回原因':issue.get('symptom') or issue['fix'],'status':'未达标 · 本轮局部重绘未获通过'})
            save(manifest,rec)
            print(rec['status'],flush=True)
            return 2
    elif args.review_print:
        from print_judge import fingerprint,reviewed_print_layout
        supplied=args.review_print.resolve()
        layout=reviewed_print_layout(approved,supplied,prompt,job/'composition-evidence.json')
        rec['修订稿构图核对']='composition-evidence.json'
        if not layout['eligible_for_inspection']:
            rec.update(status='未达标 · 修订稿与底图构图不一致',终审退回原因=layout['reading']['differences'])
            save(manifest,rec)
            return 2
        master=job/'submitted-print.png'
        expected=fingerprint(supplied)
        if not master.is_file() or fingerprint(master)!=expected:
            temp=job/(uuid.uuid4().hex+'.png')
            try:
                shutil.copyfile(supplied,temp)
                if fingerprint(temp)!=expected:raise ValueError('4K 修订稿复制校验失败')
                temp.replace(master)
            finally:temp.unlink(missing_ok=True)
        small=None
        rec['已有修订稿']={'来源':str(supplied),'指纹':fingerprint(master),'底图指纹':fingerprint(approved)}
    else:
        master,small=native_pair(approved,'native-repair') if args.keep_native else render(approved,'refine')
    from print_judge import actionable_edit, fingerprint, inspect as print_inspect, MODEL
    from production import publish
    detail_rebuilt=False
    tried_focus_fields=set()
    # The model inspects the exact finished files, including every replacement.
    for attempt in range(3):
        label='final-art' if attempt==0 else f'revision-{attempt}-final-art'
        from focus_print import fit_print_aspect
        aspect=rec.get('aspect','auto')
        master=fit_print_aspect(master,aspect,3840 if four_only else 7680)
        if small is not None:
            small=fit_print_aspect(small,aspect,3840)
        release_idle_renderer(args.port)
        if four_only and not args.repair_box and not args.keep_native:
            from lettering import clean_lettering
            master=clean_lettering(master,prompt,job,label)
        final=check(master,label,focus)
        rec['4K']=os.path.relpath(master if four_only else small,job)
        if not four_only:rec['8K']=os.path.relpath(master,job)
        rec['status']='离线逐块终审中';save(manifest,rec)
        # A decisive failed structural review must not be released on a second model's vote.
        if final['status']=='global':
            rec['终审退回原因']=final['note'];break
        print_focus=dict(rec.get('精绘区域报告') or {})
        if repair_check:print_focus['repair_check']=repair_check
        baseline=approved if four_only else small
        judged=print_inspect(master,prompt,job,final,small=baseline,focus=print_focus,
                             draft=None if four_only else approved,source_comparison=four_only)
        if four_only:rec['底图']=str(baseline)
        rec['逐块终审']={'sha256':judged['sha256'],'状态':judged['status'],'已看视图':len(judged['views']),
                      '计划视图':judged['planned'],'原像素图块':judged['planned_native'],'报告':judged['report']}
        save(manifest,rec)
        if judged['status']=='pass' and usable(final):
            if judged['comparison_sha256']!=fingerprint(baseline):raise ValueError('对照图在审查后改变')
            with Image.open(master) as finished:
                if max(finished.size)!=rec.get('delivery_edge',7680):raise ValueError('定稿尺寸不一致')
                if small is not None:
                    with Image.open(small) as four:
                        if max(four.size)!=3840 or any(abs(a-b*2)>1 for a,b in zip(finished.size,four.size)):
                            raise ValueError('定稿尺寸不一致')
            rec['自动定稿']={'model':MODEL,'模型指纹':judged['model_id'],'sha256':judged['sha256'],
                         '4K指纹':judged['sha256'] if four_only else judged['comparison_sha256'],
                         '对照指纹':judged['comparison_sha256'],'终审报告':judged['report'],'审图版本':judged['revision'],
                         '原像素图块':judged['planned_native'],'已看视图':len(judged['views'])}
            rec.pop('终审退回原因',None)
            rec['status']='已通过离线终审'
            if not args.round_worker:publish(job,job,rec)
            save(manifest,rec);print(rec['status'],flush=True);return 0
        # A localized, actionable suspicion can produce a *candidate*; it still has
        # to beat the source on the native crop and pass the full review to be released.
        finding=judged.get('issue',{})
        from production import same_repair_region
        scene_record=rec.get('高清底图返工',{})
        if (args.round_worker and not any((args.reviewed_image,args.from_image,args.candidate_image,args.repair_box))
                and scene_record.get('模型') and judged['status'] in ('local','uncertain')
                and same_repair_region(scene_record.get('目标范围'),finding.get('box'))):
            # A fully redrawn composition still defeats this same connection. A text
            # brief can choose a new layout; preserve the intermediate for inspection.
            rec['整图重绘后仍需换构图']=True
            rec['终审退回原因']=finding.get('symptom') or finding.get('fix')
            break
        if judged['status']=='local' or (judged['status']=='uncertain' and finding.get('operation') in ('redraw','remove_forbidden')):
            candidates=[finding]
        elif final['status']=='local' and judged['status']=='pass':
            candidates=final['issues']
        else:
            candidates=[]
        issue=next((item for item in candidates if region(item,final) and actionable_edit(item.get('fix'))),None)
        rec['终审退回原因']=judged.get('issue',{}).get('symptom') or final.get('note','')
        repeated=False
        if issue and repair_check:
            old=repair_check['box']
            new=region(issue,final)
            repeated=(min(old[2],new[2])>max(old[0],new[0]) and min(old[3],new[3])>max(old[1],new[1]))
        with Image.open(approved) as original:
            native_edge=max(original.size)
            low_resolution=native_edge<1024
            rebuild_size=detail_canvas(original.size)
        if four_only and attempt<2 and not (args.keep_native or args.selection_mask or args.repair_box):
            from focal_edit import failed_focus, print_constraints, repaint_definition
            pixels=failed_focus(judged)
            from restoration import available as restoration_available
            needs_detail=pixels or finding.get('category')=='unresolved_focus'
            if (attempt==0 and needs_detail and not repair_check and not args.candidate_image
                    and rec.get('初次4K方法')!='SeedVR2' and args.print_method!='tiled'
                    and native_edge<3840 and restoration_available(RUNTIME/'ComfyUI/models')):
                # Compare against the actual selected drawing. A final with a later structural
                # repair must not be replaced by an earlier, uncorrected source.
                stage='source-restoration'
                prior_master=master
                master=reconstruct_print(approved,stage,seed+28601)
                small=None
                rec['4K源图重建']={'底图':str(approved),'前一版本':str(prior_master),'模型任务':stage+'-task.json'}
                save(manifest,rec)
                continue
            if pixels and pixels not in tried_focus_fields:
                tried_focus_fields.add(pixels)
                stage=f'print-definition-{attempt+1}'
                revised=repaint_definition(master,pixels,rec.get('画材','')+' '+print_constraints(prompt),seed+17041+attempt,job,stage,
                                           args.port,prefix+'/pending-review/'+stage,local_image=local_image,
                                           post_graph=post_graph,output_image=output_image,models=RUNTIME/'ComfyUI/models')
                if revised:
                    with Image.open(master) as img:
                        focus=[round(v*1000/n) for v,n in zip(pixels,img.size*2)]
                    repair_check={'box':focus,'symptom':finding['symptom']}
                    receipt=revised.name.removesuffix('-refined.png')+'-definition.json'
                    rec.setdefault('4K焦点重绘',[]).append({'核验':receipt,'范围':focus})
                    master,small=revised,None
                    save(manifest,rec)
                    continue
        if (reference_style and not args.keep_native and not detail_rebuilt and attempt<2 and
                judged.get('issue',{}).get('category')=='unresolved_focus' and (not issue or repeated or low_resolution)
                and rebuild_size is not None):
            # A verified focus failure across the picture, or one that survived a local repair,
            # needs a higher-resolution drawing of the scene before another print pass.
            detail_rebuilt=True
            repair_check=None
            stage=f'detail-source-{attempt+1}'
            with Image.open(approved) as original:
                size=rebuild_size
                reference=local_image(original.convert('RGB'))
            fix=judged['issue'].get('fix') or judged['issue']['symptom']
            caption=('Reconstruct the same composition in its original medium. Keep subject identity, count, pose, placement, palette and light. '
                     'Resolve focal structures at this drawing scale, respecting soft focus and flat regions. Observed problem: '+fix+' Scene: '+prompt)
            graph=reference_draft(caption,seed+830+attempt,*size,prefix+'/'+stage,reference=reference,edit=True)
            history=post_graph(graph.api,args.port,job,stage,graph.ui())
            candidate=output_image(history[str(len(graph.nodes))])
            rebuilt=check(candidate,stage)
            if not usable(rebuilt):break
            from print_judge import same_composition
            if not same_composition(approved,candidate,prompt):break
            rec['高清底图返工']={'源图':str(approved),'重绘':str(candidate),'依据':fix}
            approved=candidate
            master,small=render(approved,stage+'-print')
            continue
        if not issue or attempt==2:break
        focus=region(issue,final)
        repair_check={'box':focus,'symptom':issue.get('symptom') or issue['fix']}
        with Image.open(master) as final_image:
            native=args.keep_native or max((focus[2]-focus[0])*final_image.width,(focus[3]-focus[1])*final_image.height)/1000<=1300
        source=master if native else approved
        stage=f'print-repair-{attempt+1}'
        with Image.open(source) as original:
            if issue.get('operation')=='remove_forbidden' and ERASE_MODEL.is_file():
                revised=job/(stage+'-removed-unrequested-object.png')
                corrected=remove_reviewed_feature(original.convert('RGB'),focus,issue)
                temp=revised.with_suffix('.partial.png')
                corrected.save(temp);temp.replace(revised)
            elif reference_style and issue.get('operation')=='redraw':
                revised=correct_shape(source,issue,focus,stage,seed+650+attempt)
                if revised is None:
                    candidate=redraw_scene(master,issue,focus,stage+'-scene')
                    if candidate is None:break
                    approved=candidate
                    if rec['高清底图返工'].get('保留原构图') is False:
                        focus=None
                        repair_check=None
                    master,small=render(approved,stage+'-scene-print')
                    continue
            else:
                revised=repaint(original.convert('RGB'),focus,issue['fix'],rec['positive'],seed+650+attempt,args.port,
                                job,stage,issue.get('target'),[o for o,_ in rec['parts']],reference_style,
                                rebuild_shape=issue.get('operation')=='redraw')
        if native and four_only:master=revised
        elif native:master,small=native_pair(revised,stage,(small,master))
        else:
            approved=revised
            master,small=render(approved,stage)
    rec['status']='未达标 · 终审退回'
    save(manifest,rec);print(rec['status'],flush=True);return 2


if __name__=='__main__':
    sys.exit(main())
