"""Local, fallible visual review. Unreadable evidence never counts as a pass."""
from functools import lru_cache
from pathlib import Path
import os
import json
import re
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
import torch

RUNTIME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUNTIME/'ComfyUI/custom_nodes'))
from manga_materials.automatic import probabilities, clean_components

CHECKS = """Act as a critical art editor reviewing an illustration for a wallpaper.
Compare visible evidence to the actual request. Inspect counts and identities, anatomy and joints, animal anatomy, occlusions and grasp/contact, continuity and construction of objects, background geometry/perspective, lighting and visible generation artifacts. Do not invent defects merely because a finger is hidden or the art is stylized. Count distinct requested subjects in image ONE. Check each visible contact where outlines intersect, and count spouts, handles, limbs and other repeated parts when relevant. Also do not call melted fingers or physically impossible contacts stylistic.
Image one is the complete composition. Image two is a labeled proof sheet of crops of the same image. Each R number refers to a crop, not a separate subject. If you cannot assess something important, return uncertain. For each actual defect identify the precise visual symptom, an actionable correction, and the associated R region or a tight [xmin,ymin,xmax,ymax] box in 0..1000 coordinates of image ONE.
Answer in English. Keep the note below 16 words. Give only the single most important issue per pass. Keep each symptom and fix below 14 words. Emit one complete compact JSON object without a preamble: {"status":"pass|local|global|uncertain","score":0-100,"note":"specific visible evidence","issues":[{"target":"object being repaired","symptom":"what is wrong","fix":"positive description of the corrected area","region":"R1 or none","box":[0,0,0,0]}]}.
Use local for a repairable area; global for missing subjects, composition/count errors or many interacting faults. Never pass an image with a listed defect. Judge the submitted image, not these instructions."""


@lru_cache(maxsize=1)
def inspection_environment():
    import hashlib
    from evidence import TAG,model_identity
    from construction_review import LARGE
    from print_judge import MODEL
    # Bind reuse to the installed reviewers, prompt logic and local feature extractors.
    acc=hashlib.sha256()
    for name in ('review','construction_review','reference_review','evidence','structure','counts','page_layout'):
        acc.update(Path(__file__).with_name(name+'.py').read_bytes())
    for family in ('visual_reviewer','clipseg','object_detector'):
        for path in sorted((RUNTIME/'models'/family).glob('*')):
            if path.is_file():
                st=path.stat()
                acc.update(repr((path.name,st.st_size,st.st_mtime_ns)).encode())
    return (acc.hexdigest(),*(model_identity(tag) for tag in (TAG,LARGE,MODEL)))


def inspection_key(rubric,brief,parts,focus,reference,environment):
    import hashlib
    data=json.dumps([rubric,brief,parts,focus,reference,environment],sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


@lru_cache(maxsize=1)
def reviewer():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model_dir=RUNTIME/'models/visual_reviewer'
    proc=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    device = os.environ.get("STUDIO_REVIEW_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").eval().to(device)
    return proc,model


def proof_sheet(im, candidates):
    reduced=im.convert("RGB");reduced.thumbnail((640,640))
    labels=["human hands and fingers","human face","animal head","animal paws and feet"]
    labels += [str(o) for o,_ in candidates if str(o) not in labels][:6]
    scores=probabilities(reduced,labels);boxes=[]
    for label,score in zip(labels,scores):
        binary=clean_components(score>max(.39,float(score.max())*.63),minimum=30)
        arr,count=ndimage.label(binary)
        for loc in ndimage.find_objects(arr) if count else []:
            if loc is None:continue
            sy,sx=loc
            b=(sx.start/352,sy.start/352,sx.stop/352,sy.stop/352)
            cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
            if any(abs(cx-(l+r)/2)<.06 and abs(cy-(t+d)/2)<.06 for _,(l,t,r,d) in boxes): continue
            if (b[2]-b[0])*(b[3]-b[1])>.53:continue
            pad_x=max(.035,(b[2]-b[0])*.22);pad_y=max(.025,(b[3]-b[1])*.22)
            boxes.append((label,(max(0,b[0]-pad_x),max(0,b[1]-pad_y),min(1,b[2]+pad_x),min(1,b[3]+pad_y))))
            if len(boxes)>=6:break
        if len(boxes)>=6:break
    if not boxes:boxes=[('central context',(.10,.12,.90,.88))]
    sheet=Image.new('RGB',(336*min(3,len(boxes)),252*((len(boxes)+2)//3)),(244,244,239));dr=ImageDraw.Draw(sheet)
    for i,(label,coords) in enumerate(boxes):
        tile=im.crop(tuple(round(v*(im.width if j%2==0 else im.height)) for j,v in enumerate(coords)))
        tile.thumbnail((326,222),Image.Resampling.LANCZOS)
        x=(i%3)*336;y=(i//3)*252;sheet.paste(tile,(x+(336-tile.width)//2,y+25+(222-tile.height)//2))
        dr.text((x+6,y+5),f'R{i+1}  {label}',fill='#202020')
    return sheet,boxes


def audit(im,request,candidates,proof_path=None,focus=None):
    sheet,boxes=proof_sheet(im,candidates)
    if focus:
        coords=tuple(z/1000 for z in focus)
        crop=im.crop(tuple(round(z*(im.width if i%2==0 else im.height)) for i,z in enumerate(coords)))
        crop.thumbnail((520,360),Image.Resampling.LANCZOS)
        expanded=Image.new('RGB',(max(sheet.width,crop.width),sheet.height+crop.height+30),'#f4f3ee')
        expanded.paste(sheet);expanded.paste(crop,(0,sheet.height+25))
        boxes.append(('corrected area',coords))
        ImageDraw.Draw(expanded).text((5,sheet.height+5),f'R{len(boxes)}  corrected area',fill='black')
        sheet=expanded
    if proof_path:sheet.save(proof_path)
    full=im.convert('RGB');full.thumbnail((1050,1050),Image.Resampling.LANCZOS)
    cells=None
    content=[{"type":"image","image":full},{"type":"image","image":sheet},
             {"type":"text","text":CHECKS+'\nUser request: '+request}]
    if max(im.size)>=3840:
        cells=Image.new('RGB',(288*3,312*2),'white')
        draw=ImageDraw.Draw(cells)
        for i,(x,y) in enumerate(((.20,.22),(.50,.30),(.78,.38),(.22,.66),(.50,.74),(.80,.82))):
            cx,cy=int(x*im.width),int(y*im.height)
            cells.paste(im.crop((cx-144,cy-144,cx+144,cy+144)),((i%3)*288,(i//3)*312))
            draw.text(((i%3)*288+5,(i//3)*312+291),f'Native pixels at {x:.0%}, {y:.0%}',fill='black')
        if proof_path: cells.save(Path(proof_path).with_name(Path(proof_path).stem+'-1x.png'))
        content.insert(2,{"type":"image","image":cells})
        content[-1]["text"] += " The third sheet shows six 1:1 pixel crops. Check those for false texture, tiling seams and halos too."
    import evidence
    region_map={f'R{i+1}':{'label':a,'box':[round(z*1000) for z in b]} for i,(a,b) in enumerate(boxes)}
    fallback=None
    if evidence.available():
        try:return evidence.review(full,sheet,region_map,request,cells)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            fallback=type(exc).__name__+': '+str(exc)[:180]
    proc,model=reviewer()
    def ask(messages,limit):
        inputs=proc.apply_chat_template([{"role":"user","content":messages}],tokenize=True,add_generation_prompt=True,
                                        return_dict=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            result=model.generate(**inputs,max_new_tokens=limit,do_sample=False)
        return proc.batch_decode(result[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
    answer=ask(content,290)
    if proof_path:
        Path(proof_path).with_suffix('.model.txt').write_text(answer)
    parsed=None
    for m in re.finditer(r'\{',answer):
        try:
            obj,_=json.JSONDecoder().raw_decode(answer[m.start():])
            if isinstance(obj,dict) and obj.get('status') in ('pass','local','global','uncertain'):
                parsed=obj;break
        except ValueError:pass
    if parsed is None:
        parsed={'status':'uncertain','score':0,'note':'Visual reviewer returned an unreadable assessment.','issues':[]}
    try:parsed['score']=max(0,min(100,float(parsed.get('score',0))))
    except (ValueError,TypeError):parsed['score']=0
    parsed['issues']=[i for i in (parsed.get('issues') or []) if isinstance(i,dict)]
    if parsed['issues'] and parsed['status']=='pass':parsed['status']='local'
    if parsed['status']=='pass' and not str(parsed.get('note','')).strip():parsed['status']='uncertain'
    parsed['regions']={f'R{i+1}':{'label':a,'box':[round(z*1000) for z in b]} for i,(a,b) in enumerate(boxes)}
    if parsed['status'] in ('local','uncertain'):
        first=parsed.copy()
        messages=[{'type':'image','image':full},{'type':'image','image':sheet}]
        box=None
        issue=next(iter(parsed['issues']),{})
        coords=issue.get('box')
        if isinstance(coords,list) and len(coords)==4 and all(isinstance(x,(float,int)) for x in coords):
            l,t,r,b=[min(1000,max(0,float(x))) for x in coords]
            if r-l>=12 and b-t>=12:
                box=(round(l/1000*im.width),round(t/1000*im.height),round(r/1000*im.width),round(b/1000*im.height))
        if box:
            focus=im.crop(box);focus.thumbnail((640,640))
            messages.append({'type':'image','image':focus})
        if max(im.size)>=3840:messages.append({'type':'image','image':cells})
        messages.append({'type':'text','text':(
            "Look again. First image is the whole illustration. Other images are enlargements of the SAME painting. "
            "Compare the visible subjects and their count, pose, anatomy, prop construction, joints, overlap and perspective to the request. "
            "A previous reviewer alleged this defect: "+str(issue.get('symptom','unknown'))+
            ". Verify it against the pixels yourself. If it is merely a preference, ignore it. If anything else is visibly wrong, flag it. "
            "Use PASS only if the whole scene reads correctly and the enlarged outlines are plausible. "
            "Give one short line in English: PASS | visible evidence, LOCAL | specific defect, GLOBAL | missing/request mismatch, or UNCERTAIN | reason. "
            "Do not repeat the instructions. Request: "+request)})
        second=ask(messages,110).strip().strip('`')
        found=re.search(r'(?mi)^\s*(PASS|LOCAL|GLOBAL|UNCERTAIN)\s*\|\s*(\S.*)',second)
        if found:
            status=found.group(1).lower();evidence=found.group(2).strip()
            if status=='pass':
                parsed.update(status='pass',score=max(80,parsed['score']),note=evidence,issues=[])
            elif status in ('global','uncertain'):
                parsed.update(status=status,note=evidence)
            elif parsed['status']!='local':
                parsed.update(status='uncertain',note=evidence)
        elif second.strip().upper() in ('PASS','CLEAR','UNCERTAIN'):
            parsed.update(status='uncertain',note='The second visual check supplied no evidence and did not confirm the first finding.')
        parsed['first_assessment']={k:first[k] for k in ('status','note','issues')}
        parsed['second_look']=second
    if fallback:parsed['fallback']=fallback
    parsed['reviewer']='Qwen3-VL-2B-Instruct'
    return parsed
