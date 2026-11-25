"""Independent, prompt-blind image inventory and local print review."""
import fcntl
import hashlib
import base64
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from urllib import request
import torch
from counts import reconcile, animals

RUNTIME = Path(__file__).resolve().parents[1]
TAG = 'qwen3-vl:8b-instruct'
WEIGHTS = RUNTIME/'models/visual_editor'
INVENTORY = """Look at the picture itself. No target description is available. Enumerate EACH separate human individual visible, including people in the background. Describe each by distinct position, hair and clothing. Mention a possible mirror or portrait separately. Count the people and animals depicted as subjects of this scene in whatever medium is used. In a drawing, clay diorama, felt picture or silhouette, the represented human or animal counts. Report a portrait on a book cover, shadow, reflection or other picture within the scene separately. Describe the main objects, their construction and where hands, paws or objects make contact. Specify visible anomalies or uncertainty in the outlines. Report observations in fewer than 95 English words. Do not grade the image."""
SCHEMA = {'type':'object','properties':{
    'status':{'enum':['pass','local','global','uncertain']}, 'score':{'type':'integer'},
    'note':{'type':'string'},'target':{'type':'string'},'fix':{'type':'string'},
    'region':{'type':'string'},'box':{'type':'array','items':{'type':'integer'},'minItems':4,'maxItems':4}},
    'required':['status','score','note','target','fix','region','box']}


def model_identity(tag):
    family,version=tag.split(':',1)
    for tree in ('manifests','manifests-v2'):
        path=WEIGHTS/tree/'registry.ollama.ai/library'/family/version
        if path.is_file():return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def available(tag=TAG):
    return bool(shutil.which('ollama') or Path('/Applications/Ollama.app/Contents/Resources/ollama').is_file()) and bool(model_identity(tag))


@contextmanager
def backend(tag=TAG):
    """The inference service inherits the launcher's outbound-network denial."""
    binary=shutil.which('ollama') or '/Applications/Ollama.app/Contents/Resources/ollama'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    address=f'127.0.0.1:{port}'
    env={**os.environ,'OLLAMA_HOST':address,'OLLAMA_MODELS':str(WEIGHTS),'OLLAMA_NO_CLOUD':'1','OLLAMA_NUM_PARALLEL':'1'}
    with tempfile.TemporaryFile() as log, (RUNTIME/'vision-model-lock').open('a') as memory_lock:
        fcntl.flock(memory_lock,fcntl.LOCK_EX)
        proc=subprocess.Popen([binary,'serve'],env=env,stdout=log,stderr=log)
        try:
            for _ in range(60):
                if proc.poll() is not None:raise RuntimeError('本地视觉服务未能启动')
                try:
                    with request.urlopen('http://'+address+'/api/tags',timeout=2) as response:
                        if tag in [m['name'] for m in json.load(response)['models']]:break
                except OSError:pass
                time.sleep(.3)
            else:raise RuntimeError('本地视觉模型未就绪')
            yield 'http://'+address+'/api/chat'
        finally:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


def ask(url,images,prompt,schema=None,tag=TAG,*,reason=False):
    encoded=[]
    for image in images:
        buf=io.BytesIO()
        if tag=='qwen3.8:27b':image.save(buf,format='PNG')
        else:image.save(buf,format='JPEG',quality=94)
        encoded.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    data={'model':tag,'stream':False,'options':{'temperature':0,'num_ctx':8192,'num_predict':360},
          'messages':[{'role':'user','content':prompt,'images':encoded}]}
    if tag=='qwen3.8:27b':data['think']=reason
    if schema:data['format']=schema
    for attempt,limit in enumerate((1800,3200) if reason else (360,900,1400)):
        data['options']['num_predict']=limit
        if attempt:
            data['messages'][0]['content']=prompt+' Keep every description under 20 words. Return a complete compact answer.'
            if schema:data['messages'][0]['content']+=' Use one JSON object with these keys: '+', '.join(schema.get('required',[]))+'.'
            data['options'].update(temperature=.12*attempt,seed=1709+attempt)
        req=request.Request(url,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with request.urlopen(req,timeout=300) as response:reply=json.load(response)
        text=reply['message']['content'].strip()
        if schema:
            try:
                value=json.loads(text)
                if isinstance(value,dict) and all(k in value for k in schema.get('required',[])):return text
            except ValueError:pass
        elif reply.get('done_reason')!='length' and text:return text
    raise ValueError('本地审图连续输出不完整，已保留校样')


@lru_cache(maxsize=1)
def detector():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    net=fasterrcnn_mobilenet_v3_large_fpn(weights=None,weights_backbone=None).eval()
    path=RUNTIME/'models/object_detector/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    return net,FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.meta['categories']


def objects(image):
    from torchvision.transforms.functional import pil_to_tensor
    net,names=detector()
    with torch.inference_mode():pred=net([pil_to_tensor(image).float()/255])[0]
    found=[]
    for label,score,box in zip(pred['labels'],pred['scores'],pred['boxes']):
        if score<.62:continue
        name=names[int(label)]
        l,t,r,b=[round(v*1000/n) for v,n in zip(box.tolist(),(*image.size,*image.size))]
        area=(r-l)*(b-t)
        duplicate=False
        for row in found:
            if row['name']!=name:continue
            x,y,u,v=row['box']
            inter=max(0,min(r,u)-max(l,x))*max(0,min(b,v)-max(t,y))
            if inter>.68*min(area,(u-x)*(v-y)):duplicate=True;break
        if not duplicate:found.append({'name':name,'box':[l,t,r,b],'confidence':round(float(score),2)})
    return found[:25]


COUNT_SCHEMA={'type':'object','properties':{'items':{'type':'array','items':{'type':'object','properties':{
    'category':{'type':'string'},'kind':{'enum':['person','animal','object']},'count':{'type':'integer'}},'required':['category','kind','count']}}},'required':['items']}
_request_counts={}


def classified_counts(url,text,requested):
    if requested and text in _request_counts:return _request_counts[text]
    role=('Extract only clearly requested numbers of principal entities, including singular articles, '
          'but skip ambiguous additional people or items in a crowd or unrestricted background.'
          if requested else 'Count clearly described distinct subjects depicted directly in the scene, including each drawn character, cloth doll, clay figurine or modeled animal. The visible medium does not make the subject absent. Exclude secondary pictures within the scene, faces printed on clothing, shadows and reflections. Distinguish a depicted child or turtle from a picture of one on the wall.')
    instruction=(role+' This is an independent text extraction. Use short singular English object nouns. '
                 'Aggregate every girl, child, man, and woman into category person. Keep animal species separate. '
                 'Ignore uncounted mass nouns, scenery and light. If a category count is unclear omit it. '
                 'Use 0 only for explicit absence. Do not add entities. Source text: '+text)
    data=json.loads(ask(url,[],instruction,COUNT_SCHEMA))
    counts={}
    species=set()
    for item in data['items']:
        name=str(item['category']).lower().strip()
        name={'human':'person','people':'person','girl':'person','boy':'person','woman':'person','man':'person',
              'kitten':'cat','puppy':'dog','bunny':'rabbit'}.get(name,name)
        n=item['count']
        if name and isinstance(n,int) and 0<=n<=100:
            counts[name]=n
            if item.get('kind')=='animal' and name!='animal':species.add(name)
    counts=reconcile(counts,text,observed=not requested)
    if counts.get('animal',0)>0 and not (species|animals.intersection(counts)):
        provenance='requested in the brief' if requested else 'shown as main figures, in any artistic medium'
        instruction=('Identify the animal species '+provenance+' from this text. Return only explicitly named species with their total counts. '
                     'Use a singular English species name, never the generic category animal. Omit incidental pictures, reflections and uncertain identities. Text: '+text)
        specifics=json.loads(ask(url,[],instruction,COUNT_SCHEMA))['items']
        detail={}
        for item in specifics:
            kind=item['category'].strip().lower()
            if item['kind']=='animal' and kind not in ('animal','animals','unknown') and 0<item['count']<=100:
                detail[kind]=item['count']
        if detail and sum(detail.values())==counts['animal']:
            counts.update(detail);species.update(detail)
    species=(species|animals.intersection(counts)).intersection(counts)
    if species:counts.pop('animal',None)
    result=counts,species
    if requested:_request_counts[text]=result
    return result


def enumerate_counts(url,text,requested):
    return classified_counts(url,text,requested)[0]


def count_conflict(wanted,seen,locations,wanted_species=(),visible_species=()):
    evidence=[]
    for name,number in wanted.items():
        observed=seen.get(name)
        if name=="animal" and observed is None:
            species_counts=[seen[s] for s in (animals|set(visible_species)) if s in seen]
            if species_counts:observed=sum(species_counts)
        detected=sum(item['name']==name for item in locations)
        if observed is not None and observed!=number:
            evidence.append({'object':name,'requested':number,'observed':observed,'detected':detected})
        elif observed is None and detected>number:
            evidence.append({'object':name,'requested':number,'observed':None,'detected':detected})
        elif observed is None and not detected and number>0 and name in wanted_species:
            alternatives={s:seen[s] for s in visible_species if s in seen and s not in wanted_species and seen[s]>0}
            if alternatives:
                evidence.append({'object':name,'requested':number,'observed':None,'detected':0,'other_species':alternatives})
    return evidence


def review(full,sheet,regions,request_text,native=None):
    locations=objects(full)
    with backend() as url:
        inventory=ask(url,[full],INVENTORY)
        wanted,wanted_species=classified_counts(url,request_text,True)
        seen,visible_species=classified_counts(url,inventory,False)
        conflicts=count_conflict(wanted,seen,locations,wanted_species,visible_species)
        if conflicts:
            details='；'.join(f"{c['object']} 要求 {c['requested']}，独立读图得到 {c['observed'] if c['observed'] is not None else '不确定'}，检测框 {c['detected']}" for c in conflicts)
            decisive=any(c['observed'] is not None and c['observed']>c['requested'] and c['detected']>=c['observed'] for c in conflicts)
            return {'status':'global' if decisive else 'uncertain','score':35,'note':'数量需复核。'+details,
                    'issues':[],'regions':regions,'blind_inventory':inventory,'detector':locations,
                    'expected_counts':wanted,'observed_counts':seen,'count_conflicts':conflicts,'reviewer':TAG}
        # Inspect separate region and print samples without asking the model to infer total counts from a collage.
        instructions=(
            "Review the whole illustration against the user's request. The first image is the only complete composition. "
            "The second is a set of labelled details cropped from that same image. R labels describe approximate locations, not guaranteed identities. "
            "If present, image three is six 1:1 print samples of this same illustration. Inspect for seams, repeated textures and halos. "
            "Use this separate prompt-blind observation as evidence, especially for subject counts: "+inventory+
            ". Independent detector boxes: "+json.dumps(locations)+
            ". Detectors may miss stylized or occluded objects. Do not infer absence from a missing box. "
            "Check exact requested counts, identities, pose and proportions, object continuity, occlusions/contact, architecture and perspective. "
            "Use global for an extra or missing principal subject and wrong scene, local for an actually visible, localized defect; "
            "if ambiguous use uncertain. Do not invent a correction for an intentional artistic choice. "
            "If local, identify the concrete visual error in note, describe the desired replacement in English in fix, "
            "name the target, and a tight 0..1000 XYXY box in IMAGE ONE or an R label. "
            "For pass give specific evidence. Write note in Chinese, keep under 160 characters. User request: "+request_text)
        imgs=[full,sheet]+([native] if native is not None else [])
        # Inference backends without multi-image support receive a single labelled contact sheet.
        from PIL import Image,ImageDraw
        widths=[im.width for im in imgs];heights=[im.height for im in imgs]
        montage=Image.new('RGB',(max(widths[0]+max(widths[1:]),900),max(heights[0]+26,sum(h+26 for h in heights[1:]))),'#f1f0e9')
        dr=ImageDraw.Draw(montage);montage.paste(full,(0,26));dr.text((8,5),'A: WHOLE IMAGE / box coordinates refer to this panel',fill='black')
        y=0
        for i,im in enumerate(imgs[1:]):
            dr.text((full.width+8,y+5),('B: same image, region crops','C: same image, actual pixels')[i],fill='black')
            montage.paste(im,(full.width,y+26));y+=im.height+26
        instructions=instructions.replace('first image','left panel A').replace('second is','top right panel B is').replace('image three','bottom right panel C')
        obj=json.loads(ask(url,[montage],instructions,SCHEMA))
    status=obj['status']
    issues=[]
    if status=='local':
        issues=[{'symptom':str(obj['note']),'target':str(obj['target']),'fix':str(obj['fix']),'region':str(obj['region']),'box':obj['box']}]
    return {'status':status,'score':max(0,min(100,int(obj['score']))),'note':str(obj['note']),'issues':issues,
            'regions':regions,'blind_inventory':inventory,'detector':locations,'expected_counts':wanted,'observed_counts':seen,'reviewer':TAG}
