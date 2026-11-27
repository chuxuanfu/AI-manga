"""Independent, prompt-blind image inventory and local print review."""
import hashlib
import base64
import io
import json
import marshal
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import lru_cache
from urllib import request
import torch
from model_slot import slot, auxiliary, before_auxiliary
from counts import reconcile, animals, inventory_counts

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


_active_reader=ContextVar('active_reader',default=None)
_inspection_session=ContextVar('inspection_session',default=None)


@contextmanager
def review_session():
    """Consecutive inspections of one unchanged image share a single offline service and device lease."""
    existing=_inspection_session.get()
    if existing is not None and existing['pid']==os.getpid():
        yield existing['timings']
        return
    timings={'queue_seconds':0.,'device_lease_seconds':0.,'service_starts':0,'models':[]}
    with ExitStack() as stack:
        state={'pid':os.getpid(),'stack':stack,'url':None,'timings':timings}
        token=_inspection_session.set(state)
        try:
            yield timings
        finally:
            _inspection_session.reset(token)


@contextmanager
def backend(tag=TAG):
    reader=_active_reader.get()
    if reader and reader['pid']==os.getpid():
        if not model_identity(tag):raise RuntimeError('缺少本地视觉模型 '+tag)
        yield reader['url']
        return
    state=_inspection_session.get()
    if state is None or state['pid']!=os.getpid():
        with _local_service(tag) as address:
            yield address
        return
    if state['url'] is None:
        state['url']=state['stack'].enter_context(_local_service(tag,state['timings']))
    elif not model_identity(tag):
        raise RuntimeError('缺少本地视觉模型 '+tag)
    yield state['url']


@contextmanager
def _local_service(tag,timings=None):
    """The inference service inherits the launcher's outbound-network denial."""
    binary=shutil.which('ollama') or '/Applications/Ollama.app/Contents/Resources/ollama'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    address=f'127.0.0.1:{port}'
    env={**os.environ,'OLLAMA_HOST':address,'OLLAMA_MODELS':str(WEIGHTS),'OLLAMA_NO_CLOUD':'1','OLLAMA_NUM_PARALLEL':'1','OLLAMA_MAX_LOADED_MODELS':'1'}
    with tempfile.TemporaryFile() as log, slot() as queued:
        acquired=time.monotonic()
        if timings is not None:
            timings.update(queue_seconds=round(queued,2),service_starts=1)
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
            url='http://'+address+'/api/chat'
            active={'tag':None,'pid':os.getpid(),'url':url}
            def unload():
                if active['tag']:
                    command=request.Request('http://'+address+'/api/generate',data=json.dumps(
                        {'model':active['tag'],'keep_alive':0,'stream':False}).encode(),headers={'Content-Type':'application/json'})
                    with request.urlopen(command,timeout=30) as reply:reply.read()
                    active['tag']=None
            token=_active_reader.set(active)
            callback=before_auxiliary.set(unload)
            try:yield url
            finally:
                before_auxiliary.reset(callback)
                _active_reader.reset(token)
        finally:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            if timings is not None:timings['device_lease_seconds']=round(time.monotonic()-acquired,2)


@lru_cache(maxsize=1)
def _inference_revision():
    return hashlib.sha256(marshal.dumps(_model_answer.__code__)).hexdigest()


def ask(url,images,prompt,schema=None,tag=TAG,*,reason=False):
    state=_inspection_session.get()
    if state is not None and state['pid']==os.getpid() and tag not in state['timings']['models']:
        state['timings']['models'].append(tag)
    # Pictures always go to the visual model. A retry with the identical text
    # and schema can reuse the prior extraction without evicting a loaded VLM.
    if images or schema is None:
        return _model_answer(url,images,prompt,schema,tag,reason)
    identity=model_identity(tag)
    executable=shutil.which('ollama') or '/Applications/Ollama.app/Contents/Resources/ollama'
    if not identity:
        return _model_answer(url,images,prompt,schema,tag,reason)
    try:
        stamp=Path(executable).stat()
    except OSError:
        return _model_answer(url,images,prompt,schema,tag,reason)
    key=json.dumps([_inference_revision(),stamp.st_size,stamp.st_mtime_ns,tag,identity,reason,prompt,schema],
                   ensure_ascii=False,separators=(',',':'))
    digest=hashlib.sha256(key.encode()).hexdigest()
    saved=RUNTIME/'cache'/'text-readings'/digest[:2]/(digest+'.json')
    try:
        row=json.loads(saved.read_text())
        value=json.loads(row['answer'])
        if row['request']==digest and isinstance(value,dict) and all(k in value for k in schema.get('required',[])):
            if state is not None and state['pid']==os.getpid():
                times=state['timings']
                times['reused_text_readings']=times.get('reused_text_readings',0)+1
            return row['answer']
    except (OSError,ValueError,KeyError,TypeError):
        pass
    answer=_model_answer(url,images,prompt,schema,tag,reason)
    try:
        from print_judge import store
        saved.parent.mkdir(parents=True,exist_ok=True)
        store(saved,{'request':digest,'answer':answer})
    except OSError:
        pass
    return answer


def _model_answer(url,images,prompt,schema,tag,reason):
    active=_active_reader.get()
    if active is not None and active['tag']!=tag:
        before_auxiliary.get()()
        active['tag']=tag
    encoded=[]
    for image in images:
        buf=io.BytesIO()
        # A second reader must see the same pigment boundaries, including fine colored ink.
        image.save(buf,format='PNG',compress_level=1)
        encoded.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    data={'model':tag,'stream':False,'options':{'temperature':0,'num_ctx':8192,'num_predict':360},
          'messages':[{'role':'user','content':prompt,'images':encoded}]}
    if tag=='qwen3.8:27b':data['think']=reason
    if schema:data['format']=schema
    # A two-sided four-axis assessment contains at least sixteen required fields.
    # Give wide structured answers room on their first decode, avoiding a full second vision pass.
    properties=(schema or {}).get('properties',{})
    required_fields=sum(max(1,len(field.get('required',[]))) for field in properties.values())
    initial_limit=900 if required_fields>=12 else 360
    limits=(1800,3200) if reason else ((900,1400) if initial_limit==900 else (360,900,1400))
    for attempt,limit in enumerate(limits):
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


def detector():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    net=fasterrcnn_mobilenet_v3_large_fpn(weights=None,weights_backbone=None).eval()
    path=RUNTIME/'models/object_detector/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    return net,FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.meta['categories']


@auxiliary
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
    'category':{'type':'string'},'counting_species':{'type':'string'},'kind':{'enum':['person','animal','object']},'count':{'type':'integer'}},'required':['category','kind','count','counting_species']}}},'required':['items']}
_request_counts={}


def classified_counts(url,text,requested):
    if requested and text in _request_counts:return _request_counts[text]
    role=('Extract only clearly requested numbers of principal entities, including singular articles, '
          'but skip ambiguous additional people or items in a crowd or unrestricted background.'
          if requested else 'Count clearly described distinct subjects depicted directly in the scene, including each drawn character, cloth doll, clay figurine or modeled animal. The visible medium does not make the subject absent. Exclude secondary pictures within the scene, faces printed on clothing, shadows and reflections. Distinguish a depicted child or turtle from a picture of one on the wall.')
    instruction=(role+' This is an independent text extraction. Use short singular English object nouns. '
                 'Aggregate every girl, child, man, and woman into category person. For each animal supply its common species noun in counting_species (dog for Shiba Inu, cat for calico); put the named breed or type in category. Give one row per distinct type, exclude duplicate mentions. For other kinds use an empty counting_species. '
                 'Ignore uncounted mass nouns, scenery and light. If a category count is unclear omit it. '
                 'Use 0 only for explicit absence. Do not add entities. Source text: '+text)
    data=json.loads(ask(url,[],instruction,COUNT_SCHEMA))
    counts,species=inventory_counts(data['items'])
    counts=reconcile(counts,text,observed=not requested)
    if counts.get('animal',0)>0 and not (species|animals.intersection(counts)):
        provenance='requested in the brief' if requested else 'shown as main figures, in any artistic medium'
        instruction=('Identify the animal species '+provenance+' from this text. Return only explicitly named species with their total counts. '
                     'Use a singular English species name, never the generic category animal. Omit incidental pictures, reflections and uncertain identities. Text: '+text)
        specifics=json.loads(ask(url,[],instruction,COUNT_SCHEMA))['items']
        detail,_=inventory_counts([item for item in specifics if item['kind']=='animal' and item['count']>0])
        detail={name:n for name,n in detail.items() if name not in ('animal','animals','unknown')}
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
