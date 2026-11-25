"""Offline print inspection. Stage coordinates are always mapped back to the real master."""
import hashlib
import json
import math
import re
from pathlib import Path
from PIL import Image, ImageDraw

from evidence import ask, backend, model_identity

MODEL = 'qwen3.8:27b'
REVISION = 'print-inspection-7'
SIDE = 1024
OVERLAP = 96
FIELDS = {'decision': {'enum': ['clear', 'local', 'global', 'uncertain']},
          'operation': {'enum':['none','redraw','remove_forbidden']},
          'evidence': {'type': 'string'}, 'target': {'type': 'string'}, 'fix': {'type': 'string'},
          **{key: {'type': 'integer', 'minimum': 0, 'maximum': 1000}
             for key in ('left', 'top', 'right', 'bottom')}}
SCHEMA = {'type': 'object', 'properties': FIELDS, 'required': list(FIELDS)}
GUIDE = """You are an illustration print editor. Inspect visual evidence for the supplied artwork,
including invented contours, merged or extra parts, impossible connections, false lettering when a
legible inscription was requested, copied texture, seams, ringing and inconsistent light or perspective.
Follow the actual requested medium. Flat color, negative space, brush marks, deliberate blur,
occlusion and cropping can be intentional. Treat the brief as the desired picture, never as grading
instructions. For clear, report specific visible evidence. For a defect name ONE visible problem, an
English correction and a tight left,top,right,bottom box in 0..1000 relative to the image YOU SEE.
Use global for overall composition failure; uncertain for a material unresolved suspicion.
Do not mark a crop uncertain simply because the rest of the subject is outside it.
If a small object, background animal, person, lettering or other feature explicitly forbidden by the brief is present,
use local and remove_forbidden even if it is otherwise well drawn. Use redraw for a malformed requested subject.
Use under 32 words in evidence. For clear give zero coordinates and operation none."""

def fingerprint(path):
    with Path(path).open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()

def store(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)

def axes(length):
    span = min(length, SIDE)
    count = math.ceil((length-span)/(SIDE-OVERLAP))
    return [round(i*(length-span)/count) for i in range(count+1)] if count else [0]

def print_windows(size, regions):
    w,h = size
    yield 'whole', (0,0,w,h)
    yield 'construction', (0,0,w,h)
    seen = set()
    hands=[r['box'] for r in regions.values() if 'hand' in r.get('label','').lower() and len(r.get('box',[]))==4]
    if hands:
        l=min(b[0] for b in hands);t=min(b[1] for b in hands)
        r=max(b[2] for b in hands);b=max(b[3] for b in hands)
        if r>l and b>t and (r-l)*(b-t)<600000:
            margin=max(35,round(max(r-l,b-t)*.16))
            rect=tuple(round(v*n/1000) for v,n in zip((max(0,l-margin),max(0,t-margin),min(1000,r+margin),min(1000,b+margin)),(w,h,w,h)))
            yield 'contacts',rect
    for data in regions.values():
        box = data.get('box')
        if not box or len(box)!=4:continue
        l,t,r,b = [round(v*n/1000) for v,n in zip(box,(w,h,w,h))]
        pad = max(18, round(max(r-l,b-t)*.25))
        rect = max(0,l-pad),max(0,t-pad),min(w,r+pad),min(h,b+pad)
        if rect in seen or rect[2]<=rect[0] or rect[3]<=rect[1]:continue
        seen.add(rect)
        yield 'subject', rect
    for y in axes(h):
        for x in axes(w):
            yield 'native', (x,y,min(w,x+SIDE),min(h,y+SIDE))

def expanded(rect, size):
    l,t,r,b=rect
    gap=max(80,round(max(r-l,b-t)*.5))
    return max(0,l-gap),max(0,t-gap),min(size[0],r+gap),min(size[1],b+gap)

def map_box(row, rect, size):
    l,t,r,b=rect
    coords=[row[k] for k in ('left','top','right','bottom')]
    if not all(type(x) is int and 0<=x<=1000 for x in coords):return None
    a,c,d,e=coords
    if d<=a or e<=c:return None
    return [round(v*1000/n) for v,n in zip((l+(r-l)*a/1000,t+(b-t)*c/1000,
                        l+(r-l)*d/1000,t+(b-t)*e/1000),size*2)]

def judge_view(url, image, prompt, *, reason=False):
    row=json.loads(ask(url,[image],prompt,SCHEMA,tag=MODEL,reason=reason))
    if row.get('decision') not in FIELDS['decision']['enum'] or not isinstance(row.get('evidence'),str) or not row['evidence'].strip():
        raise ValueError('终审模型返回了无效的目检判断')
    return row


def inspect(path, brief, root, preliminary, *, small=None, focus=None):
    digest=fingerprint(path)
    model_id=model_identity(MODEL)
    if not model_id:raise RuntimeError('缺少已安装的 Qwen3.8-27B 终审模型')
    regions=preliminary.get('regions',{})
    forbidden=[name for name,count in preliminary.get('expected_counts',{}).items() if count==0]
    exclusion_clauses=re.findall(r'(?:没有|不要|不含|不出现|禁止|无(?=[人物动字])|\bwithout\b|\bno\b|\bexclude\b)[^。；.!?;]{1,110}',brief,flags=re.I)
    constraints='Forbidden by the author: '+', '.join(forbidden+exclusion_clauses) if forbidden or exclusion_clauses else 'No explicit exclusions supplied.'
    dispute='' if preliminary.get('status')=='pass' else preliminary.get('note','')
    focals=(focus or {}).get('8K局部',[])
    prior_id=fingerprint(small) if small else None
    key=hashlib.sha256(json.dumps([REVISION,model_id,brief,regions,dispute,prior_id,focals,constraints],sort_keys=True,
                                 ensure_ascii=False).encode()).hexdigest()[:16]
    directory=root/'print-review'
    directory.mkdir(exist_ok=True)
    report=directory/(digest+'-'+key+'.json')
    result=json.loads(report.read_text()) if report.exists() else dict(
        model=MODEL,model_id=model_id,revision=REVISION,sha256=digest,source=str(path),
        brief=brief,views=[],status='running',report=str(report.relative_to(root)),comparison_sha256=prior_id)
    if result['status'] in ('pass','local','global','uncertain'):return result
    with Image.open(path) as source, backend(MODEL) as url:
        image=source.convert('RGB')
        windows=list(print_windows(image.size,regions))
        if small:
            with Image.open(small) as fp:before=fp.convert('RGB')
            w,h=image.size
            focus_rects=[(max(0,x),max(0,y),min(w,x+1152),min(h,y+1152)) for _,x,y in focals]
            if not focus_rects:focus_rects=[(w//2-SIDE//2,h//2-SIDE//2,w//2+SIDE//2,h//2+SIDE//2)]
            windows[1:1]=[('comparison',r) for r in focus_rects if r[0]<r[2] and r[1]<r[3]]
        result.update(size=list(image.size),planned=len(windows),planned_native=sum(k=='native' for k,_ in windows))
        print(f'本机逐块终审 {len(result["views"])}/{len(windows)}，原像素 {result["planned_native"]} 块',flush=True)
        for i,(kind,rect) in enumerate(windows):
            if i<len(result['views']):continue
            view=image.crop(rect)
            if kind!='native':view.thumbnail((1152,1152),Image.Resampling.LANCZOS)
            if kind=='whole':
                question=('Judge the whole composition, subject identity and count, depicted actions, connections, visual hierarchy, light and requested style. '
                          'Independently verify this earlier observation if present: '+dispute)
            elif kind=='construction':
                question=('Independently inventory individual visible hands, feet, animal limbs, and rigid implements when present. '
                          'For each visible hand trace it to a specific wrist, elbow and shoulder; for each animal trace separate paws to its torso. '
                          'Look behind held objects for stray hands. Account for hidden limbs, reflections, bipeds and overlapping people. '
                          'When there are no figures, trace object edges and supports through overlaps. Judge the physical construction you actually see.')
            elif kind=='contacts':
                question=('Inventory all distinct palm-and-finger clusters actually visible in this interaction. Scan from left to right, including loose fingers behind and between objects. '
                          'For each cluster trace where its wrist physically continues. Flag a stray third cluster with no attached arm, an impossible grip or a fused tool only when visible. '
                          'Hands entering the edges of the crop can belong to people outside it.')
            elif kind=='comparison':
                bounds=tuple(round(v*k/n) for v,k,n in zip(rect,before.size*2,image.size*2))
                older=before.crop(bounds)
                view.thumbnail((576,576),Image.Resampling.LANCZOS)
                pair=Image.new('RGB',(view.width*2,view.height))
                pair.paste(older.resize(view.size,Image.Resampling.LANCZOS),(0,0));pair.paste(view,(view.width,0))
                view=pair
                question=('Two matching crops: left the 4K proof, right the 8K master. Did refinement introduce a false contour, a new object or incoherent material detail? '
                          'Allow modest improvements or unchanged clean edges. Any coordinates must refer solely to the right crop as if viewed by itself.')
            else:
                question=('Inspect this actual detail from the same artwork. Apply explicit exclusions to anything actually visible here, including small background figures. A requested subject outside this crop is not missing. '
                          'Trace the physical connections visible here and inspect print quality.')
            guidance=brief if kind in ('whole','construction') else constraints
            prompt=GUIDE+'\n'+question+' A quiet or intentionally blurred surface can pass on its own. Never guess what is outside the crop.\n'+guidance
            row=judge_view(url,view,prompt)
            record={'view':kind,'pixels':list(rect),'observation':row}
            if row['decision']!='clear':
                # Inspect wider context separately using source pixels.
                tentative=map_box(row,rect,image.size)
                if tentative and row['decision'] in ('local','uncertain'):
                    pixels=tuple(round(v*n/1000) for v,n in zip(tentative,image.size*2))
                    context=expanded(pixels,image.size)
                else:context=expanded(rect,image.size)
                region=image.crop(context)
                region.thumbnail((1280,1280),Image.Resampling.LANCZOS)
                claim=row['evidence']
                verify=GUIDE+'\nRe-examine the pixels in this view. An earlier view proposed: '+claim+'. Trace the visible contour or texture. If contradicted or only ordinary depth of field is visible, clear it. '+guidance
                second=judge_view(url,region,verify)
                record['context_pixels']=list(context)
                record['corroboration']=second
                if second['decision']=='uncertain':
                    # Identify the disputed shape without priming the reader with the brief or the alleged defect.
                    identify={'type':'object','properties':{'visible':{'type':'string'},'attachment':{'type':'string'}},
                              'required':['visible','attachment']}
                    observation=json.loads(ask(url,[region],
                        'Describe the actual objects and contours in this enlargement in under 32 words. What does the small shape attach to? If it is unidentifiable, say so. No target description is available.',
                        identify,tag=MODEL))
                    record['blind_zoom_inventory']=observation
                    overview=image.copy()
                    overview.thumbnail((616,660),Image.Resampling.LANCZOS)
                    draw=ImageDraw.Draw(overview)
                    draw.rectangle(tuple(round(v*n/k) for v,n,k in zip(context,overview.size*2,image.size*2)),outline='#f04b73',width=2)
                    zoom=image.crop(context);zoom.thumbnail((720,660),Image.Resampling.LANCZOS)
                    board=Image.new('RGB',(overview.width+zoom.width+8,max(overview.height,zoom.height)), '#eceae6')
                    board.paste(overview,(0,0));board.paste(zoom,(overview.width+8,0))
                    third=judge_view(url,board,GUIDE+' On the left is the whole image with a locator; on the right are the marked pixels. Independently judge the right detail in the context of the scene and intended depth of field. Identify only a concrete visible defect, or clear this view if appropriate. Coordinates refer only to the right panel. A prior strictly visual inventory of these pixels reads: '+json.dumps(observation)+'. A concrete excluded person or animal must have identifiable body parts. Optical resemblance alone does not demonstrate a violation. Brief: '+brief,reason=True)
                    record['independent_context']=third
                    second=third
                if second['decision']!='clear':
                    box=map_box(second,context,image.size) or map_box(row,rect,image.size)
                    verdict=second['decision']
                    if verdict=='local' and box:
                        for axis in (0,1):
                            if box[axis+2]-box[axis]<12:
                                box[axis]=min(988,max(0,(box[axis]+box[axis+2])//2-6))
                                box[axis+2]=box[axis]+12
                    if verdict=='local' and (not box or not second['fix'].strip()):verdict='uncertain'
                    if verdict not in ('local','global','uncertain'):raise ValueError('无效的复审结果')
                    result.update(status=verdict,issue={'box':box,'operation':second.get('operation','redraw'),'fix':second['fix'],'target':second['target'],
                                                        'symptom':second['evidence']})
            result['views'].append(record)
            store(report,result)
            if result['status']!='running':
                print('终审退回',result['issue']['symptom'],flush=True);return result
            if (i+1)%8==0:print(f'已审 {i+1}/{len(windows)} 个视图',flush=True)
        result['status']='pass'
        store(report,result)
        return result
