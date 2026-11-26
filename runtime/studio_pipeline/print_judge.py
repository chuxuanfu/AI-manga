"""Offline print inspection. Stage coordinates are always mapped back to the real master."""
import hashlib
import json
import math
import re
import time
from pathlib import Path
from PIL import Image, ImageDraw

from evidence import ask, backend, model_identity

MODEL = 'qwen3.8:27b'
REVISION = 'print-inspection-27'
SIDE = 1024
OVERLAP = 96
FIELDS = {'decision': {'enum': ['clear', 'local', 'global', 'uncertain']},
          'operation': {'enum':['none','redraw','remove_forbidden']},
          'evidence': {'type': 'string'}, 'target': {'type': 'string'}, 'fix': {'type': 'string'},
          **{key: {'type': 'integer', 'minimum': 0, 'maximum': 1000}
             for key in ('left', 'top', 'right', 'bottom')}}
SCHEMA = {'type': 'object', 'properties': FIELDS, 'required': list(FIELDS)}
NATIVE_SCREEN = {'type': 'object', 'properties': {key: FIELDS[key] for key in ('decision', 'evidence')},
                 'required': ['decision', 'evidence']}
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
Use under 32 words in evidence. When coordinates are requested, a clear view has zero coordinates and operation none."""

SOFT_HANDLING=re.compile(r'blur(?:red|ry)?|out[- ]of[- ]focus|defocus|bokeh|shallow depth|soft(?:ened|ness|ly|[- ](?:focus|edges|background|blended))|flat[- ](?:colou?r|fills?|shading)|negative space|虚化|虚焦|失焦|浅景深|朦胧|柔焦|模糊|平涂|留白|ぼけ|ボケ',re.I)

def requested_handling(phrase,brief):
    return bool(isinstance(phrase,str) and phrase.strip() and phrase.strip().casefold() in brief.casefold()
                and SOFT_HANDLING.search(phrase))

def crop_brief(brief):
    """A pixel sample can show one object even when the requested scene contains many."""
    scene, sep, style = brief.partition(' · ')
    handling = [s.strip() for s in re.split(r'[.;。；!?，,]',scene) if SOFT_HANDLING.search(s)]
    return ('Evaluate the material and contours actually visible inside this regional crop. '
            'The composition is inspected separately in full. '
            + ('Medium and finish: '+style+'. ' if sep else 'Respect the visible artwork medium. ')
            + ('Local focus instructions: '+'. '.join(handling) if handling else ''))


def excluded_by_brief(brief):
    return re.findall(r'(?:没有|不要|不含|不出现|禁止|无(?=[人物动字])|\bwithout\b|\bno\b|\bexclude\b)[^。；.!?;]{1,110}',brief,flags=re.I)


def exclusion_repair(url, image, box, brief, clauses, claim):
    """Propose a reversible local repair when an explicit exclusion is unresolved."""
    if not clauses or not box or not 0<(box[2]-box[0])*(box[3]-box[1])<60000:return None
    location=tuple(round(v*n/1000) for v,n in zip(box,image.size*2))
    region=image.crop(expanded(location,image.size))
    region.thumbnail((850,850),Image.Resampling.LANCZOS)
    fields={'phrase':{'type':'string'},'action':{'enum':['clear_surface','remove_accessory','hold']},
            'part':{'type':'string'},'edit':{'type':'string'}}
    question=('The image contains a small disputed feature. An editor suspects it may violate an '
              'explicit exclusion. Select a local edit that unambiguously satisfies the brief while '
              'preserving every requested subject, action, pose and color. If the marks are on a '
              'required object, clear_surface means repaint only its surface while preserving the '
              'object. remove_accessory is allowed only for a separate, optional decoration whose '
              'complete removal preserves the scene. Use hold for uncertain location, unrelated '
              'defects or no applicable explicit exclusion. In phrase quote the exact applicable '
              'exclusion; describe a concrete visual edit in English. Concern: '+claim+' Brief: '+brief)
    result=json.loads(ask(url,[region],question,{'type':'object','properties':fields,'required':list(fields)},tag=MODEL))
    phrase=result['phrase'].strip().rstrip('。.!！').casefold()
    if not phrase or not any(phrase==c.strip().casefold() for c in clauses):return None
    if not result['part'].strip() or not result['edit'].strip():return None
    if result['action'] not in ('clear_surface','remove_accessory'):return None
    return result


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

def repair_extent(final_box, earlier_boxes):
    """Keep adjoining evidence for a single part when a closer view picks just one tip."""
    if not final_box:
        return None
    if (final_box[2]-final_box[0])*(final_box[3]-final_box[1]) > 120000:
        return final_box
    left,top,right,bottom = final_box
    for candidate in earlier_boxes:
        if not candidate:
            continue
        l,t,r,b = candidate
        w,h = right-left,bottom-top
        if r-l > max(40,3*w) or b-t > max(40,3*h):
            continue
        if max(l-right,left-r) > max(4,w*.12) or max(t-bottom,top-b) > max(4,h*.12):
            continue
        merged = min(l,left),min(t,top),max(r,right),max(b,bottom)
        if (merged[2]-merged[0])*(merged[3]-merged[1]) <= 120000:
            left,top,right,bottom = merged
    return [left,top,right,bottom]


def judge_view(url, image, prompt, *, reason=False, overview=None):
    images=list(image) if isinstance(image,(list,tuple)) else [image]
    if overview is not None:images.append(overview)
    row=json.loads(ask(url,images,prompt,SCHEMA,tag=MODEL,reason=reason))
    if row.get('decision') not in FIELDS['decision']['enum'] or not isinstance(row.get('evidence'),str) or not row['evidence'].strip():
        raise ValueError('终审模型返回了无效的目检判断')
    return row


def inspect_native(url,image,prompt):
    """Inspect every pixel window. Generate localization coordinates only for suspected defects."""
    row=json.loads(ask(url,[image],prompt+' Report the decision and a brief concrete observation.',
                       NATIVE_SCREEN,tag=MODEL))
    if row.get('decision') not in FIELDS['decision']['enum'] or not str(row.get('evidence') or '').strip():
        raise ValueError('原像素初检缺少有效目检记录')
    if row['decision']!='clear':
        localized=judge_view(url,image,prompt+' Locate this possible issue if visible: '+row['evidence'])
        if localized['decision']!='clear':return localized
        # Keep the original suspicion until the existing wider-context check resolves it.
    return dict(row,operation='none',target='',fix='',left=0,top=0,right=0,bottom=0)

def detail_sites(image):
    """Locate separated, structured areas without relying on figure-specific landmarks."""
    import cv2
    import numpy as np
    thumb=image.convert('L')
    thumb.thumbnail((512,512),Image.Resampling.LANCZOS)
    y=np.asarray(thumb,dtype=np.float32)
    dx=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)
    scores=cv2.boxFilter(np.minimum(cv2.magnitude(dx,dy),120),-1,(25,25))
    sites=[]
    for _ in range(3):
        _,peak,_,(x,y)=cv2.minMaxLoc(scores)
        if peak<5:break
        sites.append(((x+.5)/thumb.width,(y+.5)/thumb.height))
        cv2.circle(scores,(x,y),max(20,min(thumb.size)//5),0,-1)
    return sites

def centered_crop(size,point,span):
    w,h=size
    cw,ch=min(w,span),min(h,span)
    l=max(0,min(w-cw,round(point[0]*w-cw/2)))
    t=max(0,min(h-ch,round(point[1]*h-ch/2)))
    return l,t,l+cw,t+ch

def compare_pixels(low,high,rect,master_size):
    """Display the new crop at 1:1; enlarge the matching old field to the same size."""
    def mapped(size):
        return tuple(v*n/k for v,n,k in zip(rect,size*2,master_size*2))
    bounds=tuple(round(v) for v in mapped(high.size))
    right=high.crop(bounds)
    left=low.resize(right.size,Image.Resampling.LANCZOS,box=mapped(low.size))
    return left,right


def arbitrate_finding(url, detail, locator, original, corroboration, inventory):
    """Adjudicate one claim without substituting a different nearby feature or its coordinates."""
    props={'evidence':{'type':'string'}, 'outcome':{'enum':['confirmed','disproved','insufficient']},
           'part':{'type':'string'}}
    schema={'type':'object','properties':props,'required':list(props)}
    question=('The FIRST image is a crop containing a disputed feature. The SECOND only locates the '
              'crop in the original composition. Resolve exactly the proposed finding below. Trace the '
              'named part in the FIRST image and give visible evidence for or against that specific '
              'claim. A different feature cannot confirm or disprove this finding. Use insufficient '
              'when the named part or connection is unresolved. For part, name only the physical '
              'part that must be reconstructed together with its attachment (for instance a hand for fused digits or a wheel for misaligned spokes). Independent description of the pixels: '+json.dumps(inventory)+
              '. Proposed finding: '+original['evidence']+' Second reading: '+corroboration['evidence'])
    read=json.loads(ask(url,[detail,locator],question,schema,tag=MODEL,reason=True))
    if read['outcome']=='insufficient':
        # Resolve a contradictory decision only when another reading can account for
        # the entire disputed contour on the actual crop, independently of its locator.
        props={'target_visible':{'type':'boolean'},'entire_disputed_contour_resolved':{'type':'boolean'},
               'defect':{'enum':['present','absent','ambiguous']},'visible_trace':{'type':'string'}}
        repeat=json.loads(ask(url,[detail],
            'Inspect only the disputed physical feature in this crop. Proposed defect: '
            +original['evidence']+' A previous reader traced these pixels: '+read['evidence']
            +'. Independently trace the contour, both boundaries and attachments. Only report absent '
             'if visible evidence disproves the entire proposed defect. Use ambiguous for partial or '
             'obscured evidence or for disputed texture resolution you cannot compare at this scale.',
            {'type':'object','properties':props,'required':list(props)},tag=MODEL,reason=True))
        if (repeat['defect']=='absent' and repeat['target_visible'] is True
                and repeat['entire_disputed_contour_resolved'] is True and repeat['visible_trace'].strip()):
            read=dict(read,outcome='disproved',evidence=repeat['visible_trace'])
    if read['outcome']=='disproved':
        return dict(corroboration,decision='clear',operation='none',target='',fix='',evidence=read['evidence'])
    outcome='uncertain'
    if read['outcome']=='confirmed':
        outcome='global' if original['decision']=='global' else 'local'
    return dict(corroboration,decision=outcome,target=read['part'],evidence=read['evidence'])


FRAME_CLAIM=re.compile(r'frame|framing|cropp?ed|cut off|out of (?:the )?(?:view|picture)|extend the canvas|画框|出框|裁切|画幅|截断',re.I)

def verify_frame_exit(url,image,overview,box,claim,brief):
    """Check an alleged truncation against the physical edge and the requested framing."""
    if not box or not FRAME_CLAIM.search(claim):return None
    edges=[name for name,touches in zip(('left','top','right','bottom'),
           (box[0]<=12,box[1]<=12,box[2]>=988,box[3]>=988)) if touches]
    if not edges:return None
    pixels=tuple(round(v*n/1000) for v,n in zip(box,image.size*2))
    crop=image.crop(expanded(pixels,image.size));crop.thumbnail((960,960),Image.Resampling.LANCZOS)
    locator=overview.copy()
    ImageDraw.Draw(locator).rectangle(tuple(round(v*n/1000) for v,n in zip(box,locator.size*2)),outline='#ec407a',width=2)
    fields={'trace':{'type':'string'},'edge':{'enum':['left','top','right','bottom','none']},
            'outcome':{'enum':['continuous_at_border','broken_inside','unresolved','other_defect']},
            'brief_requires_missing_part':{'type':'boolean'}}
    result=json.loads(ask(url,[locator,crop],
        'Image 1 is the COMPLETE composition; image 2 enlarges the marked location near its actual '
        +', '.join(edges)+' edge(s). Determine whether the sole alleged defect is ordinary framing. '
        'Trace the contour to the PHYSICAL outer border of image 1. A subject may extend out of frame. '
        'Check the brief for an expressly required visible full figure or missing part. A limb or object '
        'ending or melting inside the picture counts as broken_inside. Other visible problems with this '
        'connection are other_defect. If the exact crossing cannot be traced, choose unresolved. '
        'Claim: '+claim+' Brief: '+brief,{'type':'object','properties':fields,'required':list(fields)},tag=MODEL,reason=True))
    result['explained']=(result['outcome']=='continuous_at_border' and result['edge'] in edges
                         and result['brief_requires_missing_part'] is False and bool(result['trace'].strip()))
    if result['outcome']=='unresolved' and not result['brief_requires_missing_part'] and result['edge'] in edges:
        # Ask about literal geometry on the original border rather than asking the
        # same reviewer to interpret its own uncertain verdict.
        fields={'part_reaches_border':{'type':'boolean'},'break_before_border':{'type':'boolean'},
                'another_visible_defect':{'type':'boolean'},'edge':{'enum':['left','top','right','bottom','none']},
                'required_visible_feature_absent':{'type':'boolean'},'path':{'type':'string'}}
        observation=json.loads(ask(url,[crop,locator],
           'Inspect the actual detail (first) and the entire frame with the location outlined (second). '
           'The feature under examination is described here: '+claim+
           '. Which physical image edge, if any, does this exact visible part reach? '
           'Trace the visible path to that edge. Does it actually end or change into an unrelated form '
           'BEFORE the border? Is some other defect in this same part visible on the canvas? '
           'Use only the following scene brief to decide whether an absent part was explicitly required '
           'to be entirely visible: '+brief,
           {'type':'object','properties':fields,'required':list(fields)},tag=MODEL,reason=True))
        result['edge_trace']=observation
        result['explained']=(observation['edge']==result['edge'] and observation['part_reaches_border'] is True
              and observation['break_before_border'] is False and observation['another_visible_defect'] is False
              and observation['required_visible_feature_absent'] is False and bool(observation['path'].strip()))
        if result['explained']:result['trace']=observation['path']
    return result


OPTICAL_SYMPTOM=re.compile(
    r'blur|smear|mush|out[- ]of[- ]focus|low[- ](?:res|definition)|pixelat|interpolat|upscal|'
    r'over[- ]?enlarg|soft(?:ness|ened|[- ](?:edge|boundar|transition|silhouette|focus))|'
    r'indistinct (?:edges?|strokes?)|loss of (?:fine |micro)?detail|模糊|糊成|失焦|低清|插值|ぼやけ|ピンぼけ',re.I)


def issue_category(url, region, evidence, brief):
    # A confusing relation is often reported as "unclear" even when its edges are crisp.
    # Only an observation of an optical symptom can trigger a costly focus rebuild.
    # Other findings still require structural repair and the full release checks.
    if not OPTICAL_SYMPTOM.search(evidence):return 'other'
    fields={'issue':{'enum':['unresolved_focus','other']},
            'geometric_fault':{'type':'boolean'}, 'optical_loss_visible':{'type':'boolean'},
            'pixel_trace':{'type':'string'}}
    task=('Diagnose the cause of this specific finding before selecting a repair. Trace an edge in the '
          'image and state what is actually wrong. A crisp drawing of a wrong or ambiguous shape, '
          'connection, overlap, proportion, attachment or count has a geometry problem. A missing '
          'physical connection remains a geometry problem even when its written description uses '
          'words like unresolved or unclear. Optical loss is a visibly smeared edge or disappearing '
          'fine structure on a correctly constructed intended focal subject. Ignore deliberate '
          'softness and empty fills. Choose unresolved_focus only for optical loss without a '
          'geometric fault. Evidence: '+evidence+' Brief: '+brief)
    detail=list(region) if isinstance(region,(tuple,list)) else [region]
    data=json.loads(ask(url,detail,task,{'type':'object','properties':fields,'required':list(fields)},tag=MODEL))
    if (data['issue']=='unresolved_focus' and data['optical_loss_visible'] is True
            and data['geometric_fault'] is False and data['pixel_trace'].strip()):
        return 'unresolved_focus'
    return 'other'


def independent_detail_check(url, images, overview, brief):
    """Resolve a disputed comparison only when measured detail and two blind readings agree."""
    import cv2
    import numpy as np
    low, high = images
    def bands(image):
        gray = np.asarray(image.convert('L'), dtype=np.float32)
        edge = cv2.magnitude(cv2.Sobel(gray,-1,1,0), cv2.Sobel(gray,-1,0,1)) / 8
        return float(np.percentile(edge,97)), float(np.percentile(abs(cv2.Laplacian(gray,-1)),97))
    old_edge,old_fine = bands(low)
    new_edge,new_fine = bands(high)
    report = {'edge_97':[old_edge,new_edge], 'fine_97':[old_fine,new_fine], 'confirmed':False}
    if new_edge < max(3,old_edge*.88) or new_fine < max(3,old_fine*1.08):
        return report
    fields = {'trace':{'type':'string'}, 'improvement':{'type':'string'},
              'resolvability':{'enum':['resolved','unresolved','uncertain']},
              'defined_finer_forms':{'type':'boolean'}, 'new_artifact':{'type':'boolean'}}
    question = ('First inspect separately supplied images 1 and 2 at equal display scale; image 3 locates the detail. '
                'As a print editor, follow one actual contour through a fork, turn, contact or material transition. '
                'Describe what it belongs to. Report exactly what you can resolve in image 2 that is less clear in image 1. '
                'Judge whether the focal marks in image 2 are materially resolved for this medium. True fine structure '
                'must follow a recognizable form; noise, halos and interpolation count against it. If both are soft, '
                'unchanged or implausible, do not claim finer forms. Separate optical focus from deliberate ink bleed '
                'and paper texture. Cite a position in the crop and concrete evidence. Brief: '+brief)
    def read(views,prompt,properties,reason=False):
        schema={'type':'object','properties':properties,'required':list(properties)}
        return json.loads(ask(url, views, prompt, schema, tag=MODEL, reason=reason))
    finding = read([low,high,overview],question,fields,True)
    report['trace'] = finding
    if (finding['resolvability']!='resolved' or finding['defined_finer_forms'] is not True or
            finding['new_artifact'] is not False or not finding['trace'].strip() or not finding['improvement'].strip()):
        return report
    fields = {'finer_version':{'enum':['first','second','equivalent','ambiguous']},
              'location':{'type':'string'},'evidence':{'type':'string'}}
    question = ('Treat these as two anonymous print proofs of the same region. The last image shows the scene. '
                'Which of the first two contains a more clearly resolved plausible material feature or contour? '
                'A sharper halo, noise, added random strokes or a contrast change is insufficient. Choose equivalent '
                'when no concrete additional legible structure can be located. Name the feature and its exact '
                'location in one short sentence. Brief: '+brief)
    reverse = read([high,low,overview],question,fields)
    report['reverse'] = reverse
    report['confirmed'] = reverse['finer_version']=='first' and bool(reverse['location'].strip() and reverse['evidence'].strip())
    return report


def inspect(path, brief, root, preliminary, *, small=None, focus=None, draft=None, source_comparison=False):
    digest=fingerprint(path)
    model_id=model_identity(MODEL)
    if not model_id:raise RuntimeError('缺少已安装的 Qwen3.8-27B 终审模型')
    regions=preliminary.get('regions',{})
    forbidden=[name for name,count in preliminary.get('expected_counts',{}).items() if count==0]
    exclusion_clauses=excluded_by_brief(brief)
    constraints='Forbidden by the author: '+', '.join(forbidden+exclusion_clauses) if forbidden or exclusion_clauses else 'No explicit exclusions supplied.'
    dispute='' if preliminary.get('status')=='pass' else preliminary.get('note','')
    focals=(focus or {}).get('8K局部',[])
    prior_id=fingerprint(small) if small else None
    draft_id=fingerprint(draft) if draft and small else None
    repair=(focus or {}).get('repair_check')
    key=hashlib.sha256(json.dumps([REVISION,model_id,brief,regions,dispute,prior_id,draft_id,focals,constraints,repair],sort_keys=True,
                                 ensure_ascii=False).encode()).hexdigest()[:16]
    if source_comparison:key+='-source'
    directory=root/'print-review'
    directory.mkdir(exist_ok=True)
    report=directory/(digest+'-'+key+'.json')
    result=json.loads(report.read_text()) if report.exists() else dict(
        model=MODEL,model_id=model_id,revision=REVISION,sha256=digest,source=str(path),
        brief=brief,views=[],status='running',report=str(report.relative_to(root)),comparison_sha256=prior_id,draft_sha256=draft_id)
    if result['status'] in ('pass','local','global','uncertain'):return result
    with Image.open(path) as source, backend(MODEL) as url:
        image=source.convert('RGB')
        overview_image=image.copy()
        overview_image.thumbnail((1152,1152),Image.Resampling.LANCZOS)
        windows=list(print_windows(image.size,regions))
        prior_rects={}
        for saved in sorted(directory.glob(digest+'-*.json')):
            if saved == report:continue
            old=json.loads(saved.read_text())
            if old.get('status')!='local' or old.get('brief')!=brief or old.get('model_id')!=model_id:continue
            found=old.get('issue',{}).get('box')
            if not found or not old.get('views'):continue
            last=old['views'][-1]
            original_box=map_box(last['observation'],last['pixels'],image.size)
            found=repair_extent(found,[original_box])
            candidate=tuple(round(v*k/1000) for v,k in zip(found,image.size*2))
            prior_rects[expanded(candidate,image.size)]={'symptom':old['issue'].get('symptom',''),'box':found}
        if repair:
            candidate=tuple(round(v*k/1000) for v,k in zip(repair['box'],image.size*2))
            prior_rects[expanded(candidate,image.size)]=repair
        if 'previous_regions' not in result:
            result['previous_regions']=[[list(area),prior_rects[area]] for area in sorted(prior_rects)]
        known={tuple(area):finding for area,finding in result['previous_regions']}
        for area in reversed(known):
            windows.insert(1,('previous_local',area))
        if small:
            with Image.open(small) as fp:before=fp.convert('RGB')
            w,h=image.size
            sites=detail_sites(before)
            # Also cover semantic subjects whose tones are quieter than the background.
            span=(focus or {}).get('8K图块',1152)
            for _,left,top in focals:
                center=((left+span/2)/w,(top+span/2)/h)
                if all(abs(x-center[0])*w>448 or abs(y-center[1])*h>448 for x,y in sites):
                    sites.append(center)
            if not sites:sites=[(.5,.5)]
            comparisons=[('comparison',centered_crop(image.size,site,448)) for site in sites]
            if source_comparison:
                if max(before.size)>=max(image.size):comparisons=[]
                if digest!=prior_id:comparisons.append(('palette',(0,0,w,h)))
            if draft_id:
                with Image.open(draft) as fp:draft_image=fp.convert('RGB')
                if max(draft_image.size)<max(before.size):
                    comparisons+=[('draft-comparison',centered_crop(image.size,site,896)) for site in sites]
                comparisons.append(('palette',(0,0,w,h)))
            position=1+len(result['previous_regions'])
            windows[position:position]=list(dict.fromkeys(comparisons))
        result.update(size=list(image.size),planned=len(windows),planned_native=sum(k=='native' for k,_ in windows))
        print(f'本机逐块终审 {len(result["views"])}/{len(windows)}，原像素 {result["planned_native"]} 块',flush=True)
        for i,(kind,rect) in enumerate(windows):
            if i<len(result['views']):continue
            started=time.monotonic()
            if kind in ('whole','construction'):
                view=overview_image
            elif kind not in ('comparison','draft-comparison','palette'):
                view=image.crop(rect)
                if kind!='native':view.thumbnail((1152,1152),Image.Resampling.LANCZOS)
            if kind=='whole':
                question=('Judge the whole composition, subject identity and count, depicted actions, connections, visual hierarchy, light and requested style. '
                          'Independently verify this earlier observation if present: '+dispute)
            elif kind=='palette':
                stages=(before,image) if source_comparison else (draft_image,before,image)
                panels=[im.resize((max(1,round(im.width*360/max(im.size))),max(1,round(im.height*360/max(im.size)))),Image.Resampling.LANCZOS)
                        for im in stages]
                view=panels
                order=('Image 1 is the source, image 2 is the finished 4K.' if source_comparison else
                       'Image 1 is the approved draft, image 2 is 4K, image 3 is 8K.')
                question=(order+' Compare object colors, hue relationships, shadow temperature, highlights and contrast. '
                          'Flag a visible hue drift, washed-out darks, clipped highlight shapes, or a named color contradicted by the images. Respect the requested medium and lighting. '
                          'Coordinates refer to the affected area in the last image. Ignore pixel-level sharpness in these overview images.')
            elif kind=='construction':
                question=('Independently inventory individual visible hands, feet, animal limbs, and rigid implements when present. '
                          'For each visible hand trace it to a specific wrist, elbow and shoulder; for each animal trace separate paws to its torso. '
                          'Look behind held objects for stray hands. Account for hidden limbs, reflections, bipeds and overlapping people. '
                          'When there are no figures, trace object edges and supports through overlaps. Judge the physical construction you actually see.')
            elif kind=='previous_local':
                question=('Inspect the actual part in this enlargement. A previous pass identified this concrete '
                          'problem here: '+known[rect]['symptom']+'. Trace the part independently. Clear a corrected or '
                          'visibly disproved finding. If the defect is still visible, describe and locate the '
                          'whole affected part including its attachment. Use the surrounding edge as context.')
            elif kind=='contacts':
                question=('Inventory all distinct palm-and-finger clusters actually visible in this interaction. Scan from left to right, including loose fingers behind and between objects. '
                          'For each cluster trace where its wrist physically continues. Flag a stray third cluster with no attached arm, an impossible grip or a fused tool only when visible. '
                          'Hands entering the edges of the crop can belong to people outside it.')
            elif kind in ('comparison','draft-comparison'):
                low,high=(before,image) if kind=='comparison' else (draft_image,before)
                view=compare_pixels(low,high,rect,image.size)
                question=('Image 1 is the lower resolution enlarged to match; image 2 is the same field at final pixel scale. Compare these separately supplied images. '
                          'Identify physically coherent separated tips, fine strokes or edges in image 2. '
                          'Flag smeared focal boundaries, a soft enlarged silhouette with unresolved internal structures, '
                          'invented detail, halos, doubled edges or noisy sharpening. Even if BOTH sides have the same blur, '
                          'hold an unresolved focal subject. Respect intentional softness appropriate to the brief. '
                          'If the author expressly requested this area out of focus or flat, judge the smooth tonal transitions and its role in the composition. Coordinates refer to image 2 alone.')
            else:
                question=('Inspect this actual detail from the same artwork. Apply explicit exclusions to anything actually visible here, including small background figures. A requested subject outside this crop is not missing. '
                          'Trace the physical connections visible here and inspect print quality.')
            local=kind not in ('whole','construction','palette')
            inspection_brief=crop_brief(brief) if local else brief
            guidance=inspection_brief+'\n'+constraints
            prompt=GUIDE+'\n'+question+' A quiet or intentionally blurred surface can pass on its own. Never guess what is outside the crop.\n'+guidance
            row=inspect_native(url,view,prompt) if kind=='native' else judge_view(url,view,prompt)
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
                overview=None
                if kind=='palette' and tentative and row['decision'] in ('local','uncertain'):
                    # A thumbnail is adequate for broad palette drift, not the construction
                    # of a small part. Compare this location on the actual source and master.
                    baseline = before if source_comparison else draft_image
                    pair = compare_pixels(baseline,image,context,image.size)
                    for sample in pair:sample.thumbnail((1280,1280),Image.Resampling.LANCZOS)
                    region = pair
                    verify = (GUIDE+' These are matching close-ups from the source (first) and the finished image '
                              '(second). Verify the alleged defect on the second image. Trace the local boundaries, '
                              'attachments and colour relationships using the actual pixels. Coordinates refer '
                              'to the second image. Finding: '+claim+' '+crop_brief(brief))
                elif kind in ('comparison','draft-comparison','palette'):
                    # Recheck both original pixel fields for this claim, including during arbitration.
                    region=view
                    context=rect
                    overview=overview_image.copy()
                    overview.thumbnail((800,800),Image.Resampling.LANCZOS)
                    draw=ImageDraw.Draw(overview)
                    draw.rectangle(tuple(round(v*n/k) for v,n,k in zip(rect,overview.size*2,image.size*2)),outline='#ec407a',width=2)
                    verify=GUIDE+' '+question+' The last image locates the inspected area in the full composition. Use it to identify deliberate softness. Previous claim: '+claim+' Brief: '+inspection_brief
                second=judge_view(url,region,verify,overview=overview)
                record['context_pixels']=list(context)
                record['corroboration']=second
                if second['decision']=='uncertain' and kind not in ('comparison','draft-comparison','palette'):
                    # Identify the disputed shape without priming the reader with the brief or the alleged defect.
                    identify={'type':'object','properties':{'visible':{'type':'string'},'attachment':{'type':'string'}},
                              'required':['visible','attachment']}
                    observation=json.loads(ask(url,[region],
                        'Describe the actual objects and contours in this enlargement in under 32 words. What does the small shape attach to? If it is unidentifiable, say so. No target description is available.',
                        identify,tag=MODEL))
                    record['blind_zoom_inventory']=observation
                    overview=overview_image.copy()
                    overview.thumbnail((616,660),Image.Resampling.LANCZOS)
                    draw=ImageDraw.Draw(overview)
                    draw.rectangle(tuple(round(v*n/k) for v,n,k in zip(context,overview.size*2,image.size*2)),outline='#f04b73',width=2)
                    zoom=image.crop(context);zoom.thumbnail((720,660),Image.Resampling.LANCZOS)
                    third=arbitrate_finding(url,zoom,overview,row,second,observation)
                    record['independent_context']=third
                    second=third
                if second['decision']=='uncertain' and kind in ('comparison','draft-comparison'):
                    record['issue_category']=issue_category(url,region,second['evidence'],inspection_brief)
                    if record['issue_category']=='unresolved_focus':
                        check=independent_detail_check(url,view,overview,inspection_brief)
                        record['blind_detail_check']=check
                        if check['confirmed']:
                            second=dict(second,decision='clear',evidence=check['trace']['improvement'],operation='none',fix='',target='')
                            record['independent_context']=second
                if second['decision']=='uncertain' and kind in ('comparison','draft-comparison'):
                    # A defect in one physical part can be corrected at print scale. Verify
                    # its construction on the actual newer crop before choosing a whole redraw.
                    anchored=arbitrate_finding(url,view[1],overview,row,second,
                                               {'field':'Both scale samples show this same physical part.'})
                    record['part_verification']=anchored
                    if anchored['decision']=='local':
                        second=anchored
                        record['independent_context']=second
                    elif (anchored['decision']=='clear' and record.get('issue_category')=='other'
                          and record['part_verification'].get('operation')=='none'
                          and re.search(r'attach|connect|anatom|joint|limb|digit|overlap|occlu|support|'
                                        r'merg|fus(?:e|ion)|outline|接合|连接|关节|遮挡|肢体',
                                        row['evidence']+' '+second['evidence'],re.I)):
                        second=anchored
                        record['independent_context']=second
                if second['decision']=='uncertain' and SOFT_HANDLING.search(brief):
                    overview=overview_image.copy()
                    overview.thumbnail((800,800),Image.Resampling.LANCZOS)
                    outline=ImageDraw.Draw(overview)
                    outline.rectangle(tuple(round(v*n/k) for v,n,k in zip(context,overview.size*2,image.size*2)),outline='#ec407a',width=2)
                    policy={'type':'object','properties':{'outcome':{'enum':['intentional','defect','unresolved']},'phrase':{'type':'string'},'evidence':{'type':'string'}},'required':['outcome','phrase','evidence']}
                    instruction=('Image 1 locates this region, image 2 shows its detail. Resolve the editor’s uncertainty using the author’s brief. '
                                 'Choose intentional ONLY if the sole concern is explicitly requested softness or flatness here, and the visible handling matches. '
                                 'Distinguish the intended focal subject from its background. Any other visible defect or ambiguous intent must remain held. '
                                 'Quote the exact words explicitly requesting soft or flat handling in this particular part of the picture in phrase. A medium name alone is insufficient. Give concrete visual evidence. Concern: '+second['evidence']+' Actual brief: '+brief)
                    detail=list(region) if isinstance(region,(tuple,list)) else [region]
                    intent=json.loads(ask(url,[overview,*detail],instruction,policy,tag=MODEL))
                    record['style_intent']=intent
                    if intent.get('outcome')=='intentional' and requested_handling(intent.get('phrase'),brief) and str(intent.get('evidence') or '').strip():
                        second=dict(second,decision='clear',evidence=intent['evidence'],operation='none',fix='',target='')
                        record['independent_context']=second
                if second['decision']!='clear':
                    bounds=map_box(second,context,image.size) or map_box(row,rect,image.size)
                    framing=verify_frame_exit(url,image,overview_image,bounds,
                        row['evidence']+' '+second['evidence']+' '+second['fix'],brief)
                    if framing:
                        record['physical_border_check']=framing
                        if framing['explained']:
                            second=dict(second,decision='clear',operation='none',target='',fix='',evidence=framing['trace'])
                            record['independent_context']=second
                if second['decision']!='clear':
                    box=map_box(second,context,image.size) or map_box(row,rect,image.size)
                    verdict=second['decision']
                    if verdict=='local' and box:
                        first_box=map_box(row,rect,image.size)
                        corroborating_box=map_box(record['corroboration'],context,image.size)
                        prior_box=known[rect]['box'] if kind=='previous_local' else None
                        box=repair_extent(box,[first_box,corroborating_box,prior_box])
                        for axis in (0,1):
                            if box[axis+2]-box[axis]<12:
                                box[axis]=min(988,max(0,(box[axis]+box[axis+2])//2-6))
                                box[axis+2]=box[axis]+12
                    if verdict=='local' and (not box or not second['fix'].strip()):verdict='uncertain'
                    if verdict not in ('local','global','uncertain'):raise ValueError('无效的复审结果')
                    result.update(status=verdict,issue={'box':box,'operation':second.get('operation','redraw'),'fix':second['fix'],'target':second['target'],
                                                        'symptom':second['evidence']})
                    if (verdict=='uncertain' and row.get('operation')=='remove_forbidden'
                            and second.get('operation')=='none' and box and 0<(box[2]-box[0])*(box[3]-box[1])<60000):
                        plan=exclusion_repair(url,image,box,brief,exclusion_clauses,second['evidence'])
                        if plan:
                            record['exclusion_repair']=plan
                            operation='redraw' if plan['action']=='clear_surface' else 'remove_forbidden'
                            result['issue'].update(operation=operation,target=plan['part'],fix=plan['edit'],category='other')
                    if small and result['issue']['operation']!='remove_forbidden' and 'category' not in result['issue']:
                        result['issue']['category']=record.get('issue_category') or issue_category(url,region,second['evidence'],inspection_brief)
            record['elapsed_seconds']=round(time.monotonic()-started,2)
            result['views'].append(record)
            store(report,result)
            if result['status']!='running':
                print('终审退回',result['issue']['symptom'],flush=True);return result
            if (i+1)%8==0:print(f'已审 {i+1}/{len(windows)} 个视图',flush=True)
        result['status']='pass'
        store(report,result)
        return result

def same_composition(original,candidate,brief):
    """Check that a newly sampled source still depicts the approved composition."""
    images=[]
    for path in (original,candidate):
        with Image.open(path) as source:
            image=source.convert('RGB')
            image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
            images.append(image)
    schema={'type':'object','properties':{'same':{'type':'boolean'},'evidence':{'type':'string'}},
            'required':['same','evidence']}
    question=('The first image is the approved composition. Compare the second image to it. Verify the same subject identity, number, action, pose, framing, medium, palette, lighting and spatial relationships. '
              'Allow differences in fine brush marks. If an important change is visible or uncertain, answer false. Brief: '+brief)
    with backend(MODEL) as url:
        answer=json.loads(ask(url,images,question,schema,tag=MODEL))
    return answer.get('same') is True and isinstance(answer.get('evidence'),str) and bool(answer['evidence'].strip())
