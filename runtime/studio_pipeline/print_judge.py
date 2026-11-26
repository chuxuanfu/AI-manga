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
REVISION = 'print-inspection-42'
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
    return ('Inspect only features visible inside this crop. The following is the full art direction '
            'for identifying visible materials, colors, linework and intentional focus. Judge counts, '
            'missing objects, and whole-scene placement only on the full composition. '
            'Do not infer which object this crop depicts solely from the prompt. Author brief: '+brief)



def actionable_edit(text):
    if not isinstance(text,str) or not text.strip():return False
    return not re.match(r'^(?:none\b|n/?a\b|no (?:visual )?(?:correction|repair|edit|change|action|fix)'
                        r'(?=\s+(?:is\s+)?(?:needed|required|necessary)|[.;,!]|$)|'
                        r'无需(?:修复|修改|更改)|不需要(?:修复|修改))',text.strip(),re.I)


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

def boxes_agree(first, second):
    if not first or not second:return False
    left,top,right,bottom=first
    x,y,u,v=second
    pad_x=max(5,min(right-left,u-x)*.15)
    pad_y=max(5,min(bottom-top,v-y)*.15)
    return max(left,x)<min(right,u)+pad_x and max(top,y)<min(bottom,v)+pad_y


def recheck_fixed_site(url, image, anchor, claim, brief):
    """Two views of the same location prevent confusing adjacent repeated features."""
    from construction_review import edit_locator
    pixels=tuple(round(n*v/1000) for n,v in zip(image.size*2,anchor))
    left,top,right,bottom=pixels
    pad=max(24,round(.23*max(right-left,bottom-top)))
    field=max(0,left-pad),max(0,top-pad),min(image.width,right+pad),min(image.height,bottom+pad)
    detail=image.crop(field)
    scale=min(2.5,1024/max(detail.size))
    detail=detail.resize(tuple(max(1,round(v*scale)) for v in detail.size),Image.Resampling.LANCZOS)
    reading=judge_view(url,detail,GUIDE+' The FIRST image is the intended detail. The marked rectangle '
        'in the second image identifies it in the full scene. Another reader pointed to a DIFFERENT '
        'nearby location. Follow the actual contours in the first image. Decide whether the originally '
        'reported defect is visible HERE. An unrelated neighboring feature cannot corroborate it. '
        'Name the actual shared edge, contact or transition. An obscured connection may be coherent. '
        'Use only the first image for coordinates. Original concern: '+claim+' Brief: '+brief,
        overview=edit_locator(image,pixels),reason=True)
    return reading,field


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


def inspect_repaired_region(url, image, history, brief):
    """Record the present appearance before exposing the reader to a past diagnosis."""
    observed={'type':'object','properties':{'visible':{'type':'string'},'condition':{'type':'string'}},
              'required':['visible','condition']}
    inventory=json.loads(ask(url,[image],
        'Describe this detail in the supplied image. Which surfaces and contours are visible? Trace '
        'their edges and connections. Describe any actual visible damage, anomalous geometry or '
        'unresolved detail. The last edit and the earlier image are unavailable. Be concise.',
        observed,tag=MODEL))
    fields={'trace':{'type':'string'},'defect_now':{'enum':['resolved_in_view','present','indeterminate']}}
    verdict=json.loads(ask(url,[image],
        'This image shows the CURRENT print crop. Before reading any history a reader observed: '
        +json.dumps(inventory,ensure_ascii=False)+'. An OLD version was flagged for: '+history+
        '. Compare that historical claim with the current pixels. In trace give a concrete, current '
        'visual observation in your own words. Is the historical defect visible NOW? Choose resolved_in_view only '
        'when the affected surface or attachment is in view and its current appearance resolves '
        'that claim; present for a visible defect; indeterminate if untraceable or partly out of view. '
        'Consider the current request: '+brief,{'type':'object','properties':fields,'required':list(fields)},tag=MODEL))
    if (verdict['defect_now']=='resolved_in_view' and verdict['trace'].strip()
            and inventory['visible'].strip() and inventory['condition'].strip()):
        row=dict(decision='clear',operation='none',target='',fix='',left=0,top=0,right=0,bottom=0,
                 evidence=verdict['trace'])
    else:
        row=judge_view(url,image,GUIDE+' Inspect the current pixels. Locate this reported concern only '
                       'if visible: '+verdict['trace']+' '+brief)
        if row['decision']=='clear':
            row=dict(row,decision='uncertain',evidence=verdict['trace'])
    return row,{'inventory':inventory,'current_condition':verdict}


def native_review_key(image, rect, size, prompt, model_id):
    """Bind a local tile observation to exactly what the same reviewer sees."""
    inputs=[REVISION,model_id,prompt,NATIVE_SCREEN,image.mode,image.size,rect,size]
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True,ensure_ascii=False).encode())
    key.update(image.tobytes())
    return key.hexdigest()


def previous_native_readings(root, current_report, brief, model_id):
    if root.parent.name=='rounds':
        reports=root.parent.glob('*/print-review/*.json')
    else:
        reports=root.glob('print-review/*.json')
    readings={}
    for path in reports:
        if path==current_report:continue
        report=json.loads(path.read_text())
        if (report.get('revision'),report.get('model_id'),report.get('brief'))!=(REVISION,model_id,brief):continue
        for view in report.get('views',[]):
            key=view.get('native_key')
            row=view.get('observation',{})
            if (key and view.get('view')=='native' and 'reused_from' not in view and
                    row.get('decision')=='clear' and str(row.get('evidence') or '').strip()):
                readings[key]=(row,report['sha256'])
    return readings


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



def inspect_disputed_action(url, image, box, brief):
    """An intact contour can belong to an implausible action or malformed whole part."""
    from construction_review import LARGE
    if not box or not model_identity(LARGE):
        return {'condition':'insufficient','problem':'No independent reading of the disputed part.'}
    pixels=tuple(round(v*n/1000) for v,n in zip(box,image.size*2))
    whole=image.copy()
    whole.thumbnail((1100,1100),Image.Resampling.LANCZOS)
    detail=image.crop(expanded(pixels,image.size))
    detail.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    fields={'visible_shapes':{'type':'string'},'depicted_action':{'type':'string'},'problem':{'type':'string'},
            'condition':{'enum':['physically_plausible','visible_fault','insufficient']}}
    cue=('Image 1 is the original composition. Image 2 shows the disputed area in context. Read independently. '
         'Describe the actual action at this location. Trace the functional structure of the WHOLE visible part, '
         'including its base, neighboring forms, articulations, expected visible ends, occlusion and the contact '
         'carrying its weight. A recognizable tip or individually convincing outline cannot by itself establish '
         'that the entire part and its action make physical sense. Does this arrangement work in three dimensions? '
         'If there is a visible deformity or an incompatible action, locate it relative to the visible parts. '
         'Respect stylized anatomy and illustration conventions, including believable occlusion or hidden '
         'hardware. Set visible_fault only for a concrete shape contradiction, and insufficient only when the '
         'part cannot be traced. Requested image: '+brief)
    return json.loads(ask(url,[whole,detail],cue,{'type':'object','properties':fields,'required':list(fields)},tag=LARGE))


def verified_wrist_at(preliminary, image, box, claim):
    """A wrist location can settle an allegedly absent hand, never a digit or count dispute."""
    if not box or not re.search(r'arm|hand|wrist|sleeve|手臂|手掌|袖口',claim,re.I):return None
    if not re.search(r'lack|missing|absent|merg|not visible|abrupt|cannot see|无法看到|缺|融合',claim,re.I):return None
    if re.search(r'finger|thumb|digit|extra|duplicate|surplus|指|多余',claim,re.I):return None
    audit=preliminary.get('局部连接复审',{}).get('arm_endpoint_evidence',{})
    scoped=audit.get('native_figure',{})
    if not audit.get('confirmed') or not scoped.get('confirmed'):return None
    located=[]
    for side in ('left','right'):
        item=scoped.get(side,{})
        check=item.get('crop_check',{})
        if (item.get('state')!='hand' or check.get('visible_hand') is not True or
                check.get('joins_own_wrist') is not True or check.get('visible_unsupported_end') is not False):continue
        point=item.get('source_point')
        if not point or len(point)!=2:continue
        x,y=[1000*v/n for v,n in zip(point,image.size)]
        if box[0]-30<=x<=box[2]+30 and box[1]-40<=y<=box[3]+40:
            located.append(point)
    return located[0] if len(located)==1 else None


def inspect_located_attachment(url, image, point, brief):
    """Use real native evidence of the same part when a wider reader inspected its neighbor."""
    from construction_review import LARGE
    span=max(42,round(min(image.size)*.13))
    x,y=point
    sample=image.crop((max(0,x-span),max(0,y-span),min(image.width,x+span),min(image.height,y+span)))
    factor=min(3,820/max(sample.size))
    sample=sample.resize(tuple(max(1,round(d*factor)) for d in sample.size),Image.Resampling.LANCZOS)
    overview=image.copy();overview.thumbnail((1050,1050),Image.Resampling.LANCZOS)
    px=x*overview.width/image.width;py=y*overview.height/image.height
    ImageDraw.Draw(overview).ellipse((px-15,py-15,px+15,py+15),outline='#ca3388',width=2)
    fields={'trace':{'type':'string'},'same_disputed_part':{'type':'boolean'},
            'complete_plausible_hand_and_wrist':{'type':'boolean'},'visible_fault':{'type':'boolean'},
            'requested_action_satisfied':{'type':'boolean'},'drawn_correction':{'type':'string'}}
    prompt=('Image 1 is an actual native close-up centered on a localized wrist. Image 2 marks this location '
            'in the composition. Inspect the full hand shape, sleeve termination, neighboring support or '
            'occlusion and each visible digit. Name exactly where the cuff, hand and other surface meet. '
            'The target is the human wrist at the marked location. If the detail only shows another object, '
            'same_disputed_part is false. A detached, merged, physically distorted or unresolved hand cannot '
            'be marked complete. Distinguish between a present but poorly drawn hand and a hand that is absent. '
            'Name the visible skin silhouette and knuckles. If defective, give one drawable correction grounded '
            'in these pixels. Check only the action actually requested. Actual brief: '+brief)
    result=json.loads(ask(url,[sample,overview],prompt,{'type':'object','properties':fields,'required':list(fields)},tag=LARGE))
    result['confirmed']=(result['same_disputed_part'] is True and result['complete_plausible_hand_and_wrist'] is True
                         and result['visible_fault'] is False and result['requested_action_satisfied'] is True
                         and bool(result['trace'].strip()))
    return result


def disputed_action_resolved(reading):
    return (reading.get('condition')=='physically_plausible' and
            all(str(reading.get(key) or '').strip() for key in ('visible_shapes','depicted_action')))


HARDWARE_CLAIM=re.compile(r'\b(?:brackets?|hinge[sd]?|sockets?|bolts?|welds?|rivets?|mounting|fasteners?|'
                          r'locking mechanism|articulat(?:ion|ed) joint|load path|cleats?|mechanical joint|mechanical.{0,100}(?:joint|articulation)|'
                          r'order of (?:over/under )?crossings)\b|铰链|螺栓|焊缝|紧固件|承力路径',re.I)


def compare_local_definition(url, pair):
    # With both samples visible, a reader can correctly describe blur in the source
    # and accidentally attribute its severity to the master. Inspect the master alone first.
    properties = {'visible_edges': {'type': 'string'},
                  'condition': {'enum': ['well_defined', 'optically_degraded', 'insufficient']}}
    schema = {'type': 'object', 'properties': properties, 'required': list(properties)}
    message = ('Inspect only this final print patch at its pixel scale. Trace a specific material edge, '
               'separated fine stroke or smooth intended tone. Flag smeared or interpolated focal forms, '
               'halos, noisy sharpening and invented or doubled outlines. For an ambiguous sample choose '
               'insufficient. State what the pixels actually show.')
    final = json.loads(ask(url, [pair[1]], message, schema, tag=MODEL))
    if final['condition'] != 'well_defined' or not final['visible_edges'].strip():
        return final
    message = ('Image 1 is the final print patch you are grading. Image 2 is its lower-resolution source '
               'at matched scale. Trace the same visible feature in each. In image 1 look for a cleanly '
               'resolved form or intentionally smooth tone. If image 1 merely enlarges a soft focal '
               'feature, loses real structure or adds noise or halos, choose optically_degraded. '
               'If the comparison is ambiguous choose insufficient. Classify image 1 only.')
    graded = json.loads(ask(url, [pair[1], pair[0]], message, schema, tag=MODEL))
    return dict(graded, final_patch=final)


def detail_comparison_prompt(question, brief):
    return ('You are inspecting two matched print samples for optical resolution. '+question+
            ' Grade the visible marks and tones in image 2. Whole-scene anatomy, the number of objects '
            'and connections to objects outside this field have their own inspections. For clear give '
            'specific pixel evidence, operation none and zero coordinates. When broader context is '
            'needed to distinguish intended softness choose uncertain. Named subjects may be outside '
            'both samples. Use the art direction only to interpret the marks that are present: '+brief)



CROP_OMISSION=re.compile(r'(?:crop|close.up|field of view).{0,115}?(?:exclude|cut.?off|does not (?:include|show)|'
                         r'doesn.t (?:include|show)|too (?:tight|narrow)|out of view)|'
                         r'(?:outside|beyond|out of).{0,28}?(?:crop|close.up|cropped view)|'
                         r'(?:not visible|cannot be seen|can.t see).{0,30}?(?:crop|close.up)|裁切.{0,15}(?:未包含|没有显示|未显示)',re.I)


def verify_outside_crop(url,image,brief,claim):
    """Find a claimed missing contact in the whole illustration before reviewing its pixels."""
    from construction_review import LARGE
    if not CROP_OMISSION.search(claim) or not model_identity(LARGE):return None
    whole=image.copy()
    whole.thumbnail((1440,1440),Image.Resampling.LANCZOS)
    edges=('left','top','right','bottom')
    fields={k:{'type':'integer','minimum':0,'maximum':1000} for k in edges}
    fields.update(condition={'enum':['satisfied_in_view','visible_fault','not_identified']},trace={'type':'string'},clause={'type':'string'})
    question=('An earlier crop excluded part of the scene. Using this COMPLETE illustration, find the actual '
              'required part, support or contact the crop omitted. Locate that contact itself, not merely the '
              'nearest identifiable subject. Bound the two meeting contours in normalized coordinates. '
              'If the brief asks for a relation spanning two sites, trace the visible path between them and '
              'identify the far attachment. In clause quote the exact words of the original brief requiring the relation. Mark satisfied_in_view only if the requested visible relationship '
              'can be traced. A nearby shape or a guessed off-screen endpoint is insufficient. Crop report: '
              +claim+' Author brief: '+brief)
    site=json.loads(ask(url,[whole],question,{'type':'object','properties':fields,'required':list(fields)},tag=LARGE))
    result={'search':site,'confirmed':False}
    box=[site[k] for k in edges]
    l,t,r,b=box
    clause=site.get('clause','').strip().strip('“”\"')
    if (not clause or ' '.join(clause.casefold().split()) not in ' '.join(brief.casefold().split())
            or site['condition']!='satisfied_in_view' or not site['trace'].strip()
            or not (2<l<r<998 and 2<t<b<998 and r-l<400 and b-t<400)):return result
    rect=tuple(round(n*k/1000) for n,k in zip(box,image.size*2))
    detail=image.crop(expanded(rect,image.size))
    scale=min(2.5,1024/max(detail.size))
    detail=detail.resize(tuple(max(1,round(n*scale)) for n in detail.size),Image.Resampling.LANCZOS)
    keys={'condition':{'enum':['satisfied_in_view','visible_fault','uncertain']},'evidence':{'type':'string'}}
    request=('This single enlargement is your only evidence. Trace the two actual contours and '
             'identify the physical surface on each side of their meeting point. Verify this original '
             'requirement: '+clause+'. Satisfied_in_view requires the specific parts AND the relevant '
             'connection to be visible in these pixels. If a required route or its endpoint leaves '
             'this field, or the attachment could be a different object, choose uncertain. '
             'A physical contact can occlude its own hidden side. A visible gap, fused part or '
             'contradictory crossing is a fault. Describe the evidence you see.')
    check=json.loads(ask(url,[detail],request,{'type':'object','properties':keys,'required':list(keys)},tag=MODEL))
    result.update(check=check,box=box,confirmed=check['condition']=='satisfied_in_view' and bool(check['evidence'].strip()))
    return result


def verify_depicted_connection(url,image,box,claim,brief):
    """Check visible construction separately from a demand to illustrate hidden hardware."""
    from construction_review import LARGE
    if not box or not HARDWARE_CLAIM.search(claim) or not model_identity(LARGE):return None
    if not 0<(box[2]-box[0])*(box[3]-box[1])<160000:return None
    pixels=tuple(round(v*n/1000) for v,n in zip(box,image.size*2))
    field=expanded(pixels,image.size)
    detail=image.crop(field)
    scale=min(2,1000/max(detail.size))
    detail=detail.resize(tuple(max(1,round(n*scale)) for n in detail.size),Image.Resampling.LANCZOS)
    overview=image.copy()
    overview.thumbnail((920,920),Image.Resampling.LANCZOS)
    ImageDraw.Draw(overview).rectangle(tuple(round(v*n/1000) for v,n in zip(box,overview.size*2)),outline='#c92e69',width=2)
    fields={'trace':{'type':'string'},'condition':{'enum':['visible_coherent_contact','visible_break','unresolved']}}
    read=json.loads(ask(url,[detail,overview],
        'Inspect the actual physical connection at the marked location. First image is the detail. '
        'Name both visible components and trace their contours to the shared contact or explicit occluding '
        'edge. Identify what goes in front and what goes behind. Determine whether the DRAWN shapes make '
        'a coherent contact. A real gap, shape that abruptly changes into something else, or inconsistent '
        'depth counts as a visible break. Do not invent an internal mechanism to justify a broken silhouette. '
        'Choose unresolved when the relevant contact cannot be located. Work from the drawing itself.',
        {'type':'object','properties':fields,'required':list(fields)},tag=LARGE))
    result={'blind_model':LARGE,'inventory':read,'disproved':False}
    if read['condition']!='visible_coherent_contact' or not read['trace'].strip():return result
    fields={'trace':{'type':'string'},'basis':{'enum':['only_unseen_internal_hardware','visible_fault',
                                                     'required_visible_detail_missing','unresolved']}}
    check=json.loads(ask(url,[detail,overview],
        'A print reviewer flagged: '+claim+' An independent reader traced the connection: '+read['trace']+
        ' Verify both against the detail and its location. Classify only_unseen_internal_hardware ONLY if '
        'the exterior contact and depth order are visibly coherent, the allegation merely demands a visible '
        'internal fitting, hidden strand or connector beyond what this illustration must depict, and the '
        'AUTHOR did not require that detail to be shown. An actual visible gap, contradictory crossing, '
        'false silhouette, count error or missing requested object remains a fault. If the contact is '
        'unresolved do not clear it. State the precise visible contour that distinguishes the cases. '
        'Author brief: '+brief,{'type':'object','properties':fields,'required':list(fields)},tag=MODEL))
    result['verification']=check
    result['disproved']=check['basis']=='only_unseen_internal_hardware' and bool(check['trace'].strip())
    return result


ABSENCE_CLAIM=re.compile(r'\b(?:missing|absent|cropped|cut off|out of frame|extend the canvas|feet are (?:not|never) visible|'
                         r'lettering|spelling|inscription|duplicat\w*|extra|surplus|forbidden)\b|缺失|缺少|画幅|裁切|截断|多余|重复',re.I)

def verify_part_identity(url, image, box, finding, earlier_claim=''):
    """Avoid repainting an intact neighboring object mistaken for the alleged part."""
    from construction_review import LARGE
    claim=finding.get('symptom') or finding.get('evidence','')
    target=finding.get('target','').strip()
    # Identifying a visible neighbor cannot settle an absent/off-canvas subject,
    # a count disagreement, a written instruction, or a whole-scene omission.
    if (not box or not target or target.casefold() in ('local','none','unknown','region','subject') or
            ABSENCE_CLAIM.search(' '.join((claim,earlier_claim,str(finding.get('fix',''))))) or
            finding.get('operation')!='redraw' or not model_identity(LARGE)
            or not 0<(box[2]-box[0])*(box[3]-box[1])<90000):
        return None
    width,height=image.size
    left,top,right,bottom=(round(v*k/1000) for v,k in zip(box,image.size*2))
    span=max(right-left,bottom-top)
    if span<2:return None
    side=max(span*4,min(1152,round(min(width,height)*.5)))
    def axis(lo,hi,length):
        extent=min(length,side)
        start=max(0,min(length-extent,(lo+hi-extent)//2))
        return start,start+extent
    x1,x2=axis(left,right,width)
    y1,y2=axis(top,bottom,height)
    wide=image.crop((x1,y1,x2,y2))
    draw=ImageDraw.Draw(wide)
    for x,dx in ((left-x1-5,-1),(right-x1+5,1)):
        for y,dy in ((top-y1-5,-1),(bottom-y1+5,1)):
            draw.line([(x-12*dx,y),(x,y),(x,y-12*dy)],fill='#ff00da',width=2)
    wide.thumbnail((1000,1000),Image.Resampling.LANCZOS)
    detail=image.crop((max(0,left-span//2),max(0,top-span//2),
                       min(width,right+span//2),min(height,bottom+span//2)))
    detail.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    schema={'type':'object','properties':{k:{'type':'string'} for k in ('feature','material','attachment','image_evidence')},
            'required':['feature','material','attachment','image_evidence']}
    prompt=('Inspect the shape between the magenta corner marks in the first image; the second is an unmarked '
            'enlargement from the same original. Identify the actual physical object, its material and which '
            'larger object it is part of. Trace the contour outwards in the wider picture. Look for material '
            'continuity, edges, and the actual point of attachment. If ownership is not visible, state ambiguous. '
            'No target identity has been provided. Be concise.')
    inventory=json.loads(ask(url,[wide,detail],prompt,schema,tag=LARGE))
    fields={'trace':{'type':'string'}, 'is_claimed_part':{'enum':['yes','no','unresolved']},
            'blind_identification_visible':{'type':'boolean'},'intact_contour':{'type':'boolean'}}
    audit=json.loads(ask(url,[wide,detail],
        'Question: Is the outlined physical feature a part of «'+target+'» as alleged in this finding? '
        +claim+' In trace, first name the alleged part and the actual visible part, then trace the actual '
        'owner across the corner marks to the attachment. Answer is_claimed_part=yes if it really is that '
        'part, including synonyms and different names for the same thing. Answer no only when the allegation '
        'confuses entirely distinct kinds of objects and the actual attachment is visible. Use unresolved '
        'if the marks miss the target, or ownership cannot be traced. A different material or adjacent '
        'occluder alone does not contradict identity. Check this unaided inventory against the picture: '
        +json.dumps(inventory,ensure_ascii=False)+' Set blind_identification_visible=true only for a '
        'visibly confirmed identity and attachment. Set intact_contour=true only for an intact actual form.',
        {'type':'object','properties':fields,'required':list(fields)},tag=MODEL))
    return {'blind_model':LARGE,'blind_model_id':model_identity(LARGE),'inventory':inventory,'verification':audit,
            'disproved':audit['is_claimed_part']=='no' and audit['blind_identification_visible'] is True
                        and audit['intact_contour'] is True
                        and all(str(value).strip() for value in inventory.values()) and bool(audit['trace'].strip())}

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
    # An enum can contradict the model's description of the contour actually
    # crossing the image edge. Resolve that disagreement with a separate reader.
    contradicted=(result['outcome']=='broken_inside' and bool(re.search(
        r'continues? (?:past|beyond|out of)|extends? (?:past|beyond|out of)|'
        r'(?:cross(?:es|ing)|reaches?) (?:the |its )?(?:physical |outer )?(?:edge|border)',
        result['trace'],re.I)))
    if (result['outcome']=='unresolved' or contradicted) and not result['brief_requires_missing_part'] and result['edge'] in edges:
        # Ask about literal geometry on the original border rather than asking the
        # same reviewer to interpret its own uncertain verdict.
        fields={'part_reaches_border':{'type':'boolean'},'break_before_border':{'type':'boolean'},
                'another_visible_defect':{'type':'boolean'},'edge':{'enum':['left','top','right','bottom','none']},
                'required_visible_feature_absent':{'type':'boolean'},'path':{'type':'string'}}
        from construction_review import LARGE
        auditor=LARGE if contradicted and model_identity(LARGE) else MODEL
        if contradicted and auditor==MODEL:return result
        observation=json.loads(ask(url,[crop,locator],
           'Inspect the actual detail (first) and the entire frame with the location outlined (second). '
           'The feature under examination is described here: '+claim+
           '. Which physical image edge, if any, does this exact visible part reach? '
           'Trace the visible path to that edge. Does it actually end or change into an unrelated form '
           'BEFORE the border? Is some other defect in this same part visible on the canvas? '
           'Use only the following scene brief to decide whether an absent part was explicitly required '
           'to be entirely visible: '+brief,
           {'type':'object','properties':fields,'required':list(fields)},tag=auditor,reason=auditor==MODEL))
        result['edge_trace']=observation
        result['edge_reader']=auditor
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


def inspect(path, brief, root, preliminary, *, small=None, focus=None, draft=None, source_comparison=False, construction_draft=False):
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
    if construction_draft:key+='-draft-structure'
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
        if construction_draft:
            windows=[(kind,rect) for kind,rect in windows if kind in ('whole','construction','contacts')]
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
        native_readings=None
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
            if construction_draft:
                question+=' This is the composition draft before print rendering. Check visible structure, declared material and contact at this drawing scale. Fine print resolution is evaluated on a later file.'
            prompt=GUIDE+'\n'+question+' A quiet or intentionally blurred surface can pass on its own. Never guess what is outside the crop.\n'+guidance
            if kind in ('comparison','draft-comparison'):
                prompt=detail_comparison_prompt(question,brief)
            if kind=='palette':
                # These thumbnails carry color relationships, not reliable small contours.
                # Construction and all overlapping native tiles have dedicated views.
                prompt=('Review only color fidelity across these palette previews. '+question+
                        ' Flag an issue only for a visible color change or a contradiction with a named '
                        'palette. For clear, describe a matching hue relationship, set operation to none '
                        'and coordinates to zero. For a color concern give a local box in the final '
                        'image, normalized to 0..1000, and a concrete color correction. Brief: '+brief)
            prior_reading=None
            reuse=None
            if kind=='native':
                if native_readings is None:
                    native_readings=previous_native_readings(root,report,brief,model_id)
                native_key=native_review_key(view,rect,image.size,prompt,model_id)
                reuse=native_readings.get(native_key)
                row=reuse[0] if reuse else inspect_native(url,view,prompt)
            elif kind=='previous_local':
                row,prior_reading=inspect_repaired_region(url,view,known[rect]['symptom'],guidance)
            else:
                row=judge_view(url,view,prompt)
            record={'view':kind,'pixels':list(rect),'observation':row}
            if kind=='native':record['native_key']=native_key
            if reuse:record['reused_from']=reuse[1]
            if prior_reading:record['current_pixel_check']=prior_reading
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
                    if kind in ('comparison','draft-comparison'):
                        verify=(detail_comparison_prompt(question,brief)+' Image 3 locates the samples. '
                                'Verify whether this alleged optical defect is visible in image 2: '+claim)
                second=judge_view(url,region,verify,overview=overview)
                record['context_pixels']=list(context)
                record['corroboration']=second
                palette_structure=(kind=='palette' and tentative and
                                   not actionable_edit(second.get('fix')) and
                                   re.search(r'attach|connect|limb|digit|overlap|occlu|support|merg|fus(?:e|ion)|'
                                             r'anatom|paws?|fingers?|contour|接合|连接|关节|遮挡|肢体',
                                             claim,re.I))
                if second['decision']=='uncertain' and (kind not in ('comparison','draft-comparison','palette') or palette_structure):
                    # Identify the disputed shape without priming the reader with the brief or the alleged defect.
                    identify={'type':'object','properties':{'visible':{'type':'string'},'attachment':{'type':'string'}},
                              'required':['visible','attachment']}
                    blind_detail=region[-1] if palette_structure else region
                    observation=json.loads(ask(url,[blind_detail],
                        'Describe the actual objects and contours in this enlargement in under 32 words. What does the small shape attach to? If it is unidentifiable, say so. No target description is available.',
                        identify,tag=MODEL))
                    record['blind_zoom_inventory']=observation
                    overview=overview_image.copy()
                    overview.thumbnail((616,660),Image.Resampling.LANCZOS)
                    draw=ImageDraw.Draw(overview)
                    draw.rectangle(tuple(round(v*n/k) for v,n,k in zip(context,overview.size*2,image.size*2)),outline='#f04b73',width=2)
                    zoom=image.crop(context);zoom.thumbnail((720,660),Image.Resampling.LANCZOS)
                    third=arbitrate_finding(url,zoom,overview,row,second,observation)
                    if (third['decision']=='clear' and kind in ('whole','construction','contacts')
                            and second.get('operation')=='redraw' and actionable_edit(second.get('fix'))):
                        functional=inspect_disputed_action(url,image,tentative,brief)
                        record['independent_action_check']=functional
                        if not disputed_action_resolved(functional):
                            third=dict(second,decision='uncertain')
                            if functional.get('condition')=='visible_fault':
                                third['evidence']+=' Independent whole-part reading: '+functional['problem']
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
                anchor=known[rect]['box'] if kind=='previous_local' else tentative
                if (anchor and second['decision'] in ('local','uncertain') and
                        kind not in ('comparison','draft-comparison','palette')):
                    mapped=map_box(second,context,image.size)
                    if mapped and not boxes_agree(anchor,mapped):
                        record['mislocalized_context']={'intended_box':anchor,'observed_box':mapped,'reading':second}
                        claim=known[rect]['symptom'] if kind=='previous_local' else row['evidence']
                        second,context=recheck_fixed_site(url,image,anchor,claim,brief)
                        record['localized_context_pixels']=list(context)
                        record['independent_context']=second
                        region=image.crop(context)
                        if second['decision']!='clear' and not boxes_agree(anchor,map_box(second,context,image.size)):
                            # The wider reader never corroborated this location. Keep the
                            # original issue here until it can be resolved on actual pixels.
                            second=dict(row,decision='uncertain')
                            context=rect
                            record['independent_context']=second
                if second['decision']!='clear' and kind in ('whole','construction','contacts'):
                    bounds=map_box(second,context,image.size) or map_box(row,rect,image.size)
                    claim=row['evidence']+' '+second['evidence']
                    point=verified_wrist_at(preliminary,image,bounds,claim)
                    if point:
                        located=inspect_located_attachment(url,image,point,brief)
                        record['localized_attachment']={**located,'source_point':point}
                        if located['confirmed']:
                            second=dict(second,decision='clear',operation='none',target='',fix='',evidence=located['trace'])
                            record['independent_context']=second
                        elif (located['same_disputed_part'] is True and located['visible_fault'] is True
                              and actionable_edit(located['drawn_correction']) and located['trace'].strip()):
                            second=dict(second,decision='local',operation='redraw',target='hand and wrist',
                                        evidence=located['trace'],fix=located['drawn_correction'])
                            record['independent_context']=second
                if second['decision']!='clear':
                    bounds=map_box(second,context,image.size) or map_box(row,rect,image.size)
                    if second.get('decision')=='local' and kind not in ('palette','comparison','draft-comparison'):
                        identity=verify_part_identity(url,image,bounds,second,row['evidence'])
                        if identity:
                            record['part_identity']=identity
                            if identity['disproved']:
                                second=dict(second,decision='clear',operation='none',target='',fix='',evidence=identity['verification']['trace'])
                                record['independent_context']=second
                if second['decision']!='clear':
                    bounds=map_box(second,context,image.size) or map_box(row,rect,image.size)
                    if second['decision'] in ('uncertain','local') and second.get('operation')=='redraw':
                        contact=verify_depicted_connection(url,image,bounds,row['evidence']+' '+second['evidence']+
                            ' Suggested edit: '+str(second.get('fix','')),brief)
                        if contact:
                            record['visible_connection']=contact
                            if contact['disproved']:
                                second=dict(second,decision='clear',operation='none',target='',fix='',evidence=contact['verification']['trace'])
                                record['independent_context']=second
                if second['decision']=='uncertain' and kind in ('whole','construction','contacts','subject','comparison','draft-comparison'):
                    offscreen=verify_outside_crop(url,image,brief,second['evidence'])
                    if offscreen:
                        record['whole_scene_location']=offscreen
                        if offscreen['confirmed']:
                            cross=dict(second,decision='clear',operation='none',fix='',target='',evidence=offscreen['check']['evidence'])
                            if kind in ('comparison','draft-comparison'):
                                # The missing contact now has its own native inspection. Still verify that
                                # the two ORIGINAL comparison samples agree in sharpness, ink and material.
                                optical=compare_local_definition(url,view)
                                record['definition_at_original_site']=optical
                                if optical['condition']=='well_defined' and optical['visible_edges'].strip():
                                    cross=dict(cross,evidence=optical['visible_edges'])
                                else:
                                    cross=dict(second,decision='uncertain',operation='none',fix='',target='print detail',
                                               left=0,top=0,right=1000,bottom=1000,
                                               evidence=optical['visible_edges'] or 'Definition could not be established at the original site.')
                                    record['issue_category']='unresolved_focus'
                                context=rect
                                region=view
                            second=cross
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
                    if not actionable_edit(second.get('fix')) and second.get('operation')=='redraw':
                        second=dict(second,operation='none',fix='')
                        verdict='uncertain'
                    if verdict=='local' and box:
                        first_box=map_box(row,rect,image.size)
                        corroborating_box=map_box(record['corroboration'],record['context_pixels'],image.size)
                        if not boxes_agree(box,corroborating_box):corroborating_box=None
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
