"""Independent visual construction check with zoomed evidence and a local 32B VLM."""
import json
import re
from PIL import Image, ImageOps, ImageDraw
from evidence import ask, available, backend, TAG

LARGE='qwen3-vl:32b-instruct'
SCHEMA={'type':'object','properties':{
    'verdict':{'enum':['coherent','local','global','unclear']},
    'observation':{'type':'string'},'fix':{'type':'string'},'target':{'type':'string'},
    'box':{'type':'array','items':{'type':'integer'},'minItems':4,'maxItems':4}},
    'required':['verdict','observation','fix','target','box']}
INSTRUCTION="""Inspect the whole illustration and enlarged areas of that same illustration. Trace actual visible connections: a sleeve through an elbow and forearm to its hand, a paw or wing back to its own torso, and an object to its point of support. At crossings identify which contour is in front and what reappears. Follow the exposed perimeter of rigid or straight-edged props (e.g. a card, book, window or dial). Compare corners and repeated marks to the object’s construction. Check whether a step, gap or double edge has an actual fold, tear, hinge or occluding finger to explain it. When the image shows a book cover, sign, dial or display with prominent writing, inspect the glyph shapes that are actually resolved in the supplied views. Flag an explicit requested inscription only when its actual visible wording differs, or a large unexpected pseudo-inscription when its strokes clearly imitate text without forming a readable script. Small incidental print beyond the evidence is uncertain. Check species-specific front/back anatomy, structural detail and finish in the requested medium. Normal occlusion, foreshortening, a cropped figure or a simple graphic style can be correct. If a particular visible contour really breaks, name its position, the symptom and a concrete repair. Classify one clearly localized issue as local, broad composition/subject errors as global, ambiguity as unclear, otherwise coherent. Give a tight normalized box [left,top,right,bottom] in 0..1000 of the WHOLE picture for local issues. Be brief, use under fifty English words total. Requested scene: """
def proof_regions(im,regions):
    width,height=im.size
    points=[]
    for x,y in regions:
        if all(abs(x-a)>80 or abs(y-b)>95 for a,b in points):points.append((x,y))
        if len(points)==4:break
    for x,y in ((425,440),(625,590),(480,770),(650,300)):
        if len(points)==4:break
        if all(abs(x-a)>80 or abs(y-b)>95 for a,b in points):points.append((x,y))
    side=max(1,min(width,height,round(min(width,height)*.36)))
    for x,y in points:
        l=max(0,min(width-side,round(width*x/1000-side/2)));t=max(0,min(height-side,round(height*y/1000-side/2)))
        yield x,y,(l,t,l+side,t+side)

def proof(im,regions):
    width,height=im.size
    board=Image.new('RGB',(1152,680),'#ece9e1')
    whole=ImageOps.contain(im,(555,646),Image.Resampling.LANCZOS)
    board.paste(whole,((570-whole.width)//2,22+(646-whole.height)//2))
    draw=ImageDraw.Draw(board)
    draw.text((8,4),'Complete illustration',fill='black')
    for i,(x,y,box) in enumerate(proof_regions(im,regions)):
        tile=im.crop(box).resize((282,282),Image.Resampling.LANCZOS)
        px=578+(i%2)*287;py=8+(i//2)*330
        draw.text((px,py),f'Area {i+1}  center {x},{y}',fill='black');board.paste(tile,(px,py+22))
    return board
def verify_repair_location(url,im,result,tag,indicated=None):
    """Confirm a collage finding against actual image coordinates before any inpaint."""
    target=result.get('target','')+' / '+result['observation']
    target=re.sub(r'\[\s*\d+(?:\s*,\s*\d+){3}\s*\]','',target).strip(' /')
    fields={'state':{'enum':['defect_visible','target_is_coherent','target_absent_or_unclear']},
            'target_visible':{'type':'boolean'},'evidence':{'type':'string'}}
    schema={'type':'object','properties':fields,'required':list(fields)}
    evidence=[]
    def local(box):
        if len(box)!=4 or not (0<=box[0]<box[2]<=1000 and 0<=box[1]<box[3]<=1000):return None
        area=im.crop(tuple(round(v*k/1000) for v,k in zip(box,im.size*2)))
        if min(area.size)<8:return None
        area.thumbnail((800,800),Image.Resampling.LANCZOS)
        question=('Only examine this actual crop, without assuming the proposed problem is true. Is the named part and its claimed physical defect visibly present in these pixels? '
                  'Choose target_is_coherent if the target is clearly visible and the claim is contradicted. Use target_absent_or_unclear if the part, its supporting surface or its context is missing. '
                  'Set target_visible as soon as the named part itself can be located, even when its geometry is unclear. '
                  'Allow normal occlusion and the intended illustration medium. In under 24 words describe the visible evidence. Proposed finding: '+target)
        direct=json.loads(ask(url,[area],question,schema,tag=tag))
        finding={'box':box,'direct':direct};evidence.append(finding)
        if direct['state']=='target_absent_or_unclear':
            if not direct['target_visible']:return None
            from print_judge import MODEL
            alternate=MODEL if tag!=MODEL and available(MODEL) else tag
            trace_fields={'part_visible':{'type':'boolean'},'landmark_one':{'type':'string'},
                          'landmark_two':{'type':'string'},'trace':{'type':'string'},
                          'finding':{'enum':['incompatible_shape','visible_and_coherent','unreadable']}}
            trace_schema={'type':'object','properties':trace_fields,'required':list(trace_fields)}
            read=('Locate the named actual physical part in these pixels. Follow the contour between two '
                  'visible landmarks, including any visible termination or overlap. Can both landmarks '
                  'and their physical relationship be seen? Choose unreadable if the relevant endpoint '
                  'cannot be located. Inspect the geometry directly. Concern: '+target)
            for sample in (area,ImageOps.mirror(area)):
                trace=json.loads(ask(url,[sample],read,trace_schema,tag=alternate))
                finding.setdefault('independent_traces',[]).append(trace)
                if (trace['finding']!='visible_and_coherent' or not trace['part_visible'] or
                        not all(str(trace[k]).strip() for k in ('landmark_one','landmark_two','trace'))):
                    return None
            return 'target_is_coherent'
        mirror=json.loads(ask(url,[ImageOps.mirror(area)],question,schema,tag=tag));finding['mirror']=mirror
        if direct['state']==mirror['state']:return direct['state']
        return None
    box=result.get('box',[]);state=local(box)
    if state is None and indicated is not None:
        box=indicated;state=local(box)
    if state is None:
        view=im.copy();view.thumbnail((1100,1100),Image.Resampling.LANCZOS)
        keys=('left','top','right','bottom')
        locate={'type':'object','properties':{k:{'type':'integer','minimum':0,'maximum':1000} for k in keys},'required':list(keys)}
        request=('Locate the named physical part in this complete unmarked image. Return a tight box including the part and immediate contact surface, normalized 0..1000 from the top left. '
                 'Return exactly four integer fields, all zero if absent. Do not assume the claimed defect is true. Part or connection: '+target)
        try:
            found=json.loads(ask(url,[view],request,locate,tag=tag));box=[found[k] for k in keys];state=local(box)
        except (KeyError,ValueError):
            state=None
    result['location_verification']=evidence
    if state=='defect_visible':result['box']=box
    elif state=='target_is_coherent':
        result.update(verdict='coherent',box=[0,0,0,0],observation=evidence[-1]['direct']['evidence'],fix='',target='')
    else:
        result.update(verdict='unclear',box=[0,0,0,0],observation='独立查看实际选区后，无法确认所指缺陷及其位置。',fix='')

def inspect(url,im,prompt,points,path,tag):
    board=proof(im,points);board.save(path,quality=94)
    result=json.loads(ask(url,[board],INSTRUCTION+prompt,SCHEMA,tag=tag))
    result['model']=tag
    if result['verdict']=='local':
        import re
        named=re.search(r'(?:area|region|区域)\s*([1-4])',result['observation'],re.I)
        boxes=list(proof_regions(im,points));idx=int(named[1])-1 if named else len(boxes)
        reported=[round(v*1000/d) for v,d in zip(boxes[idx][2],im.size*2)] if idx<len(boxes) else None
        verify_repair_location(url,im,result,tag,reported)
    return result
def review(im,brief,regions,path,detected=(),observed_people=0):
    points=[]
    for r in regions.values():
        b=r.get('box',[])
        if len(b)==4:points.append(((b[0]+b[2])//2,(b[1]+b[3])//2))
    tag=LARGE if available(LARGE) else TAG
    disputed_arms=False
    try:
        with backend(tag) as url:
            result=inspect(url,im,brief,points,path,tag)
            humans=observed_people or sum(r.get('name')=='person' for r in detected)
            if humans and result['verdict'] in ('coherent','unclear'):
                counts=count_visible_hands(url,im,tag)
                result['hand_inventory']=counts
                # A label alone flagged even a correct two-hand control. Require actual surplus
                # and independent agreement on the number of people before rejecting the layout.
                if counts['status']=='detached_extra' and counts['people']==humans and counts['hands']>2*humans:
                    schema={'type':'object','properties':{'intended':{'type':'boolean'}},'required':['intended']}
                    intentional=json.loads(ask(url,[],'Does this brief explicitly request more than two physical human hands per person or detached hands as a subject? '+brief,schema,tag=tag))['intended']
                    if not intentional:
                        result.update(verdict='global',observation='逐只清点可见手部后，手数超过人物可解释的范围。',fix='Arrange the requested action so each hand visibly belongs to one continuous arm. Keep objects on their own supporting surfaces.')
            if humans==1 and result['verdict'] in ('coherent','unclear'):
                held=handheld_tools(url,im,regions,brief,tag)
                if held:
                    result['held_tools']=held
                    if held.get('comparison',{}).get('relation')=='contradicts_action':
                        result.update(verdict='local',target=held['tool'],box=held['box'],observation=held['comparison']['reason'],fix='Keep only the implement needed for this action. Make the other hand perform its stated supporting action, preserving both wrists and the subject.')
            if humans==1 and result['verdict'] in ('coherent','unclear'):
                grips=inspect_single_grips(url,im,result.get('hand_inventory',{}).get('positions',[]),brief,tag)
                if grips:result['single_grips']=grips
                conflict=next((g for g in grips if g.get('brief_relation')=='contradicts_single_tool'),None)
                if conflict:
                    result.update(verdict='local',target='handheld implement',box=conflict['box'],
                                  observation=conflict['direct']['visible_parts'],
                                  fix='Redraw the grip with one continuous implement. Match the shaft on the two sides of the hand. Remove all visible pieces of the surplus shaft, preserving the hand and its working tool.')
            if humans==1 and result['verdict'] in ('coherent','unclear'):
                fingers=overlapping_digits(url,im,regions,tag)
                if fingers:result['overlapping_digits']=fingers
                faulty=next((f for f in fingers if f.get('mirror',{}).get('verdict')=='defective'),None)
                if faulty:
                    result.update(verdict='local',target='overlapping hands',box=faulty['box'],observation=faulty['direct']['contour'],fix='Reconstruct the individual fingers and palms at this contact. Preserve the intended two-hand gesture, occlusion, sleeves and both wrist connections.')
            if result['verdict'] in ('coherent','unclear') and detected:
                feet=mammal_feet(url,im,detected,tag)
                if feet:result['mammal_feet']=feet
                extra=next((f for f in feet if f['confirmed_surplus']),None)
                if extra:
                    schema={'type':'object','properties':{'intended':{'type':'boolean'}},'required':['intended']}
                    unusual=json.loads(ask(url,[],'Does this brief explicitly request an unusual number of limbs, overlapping animals, or several sequential poses of the same animal in one image? '+brief,schema,tag=tag))['intended']
                    if not unusual:
                        result.update(verdict='local',target=extra['name'],box=extra['box'],observation='两次改变分界位置，仍能在同一只四足动物上定位五个以上独立的脚爪。',fix='Reconstruct this animal as one continuous body with anatomically connected limbs. Preserve its action, scale, ground contact and medium.')
            if result['verdict']=='coherent' and detected:
                arms=trace_arms(url,im,detected,tag)
                if arms:
                    result['bilateral_trace']=arms
                    if 'abrupt_end' in (arms['left_end'],arms['right_end']) or ('initial_detail' in arms and 'unclear' in (arms['left_end'],arms['right_end'])):
                        disputed_arms=True
        if disputed_arms:
            audit_tag='qwen3.8:27b'
            audit=None
            if available(audit_tag):
                with backend(audit_tag) as address:
                    audit=verify_arm_endpoints(address,im,audit_tag)
                    if not audit['confirmed']:
                        # In a wide shot a distant full figure can span only a few
                        # tokens on the overview. Verify its own geometry at print scale.
                        nearer=verify_interior_figure(address,im,detected,audit_tag)
                        if nearer:
                            audit['native_figure']=nearer
                            audit['confirmed']=nearer['confirmed']
                result['arm_endpoint_evidence']=audit
            if not audit or not audit['confirmed']:
                result.update(verdict='unclear',observation='独立的两侧前臂追踪提示有轮廓突然结束，需核实袖口到手的连接。')
        return result
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        return {'model':tag,'verdict':'unclear','observation':str(exc)[:160]}

def edit_locator(image, bounds):
    """A neutral location marker preserves the source pose without revealing a candidate."""
    overview=image.convert('RGB').copy()
    overview.thumbnail((780,780),Image.Resampling.LANCZOS)
    marked=[round(v*n/k) for v,n,k in zip(bounds,overview.size*2,image.size*2)]
    ImageDraw.Draw(overview).rectangle(marked,outline='#be3674',width=2)
    return overview


def assess_edit_boundary(before, after, box, correction=None):
    """Follow connections outside the planned box on the actual composited before/after images."""
    from PIL import ImageChops
    from print_judge import MODEL
    old=before.convert('RGB');new=after.convert('RGB')
    difference=ImageChops.difference(old,new)
    channels=difference.split()
    changed=ImageChops.lighter(ImageChops.lighter(channels[0],channels[1]),channels[2]).point(
        lambda value: 255 if value>10 else 0).getbbox()
    if not changed:return {'status':'uncertain','trace':'No visible edit found at print scale.'}
    target=[round(v*n/1000) for v,n in zip(box,old.size*2)]
    l,t,r,b=(min(changed[0],target[0]),min(changed[1],target[1]),
             max(changed[2],target[2]),max(changed[3],target[3]))
    pad=max(72,round(max(r-l,b-t)*.45))
    field=max(0,l-pad),max(0,t-pad),min(old.width,r+pad),min(old.height,b+pad)
    import hashlib
    import numpy as np
    from scipy.ndimage import gaussian_filter
    # Localize what actually changed near each side, independent of image order.
    # Keep the native crop off the central part that is being redrawn.
    x,y,u,v=changed
    strength=np.asarray(difference.crop(changed)).max(axis=2)
    h,w=strength.shape
    score=gaussian_filter(strength.astype(np.float32),max(1,min(9,min(w,h)/18)))
    bx,by=max(1,round(w*.18)),max(1,round(h*.18))
    bands=[(0,0,bx,h),(w-bx,0,w,h),(0,0,w,by),(0,h-by,w,h)]
    radius=max(34,min(200,round(max(w,h)*.22)))
    sites=[]
    for x1,y1,x2,y2 in bands:
        cut=score[y1:y2,x1:x2]
        dy,dx=np.unravel_index(np.argmax(cut),cut.shape)
        cx=x+x1+int(dx);cy=y+y1+int(dy)
        sites.append((max(0,cx-radius),max(0,cy-radius),min(old.width,cx+radius),min(old.height,cy+radius)))
    def views(im):
        overview=im.crop(field);overview.thumbnail((900,900),Image.Resampling.LANCZOS)
        sheet=Image.new('RGB',(584,584),'#777777')
        for i,site in enumerate(sites):
            tile=ImageOps.contain(im.crop(site),(288,288),Image.Resampling.LANCZOS)
            sheet.paste(tile,((i%2)*296,(i//2)*296))
        return overview,sheet
    old_views,new_views=views(old),views(new)
    # Both directions of a comparison must show the same anonymous A/B pair.
    # Revealing the edited side prompted readers to find new damage in either order.
    first_new=hashlib.sha256(new.crop(field).tobytes()).digest()<hashlib.sha256(old.crop(field).tobytes()).digest()
    a,b=(new_views,old_views) if first_new else (old_views,new_views)
    candidate='A' if first_new else 'B'
    fields={'trace':{'type':'string'},'transition':{'enum':['A','B','both_intact','both_broken','ambiguous']}}
    text=('Two anonymous samples A and B of one illustration. Images 1 and 2 show the same field for A and B; '
          'images 3 and 4 enlarge four matching transition sites for A and B. For each grid the order is left, '
          'right, top, bottom. Inspect where a changed shape meets the surrounding drawing. Follow contours, '
          'material color and shadow across these sites. Look for a line cut off in open space, a jagged '
          'material switch across a continuous surface, a displaced edge, or a rectangular change of shading. '
          'Judge joins with the surrounding drawing independently of which central object you prefer. '
          'Choose A or B only when its transitions are coherent and the other sample shows a specific damaged '
          'transition. Choose both_intact if both join naturally, both_broken when both show breaks, ambiguous '
          'when the evidence cannot distinguish them. Name the exact location and what you see in EACH sample.')
    from illustration import pigment_edit
    recolor=pigment_edit(correction or '')
    if recolor:
        text+=(' The requested change to this illustration is: '+correction+
               ' A marking inside the recolored object can end sharply or completely replace the original '
               'pattern. Trace the EXTERIOR silhouette against the background, continuity of nearby '
               'objects, the contact shadow and common lighting. Judge a defect only where the new '
               'pigment spills onto a different object, cuts an exterior contour, leaves a rectangular '
               'patch or introduces inconsistent shading. Identify landmarks on both sides of any break.')
    with backend(MODEL) as url:
        answer=json.loads(ask(url,[a[0],b[0],a[1],b[1]],text,
                        {'type':'object','properties':fields,'required':list(fields)},tag=MODEL))
    winner=answer['transition']
    status='uncertain'
    if answer.get('trace','').strip():
        if winner in (candidate,'both_intact'):status='intact'
        elif winner in ('A','B'):status='new_damage'
    if recolor and status!='intact' and available(LARGE):
        # Independently trace the candidate itself: a blind pair can prefer a
        # continuous old pattern even when replacing it was the requested edit.
        fields={'inside_landmark':{'type':'string'},'outside_landmark':{'type':'string'},
                'trace':{'type':'string'},'edge':{'enum':['continuous','bleed_or_cut','unclear']},
                'light_and_style':{'enum':['consistent','broken','unclear']}}
        query=('Image 1 shows one detail in context. Image 2 enlarges four parts of the same image. '
               'Identify an exterior object edge and two distinct visible landmarks, one on each side. '
               'Follow the subject perimeter and contact shadow, then inspect nearby background surfaces '
               'for stray pigment, rectangular seams, double outlines, sudden texture or lighting changes. '
               'Patterns ending within the object can be intentional. What physical evidence shows the '
               'exterior edge is continuous, broken, or impossible to tell? Describe the actual drawing. '
               'An intended local change was: '+correction)
        with backend(LARGE) as url:
            proof=json.loads(ask(url,list(new_views),query,{'type':'object','properties':fields,
                                  'required':list(fields)},tag=LARGE,reason=True))
        answer['independent_perimeter']={'model':LARGE,**proof}
        landmarks=[proof[key].strip().casefold() for key in ('inside_landmark','outside_landmark')]
        if all(landmarks) and landmarks[0]!=landmarks[1] and proof['trace'].strip():
            answer['pair_trace']=answer['trace']
            answer['trace']=proof['trace']
            if proof['edge']=='continuous' and proof['light_and_style']=='consistent':
                status='intact'
            elif proof['edge']=='bleed_or_cut' or proof['light_and_style']=='broken':
                status='new_damage'
            else:
                status='uncertain'
    return {**answer,'status':status,'candidate':candidate,'field_pixels':list(field),
            'changed_pixels':list(changed),'transition_sites':sites}



def assess_local_edit(before, after, box, correction, symptom=None, brief=None):
    """Screen the composited native patch, including its connection to the unchanged surroundings."""
    from PIL import ImageChops
    from print_judge import MODEL
    width,height=before.size
    l,t,r,b=[round(v*n/1000) for v,n in zip(box,before.size*2)]
    pad=max(36,round(.30*max(r-l,b-t)))
    bounds=(max(0,l-pad),max(0,t-pad),min(width,r+pad),min(height,b+pad))
    old,new=before.crop(bounds).convert('RGB'),after.crop(bounds).convert('RGB')
    if not ImageChops.difference(old,new).getbbox():
        return {'verdict':'unchanged','evidence':'Identical source and proposed region.'}
    # A tiny crop otherwise occupies only a handful of vision-encoder patches.
    # Use identical optical zoom for both samples, without synthesizing pixels.
    edge=max(old.size)
    scale=min(3,640/edge) if edge<640 else min(1,960/edge)
    size=tuple(max(1,round(side*scale)) for side in old.size)
    old,new=(im.resize(size,Image.Resampling.LANCZOS) for im in (old,new))
    context=[] if bounds==(0,0,width,height) else [edit_locator(before,bounds)]
    props={'verdict':{'enum':['resolved','still_present','new_defect','uncertain']},
           'evidence':{'type':'string'},'next_edit':{'type':'string'}}
    question=('Inspect this single candidate crop in isolation. An earlier version had this reported defect: '
              +str(symptom or 'see correction')+'. The intended result is: '+correction+
              '. Trace the actual visible boundaries, attachments, contact shadows, material and repair seam. '
              'Resolved requires visible evidence for the requested result and no new defect. '
              'Choose still_present if the unwanted shape is visible, uncertain if you cannot locate it. '
              'Judge the physical structure in its actual perspective and occlusion. An acceptable repair must make '
              'the connection visibly plausible. For failure, describe a drawable shape that would correct it.'
              +(' Original scene and medium: '+brief if brief else ''))
    with backend(MODEL) as url:
        result=json.loads(ask(url,[new],question,
                    {'type':'object','properties':props,'required':list(props)},tag=MODEL,reason=True))
        if result.get('verdict')=='resolved':
            # The model must also identify the better image with the source/candidate
            # order hidden. Assign by content, not by which side the caller generated.
            import hashlib
            if hashlib.sha256(new.tobytes()).digest()<hashlib.sha256(old.tobytes()).digest():
                pair,expected=[new,old],'A'
            else:
                pair,expected=[old,new],'B'
            pair_fields={'evidence':{'type':'string'},'better':{'enum':['A','B','equal','uncertain']}}
            choice=json.loads(ask(url,pair+context,'Two independent samples A and B of the same detail. '
                +('Image 3 only locates the subject and establishes its pose. Compare A and B. ' if context else '')
                +'Trace the visible parts, endpoints and their connections in each sample. Trace the actual '
                'structure in each image separately. Which better satisfies this goal: '+correction+
                '? Do not infer the order in which the samples were made. Return equal for indistinguishable '
                'samples and uncertain if neither demonstrates the requested improvement.',
                {'type':'object','properties':pair_fields,'required':list(pair_fields)},tag=MODEL))
            result['blind_comparison']={**choice,'candidate':expected}
            if choice['better']!=expected or not choice['evidence'].strip():
                result.update(verdict='uncertain',next_edit='Make the corrected structure legible at the final pixel scale.')
    if (result.get('verdict')=='uncertain' and result.get('blind_comparison')
            and result.get('evidence','').strip() and available(LARGE)):
        # An isolated reading and a pair comparison disagreed. Use a different
        # local model on each anonymous crop separately; neither sees the other crop
        # or the earlier votes. Both observations must distinguish the actual fault.
        fields={'trace':{'type':'string'},'construction':{'enum':['coherent','defective','ambiguous']},
                'unrelated_damage':{'type':'boolean'}}
        query=('Inspect this single anonymous sample. Inventory the visible parts, their endpoints and '
               'their actual position relative to their parent and nearby surfaces. Do the visible '
               'contours make the requested construction legible in this perspective? Treat hidden '
               'parts and background surfaces according to the physical scene. If you cannot trace '
               'the named structure, choose ambiguous. Goal: '+correction+
               ('. The other image is only the whole-scene location marker.' if context else '')+
               ('. Scene: '+brief if brief else ''))
        traces={}
        with backend(LARGE) as addr:
            for label,frame in zip(('A','B'),pair):
                traces[label]=json.loads(ask(addr,[frame,*context],query,
                    {'type':'object','properties':fields,'required':list(fields)},tag=LARGE))
        source=traces['B' if expected=='A' else 'A']
        candidate=traces[expected]
        result['independent_construction']={'model':LARGE,'samples':traces,'candidate':expected}
        if (source['construction']=='defective' and candidate['construction']=='coherent'
                and candidate['unrelated_damage'] is False and all(v['trace'].strip() for v in traces.values())):
            result.update(verdict='resolved',next_edit='')
    if result.get('verdict')=='resolved' and bounds!=(0,0,width,height):
        boundary=assess_edit_boundary(before,after,box,correction)
        result['boundary_comparison']=boundary
        if boundary.get('status')!='intact' or not boundary.get('trace','').strip():
            result.update(verdict='new_defect' if boundary.get('status')=='new_damage' else 'uncertain',
                          next_edit='Preserve the original contours crossing the change boundary. '+boundary.get('trace',''))
    if result.get('verdict')=='resolved' and not str(result.get('evidence') or '').strip():
        result['verdict']='uncertain'
    return result


def compare_prior_version(current, earlier, box, correction, symptom, brief):
    """Select a former print for full reinspection when a later edit damaged its surroundings."""
    reading=assess_local_edit(current,earlier,box,correction,symptom,brief)
    if reading.get('verdict')!='uncertain' or not reading.get('blind_comparison'):
        return reading
    # The single-image inspection found the earlier detail plausible and the
    # narrow center comparison disagreed. A restored transition must win on
    # specific visible damage, not on a tie or an ambiguous boundary.
    comparison=reading.get('independent_construction',{})
    candidate=comparison.get('samples',{}).get(comparison.get('candidate'),{})
    if candidate.get('construction')=='defective' or candidate.get('unrelated_damage'):
        return reading
    boundary=assess_edit_boundary(current,earlier,box,correction)
    reading['surrounding_recovery']=boundary
    if (boundary['status']=='intact' and boundary.get('candidate') in ('A','B')
            and boundary['transition']==boundary['candidate']):
        reading['eligible_for_full_review']=True
    return reading


def scoped_repair_direction(planned, requested):
    """An outline repair cannot add a new action to the object in the selected area."""
    relations=(r'\b(?:rests?|resting|leans?|leaning|sits?|sitting|stands?|standing|lies|lying|press(?:es|ing)?)'
               r'(?:\s+\w+){0,2}\s+(?:on|against|upon)\b|\b(?:support|contact|load.bearing)\b|承托|倚靠|抵在|支撑',
               r'\b(?:grip(?:s|ping)?|grasp(?:s|ing)?|hold(?:s|ing)?|held|clutch(?:es|ing)?)\b|握住|抓住|握持')
    for pattern in relations:
        if re.search(pattern,planned,re.I) and not re.search(pattern,requested,re.I):
            return requested+' Preserve the existing pose and all foreground/background relationships around the subject.'
    return planned.strip()+' Required visible improvement, within the original scene: '+requested


def plan_local_repair(image, box, target, instruction, medium, directory, stage, brief=None):
    """Locate the complete local attachment when a review box describes only a damaged tip."""
    import hashlib
    model='qwen3.8:27b'
    w,h=image.size
    l,t,r,b=[v*n/1000 for v,n in zip(box,image.size*2)]
    span=min(max(w,h), max(2.8*max(r-l,b-t), .22*min(w,h)))
    width,height=min(w,round(span)),min(h,round(span))
    x=max(0,min(w-width,round((l+r-width)/2)))
    y=max(0,min(h-height,round((t+b-height)/2)))
    view=image.crop((x,y,x+width,y+height))
    view.thumbnail((960,960),Image.Resampling.LANCZOS)
    bbox=[round(1000*(v-origin)/length) for v,origin,length in zip((l,t,r,b),(x,y,x,y),(width,height,width,height))]
    locator=edit_locator(image,(x,y,x+width,y+height))
    key=hashlib.sha256(view.tobytes()+locator.tobytes()+json.dumps(['art-direction-v3',bbox,target,instruction,medium,brief,model]).encode()).hexdigest()
    path=directory/(stage+'-repair-plan.json')
    cached=json.loads(path.read_text()) if path.is_file() else {}
    if cached.get('key')==key:return cached['box'],cached['part'],cached['instruction']
    props={'part':{'type':'string'},'corrected_view':{'type':'string'},
           'box':{'type':'array','items':{'type':'integer'},'minItems':4,'maxItems':4}}
    prompt=('This is an actual cropped region of an illustration. The reported defect is near normalized '
            'left,top,right,bottom '+str(bbox)+'. Locate the smallest COMPLETE structure needed to '
            'reconstruct the damaged part with its attachment. For a malformed iris or muzzle, include '
            'the eye pair and muzzle of that same face; for fingers, include the hand and wrist; for '
            'a spoke or handle, include its immediate mounting contour. Exclude unrelated people, '
            'entire bodies and background objects. Return a box in 0..1000 RELATIVE TO THIS CROP and '
            'name the complete part. Image 2 locates image 1 in the whole scene. Trace the part to its parent '
            'there and establish whether it touches the apparent surface or is in front of it. Describe the '
            'actual connection or separation. Translate the request into a physically possible repair. '
            'Treat guesses by previous readers as hypotheses. Follow the ORIGINAL REQUEST for named colors, '
            'materials, style and actions. The broken surface is not authoritative about what the intended '
            'material looks like. If its repeated shapes are ambiguous, specify an identifiable construction '
            'appropriate to the request, including the scale, direction and grouping of marks. When the '
            'construction is unspecified, choose a simple physically credible one. Trace intact lines '
            'entering and leaving the proposed edit and keep their positions. Preserve deliberate softness. '
            'In corrected_view describe the correct local pose, material, '
            'orientation, color and connection visible in this framing for an illustrator. Use under '
            '45 English words, phrased positively. Target: '+str(target)+'. Repair: '+instruction+
            '. Medium: '+medium+('. Original scene: '+brief if brief else ''))
    with backend(model) as addr:
        result=json.loads(ask(addr,[view,locator],prompt,{'type':'object','properties':props,'required':list(props)},tag=model))
    coords=result.get('box')
    part=result.get('part')
    corrected=result.get('corrected_view')
    if not isinstance(coords,list) or len(coords)!=4 or any(type(v) is not int for v in coords):
        return box,target,instruction
    if not (0<=coords[0]<coords[2]<=1000 and 0<=coords[1]<coords[3]<=1000):
        return box,target,instruction
    if not isinstance(part,str) or not part.strip() or not isinstance(corrected,str) or not corrected.strip():
        return box,target,instruction
    local=[x+width*coords[0]/1000,y+height*coords[1]/1000,x+width*coords[2]/1000,y+height*coords[3]/1000]
    if min(local[2],r)<=max(local[0],l) or min(local[3],b)<=max(local[1],t):return box,target,instruction
    combined=[min(local[0],l),min(local[1],t),max(local[2],r),max(local[3],b)]
    bounds=[max(0,min(1000,round(v*1000/n))) for v,n in zip(combined,image.size*2)]
    name=part.strip()
    # Preserve the repair objective as well as the planner's drawable description.
    correction=scoped_repair_direction(corrected,instruction)
    data={'key':key,'box':bounds,'part':name,'instruction':correction}
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)
    return bounds,name,correction


def landmarks(url,im):
    """Locate meaningful details even in compositions with no humans or animals."""
    reduced=im.copy();reduced.thumbnail((1080,1080),Image.Resampling.LANCZOS)
    name=('interaction','secondary_subject','support','primary_subject')
    pt={'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}
    schema={'type':'object','properties':{k:pt for k in name},'required':list(name)}
    text='Locate four distinct visible details in this artwork. Give centers in [x,y] normalized to 0..1000 from top left. interaction is an important part-to-part connection, secondary_subject is a second focal feature, support is the base or foreground contact, primary_subject is a distinguishing landmark. Use actual features of the image whatever its subject, including still life or landscape.'
    pos=json.loads(ask(url,[reduced],text,schema))
    return [(k,*[max(0,min(1000,round(v))) for v in pos[k]]) for k in name]

ARM_SCHEMA={'type':'object','properties':{
    'image_left_chain':{'type':'string'},'image_right_chain':{'type':'string'},
    'left_end':{'enum':['visible_hand','named_occluder','frame_edge','abrupt_end','unclear']},
    'right_end':{'enum':['visible_hand','named_occluder','frame_edge','abrupt_end','unclear']},
    'evidence':{'type':'string'}},'required':['image_left_chain','image_right_chain','left_end','right_end','evidence']}
def trace_arms(url,im,detected,tag):
    candidates=[r['box'] for r in detected if r.get('name')=='person']
    if not candidates:return None
    box=max(candidates,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    l,t,r,b=[a*v/1000 for a,v in zip(box,im.size*2)]
    if min(r-l,b-t)<60:return None
    w,h=r-l,b-t
    # Cover both shoulder-to-wrist routes even if no hand detector fires at a missing endpoint.
    crop=im.crop((max(0,int(l-.18*w)),max(0,int(t-.05*h)),min(im.width,int(r+.18*w)),min(im.height,int(b+.13*h))))
    crop.thumbnail((980,980),Image.Resampling.LANCZOS)
    instruction=('Describe the two arms of the main person in this crop independently. Start each path at a shoulder, then name each visible sleeve, elbow, forearm, wrist, and actual hand or named occluding surface. Follow the visible outline. '
                 'Do not assign one visible hand to both arms. Describe what immediately follows each cuff. For a bare skin stump with no hand and no occluding edge use abrupt_end. For a real overlap, identify the object hiding it. '
                 'A crop edge is frame_edge. Do not infer invisible joints or treat a mitten as a missing hand. Use image-left and image-right, not the subject’s anatomical left and right. Be concise.')
    detail=json.loads(ask(url,[crop],instruction,ARM_SCHEMA,tag=tag))
    if any(detail[key] in ('abrupt_end','frame_edge') for key in ('left_end','right_end')):
        whole=ImageOps.contain(im,(1152,1152),Image.Resampling.LANCZOS)
        question=('Inspect the original, uncropped illustration. Independently trace both arms from shoulder to their visible endpoints. '
                  'Frame means the actual outer edge of this entire image. Use frame_edge when a continuous sleeve or arm goes out of frame there. '
                  'For a concealed hand, name the actual intervening body, pocket, fabric or object. Use abrupt_end only for an unexplained visible termination inside the image. '
                  'An ambiguous contour is unclear. State what follows each cuff and name the edge or occluding object.')
        confirmed=json.loads(ask(url,[whole],question,ARM_SCHEMA,tag=tag))
        confirmed['initial_detail']=detail
        return confirmed
    return detail

def verify_interior_figure(url, image, detected, tag):
    people=[r['box'] for r in detected if r.get('name')=='person']
    if len(people)!=1:return None
    l,t,r,b=[round(v*n/1000) for v,n in zip(people[0],image.size*2)]
    width,height=r-l,b-t
    if min(width,height)<30:return None
    xpad=max(12,round(width*.22))
    ypad=max(12,round(height*.18))
    bounds=(max(0,l-xpad),max(0,t-ypad),min(image.width,r+xpad),min(image.height,b+ypad))
    if (bounds[2]-bounds[0])*(bounds[3]-bounds[1])>.75*image.width*image.height:return None
    crop=image.crop(bounds)
    scale=min(3,800/max(crop.size))
    zoom=crop.resize(tuple(max(1,round(v*scale)) for v in crop.size),Image.Resampling.LANCZOS)
    check=verify_arm_endpoints(url,zoom,tag)
    check['source_pixels']=list(bounds)
    for side in ('left','right'):
        part=check[side]
        x=bounds[0]+part['x']*crop.width/1000
        y=bounds[1]+part['y']*crop.height/1000
        part['source_point']=[round(x),round(y)]
        # A crop boundary alone cannot explain missing anatomy. An endpoint must
        # also belong to the detected person rather than a second nearby shape.
        if part['state']=='frame_exit' or not (l-xpad/2<=x<=r+xpad/2 and t-ypad/2<=y<=b+ypad/2):
            check['confirmed']=False
    return check


def verify_arm_endpoints(url, im, tag):
    """Corroborate a disputed sleeve endpoint on the original and at its actual location."""
    endpoint = {
        'type': 'object',
        'properties': {
            'state': {'enum': ['hand', 'frame_exit', 'occlusion', 'hidden_far_shoulder', 'internal_break', 'ambiguous']},
            'x': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
            'y': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
            'evidence': {'type': 'string'},
        },
        'required': ['state', 'x', 'y', 'evidence'],
    }
    schema = {'type': 'object', 'properties': {'left': endpoint, 'right': endpoint},
              'required': ['left', 'right']}
    original = ImageOps.contain(im, (1280, 1280), Image.Resampling.LANCZOS)
    prompt = ('Trace each arm of the main figure from its own shoulder. Locate the last visible '
              'point of each arm in the COMPLETE image, normalized to 0..1000 from upper left. '
              'Use image-left and image-right. For an arm that leaves the frame, mark its actual '
              'intersection with the picture edge. For a hidden arm, mark the specific contour '
              'where it passes behind another surface and name that surface. If the far shoulder itself is hidden by the torso in profile, mark that torso surface as hidden_far_shoulder. For an attached '
              'hand, mark the wrist. A clearly unsupported stump inside the picture is an '
              'internal break. If the chain cannot be traced, choose ambiguous.')
    audit = json.loads(ask(url, [original], prompt, schema, tag=tag, reason=True))
    audit.update(model=tag, confirmed=False)
    for side in ('left', 'right'):
        point = audit[side]
        x, y = point['x'], point['y']
        state = point['state']
        if not (0 <= x <= 1000 and 0 <= y <= 1000) or state not in ('hand', 'frame_exit', 'occlusion', 'hidden_far_shoulder'):
            return audit
        if state == 'frame_exit' and min(x, y, 1000-x, 1000-y) > 20:
            return audit
        px, py = x*im.width/1000, y*im.height/1000
        pad = max(30, round(min(im.size)*.19))
        box = (max(0, round(px-pad)), max(0, round(py-pad)),
               min(im.width, round(px+pad)), min(im.height, round(py+pad)))
        crop = im.crop(box)
        crop.thumbnail((720,720), Image.Resampling.LANCZOS)
        if state == 'hand':
            detail = verify_attached_hand(url, crop, tag)
            point['crop_check'] = detail
            if detail['visible_hand'] is not True or detail['joins_own_wrist'] is not True or detail['visible_unsupported_end'] is not False:
                return audit
            continue
        if state == 'frame_exit':
            # This claim has a measurable boundary. Check the sleeve at the identified
            # original edge, without asking the model to translate prose into five labels.
            distances = {'left':x, 'right':1000-x, 'top':y, 'bottom':1000-y}
            edge = min(distances, key=distances.get)
            detail = verify_frame_exit(url, original, crop, side, edge, tag)
            point['crop_check'] = detail
            if detail['sleeve_or_arm_reaches_edge'] is not True or detail['visible_unsupported_end'] is not False:
                return audit
            continue
        check = ('The first image is the entire illustration. The second is the actual region at '
                 'the reported endpoint. Independently inspect where the '+side+'-side arm goes at '
                 'this specific location. Do you see a physically continuous attached hand, an '
                 'identified occluding contour, a far shoulder entirely hidden behind the torso in profile, a sleeve crossing an original frame edge, an '
                 'unsupported internal break, or insufficient evidence? A hidden far shoulder has no visible arm emerging on that side. An internal break requires a visible arm ending inside the image without an occluder. The edges of the detail '
                 'crop alone do not establish a frame exit. Briefly name the visible evidence.')
        detail_schema = {'type': 'object', 'properties': {k:endpoint['properties'][k] for k in ('state','evidence')},
                         'required': ['state','evidence']}
        detail = json.loads(ask(url, [original, crop], check, detail_schema, tag=tag, reason=True))
        point['crop_check'] = detail
        if detail['state'] != state:
            return audit
    audit['confirmed'] = True
    return audit


def verify_attached_hand(url, crop, tag):
    fields = {'evidence':{'type':'string'}, 'visible_hand':{'type':'boolean'},
              'joins_own_wrist':{'type':'boolean'}, 'visible_unsupported_end':{'type':'boolean'}}
    question = ('Inspect only the pixels of this actual crop. Identify what is near its center. Is an '
                'identifiable human hand visible here? Can its palm or back be traced continuously to its '
                'own wrist or sleeve? Is a visibly detached hand or an unsupported interior arm endpoint '
                'present? A sleeve, a paw, or an object alone cannot demonstrate a wrist connection. '
                'If the hand is absent both visible_hand and joins_own_wrist are false. If its attachment '
                'cannot be seen, joins_own_wrist is false. Briefly name what meets what.')
    schema = {'type':'object', 'properties':fields, 'required':list(fields)}
    return json.loads(ask(url, [crop], question, schema, tag=tag))


def verify_frame_exit(url, original, crop, side, edge, tag):
    fields = {'evidence': {'type':'string'}, 'sleeve_or_arm_reaches_edge': {'type':'boolean'},
              'visible_unsupported_end': {'type':'boolean'}}
    question = (f'The second image enlarges a region adjoining the true {edge} boundary of the first, '
                f'complete illustration. Follow the {side}-side arm of its principal figure. In that '
                f'region do pixels depicting the same sleeve or arm reach the {edge} edge of the '
                'complete picture? Answer false if the boundary shows only the torso, a different '
                'object, or background, or if the limb cannot be traced. Separately, is there a '
                'visible unsupported limb ending inside this region? Cite the specific material and '
                'the boundary or internal endpoint in one short sentence.')
    schema = {'type':'object', 'properties':fields, 'required':list(fields)}
    return json.loads(ask(url, [original,crop], question, schema, tag=tag))


HAND_INVENTORY={'type':'object','properties':{'people':{'type':'integer'},'hands':{'type':'array','items':{
    'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}}},'required':['people','hands']}

def count_visible_hands(url,im,tag):
    view=im.copy();view.thumbnail((1150,1150),Image.Resampling.LANCZOS)
    prompt=('Look at this single picture. Count the visible human figures. Locate the center of EACH visible HUMAN HAND actually drawn. Include stray unattached hands if visible. Scan across the entire frame. Ignore reflections, pictures within the picture, hidden hands, animal feet and separate fingertips on the same hand. Use image coordinates normalized to 0..1000 from top left. The hands field is an array of [x,y] points. Each point identifies one distinct visible hand. Keep the JSON compact.')
    def locate(image):
        result=json.loads(ask(url,[image],prompt,HAND_INVENTORY,tag=tag))
        points=[]
        for x,y in result['hands']:
            if 0<=x<=1000 and 0<=y<=1000 and all((x-a)**2+(y-b)**2>12**2 for a,b in points):
                points.append([x,y])
        return max(0,result['people']),points
    people,points=locate(view)
    data={'people':people,'hands':len(points),'positions':points,'status':'clear'}
    if len(points)>2*people:
        other,flipped=locate(ImageOps.mirror(view))
        restored=[[1000-x,y] for x,y in flipped];unused=list(restored);matches=0
        for x,y in points:
            if not unused:break
            nearest=min(unused,key=lambda p:(p[0]-x)**2+(p[1]-y)**2)
            if (nearest[0]-x)**2+(nearest[1]-y)**2<=75**2:
                unused.remove(nearest);matches+=1
        data.update(confirmation_positions=restored,status='detached_extra' if people==other and matches>2*people else 'uncertain')
    return data

FOOT_POINTS={'type':'object','properties':{'paws':{'type':'array','items':{
    'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}}},
    'required':['paws']}
FOUR_LEGGED={'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'}

def mammal_feet(url,im,detected,tag):
    """Split a detected animal in image coordinates so the usual number of legs does not prime the count."""
    animals=[r for r in detected if r.get('name') in FOUR_LEGGED]
    results=[]
    for animal in sorted(animals,key=lambda r:r.get('confidence',0),reverse=True):
        l,u,r,b=animal['box']
        if min((r-l)*im.width,(b-u)*im.height)<60000:continue
        # An adjacent animal can legitimately put its feet inside the same crop.
        if any(other is not animal and min(r,other['box'][2])-max(l,other['box'][0])>0 and min(b,other['box'][3])-max(u,other['box'][1])>0 for other in animals):continue
        frame=im.crop((max(0,(l-20)*im.width//1000),max(0,(u-20)*im.height//1000),
                       min(im.width,(r+20)*im.width//1000),min(im.height,(b+20)*im.height//1000)))
        def scan(split):
            points=[]
            for start,end in ((0,split),(split,1)):
                half=frame.crop((round(start*frame.width),0,round(end*frame.width),frame.height))
                half.thumbnail((750,750),Image.Resampling.LANCZOS)
                instruction=('Locate each visible furry foot or hoof tip in THIS partial illustration, scanning along its entire bottom silhouette. Include stretched feet and upturned paw pads. '
                             'Exclude the tail tip, background details, shoes, the entire leg shaft and anything outside this crop. Return one distinct normalized center point [x,y] 0..1000 per foot endpoint actually visible. Return an empty array if none.')
                if animal['name'] in {'sheep','cow','giraffe','horse','zebra'}:
                    instruction=('Trace each separate lower leg down to where it meets the ground. Report just ONE point at the center of the whole foot of each visible lower leg. '
                                 'A cloven hoof may show two dark toe tips under one woolly ankle. These two toes form ONE foot: mark their midpoint. '
                                 'Do not count separate toes, claw halves, a tail, or a nearby person’s shoes. Allow partial occlusion. Coordinates normalized 0..1000 relative to this crop.')
                for x,y in json.loads(ask(url,[half],instruction,FOOT_POINTS,tag=tag))['paws']:
                    if 0<=x<=1000 and 0<=y<=1000:
                        px=start+(end-start)*x/1000;py=y/1000
                        if all((px-a)**2+(py-z)**2>.043**2 for a,z in points):points.append((px,py))
            return points
        first=scan(.50)
        result={'name':animal['name'],'box':animal['box'],'visible_endpoints':first,'confirmed_surplus':False}
        if len(first)>4:
            second=scan(.55)
            unused=list(second);matched=0
            for x,y in first:
                if not unused:break
                nearest=min(unused,key=lambda p:(x-p[0])**2+(y-p[1])**2)
                if (nearest[0]-x)**2+(nearest[1]-y)**2<=.09**2:
                    matched+=1;unused.remove(nearest)
            result.update(confirmation_endpoints=second,confirmed_surplus=matched>4)
        results.append(result)
    return results

def handheld_tools(url,im,regions,brief,tag):
    boxes=[r['box'] for r in regions.values() if r.get('label')=='human hands and fingers' and len(r.get('box',[]))==4]
    if not boxes:return None
    l=min(b[0] for b in boxes);u=min(b[1] for b in boxes);r=max(b[2] for b in boxes);b=max(b[3] for b in boxes)
    px=max(48,(r-l)//3);py=max(55,(b-u)//2)
    box=[max(0,l-px),max(0,u-py),min(1000,r+px),min(1000,b+py)]
    crop=im.crop(tuple(round(v*s/1000) for v,s in zip(box,im.size*2)))
    crop.thumbnail((900,900),Image.Resampling.LANCZOS)
    props={'first_hand':{'type':'string'},'second_hand':{'type':'string'},'separate_copies':{'type':'integer'},'tool':{'type':'string'}}
    schema={'type':'object','properties':props,'required':list(props)}
    question=('Independently inspect the hands and any handheld implement in this illustration. For each hand identify the distinct tool, or a support surface, simple ingredient, shared single object, or no object. '
              'Count how many separately outlined copies of the SAME handheld implement are visibly present, one per hand with its own shaft or handle. A single tool gripped by both hands counts once. Never count fingers or background objects. Give short English labels only.')
    seen=json.loads(ask(url,[crop],question,schema,tag=tag));seen['box']=box
    if seen['separate_copies']>=2:
        group_schema={'type':'object','properties':{'same_kind':{'type':'boolean'}},'required':['same_kind']}
        grouping='Classify two object labels from an inventory: '+repr(seen['first_hand'])+' and '+repr(seen['second_hand'])+'. Are both handheld tools serving the SAME function? Pen and pencil are both writing tools. A tool and the material it is used on are different. Empty, hidden, uncertain and support surfaces count as different.'
        seen['same_kind']=json.loads(ask(url,[],grouping,group_schema,tag=tag))['same_kind']
    if seen.get('same_kind'):
        compare_schema={'type':'object','properties':{'relation':{'enum':['required_or_plausible','contradicts_action','ambiguous']},'reason':{'type':'string'}},'required':['relation','reason']}
        prompt=('A picture shows a separate '+seen['tool']+' in each of the same person’s hands. The respective objects were independently identified as '+seen['first_hand']+' and '+seen['second_hand']+'. '
                'Read the brief. Could two be intended (drumsticks, paired utensils, juggling, fantastical design)? Only choose contradicts_action when a stated action uses a single implement and the spare tool interferes with the other hand’s role. Missing information is ambiguous. Explain briefly. Brief: '+brief)
        seen['comparison']=json.loads(ask(url,[],prompt,compare_schema,tag=tag))
    return seen


def single_tool_intent(url,brief,tag):
    props={'quantity':{'enum':['single','multiple','unspecified']},'quote':{'type':'string'}}
    question=('Extract the EXPLICIT quantity of thin tools requested in the SAME grip. Return the quantity and the shortest verbatim noun phrase from the brief that states the number and names the tools. '
              'Do not count hands, people or body parts. Multiple includes a pair, two different brushes or a bundle. Single requires an explicit one or single for the implement itself. '
              'When unspecified return unspecified and an empty quote. Work in the input language. Brief: '+brief)
    read=json.loads(ask(url,[],question,{'type':'object','properties':props,'required':list(props)},tag=tag))
    phrase=read['quote'].strip().casefold()
    if read['quantity']=='multiple':return 'plausible_or_intended'
    if read['quantity']=='single' and phrase and phrase in brief.casefold():
        import re
        numbered=re.search(r'\b(?:one|single|sole)\b|[一壹](?:支|枝|根|把|柄|枚|个|個|件|本)',phrase)
        if numbered:return 'contradicts_single_tool'
    return 'ambiguous'

def inspect_single_grips(url,im,positions,brief,tag):
    assessment={'state':{'enum':['one_straight_implement','two_distinct_straight_implements','other_or_unclear']},
                'visible_parts':{'type':'string'}}
    schema={'type':'object','properties':assessment,'required':list(assessment)}
    question='Inspect this one human grip. Trace visible thin rigid tools through the fingers, matching the entering and exiting pieces by their slope and colour. Independent straight axes crossing inside a fist can belong to different complete tools. One brush has a pointed or bristled head on one side, and its wooden tail on the same axis on the other side. A cropped or occluded end is fine. Ignore clothing, fingers and scenery. Is there visual evidence for one rigid handheld implement or for two independently held implements with different, separately identifiable contours? Give names and directions in sixteen words. Scissors, hinged instruments or plier jaws form one manufactured tool.'
    records=[]
    side=max(1,round(min(im.size)*.18))
    for x,y in positions:
        left=max(0,min(im.width-side,round(im.width*x/1000-side/2)))
        top=max(0,min(im.height-side,round(im.height*y/1000-side/2)))
        edges=(left,top,left+side,top+side)
        crop=im.crop(edges).resize((600,600),Image.Resampling.LANCZOS)
        direct=json.loads(ask(url,[crop],question,schema,tag=tag))
        if direct['state']!='two_distinct_straight_implements':continue
        reverse=json.loads(ask(url,[ImageOps.mirror(crop)],question,schema,tag=tag))
        box=[round(e*1000/d) for e,d in zip(edges,im.size*2)]
        record={'box':box,'direct':direct,'mirror':reverse}
        if reverse['state']==direct['state']:
            record['brief_relation']=single_tool_intent(url,brief,tag)
        records.append(record)
    return records

def corroborate_hand_overlap(url, image, box, model):
    """Locate both contenders with wrists in view before diagnosing hand-on-hand contact."""
    width,height=image.size
    l,t,r,b=(round(n*k/1000) for n,k in zip(box,image.size*2))
    padx,pady=max(32,(r-l)//2),max(32,(b-t)//2)
    field=max(0,l-padx),max(0,t-pady),min(width,r+padx),min(height,b+pady)
    view=image.crop(field)
    scale=min(2.5,1056/max(view.size))
    view=view.resize(tuple(max(1,round(scale*k)) for k in view.size),Image.Resampling.LANCZOS)
    fields={'first_hand':{'type':'string'},'second_hand':{'type':'string'},'between':{'type':'string'},
            'relation':{'enum':['visible_skin_contact','separate_visible_contours','one_hand_only','uncertain']}}
    question=('Using this single unlabelled field, identify each visible hand by its sleeve or wrist and '
              'the object or surface it touches. Working from top to bottom trace the nearest edges of '
              'the two hands. What is between them at their closest point? Choose visible_skin_contact '
              'only if the skin of one hand visibly meets or occludes skin of the other at a specific '
              'shared edge. Holding the same object at different points can leave a visible gap. '
              'Describe this actual boundary in between. If you cannot establish it, use uncertain.')
    return json.loads(ask(url,[view],question,{'type':'object','properties':fields,'required':list(fields)},tag=model))


def overlapping_digits(url,im,regions,tag):
    relation_keys={'relation':{'enum':['fingers_of_different_hands_overlap','separated_hands','each_hand_on_a_prop','only_one_hand','unclear']},'between':{'type':'string'}}
    contour_keys={'verdict':{'enum':['coherent','defective','unclear']},'contour':{'type':'string'}}
    def schema(properties):return {'type':'object','properties':properties,'required':list(properties)}
    relation_prompt=('Describe the spatial relation between the two human hands in this close-up, without grading anything. Is there a visible gap or an object between them? Do actual fingers from one hand visibly overlap the other hand’s fingers or skin? If only one hand is visible say so. Give the name of anything between the hands, or point to the exact overlap, in fewer than 18 words.')
    contour_prompt=('Inspect only the physical construction of the drawn hands in this close-up. First identify which wrist each hand belongs to. Trace fingertip and nail contours backwards toward the base of their OWN hand. Allow natural interlocking, occlusion, perspective, or a hidden digit. Describe the actual sequence of visible edges at the contact, including any release of the farther contour. Identify a physical contradiction only if those visible edges establish one. A normal overlap or a partly hidden finger is fine. Return defective only when you can point to a specific visible contradiction. Describe the precise location in under 30 words.')
    records=[]
    for region in regions.values():
        if region.get('label')!='human hands and fingers':continue
        box=region['box']
        crop=im.crop(tuple(round(a*s/1000) for a,s in zip(box,im.size*2)))
        if min(crop.size)<16:continue
        ratio=min(2.,850/max(crop.size))
        crop=crop.resize(tuple(max(1,round(k*ratio)) for k in crop.size),Image.Resampling.LANCZOS)
        relation=json.loads(ask(url,[crop],relation_prompt,schema(relation_keys),tag=tag))
        record={'box':box,'relation':relation}
        if relation['relation']=='fingers_of_different_hands_overlap':
            from print_judge import MODEL
            if tag!=MODEL and available(MODEL):
                cross=corroborate_hand_overlap(url,im,box,MODEL)
                record['spatial_check']=cross
                if cross['relation']!='visible_skin_contact' or not cross['between'].strip():
                    records.append(record)
                    continue
            for key,view in [('direct',crop),('mirror',ImageOps.mirror(crop))]:
                record[key]=json.loads(ask(url,[view],contour_prompt,schema(contour_keys),tag=tag))
                if record[key]['verdict']!='defective':break
        records.append(record)
    return records
