"""Inspect native evidence for the most specific visible promises in the brief."""
import json
import math
import re
from pathlib import Path

from evidence import ask, model_identity
from PIL import Image


VIEWPOINT=re.compile(r'\b(?:in (?:strict |full |side )?profile|side[- ](?:on|profile|view)|'
                     r'(?:front|back|rear)[- ]view|(?:front|head)[- ]on|seen from (?:behind|the side)|'
                     r'three[- ]quarter(?:s)? (?:view|silhouette|profile))\b|'
                     r'正侧面|侧面(?:视角|轮廓)|背对镜头|正对镜头|正面视角|四分之三侧面',re.I)


# Only explicit camera-relative words can establish a mismatch. A subject looking
# toward another object says nothing about which side the camera can see.
CAMERA_ANGLES = {
    'three-quarter': r'\bthree[- ]quarters? (?:view|profile|silhouette|angle)\b|四分之三侧面',
    'side': r'\b(?:side[- ](?:profile|on|view)|(?:in|shown in|full|strict) (?:strict )?profile|'
            r'(?:seen|viewed|shown|drawn) from the side)\b|正侧面|侧面视角',
    'rear': r'\b(?:rear[- ]view|back[- ]view|(?:seen|viewed|shown|drawn) (?:directly )?from behind|'
            r'(?:its|her|his|their) back (?:is )?(?:turned |facing )?(?:to|toward) the (?:viewer|camera))\b|背对镜头',
    'front': r'\b(?:front[- ]view|(?:seen|viewed|shown|drawn) (?:directly )?from the front|'
             r'front[- ]on|head[- ]on|facing (?:directly |straight )?(?:towards? )?the (?:viewer|camera))\b|正对镜头|正面视角',
}
ANGLE_SUBJECTS = (
    r'cat|kitten|feline|tabby',r'dog|puppy|canine',r'girl|woman|lady|heroine',
    r'boy|man|gentleman',r'face|head',r'bird|owl|heron|crane',r'fox|vixen',
)


def camera_conflict(requirement,inventory):
    """Corroborate incompatible, explicitly named views on the same depicted subject."""
    def angles(text):
        return {key for key,pattern in CAMERA_ANGLES.items() if re.search(pattern,text,re.I)}
    wanted=angles(requirement)
    if len(wanted)!=1:return None
    # A turned face and a torso seen from behind may belong to the very same
    # person. Use statements which name the requested subject at that level.
    head_only=bool(re.search(r"\b(?:face|head)\s+(?:is |appears |turned |in |shown |seen )",requirement,re.I))
    for clause in re.split(r'[.;!?。；]',inventory):
        if re.search(r'\b(?:perhaps|maybe|possibly|unclear|rather than|not|or)\b',clause,re.I):continue
        seen=angles(clause)
        if len(seen)!=1 or wanted==seen:continue
        if head_only and not re.search(r"\b(?:head|face)\b",clause,re.I):continue
        if not head_only and re.search(r"\b(?:head|face) (?:is |turned |faces |shown |seen )",clause,re.I):
            continue
        def subjects(text):
            return {family for family in ANGLE_SUBJECTS if re.search(r'\b(?:'+family+r')\b',text,re.I)}
        required=subjects(requirement)
        observed=subjects(clause)
        if head_only:
            same_subject=(r'face|head' in required and r'face|head' in observed
                          and not (required-{r'face|head'} and observed-{r'face|head'} and
                                   required-{r'face|head'}!=observed-{r'face|head'}))
        else:
            required.discard(r'face|head');observed.discard(r'face|head')
            same_subject=len(required)==1 and required==observed
        if same_subject:
            return {'requested':next(iter(wanted)),'observed':next(iter(seen)),'visible_words':clause.strip()}
    return None


BODY_LEVEL=re.compile(r'\b(?:at|near|around|held at|level with) (?:the |her |his |their )?'
                      r'(chest|waist|hip|shoulder|chin|eye|knee)(?:[- ](?:level|height))?\b|'
                      r'(胸前|胸口高度|齐腰|腰部高度|与肩齐平|膝盖高度)',re.I)


def resolve_mixed_bands(url, ruled, model, item, landmark, reading):
    """Re-read named points separately if paired coordinates contradict the visual account."""
    reported={reading['item_band'],reading['landmark_band']}
    cited={int(v) for v in re.findall(r'\bband(?:\s+(?:labeled|labelled|number|numbered))?\s+#?(\d{1,2})\b',
                                      reading['trace'],re.I)}
    if not reading['visible'] or len(cited)<2 or reported==cited:return reading
    fields={'trace':{'type':'string'},'visible':{'type':'boolean'},
            'band':{'type':'integer','minimum':1,'maximum':20}}
    result={**reading,'paired':dict(reading),'single_point_checks':{}}
    for key,name in [('item_band',item),('landmark_band',landmark)]:
        question=('Locate only the center of '+name+' in this illustration. Identify its visible '
                  'physical landmarks and follow its height horizontally to the numbered margin. '
                  'Return the single corresponding band number. Mark invisible if the point cannot be located.')
        point=json.loads(ask(url,[ruled],question,{'type':'object','properties':fields,'required':list(fields)},tag=model))
        result['single_point_checks'][key]=point
        value=point['band']
        if not point['visible'] or not point['trace'].strip() or value not in cited:
            result['visible']=False
            break
        result[key]=value
    if result['visible']:
        result['trace']=' '.join(v['trace'] for v in result['single_point_checks'].values())
    return result


def check_body_level(url,overview,brief,tag):
    """Locate a specified object against a visible body landmark in the whole composition."""
    match=BODY_LEVEL.search(brief)
    if not match:return [],None
    anchor=match.group(1) or match.group(2)
    fields={'explicit':{'type':'boolean'},'object_or_point':{'type':'string'},'person':{'type':'string'}}
    plan=json.loads(ask(url,[],'Identify an explicit request that an object or part be positioned at '
        'the height of the '+anchor+'. Name the item and the reference person. Include the contact '
        'point if specified. Omit all desired positions and relative heights from the item name. '
        'Set explicit false for unrelated body descriptions. Brief: '+brief,
        {'type':'object','properties':fields,'required':list(fields)},tag=tag))
    if not plan['explicit'] or not plan['object_or_point'].strip():return [],None
    unit={'type':'integer','minimum':0,'maximum':1000}
    positions={'comparable':{'type':'boolean'},'item_x':unit,'item_y':unit,'body_x':unit,
               'level_top':unit,'level_bottom':unit,'shoulder_y':unit,'hip_y':unit,
               'trace':{'type':'string'}}
    instruction=('Look at this full illustration. Find the '+plan['object_or_point']+' and the '
                 +anchor+' of '+plan['person']+'. Locate the actual visible item or contact point '
                 'and separately locate the visible vertical extent of that anatomical landmark. '
                 'Return coordinates relative to the whole image from 0 to 1000. Determine the '
                 'shoulder and hip heights from the same person. Use comparable false if either '
                 'point cannot be identified or posture, crop or perspective prevents comparison. '
                 'Describe where the item actually appears without recommending a placement.')
    # If the detector marked the entire torso as one landmark, read the
    # actual candidate again on a labeled scale. The margins do not cover art.
    from PIL import ImageDraw,ImageFont
    ruled=None
    def measure_centers(model,row):
        nonlocal ruled
        if ruled is None:
            sample=overview.copy()
            sample.thumbnail((1000,1000),Image.Resampling.LANCZOS)
            gutter=42
            ruled=Image.new('RGB',(sample.width+2*gutter,sample.height),'white')
            ruled.paste(sample,(gutter,0))
            pen=ImageDraw.Draw(ruled)
            font=ImageFont.load_default(size=max(10,min(18,sample.height//32)))
            for band in range(20):
                y=round((band+.5)*sample.height/20)
                for margin in (gutter//2,ruled.width-gutter//2):
                    pen.text((margin,y),str(band+1),fill='black',anchor='mm',font=font)
                top=round(band*sample.height/20)
                pen.line((0,top,gutter-1,top),fill='#777777')
                pen.line((ruled.width-gutter,top,ruled.width-1,top),fill='#777777')
        band={'type':'integer','minimum':1,'maximum':20}
        fields={'visible':{'type':'boolean'},'item_band':band,'landmark_band':band,'trace':{'type':'string'}}
        check=json.loads(ask(url,[ruled],
            'The left and right margin number twenty horizontal bands from top to bottom. Read the '
            'band containing the center of '+plan['object_or_point']+' and independently the band '
            'containing the center of the '+anchor+' of '+plan['person']+'. Name both actual visible '
            'landmarks. Report their observed positions. Set visible false for an ambiguous identification.',
            {'type':'object','properties':fields,'required':list(fields)},tag=model))
        check=resolve_mixed_bands(url,ruled,model,plan['object_or_point'],anchor+' of '+plan['person'],check)
        if check['visible'] and check['trace'].strip() and all(1<=check[k]<=20 for k in ('item_band','landmark_band')):
            n=check['landmark_band']
            row=dict(row,item_y=round((check['item_band']-.5)*50),level_top=(n-1)*50,level_bottom=n*50,
                     trace=check['trace'])
        else:row=dict(row,comparable=False)
        return row,check
    notes=[]
    other='qwen3-vl:32b-instruct'
    for model in dict.fromkeys((tag,other)):
        if model!=tag and not model_identity(model):break
        row=json.loads(ask(url,[overview],instruction,{'type':'object','properties':positions,
                            'required':list(positions)},tag=model))
        note={'model':model,'location':row}
        notes.append(note)
        torso=row['hip_y']-row['shoulder_y']
        if row['comparable'] and torso>=60 and row['level_bottom']-row['level_top']>=.75*torso:
            note['original_location']=row
            row,scale_reading=measure_centers(model,row)
            note.update(location=row,scale_reading=scale_reading)
        if not row['comparable'] or torso<60 or row['level_top']>row['level_bottom'] or not row['trace'].strip():break
        margin=max(24,torso*.25)
        if row['level_top']-margin<=row['item_y']<=row['level_bottom']+margin:break
    evidence={'requirement':match.group(),'target':plan,'readings':notes}
    if len(notes)!=2:return [evidence],None
    a,b=[n['location'] for n in notes]
    def offset(v):
        if not v['comparable'] or not v['trace'].strip():return 0
        h=v['hip_y']-v['shoulder_y'];top,bottom=v['level_top'],v['level_bottom']
        if h<60 or top>bottom:return 0
        pad=max(24,h*.25)
        return 1 if v['item_y']>bottom+pad else -1 if v['item_y']<top-pad else 0
    if (not offset(a) or offset(a)!=offset(b) or abs(a['item_x']-b['item_x'])>90
            or abs(a['item_y']-b['item_y'])>90 or abs(a['level_top']-b['level_top'])>100):return [evidence],None
    xs=[a['item_x'],b['item_x'],a['body_x'],b['body_x']]
    ys=[a['item_y'],b['item_y'],a['level_top'],b['level_bottom']]
    fix='Position '+plan['object_or_point']+' '+match.group()+', preserving its visible attachment and a natural pose.'
    return [evidence],dict(decision='local',operation='redraw',target=plan['object_or_point'],fix=fix,
                           evidence=b['trace'],left=max(0,min(xs)-70),right=min(1000,max(xs)+70),
                           top=max(0,min(ys)-50),bottom=min(1000,max(ys)+50))


def identity_reference(quote):
    """Only use a reference labeled for the named species or breed being judged."""
    if re.search(r'\b(?:no|not|without|exclude|instead of)\b|不要|不含|禁止',quote,re.I):return None
    directory=Path(__file__).resolve().parents[2]/'assets/repair-guides'
    index=directory/'index.json'
    if not index.is_file():return None
    guides=[]
    for guide in json.loads(index.read_text()):
        for term in guide.get('identity_terms',[]):
            match=(re.search(r'\b'+re.escape(term)+r's?\b',quote,re.I) if term.isascii() else term in quote)
            if match and (directory/guide['file']).is_file():
                guides.append((directory/guide['file'],guide))
                break
    return guides[0] if len(guides)==1 else None


CONTACT_ACTION=re.compile(r'\b(?:turn(?:s|ing)?|press(?:es|ing)?|pinch(?:es|ing)?|'
                          r'grip(?:s|ping)?|touch(?:es|ing)?|grasps?|holding|holds?|'
                          r'tying|fasten(?:s|ing)?|plug(?:s|ging)?|steps? on|slid(?:es|ing)|insert(?:s|ing)?)\b|旋转|拧动|握住|捏住|按下|触碰|踏在|插入|滑入|抵住|贴在|推入',re.I)


def confirm_contact_gap(url,crop,overview,quote,tag,independent,*,endpoints=None):
    """Locate the nearest edges of a requested contact in the same native field."""
    if not CONTACT_ACTION.search(quote) or tag==independent:return None
    unit={'type':'integer','minimum':0,'maximum':1000}
    point={'type':'array','minItems':2,'maxItems':2,'items':unit}
    fields={'relation':{'enum':['separated','touching','occluded','uncertain']},
            'actor_edge':point,'target_edge':point,'both_visible':{'type':'boolean'},
            'nearest_relevant_target':{'type':'boolean'},'intervening_surface':{'type':'string'}}
    prompt=('First image: inspect this particular area of an illustration. The second image provides '
            'the surrounding scene. For the action '+repr(quote)+', locate the nearest actual edge of the '
            'acting part and the nearest actual edge of any appropriate target, in that order. Supply XY '
            'in coordinates 0..1000 on image 1. When several fingers or controls are available, use the '
            'closest applicable pair. Report separated only when you see a continuous strip of another '
            'surface or empty space between BOTH edges. Describe that strip. Touching, foreshortened '
            'overlap or hidden far fingers do not demonstrate a gap. If another relevant contact could '
            'be off-crop, or if the action does not require contact, select uncertain.')
    if endpoints:
        prompt+=' Locate the working end of '+endpoints[0]+' and the surface of '+endpoints[1]+'. The coordinates must describe this pair, even if the hand already grips the implement.'
    observations=[]
    for model in (tag,independent):
        read=json.loads(ask(url,[crop,overview],prompt,{'type':'object','properties':fields,'required':list(fields)},tag=model))
        observations.append(read)
        if (read['relation']!='separated' or not read['both_visible'] or not read['nearest_relevant_target']
                or not read['intervening_surface'].strip()):break
        points=[read[key] for key in ('actor_edge','target_edge')]
        if any(len(pt)!=2 or any(type(v) is not int or not 0<v<1000 for v in pt) for pt in points):break
        if math.dist(*points)<20:break
    else:
        a,b=observations
        tolerance=min(72,max(18,.55*min(math.dist(v['actor_edge'],v['target_edge']) for v in observations)))
        if all(math.dist(a[key],b[key])<=tolerance for key in ('actor_edge','target_edge')):
            return dict(confirmed=True,observations=observations,models=[tag,independent])
    return dict(confirmed=False,observations=observations,models=[tag,independent][:len(observations)])


def check_working_contact(url, image, brief, tag, independent):
    """Before a local repair, establish whether its complete action fits the current staging."""
    clauses=[s.strip() for s in re.split(r'[.!?。！？;；]',brief) if CONTACT_ACTION.search(s) and
             re.search(r'\b(?:against|into|onto)\b|(?:插入|按在|抵住|贴在|推入)',s,re.I)]
    if not clauses or not model_identity(independent):return None
    props={'required':{'type':'boolean'},'acting_item':{'type':'string'},'receiving_surface':{'type':'string'}}
    quote=clauses[0]
    plan=json.loads(ask(url,[],'Does the following sentence explicitly require a depicted part or implement '
        'to make physical contact with a separate surface? If so, quote the names of the working item '
        'and the receiving surface verbatim. For a held tool choose the tool and its working surface. '
        'Ignore figurative, future, intended, interrupted, and negated actions. Sentence: '+quote,
        {'type':'object','properties':props,'required':list(props)},tag=tag))
    names=[plan[k].strip() for k in ('acting_item','receiving_surface')]
    if (not plan['required'] or not all(n and n.casefold() in quote.casefold() for n in names)
            or names[0].casefold()==names[1].casefold()):return None
    full=image.copy();full.thumbnail((1120,1120),Image.Resampling.LANCZOS)
    overview=image.copy();overview.thumbnail((560,560),Image.Resampling.LANCZOS)
    result=confirm_contact_gap(url,full,overview,quote,tag,independent,endpoints=names)
    if not result:return None
    result.update(requirement=quote,items=names)
    if result['confirmed']:
        positions=[p[k] for p in result['observations'] for k in ('actor_edge','target_edge')]
        xs,ys=zip(*positions)
        result['box']=[max(0,min(xs)-35),max(0,min(ys)-35),min(1000,max(xs)+35),min(1000,max(ys)+35)]
        result['minimum_gap']=min(math.dist(p['actor_edge'],p['target_edge']) for p in result['observations'])
    return result


def subject_witness(url, image, overview, brief, tag):
    position={key:{'type':'integer','minimum':0,'maximum':1000} for key in ('left','top','right','bottom')}
    properties={'quote':{'type':'string'},'inspect':{'type':'string'},**position}
    count=4 if VIEWPOINT.search(brief) else 3
    pose=(' Include the explicitly requested camera-relative view as one check. Quote the subject and viewpoint '
          'together and locate its entire head and body. The direction of its gaze and the angle visible '
          'to the viewer are separate facts.' if count==4 else '')
    schema={'type':'object','properties':{'checks':{'type':'array','maxItems':count,'items':{
        'type':'object','properties':properties,'required':list(properties)}}},'required':['checks']}
    plan=json.loads(ask(url,[overview],
        f'Select at most {count} explicit, visible promises that are easy to draw incorrectly while still '
        'showing the right generic subject. Prioritize a specified physical action or contact, a named '
        'species, breed or identifying pattern, then distinctive anatomy or equipment. Omit generic '
        'beauty/style. Select a SHORT VERBATIM phrase naming a single feature; separate clauses about the same object. For each give a neutral observable question '
        'and locate the evidence in the supplied whole picture using 0..1000 coordinates. Include both '
        'endpoints of an interaction. Do not assume the depicted action or identity is correct.'+pose+' Brief: '
        +brief,schema,tag=tag))['checks']
    notes=[]
    for item in plan[:count]:
        quote=item['quote'].strip()
        if len(quote)<2 or quote.casefold() not in brief.casefold():continue
        l,t,r,b=(item[k] for k in position)
        if not (0<=l<r<=1000 and 0<=t<b<=1000):continue
        mx=max(32,(200-(r-l))//2,round((r-l)*.22));my=max(32,(180-(b-t))//2,round((b-t)*.22))
        box=(max(0,l-mx),max(0,t-my),min(1000,r+mx),min(1000,b+my))
        bounds=tuple(round(v*n/1000) for v,n in zip(box,image.size*2))
        crop=image.crop(bounds);crop.thumbnail((1024,1024),Image.Resampling.LANCZOS)
        notes.append({'quote':quote,'question':item['inspect'],'box':[l,t,r,b],'crop_extent':list(box),'crop':crop})
    if not notes:return check_body_level(url,overview,brief,tag)
    fields={'reading':{'type':'string'},'decision':{'enum':['supported','visible_contradiction','indeterminate']},
            'fix':{'type':'string'}}
    names=['crop_'+str(i+1) for i in range(len(notes))]
    checks={'type':'object','properties':{n:{'type':'object','properties':fields,'required':list(fields)} for n in names},'required':names}
    questions=' '.join(f"{k}: {n['quote']!r}. Examine: {n['question']}." for k,n in zip(names,notes))
    crops=[n.pop('crop') for n in notes]
    readings=json.loads(ask(url,[*crops,overview],
        'Inspect these native crops in order, then the entire picture. For each, first name a visible '
        'landmark of the ACTUAL drawing. For a tool or contact, name both endpoints and whether they '
        'line up. For a specific identity, describe the visible shape, markings or structure that '
        'distinguishes it from a similar generic object. Compare that evidence with its exact quoted '
        'requirement. Mark visible_contradiction only for a concrete mismatch visible in these pixels. '
        'If the crop misses the site, occlusion prevents a determination, or the limited palette makes '
        'the feature unreadable, use indeterminate. Supply a localized correction only for a mismatch. '
        +questions,checks,tag=tag))
    issue=None
    for n,k,crop in zip(notes,names,crops):
        n['reading']=readings[k]
        vote=n['reading']
        viewpoint=bool(VIEWPOINT.search(n['quote']))
        guide=identity_reference(n['quote'])
        if issue or (vote['decision']=='supported' and not viewpoint and not guide):
            continue
        # The overview reader can reverse an instrument's direction or mistake a rail
        # for its supporting post. Independently trace the same requirement in context.
        alternate='qwen3-vl:32b-instruct'
        reviewer=alternate if tag!=alternate and model_identity(alternate) else tag
        proof_fields={'actual_landmarks':{'type':'string'},'finding':{'enum':['match','contradiction','unreadable']},
                      'repair_instruction':{'type':'string'}}
        examples=[overview,crop]
        reference_instruction=''
        if guide:
            filename,description=guide
            with Image.open(filename) as source:examples.append(source.convert('RGB'))
            n['identity_reference']={'file':filename.name,'source':description['source']}
            reference_instruction=('The third image is a documented reference for the named subject. '
                'Use it only to identify characteristic anatomy or coat, allowing the pose, stylization, '
                'age, palette and any expressly requested variation in the artwork. Never count the reference '
                'as an extra subject. Relevant features: '+description['construction']+' ')
        evidence=json.loads(ask(url,examples,
            'The first image is a complete illustration and the second is an enlargement from it. '+reference_instruction+
            'First trace the actual shape, alignment, position and connection using at least two '
            'identifiable visible landmarks. Then compare with this requirement: '+repr(n['quote'])+
            ('. For a viewpoint, determine the side, front or back visible to the viewer from the silhouette '
             'and overlapping features. Distinguish gaze from body orientation.' if viewpoint else '')+
            '. Compare only landmarks you can see. Where a named species or variety is required, '
            'look for its distinctive physical markings. When a specific action is required, '
            'trace the visible endpoints of that action. A generic animal or the presence of a '
            'tool alone does not establish the specified identity or use. Select contradiction '
            'only for a demonstrable inconsistency. Give a direct imperative repair instruction '
            'only in that case.',{'type':'object','properties':proof_fields,'required':list(proof_fields)},tag=reviewer))
        n['independent_reading']={'model':reviewer,**evidence}
        marks=evidence['actual_landmarks'].strip()
        fix=evidence['repair_instruction'].strip()
        if evidence['finding']=='contradiction' and len(marks.split())>=5 and len(fix.split())>=3:
            # Read the crop without a target description. An instrument being present
            # or a generic owl being recognizable can bias a brief-conditioned observer.
            neutral=ask(url,[crop],
                'Describe only what this crop visibly depicts. Identify the visible part, silhouette, '
                'species or variety when distinguishable, facial outline and markings, material or '
                'pattern when relevant. If objects approach one another, identify exactly where one '
                'endpoint falls relative to the neighboring feature. State which surfaces face the viewer and '
                'which silhouette landmarks establish the view. Mention any visible gap or '
                'overlap. Do not grade the picture, infer intent or recommend changes. Be specific '
                'and use fewer than 65 words.',tag=reviewer).strip()
            n['blind_crop_reading']=neutral
            text_fields={'relation':{'enum':['meets_requirement','explicit_incompatibility','inconclusive']},
                         'incompatible_words':{'type':'string'}}
            reading=json.loads(ask(url,[],
                'Read the independent crop description below as written. Compare it to the required feature. '
                'Only when the observation describes an incompatible physical fact, quote the EXACT '
                'short excerpt that establishes the inconsistency. If it describes the requested '
                'relationship in other words, select meets_requirement. An omitted detail is '
                'inconclusive. Use the quoted physical shape, pattern, or point of contact from the '
                'independent description. Additional context may identify a diagnostic landmark but '
                'cannot substitute for its visibility in the independent description. Requirement: '+
                repr(n['quote'])+'. Independent crop description: '+neutral+'. Additional context: '+marks+' '+fix,
                {'type':'object','properties':text_fields,'required':list(text_fields)},tag=reviewer))
            n['evidence_comparison']=reading
            words=reading['incompatible_words'].strip()
            if reading['relation']!='explicit_incompatibility' or len(words)<5 or words.casefold() not in neutral.casefold():
                from animal_identity import named_animal_conflict
                conflict=named_animal_conflict(n['quote'],neutral)
                angle=(camera_conflict(n['quote'],neutral) if viewpoint and vote['decision']=='visible_contradiction' else None)
                if angle:
                    n['viewpoint_conflict']=angle
                elif conflict:
                    n['named_animal_conflict']=conflict
                    fix='Depict the requested '+conflict['requested']+' in this position. '+fix
                elif (vote['decision']=='visible_contradiction' and reading['relation']=='inconclusive'
                      and CONTACT_ACTION.search(n['quote'])):
                    gap=confirm_contact_gap(url,crop,overview,n['quote'],tag,reviewer)
                    n['contact_gap']=gap
                    if not gap or not gap['confirmed']:continue
                    # Include both verified ends in the correction instead of repainting a fingertip alone.
                    x0,y0,x1,y1=n['crop_extent']
                    locations=[o[key] for o in gap['observations'] for key in ('actor_edge','target_edge')]
                    xs=[round(x0+x*(x1-x0)/1000) for x,_ in locations]
                    ys=[round(y0+y*(y1-y0)/1000) for _,y in locations]
                    l,t,r,b=n['box']
                    n['box']=[max(0,min(l,*xs)-5),max(0,min(t,*ys)-5),
                              min(1000,max(r,*xs)+5),min(1000,max(b,*ys)+5)]
                else:continue
            issue={'decision':'local','operation':'redraw','target':n['quote'],'fix':fix,
                   'evidence':f"The requested detail {n['quote']!r}: {neutral}",**dict(zip(position,n['box']))}
    if issue is None:
        placement,issue=check_body_level(url,overview,brief,tag)
        notes.extend(placement)
    return notes,issue
