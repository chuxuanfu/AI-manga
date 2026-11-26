"""Inspect native evidence for the most specific visible promises in the brief."""
import json

from evidence import ask, model_identity
from PIL import Image


def subject_witness(url, image, overview, brief, tag):
    position={key:{'type':'integer','minimum':0,'maximum':1000} for key in ('left','top','right','bottom')}
    properties={'quote':{'type':'string'},'inspect':{'type':'string'},**position}
    schema={'type':'object','properties':{'checks':{'type':'array','maxItems':3,'items':{
        'type':'object','properties':properties,'required':list(properties)}}},'required':['checks']}
    plan=json.loads(ask(url,[overview],
        'Select at most three explicit, visible promises that are easy to draw incorrectly while still '
        'showing the right generic subject. Prioritize a specified physical action or contact, a named '
        'species, breed or identifying pattern, then distinctive anatomy or equipment. Omit generic '
        'beauty/style. Select a SHORT VERBATIM phrase naming a single feature; separate clauses about the same object. For each give a neutral observable question '
        'and locate the evidence in the supplied whole picture using 0..1000 coordinates. Include both '
        'endpoints of an interaction. Do not assume the depicted action or identity is correct. Brief: '
        +brief,schema,tag=tag))['checks']
    notes=[]
    for item in plan[:3]:
        quote=item['quote'].strip()
        if len(quote)<2 or quote.casefold() not in brief.casefold():continue
        l,t,r,b=(item[k] for k in position)
        if not (0<=l<r<=1000 and 0<=t<b<=1000):continue
        mx=max(32,(200-(r-l))//2,round((r-l)*.22));my=max(32,(180-(b-t))//2,round((b-t)*.22))
        box=(max(0,l-mx),max(0,t-my),min(1000,r+mx),min(1000,b+my))
        bounds=tuple(round(v*n/1000) for v,n in zip(box,image.size*2))
        crop=image.crop(bounds);crop.thumbnail((1024,1024),Image.Resampling.LANCZOS)
        notes.append({'quote':quote,'question':item['inspect'],'box':[l,t,r,b],'crop':crop})
    if not notes:return [],None
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
        if issue or vote['decision']=='supported':
            continue
        # The overview reader can reverse an instrument's direction or mistake a rail
        # for its supporting post. Independently trace the same requirement in context.
        alternate='qwen3-vl:32b-instruct'
        reviewer=alternate if tag!=alternate and model_identity(alternate) else tag
        proof_fields={'actual_landmarks':{'type':'string'},'finding':{'enum':['match','contradiction','unreadable']},
                      'repair_instruction':{'type':'string'}}
        evidence=json.loads(ask(url,[overview,crop],
            'The first image is a complete illustration and the second is an enlargement from it. '
            'First trace the actual shape, alignment, position and connection using at least two '
            'identifiable visible landmarks. Then compare with this requirement: '+repr(n['quote'])+
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
                'endpoint falls relative to the neighboring feature. Mention any visible gap or '
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
                continue
            issue={'decision':'local','operation':'redraw','target':n['quote'],'fix':fix,
                   'evidence':f"The requested detail {n['quote']!r}: {neutral}",**dict(zip(position,n['box']))}
    return notes,issue
