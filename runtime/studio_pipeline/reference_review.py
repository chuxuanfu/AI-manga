"""Compare finish at matched viewing sizes without treating the reference as the requested scene."""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageOps
from evidence import ask, backend, available, TAG
from illustration import REFERENCE

BENCHMARK=Path(__file__).resolve().parents[1]/'ComfyUI/input'/REFERENCE
AXES=('form_and_contact','material_construction','lighting','depth_and_finish')
GRADE={'type':'object','properties':{'level':{'type':'string','enum':['left','right','tie','unclear']},'evidence':{'type':'string'}},'required':['level','evidence']}
SIDE_GRADE={'type':'object','properties':{**GRADE['properties'],'left_detail':{'type':'string'},
            'right_detail':{'type':'string'}},'required':['level','evidence','left_detail','right_detail']}
SIDE_INSTRUCTION=(' Inspect each picture separately before grading. In left_detail and right_detail, name '
                  'one specific visible object or area and describe its physical evidence for this axis on that '
                  'side alone. Keep each to eight words. Describe only subjects you can see. Mark unclear if a '
                  'detail cannot be located. In evidence give a short reason for the relative grade.')


def grade_schema(axes,sided):
    return {'type':'object','properties':{axis:SIDE_GRADE if sided else GRADE for axis in axes},'required':list(axes)}



REF_BOXES=(('hands / sleeve',(.295,.535,.578,.728)),('knit / folds',(.18,.65,.445,.842)))


def detail_boxes(regions):
    boxes=[]
    # Show two different subjects whenever possible, rather than two detections of the same hand.
    families=(('corrected','hand','paw','face'),('fabric','apron','clothing','cardigan','coat','skirt','fur','ceramic','wood','animal'))
    for family in families:
        selected=None
        for token in family:
            for row in regions.values():
                coords=tuple(v/1000 for v in row['box'])
                if token in row['label'].lower() and coords not in [bounds for _,bounds in boxes]:
                    selected=(row['label'],coords)
                    break
            if selected:break
        if selected:boxes.append(selected)
    if len(boxes)<2:boxes.append(('middle distance',(.18,.28,.75,.64)))
    if len(boxes)<2:boxes.append(('foreground',(.15,.59,.77,.90)))
    return boxes


def comparison(im,regions,target):
    with Image.open(BENCHMARK) as img:
        ref=img.convert('RGB')
    page=Image.new('RGB',(1152,1104),'#f4f2eb');d=ImageDraw.Draw(page)
    def panel(image,x,title,boxes):
        d.text((x+8,6),title,fill='black')
        copy=ImageOps.contain(image,(558,730),Image.Resampling.LANCZOS)
        page.paste(copy,(x+(576-copy.width)//2,27+(730-copy.height)//2))
        for i,(name,box) in enumerate(boxes[:2]):
            l,t,r,b=box
            crop=image.crop((round(l*image.width),round(t*image.height),round(r*image.width),round(b*image.height)))
            tile=ImageOps.contain(crop,(274,286),Image.Resampling.LANCZOS)
            sx=x+i*286;page.paste(tile,(sx+(286-tile.width)//2,798+(286-tile.height)//2))
            d.text((sx+5,775),str(name)[:40],fill='black')
    panel(ref,0,'REFERENCE',REF_BOXES)
    boxes=detail_boxes(regions)
    panel(im,576,'NEW ILLUSTRATION',boxes)
    page.save(target,quality=93)
    return page


def artwork_sheet(image, boxes):
    """Give full views equal pixel area and keep both supporting details at readable scale."""
    sheet=Image.new('RGB',(960,960),'#f4f2eb')
    if image.width>=image.height:
        viewport=(0,0,960,636)
        details=((0,644,476,316),(484,644,476,316))
    else:
        viewport=(0,0,636,960)
        details=((644,0,316,476),(644,484,316,476))

    def place(picture,slot,area):
        x,y,width,height=slot
        scale=min(width/picture.width,height/picture.height,math.sqrt(area/(picture.width*picture.height)))
        size=(max(1,round(picture.width*scale)),max(1,round(picture.height*scale)))
        fitted=picture.resize(size,Image.Resampling.LANCZOS)
        sheet.paste(fitted,(x+(width-fitted.width)//2,y+(height-fitted.height)//2))

    place(image,viewport,400000)
    for (_,box),slot in zip(boxes,details):
        pixels=tuple(round(value*extent) for value,extent in zip(box,image.size*2))
        if pixels[2]>pixels[0] and pixels[3]>pixels[1]:
            place(image.crop(pixels),slot,105000)
    return sheet


def artwork_pair(image,regions):
    with Image.open(BENCHMARK) as reference:
        return [artwork_sheet(reference.convert('RGB'),REF_BOXES),artwork_sheet(image,detail_boxes(regions))]


PAIR_INSTRUCTION=(' You receive two sheets. Left means the FIRST sheet, right the SECOND. Each has a full picture '
                  'and two enlarged details. Compare the execution of the materials depicted. A paper relief can show '
                  'depth through cut-edge shadows, a woodcut through carved contours and value, and a painting through '
                  'local color and illumination. Identify a visible weakness in its own medium when rating one side '
                  'lower. A greater number of lights, objects, textures or colors alone does not establish better execution.')


def resolve_workmanship(url, im, regions, instruction, ambiguous, *, sided=False):
    from construction_review import LARGE
    if not available(LARGE):
        return {'decision':'review','reason':'缺少独立画质复审模型'}
    first,last=artwork_pair(im,regions)
    schema=grade_schema(ambiguous,sided)
    question=(instruction+('' if PAIR_INSTRUCTION in instruction else PAIR_INSTRUCTION)+' Trace visible evidence in both. '
              'Compare only these aspects: '+', '.join(ambiguous)+'.')
    readings=[json.loads(ask(url,pair,question,schema,tag=LARGE)) for pair in ([first,last],[last,first])]
    result=order_verdict(*readings,ambiguous,sided=sided)
    return dict(result,model=LARGE,orders=readings)


def order_verdict(forward,backward,axes,*,sided=False):
    supported=(('right','left'),('tie','tie'),('tie','left'),('right','tie'))
    pairs={axis:(forward[axis]['level'],backward[axis]['level']) for axis in axes}
    valid={k for k in axes if all(str(v[k].get('evidence') or '').strip() for v in (forward,backward))}
    if sided:
        def side_evidence(grade):
            marks=[str(grade.get(field) or '').strip().casefold() for field in ('left_detail','right_detail')]
            return all(len(mark.split())>=3 and not mark.startswith(('both ','both:','both pictures','both sides')) for mark in marks)
        valid={axis for axis in valid if all(side_evidence(row[axis]) for row in (forward,backward))}
    deficits=[k for k in axes if k in valid and pairs[k]==('left','right')]
    ambiguous=[k for k in axes if k not in valid or (pairs[k] not in supported and k not in deficits)]
    return dict(decision='below' if deficits else 'review' if ambiguous else 'candidate',
                consistent_deficits=deficits,unresolved_axes=ambiguous)


FINISH_EDITS={
    'form_and_contact':'Clarify silhouettes, joins, weight and the visible edge where each focal part meets its support.',
    'material_construction':'Give the focal surfaces identifiable edges, folds or joints using the marks of the requested medium.',
    'lighting':'Establish one legible light direction and use contact shadows and selective highlights consistent with the chosen medium.',
    'depth_and_finish':'Separate foreground, middle distance and far shapes by overlap, scale and value. Reserve the clearest detail for the focus.',
}


def redraw_direction(assessment,brief,*,address=None,tag=TAG,medium=None):
    """Only the candidate's visible marks and the author brief enter the next drawing prompt."""
    if assessment.get('decision')!='below':return ''
    axes=[axis for axis in AXES if axis in assessment.get('consistent_deficits',[])]
    if not axes:return ''
    general=' '.join(FINISH_EDITS[axis] for axis in axes)
    if address is None:return general
    observations=[]
    for axis in axes:
        for side,field in (('reference_on_left','right_detail'),('reference_on_right','left_detail')):
            detail=assessment.get(side,{}).get(axis,{}).get(field)
            if detail:observations.append(detail)
    schema={'type':'object','properties':{'edit':{'type':'string'}},'required':['edit']}
    question=('Write one positive drawing instruction under 95 words for a local illustration model. '
              'Preserve the exact composition, subjects, counts, palette, design and physical medium the author requested. '
              'Name concrete changes to existing parts to improve the following issues. A simple medium can show '
              'craft through deliberate joins, cut layers or shadows without imitating a different style. '
              'Include only objects from the author brief or observations of this same candidate. '
              'Author brief: '+brief+' Required improvements: '+general+' Observed candidate features: '+' '.join(observations))
    try:
        from illustration import medium_study
        sample=medium_study(medium)
        images=[]
        if sample:
            with Image.open(sample) as study:
                images=[study.convert('RGB')]
            question+=(' The grayscale studies show handling in this medium. Use their edges, layers and value '
                       'relationships to suggest specific techniques for the subjects named in the brief. '
                       'Preserve the requested colors, scene and time of day.')
        answer=json.loads(ask(address,images,question,schema,tag=tag))['edit'].strip()
    except (OSError,RuntimeError,ValueError,KeyError,AttributeError):
        return general
    return answer if 8<=len(answer.split())<=120 else general


def final_repaint_direction(assessment):
    """Only a confirmed deficit on the final master warrants repainting a whole scene."""
    from print_judge import MODEL
    if (assessment.get('decision')!='below' or assessment.get('assessment_model')!=MODEL
            or assessment.get('separate_material_evidence') is not True):
        return ''
    return assessment.get('redraw_direction') or redraw_direction(assessment,'')


def last_finish_direction(record):
    for row in reversed(record.get('reviews',[])):
        assessment=row.get('review',{}).get('参考比较') or {}
        if assessment:
            return (assessment.get('redraw_direction') or redraw_direction(assessment,'')) if assessment.get('decision')=='below' else ''
    return ''


def confirmed_final_comparison(review):
    from construction_review import LARGE
    from print_judge import MODEL
    if (review.get('decision')!='candidate' or review.get('assessment_model')!=MODEL
            or review.get('separate_material_evidence') is not True):
        return False
    try:
        first=order_verdict(review['reference_on_left'],review['reference_on_right'],AXES,sided=True)
        if first['decision']=='candidate':return True
        if first['decision']=='below':return False
        second=review.get('independent_review') or {}
        if second.get('model')!=LARGE or len(second.get('orders',[]))!=2:return False
        return order_verdict(*second['orders'],first['unresolved_axes'],sided=True)['decision']=='candidate'
    except (KeyError,TypeError,ValueError,AttributeError):
        return False


def compare(im,regions,path,brief='',*,final=False,medium=None):
    from print_judge import MODEL
    tag=MODEL if final else TAG
    board=comparison(im,regions,path)
    if not available(tag):return {'decision':'review','reason':'参考图比较模型未就绪'}
    instruction=('Compare these two illustrations as peers. They may show identical art or different scenes. For each axis indicate which side demonstrates visibly stronger workmanship, or tie when both are comparable. '
                 'Use unclear only if the evidence is too small. Check each picture on its own visible subjects. Trace a joint, supporting contact or joined object; judge how the artist depicts the materials actually present through their edges, joints or changes of surface. Compare lighting within the intended medium and the separation of focal and distant shapes. A subject present on just one side does not confer an advantage. '
                 'Ignore personal preferences, differences in subject matter, relative zoom of detail crops and which painting appears first. '
                 'Give under twelve words of concrete evidence per axis. Use left, right, tie or unclear.')
    if final:
        instruction+=SIDE_INSTRUCTION+' Judge each picture in the medium visibly used on that side.'
    elif brief:
        instruction+=' The new image has its own requested medium: '+brief+'. Evaluate construction and deliberate marks within each medium. Spare ink, flat cel shading, paper cutouts, monochrome or visible print dots may be fully resolved. Do not reward merely having more texture.'
    if final:
        pair=artwork_pair(im,regions)
        presentations=(pair,list(reversed(pair)))
        instruction+=PAIR_INSTRUCTION
    else:
        neutral=board.copy()
        draw=ImageDraw.Draw(neutral)
        draw.rectangle((0,0,1151,24),fill='#f4f2eb')
        draw.rectangle((0,768,1151,797),fill='#f4f2eb')
        reverse=Image.new('RGB',neutral.size)
        reverse.paste(neutral.crop((576,0,1152,1104)),(0,0))
        reverse.paste(neutral.crop((0,0,576,1104)),(576,0))
        presentations=([neutral],[reverse])
    try:
        with backend(tag) as url:
            schema=grade_schema(AXES,final)
            forward,backward=[json.loads(ask(url,pictures,instruction,schema,tag=tag)) for pictures in presentations]
            result=order_verdict(forward,backward,AXES,sided=final)
            if result['decision']=='review':
                independent=resolve_workmanship(url,im,regions,instruction,result['unresolved_axes'],sided=final)
                result.update(decision=independent['decision'],first_unresolved_axes=result['unresolved_axes'],
                              unresolved_axes=independent.get('unresolved_axes',result['unresolved_axes']),
                              independent_review=independent)
                if independent.get('consistent_deficits'):
                    result['consistent_deficits']=independent['consistent_deficits']
            result.update(assessment_model=tag,separate_material_evidence=final,reference_on_left=forward,reference_on_right=backward)
            if result['decision']=='below':
                result['redraw_direction']=redraw_direction(result,brief,address=url if brief else None,tag=tag,medium=medium)
        return result
    except (RuntimeError,OSError,ValueError,KeyError) as exc:
        return {'decision':'review','reason':str(exc)[:180]}
