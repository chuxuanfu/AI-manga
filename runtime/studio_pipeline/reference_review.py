"""Compare finish at matched viewing sizes without treating the reference as the requested scene."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageOps
from evidence import ask, backend, available, TAG
from illustration import REFERENCE

BENCHMARK=Path(__file__).resolve().parents[1]/'ComfyUI/input'/REFERENCE
AXES=('form_and_contact','material_construction','lighting','depth_and_finish')
GRADE={'type':'object','properties':{'level':{'type':'string','enum':['left','right','tie','unclear']},'evidence':{'type':'string'}},'required':['level','evidence']}
SCHEMA={'type':'object','properties':{key:GRADE for key in AXES},'required':list(AXES)}


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


def resolve_workmanship(url, im, regions, instruction, ambiguous):
    from construction_review import LARGE
    if not available(LARGE):
        return {'decision':'review','reason':'缺少独立画质复审模型'}
    # Preserve detail at a larger scale on individual sheets. Side-by-side
    # thumbnails can conceal thin lines and small points of contact.
    def proof(image, boxes):
        sheet=Image.new('RGB',(768,1144),'#f4f2eb')
        full=ImageOps.contain(image,(768,740),Image.Resampling.LANCZOS)
        sheet.paste(full,((768-full.width)//2,(740-full.height)//2))
        for i,(_,box) in enumerate(boxes[:2]):
            pixels=tuple(round(n*k) for n,k in zip(box,image.size*2))
            if pixels[2]<=pixels[0] or pixels[3]<=pixels[1]:continue
            patch=ImageOps.contain(image.crop(pixels),(380,396),Image.Resampling.LANCZOS)
            sheet.paste(patch,(i*384+(384-patch.width)//2,746+(396-patch.height)//2))
        return sheet
    with Image.open(BENCHMARK) as ref:
        first=proof(ref.convert('RGB'),REF_BOXES)
    last=proof(im,detail_boxes(regions))
    schema={'type':'object','properties':{axis:GRADE for axis in ambiguous},'required':list(ambiguous)}
    question=(instruction+' Images are provided as two separate sheets, each with its own enlarged details. '
              'Use left for the FIRST sheet and right for the SECOND. Trace visible evidence in both. '
              'Compare only these aspects: '+', '.join(ambiguous)+'.')
    readings=[json.loads(ask(url,pair,question,schema,tag=LARGE)) for pair in ([first,last],[last,first])]
    result=order_verdict(*readings,ambiguous)
    return dict(result,model=LARGE,orders=readings)


def order_verdict(forward,backward,axes):
    supported=(('right','left'),('tie','tie'),('tie','left'),('right','tie'))
    pairs={axis:(forward[axis]['level'],backward[axis]['level']) for axis in axes}
    valid={k for k in axes if all(str(v[k].get('evidence') or '').strip() for v in (forward,backward))}
    deficits=[k for k in axes if k in valid and pairs[k]==('left','right')]
    ambiguous=[k for k in axes if k not in valid or (pairs[k] not in supported and k not in deficits)]
    return dict(decision='below' if deficits else 'review' if ambiguous else 'candidate',
                consistent_deficits=deficits,unresolved_axes=ambiguous)


def compare(im,regions,path,brief='',*,final=False):
    from print_judge import MODEL
    tag=MODEL if final else TAG
    board=comparison(im,regions,path)
    if not available(tag):return {'decision':'review','reason':'参考图比较模型未就绪'}
    instruction=('Compare these two illustrations as peers. They may show identical art or different scenes. For each axis indicate which side demonstrates visibly stronger workmanship, or tie when both are comparable. '
                 'Use unclear only if the evidence is too small. Check each picture on its own visible subjects. Trace a joint, supporting contact or joined object; judge how the artist depicts the materials actually present, such as folded cloth, taut cord, timber or layered paper. Compare lighting within the intended medium and the separation of focal and distant shapes. A subject present on just one side does not confer an advantage. '
                 'Ignore personal preferences, differences in subject matter, relative zoom of detail crops and which painting appears first. '
                 'Give under twelve words of concrete evidence per axis. Use left, right, tie or unclear.')
    if brief:
        instruction+=' The new image has its own requested medium: '+brief+'. Evaluate construction and deliberate marks within each medium. Spare ink, flat cel shading, paper cutouts, monochrome or visible print dots may be fully resolved. Do not reward merely having more texture.'
    # Mask identifying headings and evaluate in both orders to detect position bias.
    neutral=board.copy()
    draw=ImageDraw.Draw(neutral)
    draw.rectangle((0,0,1151,24),fill='#f4f2eb')
    draw.rectangle((0,768,1151,797),fill='#f4f2eb')
    reverse=Image.new('RGB',neutral.size)
    reverse.paste(neutral.crop((576,0,1152,1104)),(0,0))
    reverse.paste(neutral.crop((0,0,576,1104)),(576,0))
    try:
        with backend(tag) as url:
            forward=json.loads(ask(url,[neutral],instruction,SCHEMA,tag=tag))
            backward=json.loads(ask(url,[reverse],instruction,SCHEMA,tag=tag))
            result=order_verdict(forward,backward,AXES)
            if result['decision']=='review':
                independent=resolve_workmanship(url,im,regions,instruction,result['unresolved_axes'])
                result.update(decision=independent['decision'],first_unresolved_axes=result['unresolved_axes'],
                              unresolved_axes=independent.get('unresolved_axes',result['unresolved_axes']),
                              independent_review=independent)
                if independent.get('consistent_deficits'):
                    result['consistent_deficits']=independent['consistent_deficits']
        return dict(result,assessment_model=tag,reference_on_left=forward,reference_on_right=backward)
    except (RuntimeError,OSError,ValueError,KeyError) as exc:
        return {'decision':'review','reason':str(exc)[:180]}
