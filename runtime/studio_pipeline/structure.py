"""Check figure continuity and supporting surfaces at a larger viewing scale."""
import json
import re

from PIL import Image, ImageDraw, ImageOps
from evidence import ask, available, backend

SCHEMA = {
    'type':'object','properties':{
        'visible_chain':{'type':'string'},
        'framing':{'type':'string','enum':['human_hips_or_knees_visible','upper_body_only','no_person']},

        'support_or_occlusion':{'type':'string'},
        'verdict':{'type':'string','enum':['coherent','defective','unclear']},
    },'required':['visible_chain','framing','support_or_occlusion','verdict'],
}

INSTRUCTION = (
    'Inspect body and object continuity in this illustration. Left is the whole picture, '
    'right is an enlargement of its supporting structures. Both are the SAME image. '
    'Describe in under 30 words exactly what joins what. For each visible figure follow the torso '
    'through hips, knees, lower legs, ankles, footwear and support. For animals follow body to paws and tail. '
    'Where cloth overlaps a limb, identify the occluding edge and what emerges. For objects inspect their structural connections. '
    'Separate real shapes from reflections by looking for a ground or water boundary. '
    'Give a second short observation about support, framing or occlusion. Never guess at hidden feet. '
    'Normal occlusion or an intentional portrait can be coherent. An unexplained change of material, duplicated '
    'body, or detached part is defective. If a silhouette cannot be traced to either a plausible occluder, '
    'frame boundary or support, return unclear. Judge visible construction independently of style and sharpness.'
)


CONTACT_PROMPT = (
    'Count the distinct HUMAN shoes or identifiable human bare feet actually visible in this crop. '
    'A shoe needs a recognizable heel or toe. Exclude animal paws, reflections and guesses. '
    'Reply with a single line: 0 | what blocks the view, 1 | location and appearance of the foot, '
    'or 2 | locations and appearance of both feet. If a bare knee is clearly resting on a solid surface '
    'and no feet are visible, answer KNEE | the visible contact. If unsure answer UNKNOWN | reason.'
)


def context_boxes(im,detected):
    people=[r['box'] for r in detected if r['name']=='person' and len(r.get('box',[]))==4]
    people.sort(key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)
    if not people:
        return [(0,round(im.height*.42),im.width,im.height)]
    boxes=[]
    for left,top,right,bottom in people[:3]:
        w, h=right-left,bottom-top
        box=(max(0,left-.22*w),max(0,top+.34*h),min(1000,right+.22*w),min(1000,bottom+.13*h))
        rect=tuple(round(v*n/1000) for v,n in zip(box,im.size*2))
        if rect[2]>rect[0] and rect[3]>rect[1]:boxes.append(rect)
    return boxes or [(0,0,im.width,im.height)]


def inspect_structure(im,detected,proof,observed_counts=None):
    boxes=context_boxes(im,detected)
    width,height=620,660
    sheet=Image.new('RGB',(width*(1+len(boxes)),height),'#f4f2eb')
    draw=ImageDraw.Draw(sheet)
    for i,view in enumerate([im]+[im.crop(box) for box in boxes]):
        tile=ImageOps.contain(view,(width-12,height-34),Image.Resampling.LANCZOS)
        sheet.paste(tile,(i*width+(width-tile.width)//2,27+(height-34-tile.height)//2))
        draw.text((i*width+8,5),'Complete scene' if i==0 else 'Figure / support detail',fill='black')
    sheet.save(proof,quality=95)
    if not available():return {'verdict':'unclear','reason':'本地结构审图模型未就绪'}
    try:
        with backend() as url:
            result=json.loads(ask(url,[sheet],INSTRUCTION,SCHEMA))
            chain=result.get('visible_chain','').lower()
            located_person=any(r['name']=='person' for r in detected)
            observed_people=(observed_counts or {}).get('person',0)
            model_framing=result['framing']
            # Use independently observed people, never the number requested by the user.
            # In fallback mode, an explicit anatomical description is stronger evidence than
            # a bare noun which may occur in a sentence such as "no human figures present".
            anatomical_description=bool(re.search(
                r"\b(?:girl|boy|woman|man|child|person|figure)(?:'s|’s)?[ :]+(?:torso|hips|knees)\b",chain))
            human=(located_person or observed_people>0 or model_framing!='no_person' or anatomical_description)
            lower_visible=(result.get('framing')=='human_hips_or_knees_visible'
                           or (human and bool(re.search(r'\b(hips?|skirt|knees?|ankles?|shoes?|trousers)\b',chain))))
            result['model_framing']=result['framing']
            if lower_visible:result['framing']='human_hips_or_knees_visible'
            result['lower_body_check']=lower_visible
            if lower_visible or human:
                observations=[]
                for left,top,right,bottom in boxes:
                    crop=im.crop((left,round(top+.42*(bottom-top)),right,bottom))
                    crop.thumbnail((800,720),Image.Resampling.LANCZOS)
                    answer=ask(url,[crop],CONTACT_PROMPT).strip()
                    match=re.match(r'(0|1|2|KNEE|UNKNOWN)\s*\|\s*(.+)',answer,re.I)
                    observed=match.group(1).upper() if match else 'UNKNOWN'
                    observations.append({'visible_support':observed,'observation':answer[:250]})
                result['support_details']=observations
                if result['verdict']=='coherent' and lower_visible and any(o['visible_support'] in ('0','UNKNOWN') for o in observations):
                    result['verdict']='unclear'
                    result['support_or_occlusion']='下半身可见，局部检查无法辨认足部或膝部的承托边界。'
            if observed_people>0 and not located_person and model_framing=='no_person':
                result['person_evidence_conflict']=True
                if result['verdict']=='coherent':
                    result['verdict']='unclear'
                    result['support_or_occlusion']='独立读图见到人物，但检测器未定位到人物，结构审查也未确认人物画幅，需目检。'
            return result
    except (OSError,RuntimeError,KeyError,ValueError) as exc:
        return {'verdict':'unclear','reason':str(exc)[:200]}
