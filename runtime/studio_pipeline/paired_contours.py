"""Cross-register independent traces when a judge disputes a pair of physical outlines."""
import json
import math
import re

from PIL import Image, ImageOps
from evidence import ask


def needs_pair_trace(reading):
    check=reading.get('consistency_check') or {}
    claim=check.get('tested_claim','')
    return (reading.get('condition')=='insufficient' and check.get('disputed_parts_in_view') is True
            and check.get('verdict')=='specific_visible_error'
            and re.match(r'(?:no (?:correction|repair|change) (?:is )?(?:needed|required)|none\b)',check.get('repair',''),re.I)
            and re.search(r'fus(?:ed|ion)|merg(?:ed|ing)|lack.{0,12}separation|joined together|粘连|融合',claim,re.I)
            and all(check.get(v,'').strip() for v in ('first_landmark','second_landmark','connection')))


def inspect_pair(url, image, box, claim, models):
    """Two anonymous readings must locate the same ends and trace their independent attachments."""
    if not box or not (0<=box[0]<box[2]<=1000 and 0<=box[1]<box[3]<=1000):return {'confirmed':False}
    left,top,right,bottom=[round(a*b/1000) for a,b in zip(box,image.size*2)]
    # Keep more than the tiny disputed tips in view so the reader can follow
    # each limb or component back to its own attachment on the parent.
    span=min(max(image.size),max(320,round(max(right-left,bottom-top)*3),round(min(image.size)*.43)))
    cx,cy=(left+right)//2,(top+bottom)//2
    w,h=min(image.width,span),min(image.height,span)
    x=max(0,min(image.width-w,cx-w//2));y=max(0,min(image.height-h,cy-h//2))
    rect=(x,y,x+w,y+h)
    field=image.crop(rect);field.thumbnail((1100,1100),Image.Resampling.LANCZOS)
    overview=image.copy();overview.thumbnail((900,900),Image.Resampling.LANCZOS)
    point={'type':'array','items':{'type':'integer','minimum':0,'maximum':1000},'minItems':2,'maxItems':2}
    props={'first_tip':point,'second_tip':point,'first_trace':{'type':'string'},'second_trace':{'type':'string'},
           'between':{'type':'string'},'shared_or_broken_interval':{'type':'boolean'},
           'separate_attachments_visible':{'type':'boolean'},'both_named_parts_visible':{'type':'boolean'}}
    schema={'type':'object','properties':props,'required':list(props)}
    prompt=('Image one is an unmarked close view, image two its whole scene in matching orientation. '
            'Independently examine the specific parts in this disputed statement: '+claim+
            '. In image one locate the centers of their two terminal ends (such as hoof or toe tips), '
            'each as [x,y] in units 0..1000 of image one. Identify the leftmost end first. Follow '
            'each distinct outline from its tip past its joint to where it joins its parent. Name the '
            'visible strip of background, separating edge or explicit occlusion between the two. '
            'Two distinct tips alone cannot establish two correct attachments; flag any unexplained '
            'shared contour, gap or malformed joint. Set both_named_parts_visible false if this field '
            'shows another part or the count cannot be traced. Report concrete visual observations.')
    result={'confirmed':False,'source_pixels':list(rect),'models':list(models),'orders':[]}
    positions=[]
    for mirrored,tag in enumerate(models):
        views=[field,overview] if not mirrored else [ImageOps.mirror(field),ImageOps.mirror(overview)]
        read=json.loads(ask(url,views,prompt,schema,tag=tag))
        result['orders'].append(read)
        if (read.get('both_named_parts_visible') is not True or read.get('separate_attachments_visible') is not True
                or read.get('shared_or_broken_interval') is not False
                or any(len(str(read.get(k) or '').split())<4 for k in ('first_trace','second_trace','between'))):
            return result
        tips=[read.get(k) for k in ('first_tip','second_tip')]
        if not all(isinstance(pt,list) and len(pt)==2 and all(type(v) is int and 0<=v<=1000 for v in pt) for pt in tips):
            return result
        if math.dist(*tips)<20:return result
        if mirrored:tips=[[1000-a,b] for a,b in tips]
        positions.append(sorted(tips))
    if len(positions)!=2 or len(set(models))!=2:return result
    # A location error as large as the inter-tip distance can match one end twice.
    tolerance=min(65,.45*min(math.dist(*row) for row in positions))
    result['tip_error']=[math.dist(a,b) for a,b in zip(*positions)]
    result['confirmed']=max(result['tip_error'])<=tolerance
    return result
