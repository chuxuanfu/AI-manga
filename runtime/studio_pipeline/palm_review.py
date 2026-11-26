"""Localize separate palms on CPU so the visual reviewers can inspect the right wrist."""
from functools import lru_cache
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from evidence import ask, model_identity

WEIGHTS = Path(__file__).resolve().parents[1] / 'models/object_detector/palm-locator.onnx'
ATTACHMENT = re.compile(r'connect|detach|separat|break|discontinu|fus(?:ed|ion)|merg(?:ed|e|ing)|attach|joins?|continu|断开|脱节|连接', re.I)
HAND = re.compile(r'\b(?:fingers?|thumbs?|wrists?|hands?|cuffs?)\b|手指|手掌|手腕|袖口', re.I)


def hand_attachment(claim):
    return bool(HAND.search(claim) and ATTACHMENT.search(claim))


@lru_cache(maxsize=1)
def detector():
    model = cv2.dnn.readNetFromONNX(str(WEIGHTS))
    # The 192px palm model has two anchors/cell on its 24px feature map and six on its 12px map.
    anchors = np.array([((x+.5)/grid, (y+.5)/grid) for grid, repeats in ((24,2),(12,6))
                        for y in range(grid) for x in range(grid) for _ in range(repeats)], dtype=np.float32)
    return model, anchors


def palms(image):
    model, anchors = detector()
    w, h = image.size
    ratio = 192 / max(w,h)
    nw, nh = max(1, round(w*ratio)), max(1, round(h*ratio))
    offset = np.array([(192-nw)//2,(192-nh)//2])
    canvas = np.zeros((192,192,3), dtype=np.uint8)
    x,y = offset
    canvas[y:y+nh,x:x+nw] = cv2.resize(np.asarray(image), (nw,nh))
    model.setInput(canvas[None].astype(np.float32)/255.)
    delta, logits = model.forward(model.getUnconnectedOutLayersNames())
    scores = 1/(1+np.exp(-np.clip(logits[0,:,0],-40,40)))
    selected = scores >= .5
    if not selected.any():
        return []
    size = delta[0,selected,2:4] / ratio
    center = (192*anchors[selected]+delta[0,selected,:2]-offset) / ratio
    boxes = np.column_stack((center-size/2,size))
    scores = scores[selected]
    kept = cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),.5,.3)
    return [tuple(boxes[i]) for i in np.asarray(kept).reshape(-1)]


def stable_palms(image):
    direct = palms(image)
    inverse = [(image.width-x-w,y,w,h) for x,y,w,h in palms(ImageOps.mirror(image))]
    matches = []
    for x,y,w,h in direct:
        best = None
        for i,(u,v,a,b) in enumerate(inverse):
            area = max(0,min(x+w,u+a)-max(x,u))*max(0,min(y+h,v+b)-max(y,v))
            overlap = area / max(1,w*h+a*b-area)
            if overlap>=.48 and (best is None or overlap>best[0]):
                best=(overlap,i)
        if best:
            inverse.pop(best[1])
            matches.append((x,y,w,h))
    return sorted(matches,key=lambda box:(box[1],box[0]))


def localized_palms(im, box):
    if not box or len(box)!=4 or not WEIGHTS.is_file():
        return None,[]
    l,t,r,b = [round(k*n/1000) for k,n in zip(box,im.size*2)]
    if r<=l or b<=t:
        return None,[]
    cx,cy=(l+r)/2,(t+b)/2
    rw,rh=max(r-l,.33*im.width),max(b-t,.26*im.height)
    field=(max(0,int(cx-rw/2)),max(0,int(cy-rh/2)),min(im.width,int(cx+rw/2)),min(im.height,int(cy+rh/2)))
    crop=im.crop(field).convert('RGB')
    matches=stable_palms(crop)
    boxes=[]
    for x,y,w,h in matches:
        # A wrist is only part of a hand. Include the adjoining cuff as well as the fingertips.
        span=2.3*max(w,h)
        mx,my=field[0]+x+w/2,field[1]+y+h/2
        left,top,right,bottom=mx-span,my-span,mx+span,my+span
        for u,v,a,bh in matches:
            dx,dy=field[0]+u+a/2-mx,field[1]+v+bh/2-my
            if max(abs(dx),abs(dy))<1.5*max(w,h,a,bh):
                continue
            if abs(dy)>abs(dx):
                if dy>0:bottom=min(bottom,my+dy/2)
                else:top=max(top,my+dy/2)
            elif dx>0:right=min(right,mx+dx/2)
            else:left=max(left,mx+dx/2)
        boxes.append((max(0,round(left)),max(0,round(top)),min(im.width,round(right)),min(im.height,round(bottom))))
    if not any(min(r,x2)>max(l,x1) and min(b,y2)>max(t,y1) for x1,y1,x2,y2 in boxes):
        return field,[]
    return field,boxes


def inspect_attachments(url, im, box, claim, readers):
    """A detection locates evidence. Only independent visual traces can settle the dispute."""
    field,boxes=localized_palms(im,box)
    if not 1<=len(boxes)<=3 or any(not model_identity(tag) for tag in readers):
        return {'decision':'unresolved','boxes':boxes}
    crops=[ImageOps.contain(im.crop(rect),(650,650),Image.Resampling.LANCZOS) for rect in boxes]
    fields={'wrist':{'type':'string'},'finger_or_occluder':{'type':'string'},'visible_contour':{'type':'string'},'object_boundary':{'type':'string'},
            'attachment':{'enum':['intact','broken','unreadable']}}
    schema={'type':'object','properties':{'hands':{'type':'array','items':{'type':'object','properties':fields,'required':list(fields)}},
            'all_disputed_parts_visible':{'type':'boolean'},'other_unattached_fragment':{'type':'boolean'}},
            'required':['hands','all_disputed_parts_visible','other_unattached_fragment']}
    readings=[]
    for index,tag in enumerate(readers):
        order=list(range(len(crops)))
        if index:order.reverse()
        context=im.crop(field).convert('RGB');draw=ImageDraw.Draw(context)
        for number,j in enumerate(order,1):
            x,y,u,v=boxes[j]
            draw.rectangle((x-field[0],y-field[1],u-field[0],v-field[1]),outline='#169eaa',width=2)
            draw.text((x-field[0]+3,max(0,y-field[1]+2)),str(number),fill='#0d5364')
        context.thumbnail((860,860),Image.Resampling.LANCZOS)
        prompt=('First image locates the numbered hands. Each following image is an unmarked crop of one '
                'hand in that order. Trace each visible finger to its OWN palm and wrist or a specific '
                'visible occluding object. Identify the edge where the hand touches or passes over the prop; '
                'describe which contour ends and which continues. If no prop is present use the adjacent '
                'surface. A fused finger-to-object boundary is broken. Report a row per crop. Check the total region for a stray '
                'fragment. Never assume a hand is intact because a crop exists. The disputed relationship '
                'must be visible in these pixels. Finding to verify independently: '+claim)
        row=json.loads(ask(url,[context]+[crops[j] for j in order],prompt,schema,tag=tag))
        readings.append(dict(model=tag,order=order,result=row))
        hands=row['hands']
        if (not row['all_disputed_parts_visible'] or row['other_unattached_fragment'] or len(hands)!=len(crops)
                or any(h['attachment']!='intact' or not all(str(h[k]).strip() for k in ('wrist','finger_or_occluder','visible_contour','object_boundary')) for h in hands)):
            return {'decision':'unresolved','boxes':boxes,'readings':readings}
    return {'decision':'coherent','boxes':boxes,'readings':readings}
