"""Locate explicitly forbidden lettering and propose a locally inpainted draft."""
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from erase import erase_region, MODEL as INPAINTER
from evidence import ask, backend, model_identity
from print_judge import MODEL, fingerprint, store

DETECTOR = Path(__file__).resolve().parents[1] / 'models/text-detection/text_detection_cn_ppocrv3_2023may.onnx'


def blank_surfaces_requested(brief):
    # A ban on captions alone still allows ordinary shop signs and writing on props.
    if re.search(r'\b(except|excluding the ban|apart from|other than|but (?:keep|allow|include))\b|除了|除外|保留.{0,8}(字|标语)', brief, re.I):
        return False
    return bool(re.search(r'\b(?:no|without)\s+(?:(?:captions?|signatures?|watermarks?|logos?|numbers?)'
                          r'(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+))*'
                          r'(?:text|lettering|writing)\b(?=\s*(?:$|[.,;!?:·]|(?:and|or)\s+(?:logos?|watermarks?|signatures?|captions?)\b))'
                          r'|(?:没有|不要|不含|禁止|不出现|无)(?:任何|可见的?)?(?:文字|字样|字母)(?=\s*(?:$|[。；、，！？,.!;·]))', brief, re.I))


def lettering_polygons(image):
    from print_judge import axes, SIDE
    model=cv2.dnn_TextDetectionModel_DB(str(DETECTOR))
    model.setInputMean((123.675,116.28,103.53))
    model.setInputScale(tuple(1/(255*std) for std in (.229,.224,.225)))
    model.setBinaryThreshold(.3)
    model.setPolygonThreshold(.80)
    model.setUnclipRatio(1.6)
    views=[(0,0,image.width,image.height)]
    if max(image.size)>2304:
        views.extend((x,y,min(x+SIDE,image.width),min(y+SIDE,image.height))
                     for y in axes(image.height) for x in axes(image.width))
    candidates=[]
    for left,top,right,bottom in views:
        crop=image.crop((left,top,right,bottom))
        scale=min(1.,1088/max(crop.size))
        model.setInputSize(tuple(max(32,round(v*scale/32)*32) for v in crop.size))
        quads,scores=model.detect(cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR))
        for quad,score in zip(quads,scores):
            if score<.80 or not 15<=cv2.contourArea(quad)<image.width*image.height*.05:continue
            quad=quad+np.array([left,top],dtype=quad.dtype)
            candidates.append((float(score),quad))
    kept=[]
    for _,quad in sorted(candidates,key=lambda item:-item[0]):
        area=cv2.contourArea(quad)
        if any(cv2.intersectConvexConvex(quad,other)[0]>.85*min(area,cv2.contourArea(other)) for other in kept):continue
        kept.append(quad)
    return [quad.tolist() for quad in kept]


def glyph_mask(image, proposal):
    """Lift high-contrast strokes without erasing the substrate or its outer edge."""
    support=np.asarray(proposal)
    x,y,width,height=cv2.boundingRect(support)
    if min(width,height)<4:return None
    span=max(5,round(min(width,height)*.29))|1
    pad=round(span*1.65)
    bounds=max(0,x-pad), max(0,y-pad), min(image.width,x+width+pad), min(image.height,y+height+pad)
    left,top,right,bottom=bounds
    luminance=cv2.cvtColor(np.asarray(image.crop(bounds)),cv2.COLOR_RGB2GRAY)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(span,span))
    responses=[cv2.morphologyEx(luminance,mode,kernel) for mode in (cv2.MORPH_TOPHAT,cv2.MORPH_BLACKHAT)]
    inside=support[top:bottom,left:right]>0
    contrast,polarity=max((np.percentile(response[inside],96),index) for index,response in enumerate(responses))
    if contrast<42:return None
    strokes=(responses[polarity]>contrast*.55).astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(strokes)
    selected=np.zeros_like(strokes)
    for index,(cx,cy,w,h,area) in enumerate(stats[1:],1):
        if area<6 or w<3 or h<3 or cx<=0 or cy<=0 or cx+w>=right-left or cy+h>=bottom-top:continue
        cx+=left;cy+=top
        # A nearby ascender or cap can extend beyond the detector's word polygon.
        if (x-.02*width<cx+w/2<x+1.02*width and y-.25*height<cy+h/2<y+1.05*height):
            selected[labels==index]=255
    if not selected.any():return None
    stroke_pad=min(9,max(3,2*round(min(width,height)/32)+1))
    selected=cv2.dilate(selected,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(stroke_pad,stroke_pad)))
    full=Image.new('L',image.size)
    full.paste(Image.fromarray(selected),bounds[:2])
    return full


def clean_lettering(path, brief, folder, stage):
    if not blank_surfaces_requested(brief) or not DETECTOR.is_file() or not INPAINTER.is_file():
        return path
    source_id = fingerprint(path)
    key = hashlib.sha256(json.dumps(['lettering-v4', source_id, brief, model_identity(MODEL)],
                                     ensure_ascii=False).encode()).hexdigest()
    report = folder / (stage + '-lettering.json')
    candidate = folder / (stage + '-plain-surfaces.png')
    if report.is_file():
        previous = json.loads(report.read_text())
        if previous.get('key') == key:
            if not previous.get('candidate_sha256'):
                return path
            if candidate.is_file() and fingerprint(candidate) == previous['candidate_sha256']:
                return candidate
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    quads = lettering_polygons(image)
    record = dict(key=key, source_sha256=source_id, polygons=quads, observations=[])
    if not quads:
        store(report, record)
        return path
    # Join adjacent lines into one region so a multi-line label is seen in context.
    masks = Image.new('L', image.size)
    draw = ImageDraw.Draw(masks)
    for quad in quads:
        draw.polygon([tuple(point) for point in quad], fill=255)
    threshold = max(5, round(min(image.size)*.008))
    joined = cv2.dilate(np.asarray(masks),np.ones((threshold,threshold),np.uint8))
    _, labels, statistics, _ = cv2.connectedComponentsWithStats(joined)
    confirmed = []
    fields = {'appearance':{'type':'string'},'feature':{'enum':['printed_or_handwritten_marks','surface_detail','insufficient_evidence']}}
    overview=image.copy()
    overview.thumbnail((750,750),Image.Resampling.LANCZOS)
    schema = {'type':'object','properties':fields,'required':list(fields)}
    with backend(MODEL) as url:
        groups=sorted(enumerate(statistics[1:],start=1),key=lambda item:-item[1][4])
        for index,(left,top,width,height,_) in groups[:12]:
            pad = max(20, round(max(width,height)*.65))
            rect = (max(0,left-pad),max(0,top-pad),min(image.width,left+width+pad),min(image.height,top+height+pad))
            view = image.crop(rect)
            factor = min(3.,800/max(view.size))
            view = view.resize(tuple(max(1,round(n*factor)) for n in view.size),Image.Resampling.LANCZOS)
            question = ('Image one enlarges markings near the center. Image two shows their original '
                        'illustration. Identify the physical surface, ink color and layout of the markings '
                        'in your own words. A row of glyph-like strokes arranged as a label, title, signature '
                        'or inscription is printed_or_handwritten_marks even if it is illegible or invented. '
                        'Lines defining object contours, seams, grain, reflections or a pictorial motif are '
                        'surface_detail. If the pixels cannot distinguish these, use insufficient_evidence.')
            result = json.loads(ask(url,[view,overview],question,schema,tag=MODEL))
            record['observations'].append(dict(box=list(map(int,(left,top,width,height))),reading=result))
            if result['feature']=='printed_or_handwritten_marks' and result['appearance'].strip():
                selected = (np.asarray(masks)>0) & (labels==index)
                strokes=glyph_mask(image,Image.fromarray(selected.astype(np.uint8)*255))
                if strokes is not None:confirmed.append(strokes)
    if confirmed:
        cleaned=image
        for mask in confirmed:
            cleaned=erase_region(cleaned,mask)
        tmp = candidate.with_suffix('.tmp')
        cleaned.save(tmp,format='PNG')
        tmp.replace(candidate)
        record['candidate_sha256'] = fingerprint(candidate)
    store(report,record)
    return candidate if 'candidate_sha256' in record else path
