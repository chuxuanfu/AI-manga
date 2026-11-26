"""Locate explicitly forbidden lettering and propose a locally inpainted draft."""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from erase import erase_region, MODEL as INPAINTER
from evidence import ask, backend, model_identity
from print_judge import MODEL, fingerprint, store

DETECTOR = Path(__file__).resolve().parents[1] / 'models/text-detection/text_detection_cn_ppocrv3_2023may.onnx'


RECOGNIZER = DETECTOR.with_name("text_recognition_CRNN_EN_2022oct_int8.onnx")


def blank_surfaces_requested(brief):
    # A ban on captions alone still allows ordinary shop signs and writing on props.
    if re.search(r'\b(except|excluding the ban|apart from|other than|but (?:keep|allow|include))\b|除了|除外|保留.{0,8}(字|标语)', brief, re.I):
        return False
    return bool(re.search(r'\b(?:no|without)\s+(?:(?:captions?|signatures?|watermarks?|logos?|numbers?)'
                          r'(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+))*'
                          r'(?:text|lettering|writing)\b(?=\s*(?:$|[.,;!?:·]|(?:and|or)\s+(?:logos?|watermarks?|signatures?|captions?)\b))'
                          r'|(?:没有|不要|不含|禁止|不出现|无)(?:任何|可见的?)?(?:文字|字样|字母)(?=\s*(?:$|[。；、，！？,.!;·]))', brief, re.I))


def preserve_instrument_scale(brief):
    """A ban on letters can coexist with a working dial. An explicit ban on numbers takes priority."""
    instrument=re.search(r'\b(?:clocks?|wristwatch(?:es)?|pocket watch(?:es)?|watch face|sundials?|gauges?|dials?|rulers?|thermometers?|barometers?|compasses|speedometers?)\b|时钟|钟面|钟表|表盘|仪表|温度计|刻度尺',brief,re.I)
    if not instrument:return False
    prohibition=r'\b(?:no|without|omit|exclude)\b[^.;!?。；]{0,110}\b(?:text|writing|numbers?|digits?|numerals?|markings?|scales?)\b|(?:没有|不要|不含|禁止|无).{0,10}(?:数字|文字|刻度)'
    return not re.search(prohibition,brief,re.I)


def lettering_polygons(image):
    from print_judge import axes, SIDE
    model=cv2.dnn_TextDetectionModel_DB(str(DETECTOR))
    model.setInputMean((123.675,116.28,103.53))
    model.setInputScale(tuple(1/(255*std) for std in (.229,.224,.225)))
    model.setBinaryThreshold(.3)
    model.setPolygonThreshold(.80)
    model.setUnclipRatio(1.6)
    views=[(0,0,image.width,image.height,1088,1.)]
    if max(image.size)>2304:
        views.extend((x,y,min(x+SIDE,image.width),min(y+SIDE,image.height),1088,1.)
                     for y in axes(image.height) for x in axes(image.width))
    # Small inscriptions disappear in the full-frame OCR input. A second pass
    # gives a six-pixel letter roughly twelve detector pixels, independent of canvas size.
    side=480
    def starts(length):
        last=max(0,length-side)
        return list(range(0,last,side-64))+[last]
    views.extend((x,y,min(x+side,image.width),min(y+side,image.height),960,2.)
                 for y in starts(image.height) for x in starts(image.width))
    candidates=[]
    for left,top,right,bottom,limit,zoom in views:
        crop=image.crop((left,top,right,bottom))
        scale=min(zoom,limit/max(crop.size))
        model.setInputSize(tuple(max(32,round(v*scale/32)*32) for v in crop.size))
        quads,scores=model.detect(cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR))
        for quad,score in zip(quads,scores):
            if score<.80 or not 15<=cv2.contourArea(quad)<image.width*image.height*.05:continue
            quad=quad+np.array([left,top],dtype=quad.dtype)
            candidates.append((float(score),quad))
    kept=[]
    for _,quad in sorted(candidates,key=lambda item:-item[0]):
        area=cv2.contourArea(quad)
        if any((overlap:=cv2.intersectConvexConvex(quad,other)[0])>
               .80*(area+cv2.contourArea(other)-overlap) for other in kept):continue
        kept.append(quad)
    return [quad.tolist() for quad in kept]


def decode_word(scores):
    """Collapse repeated CTC labels, retaining confidence for each emitted glyph."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    ids = scores.argmax(axis=1)
    emitted = (ids != 0) & (ids != np.r_[0, ids[:-1]])
    chars = ids[emitted]
    word = ''.join('0123456789abcdefghijklmnopqrstuvwxyz'[i-1] for i in chars)
    confidence = probs[emitted, chars]
    return word, float(confidence.mean()) if len(chars) else 0.


def recognized_words(image, quads):
    if not quads or not RECOGNIZER.is_file():
        return {}
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    network = cv2.dnn.readNet(str(RECOGNIZER))
    target = np.float32(((0,31),(0,0),(99,0),(99,31)))
    confirmed = {}
    for index, quad in enumerate(quads):
        points = np.asarray(quad, np.float32)
        center = points.mean(axis=0)
        readings = []
        for scale in (.96, 1.05):
            warp = cv2.getPerspectiveTransform(center + (points-center)*scale, target)
            view = cv2.warpPerspective(gray, warp, (100,32))
            network.setInput(cv2.dnn.blobFromImage(view, 1/127.5, (100,32), 127.5))
            readings.append(decode_word(network.forward()[:,0]))
        word = readings[0][0]
        if (word == readings[1][0] and len(word) >= 4 and len(set(word)) >= 3
                and any(c.isalpha() for c in word) and not re.fullmatch('[ivxlcdm]+',word)
                and min(score for _,score in readings) >= .90):
            confirmed[index] = {'word':word, 'confidence':[round(v,4) for _,v in readings]}
    return confirmed


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
    # Individual letters may extend a few pixels above the detected word.
    # A shared bounding box across separate words can also enclose a hand or an
    # envelope. Keep edits close to the detector's actual polygons.
    radius=min(9,max(2,round(min(width,height)*.12)))
    near=cv2.dilate(support[top:bottom,left:right],
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    selected=cv2.bitwise_and(selected,near)
    full=Image.new('L',image.size)
    full.paste(Image.fromarray(selected),bounds[:2])
    return full


def denies_lettering(observation):
    """A vote cannot authorize removal when its own pixel description rules out writing."""
    words=r'(?:letters?|lettering|words?|text|writing|glyphs?|inscriptions?|characters?)'
    adjectives=r'(?:(?:tiny|small|actual|printed|written|handwritten|visible|clear|alphabetic)\s+)*'
    return bool(re.search(r'\b(?:rather than|instead of|without(?: any)?|contains? no|shows? no|not(?: any)?|no)\s+'
                          +adjectives+words+r'\b|\bdo(?:es)? not (?:form|represent|resemble) (?:any )?'+adjectives+words+r'\b'
                          +r'|(?:没有|并非|不构成|而非|未见)(?:任何|可辨的?|可见的?)?(?:字母|文字|字样)', observation, re.I))


def finish_blank_surface(original, inferred, mask):
    """Suppress ghost glyphs only where the observed material around the verified label is uniform."""
    bounds=mask.getbbox()
    if bounds is None:return inferred
    x,y,u,v=bounds
    band_width=np.count_nonzero(np.asarray(mask.crop(bounds)))/max(1,u-x,v-y)
    radius=max(3,min(12,round(band_width*.12)))
    rect=(max(0,x-radius-2),max(0,y-radius-2),min(original.width,u+radius+2),min(original.height,v+radius+2))
    ink=np.asarray(mask.crop(rect))>0
    grown=cv2.dilate(ink.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))>0
    ring=grown & ~ink
    if np.count_nonzero(ring)<32:return inferred
    raw=np.asarray(original.crop(rect),dtype=np.float32)
    new=np.asarray(inferred.crop(rect),dtype=np.float32)
    ref=raw[ring];median=np.median(ref,axis=0)
    deviation=np.max(np.abs(raw-median),axis=2)
    if np.percentile(deviation[ring],85)>18:
        # A thin faceplate can border a different casing. Accept its base color
        # only when the same dominant material is visible around AND inside
        # the marked surface, including both sides along its long axis.
        clean_tone=(deviation<=17)
        if np.mean(clean_tone[ring])<.58 or np.mean(clean_tone[ink])<.62:return inferred
        axis=np.indices(ink.shape)[1 if u-x>=v-y else 0]
        middle=float(np.median(axis[ink]))
        for half in (axis<middle,axis>=middle):
            ref_side=ring & half
            if not ref_side.any() or np.mean(clean_tone[ref_side])<.48:return inferred
    usable=ring & (deviation<22)
    spread=float(np.percentile(deviation[usable],85))
    if np.percentile(np.max(abs(new[ink]-median),axis=1),95)<max(6,2.4*spread):return inferred
    width=max(5,min(48,min(u-x,v-y)*.65))
    weight=cv2.GaussianBlur(usable.astype(np.float32),(0,0),width)
    total=cv2.GaussianBlur(raw*usable[...,None],(0,0),width)
    plane=np.where(weight[...,None]>.001,total/np.maximum(.001,weight[...,None]),median)
    # Infer texture from the same clean surrounding substrate. Reusing the
    # neural prediction's own high-frequency layer copies residual glyphs back.
    from scipy.ndimage import distance_transform_edt
    nearest=distance_transform_edt(~usable,return_distances=False,return_indices=True)
    clean_residual=(raw-plane)[nearest[0],nearest[1]]
    plane+=np.clip(clean_residual,-min(2,spread*.25),min(2,spread*.25))
    proposal=Image.fromarray(plane.clip(0,255).round().astype(np.uint8))
    out=inferred.copy();out.paste(proposal,rect[:2],mask.crop(rect));return out


def erase_verified_words(image, box, *, polygons=None):
    """Restrict a graded inscription repair to OCR polygons at that physical location."""
    if not DETECTOR.is_file():
        return None
    bounds=tuple(round(v*n/1000) for v,n in zip(box,image.size*2))
    left,top,right,bottom=bounds
    if right<=left or bottom<=top:
        return None
    pad=max(16,round(max(right-left,bottom-top)*.18))
    field=(max(0,left-pad),max(0,top-pad),min(image.width,right+pad),min(image.height,bottom+pad))
    crop=image.crop(field)
    mask=Image.new('L',image.size)
    # A tight crop improves small glyph scale but can hide the label's surrounding
    # surface. Preserve the full-frame detector's words in the same coordinate space.
    detected=[np.asarray(points,dtype=np.int32)+np.asarray(field[:2],dtype=np.int32)
              for points in lettering_polygons(crop)]
    full=lettering_polygons(image) if polygons is None else polygons
    detected.extend(np.asarray(points,dtype=np.int32) for points in full)
    for quad in detected:
        x,y,w,h=cv2.boundingRect(quad)
        if not (left<=x+w/2<=right and top<=y+h/2<=bottom):
            continue
        overlap=max(0,min(right,x+w)-max(left,x))*max(0,min(bottom,y+h)-max(top,y))
        if overlap < .65*w*h:
            continue
        word=Image.new('L',image.size)
        ImageDraw.Draw(word).polygon([tuple(pt) for pt in quad],fill=255)
        strokes=glyph_mask(image,word)
        if strokes is not None:
            # The print judge has independently confirmed the inscription here.
            # Remove the interior of its detected word too. Keeping its antialiased
            # fringe as conditioning can recreate the glyphs as embossed marks.
            mask=ImageChops.lighter(mask,ImageChops.lighter(strokes,word))
    if mask.getbbox():
        return finish_blank_surface(image,erase_region(image,mask),mask)
    return None


def remove_reviewed_feature(image, box, issue):
    from erase import masked_candidate
    target=str(issue.get('target',''))+' '+str(issue.get('symptom',''))
    if re.search(r'lettering|inscription|printed (?:text|word)|glyph|pseudo.text|文字|字迹|题字',target,re.I):
        clean=erase_verified_words(image,box)
        if clean is not None:
            return clean
    return masked_candidate(image,box)


def clean_lettering(path, brief, folder, stage):
    if not blank_surfaces_requested(brief) or not DETECTOR.is_file() or not INPAINTER.is_file():
        return path
    source_id = fingerprint(path)
    functional_scale=preserve_instrument_scale(brief)
    version='lettering-scale-v1' if functional_scale else 'lettering-v9'
    key = hashlib.sha256(json.dumps([version, RECOGNIZER.is_file(), source_id, brief, model_identity(MODEL)],
                                     ensure_ascii=False).encode()).hexdigest()
    report = folder / (stage + '-lettering.json')
    candidate = folder / (stage + '-plain-surfaces.png')
    prior_readings = {}
    if report.is_file():
        previous = json.loads(report.read_text())
        old_key = hashlib.sha256(json.dumps(['lettering-v5', source_id, brief, model_identity(MODEL)],
                                            ensure_ascii=False).encode()).hexdigest()
        old_ocr_key = hashlib.sha256(json.dumps(['lettering-v6',RECOGNIZER.is_file(),source_id,brief,model_identity(MODEL)],
                                                ensure_ascii=False).encode()).hexdigest()
        old_surface_key=hashlib.sha256(json.dumps(['lettering-v7',RECOGNIZER.is_file(),source_id,brief,model_identity(MODEL)],
                                                  ensure_ascii=False).encode()).hexdigest()
        substrate_key=hashlib.sha256(json.dumps(['lettering-v8',RECOGNIZER.is_file(),source_id,brief,model_identity(MODEL)],
                                                 ensure_ascii=False).encode()).hexdigest()
        if not functional_scale and previous.get('key') in (old_key,old_ocr_key,old_surface_key,substrate_key):
            prior_readings = {tuple(v['box']):v for v in previous['observations'] if 'box' in v}
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
    words = recognized_words(image, quads)
    confirmed = []
    remaining = []
    for index, quad in enumerate(quads):
        if index in words:
            # A tightly detected, twice-read word can contain thick or low-contrast
            # strokes. Inpaint its own quadrilateral to remove interior antialiasing.
            mask = Image.new('L', image.size)
            ImageDraw.Draw(mask).polygon([tuple(pt) for pt in quad], fill=255)
            confirmed.append(mask)
            record['observations'].append({'polygon':quad, 'ocr':words[index]})
        else:
            remaining.append(quad)
    quads = remaining
    # Join adjacent lines into one region so a multi-line label is seen in context.
    masks = Image.new('L', image.size)
    draw = ImageDraw.Draw(masks)
    for quad in quads:
        draw.polygon([tuple(point) for point in quad], fill=255)
    threshold = max(5, round(min(image.size)*.008))
    joined = cv2.dilate(np.asarray(masks),np.ones((threshold,threshold),np.uint8))
    _, labels, statistics, _ = cv2.connectedComponentsWithStats(joined)
    fields = {'appearance':{'type':'string'},'feature':{'enum':['printed_or_handwritten_marks','surface_detail','insufficient_evidence']}}
    overview=image.copy()
    overview.thumbnail((750,750),Image.Resampling.LANCZOS)
    schema = {'type':'object','properties':fields,'required':list(fields)}
    with ExitStack() as stack:
        url = None
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
            if functional_scale:
                question+=(' Identify functional graduations in their surrounding object. Hour indices arranged '
                           'around a clock pivot, evenly spaced ruler divisions or calibrated gauge numbers are '
                           'surface_detail. Inspect the central marks themselves: a manufacturer name, caption '
                           'or pseudo-writing on the same instrument remains printed_or_handwritten_marks.')
            cached = prior_readings.get(tuple(map(int,(left,top,width,height))))
            if cached:
                record['observations'].append(cached)
                result = cached.get('independent_reading',cached['reading'])
            else:
                if url is None:
                    url = stack.enter_context(backend(MODEL))
                result = json.loads(ask(url,[view,overview],question,schema,tag=MODEL))
                record['observations'].append(dict(box=list(map(int,(left,top,width,height))),reading=result))
                # A schema label in the prose field is not visual evidence. Repeat
                # the close inspection without the overview when the two fields disagree.
                if result['appearance'].strip() in fields['feature']['enum']:
                    described=json.loads(ask(url,[view],
                        'Describe the marked surface and its visible strokes in one concrete sentence. '
                        'Mention the color, layout and whether these marks form tiny writing or object contours.',
                        {'type':'object','properties':{'description':{'type':'string'}},'required':['description']},tag=MODEL))
                    evidence=described['description'].strip()
                    vote=json.loads(ask(url,[view],'Look at the physical marks and verify this observation: '
                        +evidence+'. Classify the marks. Use insufficient_evidence for ambiguity.',
                        {'type':'object','properties':{'feature':fields['feature']},'required':['feature']},tag=MODEL))
                    result={'appearance':evidence,'feature':vote['feature']}
                    record['observations'][-1]['independent_reading']=result
            if result['feature']=='printed_or_handwritten_marks' and denies_lettering(result['appearance']):
                record['observations'][-1]['preserved']='pixel description rules out writing'
                continue
            if result['feature']=='printed_or_handwritten_marks' and result['appearance'].strip():
                # Mask each word separately. Neighbouring lines can have different ink
                # polarity or be printed on opposite sides of an object boundary.
                group_mask=Image.new('L',image.size)
                for quad in quads:
                    cx,cy=np.mean(quad,axis=0).astype(int)
                    if not (0<=cx<image.width and 0<=cy<image.height and labels[cy,cx]==index):continue
                    word=Image.new('L',image.size)
                    ImageDraw.Draw(word).polygon([tuple(point) for point in quad],fill=255)
                    strokes=glyph_mask(image,word)
                    if strokes is not None:group_mask=ImageChops.lighter(strokes,group_mask)
                if group_mask.getbbox():confirmed.append(group_mask)
    if confirmed:
        cleaned=image
        for mask in confirmed:
            cleaned=finish_blank_surface(cleaned,erase_region(cleaned,mask),mask)
        tmp = candidate.with_suffix('.tmp')
        cleaned.save(tmp,format='PNG')
        tmp.replace(candidate)
        record['candidate_sha256'] = fingerprint(candidate)
    store(report,record)
    return candidate if 'candidate_sha256' in record else path
