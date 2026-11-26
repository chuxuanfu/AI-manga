"""Place one existing independent subject at a different projected size and depth."""
import json
import math
import re

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy import ndimage

from erase import MODEL, erase_region
from evidence import ask, backend
from manga_materials.automatic import probabilities
from masks import refine_subject_mask


def placement_requested(instruction):
    # Enable the planner only when a correction explicitly calls for relocation.
    return bool(re.search(r'\b(?:move|reposition|relocate|place|position|seat|shift|put|rest|set)\b.{0,170}'
                          r'\b(?:farther|further|foreground|background|closer to|to the (?:left|right) of|'
                          r'far (?:behind|in front)|beside the distant|onto|on top of)\b|(?:移到|放到|移动到).{0,18}(?:远处|近处|前景|背景|左侧|右侧|上面|顶部|上)',
                          instruction,re.I|re.S))


def placement_plan(image,instruction,brief):
    if not placement_requested(instruction) or not MODEL.is_file():return None
    scene=image.copy()
    scene.thumbnail((1152,1152),Image.Resampling.LANCZOS)
    point=dict(type='integer',minimum=0,maximum=1000)
    fields={'standalone':{'type':'boolean'},'requirement_quote':{'type':'string'},'label':{'type':'string'},'parts':{
            'type':'array','maxItems':6,'items':{'type':'string'}},'source':{
            'type':'array','minItems':4,'maxItems':4,'items':point},
            'destination_foot':{'type':'array','minItems':2,'maxItems':2,'items':point},
            'scale':{'type':'number','minimum':.2,'maximum':1.8},'revealed_surface':{'type':'string'}}
    tag='qwen3.8:27b'
    with backend(tag) as url:
        plan=json.loads(ask(url,[scene],'Assess a proposed scene correction. Set standalone true '
            'only if the existing fully visible actor or object has the correct identity and pose, '
            'is separable from other subjects, and only its position or apparent scale is wrong. '
            'A limb, head, clothing on a body, attached hardware, intertwined figures or a cropped '
            'subject is ineligible. A destination behind an occluder is ineligible. Name one '
            'concrete segmentable subject. For a single subject return empty parts. When contents, '
            'their container or distinct attached pieces must travel together, supply a short local '
            'segmentation label for each visible constituent including the container. Use simple '
            'object nouns and colors. Exclude the fixed source surface and the destination support. '
            'Locate the bounding box LTRB of the entire carried assembly and the intended '
            'new bottom-center point on its support plane in whole-canvas coordinates 0..1000. '
            'Estimate the required uniform scale relative to its current size. Describe the '
            'unoccupied material and contours to restore at the vacated site. Quote verbatim the part of '
            'the original brief which specifies the requested placement. Brief: '+brief+
            '. Requested correction: '+instruction,
            {'type':'object','properties':fields,'required':list(fields)},tag=tag))
    if not isinstance(plan,dict):return None
    fields=('requirement_quote','label','revealed_surface')
    if not all(isinstance(plan.get(key),str) and plan[key].strip() for key in fields):return None
    quote=plan['requirement_quote'].strip()
    if plan.get('standalone') is True and len(quote)>=3 and quote.casefold() in brief.casefold():
        return plan
    return None


def mend_color_matched_holes(image, alpha):
    """Close small breaks within one material while retaining contrasting openings."""
    binary=np.asarray(alpha)>128
    labels,count=ndimage.label(ndimage.binary_fill_holes(binary)&~binary)
    if not count:return alpha
    pixels=np.asarray(image.convert('RGB'))
    limit=binary.sum()*.03
    modified=np.asarray(alpha).copy()
    for index,extent in enumerate(ndimage.find_objects(labels),1):
        if extent is None:continue
        vicinity=tuple(slice(max(0,axis.start-2),min(size,axis.stop+2)) for axis,size in zip(extent,labels.shape))
        hole=labels[vicinity]==index
        if hole.sum()>limit:continue
        ring=ndimage.binary_dilation(hole,iterations=2)&binary[vicinity]
        colors=pixels[vicinity]
        if ring.any() and np.max(np.abs(np.median(colors[hole],axis=0)-np.median(colors[ring],axis=0)))<38:
            modified[vicinity][hole]=255
    return Image.fromarray(modified)


def color_boundary(image, coarse):
    """Refine a small carried part using its own foreground and background colors."""
    import cv2
    ink=np.asarray(coarse)>128
    yy,xx=np.nonzero(ink)
    if not len(xx):return None
    gap=max(3,round(min(np.ptp(xx)+1,np.ptp(yy)+1)*.10))
    box=(max(0,min(xx)-2*gap),max(0,min(yy)-2*gap),
         min(image.width,max(xx)+2*gap+1),min(image.height,max(yy)+2*gap+1))
    seed=ink[box[1]:box[3],box[0]:box[2]]
    inside=ndimage.binary_erosion(seed,iterations=gap)
    outside=ndimage.binary_dilation(seed,iterations=gap)
    if inside.sum()<12 or (~outside).sum()<12:return None
    classes=np.full(seed.shape,cv2.GC_BGD,np.uint8)
    classes[outside]=cv2.GC_PR_BGD
    classes[seed]=cv2.GC_PR_FGD
    classes[inside]=cv2.GC_FGD
    pixels=np.array(image.crop(box).convert('RGB'))
    cv2.grabCut(pixels,classes,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
    matte=(classes==cv2.GC_FGD)|(classes==cv2.GC_PR_FGD)
    if min((matte&seed).sum()/seed.sum(),(matte&seed).sum()/matte.sum())<.75:return None
    result=Image.new('L',image.size)
    result.paste(Image.fromarray(matte.astype(np.uint8)*255),box[:2])
    return result


def reposition(image,plan):
    """Return a provisional local composite plus the areas requiring model inspection."""
    if not isinstance(plan,dict) or plan.get('standalone') is not True:return None
    for field,length in (('source',4),('destination_foot',2)):
        values=plan.get(field)
        if not isinstance(values,(list,tuple)) or len(values)!=length:return None
        if not all(type(v) in (int,float) and 0<=v<=1000 for v in values):return None
    if type(plan.get('scale')) not in (int,float) or not math.isfinite(plan['scale']):return None
    w,h=image.size
    l,t,r,b=plan['source']
    if not (0<=l<r<=1000 and 0<=t<b<=1000 and (r-l)*(b-t)<350000):return None
    x,y=plan['destination_foot'];scale=plan['scale']
    if not (0<=x<=1000 and 0<=y<=1000 and .2<=scale<=1.8):return None
    if not isinstance(plan.get('label'),str) or not plan['label'].strip():return None
    src=tuple(round(v*n/1000) for v,n in zip((l,t,r,b),image.size*2))
    pad=max(10,round(max(src[2]-src[0],src[3]-src[1])*.14))
    search=(max(0,src[0]-pad),max(0,src[1]-pad),min(w,src[2]+pad),min(h,src[3]+pad))
    crop=image.crop(search)
    parts=plan.get('parts',[])
    if not isinstance(parts,list) or len(parts)>6:return None
    if any(not isinstance(part,str) or not part.strip() for part in parts):return None
    labels=list(dict.fromkeys(parts or [plan['label']]))
    alpha=Image.new('L',crop.size)
    for score in probabilities(crop,labels):
        peak=float(score.max())
        if peak<.55:return None
        support=score>max(.27,peak*.44)
        if len(labels)==1:
            rough=Image.fromarray(np.uint8(support)*255).resize(crop.size,Image.Resampling.BILINEAR)
            alpha=refine_subject_mask(crop,rough)
            if alpha is rough:return None
            break
        objects,count=ndimage.label(support)
        areas=np.bincount(objects.ravel(),minlength=count+1)
        pieces=[i for i in range(1,count+1) if areas[i]>=max(12,.025*support.sum())]
        if not pieces or len(pieces)>12:return None
        for i in pieces:
            rough=Image.fromarray(np.uint8(objects==i)*255).resize(crop.size,Image.Resampling.BILINEAR)
            contour=refine_subject_mask(crop,rough)
            if contour is rough:contour=color_boundary(crop,rough)
            if contour is None or contour is rough:return None
            alpha=ImageChops.lighter(alpha,contour)
    if len(labels)>1:
        # Parts travelling together must form one spatial assembly. Do not lift
        # another copy of a named component from elsewhere in the source window.
        occupied=np.asarray(alpha)>128
        near=ndimage.binary_dilation(occupied,iterations=max(2,round(min(crop.size)*.012)))
        if ndimage.label(near)[1]!=1:return None
    border=alpha.point(lambda v:255 if v>128 else 0).getbbox()
    if not border or min(border[0],border[1],crop.width-border[2],crop.height-border[3])<1:return None
    alpha=mend_color_matched_holes(crop,alpha)
    cutout=crop.crop(border).convert('RGBA')
    cutout.putalpha(alpha.crop(border))
    bw,bh=cutout.size
    width,height=round(bw*scale),round(bh*scale)
    left,top=round(w*x/1000-width/2),round(h*y/1000-height)
    if min(width,height)<24 or min(left,top)<0 or left+width>w or top+height>h:return None
    x0,y0,x1,y1=[v+o for v,o in zip(border,search[:2]*2)]
    if min(left+width,x1)>max(left,x0) and min(top+height,y1)>max(top,y0):return None
    mask=Image.new('L',image.size)
    mask.paste(alpha,search[:2])
    dilation=min(31,max(3,2*round(.014*min(bw,bh))+1))
    mask=mask.filter(ImageFilter.MaxFilter(dilation)).point(lambda v:255 if v>30 else 0)
    # A cast shadow can extend beyond the feet. Follow only dark connected
    # strokes which touch the old silhouette within its narrow ground band.
    strip=(max(0,x0-bw//3),max(0,y1-round(bh*.048)),min(w,x1+bw//2),min(h,y1+max(2,round(bh*.025))))
    gray=np.asarray(image.crop(strip).convert('L'))
    dark=gray<min(145,np.percentile(gray,70)*.68)
    joined=ndimage.binary_closing(dark,structure=np.ones((2,7)))
    labels,_=ndimage.label(joined)
    touched=np.unique(labels[(np.asarray(mask.crop(strip))>0)&joined])
    shadow=np.isin(labels,touched[touched!=0])
    extra=Image.fromarray(np.uint8(ndimage.binary_dilation(shadow,iterations=4))*255)
    mask.paste(ImageChops.lighter(mask.crop(strip),extra),strip)
    destination=(left,top,left+width,top+height)
    if mask.crop(destination).getbbox():return None
    result=erase_region(image,mask)
    # Resize premultiplied RGBA so background colors cannot leak into ink edges.
    moved=cutout.resize((width,height),Image.Resampling.LANCZOS)
    result.paste(moved,(left,top),moved.getchannel('A'))
    def normalized(rect):return [round(1000*v/n) for v,n in zip(rect,image.size*2)]
    new=(left,top,left+width,top+height)
    old=mask.getbbox()
    return result,dict(source_subject=normalized((x0,y0,x1,y1)),vacated=normalized(old),
                       destination=normalized(new),inspection=normalized((min(old[0],new[0]),min(old[1],new[1]),
                                                                          max(old[2],new[2]),max(old[3],new[3]))))
