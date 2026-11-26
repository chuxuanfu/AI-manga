"""Preserve neighbouring objects when a localized correction names its subject."""
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter
import numpy as np
from scipy import ndimage
from manga_materials.automatic import probabilities


def object_mask(before, after, target, area, neighbours=(), *, confirmed_shape=False):
    if not target or not target.strip():return area
    target_words=set(target.lower().split())
    labels=[name for name in neighbours if name and not target_words.intersection(name.lower().split())][:7]
    masks=[]
    protected=np.zeros((352,352),bool)
    for im in (before,after):
        scores=probabilities(im,[target,*labels])
        score=scores[0]
        if len(scores)>1:
            other=scores[1:].max(axis=0)
            protected |= (other>.36) & (other>score-.12)
        if score.max()<.50:continue
        binary=ndimage.binary_fill_holes(score>max(.32,.52*float(score.max())))
        binary=ndimage.binary_dilation(binary,iterations=3)
        mask=Image.fromarray(np.uint8(binary)*255).resize(im.size,Image.Resampling.BILINEAR)
        masks.append(mask.filter(ImageFilter.GaussianBlur(max(1,min(im.size)*.005))))
    if not masks:return area
    silhouette=masks[0] if len(masks)==1 else ImageChops.lighter(*masks)
    supported=ImageChops.multiply(area,silhouette)
    if protected.any():
        protected=ndimage.binary_dilation(protected,iterations=3)
        keep=Image.fromarray((~protected).astype(np.uint8)*255).resize(before.size,Image.Resampling.BILINEAR)
        silhouette=ImageChops.multiply(silhouette,keep)
    limited=ImageChops.multiply(area,silhouette)
    if confirmed_shape:
        # A part and its parent (a fitting and its rail, for example) occupy the
        # same pixels. A coarse parent label can otherwise erase the verified repair.
        capacity=np.asarray(area).sum()
        support=np.asarray(supported).sum()
        if capacity and support < .015*capacity:
            return area
        if support and np.asarray(limited).sum() < .05*support:
            return supported
    return limited


def refine_subject_mask(image, coarse):
    """Refine a text-located object against its actual ink boundary on the CPU."""
    model_dir=Path(__file__).resolve().parents[1]/'models/subject-mask'
    if not (model_dir/'model.safetensors').is_file():return coarse
    return _refine_subject_mask(image,coarse,model_dir)


def _refine_subject_mask(image,coarse,model_dir):
    import torch
    small=image.convert('RGB')
    small.thumbnail((960,960),Image.Resampling.LANCZOS)
    if max(hi-lo for lo,hi in small.getextrema())<8:return coarse
    support=np.asarray(coarse.resize(small.size,Image.Resampling.BILINEAR))>=128
    components,total=ndimage.label(support)
    if not total:return coarse
    areas=np.bincount(components.ravel())[1:]
    support=components==(1+int(areas.argmax()))
    if support.sum()<.85*areas.sum() or support.mean()>.9:return coarse
    ys,xs=np.nonzero(support)
    pad=max(6,round(max(np.ptp(xs),np.ptp(ys))*.025))
    box=[max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
         min(small.width,int(xs.max())+pad+1),min(small.height,int(ys.max())+pad+1)]
    processor,model=boundary_model(model_dir)
    inputs=processor(images=small,input_boxes=[[box]],return_tensors='pt')
    with torch.inference_mode():result=model(**inputs,multimask_output=False)
    if result.iou_scores.item()<.86:return coarse
    mask=processor.post_process_masks(result.pred_masks.cpu(),inputs['original_sizes'])[0][0,0].numpy().astype(bool)
    common=(mask&support).sum()
    if common<.65*support.sum() or common<.65*mask.sum():return coarse
    return Image.fromarray(np.uint8(mask)*255).resize(image.size,Image.Resampling.LANCZOS)


@lru_cache(maxsize=1)
def boundary_model(model_dir):
    from transformers import Sam2Model,Sam2Processor
    return (Sam2Processor.from_pretrained(model_dir,local_files_only=True),
            Sam2Model.from_pretrained(model_dir,local_files_only=True).eval())


def focal_subject_mask(image, area, label, feature, *, require_feature=False):
    """Confine optical detail to an existing subject and locate its feature for a closer retry."""
    names=[label]+([feature] if feature and feature!=label else [])
    scores=probabilities(image,names)
    parent=scores[0]
    if parent.max()<.48:return area,None

    def support(score):
        body=score>max(.24,.45*float(score.max()))
        return Image.fromarray(np.uint8(body)*255).resize(image.size,Image.Resampling.BILINEAR)

    silhouette=refine_subject_mask(image,support(parent))
    # The outer contour stays in the source coordinates. A newly painted second
    # subject cannot enlarge its own edit mask.
    limited=ImageChops.multiply(area,silhouette)
    if limited.getbbox() is None:return limited,None
    focus=limited
    if len(scores)>1 and scores[1].max()>=.40:
        inner=ImageChops.multiply(limited,support(scores[1]))
        if inner.point(lambda x:255 if x>=128 else 0).getbbox():
            focus=inner
    if require_feature and feature and feature!=label:
        if focus is limited:return limited,None
        return focus,focus.point(lambda x:255 if x>=128 else 0).getbbox()
    return limited,focus.point(lambda x:255 if x>=128 else 0).getbbox()


def visible_part_extent(image, label):
    """Measure one confidently isolated subject in the current crop's coordinates."""
    score=probabilities(image,[label])[0]
    if score.max()<.65:return None
    parts,count=ndimage.label(score>max(.38,.56*float(score.max())))
    if not count:return None
    sizes=np.bincount(parts.ravel())[1:]
    dominant=int(sizes.argmax())+1
    if sizes[dominant-1]<.80*sizes.sum():return None
    ys,xs=np.nonzero(parts==dominant)
    extent=np.quantile(xs,[.03,.97])*image.width/score.shape[1]
    vertical=np.quantile(ys,[.03,.97])*image.height/score.shape[0]
    if min(extent[1]-extent[0],vertical[1]-vertical[0])<12:return None
    if (extent[0]<1 or vertical[0]<1 or extent[1]>image.width-2 or vertical[1]>image.height-2):
        return None
    return dict(center=[float(extent.mean()),float(vertical.mean())],width=float(extent[1]-extent[0]),
                height=float(vertical[1]-vertical[0]),area=float(len(xs))/score.size)


def unintended_portrait_scale(before, after, instruction):
    import re
    # A commissioned change of scale or vertical head tilt may intentionally
    # change the projected height. Horizontal rotations mostly change width.
    if re.search(r'\b(?:enlarge|shrink|resize|zoom|bigger|smaller|larger|taller|shorter|'
                 r'(?:looking|tilt(?:ed)?) (?:up|down)(?:ward)?)\b|放大|缩小|仰头|低头',
                 instruction,re.I):return None
    source=visible_part_extent(before,'face')
    revised=visible_part_extent(after,'face')
    if not source or not revised:return None
    height=revised['height']/source['height']
    area=revised['area']/source['area']
    if (height>1.5 and area>1.8) or (height<.65 and area<.55):
        return dict(source=source,revised=revised,height_ratio=height,area_ratio=area)
    return None


def frame_neutral_guide(guide, crop, label):
    """Put a studio geometry plate at the original part's size and position."""
    extent=visible_part_extent(crop,label)
    if extent is None:return guide,None
    # This optional metadata is used only on neutral plates whose subject fills
    # the image. Ordinary documentary photographs keep their original framing.
    size=tuple(max(1,round(v*extent['height']/guide.height)) for v in guide.size)
    center=extent['center']
    canvas=Image.new('RGB',crop.size,guide.convert('RGB').getpixel((0,0)))
    canvas.paste(guide.resize(size,Image.Resampling.LANCZOS),
                 (round(center[0]-size[0]/2),round(center[1]-size[1]/2)))
    return canvas,extent


def frame_supported_object(plate, size, bounds):
    """Register a cutout to the actual repair box and the bottom of its support."""
    if 'A' not in plate.getbands() or not plate.getchannel('A').getbbox():
        raise ValueError('A supported object guide needs a visible alpha cutout')
    l,t,r,b=bounds
    l,t=max(0,l),max(0,t)
    r,b=min(size[0],r),min(size[1],b)
    if r<=l or b<=t:
        raise ValueError('Object guide falls outside the repair field')
    shape=plate.convert('RGBA').crop(plate.getchannel('A').getbbox())
    scale=min((r-l)/shape.width,(b-t)/shape.height)
    shape=shape.resize((max(1,round(shape.width*scale)),max(1,round(shape.height*scale))),Image.Resampling.LANCZOS)
    x=l+(r-l-shape.width)//2
    y=b-shape.height
    canvas=Image.new('RGBA',size)
    canvas.alpha_composite(shape,(x,y))
    return canvas,(x,y,x+shape.width,y+shape.height)


def structural_composite(before, after, area, canvas_edges):
    """Place a rebuilt shape intact, joining the two drawings through agreeing surroundings."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order

    # Search for a join in a band around the approved correction. Cost is high on
    # displaced ink lines and color boundaries. The actual canvas edge needs no join.
    scale=min(1,192/max(before.size))
    size=tuple(max(2,round(n*scale)) for n in before.size)
    old=np.asarray(before.resize(size,Image.Resampling.BOX),dtype=np.float32)
    new=np.asarray(after.resize(size,Image.Resampling.BOX),dtype=np.float32)
    core=np.asarray(area.resize(size,Image.Resampling.BILINEAR))>=230
    if not core.any():return before.copy()
    distance=ndimage.distance_transform_edt(~core)
    band=max(6, round(max(size)*.20))
    outside=distance>=band
    left,top,right,bottom=canvas_edges
    if not left:outside[:,0]=True
    if not top:outside[0,:]=True
    if not right:outside[:,-1]=True
    if not bottom:outside[-1,:]=True
    core=core & ~outside
    if not core.any():return before.copy()
    # Positive capacities and fixed terminals guarantee a closed join.
    discrepancy=np.abs(old-new).mean(axis=2)
    discrepancy=cv2.GaussianBlur(discrepancy,(0,0),.9)
    cost=1+np.rint(3*discrepancy).astype(np.int32)
    ids=np.arange(core.size,dtype=np.int32).reshape(core.shape)
    a=np.concatenate((ids[:,:-1].ravel(),ids[:-1,:].ravel()))
    b=np.concatenate((ids[:,1:].ravel(),ids[1:,:].ravel()))
    weight=cost.ravel()[a]+cost.ravel()[b]
    src,sink=core.size,core.size+1
    inner=ids[core];outer=ids[outside]
    rows=np.concatenate((a,b,np.full(inner.size,src),outer))
    cols=np.concatenate((b,a,inner,np.full(outer.size,sink)))
    capacities=np.concatenate((weight,weight,np.full(inner.size+outer.size,1000000,dtype=np.int32)))
    graph=coo_matrix((capacities,(rows,cols)),shape=(sink+1,sink+1)).tocsr()
    residual=graph-maximum_flow(graph,src,sink).flow
    residual.eliminate_zeros()
    reachable=breadth_first_order(residual,src,directed=True,return_predecessors=False)
    weights=np.zeros(core.size,dtype=np.uint8)
    weights[reachable[reachable<core.size]]=255
    seam=Image.fromarray(weights.reshape(core.shape)).resize(before.size,Image.Resampling.BILINEAR)
    seam=seam.filter(ImageFilter.GaussianBlur(max(.6,min(before.size)*.003)))
    # Preserve ink edges at fine scales and blend broad tone changes over a
    # wider transition. A single thin feather exposes shadows as a rectangular patch.
    previous=np.asarray(before,dtype=np.float32)
    proposed=np.asarray(after,dtype=np.float32)
    alpha=np.asarray(seam,dtype=np.float32)/255
    layers=[]
    while min(alpha.shape)>12:
        old_low=cv2.pyrDown(previous)
        new_low=cv2.pyrDown(proposed)
        dims=(alpha.shape[1],alpha.shape[0])
        old_band=previous-cv2.pyrUp(old_low,dstsize=dims)
        new_band=proposed-cv2.pyrUp(new_low,dstsize=dims)
        layers.append(old_band+(new_band-old_band)*alpha[:,:,None])
        previous,proposed,alpha=old_low,new_low,cv2.pyrDown(alpha)
    result=previous+(proposed-previous)*alpha[:,:,None]
    for layer in reversed(layers):
        result=cv2.pyrUp(result,dstsize=(layer.shape[1],layer.shape[0]))+layer
    # Exact unchanged pixels at the crop boundary prevent a new line at paste time.
    fade=np.ones(before.size[::-1],np.float32)
    length=max(2,round(min(before.size)*.065))
    ramp=np.linspace(0,1,length,dtype=np.float32)
    if not left:fade[:,:length]*=ramp
    if not right:fade[:,-length:]*=ramp[::-1]
    if not top:fade[:length,:]*=ramp[:,None]
    if not bottom:fade[-length:,:]*=ramp[::-1,None]
    original=np.asarray(before,dtype=np.float32)
    result=original+(result-original)*fade[:,:,None]
    return Image.fromarray(np.uint8(np.clip(result.round(),0,255)))


def pose_outline(photo, subject):
    """Reduce a pose-only reference to its silhouette and the main internal contours."""
    import cv2
    working=photo.convert('RGB')
    working.thumbnail((640,640),Image.Resampling.LANCZOS)
    score=probabilities(working,[subject])[0]
    if score.max()<.60:return None
    rough=Image.fromarray(np.uint8(score>max(.28,float(score.max())*.48))*255).resize(working.size,Image.Resampling.BILINEAR)
    alpha=refine_subject_mask(working,rough)
    if alpha is rough:return None
    silhouette=np.uint8(np.asarray(alpha)>128)*255
    contours,_=cv2.findContours(silhouette,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return None
    contour=max(contours,key=cv2.contourArea)
    area=cv2.contourArea(contour)
    if area<200 or area<.9*sum(cv2.contourArea(c) for c in contours):return None
    # A fine fur pattern or photographic highlight must not condition the painter.
    # Keep just a few connected internal strokes and a flat tone inside the outline.
    gray=cv2.cvtColor(np.asarray(working),cv2.COLOR_RGB2GRAY)
    softened=cv2.bilateralFilter(cv2.GaussianBlur(gray,(0,0),1.9),9,30,12)
    lines=cv2.Canny(softened,25,68)
    lines&=cv2.erode(silhouette,np.ones((5,5),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(lines,connectivity=8)
    lengths=np.maximum(stats[:,cv2.CC_STAT_WIDTH],stats[:,cv2.CC_STAT_HEIGHT])
    accepted=(lengths>=max(6,min(working.size)*.018))
    accepted[0]=False
    paper=np.full(silhouette.shape,247,np.uint8)
    inside=silhouette!=0
    volumes=cv2.GaussianBlur(softened,(0,0),max(2,min(working.size)*.012))
    cutoffs=np.quantile(volumes[inside],[.25,.65])
    paper[inside]=np.choose(np.digitize(volumes[inside],cutoffs),[198,212,224])
    dark_landmark=(gray<cv2.GaussianBlur(gray,(0,0),4).astype(np.int16)-30)&inside
    _,parts,areas,_=cv2.connectedComponentsWithStats(np.uint8(dark_landmark),connectivity=8)
    anchors=(areas[:,cv2.CC_STAT_AREA]>=max(6,round(area*.0001)))&(areas[:,cv2.CC_STAT_AREA]<area*.008)
    anchors[0]=False
    paper[anchors[parts]]=108
    paper[accepted[labels]]=115
    outline=cv2.approxPolyDP(contour,1.1,True)
    cv2.polylines(paper,[outline],True,65,2,lineType=cv2.LINE_AA)
    x,y,w,h=cv2.boundingRect(contour)
    if min(x,y,working.width-x-w,working.height-y-h)<1:return None
    pad=max(3,round(max(w,h)*.014))
    return Image.fromarray(paper).crop((max(0,x-pad),max(0,y-pad),min(working.width,x+w+pad),
                                       min(working.height,y+h+pad))).convert('RGB')
