"""Preserve neighbouring objects when a localized correction names its subject."""
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
