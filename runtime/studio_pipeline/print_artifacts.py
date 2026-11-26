"""Check for unsupported geometric overlays introduced during a print repair."""
import cv2
import numpy as np
from PIL import Image

def unsupported_lines(source, candidate):
    """Find long newly drawn strokes crossing formerly quiet, unrelated image areas."""
    if source.size != candidate.size:
        raise ValueError('Unmatched crop dimensions')
    size=tuple(max(1,round(n*min(1.,640/max(source.size)))) for n in source.size)
    a,b=[np.asarray(im.convert('L').resize(size,Image.Resampling.LANCZOS),np.uint8) for im in (source,candidate)]
    span=min(a.shape)
    if span<96:return {'hold':False,'rays':0}
    original=cv2.Canny(a,38,85)
    distance=cv2.distanceTransform(255-original,cv2.DIST_L2,3)
    revised=cv2.Canny(b,45,105)
    af=a.astype(np.float32)
    smooth=cv2.blur(af,(9,9))
    std=np.sqrt(np.maximum(0,cv2.blur(af**2,(9,9))-smooth**2))
    # The candidate edge may straddle a one-pixel stroke. Look for its contrast
    # on either side without counting a shifted version of an existing contour.
    delta=cv2.dilate(cv2.absdiff(a,b),np.ones((3,3),np.uint8))
    novel=(distance>2.8)&(delta>22)
    quiet=novel&(std<14.)
    hints=(novel|cv2.dilate(quiet.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool))
    new_edges=np.where(hints,revised,0).astype(np.uint8)
    lines=cv2.HoughLinesP(new_edges,1,np.pi/360,max(12,span//25),minLineLength=max(24,int(span*.12)),maxLineGap=10)
    candidates=[]
    for line in ([] if lines is None else lines[:,0]):
        x,y,u,v=map(int,line);length=np.hypot(u-x,v-y)
        n=max(2,round(length))
        xs=np.linspace(x,u,n).round().astype(int);ys=np.linspace(y,v,n).round().astype(int)
        if np.mean(novel[ys,xs])<.30 or quiet[ys,xs].sum()<max(8,span*.023):continue
        direction=np.array([u-x,v-y],np.float32)/length
        normal=np.array([-direction[1],direction[0]])
        angle=(np.degrees(np.arctan2(direction[1],direction[0]))+180)%180
        candidates.append((angle,normal,float(normal @ [x,y]),line))
    # Require a family of distinct directions with a common origin. Individual
    # fine hairs or a long genuine railing are insufficient evidence of an overlay.
    if len(candidates)<8:
        return {'hold':False,'rays':0,'angle_bins':0,'unsupported_segments':len(candidates)}
    angles=np.array([item[0] for item in candidates])
    normals=np.array([item[1] for item in candidates])
    offsets=np.array([item[2] for item in candidates])
    widest=0;best=0
    for i,(angle,n1,d1,_) in enumerate(candidates[:120]):
        for a2,n2,d2,_ in candidates[i+1:120]:
            diff=abs(angle-a2)
            if min(diff,180-diff)<32:continue
            matrix=np.stack((n1,n2))
            center=np.linalg.solve(matrix,np.array([d1,d2]))
            if not (0<=center[0]<a.shape[1] and 0<=center[1]<a.shape[0]):continue
            passing=angles[np.abs(normals @ center-offsets)<max(4,span*.017)]
            bins=len(np.unique(passing.astype(int)//15))
            if (bins,len(passing))>(widest,best):widest,best=bins,len(passing)
    return {'hold':widest>=5 and best>=8,'rays':best,'angle_bins':widest,'unsupported_segments':len(candidates)}
