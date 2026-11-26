"""Check for unsupported geometric overlays introduced during a print repair."""
import cv2
import numpy as np
from PIL import Image

def changed_clipped_field(source, candidate):
    """Locate a broad new near-black or white fill for visual review at the edit seam."""
    if source.size != candidate.size:
        raise ValueError('Unmatched crop dimensions')
    small=source.copy()
    small.thumbnail((512,512),Image.Resampling.BOX)
    a=np.asarray(small.convert('RGB'),dtype=np.int16)
    b=np.asarray(candidate.convert('RGB').resize(small.size,Image.Resampling.BOX),dtype=np.int16)
    changed=np.abs(a-b).max(axis=2)>36
    if min(small.size)<32 or not changed.any():
        return None
    minimum=max(70,round(a.shape[0]*a.shape[1]*.009))
    candidates=[]
    for tone,darkness,original in [('black',b.max(axis=2),a.max(axis=2)),
                                    ('white',255-b.min(axis=2),255-a.min(axis=2))]:
        extreme=((darkness<=8)&changed).astype(np.uint8)
        if extreme.sum()<minimum or np.mean(original<=8)>.035:
            continue
        n,labels,stats,_=cv2.connectedComponentsWithStats(extreme,connectivity=8)
        for label in range(1,n):
            x,y,w,h,area=map(int,stats[label])
            if area<minimum or min(w,h)<8:
                continue
            core=(labels[y:y+h,x:x+w]==label).astype(np.uint8)
            radius=cv2.distanceTransform(np.pad(core,1),cv2.DIST_L2,3).max()
            if radius>=max(3,min(small.size)*.023):
                bounds=[round(v*n/d) for v,n,d in zip((x,y,x+w,y+h),source.size*2,small.size*2)]
                candidates.append((area,{'tone':tone,'pixels':bounds}))
    return max(candidates,key=lambda item:item[0])[1] if candidates else None


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


def resolved_line_pair(low, high):
    """Locate two sustained same-polarity marks that blended at the lower scale."""
    import cv2
    import numpy as np
    if low.size!=high.size or min(high.size)<64:return None
    old,new=[np.asarray(im.convert('L'),dtype=np.float32) for im in (low,high)]
    best=None
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        tx,ty=-dy,dx
        def along(im,offset,x,y):
            return np.roll(im,(offset*y,offset*x),axis=(0,1))
        # Average along each stroke so isolated speckles do not count as separable lines.
        planes=[sum(along(im,k,tx,ty) for k in range(-6,7))/13 for im in (old,new)]
        for gap in (2,3,4,5):
            samples=[[along(plane,k*gap,dx,dy) for k in (-2,-1,0,1,2)] for plane in planes]
            for polarity in (-1,1):
                readings=[]
                for left,first,mid,second,right in samples:
                    contrast=np.minimum.reduce([(left-first)*polarity,(mid-first)*polarity,
                                                 (mid-second)*polarity,(right-second)*polarity])
                    readings.append(contrast)
                old_contrast,new_contrast=readings
                # Compare an actual alternating five-sample pattern, not the
                # peak gradient of one sharp edge or its sharpening halo.
                qualify=(new_contrast>=13)&(old_contrast>=-3)&(old_contrast<=7)&(new_contrast-old_contrast>=11)
                support=sum(along(qualify.astype(np.uint8),k,tx,ty) for k in range(-4,5))
                merit=np.where(support>=7,new_contrast-np.maximum(0,old_contrast),0)
                merit[:20]=0;merit[-20:]=0;merit[:,:20]=0;merit[:,-20:]=0
                _,peak,_,(x,y)=cv2.minMaxLoc(merit)
                if peak>0 and (best is None or peak>best['strength']):
                    best=dict(center=[x,y],normal=[dx,dy],line_separation_px=2*gap*(dx*dx+dy*dy)**.5,
                              polarity='dark' if polarity==1 else 'light',lower_contrast=round(float(old_contrast[y,x]),2),
                              print_contrast=round(float(new_contrast[y,x]),2),strength=round(peak,2))
    return best


def suppress_isolated_marks(source, candidate):
    """Interpolate isolated new flecks along the direction of an existing contour."""
    if source.size != candidate.size or min(source.size)<12:
        return None
    guide, print_pixels = [np.asarray(im.convert('RGB')) for im in (source, candidate)]
    gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, -1, 1, 0)
    dy = cv2.Sobel(gray, -1, 0, 1)
    xx, xy, yy = [cv2.GaussianBlur(field, (0,0), 2.5) for field in (dx*dx, dx*dy, dy*dy)]
    angle = .5*np.arctan2(2*xy, xx-yy)
    tangent_x, tangent_y = -np.sin(angle), np.cos(angle)
    y, x = np.indices(gray.shape, dtype=np.float32)
    along = (xx+yy>200) & (np.hypot(xx-yy,2*xy)>.55*(xx+yy))

    def expected_color(field):
        middle = cv2.medianBlur(field, 5)
        samples = [cv2.remap(field, x+tangent_x*offset, y+tangent_y*offset, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REFLECT_101) for offset in (-5,-4,-3,-2,2,3,4,5)]
        middle[along] = np.median(np.stack(samples),axis=0)[along]
        return middle

    predicted = expected_color(print_pixels)
    contrast = np.max(np.abs(print_pixels.astype(np.int16)-predicted),axis=2)
    source_contrast = np.max(np.abs(guide.astype(np.int16)-expected_color(guide)),axis=2)
    isolated = np.uint8((contrast>19) & (contrast>2*source_contrast+13))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(isolated, connectivity=8)
    eligible = ((stats[:,cv2.CC_STAT_AREA]<=55) & (stats[:,cv2.CC_STAT_WIDTH]<=12)
                & (stats[:,cv2.CC_STAT_HEIGHT]<=12))
    eligible[0] = False
    selected = eligible[labels]
    # Thin outlying strokes at the field boundary may continue beyond the sampled pixels.
    selected[:5] = selected[-5:] = False
    selected[:,:5] = selected[:,-5:] = False
    count = int(selected.sum())
    if not 8<=count<=selected.size*.08:
        return None
    corrected = print_pixels.copy()
    corrected[selected] = predicted[selected]
    return Image.fromarray(corrected), count
