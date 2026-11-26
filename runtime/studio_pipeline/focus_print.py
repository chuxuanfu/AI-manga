"""Local 2x neural publication pass with independent native-pixel detail samples."""
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageFilter
from graphs import Graph
from illustration import TURBO

ENGINE=Path(__file__).resolve().parents[1]/'models/anime_superres'
PATCH=1152
PRINT_REVISION=5

def neural_double(source,destination):
    subprocess.run([str(ENGINE/'realesrgan-ncnn-vulkan'),'-i',str(source),'-o',str(destination),
                    '-m',str(ENGINE),'-n','realesr-animevideov3','-s','2','-t','256'],
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=600)
    with Image.open(source) as old, Image.open(destination) as new:
        if new.size!=tuple(v*2 for v in old.size):
            raise ValueError('神经超分结果尺寸错误')
        new.verify()

def centers(points,size):
    """Select distinct, in-frame crops from normalized visual-review coordinates."""
    width,height=size
    boxes=[]
    for name,x,y in points:
        l=min(max(round(width*x/1000-PATCH/2),0),width-PATCH)
        t=min(max(round(height*y/1000-PATCH/2),0),height-PATCH)
        if all(abs(l-a)>.65*PATCH or abs(t-b)>.65*PATCH for _,a,b in boxes):
            boxes.append((name,l,t))
    return boxes[:3]

def sampled_surfaces(source,size,focus):
    """Cover structured image regions between the semantic landmarks at print scale."""
    import cv2
    scaled=source.convert('L')
    scaled.thumbnail((768,768),Image.Resampling.LANCZOS)
    tone=cv2.GaussianBlur(np.asarray(scaled,np.float32),(0,0),1)
    dx=cv2.Sobel(tone,cv2.CV_32F,1,0,ksize=3)/8
    dy=cv2.Sobel(tone,cv2.CV_32F,0,1,ksize=3)/8
    edges=cv2.magnitude(dx,dy)
    def starts(length):
        span=length-PATCH
        if span<=0:return [0]
        return np.linspace(0,span,1+int(np.ceil(span/(PATCH-144)))).round().astype(int).tolist()
    result=[]
    for y in starts(size[1]):
        for x in starts(size[0]):
            if any(abs(x-l)<PATCH*.25 and abs(y-t)<PATCH*.25 for _,l,t in focus):continue
            l,r=[round(v*scaled.width/size[0]) for v in (x,x+PATCH)]
            t,b=[round(v*scaled.height/size[1]) for v in (y,y+PATCH)]
            patch=edges[t:b,l:r]
            if patch.size and np.mean(patch>3.0)>=.025:
                result.append(('surface',x,y))
    return result


def detail_graph(inputs,medium,seed,prefix,roles=None):
    graph=Graph('studio-native-focus-print')
    model=graph.add('UNETLoader','局部扩散',(0,0),['MODEL'],values={'unet_name':TURBO,'weight_dtype':'default'})
    clip=graph.add('CLIPLoader','画法编码',(0,220),['CLIP'],values={'clip_name':'qwen_3_06b_base.safetensors','type':'stable_diffusion','device':'default'})
    vae=graph.add('VAELoader','变分编解码器',(0,410),['VAE'],values={'vae_name':'qwen_image_vae.safetensors'})
    pos=graph.add('CLIPTextEncode','保持当前画材',(330,150),['CONDITIONING'],[('clip',(clip,0))],{'text':medium+'. Close detail of this illustration. Preserve the silhouettes and lighting. Resolve existing contours and material marks with economy. Keep unmarked surfaces calm.'})
    neg=graph.add('ConditioningZeroOut','空条件',(660,240),['CONDITIONING'],[('conditioning',(pos,0))])
    nodes=[]
    for i,name in enumerate(inputs):
        y=i*300
        role=roles[i] if roles else ('interaction','secondary_subject','support')[i%3]
        guidance={
            'animal':'Resolve the existing animal at close range, including tapered fur or feather groups, eye rims, muzzle and small anatomical transitions when visible. Interpret them in the specified medium.',
            'contact':'Follow only the hand, cuff, or boundary that is already visible. Preserve empty air, empty hands, silhouettes, depth of field and all areas of blur. Refine existing marks without adding anything to grasp.',
            'surface':'Work at the scale of the visible crop. Subdivide existing structures into fine, legible strokes; trace their taper, layering and junctions. Keep broad flat or out-of-focus areas quiet. Match the original medium.',
            'support':'Resolve the materials and layered edges actually present near the supporting surface. Follow the existing perspective and illumination.',
            'face':'Resolve the existing eye contours, locks of hair, and fabric folds, keeping the expression and skin smooth.',
            'secondary_subject':'Resolve characteristic material marks of the secondary subject already visible in this crop. Preserve its anatomy or construction.',
            'interaction':'Resolve the existing contact between forms with controlled lines, material edges and contact shadows. Preserve the exact posture.'
        }.get(role,'Resolve the visible local material and edge transitions with fine marks appropriate to this medium.')
        conditioned=graph.add('CLIPTextEncode','按局部内容用笔',(640,400+y),['CONDITIONING'],[('clip',(clip,0))],
                              {'text':medium+'. '+guidance+' Preserve this close-up composition, including blank areas and optical blur. Fine marks follow existing forms. No new objects or figures.'})
        steps=5
        strength=.34 if role in ('contact','interaction','face') else .44
        src=graph.add('LoadImage','原像素局部',(980,y),['IMAGE','MASK'],values={'image':name},widgets=[name,'image'])
        lat=graph.add('VAEEncode','编码',(1310,y),['LATENT'],[('pixels',(src,0)),('vae',(vae,0))])
        args=dict(seed=seed+i,steps=steps,cfg=1.0,sampler_name='euler',scheduler='simple',denoise=strength)
        sample=graph.add('KSampler','原位细化',(1630,y),['LATENT'],[('model',(model,0)),('positive',(conditioned,0)),('negative',(neg,0)),('latent_image',(lat,0))],
                         args,[seed+i,'fixed',steps,1.0,'euler','simple',strength])
        pic=graph.add('VAEDecode','解码',(1960,y),['IMAGE'],[('samples',(sample,0)),('vae',(vae,0))])
        nodes.append(str(graph.save('局部重绘',(2290,y),pic,prefix+'/detail-'+str(i))))
    return graph,nodes

def subject_coverage(base,paint,role):
    """Support sampled subjects with visible pixels in the source, including on textured walls."""
    import cv2
    from review import probabilities
    groups=(('the animal','an animal head','a furry animal','a bird or fish'),
            ('human hand and fingers','wrist and sleeve'),('a human face',))
    labels=[term for group in groups for term in group]
    source=probabilities(base,labels);sample=probabilities(paint,labels)
    coverage=np.ones(source.shape[1:],dtype=np.float32)
    offset=0
    for i,terms in enumerate(groups):
        a=source[offset:offset+len(terms)].max(axis=0)
        b=sample[offset:offset+len(terms)].max(axis=0)
        offset+=len(terms)
        # Compare silhouettes in both views for any newly created, disconnected subject.
        nearby=cv2.dilate((a>.19).astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
        new=((b>.46)&(nearby==0)).astype('uint8')
        count,labels_at,stats,_=cv2.connectedComponentsWithStats(new)
        flagged=stats[:,cv2.CC_STAT_AREA]>=42
        flagged[0]=False
        if count>1 and np.any(flagged):
            hold=cv2.dilate(flagged[labels_at].astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27)))
            coverage*=1-cv2.GaussianBlur(hold.astype(np.float32),(0,0),2)
        focused=(i==0 and role in ('animal','secondary_subject')) or (i==1 and role in ('contact','interaction')) or (i==2 and role in ('face','primary_subject'))
        if focused and a.max()>.56:
            in_subject=np.clip((a-.16)/.27,0,1)
            in_subject=cv2.dilate(in_subject,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            coverage*=cv2.GaussianBlur(in_subject,(0,0),1.4)
    return cv2.resize(coverage,base.size,interpolation=cv2.INTER_LINEAR)

def blend_detail(master,patch,left,top,role=None):
    # Transfer only sampled mid/high frequencies. Feather joins and protect ink edges.
    import cv2
    box=(left,top,left+patch.width,top+patch.height);base=master.crop(box)
    size=(patch.width//2,patch.height//2)
    def guide(im):
        v=cv2.resize(np.asarray(im.convert('L')),size,interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(v,(5,5),1.1)
    # Warp the sampled contours back onto the source before transferring their fine bands.
    motion=cv2.calcOpticalFlowFarneback(guide(base),guide(patch),None,.5,3,25,3,7,1.5,0)
    motion=cv2.resize(motion,(patch.width,patch.height),interpolation=cv2.INTER_LINEAR)*2
    motion=np.clip(cv2.GaussianBlur(motion,(9,9),2),-22,22)
    yy,xx=np.mgrid[:patch.height,:patch.width].astype('float32')
    patch=Image.fromarray(cv2.remap(np.asarray(patch),xx+motion[:,:,0],yy+motion[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101))
    def values(im):return np.asarray(im,dtype=np.float32)
    def blur(im,r):return values(im.filter(ImageFilter.GaussianBlur(r)))
    mid=blur(patch,7)-blur(base,7)
    delta=values(patch)-values(base)-.42*mid-.58*(blur(patch,30)-blur(base,30))
    gy,gx=np.gradient(values(base.convert('L')))
    edge=cv2.dilate(gx*gx+gy*gy,np.ones((7,7),np.uint8))
    protect=.24+.76/(1+edge/125)
    axis=np.minimum(np.arange(PATCH)+1,PATCH-np.arange(PATCH)).clip(0,72)/72
    # Estimate resolved local information on the source alone. In defocused water,
    # sky or skin a denoiser can invent a whole object inside an otherwise smooth patch.
    source_gray=values(base.convert('L'))
    high_band=np.abs(source_gray-cv2.GaussianBlur(source_gray,(0,0),4.0))
    texture=cv2.GaussianBlur(high_band,(0,0),9.0)
    confidence=np.clip((texture-1.05)/2.65,0,1)
    confidence=confidence*confidence*(3-2*confidence)
    # A lone outline in a smooth fill has a strong gradient but carries little
    # interior texture. Transfer only very fine marks there. Mid-size strokes
    # additionally need surrounding source texture and agreement on local tone.
    density=cv2.boxFilter((high_band>3.5).astype(np.float32),-1,(35,35))
    textured=np.clip((density-.10)/.34,0,1)
    coarse_difference=cv2.GaussianBlur(values(patch)-values(base),(0,0),4.0)
    tone_error=np.max(np.abs(coarse_difference),axis=2)
    agreement=1/(1+(tone_error/8)**2)
    local_signal=cv2.GaussianBlur(np.abs(gx)+np.abs(gy),(0,0),1.4)
    supported=np.clip((cv2.dilate(local_signal,np.ones((5,5),np.uint8))-1.2)/3,0,1)
    eligible=(textured**2*agreement*supported)[:,:,None]
    fine=delta-cv2.GaussianBlur(delta,(0,0),1.5)
    delta=eligible*delta+(1-eligible)*supported[:,:,None]*10*np.tanh(fine/10)
    # Painting grain can be rich even where no structural line exists. Suppress large
    # new contours there while allowing small fur lines and paper fibers to transfer.
    def coarse_edges(gray):
        smooth=cv2.GaussianBlur(gray,(0,0),4.)
        dx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)/8
        dy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)/8
        return cv2.magnitude(dx,dy)
    near=cv2.dilate(coarse_edges(source_gray),np.ones((19,19),np.uint8))
    candidate_edges=coarse_edges(values(patch.convert('L')))
    novel=((candidate_edges>.85)&(candidate_edges>2.65*near+.3)).astype('uint8')
    joined=cv2.dilate(novel,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(joined)
    pixels=np.bincount(labels.ravel(),weights=novel.ravel(),minlength=count)
    unsupported=(np.maximum(stats[:,cv2.CC_STAT_WIDTH],stats[:,cv2.CC_STAT_HEIGHT])>65)&(pixels>190)
    unsupported[0]=False
    wide=cv2.dilate(unsupported[labels].astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)))
    confidence*=1-cv2.GaussianBlur(wide.astype(np.float32),(0,0),6.)
    confidence*=subject_coverage(base,patch,role)
    alpha=(np.minimum.outer(axis,axis)*protect*confidence)[:,:,None]
    merged=np.clip(values(base)+alpha*61*np.tanh(delta/61),0,255).round().astype('uint8')
    master.paste(Image.fromarray(merged),(left,top))

def neural_weight(medium):
    medium=medium.casefold()
    if any(term in medium for term in ('watercolor','wash','pencil','pastel','chalk','graphite','charcoal','etched','etching','copperplate','woodblock','lithograph','silk','sanguine','wool','quilt','embroider','水彩','铅笔','木刻','水墨','粉彩','刺绣')):
        return .58
    if any(term in medium for term in ('cel animation','cel shading','pixel art','stained glass','cut-paper','screen-print','赛璐璐','剪纸','像素')):
        return .85
    return .78


def combine_source(source,neural,medium):
    base=source.convert('RGB').resize(neural.size,Image.Resampling.LANCZOS)
    scale_ratio=max(neural.size)/max(source.size)
    # Full-size originals supply every base pixel. Fade in neural SR as magnification approaches 2x.
    weight=neural_weight(medium)*min(1,max(0,scale_ratio-1))
    return Image.blend(base,neural.convert('RGB'),weight)


def finish_neural(small,neural,patches,target,medium=''):
    """Emit an atomic native master while retaining the source's fine tonal grain."""
    with Image.open(neural) as result, Image.open(small) as source:
        master=result.convert('RGB')
        master=combine_source(source,master,medium)
    for left,top,path,*role in patches:
        with Image.open(path) as patch:
            blend_detail(master,patch.convert('RGB'),left,top,role[0] if role else None)
    tmp=target.with_suffix('.partial.png')
    master.save(tmp,compress_level=3)
    with Image.open(tmp) as verified:
        if max(verified.size)!=7680:raise ValueError('8K 成图尺寸错误')
        verified.verify()
    tmp.replace(target)

def repair_print_pair(previous_small, previous_master, revised_master):
    """Carry just the native repair into the independently refined 4K print."""
    from PIL import ImageChops
    with Image.open(previous_small) as small, Image.open(previous_master) as old:
        if old.size != revised_master.size:
            raise ValueError('Local repair must preserve the master dimensions')
        four = small.convert('RGB')
        new = revised_master.convert('RGB')
        prior = old.convert('RGB')
        box = ImageChops.difference(prior,new).getbbox()
        if not box:
            return four
        sx,sy = prior.width/four.width, prior.height/four.height
        # The three-lobe Lanczos kernel can affect adjacent samples.
        import math
        area = (max(0,math.floor(box[0]/sx)-3), max(0,math.floor(box[1]/sy)-3),
                min(four.width,math.ceil(box[2]/sx)+3), min(four.height,math.ceil(box[3]/sy)+3))
        pixels = tuple(v*scale for v,scale in zip(area,(sx,sy,sx,sy)))
        size = area[2]-area[0],area[3]-area[1]
        def reduced(image):
            return np.asarray(image.resize(size,Image.Resampling.LANCZOS,box=pixels),dtype=np.int16)
        corrected = np.asarray(four.crop(area),dtype=np.int16) + reduced(new) - reduced(prior)
        four.paste(Image.fromarray(corrected.clip(0,255).astype(np.uint8)),area[:2])
        return four


def preserve_source_marks(reference, print_path):
    """Return aligned fine pigment that neural super-resolution removed inside forms."""
    import cv2
    with Image.open(reference) as source, Image.open(print_path) as printed:
        if max(source.size)>=max(printed.size):
            return print_path
        source.thumbnail((2304,2304),Image.Resampling.LANCZOS)
        observed=np.asarray(source.convert('RGB'),np.float32)
        reduced=np.asarray(printed.convert('RGB').resize(source.size,Image.Resampling.BOX),np.float32)
        residual=observed-reduced
        residual-=cv2.GaussianBlur(residual,(0,0),4)
        tone=cv2.cvtColor(observed,cv2.COLOR_RGB2GRAY)
        edge=cv2.magnitude(cv2.Sobel(tone,cv2.CV_32F,1,0)/8,cv2.Sobel(tone,cv2.CV_32F,0,1)/8)
        # A contour shifted by half a source pixel can leave a bright/dark echo.
        residual*=np.exp(-np.square(edge/27))[:,:,None]
        np.clip(residual,-32,32,out=residual)
        result=printed.convert('RGB')
        x=np.arange(printed.width,dtype=np.float32)
        x=(x+.5)*source.width/printed.width-.5
        for y in range(0,printed.height,256):
            end=min(y+256,printed.height)
            top=max(0,y-12)
            bottom=min(printed.height,end+12)
            strip=np.asarray(printed.crop((0,top,printed.width,bottom)).convert('RGB'),np.float32)
            gray=cv2.cvtColor(strip,cv2.COLOR_RGB2GRAY)
            guard=np.clip(1-np.abs(gray-cv2.GaussianBlur(gray,(0,0),3))/14,0,1)
            span=slice(y-top,end-top)
            rows=(np.arange(y,end,dtype=np.float32)+.5)*source.height/printed.height-.5
            xx,yy=np.meshgrid(x,rows)
            marks=cv2.remap(residual,xx,yy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            restored=strip[span]+.85*guard[span,:,None]*marks
            result.paste(Image.fromarray(restored.round().clip(0,255).astype('uint8')),(0,y))
    destination=print_path.with_name(print_path.stem+'-source-marks.png')
    temporary=destination.with_suffix('.partial.png')
    result.save(temporary,compress_level=3)
    temporary.replace(destination)
    return destination
