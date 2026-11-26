"""Detect possible paper gutters, then distinguish requested panels from scene geometry."""
import json
import re
import numpy as np
from PIL import Image
from evidence import ask,backend
from print_judge import MODEL


def gutters(im):
    overview=im.convert('RGB')
    overview.thumbnail((640,640),Image.Resampling.LANCZOS)
    pixels=np.asarray(overview)
    white=np.min(pixels,axis=2)>=239
    stripes=[]
    for name,view in [('vertical',white),('horizontal',white.T)]:
        span=view.shape[1]
        blank=view.mean(axis=0)>.97
        edges=np.flatnonzero(np.diff(np.r_[False,blank,False].astype(int)))
        for a,b in zip(edges[::2],edges[1::2]):
            if a<span*.15 or b>span*.85 or not span*.009<b-a<span*.16:continue
            margin=max(8,round(span*.09))
            if max(view[:,a-margin:a].mean(),view[:,b:b+margin].mean())>.7:continue
            stripes.append([name,round(a*1000/span),round(b*1000/span)])
    return overview,stripes


def unrequested_panels(im,stripes,brief):
    from scipy import ndimage
    pixels=np.asarray(im)
    ink=np.min(pixels,axis=2)<230
    joined=ndimage.binary_closing(ink,iterations=2)
    labels,_=ndimage.label(joined)
    panels=[]
    for index,rect in enumerate(ndimage.find_objects(labels),1):
        if not rect:continue
        y,x=rect
        width=(x.stop-x.start)/im.width;height=(y.stop-y.start)/im.height
        if width*height<.15 or np.mean(labels[rect]==index)<.70:continue
        panels.append((x.start/im.width,y.start/im.height,x.stop/im.width,y.stop/im.height))
    framed=False
    for axis,a,b in stripes:
        lo,hi=a/1000,b/1000
        for first in panels:
            for second in panels:
                l,t,r,bottom=first;u,v,s,z=second
                if axis=='horizontal':l,t,r,bottom,u,v,s,z=t,l,bottom,r,v,u,z,s
                if (r<=lo+.02 and u>=hi-.02 and min(bottom-t,z-v)>.70 and min(r-l,s-u)>.20
                        and max(abs(t-v),abs(bottom-z))<.075):
                    framed=True
    schema={'type':'object','properties':{'explicit':{'type':'boolean'},'evidence':{'type':'string'}},'required':['explicit','evidence']}
    with backend(MODEL) as url:
        intent=json.loads(ask(url,[],'Does the brief explicitly ask for several separate picture panels, comic-page frames, a diptych, or a comparison layout? Horizontal, landscape, 16:9, panoramic, watercolor and several subjects do not imply multiple frames. Cite the words requesting separate images, or say none. Brief: '+brief,schema,tag=MODEL))
        if intent['explicit']:return None
        if framed:return {'layout':'separate_panels','requested':False,'evidence':'纸白间隔两侧各有一块独立、边缘整齐且高度相近的画面。'}
        schema={'type':'object','properties':{'layout':{'enum':['continuous_scene','separate_panels']},'evidence':{'type':'string'}},'required':['layout','evidence']}
        question=('Examine the white strip across the paper. Does it divide two independently framed illustrations? A physical column in the depicted scene can be continuous. Gap positions: '+str(stripes))
        finding=json.loads(ask(url,[im],question,schema,tag=MODEL))
        return finding if finding['layout']=='separate_panels' else None


def retry_direction(note):
    """Describe how to stage a fresh composition without copying rejected content."""
    if not note or not note.strip():
        return ''
    # A defect report describes a discarded scene. Passing its nouns to the
    # diffusion encoder can introduce unrequested objects and reproduce the fault.
    issues = (
        (r'count|duplicat|surplus|extra|数量|重复|多余',
         'Arrange the requested subjects so their individual silhouettes can be distinguished.'),
        (r'connect|attach|overlap|crossing|load path|contact|grip|fused|limb|support|连接|交叠|接触|承托|肢体',
         'Compose the requested interaction from a viewpoint where the visible contours lead clearly '
         'into their attachments and the order of overlapping forms is legible.'),
        (r'blur|smeared|focus|stroke|模糊|失焦',
         'Resolve the boundaries at the intended focal point in the requested medium.'),
        (r'color|colour|palette|hue|light|色彩|颜色|光影',
         'Follow the specified local colors and light direction throughout the scene.'),
    )
    return ' '.join(direction for pattern, direction in issues if re.search(pattern, note, re.I))
