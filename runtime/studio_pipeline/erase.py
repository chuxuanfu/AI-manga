"""Local masked removal for an independently located duplicate, subject to re-review."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFilter

MODEL=Path(__file__).resolve().parents[1]/'models/inpaint/anime-manga-big-lama.pt'

from model_slot import auxiliary

@auxiliary
def erase_region(image,mask):
    mask=mask.convert('L')
    bounds=mask.getbbox()
    if not bounds or mask.size!=image.size:raise ValueError('需要与原图对齐的非空遮罩')
    x0,y0,x1,y1=bounds
    extra=max(96,round(max(x1-x0,y1-y0)*.95))
    box=(max(0,x0-extra),max(0,y0-extra),min(image.width,x1+extra),min(image.height,y1+extra))
    crop=image.crop(box).convert('RGB');area=mask.crop(box)
    working=crop.copy();working.thumbnail((1008,1008),Image.Resampling.LANCZOS)
    w,h=working.size;pad=((0,(-h)%8),(0,(-w)%8))
    rgb=np.pad(np.asarray(working,dtype=np.float32)/255,(*pad,(0,0)),mode='reflect')
    binary=np.pad(np.asarray(area.resize((w,h),Image.Resampling.NEAREST))>0,pad,mode='edge')
    model=torch.jit.load(str(MODEL),map_location='cpu').eval()
    with torch.inference_mode():
        prediction=model(torch.from_numpy(rgb).permute(2,0,1)[None],torch.from_numpy(binary.astype(np.float32))[None,None])
    remade=prediction[0,:,:h,:w].permute(1,2,0).clamp(0,1).mul(255).round().byte().numpy()
    result=image.convert('RGB').copy()
    result.paste(Image.fromarray(remade).resize(crop.size,Image.Resampling.LANCZOS),box[:2],area)
    return result

def duplicate_box(review):
    locations=review.get('detector',[])
    for conflict in review.get('count_conflicts',[]):
        count=conflict['requested'];name=conflict['object']
        instances=[r for r in locations if r['name']==name]
        if conflict.get('observed')!=count+1 or len(instances)!=count+1:continue
        boxes=[r['box'] for r in instances]
        target=min(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        l,t,r,b=target;area=(r-l)*(b-t)
        if not 0<area<250000:continue
        if any(max(0,min(r,x[2])-max(l,x[0]))*max(0,min(b,x[3])-max(t,x[1]))>area*.18 for x in (j['box'] for j in locations) if x!=target):continue
        return target
    return None

def masked_candidate(im,box):
    l,t,r,b=[round(k*v/1000) for k,v in zip(box,im.size*2)]
    pad=max(3,round(.05*max(r-l,b-t)))
    mask=Image.new('L',im.size);ImageDraw.Draw(mask).rectangle((l-pad,t-pad,r+pad,b+pad),fill=255)
    return erase_region(im,mask.filter(ImageFilter.GaussianBlur(max(1,pad/3))))
