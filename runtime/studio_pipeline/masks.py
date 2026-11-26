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
