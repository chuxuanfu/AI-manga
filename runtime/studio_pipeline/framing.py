"""Remove a generated rectangular print margin when the author requested full bleed."""
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image


FULL_BLEED = re.compile(r'\bfull[- ]bleed\b|\bedge[- ]to[- ]edge\b|'
                        r'\b(?:without|no) (?:white |outer )?(?:margin|border)s?\b|满版|滿版|无白边|無白邊', re.I)
PRESERVE = re.compile(r'\b(?:not|never|avoid) (?:a |an )?(?:full[- ]bleed|edge[- ]to[- ]edge)\b|'
                      r'\b(?:with|keep|preserve|retain|include) (?:a |the |an )?(?:wide |thin |outer |white |paper )*'
                      r'(?:border|margin)\b|(?:保留|留出|带有|帶有)(?:画面|纸张|白色)?(?:白边|白邊|外框|页边)', re.I)


def paper_bounds(image):
    """Require a common outer color and an abrupt straight ink boundary on all four sides."""
    raster = np.asarray(image.convert('RGB'))
    h, w = raster.shape[:2]
    if min(h, w) < 160:
        return None
    reference = np.median([raster[0,0], raster[-1,0], raster[0,-1], raster[-1,-1]], axis=0)
    cuts = []
    for side in (raster, raster[::-1], raster.swapaxes(0,1), raster.swapaxes(0,1)[::-1]):
        depth, length = side.shape[:2]
        strip = side[:max(8, round(depth*.09)), length//10:length-length//10]
        ink = (np.abs(strip.astype(np.int16)-reference).max(axis=2)>18)
        coverage = ink.mean(axis=1)
        # A fading wash, broad empty sky, or isolated marks have no straight
        # ink-to-paper transition extending across most of this side.
        candidates = np.flatnonzero(coverage >= .52)
        if not len(candidates):
            return None
        cut = int(candidates[0])
        if (cut < 4 or cut+3 >= len(coverage) or coverage[:cut-3].max() > .04
                or np.median(coverage[cut:cut+4]) < .55):
            return None
        # Check the complete outer band as well. A projecting mark or subject
        # outside the central strip must survive even in an otherwise square print.
        outside = side[:cut-3]
        marked = np.abs(outside.astype(np.int16)-reference).max(axis=2)>23
        if marked.mean() > .004 or np.any(marked.mean(axis=0)>.40):
            return None
        cuts.append(cut+1)
    top,bottom,left,right = cuts
    return left,top,w-right,h-bottom


def prepare_full_bleed(path, brief):
    if not FULL_BLEED.search(brief) or PRESERVE.search(brief):
        return path
    with Image.open(path) as source:
        bounds = paper_bounds(source)
        if bounds is None:
            return path
        artwork = source.crop(bounds)
        # Keep the original aspect and use real source pixels for the draft review.
        # Large conflicting aspect changes go to the ordinary composition reviewer.
        scale = min(artwork.width/source.width, artwork.height/source.height)
        w,h = round(source.width*scale),round(source.height*scale)
        if w < .94*artwork.width or h < .94*artwork.height:
            return path
        x,y = (artwork.width-w)//2,(artwork.height-h)//2
        artwork = artwork.crop((x,y,x+w,y+h))
    output = path.with_name(path.stem+'-full-bleed.png')
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.png',delete=False) as stream:
        tmp = Path(stream.name)
    try:
        artwork.save(tmp,compress_level=3)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return output
