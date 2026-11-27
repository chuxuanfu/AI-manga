"""Write only the reviewed region, then verify the encoded PNG outside that region."""
import hashlib
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


def outside_digest(image, bounds):
    x, y, right, bottom = bounds
    width, height = image.size
    digest = hashlib.sha256()
    for area in ((0, 0, width, y), (0, y, x, bottom), (right, y, width, bottom),
                 (0, bottom, width, height)):
        if area[0] < area[2] and area[1] < area[3]:
            digest.update(image.crop(area).tobytes())
    return digest.hexdigest()


def confine_edit(source, proposal, box, destination):
    from print_judge import fingerprint, store
    with Image.open(source) as original, Image.open(proposal) as proposed:
        if original.size != proposed.size:
            raise ValueError('Local edits require matching canvas dimensions.')
        mode = original.mode if original.mode in ('RGB', 'RGBA', 'L', 'LA') else 'RGBA'
        base = original.convert(mode)
        width, height = base.size
        l,t,r,b = box
        if not (0 <= l < r <= 1000 and 0 <= t < b <= 1000):
            raise ValueError('Invalid edit region.')
        bounds = (math.floor(l*width/1000), math.floor(t*height/1000),
                  math.ceil(r*width/1000), math.ceil(b*height/1000))
        x,y,right,bottom = bounds
        w,h = right-x,bottom-y
        radius = min(16, max(1, min(w,h)//12))
        horizontal = np.ones(w) * radius
        vertical = np.ones(h) * radius
        if x:horizontal = np.minimum(horizontal,np.arange(w))
        if right < width:horizontal = np.minimum(horizontal,np.arange(w)[::-1])
        if y:vertical = np.minimum(vertical,np.arange(h))
        if bottom < height:vertical = np.minimum(vertical,np.arange(h)[::-1])
        weight = np.minimum(horizontal[None,:],vertical[:,None])/radius
        mask = Image.fromarray(np.rint(255*weight*weight*(3-2*weight)).astype(np.uint8))
        expected = outside_digest(base,bounds)
        base.paste(proposed.crop(bounds).convert(mode),(x,y),mask)
        with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.png',delete=False) as stream:
            temp = Path(stream.name)
        try:
            base.save(temp,icc_profile=original.info.get('icc_profile'))
            with Image.open(temp) as encoded:
                if outside_digest(encoded,bounds) != expected:
                    raise ValueError('Pixels outside the edit mask changed.')
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
    store(destination.with_suffix('.scope.json'),dict(source_sha256=fingerprint(source),
          result_sha256=fingerprint(destination),editable_pixels=list(bounds),
          feather_inside_pixels=radius,mode=mode,outside_sha256=expected))
    return destination
