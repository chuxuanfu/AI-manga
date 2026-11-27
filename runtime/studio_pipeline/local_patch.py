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


def selection_mask(path, size):
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError('The selection and artwork must have the same canvas dimensions.')
        if 'A' in image.getbands() and image.getchannel('A').getextrema() != (255,255):
            mask = image.getchannel('A')
        else:
            mask = image.convert('L')
    if not mask.getbbox():
        raise ValueError('The selection is empty.')
    return mask


def mask_bounds(mask):
    l,t,r,b = mask.getbbox()
    w,h = mask.size
    return [l*1000//w,t*1000//h,(r*1000+w-1)//w,(b*1000+h-1)//h]


def merge_selection(source, proposal, mask, destination, *, protect_only=False):
    """Blend a proposal and verify every protected pixel, including holes in the mask."""
    from PIL import ImageChops
    from print_judge import fingerprint, store
    with Image.open(source) as original, Image.open(proposal) as change:
        if original.size != change.size or original.size != mask.size:
            raise ValueError('The selection, proposal and original must have matching dimensions.')
        mode = original.mode if original.mode in ('L','LA','RGB','RGBA') else 'RGBA'
        base = original.convert(mode)
        edited = change.convert(mode)
        support = mask.point(lambda n: 255 if n else 0)
        exterior = ImageChops.invert(support)

        def outside_changed(candidate):
            difference = ImageChops.difference(base, candidate)
            return any(ImageChops.multiply(band,exterior).getbbox() for band in difference.split())

        if protect_only and original.mode == change.mode and not outside_changed(edited):
            return proposal
        result=Image.composite(edited,base,support if protect_only else mask)
        with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.png',delete=False) as stream:
            temp=Path(stream.name)
        try:
            result.save(temp,icc_profile=original.info.get('icc_profile'))
            with Image.open(temp) as encoded:
                if outside_changed(encoded):
                    raise ValueError('Protected selection pixels changed during export.')
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
    store(destination.with_suffix('.scope.json'),dict(source_sha256=fingerprint(source),
          result_sha256=fingerprint(destination),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
          editable_pixels=mask.getbbox(),outside_changed_pixels=0))
    return destination
