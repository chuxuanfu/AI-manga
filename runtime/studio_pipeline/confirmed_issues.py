"""Keep an editor's localized rejection attached to unchanged visual evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

FOLDER = Path(__file__).resolve().parents[1] / 'confirmed-issues'


def signature(image):
    thumb = image.convert('L').resize((9,8), Image.Resampling.LANCZOS)
    data = thumb.tobytes()
    value = 0
    for y in range(8):
        for x in range(8):
            i = y*9+x
            value = (value<<1) | (data[i]>data[i+1])
    return value


def evidence_region(image, box):
    l,t,r,b = box
    if not (0<=l<r<=1000 and 0<=t<b<=1000):
        raise ValueError('Expected a nonempty box in 0..1000 coordinates')
    w,h = image.size
    rect = (l*w//1000,t*h//1000,(r*w+999)//1000,(b*h+999)//1000)
    return image.crop(rect).convert('RGB')


def register(path, box, symptom, fix, folder=FOLDER):
    from print_judge import fingerprint, store
    if not symptom.strip() or not fix.strip():
        raise ValueError('Provide a visible symptom and a local correction.')
    with Image.open(path) as image:
        region = evidence_region(image,box)
        record = dict(source_sha256=fingerprint(path),size=list(image.size),signature=signature(image),
                      sample=region.resize((32,32),Image.Resampling.BOX).tobytes().hex(),
                      evidence_sha256=hashlib.sha256(region.tobytes()).hexdigest(),box=list(box),symptom=symptom,fix=fix)
    key = hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/(key+'.json')
    store(output,record)
    return output


def ledger_version(folder=FOLDER):
    # Entries are immutable and content-addressed. Include additions and removals in review cache keys.
    return hashlib.sha256(''.join(sorted(p.stem for p in folder.glob('*.json'))).encode()).hexdigest()


def confirmed_issue(image, digest, folder=FOLDER):
    image_signature=None
    for path in sorted(folder.glob('*.json')):
        row=json.loads(path.read_text())
        if list(image.size)!=row['size']:
            continue
        if digest!=row['source_sha256']:
            if image_signature is None:
                image_signature=signature(image)
            if (image_signature ^ row['signature']).bit_count()>8:
                continue
        region=evidence_region(image,row['box'])
        exact=hashlib.sha256(region.tobytes()).hexdigest()==row['evidence_sha256']
        if not exact:
            current=region.resize((32,32),Image.Resampling.BOX).tobytes()
            prior=bytes.fromhex(row['sample'])
            # Ignore visually imperceptible changes without declaring a real repair successful.
            exact=max(abs(a-b) for a,b in zip(current,prior))<=1
        if exact:
            return dict(operation='redraw',target='confirmed visual defect',box=row['box'],
                        symptom=row['symptom'],fix=row['fix'],receipt=path.name)
    return None


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Record a confirmed local defect on this computer.')
    parser.add_argument('image',type=Path)
    parser.add_argument('--box',nargs=4,type=int,required=True,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    parser.add_argument('--symptom',required=True)
    parser.add_argument('--fix',required=True)
    args=parser.parse_args()
    print(register(args.image,args.box,args.symptom,args.fix))
