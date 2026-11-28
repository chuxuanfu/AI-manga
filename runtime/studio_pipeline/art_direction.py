"""Whole-picture selection and a bounded hand-off for cosmetic print findings."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from counts import explicit_counts
from evidence import ask, backend, model_identity
from print_judge import MODEL

POLICY = 'whole-picture-v1'
REFERENCE = Path(__file__).resolve().parents[2] / 'assets/snow-platform-finish.webp'

DIRECTION = ('Give the environment a deliberate composition of its own. Connect the subject and setting '
             'through their light, color and reflected light. Vary the edges with the material and the focus. '
             'Allow quiet passages. Reserve the most vivid color and crispest marks for chosen accents. '
             'Respect the requested medium, lighting and palette, including deliberately graphic or vivid styles.')

def anime_finish(medium):
    """Use the studio's anime finish for an unspecified medium or a plain anime request."""
    name=' '.join((medium or '').strip().casefold().replace('-',' ').split())
    return name in {'','anime','anime style','anime illustration','anime style illustration',
                              'japanese anime','japanese anime style','japanese anime illustration',
                              '日系动漫','日系动漫插画','动漫','动漫插画'}


def schema(fields):
    return {'type':'object','properties':fields,'required':list(fields)}

TEXT = {'type':'string'}
WHOLE = schema({
    'cohesion':TEXT, 'focal_hierarchy':TEXT, 'surface_and_edges':TEXT,
    'brief_and_structure':{'enum':['sound','major_failure','uncertain']},
    'structure_evidence':TEXT, 'overall_finish':{'enum':['exhibition_ready','needs_art_direction','uncertain']},
    'preference':{'type':'integer','minimum':0,'maximum':100},
    'direction':TEXT,
})

def assess(path, brief):
    with Image.open(REFERENCE) as reference, Image.open(path) as candidate:
        images=[ImageOps.contain(im.convert('RGB'),(1152,1152),Image.Resampling.LANCZOS)
                for im in (reference,candidate)]
    question=('You are choosing a finished wallpaper. Image 1 is an example the art director loves. '
              'Assess ONLY image 2 against the brief. Use the first picture as a guide to artistic finish, '
              'never require its palette, scene, people, season or style. First look at both as whole paintings. '
              'Look at the background on its own, how light and color unite it with the focus, which edges '
              'disappear, and whether materials feel intentionally painted. A minimalist or vivid design can '
              'achieve this in its own medium. Name visible evidence in image 2 for each aesthetic field. '
              'Would you hang this image because it achieves that quality? An intact but generic, harsh or '
              'visually disjointed image can need art direction. Score artistic preference from 0 to 100 for '
              'ranking viable alternatives. Separately check required subjects, counts, action and major '
              'anatomical or physical impossibilities. Obvious extra limbs or fused figures are major. '
              'Ignore small retouchable imperfections when judging overall finish. Give one concise actionable '
              'direction for a rejected picture or an empty direction if ready. Under 18 words per field. Brief: '+brief)
    with backend(MODEL) as url:
        reading=json.loads(ask(url,images,question,WHOLE,tag=MODEL))
    return {'policy':POLICY,'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            'model_id':model_identity(MODEL),'reading':reading}

def ready(evaluation):
    row=evaluation.get('reading') or {}
    return (evaluation.get('policy')==POLICY and bool(evaluation.get('model_id'))
            and evaluation.get('reference_sha256')==hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
            and row.get('brief_and_structure')=='sound' and row.get('overall_finish')=='exhibition_ready'
            and all(isinstance(row.get(k),str) and row[k].strip()
                    for k in ('cohesion','focal_hierarchy','surface_and_edges','structure_evidence'))
            and type(row.get('preference')) is int and 0<=row['preference']<=100)

def assessment_result(evaluation,brief):
    row=evaluation['reading']
    passed=ready(evaluation)
    note=row['cohesion'] if passed else ' '.join(filter(None,[row['structure_evidence'],row['direction']]))
    return dict(status='pass' if passed else 'global',score=row['preference'],note=note,
                art_direction=evaluation,expected_counts=explicit_counts(brief),regions={},issues=[])

def retouch_note(path,brief,report):
    """Keep the accepted composition when the crop is cosmetic or still inconclusive."""
    issue=report.get('issue') or {}
    box=issue.get('box')
    if (report.get('status') not in ('local','uncertain') or report.get('overlay_check') or not issue.get('symptom')
            or issue.get('operation')=='remove_forbidden' or not isinstance(box,(list,tuple))
            or len(box)!=4 or not all(type(x) in (int,float) for x in box)):
        return None
    l,t,r,b=box
    if not (0<=l<r<=1000 and 0<=t<b<=1000):return None
    with Image.open(path) as im:
        art=im.convert('RGB')
        whole=ImageOps.contain(art,(1200,1200),Image.Resampling.LANCZOS)
        rect=tuple(round(n*s/1000) for n,s in zip(box,art.size*2))
        if rect[2]<=rect[0] or rect[3]<=rect[1]:return None
        detail=ImageOps.contain(art.crop(rect),(768,768),Image.Resampling.LANCZOS)
    fields=schema({'impact':{'enum':['cosmetic','major','uncertain']},'evidence':TEXT})
    question=('Judge the reported fault in the complete wallpaper (first) and the actual disputed area '
              '(second). Cosmetic means a small, bounded touch-up leaves the identity, action, anatomy, '
              'object function, composition and overall aesthetic intact. Major includes extra/missing '
              'subjects or limbs, broken physical relationships, explicit brief violations or widespread '
              'artifacts. A small box can contain a major fault. If ambiguous choose uncertain. '
              'Give the location and observation in concise Chinese for the retouching artist. Reported issue: '+issue['symptom']+' Brief: '+brief)
    from construction_review import LARGE
    reader=LARGE if model_identity(LARGE) else MODEL
    try:
        with backend(reader) as url:
            row=json.loads(ask(url,[whole,detail],question,fields,tag=reader))
        impact=row.get('impact')
        evidence=row.get('evidence')
        if impact not in ('cosmetic','uncertain','major') or not isinstance(evidence,str) or not evidence.strip():
            raise ValueError('局部判断缺少依据')
    except (OSError,RuntimeError,ValueError) as exc:
        impact,evidence='uncertain','局部影响尚未核实。'+str(exc)[:150]
    if impact=='major':
        return None
    return dict(box=list(box),impact=impact,finding=issue['symptom'],evidence=evidence,reader=reader)
