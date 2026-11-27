"""Find the most consequential visible problem before spending work on finishing details."""
import hashlib
import json

from PIL import Image, ImageOps

from confirmed_issues import confirmed_issue, ledger_version
from evidence import ask, backend, model_identity
from print_judge import MODEL, actionable_edit, map_box, store

VERSION = 'open-review-3'
VERIFIER = 'qwen3-vl:32b-instruct'
FIELDS = {
    'observation': {'type':'string'}, 'consequence': {'type':'string'},
    'severity': {'enum':['major','minor','clear','uncertain']},
    **{key:{'type':'integer','minimum':0,'maximum':1000} for key in ('left','top','right','bottom')},
}
SCHEMA = {'type':'object','properties':FIELDS,'required':list(FIELDS)}
DISCOVERY = ('Review this artwork as an independent illustrator deciding whether to release it. What, if '
             'anything, would make an ordinary viewer stop believing the picture on a second look? Find the '
             'most consequential visible problem of ANY kind. You can name an issue nobody anticipated. '
             'Start from the actual marks and their meaning in context. A plausible story for the marks is '
             'not evidence that the drawing communicates that story. Respect intended stylization, fantasy '
             'and deliberate impossibilities. Distinguish a visible ambiguity that hurts the image from '
             'a harmless choice. For major explain the concrete consequence. Bound the evidence in image 1 '
             'in coordinates 0..1000. Use clear if it reads naturally. Keep descriptions brief. Art direction: ')


def views(size):
    w,h=size
    yield 'whole',(0,0,w,h)
    columns=3 if w>h*1.2 else 2
    rows=3 if h>w*1.2 else 2
    for y in range(rows):
        for x in range(columns):
            rect=(max(0,round((x-.08)*w/columns)), max(0,round((y-.08)*h/rows)),
                  min(w,round((x+1.08)*w/columns)), min(h,round((y+1.08)*h/rows)))
            yield f'detail-{x}-{y}',rect


def verify(url,image,overview,brief,candidate):
    box=candidate['box']
    w,h=image.size
    l,t,r,b=[round(v*n/1000) for v,n in zip(box,(w,h,w,h))]
    padx=max(12,(r-l)//4)
    pady=max(12,(b-t)//4)
    field=image.crop((max(0,l-padx),max(0,t-pady),min(w,r+padx),min(h,b+pady)))
    detail=ImageOps.contain(field,(1050,1050),Image.Resampling.LANCZOS)
    fields={'observed':{'type':'string'},'ordinary_explanation':{'type':'string'},
            'deciding_visual_cue':{'type':'string'},
            'decision':{'enum':['substantiated','visibly_ambiguous','explained','insufficient']},'fix':{'type':'string'}}
    prompt=('Image 1 is the questioned region with its surroundings. Image 2 is the whole. An editor reported: '
            +candidate['observation']+'. Test both that reading and the strongest ordinary interpretation. '
            'Which specific mark makes one reading convincing to a viewer? If the mark genuinely supports both '
            'and the difference matters, choose visibly_ambiguous. If the field is wrong or insufficient, '
            'choose insufficient. Respect the intended medium. Offer a specific local correction only when '
            'warranted. Art direction: '+brief)
    return json.loads(ask(url,[detail,overview],prompt,{'type':'object','properties':fields,'required':list(fields)},tag=VERIFIER))


def review_image(image,brief,digest,root):
    identity=model_identity(MODEL)
    if not identity:
        raise RuntimeError('本地审图模型未安装')
    verifier_id=model_identity(VERIFIER)
    ledger=ledger_version()
    key=hashlib.sha256(json.dumps([VERSION,identity,verifier_id,digest,brief,ledger],ensure_ascii=False).encode()).hexdigest()
    directory=root/'visual-triage'
    directory.mkdir(exist_ok=True)
    receipt=directory/(key+'.json')
    if receipt.is_file():
        return json.loads(receipt.read_text())
    overview=ImageOps.contain(image,(1050,1050),Image.Resampling.LANCZOS)
    report=dict(version=VERSION,model_id=identity,verifier_id=verifier_id,ledger=ledger,source_sha256=digest,views=[],status='clear',issue=None)
    known=confirmed_issue(image,digest)
    if known:
        report.update(status='local',issue=known)
        store(receipt,report)
        return report
    with backend(MODEL) as url:
        for name,rect in views(image.size):
            print(f'本地盲审 {name}',flush=True)
            whole=name=='whole'
            view=overview if whole else ImageOps.contain(image.crop(rect),(960,960),Image.Resampling.LANCZOS)
            question=(DISCOVERY+brief if whole else
                      'Look only at this detail from an illustration. Ignore what is outside the crop. '
                      'Describe the most consequential visible problem of ANY kind that would bother a viewer. '
                      'What do the upper and lower or left and right portions actually depict, and do those '
                      'readings make sense together? Consider what the artist communicates without inventing '
                      'an unseen explanation. Intentional abstraction or a crop edge alone is fine. If there '
                      'is no meaningful problem, choose clear. For a possible ambiguity use uncertain. '
                      'Give brief visual evidence and bound it in coordinates 0..1000 on this image.')
            reading=json.loads(ask(url,[view],question,SCHEMA,tag=MODEL))
            reading.update(view=name,pixels=list(rect))
            report['views'].append(reading)
    rank={'major':0,'uncertain':1}
    candidates=sorted((view for view in report['views'] if view['severity'] in rank),key=lambda item:rank[item['severity']])
    if candidates:
        adjudicate(image,overview,brief,candidates,report)
    store(receipt,report)
    return report


def adjudicate(image,overview,brief,candidates,report):
    with backend(VERIFIER) as url:
        for candidate in candidates:
            box=map_box(candidate,candidate['pixels'],image.size)
            candidate['box']=box
            precise=bool(box and box[0]<box[2] and box[1]<box[3])
            if not precise:
                report.update(status='uncertain',issue=dict(target='visual coherence',operation='reinspect',
                              box=[0,0,1000,1000],symptom=candidate['observation'],fix='Locate the reported issue on the complete image.'))
                continue
            check=verify(url,image,overview,brief,candidate)
            candidate['verification']=check
            if check['decision']=='explained':
                continue
            localized=candidate['severity']=='major' and check['decision']=='substantiated' and actionable_edit(check['fix'])
            status='local' if localized else 'uncertain'
            if status=='local' or not report['issue']:
                report.update(status=status,issue=dict(target=candidate['observation'],box=box,
                              operation='redraw' if localized else 'reinspect',symptom=check['observed']+' '+check['deciding_visual_cue'],
                              fix=check['fix'] if localized else 'Verify the location and the visible contradiction.'))
            if localized or (report['issue'] and candidate['severity']=='uncertain'):
                break
