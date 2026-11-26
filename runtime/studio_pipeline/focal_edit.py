"""Repaint a disputed area at print scale with a visual reference to its scene."""
import json

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from construction_review import LARGE
from evidence import ask, backend, model_identity
from illustration import installed_editor, instruction_edit
from print_judge import fingerprint, resize_only, store


def failed_focus(report):
    if not report.get('views'):
        return None
    last=report['views'][-1]
    issue=report.get('issue',{})
    if issue.get('category')=='unresolved_focus' and last['view'] in ('comparison','native','subject','previous_local'):
        return tuple(last['pixels'])
    # When a matched crop is ambiguous without any identified physical defect,
    # try to demonstrate actual added detail. An unsuccessful edit still fails.
    readings=[last[k] for k in ('observation','corroboration','independent_context') if last.get(k)]
    if (report.get('status')=='uncertain' and last['view']=='comparison' and issue.get('operation')=='none'
            and not issue.get('box') and readings and all(r.get('operation')=='none' for r in readings)):
        return tuple(last['pixels'])
    return None


def framing(size, pixels, *, magnified=False):
    width,height=size
    l,t,r,b=pixels
    if not (0<=l<r<=width and 0<=t<b<=height):raise ValueError('Invalid print coordinates')
    # Leave visible context around the target. The original pixels outside this
    # field, including the rest of a large print, do not enter the composite.
    side=min(width,height,1152)
    cap=min(768,side-32)
    if cap<64:return None
    if magnified:
        # A small subject occupying a few latent cells cannot acquire fine internal
        # marks. Spend the edit model's native canvas on that subject and its edge.
        side=min(side,max(192,round(min(cap,max(r-l,b-t))*1.4)))
    cx,cy=(l+r)//2,(t+b)//2
    x=min(max(0,cx-side//2),width-side);y=min(max(0,cy-side//2),height-side)
    l=max(l,cx-cap//2);r=min(r,cx+(cap+1)//2)
    t=max(t,cy-cap//2);b=min(b,cy+(cap+1)//2)
    return (x,y,x+side,y+side),(l-x,t-y,r-x,b-y)


def selection(size, target):
    l,t,r,b=target
    radius=max(2,min(20,min(r-l,b-t)//12))
    support=Image.new('L',size)
    ImageDraw.Draw(support).rectangle((l,t,r-1,b-1), fill=255)
    soft=support.filter(ImageFilter.GaussianBlur(radius))
    return Image.composite(soft,Image.new('L',size),support)


def describe_patch(url, crop, locator):
    fields={'subject':{'type':'string'},'material':{'type':'string'},'marks':{'type':'string'}}
    return json.loads(ask(url,[crop,locator],
        'Image 1 is a crop from within the pink rectangle in image 2. Name only the subject or surface actually '
        'visible INSIDE the crop, the medium and the physical marks that describe it. The rest of the artwork '
        'is solely for identification. Do not name figures or objects outside the crop. Use one short phrase per field.',
        {'type':'object','properties':fields,'required':list(fields)},tag=LARGE))


def describe_tiles(canvas, names, fields, patch_size, directory, path):
    key=[names,[list(v) for v in fields],model_identity(LARGE),'tile-subjects-1']
    cached=json.loads(path.read_text()) if path.is_file() else {}
    descriptions=cached.get('descriptions',[]) if cached.get('key')==key else []
    if len(descriptions)==len(names):return descriptions
    overview=canvas.copy();overview.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    with backend(LARGE) as url:
        for filename,(_,x,y) in zip(names[len(descriptions):],fields[len(descriptions):]):
            locator=overview.copy()
            area=(x,y,x+patch_size,y+patch_size)
            ImageDraw.Draw(locator).rectangle(tuple(round(v*n/d) for v,n,d in zip(area,overview.size*2,canvas.size*2)),
                                              outline='#ff0080',width=2)
            with Image.open(directory/filename) as sample:
                crop=sample.convert('RGB');crop.thumbnail((832,832),Image.Resampling.LANCZOS)
                observed=describe_patch(url,crop,locator)
            descriptions.append(' '.join(observed[k] for k in ('subject','material','marks')))
            store(path,{'key':key,'descriptions':descriptions})
    return descriptions


def print_constraints(brief):
    from print_judge import excluded_by_brief
    clauses=excluded_by_brief(brief)
    return ('Respect the author’s explicit exclusions: '+'. '.join(clauses)+'.') if clauses else ''


def print_subjects(source, edge, path, *, constraints=""):
    """Name actual subjects for overlapping print fields before the first diffusion pass."""
    from manga_materials import starts, TILE
    from hashlib import sha256
    size = tuple(round(v * edge / max(source.size)) for v in source.size)
    key = [sha256(source.tobytes()).hexdigest(),list(source.size),list(size),model_identity(LARGE),'field-atlas-2',constraints]
    saved = json.loads(path.read_text()) if path.is_file() else {}
    result = saved.get('subjects',{}) if saved.get('key') == key else {}
    points = [(x,y) for y in starts(0,size[1],size[1],TILE) for x in starts(0,size[0],size[0],TILE)]
    pending = [(x,y) for x,y in points if f'{x},{y}' not in result]
    if not pending:
        return result
    overview = source.copy()
    overview.thumbnail((768,768),Image.Resampling.LANCZOS)
    with backend(LARGE) as url:
        for offset in range(0,len(pending),4):
            batch = pending[offset:offset+4]
            patches = []
            names = []
            locator = overview.copy()
            draw = ImageDraw.Draw(locator)
            for i,(x,y) in enumerate(batch):
                bounds = (x,y,x+TILE,y+TILE)
                scaled = lambda dims: tuple(round(v*n/d) for v,n,d in zip(bounds,dims*2,size*2))
                piece = source.crop(scaled(source.size))
                piece.thumbnail((576,576),Image.Resampling.LANCZOS)
                patches.append(piece)
                box = scaled(overview.size)
                draw.rectangle(box,outline='#ff0080',width=2)
                draw.text((box[0]+4,box[1]+4),str(i+1),fill='white',stroke_width=1,stroke_fill='black')
                names.append('crop_'+str(i+1))
            schema = {'type':'object','properties':{name:{'type':'string'} for name in names},'required':names}
            readings = json.loads(ask(url,[*patches,locator],
                'The last image is the entire artwork, with crop locations marked. For each of the preceding '
                'images, name the particular visible subject, surface and existing marks within that crop, '
                'in the order shown. Use a short concrete phrase per crop. Avoid describing an object '
                'outside the actual crop. Describe symbols as actual shapes rather than guesses at lettering. '
                'Preserve smooth fills, deliberate softness and flat color. '+constraints,schema,tag=LARGE))
            for name,(x,y) in zip(names,batch):
                phrase = readings[name].strip()
                if not phrase:
                    raise ValueError('Empty print tile description')
                result[f'{x},{y}'] = phrase
            store(path,{'key':key,'subjects':result})
    return result


def ordered_detail(url, first, second):
    fields={'judgment':{'type':'string','enum':['A','B','equal','indeterminate']},'feature':{'type':'string'}}
    question=('Two matched crops of the same drawing, in order A then B. Compare one visible boundary and one internal mark. '
              'Choose A when A separates real smaller structures more clearly, B when B does, equal when both show the same '
              'detail, indeterminate when no visible cue distinguishes a winner from noise, halos or a style change. '
              'Look at what is drawn, especially tapered strokes or individual highlights. In feature name the structure '
              'and its location in less than 18 words. Use only the supplied images.')
    return json.loads(ask(url,[first,second],question,{'type':'object','properties':fields,'required':list(fields)},tag=LARGE))


def repaint_definition(master, pixels, medium, seed, folder, stage, port, prefix, **tools):
    for magnified in (False,True):
        suffix='-closeup' if magnified else ''
        candidate=_repaint_at_scale(master,pixels,medium,seed+int(magnified)*3877,folder,stage+suffix,port,
                                    prefix+suffix,magnified=magnified,**tools)
        if candidate:
            return candidate
    return None


def _repaint_at_scale(master, pixels, medium, seed, folder, stage, port, prefix, *, magnified, local_image, post_graph, output_image, models):
    editor=installed_editor(models)
    if not editor:return None
    with Image.open(master) as opened:original=opened.convert('RGB')
    layout=framing(original.size,pixels,magnified=magnified)
    if layout is None:return None
    bounds,target=layout
    base=original.crop(bounds)
    mask=selection(base.size,target)
    context=original.copy();context.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    locator=context.copy()
    mark=tuple(round(v*n/d) for v,n,d in zip(pixels,locator.size*2,original.size*2))
    ImageDraw.Draw(locator).rectangle(mark,outline='#ff0080',width=2)
    source_sha=fingerprint(master)
    receipt=folder/(stage+'-definition.json')
    destination=folder/(stage+'-refined.png')
    identity=[source_sha,list(pixels),medium,seed,editor,model_identity(LARGE),'focal-definition-2']
    if receipt.is_file():
        saved=json.loads(receipt.read_text())
        if saved.get('key')==identity:
            if saved.get('decision')=='hold':return None
            if destination.is_file() and saved.get('sha256')==fingerprint(destination):return destination
    with backend(LARGE) as url:
        caption=describe_patch(url,base.crop(target),locator)
        if magnified:
            details={'mark':{'type':'string'},'placement':{'type':'string'}}
            drawing=json.loads(ask(url,[base.crop(target),locator],
                'You are preparing a magnified detail drawing of the first crop. The boxed overview locates it. '
                'Specify one concrete finer structure a skilled illustrator could resolve within an existing '
                'feature of this crop. Give the physical direction, spacing or taper of the marks and exactly '
                'where they belong. Match the existing medium. Preserve smooth areas and silhouettes.',
                {'type':'object','properties':details,'required':list(details)},tag=LARGE))
            caption['detail_direction']=drawing['mark']+'. '+drawing['placement']
    direction=('Return the exact composition and framing of image 1. Image 2 shows the whole original illustration '
               'for context. Keep the boundaries and colors of image 1. Re-illustrate the selected surface at print '
               'scale with clearly separated fine marks that follow its existing physical forms. Visible subject: '
               +caption['subject']+'. Material and medium: '+caption['material']+'. Existing marks: '+caption['marks']
               +'. Requested medium: '+medium+'. Keep other subjects outside the crop. Preserve shadow direction and silhouette.')
    if magnified:
        direction+=' Draw the proposed marks only within each individual form. Each mark terminates before that '
        direction+='form’s boundary. Preserve the original background pixels. '+caption['detail_direction']
    work_size=(1152,1152) if magnified else base.size
    editable=base.resize(work_size,Image.Resampling.LANCZOS)
    editable.putalpha(ImageOps.invert(mask.resize(work_size,Image.Resampling.BILINEAR)))
    g=instruction_edit(direction,seed,local_image(editable),prefix,editor,masked=True,detail_reference=local_image(context))
    history=post_graph(g.api,port,folder,stage,g.ui())
    raw=output_image(history[str(len(g.nodes))])
    with Image.open(raw) as opened:
        paint=opened.convert('RGB')
        if paint.size!=work_size:raise ValueError('Local print edit changed the canvas size')
        if paint.size!=base.size:
            paint=paint.resize(base.size,Image.Resampling.LANCZOS)
    corrected=Image.composite(paint,base,mask)
    before=base.crop(target);after=corrected.crop(target)
    evidence={'key':identity,'caption':caption,'raw_sha256':fingerprint(raw),'field':list(bounds),'target':list(target),
              'signal':resize_only((before,after)),'decision':'hold'}
    from print_artifacts import unsupported_lines
    evidence['new_line_geometry']=unsupported_lines(before,after)
    if evidence['new_line_geometry']['hold'] or evidence['signal']['condition']!='distinct_signal':
        store(receipt,evidence);return None
    fields={'same_subject_and_pose':{'type':'boolean'},'same_medium_and_light':{'type':'boolean'},
            'introduced_defect':{'type':'boolean'},'marks_follow_surfaces':{'type':'boolean'},'trace':{'type':'string'}}
    with backend(LARGE) as url:
        check=json.loads(ask(url,[base,paint,locator],
            'Compare the ORIGINAL crop (1), raw editor output (2), and the scene locator (3). A small change to '
            'fine lines is desired. Trace an actual feature in both crops. Hold a changed species or person, '
            'new object, shifted pose, damaged contact, doubled silhouette, lettering, new seam, or a material '
            'or light change. Follow a NEW mark from its start to its end. A graphic overlay, texture crossing a '
            'silhouette, invented hatching across different materials, or rays radiating into formerly empty space '
            'cannot count as a mark belonging to the depicted surface. Subject before the edit: '+caption['subject']+'.',
            {'type':'object','properties':fields,'required':list(fields)},tag=LARGE))
        evidence['continuity']=check
        if (check['same_subject_and_pose'] is True and check['same_medium_and_light'] is True
                and check['introduced_defect'] is False and check['marks_follow_surfaces'] is True and check['trace'].strip()):
            votes=[ordered_detail(url,after,before),ordered_detail(url,before,after)]
            evidence['comparisons']=votes
            if [v['judgment'] for v in votes]==['A','B'] and all(v['feature'].strip() for v in votes):
                original.paste(corrected,bounds[:2])
                tmp=destination.with_suffix('.partial.png');original.save(tmp,compress_level=3);tmp.replace(destination)
                evidence.update(decision='improved',sha256=fingerprint(destination))
    store(receipt,evidence)
    return destination if evidence['decision']=='improved' else None
