"""Local candidate selection, repeatable regional repair, and checked 4K/8K delivery."""
import argparse
import fcntl
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
from urllib import error, request
import uuid

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch

from deadlines import checkpoint, insufficient_repair_time, REPAIR_WINDOW_CLOSED, RETOUCH
from gallery import write_gallery, write_focus_proof
from graphs import UP, draft, finish
from illustration import pigment_edit, has_drawing_model, text_composition, DRAW_UNET, installed_editor, instruction_edit, detail_canvas, NATURAL_UP, continuous_tone, continuous_tone_double, KEY_VISUAL, ANIME_PAINT, anime_color_pass, BRUSHWORK, REFERENCE, reference_draft, independent_draft, text_draft, print_graph, paint_key_visual
from reference_review import compare
from structure import inspect_structure
from construction_review import review as review_construction
from review import audit, RUNTIME, CHECKS
from manga_materials.planner import plan_scene, requested_medium
from focus_print import PATCH, centers, sampled_surfaces, combine_source, detail_graph, finish_neural, neural_double, repair_print_pair, preserve_source_marks
from local_patch import confine_edit, mask_bounds, merge_selection, selection_mask
from masks import object_mask, structural_composite
from lettering import remove_reviewed_feature
from erase import duplicate_box, erase_region, masked_candidate, MODEL as ERASE_MODEL

STUDIO = RUNTIME.parent
OUTPUT = STUDIO / 'images'
INPUT = RUNTIME / 'ComfyUI/input'
Image.MAX_IMAGE_PIXELS = 180_000_000


def save(path, obj):
    from print_judge import store
    store(path,obj)
    if path.name=="review.json":write_gallery(path.parent,obj,STUDIO)


def record_redraw(limit,manifest,rec,stage):
    """Persist a correction before starting it so interrupted rounds keep their budget."""
    if limit is None:
        return
    attempts=rec.setdefault('redraw_attempts',[])
    if len(attempts)>=limit:
        rec['status']='未达标 · 已达到重绘上限'
        rec.setdefault('终审退回原因',f'已用完 {limit} 次修订机会')
        save(manifest,rec)
        print(rec['status'],flush=True)
        raise SystemExit(2)
    attempts.append(stage)
    save(manifest,rec)


def deliver_file(source, dest):
    if dest.exists() and dest.samefile(source):
        return
    temporary=dest.with_name(uuid.uuid4().hex+'.png')
    try:
        try:os.link(source,temporary)
        except OSError:shutil.copyfile(source,temporary)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


def local_image(im):
    # Stable content-addressed inputs support editor replay and parallel jobs.
    key = hashlib.sha256(im.mode.encode() + str(im.size).encode() + im.tobytes()).hexdigest()[:24]
    name = 'studio-' + key + '.png'
    if not (INPUT/name).exists():
        temp = INPUT / (uuid.uuid4().hex + '.png')
        im.save(temp)
        temp.replace(INPUT/name)
    return name


def post_graph(api, port, directory, stage, workflow=None):
    # Unattended jobs share one graphics device. Rendering a large diffusion
    # canvas while another process loads the print judge can exhaust unified RAM.
    request_data={'prompt':api}
    if workflow:request_data['extra_data']={'extra_pnginfo':{'workflow':workflow}}
    payload=json.dumps(request_data).encode()
    digest=hashlib.sha256(payload).hexdigest()
    cache=directory/(stage+'-task.json')
    if cache.is_file():
        old=json.loads(cache.read_text())
        if (old.get('digest')==digest and old.get('renderer')=='ComfyUI' and old.get('output_sha256')
                and output_fingerprints(old.get('outputs',{}))==old['output_sha256']):
            return old['outputs']
    # The executor owns the lease. Waiting clients must leave it available to
    # the worker that runs the graph, including direct node-editor submissions.
    if os.environ.get('STUDIO_SUBMITTED_AT'):
        scripts=str(STUDIO/'scripts')
        if scripts not in sys.path:sys.path.insert(0,scripts)
        from studio import server
        server(port)
    try:
        return _post_graph(payload,digest,port,directory,stage,None)
    except (error.URLError,ConnectionError) as exc:
        cause=getattr(exc,'reason',exc)
        if not isinstance(cause,(ConnectionRefusedError,ConnectionResetError,BrokenPipeError)):raise
        sys.path.insert(0,str(STUDIO/'scripts'))
        try:
            from studio import server
            server(port)
        finally:
            sys.path.pop(0)
        return _post_graph(payload,digest,port,directory,stage,None)


def output_fingerprints(outputs):
    paths={}
    for record in outputs.values():
        for image in record.get('images',[]):
            if image.get('type')!='output':return None
            path=(OUTPUT/image.get('subfolder','')/image['filename']).resolve()
            if not path.is_relative_to(OUTPUT.resolve()) or not path.is_file():return None
            with path.open('rb') as stream:
                paths[str(path.relative_to(OUTPUT.resolve()))]=hashlib.file_digest(stream,'sha256').hexdigest()
    return paths or None


def _post_graph(payload, digest, port, directory, stage, queue_seconds):
    cache = directory / (stage + '-task.json')
    url = 'http://127.0.0.1:' + str(port)
    job = json.loads(cache.read_text()) if cache.exists() else {}
    if job.get('digest') != digest:
        if (job.get('renderer')=='ComfyUI' and job.get('output_sha256')
                and output_fingerprints(job.get('outputs',{}))==job['output_sha256']):
            # Another local graph may reuse the stage name. Its older pictures
            # remain eligible for later assessment only with a matching receipt.
            archive=directory/'render-history'
            archive.mkdir(exist_ok=True)
            save(archive/(stage+'-'+job['digest']+'.json'),job)
        job = {}
    cached=job.get('outputs')
    if isinstance(cached,dict) and job.get('output_sha256'):
        if output_fingerprints(cached)==job['output_sha256']:return cached
        # History authenticates the original render only. Save the regenerated
        # graph to a fresh filename so an altered or missing file cannot be adopted.
        submission=json.loads(payload)
        suffix='-restored-'+uuid.uuid4().hex[:10]
        for node in submission['prompt'].values():
            fields=node['inputs']
            if node['class_type']=='SaveImage' and 'filename_prefix' in fields:fields['filename_prefix']+=suffix
        payload=json.dumps(submission).encode()
        job={}
    def get(path):
        with request.urlopen(url+path,timeout=30) as response:
            return json.load(response)
    if job:
        history = get('/history/'+job['id'])
        active = get('/queue')
        if job['id'] not in history and not any(row[1]==job['id'] for rows in active.values() for row in rows):
            job = {}
    if not job:
        submitted=time.time()
        req = request.Request(url+'/prompt', data=payload, headers={'Content-Type':'application/json'})
        try:
            with request.urlopen(req,timeout=45) as response:
                ident = json.load(response)['prompt_id']
        except error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
        job = {'id':ident,'digest':digest,'gpu_queue_seconds':queue_seconds,'submitted':submitted}
        save(cache,job)
    while True:
        hist = get('/history/'+job['id']).get(job['id'])
        if hist:
            if hist['status']['status_str']!='success':
                cache.unlink(missing_ok=True)
                raise RuntimeError(str(hist['status']['messages'][-2:])[:2300])
            if 'renderer' not in job:
                # Keep the files referenced by Comfy's executed graph, including
                # optional local editors actually selected on this device.
                executed=hist.get('prompt',[])
                if len(executed)>2 and isinstance(executed[2],dict):
                    weights={v for node in executed[2].values() for k,v in node['inputs'].items()
                             if k in ('ckpt_name','unet_name','clip_name','lora_name','vae_name','model_name')
                             and isinstance(v,str)}
                    job.update(renderer='ComfyUI',endpoint=url,model_files=sorted(weights))
            timestamps={name:data['timestamp'] for name,data in hist['status'].get('messages',[])
                        if isinstance(data,dict) and isinstance(data.get('timestamp'),(int,float))}
            if 'execution_start' in timestamps and job.get('submitted'):
                job['gpu_queue_seconds']=round(max(0,timestamps['execution_start']/1000-job['submitted']),2)
            if 'execution_start' in timestamps and 'execution_success' in timestamps:
                job['renderer_seconds']=round((timestamps['execution_success']-timestamps['execution_start'])/1000,2)
            hashes=output_fingerprints(hist['outputs'])
            if hashes:job.update(outputs=hist['outputs'],output_sha256=hashes)
            save(cache,job)
            return hist['outputs']
        time.sleep(4)


def release_idle_renderer(port):
    """Return idle render caches to unified memory before the large visual reviewer starts."""
    address=f'http://127.0.0.1:{port}'
    try:
        with request.urlopen(address+'/queue',timeout=3) as reply:queue=json.load(reply)
        if not queue.get('queue_running') and not queue.get('queue_pending'):
            data=json.dumps({'unload_models':True,'free_memory':True}).encode()
            req=request.Request(address+'/free',data=data,headers={'Content-Type':'application/json'})
            with request.urlopen(req,timeout=3):pass
    except (OSError,ValueError):
        pass  # Cache release is optional. The normal checks still run.


def output_image(record):
    image = record['images'][0]
    return OUTPUT / image['subfolder'] / image['filename']


def region(issue,assessment):
    def valid(box):
        return (isinstance(box,list) and len(box)==4 and all(isinstance(x,(int,float)) and 0<=x<=1000 for x in box)
                and box[2]-box[0]>=12 and box[3]-box[1]>=12 and (box[2]-box[0])*(box[3]-box[1])<650000)
    box = issue.get('box')
    if (issue.get('user_selection') is True and isinstance(box,list) and len(box)==4
            and all(type(v) is int for v in box) and 0<=box[0]<box[2]<=1000 and 0<=box[1]<box[3]<=1000):
        return box
    if not valid(box):
        box = assessment.get('regions',{}).get(str(issue.get('region','')),{}).get('box')
    return box if valid(box) else None



def draft_repair_issue(result):
    """A localized, actionable draft finding can be repaired before paying for a print."""
    from print_judge import actionable_edit
    if result.get('status') not in ('local','uncertain'):return None
    return next((issue for issue in result.get('issues',[]) if region(issue,result)
                 and actionable_edit(issue.get('fix')) and (result['status']=='local'
                 or issue.get('operation') in ('redraw','remove_forbidden'))),None)


def repaint(im, box, description, caption, seed, port, directory, stage, target=None, neighbours=(), reference_style=False, rebuild_shape=False, reconstruction_attempt=0, allow_specialist=True, design_reference=None, allow_control=True):
    # Re-sample only the interior; the wider square gives diffusion scene context.
    original_request=(box,description,target)
    if reference_style and rebuild_shape:
        from construction_review import plan_local_repair
        box,target,description=plan_local_repair(im,box,target,description,requested_medium(caption) or '',directory,stage,caption)
    w,h = im.size
    x0,y0,x1,y1 = [round(v*n/1000) for v,n in zip(box,(w,h,w,h))]
    span = max(x1-x0,y1-y0)
    side = min(max(w,h), max(192, round(span*1.65)))
    cw,ch = min(side,w),min(side,h)
    x = min(max(0,(x0+x1-cw)//2),w-cw)
    y = min(max(0,(y0+y1-ch)//2),h-ch)
    crop = im.crop((x,y,x+cw,y+ch))
    size = tuple(max(64,round(n*960/max(cw,ch)/8)*8) for n in (cw,ch))
    mask = Image.new('L',(cw,ch))
    margin = max(2,round(span*.035))
    rect = (x0-x-margin,y0-y-margin,x1-x+margin,y1-y+margin)
    ImageDraw.Draw(mask).rounded_rectangle(rect,radius=max(2,round(span*.05)),fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(2,round(span*.022))))
    if reference_style:
        size=tuple(max(64,round(n/16)*16) for n in size)
    prepared = crop.resize(size,Image.Resampling.LANCZOS)
    prepared.putalpha(ImageOps.invert(mask.resize(size,Image.Resampling.BILINEAR)))
    prefix = (directory.relative_to(OUTPUT)/stage).as_posix()
    model_root=RUNTIME/'ComfyUI/models'
    recolor=pigment_edit(original_request[1])
    from illustration import construction_guide
    guide=construction_guide(original_request[1],attempt=reconstruction_attempt) if reference_style and rebuild_shape else None
    scene_depth='whole scene as a depth reference' in original_request[1].casefold()
    specialist=(installed_editor(model_root) if allow_specialist and reference_style and rebuild_shape
                and (design_reference or guide or scene_depth or recolor or reconstruction_attempt>=2) else None)
    # Use an explicit drawing of the corrected geometry only when it can be placed
    # in this very crop. The line map must not perpetuate the rejected silhouette.
    from registered_edit import CONTROL, controlled_drawing, drawing_inputs, registered_design
    control_available=allow_control and (model_root/'controlnet'/CONTROL).is_file()
    registered=None
    if control_available and rebuild_shape and reference_style and not recolor:
        if design_reference:
            with Image.open(design_reference) as design:
                rect=tuple(round(v*n/m) for v,n,m in zip((x,y,x+cw,y+ch),design.size*2,(w,h,w,h)))
                prior=design.convert('RGB').crop(rect).resize(crop.size,Image.Resampling.LANCZOS)
            registered=registered_design(crop,prior,mask)
        elif guide and guide[1].get('placement')=='supported_object' and reconstruction_attempt==1 and ERASE_MODEL.is_file():
            from masks import frame_supported_object
            with Image.open(guide[0]) as photo:
                placed,extent=frame_supported_object(photo,crop.size,(x0-x,y0-y,x1-x,y1-y))
            visible=mask.point(lambda v:255 if v>=128 else 0)
            if visible.getbbox():
                layout=erase_region(crop,visible)
                layout.paste(placed,(0,0),placed.getchannel('A'))
                registered=(layout,dict(asset=guide[0].name,extent=list(extent)))
    native_field=registered[0] if registered else None
    if native_field is None and reference_style and rebuild_shape and reconstruction_attempt==1:
        native_field=crop
    if allow_control and native_field is not None:
        from apple_paint import paint_local_field
        native=paint_local_field(native_field,description+'. '+caption,seed,directory,stage,
                                 strength=.28 if registered else (.41 if recolor else .54))
        if native is not None:
            patched=im.copy()
            patched.paste(Image.composite(native,crop,mask),(x,y))
            path=directory/(stage+'-composite.png')
            patched.save(path)
            save(directory/(stage+'-reference.json'),dict(model='coreml-sd15',registered=registered[1] if registered else None))
            return path
    if registered:
        layout,metadata=registered
        canvas,edges=drawing_inputs(layout,mask,size)
        from graphs import BASE
        checkpoint=ANIME_PAINT if (RUNTIME/'models/checkpoints'/ANIME_PAINT).is_file() and KEY_VISUAL in caption else BASE
        edit=controlled_drawing(local_image(canvas),local_image(edges),description+'. '+caption,seed,prefix,checkpoint)
        try:
            result=post_graph(edit.api,port,directory,stage+'-controlled',edit.ui())
        except RuntimeError as failure:
            save(directory/(stage+'-control-fallback.json'),dict(model=CONTROL,reason=str(failure)[:1000]))
        else:
            with Image.open(output_image(result[str(len(edit.nodes))])) as painted:
                part=painted.convert('RGB').resize(crop.size,Image.Resampling.LANCZOS)
            revised=im.copy()
            revised.paste(structural_composite(crop,part,mask,(x==0,y==0,x+cw==w,y+ch==h)),(x,y))
            path=directory/(stage+'-composite.png')
            revised.save(path)
            save(directory/(stage+'-reference.json'),dict(model=CONTROL,registered=metadata))
            return path
    if specialist:
        if reconstruction_attempt>=3 and not recolor and ERASE_MODEL.is_file():
            # A masked editor can still copy a malformed silhouette from its visual
            # conditioning. Give it the same surroundings on a locally cleared plate.
            clear=mask.point(lambda value:255 if value>=160 else 0)
            if clear.getbbox():
                plate=erase_region(crop,clear).resize(size,Image.Resampling.LANCZOS)
                plate.putalpha(prepared.getchannel('A'))
                prepared=plate
        source=local_image(prepared)
        direction=('Repaint the pigment and pattern of the specified surface. Keep its silhouette, pose, contact shadows '
                   'and linework. Make the requested colors and pattern clearly visible over the intended surface. '
                   if recolor else 'Reconstruct the specified part and its visible connections. ')
        reference=None
        scene_reference=None
        reference_record={}
        if design_reference:
            with Image.open(design_reference) as design:
                rect=tuple(round(v*n/m) for v,n,m in zip((x,y,x+cw,y+ch),design.size*2,(w,h,w,h)))
                sample=design.convert('RGB').crop(rect).resize(size,Image.Resampling.LANCZOS)
            reference=local_image(sample)
            reference_record={'approved_design_crop':reference,'source_file':str(design_reference)}
            direction+=('Image 2 shows this same area in the earlier selected design. Restore the '
                        'physical feature from this corresponding area. Keep the new picture direction '
                        'of light, shading, and the intact surroundings from image 1. ')
        elif guide:
            file,credit=guide
            framing=None
            with Image.open(file) as photo:
                if credit.get('placement')=='supported_object':
                    from masks import frame_supported_object
                    placed,extent=frame_supported_object(photo,crop.size,(x0-x,y0-y,x1-x,y1-y))
                    guide_image=(prepared.convert('RGB').resize(crop.size,Image.Resampling.LANCZOS)
                                 if reconstruction_attempt>=3 and ERASE_MODEL.is_file()
                                 else Image.new('RGB',crop.size,(232,228,217)))
                    guide_image.paste(placed,mask=placed.getchannel('A'))
                    framing={'pixels':list(extent),'crop_size':list(crop.size)}
                    direction+=('Image 2 places the object in the selected field. Follow its physical '
                                'construction at this scale, resting on the existing support. Orient it '
                                'as requested and render all surfaces in the original medium. ')
                else:
                    guide_image=photo.convert('RGB')
            plate_subject=credit.get('neutral_plate_subject')
            if credit.get('pose_subject') and not recolor:
                from masks import pose_outline
                guide_image=pose_outline(guide_image,credit['pose_subject'])
                plate_subject=credit['pose_subject']
            if guide_image is not None:
                if plate_subject:
                    from masks import frame_neutral_guide
                    guide_image,framing=frame_neutral_guide(guide_image,crop,plate_subject)
                    direction+=('Use the neutral outline only for the silhouette and placement of joints. '
                                'Render all interior marks, colors and shading in the original illustration medium. ')
                reference=local_image(guide_image)
                reference_record={**credit,'input':reference}
                if framing:
                    reference_record['source_part_extent']=framing
                    if plate_subject:
                        direction+=('The neutral plate is registered to the source crop. Draw the corrected '
                                    'part at that size and location relative to the surrounding anatomy. ')
            construction=credit['construction']
            direction+=(('Image 2 shows only a physical relationship: ' if reference else 'Follow this construction: ')+construction+
                        ' Orient that relationship to the perspective and physical contact in image 1. '
                        'Keep the surrounding surfaces, scale, graphic style, and palette in image 1. ')
        if scene_depth or (not recolor and reconstruction_attempt>=2 and cw*ch<.85*w*h):
            overview=im.copy()
            overview.thumbnail((880,880),Image.Resampling.LANCZOS)
            scene_reference=local_image(overview)
            slot=3 if reference else 2
            reference_record.update(whole_scene=scene_reference,scene_input=slot,source_box=box)
            direction+=(f'Image {slot} shows this same drawing in full. The edited area of image 1 '
                        f'corresponds to {box} in whole-scene coordinates 0..1000. Find the existing '
                        'subject and trace each continuous body, instrument, rope or support beyond '
                        'the crop. Join the corrected end to that same object at its existing angle. '
                        'Keep the total number of subjects and implements required by the scene. '
                        'Use image 1 for scale, color and texture. ')
        if reference_record:save(directory/(stage+'-reference.json'),reference_record)
        edit=instruction_edit('Edit this same illustration close-up. '+direction+
                              'Preserve the surrounding objects, light and medium. Target: '+
                              str(target or 'the selected area')+'. '+description,seed,source,prefix,specialist,masked=True,
                              detail_reference=reference,scene_reference=scene_reference,full=reconstruction_attempt>=4)
        graph=edit.api
        output=str(len(edit.nodes))
    elif reference_style:
        guidance=crop
        text=description
        if rebuild_shape:
            # Preserve the surroundings, attachment and light while clearing the faulty drawing.
            subject=object_mask(crop,crop,target,mask,neighbours,confirmed_shape=True)
            if reconstruction_attempt and ERASE_MODEL.is_file():
                # Remove the previous silhouette from conditioning when diffusion has
                # already copied its defect. Retain the outer context and attachment.
                clear=subject if reconstruction_attempt==1 else mask
                # LaMa receives an explicit binary support, without Gaussian tails.
                clear=clear.point(lambda pixel: 255 if pixel>=64 else 0)
                if clear.getbbox():
                    guidance=erase_region(crop,clear)
                placeholder='locally cleared area'
            else:
                guidance=Image.composite(crop.filter(ImageFilter.GaussianBlur(max(5,span*.11))),crop,subject)
                placeholder='soft placeholder'
            named=target if target and target.casefold() not in ('local','global','none') else 'subject'
            text=('Draw a complete '+named+' in the '+placeholder+
                  '. Reconstruct its silhouette, internal parts and attachment in the local pose; '
                  'match the tapered contours, local colors, material and shadows of its surroundings. '+description)
        layout=local_image(guidance.resize(size,Image.Resampling.LANCZOS))
        edit=reference_draft(text,seed,*size,prefix,reference=layout,edit=True,
                             masked_source=local_image(prepared) if rebuild_shape else None,
                             preserve_layout=rebuild_shape and reconstruction_attempt==0)
        graph=edit.api
        output=str(len(edit.nodes))
    else:
        graph = draft(description+'. '+caption,seed,*size,prefix).api
        graph['4'] = {'class_type':'LoadImage','inputs':{'image':local_image(prepared)}}
        graph['8'] = {'class_type':'VAEEncode','inputs':{'vae':['1',2],'pixels':['4',0]}}
        graph['9'] = {'class_type':'SetLatentNoiseMask','inputs':{'samples':['8',0],'mask':['4',1]}}
        graph['5']['inputs'].update(latent_image=['9',0],steps=24,denoise=.81,cfg=5.5,sampler_name='dpmpp_2m')
        output='7'
    if specialist:
        try:
            hist=post_graph(graph,port,directory,stage)
        except RuntimeError as error:
            # The baseline editor can still run on a device/operator combination
            # that rejects the specialist graph. Its result receives the same review.
            save(directory/(stage+'-editor-fallback.json'),{'model':specialist,'reason':str(error)[:1500]})
            original_box,original_description,original_target=original_request
            return repaint(im,original_box,original_description,caption,seed,port,directory,stage,
                           original_target,neighbours,reference_style,rebuild_shape,reconstruction_attempt,False,design_reference,False)
    else:
        hist=post_graph(graph,port,directory,stage)
    with Image.open(output_image(hist[output])) as new:
        patched = im.copy()
        proposed=new.convert('RGB').resize((cw,ch),Image.Resampling.LANCZOS)
        if rebuild_shape and reference_style:
            borders=(x==0,y==0,x+cw==w,y+ch==h)
            patched.paste(structural_composite(crop,proposed,mask,borders),(x,y))
        else:
            patched.paste(proposed,(x,y),object_mask(crop,proposed,target,mask,neighbours))
    target = directory/(stage+'-composite.png')
    patched.save(target)
    return target


def repair_placement(before,plan,brief,seed,port,directory,stage):
    """Restore exposed surfaces and retain the registered subject at its new site."""
    from placement import reposition
    placed=reposition(before,plan)
    if not placed:return None
    candidate,regions=placed
    path=directory/(stage+'-placement.png')
    candidate.save(path)
    save(directory/(stage+'-placement.json'),dict(plan=plan,regions=regions))
    # The masked LaMa fill already restored the exposed area. Preserve the
    # original support geometry and the moved subject for native inspection.
    # A further defect at the old site is repaired from its own localized report.
    return path,regions['inspection']


def contact_layout_graph(picture, finding, brief, seed, prefix, *, full=False):
    """Direct a new composition using two independently located endpoints of a failed action."""
    editor=installed_editor(RUNTIME/'ComfyUI/models')
    quote=finding.get('requirement','')
    names=finding.get('items',[])
    if (not editor or finding.get('confirmed') is not True or finding.get('minimum_gap',0)<=145
            or not quote or quote.casefold() not in brief.casefold() or len(names)!=2
            or any(not isinstance(n,str) or not n or n.casefold() not in quote.casefold() for n in names)):
        return None
    geometry=[v for v in finding.get('observations',[]) if v.get('relation')=='separated']
    if not geometry:return None
    sites=[geometry[0].get(key) for key in ('actor_edge','target_edge')]
    if not all(isinstance(pt,(tuple,list)) and len(pt)==2 and
               all(type(v) is int and 0<v<1000 for v in pt) for pt in sites):return None
    base=picture.convert('RGB')
    base.thumbnail((1440,1440),Image.Resampling.LANCZOS)
    locator=base.copy()
    pen=ImageDraw.Draw(locator)
    radius=max(7,round(min(locator.size)*.018))
    for (nx,ny),color in zip(sites,('#d121ae','#24aeaf')):
        x,y=round(nx*locator.width/1000),round(ny*locator.height/1000)
        pen.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=max(3,radius//4))
    direction=('Create one finished illustration. Image 1 is the unmarked earlier composition. '
               'Image 2 locates the existing working end of '+names[0]+' with a magenta circle and '
               'the actual working surface of '+names[1]+' with a teal circle. Restage the pose so the '
               'acting subject faces and reaches that surface. Move the working end into direct contact '
               'with it. Maintain the natural scale of the tool, the reach of the body and the support '
               'of both objects. Keep all requested subject identities, handedness, count, lighting and '
               'medium. Paint the finished artwork free of locator marks. Required action: '+quote+
               '. Original brief: '+brief)
    return instruction_edit(direction,seed,local_image(base),prefix,editor,detail_reference=local_image(locator),full=full)


def dimensions(prompt):
    if '3:2' in prompt.replace('：',':'):return (1248,832)
    if any(s in prompt.lower() for s in ('16:9','16：9','横屏','横幅','landscape format','landscape orientation','widescreen')):
        return (1280,720)
    if any(s in prompt.lower() for s in ('9:16','9：16','手机屏','phone wallpaper')):
        return (720,1280)
    if any(s in prompt.lower() for s in ('1:1','方形','square format','square composition')):
        return (1024,1024)
    return (832,1248)


def main(resources):
    parser = argparse.ArgumentParser(description='本地漫画壁纸选片、纠错、精绘')
    parser.add_argument('-p','--prompt',help='中文或英文画面描述')
    parser.add_argument('-o','--output-dir',type=Path,help='将选中的校样或通过终审的成品复制到此目录')
    parser.add_argument('--aspect',choices=('auto','2:3','3:2','9:16','16:9','1:1'),default='auto',help='画幅比例，默认从描述判断，未指定时为 2:3')
    parser.add_argument('--print-method',choices=('auto','restoration','tiled'),default='auto',
                        help='4K 初次精绘方式。自动模式在支持的设备上优先从底图用 SeedVR2 重建')
    parser.add_argument('--base-model',choices=('auto','qwen-image-2512','qwen-image-2512-full'),default='auto',help='纯文字底图绘制模型，默认自动搭配候选')
    parser.add_argument('--style',help='可选画材或画风，例如透明水彩')
    parser.add_argument('--accept-final',metavar='SHA256',help='目检后，按当前母版指纹标记定稿')
    parser.add_argument('--skip-focus',nargs='*',type=int,choices=(1,2,3),default=None,help='对照后撤回指定编号的8K局部采样；不带编号可全部恢复。')
    parser.add_argument('--8k',dest='eight_k',action='store_true',help='在默认 4K 成品之外生成并审查 8K')
    parser.add_argument('--full-8k-diffusion',action='store_true',help='4K、8K 两级均进行全幅重叠分块扩散')
    parser.add_argument('--deep-repair',action='store_true',help='允许自动返工升级为慢速的完整 Qwen 采样')
    parser.add_argument('--native-4k',action='store_true',help='直接在长边 3840 的画布采样，耗时较长')
    parser.add_argument('--native-only',action='store_true',help='生成并保留各张原生底图，供原尺寸查看')
    parser.add_argument('--proof-only',action='store_true',help='先选片纠错并保存校样，稍后可继续精绘')
    parser.add_argument('--job',type=Path,help='继续已有任务目录')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--candidate-image',type=Path,help='已经重绘、与底图同尺寸的备选图')
    parser.add_argument('--selection-mask',type=Path,help='灰度或透明通道遮罩，白色或不透明表示采用备选')
    parser.add_argument('--keep-native',action='store_true',help='对已有长边 7680 的母版返修，保留选区外的原像素')
    parser.add_argument('--from-image',type=Path,action='append',default=[],help='先检查已有底图，可重复')
    parser.add_argument('--candidates',type=int,choices=range(1,9),default=None)
    parser.add_argument('--max-rounds',type=int,default=0,help='完整出片的最大轮数，默认 0 表示在交付时限内持续尝试')
    parser.add_argument('--max-redraws',type=int,help='每轮最多生成多少次修订候选，含局部修改和重新构图')
    parser.add_argument('--retry-round',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--retry-observation',default='',help=argparse.SUPPRESS)
    parser.add_argument('--round-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--port',type=int,choices=(8193,8194),default=8193)
    parser.add_argument('--reviewed-image',type=Path,help='使用已经人工审定的原图，仍进行最终检查')
    parser.add_argument('--review-print',type=Path,help='对照 --reviewed-image 底图，重新验收已有的 4K 修订稿')
    parser.add_argument('--repair-box',nargs=4,type=int,metavar=('L','T','R','B'),help='手动指出需要复绘的位置，0 到 1000')
    parser.add_argument('--object',help='可选，指定需要保护轮廓的对象，如 porcelain teapot')
    parser.add_argument('--repair-text',help='这个区域应当画成什么样')
    args=parser.parse_args()
    if args.max_redraws is not None and args.max_redraws<0:
        parser.error('--max-redraws 需要非负数')
    if args.base_model!='auto':
        native_requested=any(term in (args.prompt or '').lower().replace(' ','') for term in ('原生4k','native4k'))
        if args.from_image or args.reviewed_image or args.native_4k or native_requested:
            parser.error('--base-model qwen-image-2512 适用于从文字起稿，请使用模型原生画幅')
        if not has_drawing_model(RUNTIME/'ComfyUI/models',full=args.base_model=='qwen-image-2512-full'):
            parser.error('缺少 Qwen-Image-2512 模型组件，请先运行安装器')
    args.eight_k=args.eight_k or args.full_8k_diffusion or args.keep_native or args.skip_focus is not None
    if args.print_method=='restoration':
        from restoration import available as has_restorer
        if args.eight_k:parser.error('--print-method restoration 仅适用于 4K')
        if not has_restorer(RUNTIME/'ComfyUI/models'):parser.error('缺少本地 SeedVR2，安装模型后可使用 --print-method restoration')
    if args.review_print:
        if (not args.reviewed_image or args.eight_k or args.from_image or args.repair_box or args.candidate_image
                or args.native_only or args.proof_only or args.accept_final):
            parser.error('--review-print 需要 --reviewed-image 和完整 4K 审查')
        from focus_print import validate_print_pair
        try:validate_print_pair(args.reviewed_image,args.review_print)
        except (OSError,ValueError) as exc:parser.error(str(exc))
    if args.prompt and not args.style and KEY_VISUAL not in args.prompt:
        if args.prompt.lstrip().startswith(('日系商业插画，','日系商业插画：','日系商业插画,','日系商业插画:')):
            args.style='anime-key-visual'
    if args.style and args.prompt:
        style=args.style.strip()
        if style.casefold() in ('anime-key-visual','日系商业插画'):style=KEY_VISUAL
        args.prompt=args.prompt.rstrip()+' · '+style
    manual_mask=None
    if args.selection_mask:
        if not args.from_image or not (args.candidate_image or args.repair_text):
            parser.error('--selection-mask requires --from-image and either --repair-text or --candidate-image')
        with Image.open(args.from_image[0]) as original:
            manual_mask=selection_mask(args.selection_mask,original.size)
        if not args.candidate_image and not args.repair_box:
            l,t,r,b=mask_bounds(manual_mask)
            args.repair_box=[max(0,l-12),max(0,t-12),min(1000,r+12),min(1000,b+12)]
    elif args.candidate_image:
        parser.error('--candidate-image requires --selection-mask')
    if args.repair_box and (not args.repair_text or not args.from_image or not region({'box':args.repair_box,'user_selection':True},{})):
        parser.error('--repair-box 需要有效坐标、--repair-text 和 --from-image')
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error('--seed must be between 0 and 9223372036854775807')
    if args.output_dir is not None:
        args.output_dir=args.output_dir.expanduser().resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(6,os.cpu_count() or 2))
    job = args.job.resolve() if args.job else OUTPUT/'new-jobs'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    job.relative_to(OUTPUT)
    if args.max_rounds<0:parser.error('--max-rounds 需要非负数')
    previous=job/'review.json'
    adopt_draft=False
    if not args.round_worker and not (args.proof_only or args.native_only or args.accept_final) and previous.is_file() and not (job/'production.json').exists():
        earlier=json.loads(previous.read_text())
        adopt_draft=earlier.get('status') in ('底图已选 · 待目检','待人工复核 · 尚未选出底图')
        if adopt_draft:
            if args.prompt and args.prompt.strip()!=earlier['prompt']:
                parser.error('请为新提示词建立独立任务。')
            args.prompt=earlier['prompt']
            if args.seed is None:args.seed=earlier.get('seed')
            if earlier.get('原生4K'):args.native_4k=True
            if earlier.get('输出模式')=='全幅双级扩散':args.full_8k_diffusion=True
            args.eight_k=args.eight_k or earlier.get('delivery_edge',7680)==7680
            if not args.from_image and not args.reviewed_image:
                if earlier.get('底图'):args.reviewed_image=Path(earlier['底图'])
                elif earlier.get('输入底图'):args.from_image=[Path(x) for x in earlier['输入底图']]
    if not args.round_worker and not (args.proof_only or args.native_only) and (
        (job/'production.json').exists() or not previous.exists() or adopt_draft
    ):
        from production import run as supervise
        return supervise(args,job,Path(__file__))
    job.mkdir(parents=True,exist_ok=True)
    lock=(job/'.running').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('这个任务正由另一个本地进程运行')
    from scheduler import round_slot
    resources.enter_context(round_slot(RUNTIME, job))
    manifest = job/'review.json'
    rec = json.loads(manifest.read_text()) if manifest.exists() else {}
    if args.skip_focus is not None and (not rec or rec.get('输出模式')!='全幅4K扩散与8K焦点精修' or not rec.get('精绘区域报告',{}).get('8K局部')):
        parser.error('--skip-focus 需要使用已有focus-refinement结果的 --job')
    if args.skip_focus is not None:
        if any(i>len(rec['精绘区域报告']['8K局部']) for i in args.skip_focus):parser.error('编号超过这张图的实际焦点数')
        if args.accept_final or args.repair_box or args.keep_native or args.proof_only or args.native_only:parser.error('请分别执行局部取舍和其他修订操作')
        rec['撤回焦点']=sorted(set(args.skip_focus));rec.pop('目检定稿',None)
        if not rec.get('焦点取舍依据'):
            rec['焦点取舍依据']={'4K指纹':hashlib.sha256((job/rec['4K']).read_bytes()).hexdigest(),'区域':rec['精绘区域报告']['8K局部']}
    if args.accept_final:
        if rec.get('执行模式')=='离线自动定稿':parser.error('此版本需要通过自动终审才能定稿。')
        if not args.job or not rec.get('8K'):parser.error('需要有 8K 结果的 --job')
        master=job/rec['8K'];small=job/rec['4K']
        fingerprint=hashlib.sha256(master.read_bytes()).hexdigest()
        if args.accept_final!=fingerprint:
            parser.error('本次选择与当前 8K 母版不符，请重新打开校样')
        if fingerprint in rec.get('退稿',{}):
            parser.error('这份母版已经退稿。修订图有独立的指纹。')
        release=job/'manual-release'/fingerprint[:12]
        release.mkdir(parents=True,exist_ok=True)
        for src,label in ((master,'8K'),(small,'4K')):
            dest=release/(label+'.png')
            if not dest.exists():os.link(src,dest)
        rec['目检定稿']={'sha256':fingerprint,'时间':time.strftime('%Y-%m-%d %H:%M:%S'),'8K':str((release/'8K.png').relative_to(job)),'4K':str((release/'4K.png').relative_to(job))}
        rec['状态在目检前']=rec.get('状态在目检前',rec['status']);rec['status']='已目视定稿'
        save(manifest,rec);print(rec['status'],release,flush=True);return 0
    if rec.get('退稿') and args.skip_focus is None and not any((args.reviewed_image,args.from_image,args.candidate_image,args.repair_box,args.prompt)):
        print(rec['status'],flush=True);return 2
    changing=args.skip_focus is not None or any((args.prompt,args.from_image,args.reviewed_image,args.repair_box,args.candidate_image,args.seed is not None,args.candidates))
    if rec.get('status')=='已目视定稿' and not changing:
        print(rec['status'],job/rec['目检定稿']['8K']);return 0
    if rec.get('选区返修') and not changing:
        prior=rec['选区返修']
        args.from_image=[Path(prior['原图'])];args.candidate_image=Path(prior['候选'])
        args.selection_mask=Path(prior['遮罩'])
        with Image.open(args.from_image[0]) as original:
            manual_mask=selection_mask(args.selection_mask,original.size)
            args.keep_native=max(original.size)==7680
    prompt = (args.prompt or rec.get('prompt') or '').strip()
    if not prompt: parser.error('需要 --prompt 或已有任务目录')
    if rec and rec['prompt']!=prompt: parser.error('请为新的 prompt 开始一个新任务')
    seed = rec.get('seed',args.seed if args.seed is not None else secrets.randbelow(2**45))
    print('任务目录',job,flush=True)
    from art_direction import POLICY as ART_POLICY, DIRECTION as ART_DIRECTION, anime_finish
    if not rec:
        positive,raw=plan_scene(prompt)
        rec=dict(prompt=prompt,seed=seed,positive=positive,parts=json.loads(raw),reviews=[],status='选片',工作流='reference-illustration-v1',原生4K=(args.native_4k or any(t in prompt.lower().replace(' ','') for t in ('原生4k','native4k'))),画法参考=hashlib.sha256((INPUT/REFERENCE).read_bytes()).hexdigest())
        rec['画法自适应']=True
        if not (args.reviewed_image or args.review_print or args.repair_box or args.selection_mask):
            rec['art_policy']=ART_POLICY
        rec['画材']=KEY_VISUAL if KEY_VISUAL in prompt else requested_medium(prompt)
        rec['delivery_edge']=7680 if args.eight_k else 3840
        rec['输出模式']=('全幅双级扩散' if args.full_8k_diffusion else '全幅4K扩散与8K焦点精修') if args.eight_k else '全幅4K精绘'
        rec['笔触参考']=hashlib.sha256((INPUT/BRUSHWORK).read_bytes()).hexdigest()
        save(manifest,rec)
    reserve_redraw=partial(record_redraw,args.max_redraws,manifest,rec)
    four_only=rec.get('delivery_edge',7680)==3840
    if args.aspect!='auto':rec['aspect']=args.aspect
    if args.round_worker:rec['执行模式']='离线自动定稿'
    reference_style=rec.get('工作流')=='reference-illustration-v1'
    art_first=(rec.get('art_policy')==ART_POLICY and not (args.selection_mask or args.repair_box
               or args.review_print or args.reviewed_image or rec.get('指定修正')))
    anime_painting=(KEY_VISUAL in prompt or (art_first and anime_finish(rec.get('画材')) and
                   rec.get('base_model',args.base_model)=='auto'))
    if args.native_4k and not rec.get('原生4K'):
        parser.error('继续原任务时请保持原画幅，新画幅请建立新任务。')
    if reference_style and rec['画法参考']!=hashlib.sha256((INPUT/REFERENCE).read_bytes()).hexdigest():
        parser.error('该任务的画法参考已更换，请为新参考建立新任务。')
    args.candidates_explicit=args.candidates is not None
    args.candidates=args.candidates or rec.get('候选数',3 if args.deep_repair else 2)
    if args.from_image:
        rec['输入底图']=[str(p.resolve()) for p in args.from_image]
    else:
        args.from_image=[Path(p) for p in rec.get('输入底图',[])]
    if args.base_model!='auto':rec['base_model']=args.base_model
    rec['候选数']=args.candidates
    if not args.candidate_image and not args.reviewed_image and not args.repair_box and rec.get('底图人工审定'):
        args.reviewed_image=Path(rec['底图人工审定'])
    if args.repair_box:
        rec['指定修正']={'box':args.repair_box,'text':args.repair_text,'object':args.object}
    elif rec.get('指定修正') and not args.reviewed_image:
        correction=rec['指定修正']
        args.repair_box=correction['box'];args.repair_text=correction['text'];args.object=correction['object']
    prefix = job.relative_to(OUTPUT).as_posix()
    def remember(path,stage,*,print_ready=False):
        if args.round_worker:
            if not print_ready:
                with Image.open(path) as frame:print_ready=max(frame.size)>=3840
            checkpoint(job,Path(path),stage,print_ready=print_ready)

    def reserve_source_repaint(stage):
        if args.round_worker and four_only:
            timing=insufficient_repair_time(job,stage)
            if timing:
                rec.update(status=REPAIR_WINDOW_CLOSED,修订时间预算=timing,终审退回原因=REPAIR_WINDOW_CLOSED)
                save(manifest,rec)
                raise SystemExit(2)
        reserve_redraw(stage)

    from focus_print import prepare_composition_review
    prepared_prints={}
    def reconstruct_print(source_path,stage,print_seed):
        from restoration import restoration_graph
        with Image.open(source_path) as original:
            input_name=local_image(original.convert('RGB'))
        cache_key=(input_name,print_seed)
        if cache_key in prepared_prints:
            result=prepared_prints[cache_key]
            remember(result,stage,print_ready=True)
            return result
        graph,node=restoration_graph(input_name,prefix+'/pending-review/'+stage,3840,print_seed)
        save(job/(stage+'-workflow.json'),graph.ui())
        outputs=post_graph(graph.api,args.port,job,stage,graph.ui())
        result=preserve_source_marks(source_path,output_image(outputs[node]))
        if not args.selection_mask:
            remember(result,stage,print_ready=True)
            prepared_prints[cache_key]=result
        return result

    def prepare_repaint(path,label):
        remember(path,label)
        if not args.round_worker or not four_only or args.selection_mask or args.repair_box:return False
        from restoration import prefer_source_reconstruction
        with Image.open(path) as image:
            restore=prefer_source_reconstruction(args.print_method,max(image.size),RUNTIME/'ComfyUI/models')
        if restore:
            return reconstruct_print(path,label+'-restoration',seed+28601)
        return None

    def build_draft(caption,draw_seed,label,index):
        from page_layout import retry_direction
        clean=retry_direction(args.retry_observation)
        if clean:caption += ' '+clean
        from reference_review import last_finish_direction
        direction=last_finish_direction(rec)
        if not direction and not rec.get('reviews') and args.retry_round:
            earlier=args.retry_round/'review.json'
            if earlier.is_file():direction=last_finish_direction(json.loads(earlier.read_text()))
        if art_first:
            caption+=' '+ART_DIRECTION
        if direction:caption+=' '+direction
        from brief_check import drawing_identity_details
        landmarks=drawing_identity_details(prompt)
        if landmarks:caption=caption+' '+landmarks
        if not any(v in prompt.lower() for v in ('分镜','分格','多格','四格','多联','双联','三联','两联','二联','分幅','拼贴','拼图','对照图','连环画','分屏','漫画页','panel','diptych','triptych','collage','contact sheet','comic page','storyboard','split screen','split view')):
            caption='Compose one continuous scene in an uninterrupted frame. '+caption
        aspect=rec.get('aspect','auto')
        width,height={'2:3':(832,1248),'3:2':(1248,832),'9:16':(720,1280),'16:9':(1280,720),'1:1':(1024,1024)}.get(aspect,dimensions(prompt))
        name=prefix+'/candidates/'+label
        if not reference_style:return draft(caption,draw_seed,width,height,name)
        size={(832,1248):(1248,1872),(1248,832):(1872,1248),(720,1280):(1152,2048),(1280,720):(2048,1152),(1024,1024):(1536,1536)}[width,height]
        if rec.get('原生4K'):
            size=tuple(round(v*3840/max(size)/16)*16 for v in size)
        if not prompt.isascii():
            caption+=' Original request (use its species, colours and counts): '+prompt
        # A single style reference can also copy the example pose. Give complete jobs
        # both an independent layout and a text-only alternative to select from.
        borrow=any(term in prompt.lower() for term in ('参考图','参考画','像图','reference image','reference illustration'))
        own_first=rec.get('画法自适应') and bool(rec.get('画材')) and not borrow and not rec.get('原生4K') and not args.native_only
        from illustration import medium_study
        study=medium_study(rec.get('画材'))
        # Prefer the distilled text model for a new independent composition.
        # Retain the first slot for an actual material reference when available.
        text_first=(not study and not borrow and not rec.get('原生4K') and not args.native_only
                    and not args.from_image and has_drawing_model(RUNTIME/'ComfyUI/models'))
        text_slot=index==(0 if text_first else 1) or (args.candidates==1 and args.retry_round and index==0)
        automatic_text=text_slot and not borrow and not rec.get('原生4K') and not args.native_only and not args.from_image
        requested=rec.get('base_model','auto')
        # After two rejected rounds, devote one independent text candidate to the base
        # denoiser when production actually requests a new composition.
        full=(requested=='qwen-image-2512-full' or (requested=='auto' and automatic_text and args.deep_repair
              and args.retry_round and args.retry_round.name.isdigit() and int(args.retry_round.name)>=2))
        preferred_anime=(art_first and anime_painting and index==0 and requested=='auto'
                         and not borrow and not args.native_only and not rec.get('原生4K')
                         and (RUNTIME/'models/checkpoints'/ANIME_PAINT).is_file())
        if preferred_anime:
            graph=text_draft(caption,draw_seed,*size,name)
        elif requested!='auto' or (automatic_text and has_drawing_model(RUNTIME/'ComfyUI/models',full=bool(full))):
            # The text generator can reconstruct the relationships directly from the
            # brief, independently of any earlier image or rejected silhouette.
            native={(832,1248):(1088,1632),(1248,832):(1632,1088),(720,1280):(1008,1792),(1280,720):(1792,1008),
                    (1024,1024):(1328,1328)}[width,height]
            direction=('Draw the finished illustration in the medium and palette specified by the author. '
                       'When unspecified, use a Japanese storybook illustration style. ')
            if full:
                native={(832,1248):(1056,1584),(1248,832):(1584,1056),(720,1280):(928,1664),
                        (1280,720):(1664,928),(1024,1024):(1328,1328)}[width,height]
            graph=text_composition(direction+caption,draw_seed,*native,name,full=bool(full))
        elif study and index==0 and not args.from_image and not args.native_only and not borrow:
            with Image.open(study) as technique:
                sample=local_image(technique.convert('RGB'))
            graph=reference_draft(caption,draw_seed,*size,name,reference=sample,layout=True)
            rec['画材技法样片']=study.name
        elif (text_first and index==1) or (own_first and index==0) or (not own_first and index==2 and rec.get('画法自适应')):
            direction=('Follow the medium and palette in the request. When unspecified, draw a finished Japanese storybook illustration. '
                       'Place each principal subject once. At every overlap, keep the nearer contour legible. Show a plausible support or occluding surface for visible figures, tools and animals. Preserve all requested interactions. ')
            graph=text_draft(direction+caption,draw_seed,*size,name)
        elif index==1 and not rec.get('原生4K'):
            graph=independent_draft(caption,draw_seed,*size,name)
        elif args.native_only:
            graph=reference_draft(caption,draw_seed,*size,name)
        else:
            graph=reference_draft(caption,draw_seed,*size,name,layout=None if index==2 and not own_first else True)
        save(job/(label+'workflow.json'),graph.ui())
        return graph
    def color_candidate(path,label,draw_seed):
        if not anime_painting or not reference_style or args.repair_box or args.from_image:
            return path
        with Image.open(path) as layout:
            try:
                workflow=json.loads(layout.info.get('prompt','{}'))
            except (TypeError,ValueError):
                workflow={}
            qwen_layout=isinstance(workflow,dict) and any(isinstance(node,dict) and node.get('inputs',{}).get('unet_name')==DRAW_UNET for node in workflow.values())
            size=layout.size
            specialist=(RUNTIME/'models/checkpoints'/ANIME_PAINT).is_file() and max(size)<3840
            if qwen_layout and (not specialist or rec.get('base_model','auto')!='auto'):
                return path
            source=local_image(layout.convert('RGB'))
        destination=prefix+'/candidates/'+label+'-painted'
        if specialist:
            if not rec.get('上色标签'):
                from evidence import ask,backend
                schema={'type':'object','properties':{'tags':{'type':'array','items':{'type':'string'},'minItems':6,'maxItems':50}},'required':['tags']}
                request_tags=('Translate the brief into concise English anime illustration tags, ordered by subject counts, identity, individual colors, action, materials, setting and lighting. '
                              'Keep each interaction intact, for example a named animal sitting on a named surface in one phrase. Follow the actual request. Preserve object-specific color names. '
                              'Do not introduce people into an empty scene. Brief: '+prompt.split(' · '+KEY_VISUAL)[0])
                with backend() as address:items=json.loads(ask(address,[],request_tags,schema))['tags']
                rec['上色标签']=', '.join(t.strip() for t in items if isinstance(t,str) and t.strip())
                save(manifest,rec)
            if qwen_layout:
                paint=anime_color_pass(rec['上色标签'],draw_seed,destination,source,strength=.40,edge=max(size))
            else:
                paint=anime_color_pass(rec['上色标签'],draw_seed,destination,source)
        else:
            paint=paint_key_visual(prompt,draw_seed,size,destination,source)
        stage=label+'-color'
        save(job/(stage+'workflow.json'),paint.ui())
        history=post_graph(paint.api,args.port,job,stage,paint.ui())
        colored=output_image(history[str(len(paint.nodes))])
        remember(colored,stage)
        return colored
    if args.native_only:
        args.proof_only=True
    def check(path,stage,focus=None):
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        if art_first:
            from art_direction import assess, assessment_result, REFERENCE as art_reference
            from evidence import model_identity
            from print_judge import MODEL as art_reader
            identity=model_identity(art_reader)
            reference_id=hashlib.sha256(art_reference.read_bytes()).hexdigest()
            for row in reversed(rec['reviews']):
                evidence=row['review'].get('art_direction',{})
                if (row.get('hash')==key and evidence.get('model_id')==identity and identity
                        and evidence.get('policy')==ART_POLICY and evidence.get('reference_sha256')==reference_id):
                    if row['stage']!=stage:
                        rec['reviews'].append(dict(row,stage=stage,file=os.path.relpath(path,job),elapsed_seconds=0))
                        save(manifest,rec)
                    return row['review']
            started=time.monotonic()
            result=assessment_result(assess(path,prompt),prompt)
            rec['reviews'].append(dict(file=os.path.relpath(path,job),hash=key,stage=stage,
                                       elapsed_seconds=round(time.monotonic()-started,2),review=result))
            save(manifest,rec)
            print(stage,result['status'],result['note'],flush=True)
            return result
        rubric=hashlib.sha256((CHECKS+('-construction-zoom-v17-count-triage' if rec.get('画法自适应') else '')+'-independent-inventory-v6'+('body-support-v5-grounded-figures'+rec['画法参考'] if reference_style else '')).encode()).hexdigest()[:12]
        from page_layout import gutters,unrequested_panels
        with Image.open(path) as probe:overview,gaps=gutters(probe)
        if gaps and args.round_worker:rubric+='-layout2'
        check_contact=stage.startswith(('draft-','recompose-')) or (stage=='final-art' and rec.get('单候选成品初审')==key)
        if check_contact:rubric+='-working-contact'
        from review import inspection_environment,inspection_key
        from confirmed_issues import ledger_version
        context_key=inspection_key(rubric+ledger_version(),prompt,rec['parts'],focus,rec.get('画法参考'),inspection_environment())
        from construction_review import held_pocket_reading
        def inspection_matches(row):
            detail=row['review'].get('局部连接复审') or {}
            if detail.get('verdict')=='unclear' and 'palm_attachment_review' not in detail:
                from palm_review import WEIGHTS,hand_attachment
                disputed=(detail.get('proposed_finding') or {}).get('observation','')
                if WEIGHTS.is_file() and hand_attachment(disputed):return False
            return row.get('hash')==key and row.get('inspection_key')==context_key and not held_pocket_reading(row['review'])
        def match(row):
            if not inspection_matches(row):return False
            if reference_style and stage.endswith('final-art') and row['review'].get('status')=='pass':
                from reference_review import confirmed_final_comparison
                return confirmed_final_comparison(row['review'].get('参考比较') or {})
            return True
        def eligible_structure(row):
            return (reference_style and stage.endswith('final-art') and row['stage'].endswith('final-art')
                    and row['review'].get('status')=='pass' and row['review'].get('结构审查',{}).get('verdict')=='coherent'
                    and inspection_matches(row))
        prior_structure=next((row for row in reversed(rec['reviews']) if eligible_structure(row)),None)
        prior=next((row for row in rec['reviews'] if match(row) and row['stage']==stage),None)
        if prior:return prior['review']
        if args.round_worker and args.retry_round and (args.retry_round/'review.json').is_file():
            previous=json.loads((args.retry_round/'review.json').read_text())
            if prior_structure is None:
                prior_structure=next((row for row in reversed(previous.get('reviews',[])) if eligible_structure(row)),None)
            prior=next((row for row in previous.get('reviews',[]) if match(row) and row['review']['status']=='pass'),None)
            if prior:
                for suffix in ('-inspection.jpg','-inspection-1x.png','-structure-and-support.jpg',
                               '-contact-and-occlusion.jpg','-reference-comparison.jpg'):
                    proof=args.retry_round/(prior['stage']+suffix)
                    if proof.is_file():deliver_file(proof,job/(stage+suffix))
                rec['reviews'].append(dict(prior,file=os.path.relpath(path,job),stage=stage,
                                           reused_from=str(args.retry_round),elapsed_seconds=0))
                save(manifest,rec)
                return prior['review']
        started=time.monotonic()
        if gaps and args.round_worker:
            mismatch=unrequested_panels(overview,gaps,prompt)
            if mismatch:
                result=dict(status='global',score=35,note='画面被纸白间隔切成了多个独立画幅。'+mismatch['evidence'],
                            issues=[],regions={},layout=mismatch)
                rec['reviews'].append(dict(file=os.path.relpath(path,job),hash=key,rubric=rubric,inspection_key=context_key,stage=stage,elapsed_seconds=round(time.monotonic()-started,2),review=result))
                save(manifest,rec);return result
        from evidence import review_session
        with Image.open(path) as image, review_session() as timings:
            rgb=image.convert('RGB')
            if prior_structure is None:
                result=audit(rgb,prompt,rec['parts'],job/(stage+'-inspection.jpg'),focus=focus)
            else:
                # The pixels, brief, focus, model manifests and construction checks
                # match. Only the superseded independent workmanship reading is due.
                result=dict(prior_structure['review'])
                result.pop('参考比较',None)
                timings['reused_structure_stage']=prior_structure['stage']
            count_rejection=bool(result.get('count_conflicts')) and result['status'] in ('global','uncertain')
            if reference_style and prior_structure is None and not count_rejection:
                geometry=inspect_structure(rgb,result.get('detector',[]),job/(stage+'-structure-and-support.jpg'),result.get('observed_counts'),prompt)
                result['结构审查']=geometry
                verdict=geometry.get('verdict','unclear')
                if verdict=='defective':
                    result.update(status='global',note='主体连接或承托有误。'+geometry.get('support_or_occlusion',''))
                elif verdict!='coherent' and result['status']=='pass':
                    result.update(status='uncertain',note='主体连接或承托无法确认。'+geometry.get('support_or_occlusion',''))
                if rec.get('画法自适应'):
                    detail=review_construction(rgb,prompt,result.get('regions',{}),job/(stage+'-contact-and-occlusion.jpg'),result.get('detector',()),result.get('observed_counts',{}).get('person',0))
                    result['局部连接复审']=detail
                    dv=detail['verdict']
                    if dv in ('local','global') and result['status']!='global':
                        result.update(status=dv,note=detail['observation'])
                        if dv=='local':
                            result['issues'].insert(0,dict(operation='redraw',target=detail['target'],box=detail['box'],fix=detail['fix'],symptom=detail['observation']))
                    elif dv=='unclear' and result['status']=='pass':
                        result.update(status='uncertain',note=detail['observation'])
                if (args.round_worker and not args.from_image and not args.repair_box
                        and result['status'] in ('pass','local','uncertain') and check_contact):
                    from brief_check import check_working_contact
                    from construction_review import LARGE
                    from evidence import backend
                    from print_judge import MODEL
                    with backend(MODEL) as url:
                        contact=check_working_contact(url,rgb,prompt,MODEL,LARGE)
                    if contact:
                        result['requested_contact']=contact
                        if contact['confirmed']:
                            box=contact['box'];gap=contact['minimum_gap']
                            item,surface=contact['items']
                            symptom=contact.get('gap_evidence') or contact['observations'][0]['intervening_surface']
                            fix='Arrange '+item+' so its working end visibly meets '+surface+' as requested: '+contact['requirement']
                            result['issues'].insert(0,dict(box=box,target=item+' and '+surface,
                                operation='redraw',symptom=symptom,fix=fix))
                            result.update(status='global' if gap>145 else 'local',note=symptom)
                            if gap>145:
                                result['局部连接复审']=dict(result.get('局部连接复审') or {},fix=fix)
            if result['status']!='global' and not count_rejection and 'visual_triage' not in result:
                from visual_triage import review_image
                triage=review_image(rgb,prompt,key,job)
                result['visual_triage']=triage
                if triage['status']=='local':
                    result.setdefault('issues',[]).insert(0,triage['issue'])
                    result.update(status=triage['status'],note=triage['issue']['symptom'])
            # Assess workmanship on the actual version eligible to advance.
            if reference_style and (result['status']=='pass' or not args.round_worker):
                comparison=compare(rgb,result.get('regions',{}),job/(stage+'-reference-comparison.jpg'),
                                   brief=prompt if rec.get('画法自适应') else '',final=stage.endswith('final-art'),medium=rec.get('画材'))
                result['参考比较']=comparison
                if result['status']=='pass' and comparison['decision']=='below':
                    result['status']='global'
                    result['note']+=' · 交换左右后仍发现画法差距'
                elif comparison['decision']=='review':
                    if args.round_worker and result['status']=='pass':
                        result['status']='uncertain'
                    result['note']+=' · 放大对照仍未确定作品达到了参考图的完成度'
        rec['reviews'].append(dict(file=os.path.relpath(path,job),hash=key,rubric=rubric,inspection_key=context_key,stage=stage,elapsed_seconds=round(time.monotonic()-started,2),device=timings,review=result))
        save(manifest,rec)
        print(stage,result['status'],result['note'],flush=True)
        return result
    def usable(assessment):
        return assessment['status']=='pass' and (art_first or assessment['score']>=75)
    approved=None
    focus=args.repair_box
    def painting_feedback(assessment):
        from reference_review import final_repaint_direction
        if art_first:
            reading=assessment.get('art_direction',{}).get('reading',{})
            if reading.get('brief_and_structure')=='sound' and reading.get('overall_finish')=='needs_art_direction':
                return reading.get('direction','')
            return ''
        return final_repaint_direction(assessment.get('参考比较') or {})

    def check_composition(path,label,prepared_print=None):
        if not args.from_image and not args.repair_box:
            from framing import prepare_full_bleed
            from lettering import clean_lettering
            if prepared_print is None and not rec.get('原生4K') and not args.native_only:
                path=prepare_full_bleed(path,prompt)
        result=check(prepared_print,'final-art') if prepared_print else check(path,label)
        if art_first or result.get('visual_triage',{}).get('issue'):
            return path,result
        if not args.from_image and not args.repair_box and usable(result):
            cleaned=clean_lettering(path,prompt,job,label,before_edit=reserve_redraw)
            if cleaned!=path:
                path=cleaned
                prepare_repaint(cleaned,label+'-cleaned')
                result=check(path,label+'-cleaned')
        duplicate=duplicate_box(result) if rec.get('画法自适应') and not args.repair_box and ERASE_MODEL.exists() else None
        if not duplicate and args.round_worker and ERASE_MODEL.is_file() and any(
            c.get('requested')==0 and (c.get('observed') or 0)>0 for c in result.get('count_conflicts',[])
        ):
            # The blind inventory saw a forbidden subject. Localize it on real pixels before erasing.
            from print_judge import inspect
            inspected=inspect(path,prompt,job,result)
            finding=inspected.get('issue',{})
            if inspected['status']=='local' and finding.get('operation')=='remove_forbidden':
                duplicate=region(finding,result)
        if duplicate:
            reserve_redraw(label+'-count-repair')
            repaired=job/(label+'-removed-duplicate.png')
            with Image.open(path) as source:
                masked_candidate(source,duplicate).save(repaired)
            prepare_repaint(repaired,label+'-count-repair')
            checked=check(repaired,label+'-count-repair',duplicate)
            if usable(checked):return repaired,checked
        return path,result

    def check_draft_structure(path,result):
        if art_first or not usable(result) or not args.round_worker or args.proof_only:return result
        from print_judge import inspect, fingerprint
        if rec.get('待在4K复查的底图')==fingerprint(path):
            # The draft has already passed the composition and structure screen.
            # These same whole, construction and contact views are mandatory at
            # print size. Spend the remaining deadline on the finished pixels.
            return result
        report=inspect(path,prompt,job,result,construction_draft=True)
        rec.setdefault('底图结构终审',{})[os.path.relpath(path,job)]=report['report']
        save(manifest,rec)
        if report['status']=='pass':return result
        finding=report.get('issue',{})
        return dict(result,status=report['status'],issues=[finding] if finding else [],
                    note=finding.get('symptom') or '底图结构需要返修')

    def repair_draft(path, result, index):
        """Keep the last sound source until the proposed local correction passes."""
        from construction_review import assess_local_edit
        from print_judge import actionable_edit, fingerprint
        issue=draft_repair_issue(result)
        feedback=''
        consecutive_failures=0
        for attempt in range(3):
            if not issue:break
            box=region(issue,result)
            if args.repair_box:
                l,t,r,b=args.repair_box
                if not (l<=box[0]<box[2]<=r and t<=box[1]<box[3]<=b):break
            label=f'region-repair-{index+1}-{attempt+1}'
            reserve_redraw(label)
            instruction=' '.join(filter(None,[issue['fix'],feedback]))
            strategy=2 if consecutive_failures and attempt==1 else attempt
            with Image.open(path) as original:
                if issue.get('operation')=='remove_forbidden' and ERASE_MODEL.is_file():
                    patched=job/(label+'-removed.png')
                    remove_reviewed_feature(original.convert('RGB'),box,issue).save(patched)
                else:
                    patched=repaint(original.convert('RGB'),box,instruction,rec['positive'],
                                    seed+100+index*3+attempt,args.port,job,label,issue.get('target'),
                                    [o for o,_ in rec['parts']],reference_style,
                                    rebuild_shape=reference_style and issue.get('operation')=='redraw',
                                    reconstruction_attempt=strategy)
                patched=confine_edit(path,patched,box,job/(label+'-bounded.png'))
                if manual_mask is not None:
                    patched=merge_selection(path,patched,manual_mask,job/(label+'-masked.png'))
                prepare_repaint(patched,label)
                if args.round_worker or (reference_style and issue.get('operation')=='redraw'):
                    with Image.open(patched) as candidate:
                        verdict=assess_local_edit(original,candidate,box,issue['fix'],issue.get('symptom'),prompt)
                    save(job/(label+'-draft-patch.json'),dict(verdict,source_sha256=fingerprint(path),
                         candidate_sha256=fingerprint(patched),target_box=box,instruction=instruction,strategy=strategy))
                    if verdict['verdict']!='resolved':
                        consecutive_failures+=1
                        feedback=verdict.get('next_edit','')
                        if not actionable_edit(feedback):feedback=''
                        if consecutive_failures>=2 and not args.from_image and not args.repair_box:break
                        continue
            checked=check_draft_structure(patched,check(patched,label,box))
            if usable(checked):return patched
            next_issue=draft_repair_issue(checked)
            if not next_issue or checked['score']<result['score']:break
            path,result,issue=patched,checked,next_issue
            feedback=''
            consecutive_failures=0
        return None

    retry_source=None
    retry_origin=None
    if args.round_worker and four_only and not (args.repair_box or args.selection_mask):
        from production import local_retry_source
        retry_source=local_retry_source(job,prompt,interrupted=True,review_upgrade=True)
        if retry_source:
            retry_origin=job
        elif args.retry_round:
            retry_origin=args.retry_round
            retry_source=local_retry_source(retry_origin,prompt,review_upgrade=True)
    if retry_source:
        approved=retry_source[1]
    elif args.candidate_image:
        if not args.from_image:parser.error('选区合成需要 --from-image 提供原图')
        approved=merge_selection(args.from_image[0],args.candidate_image,manual_mask,job/'merged-selection.png')
        remember(approved,'merged-selection')
        focus=mask_bounds(manual_mask)
        rec['选区返修']={'原图':str(args.from_image[0]),'候选':str(args.candidate_image.resolve()),'遮罩':str(args.selection_mask.resolve()),'范围':focus}
    elif args.reviewed_image:
        approved=args.reviewed_image.resolve()
        rec['底图人工审定']=str(approved)
    else:
        holder=rec.get('待局修底图')
        if not holder and args.retry_round and (args.retry_round/'review.json').is_file():
            preceding=json.loads((args.retry_round/'review.json').read_text())
            if preceding.get('prompt')==prompt:holder=preceding.get('待局修底图')
        if holder and not args.from_image and not args.repair_box:
            from print_judge import fingerprint as file_hash
            retained=Path(holder['file'])
            if not retained.is_file() or file_hash(retained)!=holder['sha256']:
                raise ValueError('已保留的局部修订底图发生变化')
        else:holder=None
        paths=([retained] if holder else []) or list(args.from_image) or [job/name for name in rec.get('原生底图',[]) if (job/name).is_file()]
        assessed=[]
        passing=[]
        # A selection belongs to the supplied picture. New unrelated layouts
        # cannot establish whether a patch improves that selected area.
        count=len(paths) if args.repair_box or holder else max(args.candidates,len(paths))
        streaming=args.round_worker and not args.candidates_explicit and not args.from_image and not art_first
        for i in range(count):
            label=f'draft-{i+1}'
            if i<len(paths):
                path=paths[i].resolve()
            else:
                graph=build_draft(rec['positive'],seed+i,label,i)
                hist=post_graph(graph.api,args.port,job,label,graph.ui() if reference_style else None)
                path=output_image(hist[str(len(graph.nodes))])
                native=rec.setdefault('原生底图',[])
                local=str(path.relative_to(job))
                if local not in native:native.append(local)
                save(manifest,rec)
                with Image.open(path) as frame:
                    remember(path,label,print_ready=max(frame.size)==3840)
                print('原生底图',path,flush=True)
            if args.repair_box and i==0:
                # The caller identified this source defect. Inspect the edited pixels
                # against the original, then apply every ordinary draft and print gate.
                result={'status':'local','score':100,'issues':[{'box':args.repair_box,'user_selection':True,'fix':args.repair_text,
                                                               'operation':'redraw','target':args.object}]}
            elif holder:
                result=check(path,label)
            else:
                path=color_candidate(path,label,seed+373+i)
                prepared_print=None
                # Prepare the first real print before potentially lengthy draft audits.
                # An unchanged source produces the same graph digest at final refinement.
                if i==0 and args.round_worker and four_only and not args.from_image and not holder:
                    from restoration import prefer_source_reconstruction
                    with Image.open(path) as frame:
                        restore=prefer_source_reconstruction(args.print_method,max(frame.size),RUNTIME/'ComfyUI/models')
                    if restore:
                        from framing import prepare_full_bleed
                        path=prepare_full_bleed(path,prompt)
                        early=reconstruct_print(path,'first-source-restoration',seed+28601)
                        if count==1:
                            prepared_print=prepare_composition_review(path,early,rec)
                            save(manifest,rec)
                path,result=check_composition(path,label,prepared_print)
                if prepared_print and result.get('status')=='global':
                    if painting_feedback(result):
                        # Route the recorded rejection into the final repair stage.
                        # The exact print still needs its complete release receipt.
                        approved=path
                        break
            if usable(result) and not args.repair_box:
                if streaming or holder:
                    result=check_draft_structure(path,result)
                    if draft_repair_issue(result):
                        from print_judge import fingerprint as file_hash
                        holder=dict(file=str(path.resolve()),sha256=file_hash(path))
                        rec['待局修底图']=holder
                        save(manifest,rec)
                    if usable(result):
                        approved=path
                        break
                else:
                    passing.append((result,path,i))
            if holder and result['status']=='global':
                holder=None
                rec.pop('待局修底图',None)
                save(manifest,rec)
            if draft_repair_issue(result):
                if streaming or holder:
                    approved=repair_draft(path,result,i)
                    if approved or holder:break
                else:
                    assessed.append((result,path,i))
        for result,path,i in sorted(passing,key=lambda item:-item[0]['score']):
            result=check_draft_structure(path,result)
            if usable(result):
                approved=path
                break
            if draft_repair_issue(result):assessed.append((result,path,i))
        if not approved:
            for result,path,i in sorted(assessed,key=lambda x:-x[0]['score']):
                approved=repair_draft(path,result,i)
                if approved:break
        if holder and not approved:
            rec.update({'待局修底图':holder,'status':'未达标 · 保留构图待局部返修','终审退回原因':result.get('note','局部修复尚未通过')})
            save(manifest,rec)
            return 2
        if approved:rec.pop('待局修底图',None)
    if not approved and not args.from_image and not args.repair_box:
        # Allow fresh layouts for unattended production even when reviewers cannot localize an issue.
        conflicts=[c for row in rec['reviews'] for c in row['review'].get('count_conflicts',[])]
        terms={c['object']:c['requested'] for c in conflicts}
        if terms or not args.proof_only or any(a['review']['status']=='global' for a in rec['reviews']):
            constraint=', '.join(f"exactly {n} {name}" if n>0 else f"no {name}" for name,n in terms.items())
            instructions=list(dict.fromkeys(a['review'].get('局部连接复审',{}).get('fix','') for a in rec['reviews'] if a['review']['status'] in ('local','global')))
            positive=' '.join(filter(None,[constraint,rec['positive'],*instructions[-3:]]))
            for i in range(2):
                label=f'recompose-{i+1}'
                reserve_source_repaint(label)
                graph=None
                if i==0 and reference_style:
                    for row in reversed(rec['reviews']):
                        finding=row['review'].get('requested_contact') or {}
                        if not finding.get('confirmed') or finding.get('minimum_gap',0)<=145:continue
                        source=job/row['file']
                        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=row['hash']:continue
                        with Image.open(source) as frame:
                            graph=contact_layout_graph(frame,finding,prompt,seed+800,prefix+'/'+label)
                        if graph:break
                if graph is None:graph=build_draft(positive,seed+800+i,label,i)
                avoid=', '.join('duplicate '+o+', multiple '+o for o,n in terms.items() if n==1)
                if avoid and not reference_style:graph.api['3']['inputs']['text']+=', '+avoid
                hist=post_graph(graph.api,args.port,job,label,graph.ui() if reference_style else None)
                new_path=output_image(hist[str(len(graph.nodes))])
                remember(new_path,label)
                new_path=color_candidate(new_path,label,seed+1173+i)
                if args.round_worker and four_only and not args.proof_only and not args.native_only and not rec.get('原生4K'):
                    from framing import prepare_full_bleed
                    new_path=prepare_full_bleed(new_path,prompt)
                restored=prepare_repaint(new_path,label)
                prepared=prepare_composition_review(new_path,restored,rec) if restored and not args.proof_only else None
                if prepared:save(manifest,rec)
                new_path,result=check_composition(new_path,label,prepared)
                if prepared and result['status']=='global':
                    if painting_feedback(result):
                        approved=new_path;break
                if usable(check_draft_structure(new_path,result)):
                    approved=new_path;break
    if not approved:
        rec['终审退回原因']=next((v['review']['note'] for v in reversed(rec['reviews']) if v['review'].get('status')!='pass' and v['review'].get('note')), 'No draft cleared the composition review.')
        old_proof=job/'selected-proof.png'
        if old_proof.is_file():
            archive=job/'archived-proofs'
            archive.mkdir(exist_ok=True)
            key=hashlib.sha256(old_proof.read_bytes()).hexdigest()[:16]
            old_proof.replace(archive/(key+'.png'))
        rec.pop('底图',None)
        rec['status']='未达标 · 所有底图均未通过' if args.round_worker else '待人工复核 · 尚未选出底图';save(manifest,rec)
        print(rec['status'],flush=True);return 2
    rec['底图']=str(approved);rec['status']='精绘中';save(manifest,rec)
    if args.proof_only:
        proof=job/'selected-proof.png'
        deliver_file(approved,proof)
        if args.output_dir:
            from delivery import export_files
            export_files(args.output_dir,job,{'selected-proof.png':proof})
        rec['status']='底图已选 · 待目检';save(manifest,rec);print(rec['status'],proof,flush=True);return 0
    def render(source_path,stage):
        if torch.backends.mps.is_available():torch.mps.empty_cache()
        with Image.open(source_path) as selected:
            composition=selected.convert('RGB')
            source_edge=max(composition.size)
            native_four=reference_style and source_edge>=3840
            if native_four:
                composition.thumbnail((3840,3840),Image.Resampling.LANCZOS)
            else:
                rebuilt_source=rec.get('高清底图返工',{}).get('重绘')==str(source_path)
                bound=3840 if rebuilt_source else (2304 if reference_style else 1536)
                composition.thumbnail((bound,bound),Image.Resampling.LANCZOS)
            source=local_image(composition)
        if four_only and source_edge<3840:
            from restoration import prefer_source_reconstruction
            # Resume a previously submitted graph with its own files and seed.
            if not (job/(stage+'-task.json')).is_file() and prefer_source_reconstruction(
                    args.print_method,source_edge,RUNTIME/'ComfyUI/models'):
                restored_stage='first-source-restoration' if stage=='refine' else stage+'-source-restoration'
                master=reconstruct_print(source_path,restored_stage,seed+28601)
                if stage=='refine':rec['初次4K方法']='SeedVR2'
                from print_judge import fingerprint as image_fingerprint
                rec['当前4K重建底图指纹']=image_fingerprint(source_path)
                rec['精绘区域报告']={'覆盖':'全幅源图条件扩散重建','长边':[3840]}
                save(manifest,rec)
                return master,None
        focus_mode=rec.get('输出模式')=='全幅4K扩散与8K焦点精修'
        medium=rec.get('画材','')
        natural_sr=continuous_tone(medium) and (RUNTIME/'models/upscale_models'/NATURAL_UP).is_file()
        if reference_style or four_only:
            from focal_edit import print_constraints, print_subjects
            restrictions=print_constraints(prompt)
            scales=([3840] if not native_four else [])+([] if four_only or focus_mode else [7680])
            subjects={edge:print_subjects(composition,edge,job/(stage+'-tile-subjects-'+str(edge)+'.json'),constraints=restrictions) for edge in scales}
            graph,ids=print_graph(source,prefix+'/pending-review/'+stage,seed+320,native_four=native_four,
                                  medium=medium+' '+restrictions,four_steps=4,four_only=four_only or focus_mode,upscale_model=NATURAL_UP if natural_sr else UP,tile_subjects=subjects)
        else:
            graph,ids=finish(source,prompt,prefix+'/pending-review/'+stage,seed+320)
        save(job/(stage+'workflow.json'),graph.ui())
        hist=post_graph(graph.api,args.port,job,stage,graph.ui() if reference_style else None)
        if four_only:
            four=output_image(hist[ids['small']])
            rec['精绘区域报告']={'覆盖':'原生4K' if native_four else '4K全幅分块扩散','长边':[3840]}
            return preserve_source_marks(source_path,four),None
        if focus_mode:
            small=output_image(hist[ids['small']])
            if not args.selection_mask:remember(small,stage+'-4k',print_ready=True)
            # Use actual landmarks from the approved image; there are no scene-specific coordinates.
            from evidence import ask, backend
            from construction_review import landmarks
            prior=rec.get('焦点取舍依据',{})
            small_hash=hashlib.sha256(small.read_bytes()).hexdigest()
            reuse=bool(prior) and prior['4K指纹']==small_hash
            if not reuse:
                with backend() as address:
                    points=landmarks(address,composition)
            neural=small.with_name(small.stem+'-neural-2x.png')
            if natural_sr:
                with Image.open(small) as four:
                    sr_input=local_image(four.convert('RGB'))
                sr_graph,node=continuous_tone_double(sr_input,prefix+'/pending-review/'+stage+'/continuous-tone-2x')
                sr_hist=post_graph(sr_graph.api,args.port,job,stage+'-continuous-tone-2x',sr_graph.ui())
                neural=output_image(sr_hist[node])
            elif not neural.exists():
                temp=neural.with_suffix('.partial.png')
                neural_double(small,temp);temp.replace(neural)
            # Retain source frequencies above the 4K sampling limit for large imported art.
            base_guide=source_path if source_edge>3840 else small
            with Image.open(neural) as sr, Image.open(base_guide) as original:
                canvas=combine_source(original,sr,rec.get('画材',''))
                boxes=prior['区域'] if reuse else centers(points,canvas.size)
                rec['焦点取舍依据']={'4K指纹':small_hash,'区域':boxes}
                if not reuse:save(manifest,rec)
                surfaces=sampled_surfaces(original,canvas.size,boxes)
                redraw=[*boxes,*surfaces]
                names=[local_image(canvas.crop((l,t,l+PATCH,t+PATCH))) for _,l,t in redraw]
                from focal_edit import describe_tiles
                descriptions=describe_tiles(canvas,names,redraw,PATCH,INPUT,job/(stage+'-tile-subjects.json'))
            master=small.with_name('8K-focus-refinement.png')
            skipped=set(rec.get('撤回焦点',[])) if reuse else set()
            chosen=[]
            for offset in range(0,len(redraw),6):
                batch=redraw[offset:offset+6]
                batch_stage=stage+'-detail-'+str(offset)
                prefix8=prefix+'/pending-review/'+batch_stage
                detail,nodes=detail_graph(names[offset:offset+6],medium+' '+restrictions,seed+12020+offset,
                                          prefix8,roles=[role for role,_,_ in batch],descriptions=descriptions[offset:offset+6])
                save(job/(batch_stage+'-workflow.json'),detail.ui())
                patches=post_graph(detail.api,args.port,job,batch_stage,detail.ui())
                chosen.extend((l,t,output_image(patches[k]),role) for i,((role,l,t),k) in enumerate(zip(batch,nodes),offset+1) if i not in skipped)
            finish_neural(base_guide,neural,chosen,master,rec.get('画材',''))
            master=preserve_source_marks(source_path,master)
            small=preserve_source_marks(source_path,small)
            focus_sheet=job/(stage+'-detail-comparison.jpg')
            if boxes:write_focus_proof(master,[INPUT/name for name in names[:len(boxes)]],boxes,focus_sheet)
            rec['精绘区域报告']={'同位校样':focus_sheet.name,'覆盖':'4K全幅扩散或原生4K，8K全幅神经超分，焦点与有结构区域按原像素重绘','8K图块':PATCH,'8K局部':boxes,'全幅纹理选区':surfaces,'实际撤回的区域':sorted(skipped),'长边':[3840,7680]}
            return master,small
        if 'report' in ids:rec['精绘区域报告']=hist.get(ids['report'])
        else:rec['精绘区域报告']={'覆盖':'原生 4K、8K 分块扩散' if native_four else '两次全幅重叠分块扩散','图块':1152,'布局':'全幅等距，保形融合','长边':[3840,7680]}
        master=output_image(hist[ids['master']]);small=output_image(hist[ids['small']])
        for path,edge in ((master,7680),(small,3840)):
            with Image.open(path) as image:
                if max(image.size)!=edge:raise RuntimeError('尺寸校验失败 '+str(path))
        if reference_style:
            master=preserve_source_marks(source_path,master)
            small=preserve_source_marks(source_path,small)
        return master,small
    def native_pair(source_path,stage,prior=None):
        dest=job/'pending-review'/stage
        dest.mkdir(parents=True,exist_ok=True)
        with Image.open(source_path) as im:
            if max(im.size)!=7680:raise ValueError('--keep-native 仅接受长边 7680 的既有母版')
            ratio=3840/max(im.size)
            scaled=(repair_print_pair(*prior,im) if prior else
                    im.convert('RGB').resize(tuple(round(v*ratio) for v in im.size),Image.Resampling.LANCZOS))
        master=dest/'8K.png';small=dest/'4K.png'
        deliver_file(source_path,master)
        temp=dest/(uuid.uuid4().hex+'.png');scaled.save(temp);temp.replace(small)
        return master,small
    def correct_shape(source,issue,box,stage,edit_seed,previous=None):
        from construction_review import assess_local_edit
        instruction=issue['fix']
        with Image.open(source) as original:
            before=original.convert('RGB')
        if reference_style:
            from placement import placement_plan
            plan=placement_plan(before,instruction,prompt)
            if plan:
                reserve_redraw(stage+'-layout')
                trial=repair_placement(before,plan,rec['positive'],edit_seed,args.port,job,stage+'-layout')
                if trial:
                    path,_=trial
                    path=confine_edit(source,path,box,job/(stage+'-layout-bounded.png'))
                    if not (args.selection_mask or args.repair_box):remember(path,stage+'-layout')
                    with Image.open(path) as proposed:
                        checked=assess_local_edit(before,proposed,box,instruction,issue.get('symptom'),prompt)
                    from print_judge import fingerprint as print_fingerprint
                    checked.update(source_sha256=print_fingerprint(source),candidate_sha256=print_fingerprint(path),box=box)
                    save(job/(stage+'-layout-review.json'),checked)
                    if checked['verdict']=='resolved':return path
        from production import repeated_rejection
        recurring=bool(previous and repeated_rejection(previous,box))
        strategies=[2,0,1] if recurring else [0,1,2]
        for variant,strategy in enumerate(strategies):
            name=stage if variant==0 else f'{stage}-variant-{variant+1}'
            reserve_redraw(name)
            revised=repaint(before,box,instruction,rec['positive'],edit_seed+variant*1753,args.port,
                            job,name,issue.get('target'),[o for o,_ in rec['parts']],reference_style,rebuild_shape=True,
                            reconstruction_attempt=strategy)
            revised=confine_edit(source,revised,box,job/(name+'-bounded.png'))
            if not (args.selection_mask or args.repair_box):remember(revised,name)
            release_idle_renderer(args.port)
            with Image.open(revised) as proposed:
                verdict=assess_local_edit(before,proposed,box,issue['fix'],issue.get('symptom'),prompt)
            with source.open('rb') as original_file, revised.open('rb') as edited_file:
                verdict.update(source_sha256=hashlib.file_digest(original_file,'sha256').hexdigest(),
                               candidate_sha256=hashlib.file_digest(edited_file,'sha256').hexdigest(),
                               source_file=os.path.relpath(source,job),candidate_file=os.path.relpath(revised,job))
            verdict.update(target_box=box,reconstruction_attempt=strategy)
            if strategy==2 and verdict.get('verdict') in ('unchanged','still_present') and not pigment_edit(instruction):
                # The specialist reproduced a known geometric error. The next attempt
                # rebuilds the interior using a cleared visual reference.
                if 3 not in strategies:strategies.append(3)
            if (args.deep_repair and strategy==3 and recurring and verdict.get('verdict') in ('unchanged','still_present')
                    and installed_editor(RUNTIME/'ComfyUI/models')):
                # A sequence of rejected short edits can repeat the same geometry. Give
                # the full base editor one attempt before a complete scene redraw.
                strategies.append(4)
            save(job/(name+'-patch-review.json'),verdict)
            if verdict['verdict']=='resolved':
                return revised
            instruction=(issue['fix']+' Recompose the faulty local forms and their contact within the same framing. '
                         +str(verdict.get('next_edit') or 'Redraw the silhouette and its point of support.'))
        return None

    def repaint_finish(master,approved,assessment):
        instruction=painting_feedback(assessment)
        if (not instruction or not args.round_worker or rec.get('原生4K') or args.keep_native
                or args.repair_box or args.selection_mask):
            return None
        from illustration import finish_repaint_graph, medium_study
        from print_judge import same_composition
        editor=installed_editor(RUNTIME/'ComfyUI/models')
        if not editor:return None
        stage='finish-repaint'
        with Image.open(master) as art:
            drawing=art.convert('RGB')
        drawing.thumbnail((1536,1536),Image.Resampling.LANCZOS)
        sample=medium_study(rec.get('画材'))
        study=None
        if sample:
            with Image.open(sample) as image:
                study=local_image(image.convert('RGB'))
        prior=args.retry_round/'review.json' if args.retry_round else None
        earlier=json.loads(prior.read_text()).get('画法整图返工',{}) if prior and prior.is_file() else {}
        repeated=earlier.get('重绘指纹')==fingerprint(approved) or earlier.get('前一版本指纹')==fingerprint(master)
        if repeated and not args.deep_repair:
            # The same composition already received a full-frame edit. Return
            # its rejection so that the next round can draw a different layout.
            return None
        full=bool(repeated and args.deep_repair)
        reserve_source_repaint(stage)
        graph=finish_repaint_graph(prompt,instruction,seed+882501,local_image(drawing),
                                  prefix+'/'+stage,editor,study=study,full=full)
        save(job/(stage+'-workflow.json'),graph.ui())
        outputs=post_graph(graph.api,args.port,job,stage,graph.ui())
        candidate=output_image(outputs[str(len(graph.nodes))])
        restored=prepare_repaint(candidate,stage)
        from reference_review import review_finish_changes
        finding=review_finish_changes(approved,candidate,prompt)
        save(job/(stage+'-design-review.json'),finding)
        aligned=finding['verdict']=='coherent'
        if finding['verdict']=='local':
            from construction_review import assess_local_edit
            with Image.open(candidate) as picture:before=picture.convert('RGB')
            target_box=finding['box']
            from registered_edit import CONTROL
            for use_edges in (True,False):
                retry_stage=stage+('-design-repair' if use_edges else '-design-edit')
                reserve_redraw(retry_stage)
                corrected=repaint(before,target_box,finding['fix'],prompt,seed+882502,args.port,job,
                                  retry_stage,finding.get('target'),reference_style=True,
                                  rebuild_shape=True,design_reference=approved,allow_control=use_edges)
                corrected=confine_edit(candidate,corrected,target_box,job/(retry_stage+'-bounded.png'))
                corrected_restored=prepare_repaint(corrected,retry_stage)
                with Image.open(corrected) as picture:
                    repair=assess_local_edit(before,picture.convert('RGB'),target_box,
                                             finding['fix'],finding.get('observation'),prompt)
                save(job/(retry_stage+'-review.json'),repair)
                if repair['verdict']=='resolved':
                    candidate=corrected
                    restored=corrected_restored
                    aligned=same_composition(approved,candidate,prompt)
                    break
                reference_record=job/(retry_stage+'-reference.json')
                if not reference_record.is_file() or json.loads(reference_record.read_text()).get('model') not in (CONTROL,'coreml-sd15'):
                    break
        elif finding.get('independent',{}).get('verdict')=='faithful':
            aligned=same_composition(approved,candidate,prompt)
        rec['画法整图返工']={'前一版本':str(master),'前一版本指纹':fingerprint(master),
                           '重绘':str(candidate),'重绘指纹':fingerprint(candidate),
                           '保留构图':aligned,'修改要求':instruction,'步数':40 if full else 4}
        save(manifest,rec)
        if aligned and (restored or usable(check_draft_structure(candidate,check(candidate,stage+'-draft')))):
            return candidate
        return None

    repair_check=None
    retained_failure=None
    if retry_source:
        last,_,issue=retry_source
        focus=region(issue,{})
        # Check a reported border problem on the retained pixels before redrawing
        # a body or object that intentionally continues beyond the canvas.
        from print_judge import FRAME_CLAIM, verify_frame_exit, excluded_by_brief, exclusion_repair, MODEL as PRINT_MODEL
        # Earlier generated edits can introduce a seam that becomes part of the
        # next editor's conditioning. Test a previous version against current pixels
        # only when every pixel outside the damaged region is exactly the same.
        reverted=None
        if focus and issue.get('operation')=='redraw':
            from production import earlier_local_version, repeated_rejection
            from print_judge import REVISION
            if repeated_rejection(retry_origin,focus):
                alternative=earlier_local_version(job,last,retry_source[1],focus,prompt,REVISION)
                if alternative:
                    earlier,history=alternative
                    from construction_review import compare_prior_version
                    with Image.open(last) as current_art, Image.open(earlier) as older_art:
                        verdict=compare_prior_version(current_art,older_art,focus,issue['fix'],issue.get('symptom'),prompt)
                    save(job/('prior-version-'+history['current'][:12]+'-'+history['earlier'][:12]+'-review.json'),
                         {**history,'verdict':verdict})
                    if verdict.get('verdict')=='resolved' or verdict.get('eligible_for_full_review'):
                        reverted=earlier
        if not reverted and focus and issue.get('verify_part_identity'):
            from print_judge import verify_depicted_connection, verify_part_identity
            from evidence import backend
            with Image.open(last) as source,backend(PRINT_MODEL) as address:
                identity=verify_part_identity(address,source,focus,issue)
                contact=None
                if not (identity and identity['disproved']):
                    contact=verify_depicted_connection(address,source,focus,issue.get('symptom',''),prompt)
            if contact:
                save(job/'continued-visible-connection.json',contact)
                if contact['disproved']:
                    issue=dict(issue,operation='reinspect',symptom=contact['verification']['trace'])
            if identity:
                save(job/'continued-part-identity.json',identity)
                if identity['disproved']:
                    issue=dict(issue,operation='reinspect',symptom='Check the visible attachment. '+identity['verification']['trace'])
        if focus and issue.get('unresolved_exclusion'):
            from evidence import backend
            with Image.open(last) as source,backend(PRINT_MODEL) as address:
                plan=exclusion_repair(address,source,focus,prompt,excluded_by_brief(prompt),issue.get('symptom',''))
            if plan:
                save(job/'continued-exclusion-plan.json',plan)
                issue=dict(issue,operation='redraw' if plan['action']=='clear_surface' else 'remove_forbidden',
                           target=plan['part'],fix=plan['edit'])
        claim=issue.get('symptom','')+' '+issue['fix']
        framing=None
        if focus and FRAME_CLAIM.search(claim) and (min(focus[:2])<=12 or max(focus[2:])>=988):
            from evidence import backend
            with Image.open(last) as source, backend(PRINT_MODEL) as address:
                overview=source.copy();overview.thumbnail((1152,1152),Image.Resampling.LANCZOS)
                framing=verify_frame_exit(address,source,overview,focus,claim,prompt)
            if framing:save(job/'continued-framing-review.json',framing)
        if reverted:
            repaired=reverted
        elif issue.get('operation')=='reinspect' or (framing and framing['explained']):
            repaired=last
        else:
            stage='continued-repair'
            if retry_origin==job:
                with last.open('rb') as stream:
                    stage='resume-'+hashlib.file_digest(stream,'sha256').hexdigest()[:16]+'-repair'
            if focus and issue.get('operation')=='remove_forbidden' and ERASE_MODEL.is_file():
                repaired=job/(stage+'-removed-unrequested-object.png')
                reserve_redraw(stage)
                with Image.open(last) as retained:remove_reviewed_feature(retained.convert('RGB'),focus,issue).save(repaired)
                repaired=confine_edit(last,repaired,focus,job/(stage+'-bounded.png'))
            else:
                repaired=correct_shape(last,issue,focus,stage,seed+820,retry_origin) if focus else None
        if repaired:
            master,small=repaired,None
            repair_check={'box':focus,'symptom':issue.get('symptom') or issue['fix']} if focus else None
            rec['续修来源']=str(last)
        else:
            # None of the new candidates improved the defect. The prior, hash-verified
            # negative receipt still describes these exact pixels and can be carried
            # forward without asking a vision model to regrade an unchanged failure.
            previous=json.loads((retry_origin/'review.json').read_text())
            entry=dict(previous['逐块终审'])
            entry['报告']=os.path.relpath((retry_origin/entry['报告']).resolve(),job)
            rec.update({'4K':os.path.relpath(last,job),'逐块终审':entry,
                        '终审退回原因':issue.get('symptom') or issue['fix'],'status':'未达标 · 本轮局部重绘未获通过'})
            save(manifest,rec)
            print(rec['status'],flush=True)
            return 2
        if repaired and issue.get('operation')!='reinspect':
            prior=json.loads((retry_origin/'review.json').read_text())
            entry=dict(prior['逐块终审'])
            entry['报告']=os.path.relpath((retry_origin/entry['报告']).resolve(),job)
            retained_failure={'4K':os.path.relpath(last,job),'底图':str(retry_source[1]),
                              '逐块终审':entry,'终审退回原因':issue.get('symptom') or issue['fix']}
    elif args.review_print:
        from print_judge import fingerprint,reviewed_print_layout
        supplied=args.review_print.resolve()
        layout=reviewed_print_layout(approved,supplied,prompt,job/'composition-evidence.json')
        rec['修订稿构图核对']='composition-evidence.json'
        if not layout['eligible_for_inspection']:
            rec.update(status='未达标 · 修订稿与底图构图不一致',终审退回原因=layout['reading']['differences'])
            save(manifest,rec)
            return 2
        master=job/'submitted-print.png'
        expected=fingerprint(supplied)
        if not master.is_file() or fingerprint(master)!=expected:
            temp=job/(uuid.uuid4().hex+'.png')
            try:
                shutil.copyfile(supplied,temp)
                if fingerprint(temp)!=expected:raise ValueError('4K 修订稿复制校验失败')
                temp.replace(master)
            finally:temp.unlink(missing_ok=True)
        small=None
        rec['已有修订稿']={'来源':str(supplied),'指纹':fingerprint(master),'底图指纹':fingerprint(approved)}
    else:
        with Image.open(approved) as img:
            local_master=bool((args.repair_box or args.selection_mask) and four_only and max(img.size)==3840)
        if local_master:
            master=job/'pending-review'/'local-master.png'
            master.parent.mkdir(exist_ok=True)
            deliver_file(approved,master)
            small=None
        else:
            master,small=native_pair(approved,'native-repair') if args.keep_native else render(approved,'refine')
    from print_judge import actionable_edit, fingerprint, grade_finished_print, MODEL
    from production import publish
    detail_rebuilt=False
    tried_focus_fields=set()
    # The model inspects the exact finished files, including every replacement.
    for attempt in range(3):
        label='final-art' if attempt==0 else f'revision-{attempt}-final-art'
        from focus_print import fit_print_aspect
        aspect=rec.get('aspect','auto')
        if not (repair_check or args.repair_box or args.selection_mask):
            master=fit_print_aspect(master,aspect,3840 if four_only else 7680)
            if small is not None:
                small=fit_print_aspect(small,aspect,3840)
        if manual_mask is not None:
            with Image.open(master) as img:
                same_canvas=img.size==manual_mask.size
            if same_canvas:
                master=merge_selection(args.from_image[0],master,manual_mask,
                                       job/(label+'-protected.png'),protect_only=True)
        release_idle_renderer(args.port)
        rec['4K']=os.path.relpath(master if four_only else small,job)
        if not four_only:rec['8K']=os.path.relpath(master,job)
        remember(master if four_only else small,label,print_ready=True)
        rec['status']='离线逐块终审中';save(manifest,rec)
        print_focus=dict(rec.get('精绘区域报告') or {})
        if repair_check:print_focus['repair_check']=repair_check
        baseline=approved if four_only else small
        final,judged,timings=grade_finished_print(master,prompt,job,lambda:check(master,label,focus),
            small=baseline,focus=print_focus,draft=None if four_only else approved,source_comparison=four_only)
        rec.setdefault('终审设备耗时',{})[label]=timings
        if judged is None:
            rec['终审退回原因']=final['note']
            candidate=repaint_finish(master,approved,final) if attempt==0 and not (repair_check or args.repair_box or args.selection_mask) else None
            if candidate:
                approved=candidate
                focus=None
                repair_check=None
                rec['底图']=str(approved)
                master,small=render(approved,'finish-repaint-print')
                continue
            break
        if four_only:rec['底图']=str(baseline)
        rec['逐块终审']={'sha256':judged['sha256'],'状态':judged['status'],'已看视图':len(judged['views']),
                      '计划视图':judged['planned'],'原像素图块':judged['planned_native'],'报告':judged['report']}
        save(manifest,rec)
        if judged['status']=='pass' and usable(final):
            if judged['comparison_sha256']!=fingerprint(baseline):raise ValueError('对照图在审查后改变')
            with Image.open(master) as finished:
                if max(finished.size)!=rec.get('delivery_edge',7680):raise ValueError('定稿尺寸不一致')
                if small is not None:
                    with Image.open(small) as four:
                        if max(four.size)!=3840 or any(abs(a-b*2)>1 for a,b in zip(finished.size,four.size)):
                            raise ValueError('定稿尺寸不一致')
            rec['自动定稿']={'model':MODEL,'模型指纹':judged['model_id'],'sha256':judged['sha256'],
                         '4K指纹':judged['sha256'] if four_only else judged['comparison_sha256'],
                         '对照指纹':judged['comparison_sha256'],'终审报告':judged['report'],'审图版本':judged['revision'],
                         '原像素图块':judged['planned_native'],'已看视图':len(judged['views'])}
            if art_first:rec['自动定稿']['art_direction']=final['art_direction']
            rec.pop('终审退回原因',None)
            rec['status']='已通过离线终审'
            if not args.round_worker:publish(job,job,rec)
            save(manifest,rec);print(rec['status'],flush=True);return 0
        # A localized, actionable suspicion can produce a *candidate*; it still has
        # to beat the source on the native crop and pass the full review to be released.
        if art_first and usable(final) and judged['status'] in ('local','uncertain'):
            from art_direction import retouch_note
            note=retouch_note(master,prompt,judged)
            if note:
                rec.update(status=RETOUCH,retouch_note=note,终审退回原因=note['evidence'])
                save(manifest,rec)
                print(RETOUCH,note['evidence'],flush=True)
                return 2
        finding=judged.get('issue',{})
        if judged['status']=='local' or (judged['status']=='uncertain' and finding.get('operation') in ('redraw','remove_forbidden')):
            candidates=[finding]
        elif final['status']=='local' and judged['status']=='pass':
            candidates=final['issues']
        else:
            candidates=[]
        issue=next((item for item in candidates if region(item,final) and actionable_edit(item.get('fix'))),None)
        rec['终审退回原因']=judged.get('issue',{}).get('symptom') or final.get('note','')
        if four_only and issue and judged['status'] in ('local','uncertain'):
            retained_failure={k:rec[k] for k in ('4K','底图','逐块终审','终审退回原因')}
        locked_box=args.repair_box or (rec.get('选区返修',{}).get('范围') if args.selection_mask else None)
        if locked_box and issue:
            l,t,r,b=locked_box
            box=region(issue,final)
            if not (l<=box[0]<box[2]<=r and t<=box[1]<box[3]<=b):break
        with Image.open(approved) as original:
            native_edge=max(original.size)
            rebuild_size=detail_canvas(original.size)
        if four_only and attempt<2 and not (args.keep_native or args.selection_mask or args.repair_box):
            from focal_edit import failed_focus, print_constraints, repaint_definition
            pixels=failed_focus(judged)
            if not rec.get('4K散点修复'):
                from focal_edit import repair_reported_speckles
                polished=repair_reported_speckles(master,approved,judged,job,f'print-clean-{attempt+1}',before_edit=reserve_redraw)
                if polished:
                    master,pixel_rect=polished
                    size=judged['size']
                    repair_check={'box':[round(v*1000/n) for v,n in zip(pixel_rect,size*2)],
                                  'symptom':judged['views'][-1]['blind_detail_check']['trace']['improvement']}
                    small=None
                    rec['4K散点修复']=master.with_suffix('.json').name
                    save(manifest,rec)
                    continue
            from restoration import available as restoration_available
            local_defect=bool(issue or pixels or finding.get('box')) and judged['status'] in ('local','uncertain')
            needs_detail=pixels or finding.get('category')=='unresolved_focus'
            restored_source=rec.get('当前4K重建底图指纹')
            already_restored=(restored_source==fingerprint(approved) if restored_source else rec.get('初次4K方法')=='SeedVR2')
            if (attempt==0 and needs_detail and not local_defect and not repair_check and not args.candidate_image
                    and not already_restored and args.print_method!='tiled'
                    and native_edge<3840 and restoration_available(RUNTIME/'ComfyUI/models')):
                # Compare against the actual selected drawing. A final with a later structural
                # repair must not be replaced by an earlier, uncorrected source.
                stage='source-restoration'
                reserve_redraw(stage)
                prior_master=master
                master=reconstruct_print(approved,stage,seed+28601)
                small=None
                rec['当前4K重建底图指纹']=fingerprint(approved)
                rec['4K源图重建']={'底图':str(approved),'前一版本':str(prior_master),'模型任务':stage+'-task.json'}
                save(manifest,rec)
                continue
            if pixels and pixels not in tried_focus_fields:
                tried_focus_fields.add(pixels)
                stage=f'print-definition-{attempt+1}'
                revised=repaint_definition(master,pixels,rec.get('画材','')+' '+print_constraints(prompt),seed+17041+attempt,job,stage,
                                           args.port,prefix+'/pending-review/'+stage,before_edit=reserve_redraw,local_image=local_image,
                                           post_graph=post_graph,output_image=output_image,models=RUNTIME/'ComfyUI/models')
                if revised:
                    with Image.open(master) as img:
                        focus=[round(v*1000/n) for v,n in zip(pixels,img.size*2)]
                    repair_check={'box':focus,'symptom':finding['symptom']}
                    receipt=revised.name.removesuffix('-refined.png')+'-definition.json'
                    rec.setdefault('4K焦点重绘',[]).append({'核验':receipt,'范围':focus})
                    master,small=revised,None
                    save(manifest,rec)
                    continue
        if (reference_style and not (args.keep_native or args.selection_mask or args.repair_box or repair_check) and not detail_rebuilt and attempt<2 and
                judged.get('issue',{}).get('category')=='unresolved_focus' and not issue and not finding.get('box') and judged['status']=='global'
                and rebuild_size is not None):
            # A verified focus failure across the picture, or one that survived a local repair,
            # needs a higher-resolution drawing of the scene before another print pass.
            detail_rebuilt=True
            repair_check=None
            stage=f'detail-source-{attempt+1}'
            reserve_redraw(stage)
            with Image.open(approved) as original:
                size=rebuild_size
                reference=local_image(original.convert('RGB'))
            fix=judged['issue'].get('fix') or judged['issue']['symptom']
            caption=('Reconstruct the same composition in its original medium. Keep subject identity, count, pose, placement, palette and light. '
                     'Resolve focal structures at this drawing scale, respecting soft focus and flat regions. Observed problem: '+fix+' Scene: '+prompt)
            graph=reference_draft(caption,seed+830+attempt,*size,prefix+'/'+stage,reference=reference,edit=True)
            history=post_graph(graph.api,args.port,job,stage,graph.ui())
            candidate=output_image(history[str(len(graph.nodes))])
            remember(candidate,stage)
            rebuilt=check(candidate,stage)
            if not usable(rebuilt):break
            from print_judge import same_composition
            if not same_composition(approved,candidate,prompt):break
            rec['高清底图返工']={'源图':str(approved),'重绘':str(candidate),'依据':fix}
            approved=candidate
            master,small=render(approved,stage+'-print')
            continue
        if not issue or attempt==2:break
        focus=region(issue,final)
        repair_check={'box':focus,'symptom':issue.get('symptom') or issue['fix']}
        source=master
        stage=f'print-repair-{attempt+1}'
        with Image.open(source) as original:
            if issue.get('operation')=='remove_forbidden' and ERASE_MODEL.is_file():
                reserve_redraw(stage)
                revised=job/(stage+'-removed-unrequested-object.png')
                corrected=remove_reviewed_feature(original.convert('RGB'),focus,issue)
                temp=revised.with_suffix('.partial.png')
                corrected.save(temp);temp.replace(revised)
            elif reference_style and issue.get('operation')=='redraw':
                revised=correct_shape(source,issue,focus,stage,seed+650+attempt)
                if revised is None:break
            else:
                reserve_redraw(stage)
                revised=repaint(original.convert('RGB'),focus,issue['fix'],rec['positive'],seed+650+attempt,args.port,
                                job,stage,issue.get('target'),[o for o,_ in rec['parts']],reference_style,
                                rebuild_shape=issue.get('operation')=='redraw')
        if not (reference_style and issue.get('operation')=='redraw'):
            revised=confine_edit(source,revised,focus,job/(stage+'-bounded.png'))
        if four_only:master=revised
        else:master,small=native_pair(revised,stage,(small,master))
    if retained_failure:rec.update(retained_failure)
    rec['status']='未达标 · 终审退回'
    save(manifest,rec);print(rec['status'],flush=True);return 2


if __name__=='__main__':
    from contextlib import ExitStack
    with ExitStack() as resources:
        sys.exit(main(resources))
