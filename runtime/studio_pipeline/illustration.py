"""Offline illustration graphs based on the publishers' ComfyUI examples."""
from graphs import Graph, UP

KEY_VISUAL = ('Japanese anime key visual, clean tapered linework with varied line weight. Group fine strokes into readable forms, picking out a few crisp accents at the focal point. '
              'Model materials with deliberate shadow shapes, controlled highlights and restrained reflected light. Keep requested local colors identifiable in light and shadow. '
              'Maintain a clear value hierarchy and quieter distant edges. Follow the scene lighting and palette in the brief.')

REFERENCE = 'Studio-illustration-reference.png'
BRUSHWORK = 'Studio-illustration-brushwork.png'
KLEIN = 'flux-2-klein-4b.safetensors'
TURBO = 'anima-turbo-v1.1.safetensors'
AESTHETIC = 'anima-aesthetic-v1.1.safetensors'
DIRECTION = ('Create a new finished Japanese storybook illustration of the requested scene. Follow the requested medium and palette. Use the supplied painting for the level of finish, material construction, observed lighting and spatial coherence. '
             'Design a different composition appropriate to the new scene. Follow its specified people, animals, action and location exactly. '
             'Give the primary subjects enough space and scale to read. Let folds turn around forms, cast shadows meet their sources, and nearby surfaces share reflected light. '
             'Preserve quiet resting areas around the focal point. Render the requested scene: ')
NEGATIVE = 'unintentionally blurry, waxy skin, artificial noise, oversharpened edges, broken contours, duplicate fingers, floating props, watermark, nsfw'


def prompt_field(graph, text, position):
    return graph.add('PrimitiveStringMultiline','本次画面描述 · 修改这里',position,['STRING'],
                     values={'value':text},size=(440,240))


def encode_brief(graph, clip, brief, direction, title, position):
    x,y=position
    combined=graph.add('StringConcatenate',title+' · 画法',(x,y),['STRING'],[('string_b',(brief,0))],
                       values={'string_a':direction,'delimiter':' '},widgets=[direction,'',' '],widget_links=['string_b'])
    return graph.add('CLIPTextEncode',title,(x+335,y),['CONDITIONING'],[('clip',(clip,0)),('text',(combined,0))],
                     widgets=[''],widget_links=['text'])


EDIT_UNET='qwen_image_edit_2511_int8_convrot.safetensors'
EDIT_CLIP='qwen_2.5_vl_7b.safetensors'
EDIT_LORA='Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'


def installed_editor(model_root):
    if not (model_root/'text_encoders'/EDIT_CLIP).is_file() or not (model_root/'loras'/EDIT_LORA).is_file():
        return None
    for name in (EDIT_UNET,'qwen_image_edit_2511_bf16.safetensors'):
        if (model_root/'diffusion_models'/name).is_file():return name
    return None


DRAW_UNET='qwen_image_2512_int8_convrot.safetensors'
DRAW_LORA='Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors'


def has_drawing_model(model_root):
    files=(('diffusion_models',DRAW_UNET),('loras',DRAW_LORA),('text_encoders',EDIT_CLIP),
           ('vae','qwen_image_vae.safetensors'))
    return all((model_root/folder/name).is_file() for folder,name in files)


def text_composition(text,seed,width,height,prefix):
    """Qwen-Image-2512 draws from text on an empty canvas with its own Lightning weights."""
    g=Graph('studio-qwen-text-composition')
    unet=g.add('UNETLoader','Drawing model',(0,0),['MODEL'],values={'unet_name':DRAW_UNET,'weight_dtype':'default'})
    clip=g.add('CLIPLoader','Text encoder',(0,205),['CLIP'],values={'clip_name':EDIT_CLIP,'type':'qwen_image','device':'default'})
    vae=g.add('VAELoader','Image codec',(0,410),['VAE'],values={'vae_name':'qwen_image_vae.safetensors'})
    model=g.add('LoraLoaderModelOnly','Four-step drawing',(340,0),['MODEL'],[('model',(unet,0))],
                values={'lora_name':DRAW_LORA,'strength_model':1.})
    model=g.add('ModelSamplingAuraFlow','Sampling shift',(695,0),['MODEL'],[('model',(model,0))],values={'shift':3.1})
    pos=g.add('CLIPTextEncode','Art direction',(340,190),['CONDITIONING'],[('clip',(clip,0))],{'text':text})
    neg=g.add('CLIPTextEncode','Unconditional',(340,425),['CONDITIONING'],[('clip',(clip,0))],{'text':''})
    lat=g.add('EmptySD3LatentImage','Blank canvas',(695,420),['LATENT'],
              values={'width':width,'height':height,'batch_size':1})
    params=dict(seed=seed,steps=4,cfg=1.,sampler_name='euler',scheduler='simple',denoise=1.)
    sample=g.add('KSampler','Draw composition',(1020,125),['LATENT'],[('model',(model,0)),('positive',(pos,0)),
                 ('negative',(neg,0)),('latent_image',(lat,0))],params,[seed,'fixed',4,1.,'euler','simple',1.])
    decoded=g.add('VAEDecode','Decode',(1340,145),['IMAGE'],[('samples',(sample,0)),('vae',(vae,0))])
    g.save('Local drawing',(1620,145),decoded,prefix)
    return g


def instruction_edit(text, seed, reference, prefix, unet_name=EDIT_UNET, *, masked=False, detail_reference=None):
    """Comfy's Qwen-Image-Edit-2511 Lightning graph for a local repair crop."""
    g=Graph('studio-instruction-edit')
    source=g.add('LoadImage','Source crop',(0,0),['IMAGE','MASK'],values={'image':reference})
    unet=g.add('UNETLoader','Edit model',(0,300),['MODEL'],values={'unet_name':unet_name,'weight_dtype':'default'})
    clip=g.add('CLIPLoader','Instruction encoder',(0,450),['CLIP'],values={'clip_name':EDIT_CLIP,'type':'qwen_image','device':'default'})
    vae=g.add('VAELoader','Image codec',(0,580),['VAE'],values={'vae_name':'qwen_image_vae.safetensors'})
    model=g.add('LoraLoaderModelOnly','Four-step edit',(350,300),['MODEL'],[('model',(unet,0))],
                values={'lora_name':EDIT_LORA,'strength_model':1.0})
    model=g.add('ModelSamplingAuraFlow','Edit timestep shift',(670,300),['MODEL'],[('model',(model,0))],values={'shift':3.1})
    inputs=[('clip',(clip,0)),('vae',(vae,0)),('image1',(source,0))]
    if detail_reference:
        detail=g.add('LoadImage','Detail of the existing drawing',(0,760),['IMAGE','MASK'],values={'image':detail_reference})
        inputs.append(('image2',(detail,0)))
    pos=g.add('TextEncodeQwenImageEditPlus','Edit instruction',(350,0),['CONDITIONING'],inputs,values={'prompt':text})
    neg=g.add('TextEncodeQwenImageEditPlus','Unconditional',(350,170),['CONDITIONING'],inputs,values={'prompt':''})
    latent=g.add('VAEEncode','Crop latent',(680,0),['LATENT'],[('pixels',(source,0)),('vae',(vae,0))])
    if masked:
        latent=g.add('SetLatentNoiseMask','Preserve surrounding drawing',(820,110),['LATENT'],
                     [('samples',(latent,0)),('mask',(source,1))])
    sample=g.add('KSampler','Repair crop',(990,0),['LATENT'],[('model',(model,0)),('positive',(pos,0)),
                 ('negative',(neg,0)),('latent_image',(latent,0))],values={
                 'seed':seed,'steps':4,'cfg':1.0,'sampler_name':'euler','scheduler':'simple','denoise':1.0})
    decoded=g.add('VAEDecode','Decode',(1300,0),['IMAGE'],[('samples',(sample,0)),('vae',(vae,0))])
    g.save('Edited region',(1600,0),decoded,prefix)
    return g


def reference_draft(text, seed, width, height, prefix, reference=REFERENCE, layout=None, edit=False, masked_source=None, preserve_layout=False):
    g=Graph('studio-reference-conditioned-illustration')
    brief=prompt_field(g,text,(-2600,-60))
    if layout and reference==REFERENCE:reference=BRUSHWORK
    src=g.add('LoadImage','画法参考',(0,0),['IMAGE','MASK'],values={'image':reference},widgets=[reference,'image'],size=(310,320))
    # The publisher resizes the reference to one megapixel before reference conditioning.
    scaled=g.add('ImageScaleToTotalPixels','参考画幅',(365,0),['IMAGE'],[('image',(src,0))],
                 values={'upscale_method':'lanczos','megapixels':1.0,'resolution_steps':1})
    vae=g.add('VAELoader','画面编码器',(365,220),['VAE'],values={'vae_name':'flux2-vae.safetensors'})
    enc=g.add('VAEEncode','参考图视觉信息',(740,0),['LATENT'],[('pixels',(scaled,0)),('vae',(vae,0))])
    clip=g.add('CLIPLoader','文本编码器',(0,440),['CLIP'],values={'clip_name':'qwen_3_4b.safetensors','type':'flux2','device':'default'})
    direction=DIRECTION
    layout_latent=None
    if layout:
        if layout is True:
            scale=min(1,(1536**2/(width*height))**.5)
            sw,sh=(round(n*scale/16)*16 for n in (width,height))
            structure=layout_pass(g,brief,seed,sw,sh)
        else:
            structure=g.add('LoadImage','本次构图',(-375,-250),['IMAGE','MASK'],values={'image':layout},widgets=[layout,'image'])
        resized=g.add('ImageScaleToTotalPixels','构图上下文',(365,-270),['IMAGE'],[('image',(structure,0))],
                      values={'upscale_method':'lanczos','megapixels':1.0,'resolution_steps':1})
        layout_latent=g.add('VAEEncode','构图参考',(740,-270),['LATENT'],[('pixels',(resized,0)),('vae',(vae,0))])
        direction=('Repaint image 1 as an original finished Japanese illustration. Image 1 supplies the new scene, poses, framing and objects. '
                   'Image 2 is a brushwork sample sheet. Borrow only how light and shade, contours and surfaces are painted. Draw one continuous scene from image 1. Preserve its subject count, silhouettes, clothing, anatomy and background layout. '
                   'Resolve vague marks and contact in image 1. Keep its characters and scene, with their own colours and costume. New scene: ')
    if edit:
        direction=('Edit the supplied crop in place in its original medium. Preserve camera angle, scale, palette, illumination and all surrounding borders. '
                   'Redraw the described construction. Only objects belonging inside this close-up may appear. Specific correction: ')
    pos=encode_brief(g,clip,brief,direction,'新图描述',(315,430))
    zero=g.add('ConditioningZeroOut','空白条件',(790,620),['CONDITIONING'],[('conditioning',(pos,0))])
    def condition(node,y):
        if layout_latent:
            node=g.add('ReferenceLatent','第一张 · 构图',(755,y-75),['CONDITIONING'],[('conditioning',(node,0)),('latent',(layout_latent,0))])
        return g.add('ReferenceLatent','结合画法参考',(1090,y),['CONDITIONING'],[('conditioning',(node,0)),('latent',(enc,0))])
    positive=condition(pos,305);negative=condition(zero,545)
    initial=None
    if masked_source:
        src=g.add('LoadImage','Original crop and edit mask',(740,760),['IMAGE','MASK'],values={'image':masked_source})
        encoded=g.add('VAEEncode','Preserve original geometry',(1060,760),['LATENT'],[('pixels',(src,0)),('vae',(vae,0))])
        initial=g.add('SetLatentNoiseMask','Limit sampling to the repair',(1260,900),['LATENT'],
                      [('samples',(encoded,0)),('mask',(src,1))])
    return flux_canvas(g,positive,negative,vae,seed,width,height,prefix,initial=initial,preserve_layout=preserve_layout)


ANIME_PAINT='animagine-xl-4.0-opt.safetensors'


def anime_color_pass(tags,seed,prefix,layout,strength=.64):
    """Retain a layout while painting its materials with an anime-specialized SDXL checkpoint."""
    g=Graph('studio-animagine-key-visual')
    model=g.checkpoint('商业插画上色',(0,0),ANIME_PAINT)
    positive=g.text('场景与颜色标签',(350,0),model,tags+', painterly anime illustration, delicate colored lineart, reflected light, precise material highlights, detailed background, safe, masterpiece, high score, great score, absurdres')
    negative=g.text('负向提示词',(350,200),model,'lowres, blurry focal subject, bad anatomy, bad hands, extra limbs, watermark, low quality, worst quality, color banding, uniform heavy outlines')
    source=g.add('LoadImage','本次构图',(0,320),['IMAGE','MASK'],values={'image':layout},widgets=[layout,'image'])
    guide=g.scale('上色画布',(350,430),source,1536)
    latent=g.encode('构图编码',(690,300),guide,model)
    sample=g.sample('保留构图并上色',(1010,100),model,positive,negative,latent,seed,25,5,strength,'euler_ancestral')
    picture=g.decode('解码',(1350,100),sample,model)
    g.save('上色校样',(1680,100),picture,prefix)
    return g


def paint_key_visual(brief,seed,size,prefix,layout):
    """Use the selected layout and an independent brushwork sheet for the color pass."""
    scale=max(1,2304/max(size))
    width,height=(round(v*scale/16)*16 for v in size)
    direction=('High-end painted anime key visual. Refine the approved framing and objects of reference 1. Let planes of color model the forms. Selective fine colored contours lose and find their edges in the light. '
               'Let highlights break across overlapping forms, use delicate reflected color in shadow, varied satin and matte surfaces, intentional close details and atmospheric separation. '
               'Avoid uniform black outlines, flat sticker-like shapes and repeated hatching. Keep all requested local colors. ')
    return reference_draft(direction+brief,seed,width,height,prefix,layout=layout,reference=BRUSHWORK)


def text_draft(text,seed,width,height,prefix):
    """Independent text composition for briefs with their own medium and palette."""
    g=Graph('studio-independent-flux-illustration')
    brief=prompt_field(g,text,(0,0))
    clip=g.add('CLIPLoader','文本编码器',(0,270),['CLIP'],values={'clip_name':'qwen_3_4b.safetensors','type':'flux2','device':'default'})
    vae=g.add('VAELoader','画面编码器',(0,470),['VAE'],values={'vae_name':'flux2-vae.safetensors'})
    pos=g.add('CLIPTextEncode','画面与画材',(390,100),['CONDITIONING'],[('clip',(clip,0)),('text',(brief,0))],widgets=[''],widget_links=['text'])
    neg=g.add('ConditioningZeroOut','空白条件',(740,330),['CONDITIONING'],[('conditioning',(pos,0))])
    return flux_canvas(g,pos,neg,vae,seed,width,height,prefix)


def flux_canvas(g,positive,negative,vae,seed,width,height,prefix,*,initial=None,preserve_layout=False):
    model=g.add('UNETLoader','FLUX.2 klein 4B',(1090,75),['MODEL'],values={'unet_name':KLEIN,'weight_dtype':'default'})
    guide=g.add('CFGGuider','蒸馏引导',(1430,300),['GUIDER'],[('model',(model,0)),('positive',(positive,0)),('negative',(negative,0))],{'cfg':1.0})
    noise=g.add('RandomNoise','种子',(1430,90),['NOISE'],values={'noise_seed':seed},widgets=[seed,'fixed'])
    sampler=g.add('KSamplerSelect','采样器',(1430,505),['SAMPLER'],values={'sampler_name':'euler'})
    # Beyond the tested illustration area, the default exponential shift puts
    # the first three distilled steps almost entirely at sigma=1. Keep the
    # proven noise schedule while still drawing on the full requested latent.
    schedule_scale=min(1,(1248*1872/(width*height))**.5)
    sw,sh=(round(n*schedule_scale) for n in (width,height))
    # Keep four denoising evaluations. The first local attempt starts later on
    # the actual model schedule, retaining the source geometry inside the mask.
    steps=16 if initial and preserve_layout else 4
    sigma=g.add('Flux2Scheduler','四步去噪曲线',(1430,680),['SIGMAS'],values={'steps':steps,'width':sw,'height':sh})
    schedule=(sigma,0)
    if steps>4:
        trimmed=g.add('SplitSigmas','Four-step local reconstruction',(1630,730),['SIGMAS','SIGMAS'],
                      [('sigmas',schedule)],values={'step':steps-4})
        schedule=(trimmed,1)
    lat=initial or g.add('EmptyFlux2LatentImage','新画幅',(1430,910),['LATENT'],values={'width':width,'height':height,'batch_size':1})
    sample=g.add('SamplerCustomAdvanced','参考图条件生成',(1790,360),['LATENT','LATENT'],
                 [('noise',(noise,0)),('guider',(guide,0)),('sampler',(sampler,0)),('sigmas',schedule),('latent_image',(lat,0))])
    inputs=[('samples',(sample,0)),('vae',(vae,0))]
    if max(width,height)>=3840 or width*height>2048*1152:
        image=g.add('VAEDecodeTiled','分块解码',(2130,360),['IMAGE'],inputs,dict(tile_size=768,overlap=96,temporal_size=64,temporal_overlap=8))
    else:
        image=g.add('VAEDecode','解码',(2130,360),['IMAGE'],inputs)
    g.save('构图候选',(2490,360),image,prefix)
    return g


def detail_canvas(size):
    """Increase sampling density on a rejected draft within the native 4K budget."""
    longest = max(size)
    if longest >= 3840:
        return None
    usual = min(3072 / longest, 2304 / (size[0] * size[1])**.5)
    scale = min(3840 / longest, max(4/3, usual))
    return tuple(max(64, round(n * scale / 16) * 16) for n in size)


NATURAL_UP = 'RealESRGAN_x2plus.pth'

def continuous_tone(medium):
    terms = ('photograph', 'photogram', 'cyanotype', 'film grain', 'silver gelatin',
             'watercolor', 'watercolour', 'gouache', 'pastel', 'charcoal', 'graphite',
             'oil paint', 'textile', 'embroidery', 'tapestry', 'wool', 'acrylic',
             '摄影', '氰版', '蓝晒', '水彩', '水粉', '油画', '粉彩', '炭笔', '刺绣', '织物')
    return any(term in medium.casefold() for term in terms)


def continuous_tone_double(reference, prefix):
    g = Graph('studio-continuous-tone-super-resolution')
    source = g.add('LoadImage', 'Source', (0,0), ['IMAGE','MASK'],
                   values={'image': reference}, widgets=[reference,'image'])
    model = g.add('UpscaleModelLoader', 'Continuous-tone 2x', (0,280), ['UPSCALE_MODEL'],
                  values={'model_name': NATURAL_UP})
    image = g.upscale((380,0), source, model)
    output = g.save('2x print guide', (750,0), image, prefix)
    return g, str(output)


def print_graph(reference,prefix,seed,native_four=False,medium='',four_steps=4,eight_steps=4,four_only=False,upscale_model=UP):
    g=Graph('studio-illustration-print')
    src=g.add('LoadImage','已选插画',(0,0),['IMAGE','MASK'],values={'image':reference},widgets=[reference,'image'])
    model=g.add('UNETLoader','Anima Turbo',(0,240),['MODEL'],values={'unet_name':TURBO,'weight_dtype':'default'})
    clip=g.add('CLIPLoader','Anima 文本编码器',(0,450),['CLIP'],values={'clip_name':'qwen_3_06b_base.safetensors','type':'stable_diffusion','device':'default'})
    vae=g.add('VAELoader','Anima 画面编码器',(0,650),['VAE'],values={'vae_name':'qwen_image_vae.safetensors'})
    positive=g.add('CLIPTextEncode','局部画法',(370,420),['CONDITIONING'],[('clip',(clip,0))],{'text':('A detail from a finished illustration. '+medium+' Preserve the specified medium and palette. Refine marks already supported by the source. Keep quiet fills smooth, preserve the light direction and contours, and resolve existing material edges.').strip()})
    negative=g.add('ConditioningZeroOut','空白条件',(750,550),['CONDITIONING'],[('conditioning',(positive,0))])
    def pass_at(image,edge,steps,strength,x):
        guide=g.scale(f'{edge} 像素画布',(x,0),image,edge)
        split=g.add('MangaStudioMaterialTiles','全幅均匀分块',(x+345,0),['IMAGE','STRING','MANGA_TILE_PLAN'],
                    [('image',(guide,0)),('source_mask',(src,1))],
                    dict(profile='整幅通用重绘',study='none',material='painted illustration'))
        lat=g.add('VAEEncode','编码图块',(x+690,0),['LATENT'],[('pixels',(split,0)),('vae',(vae,0))])
        params=dict(seed=seed+edge,steps=steps,cfg=1.0,sampler_name='euler',scheduler='simple',denoise=strength)
        sample=g.add('KSampler','按新像素重绘',(x+1020,0),['LATENT'],[('model',(model,0)),('positive',(positive,0)),('negative',(negative,0)),('latent_image',(lat,0))],
                     params,[seed+edge,'fixed',steps,1.0,'euler','simple',strength],size=(310,265))
        decoded=g.add('VAEDecode','解码图块',(x+1380,0),['IMAGE'],[('samples',(sample,0)),('vae',(vae,0))])
        return g.add('MangaStudioMaterialComposite','保留原稿线条与明暗',(x+1710,0),['IMAGE'],
                     [('base',(guide,0)),('painted',(decoded,0)),('plan',(split,2))])
    if native_four:
        four=src
    else:
        # Restore contours on the input lattice before diffusion sees a 4K canvas.
        up=g.add('UpscaleModelLoader','构图轮廓超分',(300,-300),['UPSCALE_MODEL'],values={'model_name':upscale_model})
        restored=g.upscale((630,-300),src,up)
        four=pass_at(restored,3840,four_steps,.34,750)
    small=g.save('4K 分块精绘',(2820,0),four,prefix+'/4K')
    if four_only:return g,dict(small=str(small))
    eight=pass_at(four,7680,eight_steps,.32,2820)
    master=g.save('8K 分块精绘',(4910,0),eight,prefix+'/8K')
    return g,dict(master=str(master),small=str(small))


def layout_pass(g,brief,seed,width,height):
    model=g.add('UNETLoader','插画构图模型',(-2050,-280),['MODEL'],values={'unet_name':AESTHETIC,'weight_dtype':'default'})
    clip=g.add('CLIPLoader','构图文本编码器',(-2050,-80),['CLIP'],values={'clip_name':'qwen_3_06b_base.safetensors','type':'stable_diffusion','device':'default'})
    vae=g.add('VAELoader','构图编码器',(-2050,140),['VAE'],values={'vae_name':'qwen_image_vae.safetensors'})
    direction='Honor the specified medium, palette and graphic treatment. If style is unspecified, use a Japanese illustrated storybook look. Expressive gestures, consistent supports and directional light. Paint this scene:'
    pos=encode_brief(g,clip,brief,direction,'场景与画法',(-1700,-365))
    neg=g.add('CLIPTextEncode','构图约束',(-1690,0),['CONDITIONING'],[('clip',(clip,0))],{'text':NEGATIVE})
    lat=g.add('EmptyLatentImage','原生构图尺寸',(-1690,200),['LATENT'],values={'width':width,'height':height,'batch_size':1})
    params=dict(seed=seed,steps=32,cfg=4.0,sampler_name='euler',scheduler='simple',denoise=1.0)
    sm=g.add('KSampler','原生插画构图',(-1270,-270),['LATENT'],[('model',(model,0)),('positive',(pos,0)),('negative',(neg,0)),('latent_image',(lat,0))],params,[seed,'fixed',32,4.0,'euler','simple',1.0])
    return g.add('VAEDecode','完整构图',(-880,-270),['IMAGE'],[('samples',(sm,0)),('vae',(vae,0))])

def independent_draft(text,seed,width,height,prefix):
    g=Graph('studio-text-illustration')
    brief=prompt_field(g,text,(-2600,-60))
    composition=layout_pass(g,brief,seed,width,height)
    g.save('全新构图',(-500,-270),composition,prefix)
    return g
