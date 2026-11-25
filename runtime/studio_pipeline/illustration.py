"""Offline illustration graphs based on the publishers' ComfyUI examples."""
from graphs import Graph

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


def reference_draft(text, seed, width, height, prefix, reference=REFERENCE, layout=None, edit=False):
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
    return flux_canvas(g,positive,negative,vae,seed,width,height,prefix)


def text_draft(text,seed,width,height,prefix):
    """Independent text composition for briefs with their own medium and palette."""
    g=Graph('studio-independent-flux-illustration')
    brief=prompt_field(g,text,(0,0))
    clip=g.add('CLIPLoader','文本编码器',(0,270),['CLIP'],values={'clip_name':'qwen_3_4b.safetensors','type':'flux2','device':'default'})
    vae=g.add('VAELoader','画面编码器',(0,470),['VAE'],values={'vae_name':'flux2-vae.safetensors'})
    pos=g.add('CLIPTextEncode','画面与画材',(390,100),['CONDITIONING'],[('clip',(clip,0)),('text',(brief,0))],widgets=[''],widget_links=['text'])
    neg=g.add('ConditioningZeroOut','空白条件',(740,330),['CONDITIONING'],[('conditioning',(pos,0))])
    return flux_canvas(g,pos,neg,vae,seed,width,height,prefix)


def flux_canvas(g,positive,negative,vae,seed,width,height,prefix):
    model=g.add('UNETLoader','FLUX.2 klein 4B',(1090,75),['MODEL'],values={'unet_name':KLEIN,'weight_dtype':'default'})
    guide=g.add('CFGGuider','蒸馏引导',(1430,300),['GUIDER'],[('model',(model,0)),('positive',(positive,0)),('negative',(negative,0))],{'cfg':1.0})
    noise=g.add('RandomNoise','种子',(1430,90),['NOISE'],values={'noise_seed':seed},widgets=[seed,'fixed'])
    sampler=g.add('KSamplerSelect','采样器',(1430,505),['SAMPLER'],values={'sampler_name':'euler'})
    # Beyond the tested illustration area, the default exponential shift puts
    # the first three distilled steps almost entirely at sigma=1. Keep the
    # proven noise schedule while still drawing on the full requested latent.
    schedule_scale=min(1,(1248*1872/(width*height))**.5)
    sw,sh=(round(n*schedule_scale) for n in (width,height))
    sigma=g.add('Flux2Scheduler','四步去噪曲线',(1430,680),['SIGMAS'],values={'steps':4,'width':sw,'height':sh})
    lat=g.add('EmptyFlux2LatentImage','新画幅',(1430,910),['LATENT'],values={'width':width,'height':height,'batch_size':1})
    sample=g.add('SamplerCustomAdvanced','参考图条件生成',(1790,360),['LATENT','LATENT'],
                 [('noise',(noise,0)),('guider',(guide,0)),('sampler',(sampler,0)),('sigmas',(sigma,0)),('latent_image',(lat,0))])
    inputs=[('samples',(sample,0)),('vae',(vae,0))]
    if max(width,height)>=3840:
        image=g.add('VAEDecodeTiled','分块解码',(2130,360),['IMAGE'],inputs,dict(tile_size=768,overlap=96,temporal_size=64,temporal_overlap=8))
    else:
        image=g.add('VAEDecode','解码',(2130,360),['IMAGE'],inputs)
    g.save('构图候选',(2490,360),image,prefix)
    return g


def print_graph(reference,prefix,seed,native_four=False,medium='',four_steps=4,eight_steps=4,four_only=False):
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
    four=src if native_four else pass_at(src,3840,four_steps,.23,750)
    small=g.save('4K 分块精绘',(2820,0),four,prefix+'/4K')
    if four_only:return g,dict(small=str(small))
    eight=pass_at(four,7680,eight_steps,.16,2820)
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
