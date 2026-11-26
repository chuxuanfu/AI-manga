"""Source-conditioned, single-step diffusion restoration using native ComfyUI nodes."""
from graphs import Graph
RESTORER='seedvr2_3b_fp16.safetensors'
CODEC='seedvr2_ema_vae_fp16.safetensors'
def available(root):
    return (root/'diffusion_models'/RESTORER).is_file() and (root/'vae'/CODEC).is_file()


def prefer_source_reconstruction(method,source_edge,models):
    return (method!='tiled' and 0<source_edge<3840 and (method=='restoration' or source_edge>=1024)
            and available(models))


def restoration_graph(reference,prefix,edge,seed):
    g=Graph('studio-reference-restoration')
    src=g.add('LoadImage','Drawing',(0,0),['IMAGE','MASK'],values={'image':reference},widgets=[reference,'image'])
    guide=g.scale('Print dimensions',(335,0),src,edge)
    normalized=g.add('SeedVR2Preprocess','Prepare drawing',(670,0),['IMAGE'],[('resized_images',(guide,0))])
    vae=g.add('VAELoader','Restoration codec',(330,420),['VAE'],values={'vae_name':CODEC})
    model=g.add('UNETLoader','Image restoration',(0,550),['MODEL'],values={'unet_name':RESTORER,'weight_dtype':'default'})
    tiles={'tile_size':512,'overlap':128,'temporal_size':64,'temporal_overlap':8}
    z=g.add('VAEEncodeTiled','Encode source pixels',(985,0),['LATENT'],[('pixels',(normalized,0)),('vae',(vae,0))],tiles)
    c=g.add('SeedVR2Conditioning','Condition on source geometry',(1000,300),['CONDITIONING','CONDITIONING'],[('model',(model,0)),('vae_conditioning',(z,0))])
    k={'seed':seed,'steps':1,'cfg':1.,'sampler_name':'euler','scheduler':'simple','denoise':1.}
    n=g.add('KSampler','One-step restoration',(1340,0),['LATENT'],[('model',(model,0)),('positive',(c,0)),('negative',(c,1)),('latent_image',(z,0))],k,[seed,'fixed',1,1.,'euler','simple',1.])
    image=g.add('VAEDecodeTiled','Decode print',(1680,0),['IMAGE'],[('samples',(n,0)),('vae',(vae,0))],tiles)
    corrected=g.add('SeedVR2PostProcessing','Restore original palette',(2010,0),['IMAGE'],[('images',(image,0)),('original_resized_images',(guide,0))],{'color_correction_method':'wavelet'})
    end=g.save('Candidate print',(2350,0),corrected,prefix)
    return g,str(end)
