"""Emit paired ComfyUI editor and API graphs from a single definition."""
import uuid

ANIME="animagine-xl-3.1.safetensors"
BASE="sd_xl_base_1.0.safetensors"
UP="RealESRGAN_x4plus_anime_6B.pth"
NEG="nsfw, low quality, blurry, malformed anatomy, extra limbs, fused fingers, duplicated subjects, disconnected parts, inconsistent perspective, textual artifacts, watermark, jpeg artifacts"


class Graph:
    def __init__(self, name):
        self.name=name; self.nodes=[];self.links=[];self.api={}

    def add(self,typ,title,xy,outputs,connections=(),values=None,widgets=None,size=(285,125),widget_links=()):
        idx=len(self.nodes)+1; inputs=dict(values or {});ports=[]
        for param,(src,slot) in connections:
            port=self.nodes[src-1]['outputs'][slot]; lid=len(self.links)+1
            self.links.append([lid,src,slot,idx,len(ports),port['type']]);port['links'].append(lid)
            pin={'name':param,'type':port['type'],'link':lid}
            if param in widget_links:pin['widget']={'name':param}
            ports.append(pin);inputs[param]=[str(src),slot]
        self.nodes.append({'id':idx,'type':typ,'title':title,'pos':list(xy),'size':list(size),'flags':{},'order':idx-1,'mode':0,
                           'inputs':ports,'outputs':[{'name':t,'type':t,'links':[],'slot_index':i} for i,t in enumerate(outputs)],
                           'properties':{'Node name for S&R':typ},'widgets_values':list(widgets if widgets is not None else (values or {}).values())})
        self.api[str(idx)]={'class_type':typ,'inputs':inputs,'_meta':{'title':title}}
        return idx

    def ui(self):
        return {'id':str(uuid.uuid5(uuid.NAMESPACE_DNS,self.name)),'revision':1,'last_node_id':len(self.nodes),
          'last_link_id':len(self.links),'nodes':self.nodes,'links':self.links,'groups':[],'config':{},
          'extra':{'ds':{'scale':.58,'offset':[80,95]}},'version':.4}

    def checkpoint(self,title,xy,filename):
        return self.add('CheckpointLoaderSimple',title,xy,['MODEL','CLIP','VAE'],values={'ckpt_name':filename})
    def text(self,title,xy,model,text):
        if isinstance(text,tuple):
            return self.add('CLIPTextEncode',title,xy,['CONDITIONING'],[('clip',(model,1)),('text',text)],widgets=[''],widget_links=['text'])
        return self.add('CLIPTextEncode',title,xy,['CONDITIONING'],[('clip',(model,1))],{'text':text},size=(330,170))
    def sample(self,title,xy,model,pos,neg,lat,seed,steps,cfg,denoise,sampler='dpmpp_2m'):
        params=dict(seed=seed,steps=steps,cfg=cfg,sampler_name=sampler,scheduler='normal',denoise=denoise)
        return self.add('KSampler',title,xy,['LATENT'],[('model',(model,0)),('positive',(pos,0)),('negative',(neg,0)),('latent_image',(lat,0))],
                        params,[seed,'fixed',steps,cfg,sampler,'normal',denoise],size=(310,280))
    def decode(self,title,xy,sample,model):
        return self.add('VAEDecode',title,xy,['IMAGE'],[('samples',(sample,0)),('vae',(model,2))])
    def encode(self,title,xy,img,model):
        return self.add('VAEEncode',title,xy,['LATENT'],[('pixels',(img,0)),('vae',(model,2))])
    def save(self,title,xy,img,prefix):
        return self.add('SaveImage',title,xy,['IMAGE'],[('images',(img,0))],{'filename_prefix':prefix},size=(310,310))
    def scale(self,title,xy,img,long_edge):
        return self.add('ImageScaleToMaxDimension',title,xy,['IMAGE'],[('image',(img,0))],{'upscale_method':'lanczos','largest_size':long_edge})
    def upscale(self,xy,img,model):
        return self.add('ImageUpscaleWithModel','本机动漫超分',xy,['IMAGE'],[('upscale_model',(model,0)),('image',(img,0))])


def draft(text,seed,width,height,prefix):
    g=Graph('studio-general-draft')
    model=g.checkpoint('构图模型',(0,0),ANIME)
    pos=g.text('画面',(355,0),model,text);neg=g.text('负向',(355,230),model,NEG)
    lat=g.add('EmptyLatentImage','构图尺寸',(355,470),['LATENT'],values=dict(width=width,height=height,batch_size=1))
    sample=g.sample('构图采样',(770,0),model,pos,neg,lat,seed,24,6,1,'euler_ancestral')
    pic=g.decode('底图',(1150,0),sample,model);g.save('备选构图',(1490,0),pic,prefix)
    return g


def finish(reference, request, prefix, seed):
    g=Graph('studio-general-approved-to-print')
    load=g.add('LoadImage','已校样的底图',(0,0),['IMAGE','MASK'],values={'image':reference},widgets=[reference,'image'],size=(300,325))
    lang=g.add('MangaStudioScenePlanner','原始新 prompt · 自动分析',(0,450),['STRING','STRING'],values={'request':request},size=(320,215))
    up=g.add('UpscaleModelLoader','本地超分模型',(340,0),['UPSCALE_MODEL'],values={'model_name':UP})
    x=g.upscale((685,0),load,up); guide=g.scale('全幅细化画布',(1030,0),x,2304)
    split=g.add('SplitImageToTileList','全图分块',(1370,0),['IMAGE'],[('image',(guide,0))],dict(tile_width=1024,tile_height=1024,overlap=144))
    anime=g.checkpoint('全幅画法',(690,480),ANIME)
    pos=g.text('局部画法',(1050,340),anime,'fine anime illustration, carefully inked silhouettes, painterly scenery, preserve existing subjects and shapes, subtle fine surface details, best quality')
    neg=g.text('保护画面',(1050,570),anime,NEG)
    lat=g.encode('编码全幅图块',(1720,0),split,anime);s=g.sample('低强度全幅细化',(2060,0),anime,pos,neg,lat,seed,8,4.5,.22)
    img=g.decode('解码全幅图块',(2440,0),s,anime)
    size=g.add('GetImageSize','读取实际构图比例',(1720,365),['INT','INT','INT'],[('image',(guide,0))])
    merge=g.add('ImageMergeTileList','合并全幅',(2780,0),['IMAGE'],[('image_list',(img,0)),('final_width',(size,0)),('final_height',(size,1))],
                values={'overlap':144},widgets=[1536,2304,144],widget_links=['final_width','final_height'])
    high=g.upscale((3120,0),merge,up);native=g.scale('8K 原位画布',(3460,0),high,7680)
    auto=g.add('MangaStudioAutoMaterialTiles','根据新图自动定位材质',(3820,0),['IMAGE','STRING','MANGA_TILE_PLAN','IMAGE','STRING'],
               [('guide',(native,0)),('composition',(load,0)),('surface_candidates',(lang,1))],size=(340,160))
    base=g.checkpoint('8K 材质模型',(3500,510),BASE)
    positive=g.text('每块独立的材质描述',(3900,395),base,(auto,1))
    negative=g.text('细节约束',(3900,610),base,'plastic surface, new subjects, duplicate outlines, malformed joints, text, noise, block artifacts')
    lat=g.encode('8K 原像素图块',(4240,0),auto,base)
    s=g.sample('目标分辨率重绘',(4620,0),base,positive,negative,lat,seed+1,9,5,.53)
    img=g.decode('解码材质',(5000,0),s,base)
    composite=g.add('MangaStudioMaterialComposite','按遮罩合回审定构图',(5340,0),['IMAGE'],[('base',(native,0)),('painted',(img,0)),('plan',(auto,2))])
    master=g.save('待终审 8K',(5710,0),composite,prefix+'/8K')
    down=g.scale('同母版 4K',(5710,415),composite,3840)
    small=g.save('待终审 4K',(6050,415),down,prefix+'/4K')
    preview=g.add('PreviewImage','检查识别区域',(4250,450),['IMAGE'],[('images',(auto,3))],size=(360,345))
    report=g.add('PreviewAny','实际命中与回退报告',(4660,450),['STRING'],[('source',(auto,4))],size=(400,260))
    return g,dict(master=str(master),small=str(small),preview=str(preview),report=str(report))
