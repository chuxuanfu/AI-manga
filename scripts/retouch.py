"""Loopback-only interface for local generation and masked repair."""
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from urllib.request import urlopen

from PIL import Image, ImageDraw, UnidentifiedImageError

from studio import ROOT, RUNTIME, IMAGES, PYTHON, protected, settings

PORT = 8195
ADDRESS = f'http://127.0.0.1:{PORT}'
UPLOAD_LIMIT = 256 * 1024**2
ID = re.compile(r'[0-9a-f]{32}')


def drawing_mask(size, strokes):
    if not isinstance(strokes, list) or not 1 <= len(strokes) <= 2000:
        raise ValueError('请先在原图上涂出要修改的区域。')
    mask = Image.new('L', size)
    draw = ImageDraw.Draw(mask)
    total = 0
    for stroke in strokes:
        if not isinstance(stroke,dict):
            raise ValueError('笔画格式无效。')
        mode = stroke['tool']
        diameter = stroke['width']
        points = stroke['points']
        if mode not in ('paint','erase') or not isinstance(diameter,(float,int)) or not 0 < diameter <= max(size):
            raise ValueError('画笔参数无效，请重试。')
        if not isinstance(points,list) or not points:
            raise ValueError('笔画为空。')
        total += len(points)
        if total > 200000:
            raise ValueError('笔画过多，请简化选区。')
        coords=[]
        for point in points:
            if (not isinstance(point,list) or len(point)!=2 or
                    not all(isinstance(v,(float,int)) and math.isfinite(v) and 0 <= v <= n for v,n in zip(point,size))):
                raise ValueError('笔画超出画布。')
            coords.append(tuple(min(n-1,round(v)) for v,n in zip(point,size)))
        color = 255 if mode=='paint' else 0
        width = max(1,round(diameter))
        if len(coords)>1:
            draw.line(coords,fill=color,width=width,joint='curve')
        radius=(width-1)/2
        for x,y in coords:
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
            if width==1:
                draw.point((x,y),fill=color)
    if mask.getbbox() is None:
        raise ValueError('选区为空，请用画笔涂出要修改的部分。')
    return mask


def image_info(path):
    with Image.open(path) as im:
        if im.format!='PNG' or max(im.size) not in (3840,7680) or getattr(im,'is_animated',False):
            raise ValueError('请选择项目生成的 4K 或 8K 静态 PNG 原图。')
        if im.getexif().get(274,1)!=1:
            raise ValueError('请先将原图保存为方向已校正的 PNG。')
        size=im.size
    with Image.open(path) as check:
        check.verify()
    return size


def source_prompt(path,digest):
    for parent in path.parents:
        if parent==IMAGES or not parent.is_relative_to(IMAGES):
            break
        for name in ('production.json','review.json'):
            file=parent/name
            if file.is_file():
                text=json.loads(file.read_text()).get('prompt')
                if text:
                    return text
    for file in (IMAGES/'new-jobs').glob('*/production.json'):
        try:
            data=json.loads(file.read_text())
            if data.get('sha256')==digest:
                return data.get('prompt','')
        except (OSError,ValueError):
            continue
    return ''


class Editor(ThreadingHTTPServer):
    daemon_threads=True

    def __init__(self, address, cache):
        super().__init__(address,Handler)
        self.cache=cache
        self.sources={}
        self.jobs={}
        self.guard=threading.Lock()

    def add_source(self,path,original):
        size=image_info(path)
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        token=digest[:32]
        target=self.cache/(token+'.png')
        path.replace(target)
        self.sources[token]=(target,size)
        return dict(id=token,width=size[0],height=size[1],prompt=source_prompt(original,digest))

    def launch(self, token, job, options):
        args=[PYTHON, ROOT/'scripts/studio.py', '--job', job, *options]
        destination=settings().get('output_dir')
        if destination:
            args.extend(['--output-dir',destination])
        with (job/'run.log').open('wb') as log:
            self.jobs[token]=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)

    def generate(self, data):
        if not isinstance(data,dict):
            raise ValueError('生成参数无效。')
        token=data.get('request')
        prompt=data.get('prompt')
        if not isinstance(token,str) or not ID.fullmatch(token):
            raise ValueError('提交编号无效。')
        if not isinstance(prompt,str) or not 1 <= len(prompt.strip()) <= 20000:
            raise ValueError('请填写画面描述（最多 20000 字）。')
        aspect=data.get('aspect','auto')
        quality=data.get('quality','4K')
        style=data.get('style','')
        seed=data.get('seed','')
        rounds=data.get('rounds',1)
        if aspect not in ('auto','2:3','3:2','9:16','16:9','1:1') or quality not in ('4K','8K'):
            raise ValueError('画幅或输出规格无效。')
        if not isinstance(style,str) or len(style)>500 or type(rounds) is not int or not 0<=rounds<=10:
            raise ValueError('画风或轮数无效。')
        if not isinstance(seed,str) or seed and (not re.fullmatch(r'[0-9]{1,19}',seed) or int(seed)>=2**63):
            raise ValueError('种子须在 0 到 9223372036854775807 之间。')
        options=['--prompt='+prompt.strip(), '--aspect',aspect, '--max-rounds',str(rounds)]
        if style.strip():
            options.append('--style='+style.strip())
        if seed:
            options.extend(['--seed',seed])
        if quality=='8K':
            options.append('--8k')
        job=IMAGES/'new-jobs'/('generate-'+token)
        with self.guard:
            if job.is_dir():
                return token
            if token in self.jobs or job.with_name('retouch-'+token).exists():
                raise ValueError('提交编号已被使用。')
            job.mkdir(parents=True)
            try:
                (job/'request.json').write_text(json.dumps({'prompt':prompt.strip()},ensure_ascii=False))
                self.launch(token,job,options)
            except Exception:
                shutil.rmtree(job)
                raise
        return token

    def submit(self,data):
        if not isinstance(data,dict) or not all(isinstance(data.get(k),str) for k in ('request','source','change')) or not isinstance(data.get('prompt',''),str):
            raise ValueError('提交内容无效。')
        token=data['request']
        if not isinstance(token,str) or not ID.fullmatch(token):
            raise ValueError('提交编号无效。')
        text=data['change'].strip()
        if not text or len(text)>6000:
            raise ValueError('请填写简短的修改要求。')
        source,size=self.sources[data['source']]
        with self.guard:
            if token in self.jobs:
                return token
            mask=drawing_mask(size,data['strokes'])
            prompt=data.get('prompt','').strip() or '保留所给原图的主体、构图、姿势、色彩、光照与画风，按选区修改要求完成局部修订。'
            if len(prompt)>20000:
                raise ValueError('原画描述过长。')
            job=IMAGES/'new-jobs'/('retouch-'+token)
            job.mkdir(parents=True)
            try:
                inputs=job/'inputs'
                inputs.mkdir()
                original=inputs/'original.png'
                selected=inputs/'selection.png'
                shutil.copyfile(source,original)
                mask.save(selected)
                options=['--from-image',original,'--selection-mask',selected,'--prompt',prompt,'--repair-text',text,'--max-rounds','1']
                if max(size)==7680:
                    options.append('--keep-native')
                self.launch(token,job,options)
            except Exception:
                shutil.rmtree(job)
                raise
        return token


class Handler(BaseHTTPRequestHandler):
    def send_data(self, data, mime, status=200):
        is_file=isinstance(data,Path)
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(data.stat().st_size if is_file else len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','DENY')
        self.end_headers()
        if is_file:
            with data.open('rb') as stream:
                shutil.copyfileobj(stream,self.wfile)
        else:
            self.wfile.write(data)

    def reply(self,obj,status=200):
        self.send_data(json.dumps(obj,ensure_ascii=False).encode(),'application/json',status)

    def route(self, post):
        host=f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host')!=host:
            return self.reply({'error':'无效的本机地址。'},403)
        if post and (self.headers.get('Origin')!='http://'+host or self.headers.get('X-Studio-Retouch')!='1'):
            return self.reply({'error':'请从本机编辑器提交。'},403)
        request=urlsplit(self.path)
        path=request.path
        if not post and path=='/studio':
            return self.send_data(Path(__file__).with_name('generate.html'),'text/html; charset=utf-8')
        if not post and path=='/':
            return self.send_data(Path(__file__).with_name('retouch.html'),'text/html; charset=utf-8')
        if not post and path=='/identity':
            return self.reply({'studio':str(ROOT)})
        if post and path=='/source':
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=UPLOAD_LIMIT:
                raise ValueError('文件太大或为空。')
            with tempfile.NamedTemporaryFile(dir=self.server.cache,delete=False) as stream:
                tmp=Path(stream.name)
                try:
                    remaining=length
                    while remaining:
                        chunk=self.rfile.read(min(1024**2,remaining))
                        if not chunk:
                            raise ValueError('文件传输中断。')
                        stream.write(chunk)
                        remaining-=len(chunk)
                    stream.flush()
                    return self.reply(self.server.add_source(tmp,tmp))
                finally:
                    tmp.unlink(missing_ok=True)
        if post and path=='/library-source':
            item=parse_qs(request.query).get('image',[''])[0]
            original=(IMAGES/item).resolve()
            if not original.is_relative_to(IMAGES.resolve()) or original.suffix.lower()!='.png' or not original.is_file():
                raise ValueError('图库中找不到这张原图。')
            tmp=self.server.cache/uuid.uuid4().hex
            try:
                shutil.copyfile(original,tmp)
                return self.reply(self.server.add_source(tmp,original))
            finally:
                tmp.unlink(missing_ok=True)
        if not post and path=='/generations':
            jobs=sorted((IMAGES/'new-jobs').glob('generate-*'),key=lambda j:j.stat().st_mtime,reverse=True)
            recent=[]
            for job in jobs:
                token=job.name.removeprefix('generate-')
                if ID.fullmatch(token) and (job/'request.json').is_file():
                    recent.append(dict(id=token,prompt=json.loads((job/'request.json').read_text())['prompt'][:100]))
                if len(recent)==25:
                    break
            return self.reply(recent)
        if post and path in ('/jobs','/generations'):
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8*1024**2:
                raise ValueError('选区数据大小无效。')
            submit=self.server.generate if path=='/generations' else self.server.submit
            return self.reply({'job':submit(json.loads(self.rfile.read(length)))},202)
        parts=path.strip('/').split('/')
        if not post and len(parts)==2 and parts[0]=='source' and ID.fullmatch(parts[1]):
            return self.send_data(self.server.sources[parts[1]][0],'image/png')
        if not post and len(parts) in (2,3) and parts[0]=='jobs' and ID.fullmatch(parts[1]):
            job=IMAGES/'new-jobs'/('retouch-'+parts[1])
            if not job.is_dir():
                job=job.with_name('generate-'+parts[1])
            if not job.is_dir():
                return self.reply({'error':'找不到任务。'},404)
            state=json.loads((job/'production.json').read_text()) if (job/'production.json').is_file() else {}
            proc=self.server.jobs.get(parts[1])
            done=state.get('status')=='已自动定稿'
            filename='8K.png' if state.get('delivery_edge')==7680 else '4K.png'
            available=[name for name in ('4K.png','8K.png') if done and (job/name).is_file()]
            if len(parts)==3:
                name=filename if parts[2]=='image' else parts[2]
                if name in available:
                    return self.send_data(job/name,'image/png')
                return self.reply({'error':'暂无定稿。'},404)
            ended=proc is not None and proc.poll() is not None
            rounds=state.get('rounds') or []
            last=rounds[-1] if rounds else {}
            terminal=state.get('status','').startswith(('达到指定轮数','本地运行故障'))
            message=state.get('status') or '正在准备本地模型'
            if not done and not terminal and proc is None:
                # A detached CLI worker can still own this job after the UI restarts.
                lines=subprocess.check_output(['ps','-axo','command='],text=True).splitlines()
                ended=not any(str(job) in line and ('studio.py' in line or 'run.py' in line) for line in lines)
            if done and filename not in available:
                message='定稿文件缺失，请查看任务目录'
            if ended and not done and not terminal:
                message='任务已结束，请查看运行日志'
            log=job/'run.log'
            detail=''
            if log.is_file():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-4000))
                    detail=stream.read().decode('utf-8','replace').strip()[-2000:]
            return self.reply(dict(status=message,reason=last.get('原因',''),round=len(rounds),log=detail,
                                   done=ended or done or terminal,accepted=filename in available,folder=str(job),
                                   images=available,relative=job.relative_to(IMAGES).as_posix(),exported=state.get('exported_to')))
        self.reply({'error':'找不到页面。'},404)

    def dispatch(self,post):
        try:
            self.route(post)
        except (ValueError,TypeError,KeyError,OSError,Image.DecompressionBombError,UnidentifiedImageError) as err:
            if isinstance(err,(BrokenPipeError,ConnectionResetError)):
                return
            self.reply({'error':str(err) or '文件或选区无效。'},400)

    def do_GET(self):
        self.dispatch(False)

    def do_POST(self):
        self.dispatch(True)


def start(open_page=True, page="/"):
    def ready():
        try:
            with urlopen(ADDRESS+'/identity',timeout=1) as response:
                return json.load(response).get('studio')==str(ROOT)
        except (OSError,ValueError):
            return False
    if not ready():
        with (RUNTIME/'retouch.log').open('ab') as log:
            proc=subprocess.Popen(protected([PYTHON,Path(__file__),'--serve']),stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=log,start_new_session=True)
        for _ in range(50):
            if ready():
                break
            if proc.poll() is not None:
                raise SystemExit('局部微调服务未能启动，请查看 runtime/retouch.log。')
            time.sleep(.1)
        else:
            raise SystemExit('局部微调服务启动超时。')
    if open_page:
        webbrowser.open(ADDRESS+page)
    return ADDRESS+page


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='ai-manga-retouch-') as folder, Editor(('127.0.0.1',PORT),Path(folder)) as app:
        app.serve_forever()
