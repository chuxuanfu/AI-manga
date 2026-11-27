"""Interactive and CLI entry for the same local studio."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib import request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
PYTHON = RUNTIME / 'venv/bin/python'
PIPELINE = RUNTIME / 'studio_pipeline'
IMAGES = ROOT / 'images'
SETTINGS = RUNTIME / 'settings.json'
MAC = sys.platform == 'darwin'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OLLAMA_NO_CLOUD='1',
                  PYTORCH_ENABLE_MPS_FALLBACK='1', TOKENIZERS_PARALLELISM='false')
os.environ['PATH'] = str(RUNTIME / 'tools/bin') + os.pathsep + os.environ.get('PATH','')

def protected(command):
    if MAC:
        return ['/usr/bin/sandbox-exec', '-f', str(RUNTIME/'local-only.sb'), *command]
    return command

def server(port=8193):
    with (RUNTIME/f'server-{port}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _ensure_server(port)


def _ensure_server(port):
    entry = RUNTIME/'ComfyUI/main.py'
    address = f'http://127.0.0.1:{port}'
    def ready():
        try:
            with request.urlopen(address+'/system_stats', timeout=3) as response:
                args = json.load(response)['system']['argv']
        except OSError:
            return False
        if not args or Path(args[0]).resolve() != entry.resolve():
            raise SystemExit(f'Port {port} is in use by another installation.')
        with request.urlopen(address+'/object_info/MangaStudioAutoMaterialTiles',timeout=6) as response:
            if not json.load(response).get('MangaStudioAutoMaterialTiles'):
                raise SystemExit('Missing studio nodes. Run Studio.command --install.')
        return True
    if not ready():
        cmd=[PYTHON,entry,'--listen','127.0.0.1','--port',str(port),'--offline','--fp32-vae',
             '--extra-model-paths-config',RUNTIME/'model-paths.yaml','--output-directory',IMAGES,'--disable-auto-launch']
        with (RUNTIME/f'server-{port}.log').open('ab') as log:
            proc=subprocess.Popen(protected(cmd),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        for _ in range(180):
            if ready():break
            if proc.poll() is not None:raise SystemExit(f'Server failed. See runtime/server-{port}.log')
            time.sleep(1)
        else:raise SystemExit(f'Server startup timed out. See runtime/server-{port}.log')
    return address

def settings():
    return json.loads(SETTINGS.read_text()) if SETTINGS.is_file() else {}

def set_output(value):
    cfg=settings()
    if value:
        folder=Path(value).expanduser().resolve()
        folder.mkdir(parents=True,exist_ok=True)
        cfg['output_dir']=str(folder)
    else:cfg.pop('output_dir',None)
    with tempfile.NamedTemporaryFile(mode='w',dir=RUNTIME,delete=False) as stream:
        json.dump(cfg,stream,indent=2)
    Path(stream.name).replace(SETTINGS)
    print('默认成品位置',cfg.get('output_dir',IMAGES/'new-jobs'))

def main(args):
    if not args or args==['--ui']:
        from retouch import start
        print(start(page='/studio'))
        return
    interactive=args==['--menu']
    background = '--background' in args
    if background:
        args=[arg for arg in args if arg!='--background']
        if not args:raise SystemExit('--background requires a prompt or --job')
    if interactive:
        print('AI Manga Studio\n\n1  生成 4K 壁纸（默认）\n2  打开图库\n3  设置成品位置\n4  仅生成底图\n5  打开节点编辑器\n6  在原图上涂抹微调\n0  退出')
        choice=input('选择 [1] ').strip() or '1'
        if choice=='0':return
        if choice=='3':
            set_output(input('输入文件夹路径，留空恢复默认：').strip());return
        if choice in ('1','4'):
            scene=input('画面描述：').strip()
            if not scene:return
            args=['--prompt',scene]+(['--draft'] if choice=='4' else [])
        elif choice in ('2','5','6'):args=[{'2':'--gallery','5':'--editor','6':'--retouch'}[choice]]
        else:raise SystemExit('请输入菜单中的数字')
    if args==['--advanced-help']:
        subprocess.run([PYTHON,PIPELINE/'run.py','--help'],check=True);return
    if args==['--retouch']:
        from retouch import start
        print(start());return
    if args==['--gallery']:
        from retouch import start
        start(open_page=False)
        subprocess.run([PYTHON,PIPELINE/'library.py'],check=True)
        webbrowser.open((IMAGES/'gallery.html').as_uri());return
    if args==['--editor']:
        webbrowser.open(server()+'/?workflows=MangaStudio');return
    if len(args)==2 and args[0]=='--set-output-dir':
        set_output(args[1]);return
    if args==['--reset-output-dir']:
        set_output(None);return
    if '--draft' in args:
        args=[a if a!='--draft' else '--native-only' for a in args]
        if not any(a=='--candidates' or a.startswith('--candidates=') for a in args):
            args+=['--candidates','1']
    # Resume uses the destination persisted in its job unless explicitly overridden.
    if not any(a.split('=')[0] in ('-o','--output-dir','--job') or (a.startswith('-o') and not a.startswith('--')) for a in args):
        folder=settings().get('output_dir')
        if folder:args+=['--output-dir',folder]
    port=8193
    for i,arg in enumerate(args):
        if arg=='--port' and i+1<len(args):port=int(args[i+1])
        elif arg.startswith('--port='):port=int(arg.split('=',1)[1])
    if port not in (8193,8194):raise SystemExit('Available ports: 8193 and 8194')
    server(port)
    cmd=protected([PYTHON,PIPELINE/'run.py',*args])
    if MAC:cmd=['/usr/bin/caffeinate','-i',*cmd]
    if interactive or background:
        folder=IMAGES/'new-jobs';folder.mkdir(parents=True,exist_ok=True)
        supplied=next((arg.split('=',1)[1] for arg in args if arg.startswith('--job=')),None)
        if '--job' in args:
            index=args.index('--job')
            if index+1>=len(args):raise SystemExit('--job requires a directory')
            supplied=args[index+1]
        if supplied:
            job=Path(supplied).expanduser().resolve()
            if not job.is_relative_to(IMAGES):raise SystemExit('--job must be inside images/')
            job.mkdir(parents=True,exist_ok=True)
        else:
            job=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S-'),dir=folder))
            cmd.extend(['--job',job])
        with (job/'run.log').open('ab') as log:
            subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print('任务已在后台启动。进度和日志保存在',job)
    else:
        os.execv(str(cmd[0]),list(map(str,cmd)))

if __name__=='__main__':
    main(sys.argv[1:])
