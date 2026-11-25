"""Install the pinned upstream engine, local inference dependencies, and model files."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib import request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RT=ROOT/'runtime'
CACHE=RT/'downloads'
TOOLS=RT/'tools'
MAC=sys.platform=='darwin'
COMFY=RT/'ComfyUI'
COMMIT='a277b5bf213e8a44fa34817b7770b54f86ef4cea'
TAGS=('qwen3-vl:8b-instruct','qwen3-vl:32b-instruct','qwen3.8:27b')

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()

def fetch(url,dest,digest):
    if dest.is_file() and sha(dest)==digest:return
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_name(dest.name+'.part')
    for attempt in range(2):
        print('Downloading',dest.name,flush=True)
        cmd=['curl','--fail','--location','--retry','4','--connect-timeout','25','--output',str(partial)]
        if partial.exists() and attempt==0:cmd+=['--continue-at','-']
        outcome=subprocess.run([*cmd,url])
        if outcome.returncode==0 and sha(partial)==digest:
            partial.replace(dest);return
        if outcome.returncode not in (0,33,36):outcome.check_returncode()
        partial.unlink(missing_ok=True)
    raise IOError(f'Download verification failed: {dest.name}')

def unpack(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:*') as tar:
        tar.extractall(destination,filter='data')

def install_engine():
    if not (COMFY/'main.py').is_file():
        archive=CACHE/'comfy.tar.gz'
        fetch(f'https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/{COMMIT}',archive,
              'adc831a3215d6181a8b3f6e801222a710f3baa8056101b9afdca1a05ed4973a3')
        with tempfile.TemporaryDirectory(dir=RT) as temp:
            unpack(archive,Path(temp))
            source=Path(temp)/('ComfyUI-'+COMMIT)
            if COMFY.exists():shutil.copytree(source,COMFY,dirs_exist_ok=True)
            else:source.rename(COMFY)
    pip=[TOOLS/'bin/uv','pip','install','--python',sys.executable]
    subprocess.run([*pip,'--torch-backend','auto','-r',COMFY/'requirements.txt','-r',ROOT/'scripts/requirements.txt'],check=True)
    shutil.copytree(ROOT/'extensions/manga_materials',COMFY/'custom_nodes/manga_materials',dirs_exist_ok=True)
    editors=COMFY/'user/default/workflows/MangaStudio'
    editors.mkdir(parents=True,exist_ok=True)
    for wf in (ROOT/'workflows').glob('*.json'):
        shutil.copy2(wf,editors/wf.name)

def install_superres():
    where='macos' if MAC else 'ubuntu'
    archive=CACHE/f'super-resolution-{where}.zip'
    checksum=('e0ad05580abfeb25f8d8fb55aaf7bedf552c375b5b4d9bd3c8d59764d2cc333a' if MAC else
              'e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96')
    fetch('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'
          f'realesrgan-ncnn-vulkan-20220424-{where}.zip',archive,checksum)
    dest=RT/'models/anime_superres';dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in ['realesrgan-ncnn-vulkan','models/realesr-animevideov3-x2.bin','models/realesr-animevideov3-x2.param']:
            path=dest/Path(name).name
            path.write_bytes(z.read(name))
    (dest/'realesrgan-ncnn-vulkan').chmod(0o755)

def install_ollama():
    ollama=TOOLS/'bin/ollama'
    if not ollama.is_file():
        name='ollama-darwin.tgz' if MAC else 'ollama-linux-amd64.tar.zst'
        digest=('015261f434d31d7f0781f8ba6b59cc8e1f70ecb562a4e6db44dfdd5e02f7e562' if MAC else
                'b82649006cb5b54ed1237fa2d1a26a7f939a4fd03150033759acbff7ff315adf')
        archive=CACHE/name
        fetch('https://github.com/ollama/ollama/releases/download/v0.40.18/'+name,archive,digest)
        dest=TOOLS/'ollama-runtime'
        if MAC:unpack(archive,dest)
        else:
            import zstandard
            dest.mkdir(exist_ok=True)
            with archive.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream:
                with tarfile.open(fileobj=stream,mode='r|') as tar:tar.extractall(dest,filter='data')
        binary=next(p for p in dest.rglob('ollama') if p.is_file() and os.access(p,os.X_OK))
        ollama.symlink_to(os.path.relpath(binary,ollama.parent))
    return ollama

def pull_models(binary):
    weights=RT/'models/visual_editor'
    weights.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    host=f'127.0.0.1:{port}'
    env={**os.environ,'OLLAMA_MODELS':str(weights),'OLLAMA_HOST':host,'OLLAMA_NO_CLOUD':'1'}
    with (RT/'model-install.log').open('ab') as log:
        proc=subprocess.Popen([binary,'serve'],env=env,stdout=log,stderr=log)
        try:
            for _ in range(100):
                if proc.poll() is not None:raise RuntimeError('Ollama startup failed; see runtime/model-install.log')
                try:
                    with request.urlopen('http://'+host+'/api/tags',timeout=2) as resp:json.load(resp)
                    break
                except OSError:time.sleep(.3)
            else:raise RuntimeError('Local model server timed out')
            for tag in TAGS:
                subprocess.run([binary,'pull',tag],env=env,check=True)
        finally:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

def reference():
    from PIL import Image
    folder=COMFY/'input';folder.mkdir(exist_ok=True)
    ref=folder/'Studio-illustration-reference.png'
    if not ref.exists():
        with Image.open(ROOT/'assets/illustration-reference.webp') as im:im.convert('RGB').save(ref)
    brush=folder/'Studio-illustration-brushwork.png'
    if not brush.exists():
        with Image.open(ref) as im:
            sheet=Image.new('RGB',(768,768))
            for i,(x,y) in enumerate(((.1,.15),(.62,.12),(.2,.66),(.63,.58))):
                x,y=int(im.width*x),int(im.height*y)
                box=(x,y,min(x+im.width//3,im.width),min(y+im.height//3,im.height))
                sheet.paste(im.crop(box).resize((384,384)),((i%2)*384,(i//2)*384))
            sheet.save(brush)

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    with (RT/'.install.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        install_engine()
        for row in json.loads((ROOT/'scripts/models.json').read_text()):
            fetch(row['url'],ROOT/row['path'],row['sha256'])
        install_superres()
        pull_models(install_ollama())
        reference()
        subprocess.run([sys.executable,RT/'studio_pipeline/library.py'],check=True)
        (RT/'.installed-v1').write_text(COMMIT+'\n')
    print('Installation complete. Open Studio.command to generate a wallpaper.')

if __name__=='__main__':main()
