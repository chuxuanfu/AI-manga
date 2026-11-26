"""Draw a masked candidate on the local Neural Engine while the desktop GPU is occupied."""
import fcntl
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image

RUNTIME = Path(__file__).resolve().parents[1]


def paint_local_field(image, prompt, seed, directory, stage, strength=.28):
    binary = RUNTIME/'tools/bin/coreml-painter'
    weights = RUNTIME/'models/coreml-sd15'
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64' or not binary.is_file()
            or not (weights/'VAEEncoder.mlmodelc/coremldata.bin').is_file()):
        return None
    with (RUNTIME/'neural-engine.lock').open('a') as lane, tempfile.TemporaryDirectory(dir=directory) as temporary:
        fcntl.flock(lane, fcntl.LOCK_EX)
        root = Path(temporary)
        image = image.convert('RGB')
        ratio = 512 / max(image.size)
        w, h = (max(1, round(n*ratio)) for n in image.size)
        pixels = np.asarray(image.resize((w,h), Image.Resampling.LANCZOS))
        x, y = (512-w)//2, (512-h)//2
        canvas = np.pad(pixels, ((y,512-h-y), (x,512-w-x), (0,0)), mode='edge')
        Image.fromarray(canvas).save(root/'input.png')
        # The upstream executable derives its filename from the first 75 characters of the prompt.
        instruction = ('Render this same small region in the source drawing with its existing lighting and medium. ' + prompt)
        args = [str(binary), '--compute-units', 'cpuAndNeuralEngine', '--resource-path', str(weights),
                '--image', str(root/'input.png'), '--output-path', str(root), '--seed', str(seed % 2**32),
                '--step-count', '26', '--strength', str(strength), '--guidance-scale', '5.9',
                '--negative-prompt', 'duplicate contours, ringing, floating pieces, plastic, added lettering',
                '--reduce-memory', instruction]
        with (directory/(stage+'-neural-engine.log')).open('wb') as log:
            proc = subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            warning_since = None
            try:
                while proc.poll() is None:
                    level = subprocess.run(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],
                                           capture_output=True,text=True)
                    pressure = int(level.stdout.strip()) if level.returncode==0 and level.stdout.strip().isdigit() else 1
                    if pressure>=2:
                        warning_since = warning_since or time.monotonic()
                        if pressure>=4 or time.monotonic()-warning_since>10:
                            return None
                    else:
                        warning_since = None
                    time.sleep(.7)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        files = list(root.glob('*.final.png'))
        if proc.returncode or len(files)!=1:
            return None
        with Image.open(files[0]) as result:
            if result.size != (512,512):
                return None
            return result.convert('RGB').crop((x,y,x+w,y+h)).resize(image.size,Image.Resampling.LANCZOS)
