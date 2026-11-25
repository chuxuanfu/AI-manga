"""Generate Finder-friendly offline albums. Public jobs expose only reviewed masters."""
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from PIL import Image

STUDIO=Path(__file__).resolve().parents[2]
IMAGES=STUDIO/'images'
CACHE=IMAGES/'.thumbnails'
HOME='gallery.html'
ALBUM='index.html'
LABELS={'fifty-scenes':'五十景画集','new-jobs':'新提示词与定稿','Wallpaper':'屏幕壁纸','Spring':'春日插画与修订版',
        'Detail':'分块精绘','Draft':'构图草稿','SpringDraft':'春日底图','reference-studies':'参考画法实测','workflow-tests':'workflow-tests'}
STYLE="""<meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}html{background:#f5f3ed;color:#232a26;font:16px/1.55 system-ui,sans-serif}body{max-width:1220px;margin:auto;padding:40px 5vw 80px}
h1{font-size:clamp(2rem,4.6vw,3.7rem);font-weight:650;letter-spacing:-.06em;margin:10px 0 12px}h2{margin:36px 0 14px;font-size:1.2rem}
a{color:#325a4b;text-underline-offset:4px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(210px,100%),1fr));gap:21px}
.card{display:flex;flex-direction:column;text-decoration:none;color:inherit}.card img,.empty{height:clamp(155px,16vw,212px);width:100%;object-fit:cover;background:#e7e3d6;border-radius:4px}
.card:hover strong{text-decoration:underline;text-underline-offset:4px}.card strong{margin-top:8px;overflow-wrap:anywhere}.card small{color:#565d56}.empty{display:grid;place-items:center;font-size:1.15rem}
nav,p{margin:0 0 15px}a:focus-visible{outline:3px solid #27624c;outline-offset:3px}
</style>"""

def relative(target,folder):
    return quote(os.path.relpath(target,folder))

def write(path,body):
    tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(body);tmp.replace(path)

def thumbnail(path):
    if not path or not path.is_file():return None
    stat=path.stat()
    key=hashlib.sha256((str(path.relative_to(IMAGES))+str(stat.st_mtime_ns)+str(stat.st_size)).encode()).hexdigest()[:24]
    target=CACHE/(key+'.jpg')
    if not target.exists():
        try:
            with Image.open(path) as picture:
                picture.thumbnail((560,400),Image.Resampling.LANCZOS)
                tmp=target.with_suffix('.tmp')
                picture.convert('RGB').save(tmp,'JPEG',quality=86)
                tmp.replace(target)
        except (OSError,ValueError):
            return None
    return target

def card(folder,title,href,cover=None,subtitle=''):
    t=html.escape(title);dest=relative(href,folder)
    img=thumbnail(cover)
    visual=('<img loading="lazy" alt="" src="'+relative(img,folder)+'">') if img else '<span class="empty">查看任务</span>'
    return f'<a class="card" href="{dest}">{visual}<strong>{t}</strong><small>{html.escape(subtitle)}</small></a>'

def page(path,title,contents):
    nav='' if path.parent==IMAGES else '<nav><a href="'+relative(IMAGES/HOME,path.parent)+'">返回图片库</a></nav>'
    write(path,'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(title)+'</title>'+STYLE+nav+'<h1>'+html.escape(title)+'</h1>'+contents+'</html>')

def grid(cards):
    return '<div class="grid">'+''.join(cards)+'</div>'

def record(path):
    try:return json.loads(path.read_text())
    except (ValueError,OSError):return {}

def jobs(folder):
    tiles=[];cover=None
    for job in sorted((p for p in folder.iterdir() if p.is_dir()),key=lambda p:p.name,reverse=True):
        data=record(job/'production.json') or record(job/'review.json')
        dashboard=job/'proofs.html'
        if not dashboard.exists():continue
        art=job/'4K.png'
        ready=art.is_file() and (job/'final').is_dir()
        if ready and cover is None:cover=art
        tiles.append(card(folder,data.get('prompt',job.name),dashboard,art if ready else None,data.get('status',job.name)))
    page(folder/ALBUM,'新提示词与定稿',grid(tiles) if tiles else '<p>新任务会出现在这里。</p>')
    return folder/ALBUM,cover

def history(folder):
    groups={}
    for image in sorted(folder.rglob('*.png')):
        if image.is_file():groups.setdefault(image.parent,[]).append(image)
    parts=[]
    for sub,images in groups.items():
        if sub!=folder:parts.append('<h2>'+html.escape(str(sub.relative_to(folder)))+'</h2>')
        if (sub/'proofs.html').is_file():parts.append('<p><a href="'+relative(sub/'proofs.html',folder)+'">打开该任务的审图页</a></p>')
        parts.append(grid([card(folder,p.stem,p,p) for p in images]))
    page(folder/ALBUM,LABELS.get(folder.name,folder.name),''.join(parts))
    cover=next((p for files in groups.values() for p in files if '对照' not in p.name and '检查' not in p.name),None)
    return folder/ALBUM,cover

def refresh():
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        albums=[]
        extras=sorted(p.name for p in IMAGES.iterdir() if p.is_dir() and not p.name.startswith('.') and p.name not in LABELS)
        for name in [*LABELS,*extras]:
            folder=IMAGES/name
            if not folder.is_dir():continue
            if name=='new-jobs':entry,cover=jobs(folder)
            elif name=='fifty-scenes':
                entry=folder/'index.html';cover=folder/'proofs/01-cherry-rain-post-office.jpg'
            else:entry,cover=history(folder)
            if entry.is_file():albums.append(card(IMAGES,LABELS.get(name,name),entry,cover,name))
        page(IMAGES/HOME,'本地图片库',grid(albums))
        write(STUDIO/HOME,'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=images/gallery.html"><title>本地图片库</title><a href="images/gallery.html">打开本地图片库</a></html>')

if __name__=='__main__':refresh()
