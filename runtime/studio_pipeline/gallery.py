"""A self-contained offline contact sheet and actual-pixel inspector for each job."""
import json
from pathlib import Path
import shlex
from urllib.parse import quote

TEMPLATE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>本地壁纸校样</title>
<style>*{box-sizing:border-box}body{margin:24px auto;padding:0 18px;max-width:1250px;background:#f4f1e9;color:#202321;font:16px/1.55 system-ui,sans-serif}h1{margin-bottom:0}header p{margin:.5em 0}section{display:grid;grid-template-columns:minmax(260px,1fr) minmax(280px,1fr);gap:18px}#frame{position:relative;display:inline-block;line-height:0;max-width:100%}#scene{max-width:100%;max-height:77vh;touch-action:none;cursor:crosshair}#box{position:absolute;border:2px dashed #e43;background:#e443331b;pointer-events:none}canvas{border:1px solid #aaa;max-width:100%;background:#eee}select,input,textarea,button{font:inherit;padding:5px;max-width:100%}textarea{width:100%}small{display:block}label{display:block;margin:.6em 0}#note{white-space:pre-wrap}code{overflow-wrap:anywhere}details{margin:12px 0}@media(max-width:680px){section{grid-template-columns:1fr}}</style>
<header><h1>本地壁纸校样</h1><p id="prompt"></p><p id="status"></p></header>
<label>选择版本 <select id="pick"></select></label><section><div><div id="frame"><img id="scene" alt="构图或成图"><div id="box" hidden></div></div><small>在画上移动查看原像素。拖动可框出任意要重绘的区域。</small></div>
<div><strong>原像素窗口</strong><br><canvas id="zoom" width="320" height="320"></canvas><p id="note"></p><a id="compare" hidden>查看同尺度参考对照</a> <a id="structure" hidden>查看主体与承托局部</a><details id="finish" hidden><summary>目检后标记定稿</summary><small>按实际像素检查整图后，可在本机终端运行。</small><code id="accept"></code></details>
<details id="focusTools" hidden><summary>比较和取舍 8K 局部精绘</summary><a id="focusProof">打开每个编号的前后对照</a><p>上排为整幅神经超分后的局部，下排为最终 8K 同一位置。可在本机运行对应命令撤回一处，或恢复全部。</p><div id="focusCommands"></div></details>
<details id="fix"><summary>修正这个区域</summary><label>描述这里应当画成什么样（英文更准确）<textarea id="desc" rows="2"></textarea></label><small>框选后，下方会生成可在本机终端运行的命令。修订会创建独立任务并重新审图。</small><code id="cmd"></code></details></div></section>
<script>
const data=__DATA__, get=id=>document.getElementById(id), scene=get('scene'),menu=get('pick'),canvas=get('zoom'),ctx=canvas.getContext('2d'),box=get('box'); let start=null,coords=null;
if(data.focus){get('focusTools').hidden=false;get('focusProof').href=data.focus.proof;data.focus.commands.forEach(([name,cmd])=>{const p=document.createElement('p'),title=document.createElement('strong'),code=document.createElement('code');title.textContent=name;code.textContent=cmd;p.append(title,document.createElement('br'),code);get('focusCommands').append(p)})}
if(data.accept){get('finish').hidden=false;get('accept').textContent=data.accept}get('prompt').textContent=data.prompt;get('status').textContent=data.status;
data.items.forEach((i,n)=>menu.add(new Option(i.stage,n)));
function show(){const item=data.items[menu.selectedIndex];if(!item)return;scene.src=item.url;get('note').textContent=item.note;get('compare').hidden=!item.comparison;if(item.comparison)get('compare').href=item.comparison;get('structure').hidden=!item.structure;if(item.structure)get('structure').href=item.structure;coords=null;box.hidden=true;command()}
function point(e){const r=scene.getBoundingClientRect();return [Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))]}
function inspect(p){const x=Math.round(p[0]*scene.naturalWidth),y=Math.round(p[1]*scene.naturalHeight);ctx.clearRect(0,0,320,320);ctx.drawImage(scene,160-x,160-y)}
function setbox(p){const r=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];coords=r.map(x=>Math.round(x*1000));box.hidden=false;Object.assign(box.style,{left:r[0]*100+'%',top:r[1]*100+'%',width:(r[2]-r[0])*100+'%',height:(r[3]-r[1])*100+'%'})}
scene.onload=()=>inspect([.5,.5]);scene.onpointermove=e=>{const p=point(e);inspect(p);if(start)setbox(p)};scene.onpointerdown=e=>{start=point(e);scene.setPointerCapture(e.pointerId);e.preventDefault()};scene.onpointerup=e=>{if(!start)return;setbox(point(e));start=null;get('fix').open=true;command()};
function shell(s){return "'"+s.replaceAll("'","'\"'\"'")+"'"}function command(){const t=get('desc').value.trim();if(!t||!coords||coords[2]-coords[0]<12||coords[3]-coords[1]<12){get('cmd').textContent='';return}get('cmd').textContent=data.entry+' --prompt '+shell(data.prompt)+' --from-image '+shell(data.items[menu.selectedIndex].repairSource)+(data.items[menu.selectedIndex].native?' --keep-native':'')+' --candidates 1 --repair-box '+coords.join(' ')+' --repair-text '+shell(t)}
menu.onchange=show;get('desc').oninput=command;menu.selectedIndex=data.items.length-1;show();
</script></html>"""


def write_focus_proof(master, inputs, boxes, dest):
    from PIL import Image, ImageDraw
    side=380
    canvas=Image.new('RGB',(side*len(boxes),2*(side+27)), '#f3f0e9')
    labels=ImageDraw.Draw(canvas)
    with Image.open(master) as final:
        for idx,((role,x,y),path) in enumerate(zip(boxes,inputs)):
            with Image.open(path) as source:
                actual=final.crop((x,y,x+source.width,y+source.height))
                for row,(image,description) in enumerate(((source,'Before local diffusion'),(actual,'Final 8K'))):
                    top=row*(side+27)
                    labels.text((idx*side+7,top+7),f'{idx+1} / {role} / {description}',fill='#262c29')
                    canvas.paste(image.resize((side,side),Image.Resampling.LANCZOS),(idx*side,top+27))
    canvas.save(dest,quality=95)


def write_gallery(job, rec, studio):
    items = []
    for row in rec['reviews']:
        source = (job/row['file']).resolve()
        if not source.exists():continue
        assessment = row['review']
        items.append(dict(stage=row['stage']+' · '+assessment['status'],url=quote(row['file'],safe='/.'),note=assessment.get('note',''),
                          native=row['stage'].endswith('final-art'),repairSource=str(source),
                          comparison=quote(row['stage']+'-reference-comparison.jpg') if assessment.get('参考比较') else None,
                          structure=quote(row['stage']+'-structure-and-support.jpg') if assessment.get('结构审查') else None))
    if items:
        data=dict(prompt=rec['prompt'],status=rec['status'],items=items,entry=shlex.quote(str(studio/'Studio.command')))
        full=rec.get('精绘区域报告',{})
        if rec.get('8K') and full.get('同位校样') and (job/full['同位校样']).is_file():
            cmd=data['entry']+' --job '+shlex.quote(str(job))+' --skip-focus'
            skipped=set(full.get('实际撤回的区域',[]))
            commands=[('撤回局部 '+str(i),cmd+' '+' '.join(str(v) for v in sorted(skipped|{i}))) for i in range(1,len(full['8K局部'])+1) if i not in skipped]
            if skipped:commands.append(('恢复全部局部',cmd))
            data['focus']={'proof':quote(full['同位校样']),'commands':commands}
        selected=next((a for a in reversed(rec['reviews']) if a['file']==rec.get('8K')),None)
        if selected and rec.get('执行模式')!='离线自动定稿' and not rec.get('逐块终审') and selected['hash'] not in rec.get('退稿',{}):
            data['accept']=data['entry']+' --job '+shlex.quote(str(job))+' --accept-final '+shlex.quote(selected['hash'])
        (job/'proofs.html').write_text(TEMPLATE.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
