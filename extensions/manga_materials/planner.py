"""Translate a new scene request and propose visible objects for local segmentation."""
from functools import lru_cache
from pathlib import Path
import gc
import hashlib
import uuid
import json
import re
import sys

import torch

MODEL_ROOT = Path(__file__).resolve().parents[3] / "models"
INSTRUCTION = """You plan a Japanese anime illustration and the actual surfaces worth detailing.
Read the user's scene in any language. Return ONLY a JSON object with these keys:
"caption": an English image-generation prompt, a concise faithful translation of the user's description.
"medium": a short English phrase containing ONLY an explicitly requested visual medium or style. Use an empty string if unspecified.
"palette": array of {"source":"exact color phrase copied from the request","english":"precise English color name"}. For a requested overall palette only. Translate modifiers (e.g. 奶油白 means cream white, not generic white); one entry per named shade. 黑白木刻 has a black and white palette. Empty array when only individual objects have specified colors.
"art_direction": faithfully translate explicit stylistic tradition, characteristics of the requested paint, lighting direction, treatment of edges and composition constraints into English. Do not invent a lighting or edge style. Keep every named color in a limited palette even if an object also has that color. Empty string when unspecified. Use established medium terminology, for example 水粉 is gouache, 水彩 is watercolor, 木刻 is woodcut.
"surfaces": up to six different entries {"part":"short English noun phrase naming a concrete visible object","material":"short English material name"}.
Use an object's visible noun for part, for example white rabbit, blue apron, ceramic teapot. No positions or coordinates.
Keep the specified animals, people, actions, colors, location, medium, palette and graphic treatment. Never add people, style names or objects that were not requested.
For an identifiable material use fur, knit, fiber, wicker, wood, stone, crust, metal, ceramic, leather, glass, paper or another accurate material. Do not describe skin, eyes or the sky as textured material.
Example output: {"caption":"a child reading on a bench beside a copper kettle","medium":"","palette":[],"art_direction":"","surfaces":[{"part":"copper kettle","material":"copper"},{"part":"wooden bench","material":"wood"},{"part":"open book","material":"paper"}]}"""



# Preserve common palette names when a local translator generalizes the shade.
COLOR_TERMS={"青绿":"blue-green","青綠":"blue-green","草莓红":"strawberry red","草莓紅":"strawberry red","奶油白":"cream white","黑白":"black and white","靛蓝":"indigo","靛藍":"indigo","朱红":"vermilion","朱紅":"vermilion","赭石":"ochre","群青":"ultramarine"}

def english_request(prompt):
    return prompt.replace(" · ", " ").isascii()


def clean_plan(plan,prompt):
    caption=prompt if english_request(prompt) else str(plan.get("caption","")).strip()
    if not caption or (not english_request(prompt) and re.search(r"[\u3040-\u30ff\u3400-\u9fff]",caption)):
        return None
    surfaces=[]
    for entry in plan.get("surfaces",[]):
        if not isinstance(entry,dict):continue
        part=str(entry.get("part","")).strip().lower()[:80]
        material=str(entry.get("material","")).strip().lower()[:60]
        if not part or not material or material in ("flesh","human skin","skin tissue"):continue
        if (part,material) not in surfaces:surfaces.append((part,material))
    medium=plan.get("medium","")
    medium=medium.strip()[:200] if isinstance(medium,str) else ""
    if not english_request(prompt) and medium and medium.casefold() not in caption.casefold():caption=medium+". "+caption
    if not english_request(prompt):
        palette=plan.get("palette",[])
        if isinstance(palette,list):
            shades=[COLOR_TERMS.get(v["source"],v["english"].strip()) for v in palette if isinstance(v,dict) and isinstance(v.get("source"),str) and isinstance(v.get("english"),str) and v["source"] and v["source"] in prompt and v["english"].strip()]
            # Some tokenizers split 黑白 into two non-verbatim terms such as 黑色 and 白色.
            if re.search(r"(?:^|[，。；、,;\s])(?:日系)?黑白(?:木刻|版画|漫畫|漫画|素描|摄影|攝影|速写|速寫|插画|插畫)",prompt) and "black and white" not in shades:
                shades.append("black and white")
            palette="Palette of "+", ".join(shades) if shades else ""
        for item in (palette,plan.get("art_direction","")):
            if isinstance(item,str) and item.strip() and item.strip().casefold() not in caption.casefold():
                caption=caption.rstrip(". ")+". "+item.strip()
    return caption,json.dumps(surfaces[:6],ensure_ascii=False),medium


def dedicated_plan(prompt):
    pipeline=str(MODEL_ROOT.parent/"studio_pipeline")
    if pipeline not in sys.path:sys.path.insert(0,pipeline)
    from evidence import TAG,ask,available,backend,model_identity
    if not available():return None
    schema={"type":"object","properties":{"caption":{"type":"string"},"medium":{"type":"string"},"palette":{"type":"array","items":{"type":"object","properties":{"source":{"type":"string"},"english":{"type":"string"}},"required":["source","english"]}},"art_direction":{"type":"string"},
            "surfaces":{"type":"array","items":{"type":"object","properties":{"part":{"type":"string"},"material":{"type":"string"}},"required":["part","material"]}}},"required":["caption","medium","palette","art_direction","surfaces"]}
    instructions=INSTRUCTION+" Translate every visual detail into idiomatic English, including media such as stop-motion clay. Only describe materials belonging to actual objects. Scene: "
    key=hashlib.sha256(json.dumps([prompt,instructions,schema,model_identity(TAG)],sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest()
    cache=MODEL_ROOT.parent/'cache/scene-plans'/(key+'.json')
    def read():
        try:return clean_plan(json.loads(cache.read_text()),prompt)
        except (OSError,ValueError,KeyError,TypeError,AttributeError):return None
    cached=read()
    if cached:return cached
    with backend() as url:
        cached=read()
        if cached:return cached
        result=json.loads(ask(url,[],instructions+prompt,schema))
        planned=clean_plan(result,prompt)
        if planned:
            temporary=cache.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            try:
                cache.parent.mkdir(parents=True,exist_ok=True)
                temporary.write_text(json.dumps(result,ensure_ascii=False))
                temporary.replace(cache)
            except OSError:pass
            finally:temporary.unlink(missing_ok=True)
        return planned

@lru_cache(maxsize=12)
def _plan(request):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompt = request.strip()
    if not prompt:
        raise ValueError("请填写一段中文或英文画面描述。")
    try:
        structured=dedicated_plan(prompt)
        if structured:return structured
    except (ImportError,OSError,RuntimeError,ValueError,KeyError):
        pass
    location = MODEL_ROOT / "scene_planner"
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(location, local_files_only=True, dtype=torch.bfloat16).eval()
    def ask(instruction, limit):
        chat = [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
        batch = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(**batch, max_new_tokens=limit, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(ids[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    reply = ask(INSTRUCTION, 360)
    plan = {}
    for match in re.finditer(r"\{", reply):
        try:
            candidate, _ = json.JSONDecoder().raw_decode(reply[match.start():])
            if isinstance(candidate, dict) and isinstance(candidate.get("caption"), str):
                plan = candidate
                break
        except ValueError:
            continue
    # Keep English art direction exactly as entered. Translate other languages locally.
    if english_request(prompt):
        caption = prompt
    else:
        caption = str(plan.get("caption") or "").strip()
        if not caption or re.search(r"[\u3400-\u9fff]", caption):
            caption = ask("Translate the user's illustration request faithfully into English. Return only the translation.", 200).strip()
    del model
    gc.collect()
    plan["caption"]=caption
    return clean_plan(plan,prompt) or (prompt,"[]","")



def plan_scene(request):
    caption,surfaces,_=_plan(request)
    return caption,surfaces


def requested_medium(request):
    return _plan(request)[2]


class ScenePlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"request": ("STRING", {"multiline": True, "default":
                    "日系水彩漫画，一位短发女孩系着靛蓝棉围裙，坐在陶艺室里，一只白兔依偎在她身旁。木桌上放着一只青瓷茶壶，窗外下过雨，柔和的晨光。竖幅。"})}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "surface_candidates")
    FUNCTION = "run"
    CATEGORY = "MangaStudio/8K"

    def run(self, request):
        caption, candidates = plan_scene(request)
        return (caption + ", follow the requested medium and light, coherent anatomy, legible contacts, intentional foreground and background", candidates)
