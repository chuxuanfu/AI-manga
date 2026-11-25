"""Identify surfaces from the actual pixels of a new illustration, with no scene coordinates."""
from functools import lru_cache
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
import torch

from .planner import MODEL_ROOT
from . import make_tiles, tensor_bytes

# Unknown material names keep their own text conditioning and use no stock texture.
MATERIALS = {
    "crust": (("bread", "crust", "pastry", "biscuit"), "matte baked crust, fine pores and shallow toasted cracks", .40, 14),
    "knit": (("knit", "wool", "yarn"), "interlocking small knitted stitches following the folds, soft yarn fibers", .09, 12),
    "fiber": (("fiber", "cotton", "linen", "fabric", "textile", "silk", "denim", "cloth"), "fine directional woven threads, subtle textile sheen and stitching", .24, 9),
    "fur": (("fur", "animal hair"), "fine tapered guard hairs, soft directional undercoat following the existing coat markings", .24, 12),
    "wood": (("wood", "timber", "bamboo"), "natural wood fibers, fine longitudinal grain, minute pores", .45, 9),
    "stone": (("stone", "granite", "slate", "sandstone"), "mineral flecks and irregular fine surface grains", .50, 8),
    "wicker": (("wicker", "rattan"), "over-under weave and longitudinal split willow fibers", .25, 12),
}
DEFAULTS = [("a knitted sweater", "knit"), ("furry animal", "fur"), ("a wooden surface", "wood"),
            ("fabric clothing", "fiber"), ("stone pavement", "stone"), ("a bread roll", "crust"),
            ("a woven basket", "wicker")]
PROTECT = ["a human face", "human hands and fingers", "bare human skin", "animal eyes", "open sky"]
COLORS = ["#f79944", "#5abcc2", "#a087e8", "#d7ae45", "#68ac72", "#c3689d", "#798ecd",
          "#a8b246", "#dd816a", "#57b09a", "#cb84c8", "#8d9fb8", "#d79d69"]


def material_terms(name):
    for key, (terms, detail, scale, contrast) in MATERIALS.items():
        if any(term in name for term in terms):
            return detail, (key, 8, scale, contrast)
    return f"surface of {name}, fine physically plausible finish, material-specific microstructure and reflected light", None


@lru_cache(maxsize=1)
def segmenter():
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    folder = MODEL_ROOT / "clipseg"
    processor = CLIPSegProcessor.from_pretrained(folder, local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(folder, local_files_only=True).eval()
    return processor, model


def probabilities(image, labels):
    processor, model = segmenter()
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(labels), 5):
            names = labels[start:start+5]
            inputs = processor(text=names, images=[image]*len(names), padding=True, return_tensors="pt")
            logits = model(**inputs).logits.reshape(len(names), 352, 352)
            blocks.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(blocks)


def clean_components(binary, minimum=95):
    labels, _ = ndimage.label(binary)
    sizes = np.bincount(labels.ravel())
    return (labels != 0) & (sizes[labels] >= minimum)


def candidates(raw):
    seen = set()
    result = []
    for pair in json.loads(raw):
        if not isinstance(pair, list) or len(pair)!=2: continue
        obj, mat = (str(x).strip().lower() for x in pair)
        if not obj or not mat or obj in seen: continue
        if obj in ("girl","woman","child","boy","human","person","face","skin","hair","eyes","hands","sky"): continue
        result.append((obj, mat, True))
        seen.add(obj)
    for obj, mat in DEFAULTS:
        if obj not in seen and not any(m==mat for _,m,_ in result):
            result.append((obj, mat, False))
    return result


def visual_masks(image, raw):
    items = candidates(raw)
    labels = [obj for obj,_,_ in items] + PROTECT
    scores = probabilities(image, labels)
    protection = scores[len(items):].max(axis=0)
    scores = scores[:len(items)]
    # Unprompted probes require stronger visual evidence than requested subjects.
    thresholds = np.array([.37 if user else .54 for _,_,user in items], dtype=np.float32)
    adjusted = scores - thresholds[:,None,None]
    winner = np.argmax(adjusted, axis=0)
    accepted = []
    for i,(obj, mat, user) in enumerate(items):
        prob = scores[i]
        if prob.max() < (.50 if user else .68): continue
        binary = (winner==i) & (adjusted[i]>0) & ((protection<.32) | (prob>protection+.16))
        binary = clean_components(ndimage.binary_opening(binary, iterations=1))
        if binary.sum() < 140: continue
        # Contract the uncertain boundary. Skin and hands are excluded again after smoothing.
        binary &= ~((protection>.56) & (protection>prob-.05))
        confidence = np.clip((prob - thresholds[i])/.19, 0, 1) * binary
        mask = Image.fromarray((confidence*255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
        accepted.append((obj, mat, mask, float(prob.max())))
    return accepted


class AutoMaterialTiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"guide": ("IMAGE",), "composition": ("IMAGE",), "surface_candidates": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "STRING", "MANGA_TILE_PLAN", "IMAGE", "STRING")
    RETURN_NAMES = ("tiles", "prompts", "plan", "mask_preview", "report")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "run"
    CATEGORY = "MangaStudio/8K"

    def run(self, guide, composition, surface_candidates):
        if guide.shape[0]!=1 or composition.shape[0]!=1:
            raise ValueError("请每次处理一张画。")
        h,w = guide.shape[1:3]
        im = Image.fromarray(tensor_bytes(composition[0,:,:,:3])); im.thumbnail((900,900))
        accepted = visual_masks(im, surface_candidates)
        regions=[]; report=[]; preview=im.copy()
        for i,(obj,mat,mask,peak) in enumerate(accepted):
            desc,study = material_terms(mat)
            color = COLORS[i % len(COLORS)]
            preview.paste(Image.new("RGB", im.size, color), (0,0), mask.point(lambda a:int(a*.53)))
            native = mask.resize((w,h), Image.Resampling.BILINEAR)
            description = f"Close-up material study of {obj}. {desc}. Retain existing form and markings"
            regions.append((native, description, 0.85, False, study))
            coverage=float(np.array(mask).mean()/255)
            report.append({"object":obj,"material":mat,"study":study[0] if study else None,
                           "coverage":round(coverage,4),"peak_score":round(peak,3),"color":color})
        if not regions:
            regions=[(Image.new("L",(w,h),255), "Existing illustrated surfaces, delicate material-specific small details, preserve original composition",.43,False,None)]
            report=[{"fallback":"No confident material match. Conservative full-frame diffusion with no texture sample."}]
        tiles,texts,plan=make_tiles(guide,regions)
        legend=Image.new('RGB',(preview.width,preview.height+24*len(report)),(242,241,235))
        legend.paste(preview); draw=ImageDraw.Draw(legend)
        for i,entry in enumerate(report):
            y=preview.height+i*24
            if 'object' in entry:
                draw.rectangle((7,y+5,21,y+19),fill=entry['color'])
                draw.text((28,y+4),entry['object']+' / '+entry['material'],fill='#20292c')
            else:draw.text((8,y+4),'No confident match / conservative redraw',fill='#20292c')
        view=torch.from_numpy(np.array(legend)).float().div_(255).unsqueeze(0)
        return tiles,texts,plan,view,json.dumps({"regions":report,"tiles":len(tiles)},ensure_ascii=False)
