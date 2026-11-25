"""Native-resolution material repainting nodes for the local MangaStudio."""
import math
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from .surface import surface_light

HERE = Path(__file__).resolve().parent
TILE = 1152
OVERLAP = 192
CUSTOM = "使用手绘遮罩"
WHOLE = "整幅通用重绘"
PRESETS = [CUSTOM, WHOLE]
if (HERE / "regions.json").is_file():
    PRESETS.extend(json.loads((HERE / "regions.json").read_text()))


def polygon_mask(size, regions, exclusions):
    mask = Image.new("L", size)
    draw = ImageDraw.Draw(mask)
    def xy(poly):
        return [(round(x * size[0] / 480), round(y * size[1] / 720)) for x, y in poly]
    for poly in regions:
        draw.polygon(xy(poly), fill=255)
    for poly in exclusions:
        draw.polygon(xy(poly), fill=0)
    return mask


def starts(lo, hi, full, tile):
    # Give the sampler enough context outside the painted object.
    lo, hi = max(0, lo - 64), min(full, hi + 64)
    if hi - lo <= tile:
        return [max(0, min(full - tile, (lo + hi - tile) // 2))]
    count = math.ceil((hi - lo - tile) / (tile - OVERLAP)) + 1
    return [round(v) for v in np.linspace(lo, hi - tile, count)]


def tensor_bytes(image):
    return image.detach().to("cpu").mul(255).round().clamp(0, 255).to(torch.uint8).numpy()


class MaterialTiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "source_mask": ("MASK",),
            "profile": (PRESETS,),
            "study": (["none", "crust", "knit", "fiber", "fur", "wood", "stone", "wicker"],),
            "material": ("STRING", {"multiline": True, "default":
                "fine woven linen, visible individual threads, illustrated in fine ink and watercolor"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "MANGA_TILE_PLAN")
    RETURN_NAMES = ("tiles", "prompts", "plan")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "prepare"
    CATEGORY = "MangaStudio/8K"

    def prepare(self, image, source_mask, profile, study, material):
        if image.shape[0] != 1:
            raise ValueError("请每次处理一张底图。")
        h, w = image.shape[1:3]
        if min(h, w) < TILE:
            raise ValueError("请先将输入调整到目标分辨率，短边至少 1152 像素。")
        if profile == CUSTOM:
            mask = Image.fromarray(tensor_bytes(source_mask[0])).resize((w, h), Image.Resampling.BILINEAR)
            regions = [(mask, material, 1.0, True, [study, 0, 0.65, 13] if study != "none" else None)]
        elif profile == WHOLE:
            regions = [(Image.new("L", (w, h), 255), material, 0.9, False, None)]
        else:
            data = json.loads((HERE / "regions.json").read_text())[profile]
            thumbnail = torch.nn.functional.interpolate(image[:, :, :, :3].movedim(-1, 1), size=(36, 24), mode="area")[0].movedim(0, -1)
            fingerprint = np.asarray(data["preview"], dtype=np.float32)
            if np.mean(np.abs(tensor_bytes(thumbnail).astype(np.float32) - fingerprint)) > 16:
                raise ValueError("该预设与底图不匹配。请选对应的春日壁纸，或用手绘遮罩标出新图里的材质。")
            regions = ((polygon_mask((w, h), r["polygons"], data["protect"] + r.get("exclude", [])),
                        r["prompt"], r.get("strength", 1.0), r.get("replace_surface", False), r.get("study")) for r in data["regions"])
        return make_tiles(image, regions)


def make_tiles(image, regions):
    h, w = image.shape[1:3]
    tiles, prompts, plan = [], [], []
    for mask, description, strength, replace_surface, study_setting in regions:
        bbox = mask.getbbox()
        if not bbox:
            continue
        for y in starts(bbox[1], bbox[3], h, TILE):
            for x in starts(bbox[0], bbox[2], w, TILE):
                local = mask.crop((x, y, x + TILE, y + TILE))
                # Drop empty intersections, while preserving small painted islands.
                if not local.getbbox():
                    continue
                tiles.append(image[:, y:y + TILE, x:x + TILE, :3])
                prompts.append(description + ", close-up study of the existing material, tactile natural microtexture, grazing light across the surface, fine editorial illustration, maintain the original palette and perspective")
                plan.append((x, y, local, strength, replace_surface, study_setting))
    if not tiles:
        raise ValueError("遮罩为空。请在输入图的遮罩编辑器中涂出材质区域，或选择整幅通用重绘。")
    return tiles, prompts, plan


def locked_residual(base, painted, replace_surface=False):
    # Preserve broad shading and silhouettes. Transfer newly sampled fine and mid bands.
    def blur(arr, radius):
        return np.asarray(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    delta = painted.astype(np.float32) - base
    if replace_surface:
        # An explicit object mask protects its external silhouette. Replace internal
        # edges instead of superimposing the new grooves over the old drawing.
        lighting = blur(painted, 35) - blur(base, 35)
        return np.clip(delta - 0.83 * lighting, -180, 180)
    mid = blur(painted, 10) - blur(base, 10)
    low = blur(painted, 40) - blur(base, 40)
    detail = (delta - mid) + 0.38 * (mid - low)
    luma = np.asarray(Image.fromarray(base).convert("L"), dtype=np.float32)
    gy, gx = np.gradient(luma)
    protection = 0.18 + 0.82 / (1.0 + ((gx * gx + gy * gy) / 90) ** 1.25)
    return 48 * np.tanh(detail / 48) * protection[:, :, None]


class MaterialComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base": ("IMAGE",), "painted": ("IMAGE",), "plan": ("MANGA_TILE_PLAN",)}}

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    FUNCTION = "compose"
    CATEGORY = "MangaStudio/8K"

    def compose(self, base, painted, plan):
        image = tensor_bytes(base[0][0, :, :, :3])
        entries = plan[0]
        if len(entries) != len(painted):
            raise ValueError("分块数量与布局不一致，无法合成。")
        h, w = image.shape[:2]
        acc = np.zeros((h, w, 3), np.float32)
        weights = np.zeros((h, w), np.float32)
        opacity = np.zeros((h, w), np.float32)
        axis = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE))
        ramp = np.minimum(1, axis / 80).astype(np.float32)
        for tile, (x, y, mask, strength, replace_surface, study_setting) in zip(painted, entries):
            win = np.s_[y:y + TILE, x:x + TILE]
            source = image[win]
            delta = locked_residual(source, tensor_bytes(tile[0, :, :, :3]), replace_surface)
            if study_setting:
                if study_setting[0] == "crust":
                    # Keep the existing scoring pattern and contact edges on the bread.
                    delta = 0.36 * locked_residual(source, tensor_bytes(tile[0, :, :, :3]))
                delta += surface_light(source, x, y, study_setting)
            raw = np.asarray(mask, dtype=np.float32) / 255
            feather = np.minimum(raw, np.asarray(mask.filter(ImageFilter.GaussianBlur(8)), dtype=np.float32) / 255)
            ry, rx = ramp.copy(), ramp.copy()
            if y == 0: ry[:80] = 1
            if x == 0: rx[:80] = 1
            if y + TILE == h: ry[-80:] = 1
            if x + TILE == w: rx[-80:] = 1
            coverage = ry[:, None] * rx[None, :] * feather
            acc[win] += coverage[:, :, None] * delta
            weights[win] += coverage
            np.maximum(opacity[win], coverage * strength, out=opacity[win])
        # Work in strips to keep the full master from needing extra float-sized buffers.
        for y in range(0, h, 128):
            rows = slice(y, y + 128)
            scale = opacity[rows] / np.maximum(weights[rows], 1e-8)
            image[rows] = np.clip(np.rint(image[rows] + acc[rows] * scale[:, :, None]), 0, 255).astype(np.uint8)
        return (torch.from_numpy(image).float().div_(255).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"MangaStudioMaterialTiles": MaterialTiles, "MangaStudioMaterialComposite": MaterialComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"MangaStudioMaterialTiles": "8K 材质区域与提示词", "MangaStudioMaterialComposite": "8K 保形融合"}

from .planner import ScenePlanner
from .automatic import AutoMaterialTiles

NODE_CLASS_MAPPINGS.update({"MangaStudioScenePlanner": ScenePlanner, "MangaStudioAutoMaterialTiles": AutoMaterialTiles})
NODE_DISPLAY_NAME_MAPPINGS.update({"MangaStudioScenePlanner": "本地场景与材质规划", "MangaStudioAutoMaterialTiles": "自动定位材质 · CLIPSeg"})
