"""Transfer the fine spatial bands of locally generated material studies."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import map_coordinates


@lru_cache(maxsize=32)
def study_band(name, pixel_scale):
    allowed = {"crust", "knit", "fiber", "fur", "wood", "stone", "wicker"}
    if name not in allowed:
        raise ValueError("未知材质样片")
    with Image.open(Path(__file__).parent / "studies" / (name + ".png")) as im:
        gray = im.convert("L").resize((max(16, round(im.width * pixel_scale)), max(16, round(im.height * pixel_scale))), Image.Resampling.LANCZOS)
        tone = np.asarray(gray, dtype=np.float32)
        fine = np.asarray(gray.filter(ImageFilter.GaussianBlur(2.5)), dtype=np.float32)
        shading = np.asarray(gray.filter(ImageFilter.GaussianBlur(8)), dtype=np.float32)
    field = tone - fine + 0.46 * (fine - shading)
    scale = max(2.0, np.percentile(np.abs(field), 87))
    return (2.5 * np.tanh(field / (2.5 * scale))).astype(np.float32)


def surface_light(source, x, y, setting):
    name, angle, pixel_scale, contrast = setting
    field = study_band(name, pixel_scale)
    h, w = source.shape[:2]
    # Coordinates are global to the master so overlap tiles sample the same relief.
    xx = np.arange(x, x + w, dtype=np.float32)[None, :]
    yy = np.arange(y, y + h, dtype=np.float32)[:, None]
    radians = np.deg2rad(angle)
    a, b = np.cos(radians), np.sin(radians)
    tone = np.asarray(Image.fromarray(source).convert("L"), dtype=np.float32) / 255.0
    bend = np.asarray(Image.fromarray(source).convert("L").filter(ImageFilter.GaussianBlur(19)), dtype=np.float32) / 255.0
    u = (a * xx + b * yy + 4 * np.sin(yy / 183) + 4 * bend)
    v = (-b * xx + a * yy + 4 * np.sin(xx / 237) - 6 * bend)
    sampled = map_coordinates(field, [v, u], order=1, mode="reflect", prefilter=False)
    gy, gx = np.gradient(tone)
    ink_guard = 1.0 / (1.0 + 70 * (np.abs(gx) + np.abs(gy)))
    light_response = np.clip(4.8 * tone * (1 - tone), 0.12, 1)
    return (sampled * contrast * light_response * ink_guard)[:, :, None]
