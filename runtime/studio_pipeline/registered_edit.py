"""Constrain a local diffusion repair to geometry registered to the current drawing."""
import cv2
import numpy as np
from PIL import Image, ImageOps
from graphs import Graph, BASE

CONTROL = 'xinsir-canny-v2-sdxl.safetensors'


def registered_design(original, previous, editable):
    """Map an earlier design to this crop using intact features around the requested repair."""
    if original.size != previous.size or original.size != editable.size:
        return None
    width, height = original.size
    if min(width, height) < 80:
        return None
    current, earlier = [np.asarray(image.convert('L')) for image in (original, previous)]
    context = np.where(np.asarray(editable.convert('L')) < 12, 255, 0).astype('uint8')
    context = cv2.erode(context, np.ones((5, 5), np.uint8))
    detector = cv2.SIFT_create(nfeatures=550, contrastThreshold=.015)
    old_points, old_features = detector.detectAndCompute(earlier, context)
    new_points, new_features = detector.detectAndCompute(current, context)
    if old_features is None or new_features is None or min(len(old_points), len(new_points)) < 8:
        return None
    pairs = cv2.BFMatcher().knnMatch(old_features, new_features, k=2)
    matches = [a for a, b in pairs if a.distance < .72 * b.distance]
    if len(matches) < 8:
        return None
    old = np.float32([old_points[m.queryIdx].pt for m in matches])
    new = np.float32([new_points[m.trainIdx].pt for m in matches])
    warp, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if warp is None or inliers is None:
        return None
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 7 or keep.mean() < .65:
        return None
    # Repeated paper textures or a few matches on just one plank cannot register a face or joint.
    hull = cv2.convexHull(new[keep])
    bounds = editable.getbbox()
    if not bounds:
        return None
    center = ((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2)
    if cv2.pointPolygonTest(hull, center, True) < -min(width, height)*.05:
        return None
    scale = np.linalg.norm(warp[0, :2])
    angle = abs(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    if not .8 <= scale <= 1.25 or angle > 15:
        return None
    pixels = cv2.warpAffine(np.asarray(previous.convert('RGB')), warp, (width, height),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    result = Image.composite(Image.fromarray(pixels), original.convert('RGB'), editable.convert('L'))
    return result, dict(transform=warp.tolist(), context_matches=int(keep.sum()))


def drawing_inputs(layout, editable, size):
    canvas = layout.convert('RGB').resize(size, Image.Resampling.LANCZOS)
    mask = editable.convert('L').resize(size, Image.Resampling.BILINEAR)
    ink = cv2.Canny(cv2.bilateralFilter(np.asarray(canvas), 7, 26, 4), 58, 135)
    canvas.putalpha(ImageOps.invert(mask))
    return canvas, Image.fromarray(ink).convert('RGB')


def controlled_drawing(layout, edges, description, seed, prefix, checkpoint=BASE):
    graph = Graph('registered-geometry-repair')
    model = graph.checkpoint('Drawing model', (0, 0), checkpoint)
    canvas = graph.add('LoadImage', 'Registered drawing and repair mask', (0, 230),
                       ['IMAGE', 'MASK'], values={'image': layout}, widgets=[layout, 'image'])
    guide = graph.add('LoadImage', 'Registered edges', (0, 460),
                      ['IMAGE', 'MASK'], values={'image': edges}, widgets=[edges, 'image'])
    control = graph.add('ControlNetLoader', 'Local SDXL edge control', (0, 690), ['CONTROL_NET'],
                        values={'control_net_name': CONTROL})
    positive = graph.text('Material and construction', (360, 0), model, description)
    negative = graph.text('Prevent artifacts', (360, 210), model,
                          'stray marks, floating parts, added lettering, doubled outline, blur, halo')
    conditions = graph.add('ControlNetApplyAdvanced', 'Hold the registered construction', (740, 160),
                           ['CONDITIONING', 'CONDITIONING'],
                           [('positive', (positive, 0)), ('negative', (negative, 0)),
                            ('control_net', (control, 0)), ('image', (guide, 0)), ('vae', (model, 2))],
                           dict(strength=.88, start_percent=0., end_percent=.94))
    latent = graph.encode('Encode the drawing', (360, 440), canvas, model)
    masked = graph.add('SetLatentNoiseMask', 'Repaint inside the repair', (740, 455), ['LATENT'],
                       [('samples', (latent, 0)), ('mask', (canvas, 1))])
    sampled = graph.add('KSampler', 'Draw the local material', (1080, 160), ['LATENT'],
                        [('model', (model, 0)), ('positive', (conditions, 0)),
                         ('negative', (conditions, 1)), ('latent_image', (masked, 0))],
                        dict(seed=seed, steps=14, cfg=5., sampler_name='euler', scheduler='normal', denoise=.33),
                        [seed, 'fixed', 14, 5., 'euler', 'normal', .33])
    painted = graph.decode('Decode the repair', (1420, 160), sampled, model)
    graph.save('Candidate for local inspection', (1750, 160), painted, prefix)
    return graph
