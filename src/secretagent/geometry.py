"""Native-pixel masks and crop coordinates; no image blur or coordinate repair."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


def core_mask(region, shape):
    h, w = shape
    if region.kind == "global":
        raise ValueError("Executable regions must be local")
    canvas = Image.new("L", (w, h))
    points = [(min(w-1, int(x*w)), min(h-1, int(y*h))) for x, y in region.points]
    draw = ImageDraw.Draw(canvas)
    if region.kind == "bbox":
        draw.rectangle(points, fill=255)
    else:
        draw.polygon(points, fill=255)
    core = np.asarray(canvas) > 0
    if not core.any() or core.all():
        raise ValueError("Each rasterized core must be nonempty and local")
    return core


def outward_mask(core, width):
    if not np.isfinite(width) or not 0 < width <= 64:
        raise ValueError("Feather width must be in (0,64]")
    d = distance_transform_edt(~core)
    t = np.clip(d/width, 0, 1)
    return (1-t**3*(6*t*t-15*t+10)).astype(np.float32)


def bbox(core):
    y, x = np.where(core)
    return (int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1))


def gradient_regions(plan, params, shape):
    widths = {r.region_id: r.feather_pixels for r in params.regions}
    if len(widths) != len(params.regions) or set(widths) != {r.region_id for r in plan.regions}:
        raise ValueError("A3 must configure every A2 region exactly once")
    cores = [core_mask(r.geometry.canonical(), shape) for r in plan.regions]
    union = np.logical_or.reduce(cores)
    if union.all():
        raise ValueError("Region cores must not jointly cover the full image")
    masks = [outward_mask(c, widths[r.region_id]) for r, c in zip(plan.regions, cores)]
    return np.maximum.reduce(masks), cores, [bbox(c) for c in cores]
