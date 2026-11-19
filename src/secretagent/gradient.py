"""One original-relative masked sign step per outer round, without momentum."""
import numpy as np
import torch
import torch.nn.functional as F
from .features import tensor, weighted_concat, fuse
from .geometry import gradient_regions


def project(x, delta, cap):
    return ((x+torch.maximum(torch.minimum(delta, cap), -cap)).clamp(0, 1)-x).detach()


def fixed_update(x, delta, gradient, mask, epsilon_pixels, alpha_pixels):
    cap = mask*epsilon_pixels/255
    before = project(x, delta, cap)
    return project(x, before+alpha_pixels/255*mask*gradient.sign(), cap)


def statistics(original, candidate, core):
    delta = candidate.astype(np.int16)-original.astype(np.int16)
    inside = np.abs(delta[core]).astype(float)
    return dict(linf_pixels=int(np.abs(delta).max()), no_change=not bool(delta.any()),
                mean_abs_pixels=float(np.abs(delta).mean()),
                rms_pixels=float(np.sqrt(np.mean(delta.astype(float)**2))),
                interior_mean_abs_pixels=float(inside.mean()), core_fraction=float(core.mean()))


class GradientRenderer:
    def __init__(self, surrogates):
        self.bank = surrogates

    def render(self, original, previous, plan, params, epsilon_pixels, alpha_pixels, clean_text, candidate_text):
        mask, cores, bounds = gradient_regions(plan, params, original.shape[:2])
        x = tensor(original, self.bank.device)
        prev = torch.zeros_like(x) if previous is None else tensor(previous, self.bank.device)-x
        m = torch.from_numpy(mask).to(x.device)[None, None]
        cap = m*epsilon_pixels/255
        delta = project(x, prev, cap)
        gen = torch.Generator(device=x.device).manual_seed(params.seed)
        probe = (torch.rand(x.shape, generator=gen, device=x.device)*2-1)*(.25/255)*m
        probe = torch.where(delta.abs() < 1e-8, probe, torch.zeros_like(probe))
        # The probe evaluates the derivative only; it is not persistent noise.
        z = (x+delta+probe).detach().requires_grad_(True)
        h, w = original.shape[:2]
        boxes = [(0, 0, w, h)]+bounds
        view_weights = [.6]+[.4/len(bounds)]*len(bounds)
        weights = params.encoder_weights.model_dump()
        clean_vectors, candidate_vectors = self.bank.text(clean_text), self.bank.text(candidate_text)
        gradient = torch.zeros_like(x)
        # Accumulate raw view gradients, then take SIGN once. Encoder gradients
        # are combined before backpropagation within each view.
        for box, view_weight in zip(boxes, view_weights):
            l, t, r, b = box
            with torch.no_grad():
                reference = weighted_concat(fuse(self.bank.image(x[:, :, t:b, l:r]), clean_vectors), weights)
            feature = weighted_concat(fuse(self.bank.image(z[:, :, t:b, l:r]), candidate_vectors), weights)
            loss = 1-F.cosine_similarity(feature, reference).mean()
            g = torch.autograd.grad(loss, z)[0]
            if not torch.isfinite(g).all():
                raise FloatingPointError("Nonfinite pixel gradient")
            gradient += view_weight*g
        updated = fixed_update(x, delta, gradient, m, epsilon_pixels, alpha_pixels)
        pixels = torch.round((x+updated)*255).clamp(0, 255).byte()[0].permute(1, 2, 0).cpu().numpy()
        change = pixels.astype(np.int16)-original.astype(np.int16)
        if np.abs(change).max() > epsilon_pixels or np.any(change[mask == 0]):
            raise AssertionError("Quantized candidate violates support or budget")
        regions = [dict(region_id=r.region_id, geometry=r.geometry.model_dump(), crop_box_pixels=list(b),
                        feather_pixels=f.feather_pixels,
                        actual_composite=statistics(original, pixels, c))
                   for r, b, c in zip(plan.regions, bounds, cores)
                   for f in params.regions if f.region_id == r.region_id]
        return pixels, regions, statistics(original, pixels, np.logical_or.reduce(cores))
