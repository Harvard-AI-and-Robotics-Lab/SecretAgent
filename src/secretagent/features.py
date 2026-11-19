"""Six-encoder features for the objective and forward-only search feedback."""
import numpy as np
import torch
import torch.nn.functional as F
from .schemas import IDS


def unit(x):
    if not torch.isfinite(x).all():
        raise FloatingPointError("Nonfinite feature")
    return F.normalize(x.float(), dim=-1)


def weighted_concat(features, weights):
    if set(features) != set(IDS) or set(weights) != set(IDS):
        raise ValueError("Exactly six feature blocks and weights are required")
    w = torch.tensor([weights[k] for k in IDS], device=features[IDS[0]].device, dtype=torch.float64)
    if not torch.isfinite(w).all() or not (w > 0).all():
        raise ValueError("Weights must be positive and finite")
    w = w/w.max()
    w = w/w.sum()
    return torch.cat([unit(features[k])*w[i].sqrt().to(features[k].dtype) for i, k in enumerate(IDS)], dim=-1)


def fuse(features, texts):
    out = dict(features)
    for k in IDS[:3]:
        out[k] = unit(.7*unit(features[k]) + .3*texts[k].detach())
    return out


def tensor(rgb, device):
    return torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)/255


class Surrogates:
    def __init__(self, encoders, device):
        if set(encoders) != set(IDS):
            raise ValueError("The method requires all six encoders")
        self.encoders, self.device = encoders, device
        for model in encoders.values():
            model.eval().requires_grad_(False)

    def image(self, x):
        features = {}
        for name in IDS:
            tokens = self.encoders[name](x)
            if tokens.ndim != 3 or tokens.shape[0] != x.shape[0] or tokens.shape[1] < 1:
                raise ValueError(f'{name} must return [batch,tokens,hidden]')
            features[name] = unit(tokens.mean(1))
        return features

    @torch.no_grad()
    def text(self, sentences):
        if not sentences or not any(s.strip() for s in sentences):
            raise ValueError("A1 supplied no usable answer/evidence text")
        return {k: unit(self.encoders[k].text(sentences).mean(0, keepdim=True)) for k in IDS[:3]}

    @torch.no_grad()
    def measure(self, original, candidate, regions, previous=None, fusion_context=None):
        x, y = tensor(original, self.device), tensor(candidate, self.device)
        h, w = original.shape[:2]
        boxes = [(0, 0, w, h)]+[tuple(r['crop_box_pixels']) for r in regions]
        measured, fused = {}, {}
        if fusion_context:
            clean_text = self.text(fusion_context['clean'])
            candidate_text = self.text(fusion_context['candidate'])
        for box in dict.fromkeys(boxes):
            l, t, r, b = box
            a, c = self.image(x[:, :, t:b, l:r]), self.image(y[:, :, t:b, l:r])
            measured[box] = {k: float(F.cosine_similarity(a[k].double(), c[k].double()).clamp(-1, 1).mean()) for k in IDS}
            if fusion_context:
                fa, fc = fuse(a, clean_text), fuse(c, candidate_text)
                fused[box] = {k: float(F.cosine_similarity(fa[k].double(), fc[k].double()).clamp(-1, 1).mean()) for k in IDS}
        local = []
        for region in regions:
            current = dict(region, similarity=measured[tuple(region['crop_box_pixels'])])
            old = next((v for v in (previous or {}).get('regions', [])
                        if v['region_id'] == region['region_id'] and v['geometry'] == region['geometry']
                        and tuple(v['crop_box_pixels']) == tuple(region['crop_box_pixels'])), None)
            current['previous_similarity'] = old['similarity'] if old else None
            current['same_geometry_as_previous'] = old is not None
            current['similarity_change'] = {k: current['similarity'][k]-old['similarity'][k] for k in IDS} if old else None
            if fusion_context:
                current['fused_similarity'] = fused[tuple(region['crop_box_pixels'])]
            local.append(current)
        full = measured[boxes[0]]
        result = dict(full_image=full, regions=local, no_similarity_gate=True,
                    previous_full_image=previous['full_image'] if previous else None,
                    full_image_change={k: full[k]-previous['full_image'][k] for k in IDS} if previous else None)
        if fusion_context:
            result['fused_full_image'] = fused[boxes[0]]
            result['fusion_context'] = fusion_context
        return result

    def close(self):
        self.encoders.clear()
        if str(self.device).startswith('cuda'):
            torch.cuda.empty_cache()
