"""Compile complete Tool plans, preserve keep components, and render from x."""
import copy
import hashlib
import json
from typing import Literal, Union
import numpy as np
from pydantic import Field, create_model
from .schemas import Region, NumericConfig, StrictModel, ToolParameters
from .geometry import core_mask, outward_mask, bbox
from .noise.fields import DOMAINS, generate, parameters


def compile_plan(plan, latest, best, shape):
    parents = {r['candidate_id']: r for r in (latest, best) if r and r.get('config')}
    if plan.parent_candidate_id is not None and plan.parent_candidate_id not in parents:
        raise ValueError("Unknown parent; use latest, best, or null")
    parent = parents.get(plan.parent_candidate_id, {})
    old = {c['component_id']: c for c in parent.get('config', [])}
    if len({c.component_id for c in plan.components}) != len(plan.components):
        raise ValueError("Duplicate component IDs")
    compiled = []
    for c in plan.components:
        previous = old.get(c.component_id)
        if c.action == 'keep':
            if previous is None:
                raise ValueError("keep requires a parent component")
            compiled.append(dict(copy.deepcopy(previous), action='keep', adjustment=c.adjustment))
            continue
        if (previous is None) != (c.action == 'add'):
            raise ValueError("Use add for new IDs, modify/replace for existing IDs")
        if c.noise_type is None or c.target_region is None:
            raise ValueError("Editable components require family and region")
        if hasattr(c, 'target_attributes'):
            if not c.target_attributes or c.clues is None:
                raise ValueError("Joint facial components require target_attributes and clues")
            clues = c.clues.model_dump()
            target_clue = '; '.join(f'{name}: {value}' for name, value in clues.items() if value)
            attributes = dict(target_attributes=list(c.target_attributes), clues=clues)
        else:
            if c.target_clue is None:
                raise ValueError("Editable components require a target clue")
            target_clue, attributes = c.target_clue, {}
        compiled.append(dict(component_id=c.component_id, noise_type=c.noise_type,
            target_region=c.target_region.canonical().model_dump(), target_clue=target_clue,
            **attributes, action=c.action, adjustment=c.adjustment, config=None,
            initial_config=copy.deepcopy(previous['config']) if previous else None))
    union = np.logical_or.reduce([core_mask(Region.model_validate(c['target_region']), shape) for c in compiled])
    if union.all():
        raise ValueError("Combined cores cover the entire image")
    return compiled


def execution_schema(components):
    variants = []
    for i, c in enumerate(components):
        if c['action'] == 'keep':
            continue
        fields = {k: (int if integer else float, Field(ge=lo, le=hi))
                  for k, (lo, hi, _, integer) in DOMAINS[c['noise_type']].items()}
        family = create_model(f'FamilyParameters{i}', __base__=StrictModel, **fields)
        numeric = {k: (v.annotation, copy.deepcopy(v)) for k, v in NumericConfig.model_fields.items() if k != 'parameters'}
        numeric['parameters'] = (family, ...)
        config = create_model(f'ComponentConfig{i}', __base__=StrictModel, **numeric)
        variants.append(create_model(f'EditableComponent{i}', __base__=StrictModel,
            component_id=(Literal[c['component_id']], ...), config=(config, ...)))
    if not variants:
        return None
    item = variants[0] if len(variants) == 1 else Union[tuple(variants)]
    return create_model('TypedToolParameters', __base__=StrictModel,
        components=(list[item], Field(min_length=len(variants), max_length=len(variants))), explanation=(str, ...))


def merge_parameters(components, response, shape):
    raw = response.model_dump() if response is not None else {'components': [], 'explanation': 'All components kept'}
    for c in raw['components']:
        values = c['config']['parameters']
        if isinstance(values, dict):
            c['config']['parameters'] = [dict(name=k, value=v) for k, v in values.items()]
    execution = ToolParameters.model_validate(raw)
    returned = {c.component_id: c.config for c in execution.components}
    editable = {c['component_id'] for c in components if c['action'] != 'keep'}
    if len(returned) != len(execution.components) or set(returned) != editable:
        raise ValueError("A3 must configure exactly the editable IDs")
    out = copy.deepcopy(components)
    h, w = shape
    for c in out:
        cfg = NumericConfig.model_validate(c['config']) if c['action'] == 'keep' else returned[c['component_id']]
        values = parameters(c['noise_type'], cfg)
        if c['noise_type'] == 'single_fourier' and (values['frequency_x'] > w//2 or values['frequency_y'] > h//2):
            raise ValueError("Fourier frequencies exceed image Nyquist limits")
        c['config'] = cfg.model_dump()
    return out


def render(original, components, epsilon_pixels):
    from .gradient import statistics
    x = original.astype(np.float32)/255
    residual = np.zeros(original.shape, dtype=np.float64)
    core_union = np.zeros(original.shape[:2], dtype=bool)
    support = np.zeros_like(core_union)
    regions = {}
    for c in components:
        cfg = NumericConfig.model_validate(c['config'])
        core = core_mask(Region.model_validate(c['target_region']), original.shape[:2])
        mask = outward_mask(core, cfg.feather_pixels)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            residual += epsilon_pixels*cfg.strength*generate(c['noise_type'], x, cfg).astype(np.float64)*mask[..., None]
        core_union |= core
        support |= mask > 0
        key = json.dumps(c['target_region'], sort_keys=True)
        if key not in regions:
            regions[key] = dict(region_id='geometry_'+hashlib.sha256(key.encode()).hexdigest()[:16],
                                geometry=c['target_region'], crop_box_pixels=list(bbox(core)), components=[])
        regions[key]['components'].append(copy.deepcopy(c))
    if core_union.all() or not np.isfinite(residual).all():
        raise ValueError("Invalid core coverage or nonfinite composite")
    # One cap AFTER all gains, masks, and overlapping fields. No normalization.
    candidate = np.clip(np.rint(original.astype(np.float32)+np.clip(residual, -epsilon_pixels, epsilon_pixels)), 0, 255).astype(np.uint8)
    difference = candidate.astype(np.int16)-original.astype(np.int16)
    if np.abs(difference).max() > epsilon_pixels or np.any(difference[~support]):
        raise AssertionError("Saved pixels violate support or budget")
    for region in regions.values():
        core = core_mask(Region.model_validate(region['geometry']), original.shape[:2])
        region['actual_composite'] = statistics(original, candidate, core)
    return candidate, list(regions.values()), statistics(original, candidate, core_union)
