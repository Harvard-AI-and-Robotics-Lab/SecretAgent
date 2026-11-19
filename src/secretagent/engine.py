"""The four-agent loop shared by both variants; all history stays in memory."""
import copy
import io
from dataclasses import dataclass
import numpy as np
from PIL import Image
from .schemas import GradientPlan, GradientParameters, ToolPlan, Appearance, IDS
from .geometry import core_mask, gradient_regions
from .gradient import GradientRenderer
from .noise.fields import catalog
from . import tool


def png_roundtrip(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    stream.seek(0)
    with Image.open(stream) as image:
        return np.array(image, dtype=np.uint8)


def selection_key(record, method):
    if not record['appearance']['passed'] or record['prediction'] is None or record['actual_composite']['no_change']:
        return None
    full = record['encoder_feedback']['full_image']
    similarity = full['clip_openai_b32'] if method == 'gradient' else sum(full[k] for k in IDS)/len(IDS)
    return record['privacy_score'], -similarity


@dataclass
class Result:
    image: np.ndarray
    selected_round: int | None
    selected_recipe: dict | None
    history: list[dict]
    original_prediction: dict


class Search:
    def __init__(self, settings, router, surrogates, task):
        if settings.dataset != task.dataset:
            raise ValueError('Dataset in task and runtime config must match')
        self.settings, self.router, self.bank, self.task = settings, router, surrogates, task
        self.gradient = GradientRenderer(surrogates) if settings.method == 'gradient' else None

    def run(self, original):
        if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
            raise ValueError('Input must be a native-resolution uint8 RGB array')
        original = png_roundtrip(original)
        clean = self.task.assess(self.router, original)
        clean_record = {'candidate_id': 'original', 'prediction': clean, 'leaked': self.task.correctness(clean)}
        clean_text = self.task.text(clean)
        if self.settings.method == 'gradient' and not clean_text:
            raise ValueError('Clean A1 has no usable answer/evidence text for the CLIP objective')
        history, latest_pixels, best_record, best_pixels, best_key = [], None, None, None, None
        last_text, last_text_source = clean_text, 'original'
        h, w = original.shape[:2]
        for number in range(1, self.settings.rounds+1):
            latest = history[-1] if history else None
            payload = dict(self.task.public_context,
                round=number, rounds=self.settings.rounds,
                remaining_rounds=self.settings.rounds-number+1,
                epsilon_pixels=self.settings.epsilon_pixels,
                native_dimensions={'width': w, 'height': h},
                clean=clean_record, latest=latest, best=best_record,
                history=history, encoder_feedback=latest['encoder_feedback'] if latest else None)
            if self.settings.method == 'gradient':
                payload.update(alpha_pixels=self.settings.alpha_pixels, gradient_updates=1,
                               A1_text={'clean': clean_text, 'candidate': last_text, 'candidate_source': last_text_source})
            images = [('ORIGINAL', original)]+([('LATEST CANDIDATE', latest_pixels)] if latest_pixels is not None else [])
            if self.settings.method == 'gradient':
                def valid_plan(plan):
                    cores = [core_mask(r.geometry.canonical(), original.shape[:2]) for r in plan.regions]
                    if np.logical_or.reduce(cores).all():
                        raise ValueError('Combined region cores must remain local')
                plan = self.router.call('a2', payload, images, self.task.gradient_plan_schema, valid_plan)
                parameters = self.router.call('a3', {
                    'plan': plan.model_dump(), 'epsilon_pixels': self.settings.epsilon_pixels,
                    'alpha_pixels': self.settings.alpha_pixels, 'gradient_updates': 1,
                    'native_dimensions': {'width': w, 'height': h},
                    'A1_text': {'clean': clean_text, 'candidate': last_text, 'candidate_source': last_text_source},
                    'encoder_feedback': payload['encoder_feedback']}, [('ORIGINAL', original)],
                    GradientParameters, lambda p: gradient_regions(plan, p, original.shape[:2]))
                pixels, regions, actual = self.gradient.render(original, latest_pixels, plan, parameters,
                    self.settings.epsilon_pixels, self.settings.alpha_pixels, clean_text, last_text)
                executed = parameters.model_dump()
            else:
                payload['noise_catalog'] = catalog()
                plan = self.router.call('a2', payload, images, self.task.tool_plan_schema,
                    lambda p: tool.compile_plan(p, latest, best_record, original.shape[:2]))
                compiled = tool.compile_plan(plan, latest, best_record, original.shape[:2])
                schema = tool.execution_schema(compiled)
                numeric = None if schema is None else self.router.call('a3', {
                    'plan': compiled, 'instruction': plan.instruction,
                    'expected_component_ids': [c['component_id'] for c in compiled if c['action'] != 'keep'],
                    'locked_components': [c for c in compiled if c['action'] == 'keep'],
                    'epsilon_pixels': self.settings.epsilon_pixels,
                    'native_dimensions': {'width': w, 'height': h}, 'noise_catalog': catalog()},
                    [('ORIGINAL', original)], schema,
                    lambda p: tool.merge_parameters(compiled, p, original.shape[:2]))
                executed = tool.merge_parameters(compiled, numeric, original.shape[:2])
                pixels, regions, actual = tool.render(original, executed, self.settings.epsilon_pixels)
            pixels = png_roundtrip(pixels)
            appearance_payload, appearance_images, appearance_schema = self.task.appearance_request(original, pixels)
            appearance = self.router.call('a4', appearance_payload, appearance_images, appearance_schema)
            passed = appearance.passes()
            prediction = self.task.assess(self.router, pixels) if passed else None
            fusion_context = dict(clean=clean_text, candidate=last_text, candidate_source=last_text_source) if self.settings.method == 'gradient' else None
            measured = self.bank.measure(original, pixels, regions, latest['encoder_feedback'] if latest else None,
                                         fusion_context=fusion_context)
            record = dict(candidate_id=f'c{number:03d}', round=number, plan=plan.model_dump(), config=executed,
                appearance=dict(appearance.model_dump(), passed=passed), prediction=prediction,
                assessment_status='ASSESSED' if prediction is not None else 'NOT_EVALUATED',
                leaked=self.task.correctness(prediction) if prediction is not None else None,
                privacy_score=self.task.score(prediction, clean) if prediction is not None else None,
                actual_composite=actual, encoder_feedback=measured)
            key = selection_key(record, self.settings.method)
            if key is not None and (best_key is None or key > best_key):
                best_record, best_pixels, best_key = copy.deepcopy(record), pixels.copy(), key
            history.append(record)
            latest_pixels = pixels
            if prediction is not None:
                text = self.task.text(prediction)
                if text:
                    last_text, last_text_source = text, record['candidate_id']
            # Deliberately no protection-dependent early exit.
        return Result(original.copy() if best_pixels is None else best_pixels,
                      best_record['round'] if best_record else None,
                      best_record['config'] if best_record else None, history, clean)
