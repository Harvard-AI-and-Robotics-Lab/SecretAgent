"""Single-image inference. No benchmark scheduling, scoring tables, or data download."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from .config import Settings
from .providers import Router, load_prompt
from .tasks import InputTask, Task


def check(settings, require_weights=False):
    for role in settings.agents:
        if not load_prompt(settings, role).strip():
            raise ValueError(f'Empty {role} prompt')
    status = {'config': 'valid', 'dataset': settings.dataset, 'method': settings.method,
              'prompts': 'present', 'models_loaded': False, 'api_requests': 0,
              'model_paths_present': {}, 'credentials_present': {}}
    for name, path in settings.encoders.items():
        present = Path(os.path.expandvars(path)).expanduser().is_dir()
        status['model_paths_present'][name] = present
        if require_weights and not present:
            raise FileNotFoundError(f'Missing model directory: {name}')
    for role, provider in settings.agents.items():
        status['credentials_present'][role] = bool(os.environ.get(provider.key_env))
    status['environment'] = {}
    for name in ('numpy', 'scipy', 'PIL', 'pydantic', 'httpx', 'openai', 'torch', 'torchvision', 'transformers', 'safetensors'):
        try:
            module = importlib.import_module(name)
            status['environment'][name] = module.__version__
        except ImportError:
            status['environment'][name] = 'not installed'
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(prog='secretagent')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('check', help='Offline config, prompt, and dependency check; no API calls')
    inspect.add_argument('--config', required=True)
    inspect.add_argument('--require-weights', action='store_true')
    smoke = sub.add_parser('smoke', help='Offline synthetic forward/backward check using all six prepared encoders')
    smoke.add_argument('--config', required=True)
    smoke.add_argument('--device', default=None)
    run = sub.add_parser('protect', help='Run one complete search on a canonical RGB image')
    run.add_argument('--config', required=True)
    run.add_argument('--task', required=True)
    run.add_argument('--image', required=True)
    run.add_argument('--output', required=True, help='New PNG path; will not overwrite')
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    if args.command == 'check':
        print(json.dumps(check(settings, args.require_weights), indent=2))
        return 0
    if args.command == 'smoke':
        import torch
        from .encoders.bank import load_encoders
        from .features import Surrogates
        from .gradient import GradientRenderer
        from .schemas import GradientPlan, GradientParameters, IDS
        device = args.device or settings.device
        bank = None
        try:
            bank = Surrogates(load_encoders(settings.encoders, device), device)
            original = np.random.default_rng(7).integers(32,224,(48,64,3),dtype=np.uint8)
            plan = GradientPlan(regions=[dict(region_id='synthetic',clue='synthetic texture',
                geometry=dict(kind='bbox',bounds=dict(x_min=.25,y_min=.25,x_max=.75,y_max=.75),vertices=None))],instruction='Offline gradient check')
            params = GradientParameters(encoder_weights={k:1 for k in IDS},regions=[dict(region_id='synthetic',feather_pixels=4)],seed=7,explanation='Offline check')
            protected, regions, stats = GradientRenderer(bank).render(original,None,plan,params,4,2,['Synthetic colored texture.'],['Synthetic colored texture.'])
            feedback = bank.measure(original,protected,regions)
            if stats['linf_pixels'] > 2 or stats['no_change']:
                raise AssertionError('Offline one-step budget/nonzero check failed')
            if any(p.requires_grad or p.grad is not None for encoder in bank.encoders.values() for p in encoder.parameters()):
                raise AssertionError('Encoder weights must remain frozen')
            print(json.dumps({'status':'passed','device':device,'encoders':list(IDS),
                              'api_requests':0,'linf_pixels':stats['linf_pixels'],
                              'similarity':feedback['full_image'],'weights_frozen':True}))
        finally:
            if bank is not None:
                bank.close()
        return 0
    task = Task(InputTask.model_validate_json(Path(args.task).read_text()))
    if task.dataset != settings.dataset:
        raise ValueError('Task and config dataset must match')
    output = Path(args.output)
    if output.exists() or output.suffix.lower() != '.png':
        raise ValueError('Output must be a new .png file')
    with Image.open(args.image) as image:
        if image.mode != 'RGB' or image.getexif().get(274, 1) != 1 or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Supply a single-frame, orientation-correct RGB image. No implicit resizing/conversion is performed.')
        original = np.array(image, dtype=np.uint8)
    # Fail on missing keys before allocating any GPU memory.
    for provider in settings.agents.values():
        if not os.environ.get(provider.key_env):
            raise ValueError(f'Missing environment variable {provider.key_env}')
    from .encoders.bank import load_encoders
    from .features import Surrogates
    from .engine import Search
    bank = None
    try:
        bank = Surrogates(load_encoders(settings.encoders, settings.device), settings.device)
        router = Router(settings)
        result = Search(settings, router, bank, task).run(original)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            Image.fromarray(result.image).save(stream, format='PNG')
        print(json.dumps({'output': str(output), 'selected_round': result.selected_round,
                          'rounds_completed': len(result.history), 'requests': router.requests,
                          'original_fallback': result.selected_round is None}))
    finally:
        if bank is not None:
            bank.close()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
