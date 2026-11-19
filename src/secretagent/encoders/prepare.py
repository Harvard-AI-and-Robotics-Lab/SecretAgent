"""Extract only frozen vision/projector tensors from an official local snapshot.

This does not download datasets, train models, or load a language decoder.
Obtain checkpoints under their upstream access and license terms first.
"""
import argparse
import json
from importlib.resources import files
from pathlib import Path
import shutil
from safetensors import safe_open
from safetensors.torch import save_file


def extract(source, output, name):
    manifest = json.loads(files('secretagent.encoders').joinpath('checkpoints.json').read_text())
    spec = manifest[name]
    if name not in ('internvl35_projected', 'gemma4_projected'):
        raise ValueError('Use the complete local CLIP/DINO snapshot directly; no extraction is needed')
    if output.exists():
        raise FileExistsError('Prepared output must be a new directory')
    source = source.resolve(strict=True)
    index_path = source/'model.safetensors.index.json'
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())['weight_map']
        shard_names = sorted({v for k, v in weight_map.items() if k.startswith((spec['prefix'], spec['projector_prefix']))})
    elif (source/'model.safetensors').is_file():
        shard_names = ['model.safetensors']
    else:
        raise FileNotFoundError('No safetensors checkpoint/index in supplied snapshot')
    tensors = {'vision': {}, 'projector': {}}
    for shard_name in shard_names:
        if Path(shard_name).name != shard_name:
            raise ValueError('Invalid checkpoint shard name')
        with safe_open(source/shard_name, framework='pt', device='cpu') as shard:
            for key in shard.keys():
                for part, prefix in [('vision', spec['prefix']), ('projector', spec['projector_prefix'])]:
                    if key.startswith(prefix):
                        short = key[len(prefix):]
                        if short in tensors[part]:
                            raise ValueError('Duplicate parameter in checkpoint shards')
                        tensors[part][short] = shard.get_tensor(key).contiguous()
    if not all(tensors.values()):
        raise ValueError('Checkpoint is missing the specified vision/projector prefixes')
    names = ['config.json'] + (['configuration_intern_vit.py', 'modeling_intern_vit.py']
                               if name == 'internvl35_projected' else ['processor_config.json'])
    for filename in names:
        if not (source/filename).is_file():
            raise FileNotFoundError(f'Required upstream file: {filename}')
    output.mkdir(parents=True)
    for filename in names:
        shutil.copy2(source/filename, output/filename)
    for part, state in tensors.items():
        save_file(state, output/f'{part}.safetensors')
    print(json.dumps({'encoder': name, 'output': str(output),
                      'vision_tensors': len(tensors['vision']), 'projector_tensors': len(tensors['projector'])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['internvl35_projected', 'gemma4_projected'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.source, args.output, args.encoder)


if __name__ == '__main__':
    main()
