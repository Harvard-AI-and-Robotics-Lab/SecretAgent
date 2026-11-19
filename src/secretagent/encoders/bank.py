"""Frozen visual encoders and pretrained vision-to-language projectors."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import functional as TV
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file
from transformers import CLIPModel, AutoTokenizer, AutoModel, Gemma4Config, Gemma4VisionConfig, Gemma4VisionModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from ..schemas import IDS

def deterministic_runtime():
    # Must be set before CUDA model initialization. Keep it the same for smoke,
    # serial/parallel benchmarks and production; iterative sign updates amplify
    # tiny non-deterministic gradient differences.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def normalized(x):
    x = x.float()
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite encoder features')
    return F.normalize(x, dim=-1)

def deterministic_resize(x, size, interpolation):
    # CUDA antialiased resize backward uses nondeterministic atomic additions.
    # Retain the same interpolation and autograd, executing only resizing on CPU.
    return TV.resize(x.cpu(), size, interpolation=interpolation, antialias=True).to(x.device)

class Clip(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = CLIPModel.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.register_buffer('mean', torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1))

    def preprocess(self, x):
        x = deterministic_resize(x, 224, InterpolationMode.BICUBIC)
        x = TV.center_crop(x, [224,224])
        return (x-self.mean)/self.std

    def forward(self, x):
        y = self.model.get_image_features(pixel_values=self.preprocess(x))
        return normalized(y.pooler_output).unsqueeze(1)

    @torch.no_grad()
    def text(self, texts):
        # Split each location/clue sentence into <=75-token pieces; never silently
        # discard later clues at CLIP's 77-token context limit.
        chunks=[]
        for text in texts:
            ids=self.tokenizer.encode(text, add_special_tokens=False)
            chunks.extend(ids[i:i+75] for i in range(0,len(ids),75))
        if not chunks: raise ValueError('No A1 text to encode')
        rows=[{'input_ids':[self.tokenizer.bos_token_id]+c+[self.tokenizer.eos_token_id],
               'attention_mask':[1]*(len(c)+2)} for c in chunks]
        t = self.tokenizer.pad(rows, padding=True, return_tensors='pt').to(self.mean.device)
        return normalized(self.model.get_text_features(**t).pooler_output)

def import_intern(path):
    name = 'afp_pinned_internvl35'
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
        for filename in ('configuration_intern_vit', 'modeling_intern_vit'):
            spec = importlib.util.spec_from_file_location(f'{name}.{filename}', path / f'{filename}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return (sys.modules[name+'.configuration_intern_vit'].InternVisionConfig,
            sys.modules[name+'.modeling_intern_vit'].InternVisionModel)

class Intern(nn.Module):
    def __init__(self, path):
        super().__init__()
        Config, Model = import_intern(path)
        config = Config(**json.loads((path/'config.json').read_text())['vision_config'])
        config.use_flash_attn = False
        self.model = Model(config)
        self.model.load_state_dict(load_file(path/'vision.safetensors'), strict=True)
        full=json.loads((path/'config.json').read_text())
        assert full['downsample_ratio']==.5 and full['ps_version']=='v2' and full['select_layer']==-1
        hidden=config.hidden_size; language=full['llm_config']['hidden_size']
        self.mlp1=nn.Sequential(nn.LayerNorm(hidden*4),nn.Linear(hidden*4,language),nn.GELU(),nn.Linear(language,language))
        self.mlp1.load_state_dict(load_file(path/'projector.safetensors'),strict=True)
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self, x):
        x = deterministic_resize(x, [448,448], InterpolationMode.BICUBIC)
        x = (x-self.mean)/self.std
        tokens=self.model(pixel_values=x, return_dict=True).last_hidden_state[:,1:]
        n, count, channels=tokens.shape; side=int(count**.5)
        assert side*side==count and side%2==0
        # Exact official InternVL3.5 pixel_shuffle(v2, scale_factor=.5).
        tokens=tokens.reshape(n,side,side,channels).view(n,side,side//2,channels*2)
        tokens=tokens.permute(0,2,1,3).contiguous().view(n,side//2,side//2,channels*4)
        tokens=tokens.permute(0,2,1,3).contiguous().reshape(n,-1,channels*4)
        return self.mlp1(tokens)

class Gemma(nn.Module):
    def __init__(self, path):
        super().__init__()
        config = Gemma4VisionConfig(**json.loads((path/'config.json').read_text())['vision_config'])
        config._attn_implementation = 'sdpa'
        self.model = Gemma4VisionModel(config)
        self.model.load_state_dict(load_file(path/'vision.safetensors'), strict=True)
        full=Gemma4Config(**json.loads((path/'config.json').read_text()))
        self.embed_vision=Gemma4MultimodalEmbedder(full.vision_config,full.text_config)
        self.embed_vision.load_state_dict(load_file(path/'projector.safetensors'),strict=True)
        settings = json.loads((path/'processor_config.json').read_text())['image_processor']
        self.processor = Gemma4ImageProcessor(**settings)

    def preprocess(self, x):
        assert x.shape[0] == 1
        # Direct tensor backend: retain autograd and avoid PIL/NumPy conversion.
        batch = self.processor._preprocess(
            images=[x[0].cpu()], do_resize=True, resample=InterpolationMode.BICUBIC,
            do_rescale=False, rescale_factor=1., do_normalize=False,
            image_mean=(0.,0.,0.), image_std=(1.,1.,1.), return_tensors='pt',
            patch_size=16, max_soft_tokens=280, pooling_kernel_size=3)
        return dict(pixel_values=batch['pixel_values'].to(x.device),
                    pixel_position_ids=batch['image_position_ids'].to(x.device))

    def forward(self, x):
        projected = self.embed_vision(self.model(**self.preprocess(x)).last_hidden_state)
        # Gemma4 5.5.4 flattens the single image to [tokens,hidden]. Retain a
        # batch dimension so the shared feature bank can mean-pool tokens.
        if projected.ndim == 2 and x.shape[0] == 1:
            projected = projected.unsqueeze(0)
        if projected.ndim != 3 or projected.shape[0] != x.shape[0]:
            raise ValueError('Gemma projected tokens must have shape [batch,tokens,hidden]')
        return projected

class Dino(nn.Module):
    def __init__(self, path):
        super().__init__()
        settings=json.loads((path/'preprocessor_config.json').read_text())
        config=json.loads((path/'config.json').read_text())
        assert config['model_type']=='dinov3_vit' and config['hidden_size']==1024
        assert settings['resample']==2 and not settings.get('do_center_crop')
        self.size=[settings['size']['height'],settings['size']['width']]
        self.register_buffer('mean',torch.tensor(settings['image_mean']).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(settings['image_std']).view(1,3,1,1))
        self.model=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')

    def forward(self,x):
        x=deterministic_resize(x,self.size,InterpolationMode.BILINEAR)
        return normalized(self.model(pixel_values=(x-self.mean)/self.std).last_hidden_state[:,0]).unsqueeze(1)

def load_encoders(paths, device="cuda:0"):
    if set(paths) != set(IDS):
        raise ValueError("Provide a local path for each of the six encoders")
    deterministic_runtime()
    constructors = (Clip, Clip, Clip, Intern, Gemma, Dino)
    out = {}
    try:
        for name, constructor in zip(IDS, constructors):
            path = Path(os.path.expandvars(str(paths[name]))).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(f"Missing prepared encoder directory: {name}")
            out[name] = constructor(path).float().eval().requires_grad_(False).to(device)
        return out
    except BaseException:
        out.clear()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        raise
