"""Explicit runtime configuration, separate from input labels and credentials."""
import json
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from .schemas import StrictModel, IDS


class Provider(StrictModel):
    backend: Literal['openai_responses', 'anthropic', 'openai_chat']
    model: str
    key_env: str
    base_url: str
    reasoning: Literal['none', 'medium'] = 'none'
    temperature: float | None = None


class Settings(StrictModel):
    dataset: Literal['pristreet', 'p5', 'frll_age_gender_race', 'rhpe', 'papila']
    method: Literal['gradient', 'tool']
    epsilon_pixels: int = Field(default=4, ge=1, le=16)
    alpha_pixels: float = Field(default=2, gt=0, le=16)
    rounds: int = Field(default=10, ge=1, le=30)
    validation_attempts: int = Field(default=3, ge=1, le=3)
    device: str = 'cuda:0'
    encoders: dict[str, str]
    agents: dict[str, Provider]

    @model_validator(mode='after')
    def complete(self):
        if set(self.encoders) != set(IDS):
            raise ValueError('Exactly the six declared encoder paths are required')
        if set(self.agents) != {'a1', 'a2', 'a3', 'a4'}:
            raise ValueError('Exactly four agent provider configurations are required')
        if self.agents['a1'].reasoning != 'none' or self.agents['a4'].reasoning != 'none':
            raise ValueError('The release protocol disables A1/A4 reasoning')
        return self

    @property
    def domain(self):
        return dict(pristreet='geolocation', p5='pii', frll_age_gender_race='face',
                    rhpe='medical', papila='medical')[self.dataset]

    @classmethod
    def load(cls, path):
        return cls.model_validate(json.loads(Path(path).read_text()))
