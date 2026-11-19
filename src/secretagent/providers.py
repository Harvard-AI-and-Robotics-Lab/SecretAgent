"""Stateless structured requests. Keys come only from named environment variables."""
import base64
import io
import json
import os
from importlib.resources import files
import httpx
from PIL import Image
from pydantic import ValidationError
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema


class ProviderError(RuntimeError):
    """A transport failure is not a privacy refusal or a successful candidate."""


def image_base64(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')


def anthropic_schema(value):
    if isinstance(value, list):
        return [anthropic_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    unsupported = {'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
                   'multipleOf', 'minLength', 'maxLength', 'maxItems', 'pattern'}
    out, constraints = {}, {}
    for key, item in value.items():
        if key in unsupported or (key == 'minItems' and item not in (0, 1)):
            constraints[key] = item
        else:
            out[key] = anthropic_schema(item)
    if constraints:
        out['description'] = out.get('description', '')+' Required constraints: '+json.dumps(constraints)
    return out


def load_prompt(settings, role):
    suffix = f'_{settings.method}' if role in ('a2', 'a3') else ''
    # Medical datasets use different A1/A4 image modalities.
    name = f'{role}{suffix}.txt'
    root = files('secretagent').joinpath('prompts', settings.domain)
    if settings.domain == 'medical':
        root = root.joinpath(settings.dataset)
    elif settings.dataset == 'frll_age_gender_race':
        root = root.joinpath('frll_age_gender_race')
    return root.joinpath(name).read_text()


class Router:
    def __init__(self, settings):
        self.settings = settings
        self.prompts = {r: load_prompt(settings, r) for r in settings.agents}
        self.requests = {r: 0 for r in settings.agents}

    def call(self, role, payload, images, schema, validate=None):
        # Only invalid structured responses get a same-round repair request.
        # Network/auth/credit/model failures stop; nothing is scored as abstention.
        message = dict(payload)
        for attempt in range(self.settings.validation_attempts):
            self.requests[role] += 1
            raw = self._request(role, message, images, schema)
            try:
                parsed = schema.model_validate_json(raw)
                if validate:
                    validate(parsed)
                return parsed
            except (ValidationError, ValueError) as error:
                if attempt+1 == self.settings.validation_attempts:
                    raise ValueError(f'{role}: structured validation failed after {attempt+1} attempts') from None
                detail = error.errors(include_input=False, include_context=False, include_url=False) if isinstance(error, ValidationError) else str(error)
                message = dict(payload, validation_feedback=detail)
        raise AssertionError('Unreachable')

    def _request(self, role, payload, images, schema):
        config = self.settings.agents[role]
        key = os.environ.get(config.key_env)
        if not key:
            raise ProviderError(f'Missing environment variable {config.key_env}')
        cap = 65536 if role == 'a2' else 16384 if role == 'a3' else 8192
        packed = [(label, image_base64(rgb)) for label, rgb in images]
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if config.backend == 'openai_responses':
            content = [{'type': 'input_text', 'text': text}]
            for label, data in packed:
                content += [{'type': 'input_text', 'text': label},
                            {'type': 'input_image', 'image_url': 'data:image/png;base64,'+data, 'detail': 'high'}]
            body = dict(model=config.model, instructions=self.prompts[role],
                        input=[{'role': 'user', 'content': content}], store=False,
                        max_output_tokens=cap, reasoning={'effort': config.reasoning},
                        text={'format': {'type': 'json_schema', 'name': schema.__name__,
                                         'strict': True, 'schema': to_strict_json_schema(schema)}})
            if config.temperature is not None:
                body['temperature'] = config.temperature
            try:
                with OpenAI(api_key=key, base_url=config.base_url, max_retries=0, timeout=300) as client:
                    result = client.responses.create(**body)
            except Exception as error:
                raise ProviderError(f'{role}: OpenAI request failed ({type(error).__name__})') from None
            if result.status != 'completed' or result.model != config.model or not result.output_text:
                raise ProviderError(f'{role}: incomplete response or unexpected model; no automatic substitution')
            return result.output_text
        if config.backend == 'anthropic':
            content = [{'type': 'text', 'text': text}]
            for label, data in packed:
                content += [{'type': 'text', 'text': label}, {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/png', 'data': data}}]
            body = dict(model=config.model, max_tokens=cap, system=self.prompts[role],
                        thinking={'type': 'disabled' if config.reasoning == 'none' else 'adaptive'},
                        output_config={'effort': 'medium', 'format': {'type': 'json_schema', 'schema': anthropic_schema(schema.model_json_schema())}},
                        messages=[{'role': 'user', 'content': content}])
            if config.temperature is not None:
                body['temperature'] = config.temperature
            result = self._http(config.base_url.rstrip('/')+'/messages', body,
                                {'x-api-key': key, 'anthropic-version': '2023-06-01'})
            if result.get('model') != config.model or result.get('stop_reason') != 'end_turn':
                raise ProviderError(f'{role}: incomplete response or unexpected model')
            if config.reasoning == 'none' and any(x['type'] in ('thinking', 'redacted_thinking') for x in result['content']):
                raise ProviderError(f'{role}: disabled reasoning contract was violated')
            return ''.join(x['text'] for x in result['content'] if x['type'] == 'text')
        # A pre-adapted medical VLM exposed by an OpenAI-compatible endpoint.
        content = [{'type': 'text', 'text': text}]
        for label, data in packed:
            content += [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+data}}]
        body = dict(model=config.model, messages=[{'role': 'system', 'content': self.prompts[role]},
                    {'role': 'user', 'content': content}], max_tokens=cap,
                    response_format={'type': 'json_schema', 'json_schema': {
                        'name': schema.__name__, 'strict': True, 'schema': to_strict_json_schema(schema)}})
        if config.temperature is not None:
            body['temperature'] = config.temperature
        result = self._http(config.base_url.rstrip('/')+'/chat/completions', body, {'Authorization': 'Bearer '+key})
        if result.get('model') != config.model or result['choices'][0]['finish_reason'] != 'stop':
            raise ProviderError(f'{role}: incomplete response or unexpected model')
        message = result['choices'][0]['message']
        if message.get('reasoning_content') or message.get('reasoning'):
            raise ProviderError(f'{role}: medical endpoint returned unexpected reasoning')
        return message['content']

    @staticmethod
    def _http(url, body, headers):
        try:
            with httpx.Client(timeout=httpx.Timeout(300, connect=30), trust_env=False) as client:
                response = client.post(url, json=body, headers=headers)
        except httpx.HTTPError as error:
            raise ProviderError(f'HTTP transport failed ({type(error).__name__}); no automatic retry') from None
        if response.status_code != 200:
            raise ProviderError(f'HTTP {response.status_code}; provider body and credentials are not logged')
        return response.json()
